from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import torch

from config import METHODS, RESULTS, SEEDS, ExperimentConfig
from data_pipeline import build_client_datasets, dataset_summary, load_signal_data
from federated import run_federated


def configure(args, dataset: str, method: str, seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig(dataset=dataset, method=method, seed=seed)
    if dataset == "urbansound8k":
        cfg.rounds = 48
        cfg.prototype_fusion = 0.03
    if dataset == "esc50":
        cfg.num_clients = 8
        cfg.clients_per_round = 4
        cfg.rounds = 48
        cfg.min_client_samples = 30
        cfg.prototype_fusion = 0.03
        cfg.esc50_test_fold = args.fold
        cfg.esc50_val_fold = args.fold % 5 + 1
    if args.quick:
        cfg.rounds = 4
        cfg.local_steps = 2
        cfg.num_clients = 8
        cfg.clients_per_round = 4
        cfg.min_client_samples = 20
        cfg.eval_interval = 1
    if args.rounds is not None:
        cfg.rounds = args.rounds
    return cfg


def run_one(cfg: ExperimentConfig) -> Dict:
    data = load_signal_data(cfg)
    train_clients, val_clients, test_clients = build_client_datasets(cfg, data)
    summary = dataset_summary(cfg, data, train_clients)
    print(json.dumps({"event": "dataset_ready", **summary}, ensure_ascii=False), flush=True)
    result = run_federated(cfg, train_clients, val_clients, test_clients)
    print(json.dumps({"event": "run_complete", "dataset": cfg.dataset, "method": cfg.method,
                      "variant": cfg.variant, "seed": cfg.seed,
                      "accuracy": result["metrics"]["pooled_accuracy"],
                      "macro_f1": result["metrics"]["pooled_macro_f1"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


def collect_main_table() -> pd.DataFrame:
    rows = []
    paths = sorted((RESULTS / "models").glob("run_*_seed*.json"))
    has_esc50_v2 = any("run_esc50_" in path.name and "_v2_seed" in path.name for path in paths)
    has_esc50_v3 = any("run_esc50_" in path.name and "FedSPECTRA_v3_seed" in path.name for path in paths)
    has_esc50_v4 = any("run_esc50_" in path.name and "FedSPECTRA_v4_seed" in path.name for path in paths)
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        cfg = record["config"]
        if cfg.get("variant", "full") != "full" or cfg["method"] not in METHODS:
            continue
        if cfg["dataset"] == "esc50":
            desired_tag = "v4" if cfg["method"] == "FedSPECTRA" and has_esc50_v4 else (
                "v3" if cfg["method"] == "FedSPECTRA" and has_esc50_v3 else (
                "v2" if has_esc50_v2 else "")
            )
            if cfg.get("run_tag", "") != desired_tag:
                continue
        row = {"dataset": cfg["dataset"], "method": cfg["method"], "seed": cfg["seed"],
               "fold": cfg.get("esc50_test_fold") if cfg["dataset"] == "esc50" else None,
               "elapsed_seconds": record["elapsed_seconds"], "communication_mb": record["communication_mb"],
               "peak_gpu_memory_mb": record["peak_gpu_memory_mb"], "parameter_count": record["parameter_count"]}
        row.update(record["metrics"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(RESULTS / "tables" / "main_comparison_seedwise.csv", index=False)
        summary = frame.groupby(["dataset", "method"], as_index=False).agg(
            accuracy_mean=("pooled_accuracy", "mean"), accuracy_std=("pooled_accuracy", "std"),
            macro_f1_mean=("pooled_macro_f1", "mean"), macro_f1_std=("pooled_macro_f1", "std"),
            ece_mean=("pooled_ece", "mean"), ece_std=("pooled_ece", "std"),
            worst_client_mean=("accuracy_worst", "mean"), client_gap_mean=("accuracy_gap", "mean"),
            communication_mb=("communication_mb", "mean"), elapsed_seconds=("elapsed_seconds", "mean"),
            peak_gpu_memory_mb=("peak_gpu_memory_mb", "mean"), parameter_count=("parameter_count", "mean"),
        )
        summary.to_csv(RESULTS / "tables" / "main_comparison.csv", index=False)
    return frame


def run_ablation(base: ExperimentConfig) -> None:
    variants = [
        ("no_transport", {"transport_weight": 0.0}),
        ("no_feature_prototype", {"prototype_weight": 0.0, "prototype_fusion": 0.0}),
        ("no_label_reliability", {"use_label_reliability": False}),
        ("no_spectral_reliability", {"use_spectral_reliability": False}),
        ("no_trajectory", {"use_trajectory_stabilization": False}),
        ("full", {}),
    ]
    rows = []
    for variant, changes in variants:
        cfg = replace(base, method="FedSPECTRA", variant=variant, **changes)
        result = run_one(cfg)
        rows.append({"dataset": cfg.dataset, "variant": variant, **result["metrics"],
                     "communication_mb": result["communication_mb"],
                     "elapsed_seconds": result["elapsed_seconds"]})
    pd.DataFrame(rows).to_csv(RESULTS / "tables" / f"ablation_{base.dataset}.csv", index=False)


def run_sensitivity(base: ExperimentConfig) -> None:
    settings = []
    for transport in [0.0, 0.005, 0.01, 0.02, 0.04]:
        settings.append((f"transport_{transport:.2f}", {"transport_weight": transport}))
    for blend in [0.0, 0.20, 0.35, 0.50, 0.75]:
        settings.append((f"blend_{blend:.2f}", {"reliability_blend": blend}))
    records = []
    for variant, changes in settings:
        cfg = replace(base, method="FedSPECTRA", variant=variant, rounds=max(18, base.rounds // 2), **changes)
        result = run_one(cfg)
        records.append({"variant": variant, "dataset": base.dataset,
                        "accuracy": result["metrics"]["pooled_accuracy"],
                        "macro_f1": result["metrics"]["pooled_macro_f1"],
                        "ece": result["metrics"]["pooled_ece"]})
    (RESULTS / "models" / f"sensitivity_{base.dataset}.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")


def run_heterogeneity(base: ExperimentConfig) -> None:
    records = []
    for alpha in [0.10, 0.30, 1.00]:
        for shift in [0.16, 0.32, 0.48]:
            for method in ["FedAvg", "FPL", "FedSPECTRA"]:
                cfg = replace(base, method=method, variant=f"a{alpha:.2f}_s{shift:.2f}",
                              dirichlet_alpha=alpha, device_shift_strength=shift,
                              rounds=max(18, base.rounds // 2))
                result = run_one(cfg)
                records.append({"alpha": alpha, "shift": shift, "method": method,
                                "accuracy": result["metrics"]["pooled_accuracy"],
                                "macro_f1": result["metrics"]["pooled_macro_f1"],
                                "worst_client": result["metrics"]["accuracy_worst"],
                                "ece": result["metrics"]["pooled_ece"]})
    (RESULTS / "models" / f"heterogeneity_{base.dataset}.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["urbansound8k", "esc50"], default="urbansound8k")
    p.add_argument("--method", choices=METHODS, default="FedSPECTRA")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--rounds", type=int)
    p.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--all", action="store_true",
                   help="Run the UrbanSound8K main comparison with all methods and seeds")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--sensitivity", action="store_true")
    p.add_argument("--heterogeneity", action="store_true")
    p.add_argument("--collect", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    for d in [RESULTS, RESULTS / "models", RESULTS / "tables", RESULTS / "figures"]:
        d.mkdir(parents=True, exist_ok=True)
    if args.collect:
        print(collect_main_table())
        return
    if args.all:
        for seed in SEEDS:
            for method in METHODS:
                run_one(configure(args, "urbansound8k", method, seed))
        collect_main_table()
        return
    cfg = configure(args, args.dataset, args.method, args.seed)
    if args.ablation:
        run_ablation(cfg)
    elif args.sensitivity:
        run_sensitivity(cfg)
    elif args.heterogeneity:
        run_heterogeneity(cfg)
    else:
        run_one(cfg)
        collect_main_table()


if __name__ == "__main__":
    main()
