"""Run and collect the paper's UrbanSound8K ablation suites."""

from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import argparse
import json
from dataclasses import replace

import pandas as pd

from config import RESULTS, ExperimentConfig
from run_experiments import configure, run_one


SEEDS = [2027, 2028, 2029]
REMOVAL_VARIANTS = [
    ("no_transport", {"transport_weight": 0.0}),
    ("no_feature_prototype", {"prototype_weight": 0.0, "prototype_fusion": 0.0}),
    ("no_label_reliability", {"use_label_reliability": False}),
    ("no_spectral_reliability", {"use_spectral_reliability": False}),
    ("no_trajectory", {"use_trajectory_stabilization": False}),
]
SHIFTS = {"standard": 0.32, "strong": 0.48}
STAGES = [
    ("trajectory", dict(prototype_weight=0.0, transport_weight=0.0,
                        prototype_fusion=0.0, use_reliability=False)),
    ("feature", dict(prototype_weight=0.01, transport_weight=0.0,
                     prototype_fusion=0.03, use_reliability=False)),
    ("transport", dict(prototype_weight=0.01, transport_weight=0.01,
                       prototype_fusion=0.03, use_reliability=False)),
    ("label_reliability", dict(prototype_weight=0.01, transport_weight=0.01,
                               prototype_fusion=0.03, use_reliability=True,
                               use_label_reliability=True, use_spectral_reliability=False)),
    ("full", {}),
]


def _load_or_run(path, cfg: ExperimentConfig, collect_only: bool):
    if path.exists():
        print(f"cached: {path.name}", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))
    if collect_only:
        raise FileNotFoundError(f"Missing cached result required by --collect-only: {path}")
    return run_one(cfg)


def run_removal(collect_only: bool) -> None:
    rows = []
    variants = REMOVAL_VARIANTS + [("full", {})]
    for seed in SEEDS:
        base = configure(argparse.Namespace(quick=False, rounds=None), "urbansound8k", "FedSPECTRA", seed)
        for variant, changes in variants:
            suffix = "" if variant == "full" else f"_{variant}"
            path = RESULTS / "models" / f"run_urbansound8k_FedSPECTRA{suffix}_seed{seed}.json"
            cfg = replace(base, variant=variant, **changes)
            result = _load_or_run(path, cfg, collect_only)
            rows.append({"variant": variant, "seed": seed, **result["metrics"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "tables" / "ablation_urbansound8k_seedwise.csv", index=False)
    frame.groupby("variant", as_index=False).agg(
        accuracy_mean=("pooled_accuracy", "mean"), accuracy_std=("pooled_accuracy", "std"),
        macro_f1_mean=("pooled_macro_f1", "mean"), macro_f1_std=("pooled_macro_f1", "std"),
        ece_mean=("pooled_ece", "mean"), worst_client_mean=("accuracy_worst", "mean"),
    ).to_csv(RESULTS / "tables" / "ablation_urbansound8k.csv", index=False)


def run_staged(collect_only: bool) -> None:
    rows = []
    for shift_name, strength in SHIFTS.items():
        for stage, changes in STAGES:
            for seed in SEEDS:
                if stage == "full":
                    variant = "full" if shift_name == "standard" else "highshift_full"
                    suffix = "" if variant == "full" else f"_{variant}"
                else:
                    variant = f"stage_{shift_name}_{stage}"
                    suffix = f"_{variant}"
                path = RESULTS / "models" / f"run_urbansound8k_FedSPECTRA{suffix}_seed{seed}.json"
                cfg = ExperimentConfig(
                    dataset="urbansound8k", method="FedSPECTRA", seed=seed,
                    variant=variant, rounds=48, prototype_fusion=0.03,
                    device_shift_strength=strength,
                )
                result = _load_or_run(path, replace(cfg, **changes), collect_only)
                rows.append({"shift": shift_name, "shift_strength": strength, "stage": stage,
                             "seed": seed, **result["metrics"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "tables" / "staged_ablation_seedwise.csv", index=False)
    summary = frame.groupby(["shift", "shift_strength", "stage"], as_index=False).agg(
        accuracy_mean=("pooled_accuracy", "mean"), accuracy_std=("pooled_accuracy", "std"),
        macro_f1_mean=("pooled_macro_f1", "mean"), macro_f1_std=("pooled_macro_f1", "std"),
        worst_client_mean=("accuracy_worst", "mean"), ece_mean=("pooled_ece", "mean"),
    )
    summary["stage_order"] = summary.stage.map({name: i for i, (name, _) in enumerate(STAGES)})
    summary = summary.sort_values(["shift_strength", "stage_order"])
    summary["delta_macro_f1_pp"] = summary.groupby("shift").macro_f1_mean.diff() * 100
    summary.to_csv(RESULTS / "tables" / "staged_ablation.csv", index=False)


def run_high_shift(collect_only: bool) -> None:
    rows = []
    variants = [("highshift_full", {}), ("highshift_no_transport", {"transport_weight": 0.0})]
    for seed in SEEDS:
        base = configure(argparse.Namespace(quick=False, rounds=None), "urbansound8k", "FedSPECTRA", seed)
        for variant, changes in variants:
            path = RESULTS / "models" / f"run_urbansound8k_FedSPECTRA_{variant}_seed{seed}.json"
            cfg = replace(base, variant=variant, device_shift_strength=0.48, **changes)
            result = _load_or_run(path, cfg, collect_only)
            rows.append({"variant": variant, "seed": seed, **result["metrics"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "tables" / "high_shift_ablation_seedwise.csv", index=False)
    frame.groupby("variant", as_index=False).agg(
        accuracy_mean=("pooled_accuracy", "mean"), accuracy_std=("pooled_accuracy", "std"),
        macro_f1_mean=("pooled_macro_f1", "mean"), macro_f1_std=("pooled_macro_f1", "std"),
        worst_client_mean=("accuracy_worst", "mean"), ece_mean=("pooled_ece", "mean"),
    ).to_csv(RESULTS / "tables" / "high_shift_ablation.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="UrbanSound8K ablation runner")
    parser.add_argument("suite", choices=["removal", "staged", "high-shift", "all"],
                        nargs="?", default="all")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if args.suite in {"removal", "all"}:
        run_removal(args.collect_only)
    if args.suite in {"staged", "all"}:
        run_staged(args.collect_only)
    if args.suite in {"high-shift", "all"}:
        run_high_shift(args.collect_only)


if __name__ == "__main__":
    main()
