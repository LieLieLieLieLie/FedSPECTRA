"""Matched incremental ablation under standard and stronger acquisition shift."""

from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import json
from dataclasses import replace

import pandas as pd

from config import RESULTS, ExperimentConfig
from run_experiments import run_one


SEEDS = [2027, 2028, 2029]
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


def _cached_full(shift_name: str, seed: int):
    if shift_name == "standard":
        path = RESULTS / "models" / f"run_urbansound8k_FedSPECTRA_seed{seed}.json"
    else:
        path = RESULTS / "models" / f"run_urbansound8k_FedSPECTRA_highshift_full_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    rows = []
    for shift_name, strength in SHIFTS.items():
        for stage, changes in STAGES:
            for seed in SEEDS:
                if stage == "full":
                    result = _cached_full(shift_name, seed)
                    if result is None:
                        variant = "full" if shift_name == "standard" else "highshift_full"
                        cfg = ExperimentConfig(dataset="urbansound8k", method="FedSPECTRA", seed=seed,
                                               variant=variant, rounds=48, prototype_fusion=0.03,
                                               device_shift_strength=strength)
                        result = run_one(cfg)
                else:
                    variant = f"stage_{shift_name}_{stage}"
                    path = RESULTS / "models" / f"run_urbansound8k_FedSPECTRA_{variant}_seed{seed}.json"
                    if path.exists():
                        result = json.loads(path.read_text(encoding="utf-8"))
                        print(f"cached: {path.name}", flush=True)
                    else:
                        base = ExperimentConfig(dataset="urbansound8k", method="FedSPECTRA", seed=seed,
                                                variant=variant, rounds=48, prototype_fusion=0.03,
                                                device_shift_strength=strength)
                        result = run_one(replace(base, **changes))
                rows.append({"shift": shift_name, "shift_strength": strength, "stage": stage,
                             "seed": seed, **result["metrics"]})

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "tables" / "staged_ablation_seedwise.csv", index=False)
    summary = frame.groupby(["shift", "shift_strength", "stage"], as_index=False).agg(
        accuracy_mean=("pooled_accuracy", "mean"), accuracy_std=("pooled_accuracy", "std"),
        macro_f1_mean=("pooled_macro_f1", "mean"), macro_f1_std=("pooled_macro_f1", "std"),
        worst_client_mean=("accuracy_worst", "mean"), ece_mean=("pooled_ece", "mean"),
    )
    order = {name: i for i, (name, _) in enumerate(STAGES)}
    summary["stage_order"] = summary.stage.map(order)
    summary = summary.sort_values(["shift_strength", "stage_order"])
    summary["delta_macro_f1_pp"] = summary.groupby("shift").macro_f1_mean.diff() * 100
    summary.to_csv(RESULTS / "tables" / "staged_ablation.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
