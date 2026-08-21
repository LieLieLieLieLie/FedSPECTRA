from runtime_compat import bootstrap

bootstrap()

import argparse
import json
from dataclasses import replace

import pandas as pd

from config import RESULTS, SEEDS
from run_experiments import configure, run_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    dummy = argparse.Namespace(quick=False, rounds=None)
    variants = [("highshift_full", {}), ("highshift_no_transport", {"transport_weight": 0.0})]
    if not args.collect_only:
        for seed in SEEDS:
            base = configure(dummy, "urbansound8k", "FedSPECTRA", seed)
            for variant, changes in variants:
                run_one(replace(base, variant=variant, device_shift_strength=0.48, **changes))
    rows = []
    for variant, _ in variants:
        for seed in SEEDS:
            path = RESULTS / "models" / f"run_urbansound8k_FedSPECTRA_{variant}_seed{seed}.json"
            r = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"variant": variant, "seed": seed, **r["metrics"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "tables" / "high_shift_ablation_seedwise.csv", index=False)
    frame.groupby("variant", as_index=False).agg(
        accuracy_mean=("pooled_accuracy", "mean"), accuracy_std=("pooled_accuracy", "std"),
        macro_f1_mean=("pooled_macro_f1", "mean"), macro_f1_std=("pooled_macro_f1", "std"),
        worst_client_mean=("accuracy_worst", "mean"), ece_mean=("pooled_ece", "mean"),
    ).to_csv(RESULTS / "tables" / "high_shift_ablation.csv", index=False)


if __name__ == "__main__":
    main()
