from runtime_compat import bootstrap

bootstrap()

import argparse
import json
from dataclasses import replace

import pandas as pd

from config import RESULTS
from run_experiments import configure, run_one


VARIANTS = [
    ("no_transport", {"transport_weight": 0.0}),
    ("no_feature_prototype", {"prototype_weight": 0.0, "prototype_fusion": 0.0}),
    ("no_label_reliability", {"use_label_reliability": False}),
    ("no_spectral_reliability", {"use_spectral_reliability": False}),
    ("no_trajectory", {"use_trajectory_stabilization": False}),
]


def collect():
    rows = []
    for variant, _ in VARIANTS + [("full", {})]:
        for seed in [2027, 2028, 2029]:
            suffix = "" if variant == "full" else f"_{variant}"
            path = RESULTS / "models" / f"run_urbansound8k_FedSPECTRA{suffix}_seed{seed}.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"variant": variant, "seed": seed, **r["metrics"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "tables" / "ablation_urbansound8k_seedwise.csv", index=False)
    summary = frame.groupby("variant", as_index=False).agg(
        accuracy_mean=("pooled_accuracy", "mean"), accuracy_std=("pooled_accuracy", "std"),
        macro_f1_mean=("pooled_macro_f1", "mean"), macro_f1_std=("pooled_macro_f1", "std"),
        ece_mean=("pooled_ece", "mean"), worst_client_mean=("accuracy_worst", "mean"),
    )
    summary.to_csv(RESULTS / "tables" / "ablation_urbansound8k.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    dummy = argparse.Namespace(quick=False, rounds=None)
    if not args.collect_only:
        for seed in [2028, 2029]:
            base = configure(dummy, "urbansound8k", "FedSPECTRA", seed)
            for variant, changes in VARIANTS:
                run_one(replace(base, variant=variant, **changes))
    collect()


if __name__ == "__main__":
    main()
