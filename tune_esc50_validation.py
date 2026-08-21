from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import argparse
import json
from dataclasses import replace

import pandas as pd

from config import RESULTS
from run_experiments import configure, run_one


SETTINGS = [
    ("esc_base", {}),
    ("esc_no_traj", {"use_trajectory_stabilization": False}),
    ("esc_fusion0", {"prototype_fusion": 0.0}),
    ("esc_no_traj_fusion0", {"use_trajectory_stabilization": False, "prototype_fusion": 0.0}),
    ("esc_shrink15", {"use_trajectory_stabilization": False, "reliability_blend": 0.15}),
    ("esc_shrink10", {"use_trajectory_stabilization": False, "reliability_blend": 0.10}),
    ("esc_light", {"use_trajectory_stabilization": False, "reliability_blend": 0.15,
                   "prototype_weight": 0.005, "transport_weight": 0.005}),
    ("esc_light_fusion0", {"use_trajectory_stabilization": False, "reliability_blend": 0.15,
                           "prototype_weight": 0.005, "transport_weight": 0.005,
                           "prototype_fusion": 0.0}),
]


def main():
    dummy = argparse.Namespace(quick=False, rounds=None, fold=1)
    base = configure(dummy, "esc50", "FedSPECTRA", 2027)
    rows = []
    for variant, changes in SETTINGS:
        output = RESULTS / "models" / f"run_esc50_fold1_FedSPECTRA_{variant}_seed2027.json"
        if output.exists():
            result = json.loads(output.read_text(encoding="utf-8"))
            print(f"skip existing: {output.name}", flush=True)
        else:
            result = run_one(replace(base, variant=variant, **changes))
        final_validation = next(
            record for record in reversed(result["history"]) if "val_pooled_macro_f1" in record
        )
        rows.append({
            "variant": variant,
            "validation_accuracy": final_validation["val_pooled_accuracy"],
            "validation_macro_f1": final_validation["val_pooled_macro_f1"],
            "validation_ece": final_validation["val_pooled_ece"],
            "test_accuracy_not_for_selection": result["metrics"]["pooled_accuracy"],
            "test_macro_f1_not_for_selection": result["metrics"]["pooled_macro_f1"],
        })
    frame = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    table_output = RESULTS / "tables" / "esc50_validation_tuning.csv"
    frame.to_csv(table_output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
