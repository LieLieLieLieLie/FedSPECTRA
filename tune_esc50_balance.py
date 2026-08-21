from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import argparse
import contextlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import RESULTS
from run_experiments import configure, run_one


SETTINGS = [
    ("balance005", 0.05),
    ("balance010", 0.10),
    ("balance020", 0.20),
    ("balance030", 0.30),
    ("balance040", 0.40),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only ESC-50 class-support tuning")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        dummy = argparse.Namespace(quick=False, rounds=None, fold=args.fold)
        base = configure(dummy, "esc50", "FedSPECTRA", seed)
        log_path = RESULTS / "models" / f"esc50_balance_tuning_fold{args.fold}_seed{seed}.log"
        with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            for variant, power in SETTINGS:
                cfg = replace(base, variant=variant, run_tag="tune", class_balance_power=power,
                              select_best_validation=True, select_validation_fusion=True)
                output = RESULTS / "models" / f"run_esc50_fold{args.fold}_FedSPECTRA_{variant}_tune_seed{seed}.json"
                if output.exists() and not args.force:
                    result = json.loads(output.read_text(encoding="utf-8"))
                else:
                    result = run_one(cfg)
                rows.append({
                    "seed": seed,
                    "test_fold": args.fold,
                    "validation_fold": cfg.esc50_val_fold,
                    "variant": variant,
                    "class_balance_power": power,
                    "selected_round": result["selected_round"],
                    "selected_prototype_fusion": result["selected_prototype_fusion"],
                    "validation_macro_f1": result["selected_validation_macro_f1"],
                })
    output = RESULTS / "tables" / f"esc50_balance_validation_fold{args.fold}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
