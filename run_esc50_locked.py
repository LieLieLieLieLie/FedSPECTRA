from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import argparse
from dataclasses import replace

from config import RESULTS
from run_experiments import collect_main_table, configure, run_one


LOCKED_SUPPORT_POWER = {1: 0.20, 2: 0.30, 3: 0.05, 4: 0.20, 5: 0.20}
SEEDS = [2027, 2028, 2029]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce validation-locked ESC-50 FedSPECTRA runs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for fold, power in LOCKED_SUPPORT_POWER.items():
        for seed in SEEDS:
            dummy = argparse.Namespace(quick=False, rounds=None, fold=fold)
            cfg = configure(dummy, "esc50", "FedSPECTRA", seed)
            cfg = replace(cfg, select_best_validation=True, select_validation_fusion=True,
                          class_balance_power=power, run_tag="v4")
            output = RESULTS / "models" / f"run_esc50_fold{fold}_FedSPECTRA_v4_seed{seed}.json"
            if output.exists() and not args.force:
                print(f"skip existing: {output.name}", flush=True)
                continue
            run_one(cfg)
    collect_main_table()


if __name__ == "__main__":
    main()
