from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import argparse
from dataclasses import replace
from pathlib import Path

from config import METHODS, RESULTS
from run_experiments import collect_main_table, configure, run_one


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable ESC-50 five-fold federated evaluation")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-tag", default="v2")
    parser.add_argument("--class-balance-power", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    for fold in args.folds:
        for seed in args.seeds:
            for method in args.methods:
                dummy = argparse.Namespace(quick=args.quick, rounds=args.rounds, fold=fold)
                cfg = configure(dummy, "esc50", method, seed)
                cfg = replace(cfg, select_best_validation=True, select_validation_fusion=True,
                              class_balance_power=(args.class_balance_power
                                                   if method == "FedSPECTRA" else 0.0),
                              run_tag=args.run_tag)
                tag = f"_{args.run_tag}" if args.run_tag else ""
                output = RESULTS / "models" / f"run_esc50_fold{fold}_{method}{tag}_seed{seed}.json"
                if output.exists() and not args.force:
                    print(f"skip existing: {output.name}", flush=True)
                    continue
                run_one(cfg)
    collect_main_table()


if __name__ == "__main__":
    main()
