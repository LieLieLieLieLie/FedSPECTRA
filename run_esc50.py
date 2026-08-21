"""Reproduce the ESC-50 cross-validation experiments used in the paper."""

from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import argparse
from dataclasses import replace

from config import METHODS, RESULTS
from run_experiments import collect_main_table, configure, run_one


LOCKED_SUPPORT_POWER = {1: 0.20, 2: 0.30, 3: 0.05, 4: 0.20, 5: 0.20}
DEFAULT_SEEDS = [2027, 2028, 2029]


def _base_config(fold: int, seed: int, rounds: int | None, quick: bool):
    args = argparse.Namespace(quick=quick, rounds=rounds, fold=fold)
    return configure(args, "esc50", "FedSPECTRA", seed)


def run_baselines(args: argparse.Namespace) -> None:
    for fold in args.folds:
        for seed in args.seeds:
            for method in args.methods:
                cfg = replace(
                    _base_config(fold, seed, args.rounds, args.quick),
                    method=method,
                    select_best_validation=True,
                    select_validation_fusion=True,
                    class_balance_power=(args.class_balance_power if method == "FedSPECTRA" else 0.0),
                    run_tag=args.run_tag,
                )
                tag = f"_{args.run_tag}" if args.run_tag else ""
                output = RESULTS / "models" / f"run_esc50_fold{fold}_{method}{tag}_seed{seed}.json"
                if output.exists() and not args.force:
                    print(f"skip existing: {output.name}", flush=True)
                else:
                    run_one(cfg)


def run_locked(args: argparse.Namespace) -> None:
    for fold in args.folds:
        power = LOCKED_SUPPORT_POWER[fold]
        for seed in args.seeds:
            cfg = replace(
                _base_config(fold, seed, args.rounds, args.quick),
                select_best_validation=True,
                select_validation_fusion=True,
                class_balance_power=power,
                run_tag="v4",
            )
            output = RESULTS / "models" / f"run_esc50_fold{fold}_FedSPECTRA_v4_seed{seed}.json"
            if output.exists() and not args.force:
                print(f"skip existing: {output.name}", flush=True)
            else:
                run_one(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable ESC-50 five-fold evaluation")
    parser.add_argument("protocol", choices=["baselines", "locked", "all"], nargs="?", default="all")
    parser.add_argument("--folds", type=int, nargs="+", choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=["FedAvg", "FedExP", "FedDisco", "FPL", "FedLESAM"])
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-tag", default="v2")
    parser.add_argument("--class-balance-power", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.protocol in {"baselines", "all"}:
        run_baselines(args)
    if args.protocol in {"locked", "all"}:
        run_locked(args)
    collect_main_table()


if __name__ == "__main__":
    main()
