"""Complete the five-seed UrbanSound8K comparison without rerunning cached seeds."""

from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

from argparse import Namespace

from config import METHODS, RESULTS, SEEDS
from run_experiments import collect_main_table, configure, run_one


def main() -> None:
    args = Namespace(quick=False, rounds=None)
    for seed in SEEDS:
        for method in METHODS:
            path = RESULTS / "models" / f"run_urbansound8k_{method}_seed{seed}.json"
            if path.exists():
                print(f"cached: {path.name}", flush=True)
                continue
            run_one(configure(args, "urbansound8k", method, seed))
    collect_main_table()


if __name__ == "__main__":
    main()
