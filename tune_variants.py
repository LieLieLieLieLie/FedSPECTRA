from runtime_compat import bootstrap

bootstrap()

import argparse
from dataclasses import replace

from config import SEEDS
from run_experiments import configure, run_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="urbansound8k", choices=["urbansound8k"])
    parser.add_argument("--setting", required=True,
                        choices=["count_no_fusion", "count_fusion", "spectral_no_trajectory", "transport_004"])
    args = parser.parse_args()
    dummy = argparse.Namespace(quick=False, rounds=None)
    changes = {
        "count_no_fusion": dict(use_reliability=False, use_trajectory_stabilization=False,
                                prototype_fusion=0.0),
        "count_fusion": dict(use_reliability=False, use_trajectory_stabilization=False,
                             prototype_fusion=0.20),
        "spectral_no_trajectory": dict(use_reliability=True, use_trajectory_stabilization=False,
                                       prototype_fusion=0.20),
        "transport_004": dict(transport_weight=0.04),
    }[args.setting]
    for seed in SEEDS:
        cfg = configure(dummy, args.dataset, "FedSPECTRA", seed)
        cfg = replace(cfg, variant=f"tune_{args.setting}", **changes)
        run_one(cfg)


if __name__ == "__main__":
    main()
