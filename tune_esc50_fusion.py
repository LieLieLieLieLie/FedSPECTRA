from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import json
from dataclasses import replace

import pandas as pd
import torch

from config import ExperimentConfig, RESULTS
from data_pipeline import build_client_datasets, load_signal_data
from federated import PrototypeBank, evaluate
from models import SpectralEncoder


# Each outer test fold uses only its designated cyclic validation fold to
# select the fusion coefficient.  No test prediction is read.
FUSION_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
SEEDS = [2027, 2028, 2029]


def main() -> None:
    rows = []
    for fold in range(1, 6):
        for seed in SEEDS:
            run_path = RESULTS / "models" / f"run_esc50_fold{fold}_FedSPECTRA_seed{seed}.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            cfg = ExperimentConfig(**run["config"])
            data = load_signal_data(cfg)
            _, val_clients, _ = build_client_datasets(cfg, data)
            checkpoint = torch.load(run["checkpoint"], map_location=cfg.device)
            model = SpectralEncoder(cfg.num_classes, cfg.embedding_dim, cfg.spectral_bands).to(cfg.device)
            model.load_state_dict(checkpoint["state_dict"])
            raw_bank = checkpoint["bank"]
            bank = PrototypeBank(
                raw_bank["embedding"].to(cfg.device),
                raw_bank["spectral"].to(cfg.device),
                raw_bank["dispersion"].to(cfg.device),
                raw_bank["valid"].to(cfg.device),
            )
            for fusion in FUSION_GRID:
                metrics, _ = evaluate(model, val_clients, replace(cfg, prototype_fusion=fusion), bank=bank)
                rows.append({
                    "test_fold": fold,
                    "validation_fold": cfg.esc50_val_fold,
                    "seed": seed,
                    "prototype_fusion": fusion,
                    "validation_accuracy": metrics["pooled_accuracy"],
                    "validation_macro_f1": metrics["pooled_macro_f1"],
                    "validation_ece": metrics["pooled_ece"],
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "tables" / "esc50_fusion_validation_seedwise.csv", index=False)
    summary = frame.groupby(["test_fold", "validation_fold", "prototype_fusion"], as_index=False).agg(
        validation_accuracy=("validation_accuracy", "mean"),
        validation_macro_f1=("validation_macro_f1", "mean"),
        validation_ece=("validation_ece", "mean"),
    ).sort_values("validation_macro_f1", ascending=False)
    summary.to_csv(RESULTS / "tables" / "esc50_fusion_validation.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
