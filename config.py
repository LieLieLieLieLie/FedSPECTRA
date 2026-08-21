from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results"
DATA_ROOT = Path(os.environ.get("FEDSPECTRA_DATA_ROOT", ROOT / "data")).expanduser()
URBANSOUND_ROOT = Path(
    os.environ.get("FEDSPECTRA_URBANSOUND_ROOT", DATA_ROOT / "UrbanSound8K")
).expanduser()
ESC50_ROOT = Path(
    os.environ.get("FEDSPECTRA_ESC50_ROOT", DATA_ROOT / "ESC-50-master")
).expanduser()


@dataclass
class ExperimentConfig:
    dataset: str = "urbansound8k"
    seed: int = 2027
    method: str = "FedSPECTRA"
    variant: str = "full"
    num_clients: int = 16
    clients_per_round: int = 8
    rounds: int = 36
    local_steps: int = 6
    batch_size: int = 64
    learning_rate: float = 0.025
    weight_decay: float = 1e-4
    momentum: float = 0.9
    dirichlet_alpha: float = 0.3
    min_client_samples: int = 80
    mel_bins: int = 64
    time_frames: int = 96
    embedding_dim: int = 96
    spectral_bands: int = 8
    prototype_weight: float = 0.01
    transport_weight: float = 0.01
    prototype_fusion: float = 0.50
    fpl_weight: float = 0.20
    fedlesam_rho: float = 0.05
    conflict_floor: float = 0.90
    consensus_energy: float = 0.82
    reliability_temperature: float = 0.45
    reliability_blend: float = 0.35
    use_reliability: bool = True
    use_label_reliability: bool = True
    use_spectral_reliability: bool = True
    use_trajectory_stabilization: bool = True
    use_consensus: bool = False
    participation_rate: float = 0.5
    device_shift_strength: float = 0.32
    noise_strength: float = 0.10
    eval_interval: int = 2
    device: str = "cuda"
    cache_version: str = "v1"
    quick: bool = False
    esc50_test_fold: int = 1
    esc50_val_fold: int = 2
    select_best_validation: bool = False
    select_validation_fusion: bool = False
    class_balance_power: float = 0.0
    run_tag: str = ""

    @property
    def num_classes(self) -> int:
        return {"urbansound8k": 10, "esc50": 50}[self.dataset]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


METHODS = ["FedAvg", "FedExP", "FedDisco", "FPL", "FedLESAM", "FedSPECTRA"]
SEEDS = [2027, 2028, 2029, 2030, 2031]
