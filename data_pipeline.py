from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import gc
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from config import ESC50_ROOT, RESULTS, URBANSOUND_ROOT, ExperimentConfig


@dataclass
class SignalData:
    x: torch.Tensor
    y: torch.Tensor
    domain: torch.Tensor
    class_names: List[str]


def _sample_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
    return (x - mean) / std


def _resize_spec(spec: torch.Tensor, bins: int, frames: int) -> torch.Tensor:
    if spec.ndim == 3:
        spec = spec.unsqueeze(1)
    return F.interpolate(spec, size=(bins, frames), mode="bilinear", align_corners=False).squeeze(1)


def prepare_urbansound(cfg: ExperimentConfig, force: bool = False) -> Path:
    cache = RESULTS / "models" / f"urbansound8k_spectrogram_{cfg.cache_version}.pt"
    if cache.exists() and not force:
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, n_fft=1024, win_length=882, hop_length=441,
        n_mels=cfg.mel_bins, f_min=40.0, f_max=10000.0, power=2.0,
    ).to(cfg.device)
    all_x, all_y, all_fold = [], [], []
    processed = URBANSOUND_ROOT / "processed"
    metadata_candidates = [
        URBANSOUND_ROOT / "metadata" / "UrbanSound8K.csv",
        URBANSOUND_ROOT / "UrbanSound8K.csv",
    ]
    metadata = next((path for path in metadata_candidates if path.exists()), None)
    audio_root = URBANSOUND_ROOT / "audio" if (URBANSOUND_ROOT / "audio").exists() else URBANSOUND_ROOT
    if not processed.exists() and metadata is None:
        raise FileNotFoundError(
            f"UrbanSound8K was not found at {URBANSOUND_ROOT}. "
            "Expected either the official audio/ and metadata/ directories or "
            "the processed/fold_*.npy layout."
        )
    metadata_rows = None
    if metadata is not None:
        with metadata.open("r", encoding="utf-8-sig", newline="") as stream:
            metadata_rows = list(csv.DictReader(stream))
        metadata_rows.sort(key=lambda row: (int(row["fold"]), row["slice_file_name"]))
    resamplers: Dict[int, torchaudio.transforms.Resample] = {}
    target_length = 4 * 22050
    for fold in range(1, 11):
        processed_fold = processed / f"fold_{fold}.npy"
        if processed_fold.exists():
            records = np.load(processed_fold, allow_pickle=True)
            batches = [records[start:start + 96] for start in range(0, len(records), 96)]
            iterator = ((torch.from_numpy(np.stack([r["waveform"] for r in batch])),
                         [int(r["target"]) for r in batch]) for batch in batches)
        else:
            if metadata_rows is None:
                raise FileNotFoundError(
                    f"Missing {processed_fold.name} and UrbanSound8K metadata under {URBANSOUND_ROOT}."
                )
            fold_rows = [row for row in metadata_rows if int(row["fold"]) == fold]
            raw_batches = [fold_rows[start:start + 32] for start in range(0, len(fold_rows), 32)]

            def raw_iterator():
                for batch in raw_batches:
                    waves = []
                    for row in batch:
                        wave, sample_rate = torchaudio.load(
                            audio_root / f"fold{fold}" / row["slice_file_name"]
                        )
                        wave = wave.mean(dim=0)
                        if sample_rate != 22050:
                            if sample_rate not in resamplers:
                                resamplers[sample_rate] = torchaudio.transforms.Resample(sample_rate, 22050)
                            wave = resamplers[sample_rate](wave)
                        if wave.numel() < target_length:
                            wave = F.pad(wave, (0, target_length - wave.numel()))
                        waves.append(wave[:target_length])
                    yield torch.stack(waves), [int(row["classID"]) for row in batch]

            iterator = raw_iterator()
        for wave, labels in iterator:
            wave = wave.to(cfg.device)
            with torch.no_grad():
                spec = torch.log1p(mel(wave))
                spec = _resize_spec(spec, cfg.mel_bins, cfg.time_frames)
                spec = _sample_normalize(spec).half().cpu()
            all_x.append(spec)
            all_y.append(torch.tensor(labels, dtype=torch.long))
            all_fold.append(torch.full((len(labels),), fold, dtype=torch.long))
        gc.collect()
    payload = {
        "x": torch.cat(all_x), "y": torch.cat(all_y), "domain": torch.cat(all_fold),
        "class_names": ["air conditioner", "car horn", "children playing", "dog bark",
                        "drilling", "engine idling", "gun shot", "jackhammer", "siren", "street music"],
        "source": str(URBANSOUND_ROOT),
    }
    torch.save(payload, cache)
    return cache
def prepare_esc50(cfg: ExperimentConfig, force: bool = False) -> Path:
    """Create a compact log-mel cache while preserving ESC-50's official folds."""
    cache = RESULTS / "models" / f"esc50_spectrogram_{cfg.cache_version}.pt"
    if cache.exists() and not force:
        return cache
    metadata = ESC50_ROOT / "meta" / "esc50.csv"
    audio_root = ESC50_ROOT / "audio"
    if not metadata.exists() or not audio_root.exists():
        raise FileNotFoundError(
            f"ESC-50 was not found at {ESC50_ROOT}. Clone https://github.com/karolpiczak/esc-50 there."
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with metadata.open("r", encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    records.sort(key=lambda row: (int(row["fold"]), row["filename"]))
    target_to_name = {int(row["target"]): row["category"] for row in records}
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, n_fft=1024, win_length=882, hop_length=441,
        n_mels=cfg.mel_bins, f_min=40.0, f_max=10000.0, power=2.0,
    ).to(cfg.device)
    resample = torchaudio.transforms.Resample(44100, 22050)
    target_length = 5 * 22050
    all_x, all_y, all_fold = [], [], []
    for start in range(0, len(records), 32):
        batch = records[start:start + 32]
        waves = []
        for row in batch:
            wave, sample_rate = torchaudio.load(audio_root / row["filename"])
            wave = wave.mean(dim=0)
            if sample_rate != 22050:
                wave = resample(wave) if sample_rate == 44100 else torchaudio.functional.resample(
                    wave, sample_rate, 22050
                )
            if wave.numel() < target_length:
                wave = F.pad(wave, (0, target_length - wave.numel()))
            waves.append(wave[:target_length])
        wave_batch = torch.stack(waves).to(cfg.device)
        with torch.no_grad():
            spec = torch.log1p(mel(wave_batch))
            spec = _resize_spec(spec, cfg.mel_bins, cfg.time_frames)
            spec = _sample_normalize(spec).half().cpu()
        all_x.append(spec)
        all_y.append(torch.tensor([int(row["target"]) for row in batch], dtype=torch.long))
        all_fold.append(torch.tensor([int(row["fold"]) for row in batch], dtype=torch.long))
    torch.save({
        "x": torch.cat(all_x), "y": torch.cat(all_y), "domain": torch.cat(all_fold),
        "class_names": [target_to_name[i] for i in range(50)], "source": str(ESC50_ROOT),
    }, cache)
    return cache


def load_signal_data(cfg: ExperimentConfig) -> SignalData:
    if cfg.dataset == "urbansound8k":
        path = prepare_urbansound(cfg)
    elif cfg.dataset == "esc50":
        path = prepare_esc50(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    payload = torch.load(path, map_location="cpu")
    return SignalData(payload["x"].float(), payload["y"], payload["domain"], payload["class_names"])


def train_val_test_indices(data: SignalData, cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = data.domain.numpy()
    if cfg.dataset == "urbansound8k":
        return np.flatnonzero(d <= 8), np.flatnonzero(d == 9), np.flatnonzero(d == 10)
    if cfg.dataset == "esc50":
        held_out = {cfg.esc50_test_fold, cfg.esc50_val_fold}
        train = np.flatnonzero(~np.isin(d, list(held_out)))
        return train, np.flatnonzero(d == cfg.esc50_val_fold), np.flatnonzero(d == cfg.esc50_test_fold)
    return np.flatnonzero(d <= 1), np.flatnonzero(d == 2), np.flatnonzero(d == 3)


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float,
                        min_size: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    best: List[List[int]] = []
    for _ in range(200):
        buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for cls in classes:
            idx = np.flatnonzero(labels == cls)
            rng.shuffle(idx)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            capacity = np.array([len(b) < len(labels) / num_clients * 1.4 for b in buckets], dtype=float)
            proportions = proportions * capacity
            proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
            for client, part in enumerate(np.split(idx, cuts)):
                buckets[client].extend(part.tolist())
        if min(map(len, buckets)) >= min_size:
            best = buckets
            break
        if not best or min(map(len, buckets)) > min(map(len, best)):
            best = buckets
    if min(map(len, best)) < min_size:
        order = rng.permutation(len(labels))
        best = [chunk.tolist() for chunk in np.array_split(order, num_clients)]
    return [np.asarray(sorted(b), dtype=np.int64) for b in best]


def make_frequency_response(client_id: int, bins: int, strength: float, seed: int,
                            unseen: bool = False) -> torch.Tensor:
    rng = np.random.default_rng(seed + 104729 * (client_id + (1000 if unseen else 0)))
    knots = rng.normal(0.0, strength * (1.25 if unseen else 1.0), size=6)
    response = np.interp(np.linspace(0, 5, bins), np.arange(6), knots)
    response += (0.08 if unseen else 0.04) * np.sin(np.linspace(0, rng.uniform(2, 5) * np.pi, bins))
    response -= response.mean()
    return torch.tensor(response, dtype=torch.float32).view(1, bins, 1)


class ClientSignalDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, indices: Sequence[int], response: torch.Tensor,
                 noise: float, train: bool, seed: int):
        self.x = x
        self.y = y
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.response = response
        self.noise = noise
        self.train = train
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        x = self.x[idx].unsqueeze(0) + self.response
        if self.train and self.noise > 0:
            x = x + torch.randn(x.shape, generator=self.generator) * self.noise
            if torch.rand((), generator=self.generator) < 0.25:
                width = int(torch.randint(2, 7, (), generator=self.generator))
                start = int(torch.randint(0, max(1, x.shape[-1] - width), (), generator=self.generator))
                x[:, :, start:start + width] = 0
        return x, self.y[idx]


def build_client_datasets(cfg: ExperimentConfig, data: SignalData):
    train_idx, val_idx, test_idx = train_val_test_indices(data, cfg)
    local_partition = dirichlet_partition(data.y[train_idx].numpy(), cfg.num_clients,
                                          cfg.dirichlet_alpha, cfg.min_client_samples, cfg.seed)
    train_clients = []
    for cid, local in enumerate(local_partition):
        absolute = train_idx[local]
        response = make_frequency_response(cid, cfg.mel_bins, cfg.device_shift_strength, cfg.seed)
        train_clients.append(ClientSignalDataset(data.x, data.y, absolute, response,
                                                  cfg.noise_strength, True, cfg.seed + cid))

    def _eval_group(indices: np.ndarray, count: int, offset: int):
        relative = dirichlet_partition(data.y[indices].numpy(), count, 0.65,
                                       max(5, len(indices) // (count * 4)), cfg.seed + offset)
        chunks = [indices[r] for r in relative]
        return [ClientSignalDataset(data.x, data.y, c,
                                    make_frequency_response(i + offset, cfg.mel_bins,
                                                            cfg.device_shift_strength, cfg.seed, unseen=True),
                                    0.0, False, cfg.seed + offset + i)
                for i, c in enumerate(chunks)]

    val_clients = _eval_group(val_idx, min(6, len(val_idx)), 100)
    test_clients = _eval_group(test_idx, min(8, len(test_idx)), 200)
    return train_clients, val_clients, test_clients


def dataset_summary(cfg: ExperimentConfig, data: SignalData, clients: Sequence[Dataset]) -> Dict[str, object]:
    train_idx, val_idx, test_idx = train_val_test_indices(data, cfg)
    counts = [len(c) for c in clients]
    return {
        "dataset": cfg.dataset,
        "samples": len(data.y),
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx),
        "classes": len(data.class_names),
        "clients": len(clients),
        "client_min": min(counts),
        "client_median": float(np.median(counts)),
        "client_max": max(counts),
        "test_fold": cfg.esc50_test_fold if cfg.dataset == "esc50" else None,
        "validation_fold": cfg.esc50_val_fold if cfg.dataset == "esc50" else None,
    }

