from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import copy
import json
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from config import RESULTS, ExperimentConfig
from metrics import aggregate_client_metrics, classification_metrics
from models import SpectralEncoder, assign_flat_parameters, flatten_parameters, parameter_count


@dataclass
class PrototypeBank:
    embedding: torch.Tensor
    spectral: torch.Tensor
    dispersion: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def empty(cls, classes: int, dim: int, bands: int, device: str):
        return cls(torch.zeros(classes, dim, device=device),
                   torch.full((classes, bands), 1.0 / bands, device=device),
                   torch.ones(classes, device=device),
                   torch.zeros(classes, dtype=torch.bool, device=device))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _prototype_losses(embedding: torch.Tensor, spectral: torch.Tensor, y: torch.Tensor,
                      bank: PrototypeBank) -> Tuple[torch.Tensor, torch.Tensor]:
    available = bank.valid[y]
    if not available.any():
        zero = embedding.sum() * 0.0
        return zero, zero
    z = F.normalize(embedding[available], dim=1)
    p = F.normalize(bank.embedding, dim=1)
    logits = z @ p.t() / 0.15
    logits[:, ~bank.valid] = -1e4
    proto = F.cross_entropy(logits, y[available])
    cdf = spectral[available].cumsum(dim=1)
    target_cdf = bank.spectral[y[available]].cumsum(dim=1)
    weights = (1.0 / (bank.dispersion[y[available]] + 0.1)).clamp(max=5.0)
    transport = ((cdf - target_cdf).abs().mean(dim=1) * weights).mean()
    return proto, transport


def _stats_from_accumulators(sums_z, sums_z2, sums_s, counts) -> Dict[int, Dict[str, torch.Tensor]]:
    stats: Dict[int, Dict[str, torch.Tensor]] = {}
    for c, count in counts.items():
        n = max(1, int(count))
        mean = sums_z[c] / n
        second = sums_z2[c] / n
        dispersion = torch.sqrt((second - mean.square()).clamp_min(0).mean() + 1e-8)
        spectral = (sums_s[c] / n).clamp_min(1e-7)
        spectral = spectral / spectral.sum()
        stats[c] = {"count": torch.tensor(float(n)), "embedding": mean,
                    "spectral": spectral, "dispersion": dispersion}
    return stats


def train_client(global_model: nn.Module, dataset, cfg: ExperimentConfig, method: str,
                 bank: PrototypeBank, previous_received: Optional[torch.Tensor],
                 round_idx: int = 0) -> Tuple[nn.Module, Dict, float]:
    model = copy.deepcopy(global_model).to(cfg.device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
                        drop_last=len(dataset) >= cfg.batch_size)
    class_weights = None
    if method.startswith("FedSPECTRA") and cfg.class_balance_power > 0:
        local_labels = dataset.y[dataset.indices]
        class_counts = torch.bincount(local_labels, minlength=cfg.num_classes).float().to(cfg.device)
        present = class_counts > 0
        class_weights = torch.zeros_like(class_counts)
        class_weights[present] = (class_counts[present] + 1.0).pow(-cfg.class_balance_power)
        class_weights[present] /= class_weights[present].mean()
    iterator = iter(loader)
    current_flat = flatten_parameters(global_model).to(cfg.device)
    perturb = None
    if (method == "FedLESAM" or
            (method.startswith("FedSPECTRA") and cfg.use_trajectory_stabilization)) and previous_received is not None:
        direction = previous_received.to(cfg.device) - current_flat
        perturb = cfg.fedlesam_rho * direction / (direction.norm() + 1e-12)

    sums_z, sums_z2, sums_s, counts = {}, {}, {}, {}
    total_loss = 0.0
    for _ in range(cfg.local_steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
        if perturb is not None:
            with torch.no_grad():
                offset = 0
                for p in model.parameters():
                    n = p.numel()
                    p.add_(perturb[offset:offset + n].view_as(p))
                    offset += n
        optimizer.zero_grad(set_to_none=True)
        logits, embedding, spectral = model(x)
        ce = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=0.03)
        proto, transport = _prototype_losses(embedding, spectral, y, bank)
        loss = ce
        if method == "FPL":
            loss = loss + cfg.fpl_weight * proto
        elif method.startswith("FedSPECTRA"):
            ramp = min(1.0, max(0.0, (round_idx - 1) / 5.0))
            loss = loss + ramp * (cfg.prototype_weight * proto + cfg.transport_weight * transport)
        loss.backward()
        if perturb is not None:
            with torch.no_grad():
                offset = 0
                for p in model.parameters():
                    n = p.numel()
                    p.sub_(perturb[offset:offset + n].view_as(p))
                    offset += n
        torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
        optimizer.step()
        total_loss += float(loss.detach())
        with torch.no_grad():
            for c in y.unique().tolist():
                mask = y == c
                zc, sc = embedding[mask], spectral[mask]
                if c not in counts:
                    sums_z[c] = zc.sum(0)
                    sums_z2[c] = zc.square().sum(0)
                    sums_s[c] = sc.sum(0)
                    counts[c] = int(mask.sum())
                else:
                    sums_z[c] += zc.sum(0)
                    sums_z2[c] += zc.square().sum(0)
                    sums_s[c] += sc.sum(0)
                    counts[c] += int(mask.sum())
    return model, _stats_from_accumulators(sums_z, sums_z2, sums_s, counts), total_loss / cfg.local_steps


def _weighted_median(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, dim=0)
    sorted_values = torch.gather(values, 0, order)
    expanded = weights[:, None].expand_as(values)
    sorted_weights = torch.gather(expanded, 0, order)
    cdf = sorted_weights.cumsum(0)
    threshold = weights.sum() * 0.5
    index = (cdf >= threshold).float().argmax(0, keepdim=True)
    return torch.gather(sorted_values, 0, index).squeeze(0)


@torch.no_grad()
def update_bank(bank: PrototypeBank, client_stats: Sequence[Dict], momentum: float = 0.35) -> PrototypeBank:
    new = PrototypeBank(bank.embedding.clone(), bank.spectral.clone(), bank.dispersion.clone(), bank.valid.clone())
    for c in range(len(bank.valid)):
        entries = [s[c] for s in client_stats if c in s]
        if not entries:
            continue
        emb = torch.stack([e["embedding"] for e in entries])
        spec = torch.stack([e["spectral"] for e in entries])
        disp = torch.stack([e["dispersion"] for e in entries])
        count = torch.stack([e["count"] for e in entries]).to(emb.device)
        weights = count / (disp + 0.08)
        weights = weights / weights.sum()
        emb_target = (emb * weights[:, None]).sum(0)
        cdfs = spec.cumsum(1)
        median_cdf = _weighted_median(cdfs, weights)
        median_cdf = torch.cummax(median_cdf.clamp(0, 1), dim=0).values
        spec_target = torch.diff(torch.cat([torch.zeros(1, device=spec.device), median_cdf])).clamp_min(1e-6)
        spec_target = spec_target / spec_target.sum()
        disp_target = (disp * weights).sum()
        if bank.valid[c]:
            new.embedding[c] = (1 - momentum) * bank.embedding[c] + momentum * emb_target
            new.spectral[c] = (1 - momentum) * bank.spectral[c] + momentum * spec_target
            new.spectral[c] /= new.spectral[c].sum()
            new.dispersion[c] = (1 - momentum) * bank.dispersion[c] + momentum * disp_target
        else:
            new.embedding[c] = emb_target
            new.spectral[c] = spec_target
            new.dispersion[c] = disp_target
            new.valid[c] = True
    return new


def _spectral_residual(stats: Dict, bank: PrototypeBank) -> float:
    values, weights = [], []
    for c, entry in stats.items():
        if bank.valid[c]:
            w1 = (entry["spectral"].cumsum(0) - bank.spectral[c].cumsum(0)).abs().mean()
            values.append(float(w1))
            weights.append(float(entry["count"]))
    return float(np.average(values, weights=weights)) if values else 0.0


def _class_distribution(dataset, classes: int) -> np.ndarray:
    labels = dataset.y[dataset.indices].numpy()
    hist = np.bincount(labels, minlength=classes).astype(float)
    return hist / hist.sum()


@torch.no_grad()
def aggregate(global_model: nn.Module, local_models: Sequence[nn.Module], selected_datasets,
              client_stats: Sequence[Dict], bank: PrototypeBank, cfg: ExperimentConfig,
              method: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    base = flatten_parameters(global_model).to(cfg.device)
    updates = torch.stack([flatten_parameters(m).to(cfg.device) - base for m in local_models])
    sample_counts = torch.tensor([len(d) for d in selected_datasets], device=cfg.device, dtype=torch.float32)
    weights = sample_counts / sample_counts.sum()
    diagnostics: Dict[str, float] = {}

    if method == "FedDisco":
        priors = np.stack([_class_distribution(d, cfg.num_classes) for d in selected_datasets])
        global_prior = np.average(priors, axis=0, weights=sample_counts.cpu().numpy())
        discrepancy = np.abs(priors - global_prior).sum(1)
        modifier = torch.tensor(np.exp(-0.75 * discrepancy), device=cfg.device, dtype=torch.float32)
        weights = weights * modifier
        weights /= weights.sum()
        diagnostics["mean_label_discrepancy"] = float(discrepancy.mean())

    if method.startswith("FedSPECTRA"):
        priors = np.stack([_class_distribution(d, cfg.num_classes) for d in selected_datasets])
        global_prior = np.average(priors, axis=0, weights=sample_counts.cpu().numpy())
        label_discrepancy = np.abs(priors - global_prior).sum(1)
        label_modifier = torch.tensor(np.exp(-0.75 * label_discrepancy), device=cfg.device,
                                      dtype=torch.float32)
        residual = torch.tensor([_spectral_residual(s, bank) for s in client_stats],
                                device=cfg.device, dtype=torch.float32)
        dispersion = torch.tensor([
            np.average([float(e["dispersion"]) for e in s.values()],
                       weights=[float(e["count"]) for e in s.values()]) if s else 1.0
            for s in client_stats], device=cfg.device, dtype=torch.float32)
        reliability = torch.exp(-0.05 * residual / cfg.reliability_temperature - 0.02 * dispersion)
        if cfg.use_reliability:
            modifier = torch.ones_like(weights)
            if cfg.use_label_reliability:
                modifier = modifier * label_modifier
            if cfg.use_spectral_reliability:
                modifier = modifier * reliability
            # Conservative shrinkage toward sample-count weighting prevents a noisy
            # early prototype bank from dominating the server update.
            modifier = (1.0 - cfg.reliability_blend) + cfg.reliability_blend * modifier
            weights = weights * modifier
        weights = weights / weights.sum().clamp_min(1e-12)
        anchor = (weights[:, None] * updates).sum(0)
        if cfg.use_consensus:
            normalized = updates / updates.norm(dim=1, keepdim=True).clamp_min(1e-12)
            _, singular, vh = torch.linalg.svd(normalized, full_matrices=False)
            energy = singular.square()
            rank = int((energy.cumsum(0) / energy.sum() < cfg.consensus_energy).sum().item() + 1)
            basis = vh[:rank]
            projected = (updates @ basis.t()) @ basis
            projected_anchor = (weights[:, None] * projected).sum(0)
            cosine = F.cosine_similarity(projected, projected_anchor.unsqueeze(0), dim=1)
            full_dot = updates @ anchor
            negative = full_dot.clamp_max(0.0)
            corrected = updates - (negative / anchor.square().sum().clamp_min(1e-12))[:, None] * anchor
            delta = (weights[:, None] * corrected).sum(0)
        else:
            rank = updates.shape[0]
            cosine = F.cosine_similarity(updates, anchor.unsqueeze(0), dim=1)
            delta = anchor
        conflict_rate = float((cosine < 0).float().mean())
        diagnostics.update({"spectral_residual": float((weights * residual).sum()),
                            "mean_reliability": float(reliability.mean()),
                            "mean_label_discrepancy": float(label_discrepancy.mean()),
                            "conflict_rate": conflict_rate, "consensus_rank": float(rank),
                            "retained_update_energy": float(delta.square().sum() / (anchor.square().sum() + 1e-12))})
    else:
        delta = (weights[:, None] * updates).sum(0)

    if method == "FedExP":
        numerator = (weights * updates.square().sum(1)).sum()
        denominator = 2.0 * delta.square().sum().clamp_min(1e-12)
        eta = float(torch.clamp(numerator / denominator, 1.0, 1.5))
        delta = eta * delta
        diagnostics["server_extrapolation"] = eta
    assign_flat_parameters(global_model, base + delta)
    diagnostics["update_norm"] = float(delta.norm())
    diagnostics["weight_entropy"] = float(-(weights * weights.clamp_min(1e-12).log()).sum())
    return weights.detach().cpu(), diagnostics


@torch.no_grad()
def evaluate(model: nn.Module, clients, cfg: ExperimentConfig, save_predictions: Optional[Path] = None,
             bank: Optional[PrototypeBank] = None):
    model.eval()
    client_results, all_probs, all_labels, all_client = [], [], [], []
    for cid, dataset in enumerate(clients):
        probs_list, labels_list = [], []
        loader = DataLoader(dataset, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0)
        for x, y in loader:
            logits, embedding, _ = model(x.to(cfg.device))
            if bank is not None and bank.valid.any():
                proto_logits = F.normalize(embedding, dim=1) @ F.normalize(bank.embedding, dim=1).t() / 0.15
                proto_logits[:, ~bank.valid] = -1e4
                logits = logits + cfg.prototype_fusion * proto_logits
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
            labels_list.append(y.numpy())
        probs = np.concatenate(probs_list)
        labels = np.concatenate(labels_list)
        client_results.append(classification_metrics(probs, labels))
        all_probs.append(probs)
        all_labels.append(labels)
        all_client.append(np.full(len(labels), cid))
    aggregate_metrics = aggregate_client_metrics(client_results)
    pooled = classification_metrics(np.concatenate(all_probs), np.concatenate(all_labels))
    aggregate_metrics.update({f"pooled_{k}": v for k, v in pooled.items()})
    if save_predictions:
        np.savez_compressed(save_predictions, probs=np.concatenate(all_probs), labels=np.concatenate(all_labels),
                            client=np.concatenate(all_client))
    return aggregate_metrics, client_results


def run_federated(cfg: ExperimentConfig, train_clients, val_clients, test_clients) -> Dict:
    seed_everything(cfg.seed)
    model = SpectralEncoder(cfg.num_classes, cfg.embedding_dim, cfg.spectral_bands).to(cfg.device)
    bank = PrototypeBank.empty(cfg.num_classes, cfg.embedding_dim, cfg.spectral_bands, cfg.device)
    rng = np.random.default_rng(cfg.seed + 17)
    previous_received: Dict[int, torch.Tensor] = {}
    history: List[Dict] = []
    best_validation = -float("inf")
    best_round = cfg.rounds
    best_state = None
    best_bank = None
    best_fusion = cfg.prototype_fusion
    start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for round_idx in range(cfg.rounds):
        selected = rng.choice(len(train_clients), size=min(cfg.clients_per_round, len(train_clients)), replace=False)
        received = flatten_parameters(model).detach().cpu()
        local_models, stats, losses = [], [], []
        for cid in selected:
            local, stat, loss = train_client(model, train_clients[cid], cfg, cfg.method, bank,
                                             previous_received.get(int(cid)), round_idx)
            local_models.append(local)
            stats.append(stat)
            losses.append(loss)
            previous_received[int(cid)] = received.clone()
        weights, diagnostics = aggregate(model, local_models, [train_clients[i] for i in selected],
                                         stats, bank, cfg, cfg.method)
        if cfg.method == "FPL" or cfg.method.startswith("FedSPECTRA"):
            bank = update_bank(bank, stats)
        record = {"round": round_idx + 1, "train_loss": float(np.mean(losses)), **diagnostics}
        if (round_idx + 1) % cfg.eval_interval == 0 or round_idx == cfg.rounds - 1:
            validation_bank = bank if cfg.method.startswith("FedSPECTRA") else None
            validation_fusion = cfg.prototype_fusion
            if cfg.method.startswith("FedSPECTRA") and cfg.select_validation_fusion:
                candidates = []
                for fusion in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
                    metrics, _ = evaluate(model, val_clients,
                                          replace(cfg, prototype_fusion=fusion), bank=validation_bank)
                    candidates.append((float(metrics["pooled_macro_f1"]), -fusion, fusion, metrics))
                _, _, validation_fusion, val_metrics = max(candidates, key=lambda item: (item[0], item[1]))
                record["val_prototype_fusion"] = validation_fusion
            else:
                val_metrics, _ = evaluate(model, val_clients, cfg, bank=validation_bank)
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            score = float(val_metrics["pooled_macro_f1"])
            if cfg.select_best_validation and score > best_validation:
                best_validation = score
                best_round = round_idx + 1
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
                best_bank = PrototypeBank(bank.embedding.detach().cpu().clone(),
                                          bank.spectral.detach().cpu().clone(),
                                          bank.dispersion.detach().cpu().clone(),
                                          bank.valid.detach().cpu().clone())
                best_fusion = validation_fusion
        history.append(record)
        del local_models

    elapsed = time.perf_counter() - start_time
    if cfg.select_best_validation and best_state is not None:
        model.load_state_dict(best_state)
        bank = PrototypeBank(best_bank.embedding.to(cfg.device), best_bank.spectral.to(cfg.device),
                             best_bank.dispersion.to(cfg.device), best_bank.valid.to(cfg.device))
        cfg.prototype_fusion = best_fusion
    variant_suffix = "" if cfg.variant == "full" else f"_{cfg.variant}"
    fold_suffix = f"_fold{cfg.esc50_test_fold}" if cfg.dataset == "esc50" else ""
    tag_suffix = f"_{cfg.run_tag}" if cfg.run_tag else ""
    stem = f"{cfg.dataset}{fold_suffix}_{cfg.method}{variant_suffix}{tag_suffix}_seed{cfg.seed}"
    pred_path = RESULTS / "models" / f"predictions_{stem}.npz"
    inference_bank = bank if cfg.method.startswith("FedSPECTRA") else None
    test_metrics, per_client = evaluate(model, test_clients, cfg, pred_path, inference_bank)
    checkpoint_path = RESULTS / "models" / f"checkpoint_{stem}.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                "bank": {"embedding": bank.embedding.cpu(), "spectral": bank.spectral.cpu(),
                         "dispersion": bank.dispersion.cpu(), "valid": bank.valid.cpu()}}, checkpoint_path)
    params = parameter_count(model)
    model_mb = params * 4 / 1e6
    extra_stats = 0.0
    if cfg.method == "FPL":
        extra_stats = cfg.num_classes * (cfg.embedding_dim + 2) * 4 / 1e6
    elif cfg.method.startswith("FedSPECTRA"):
        extra_stats = cfg.num_classes * (cfg.embedding_dim + cfg.spectral_bands + 3) * 4 / 1e6
    communication = cfg.rounds * cfg.clients_per_round * (2 * model_mb + extra_stats)
    result = {
        "config": cfg.to_dict(), "metrics": test_metrics, "per_client": per_client,
        "history": history, "elapsed_seconds": elapsed, "parameter_count": params,
        "model_size_mb": model_mb, "communication_mb": communication,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0,
        "selected_round": best_round if cfg.select_best_validation else cfg.rounds,
        "selected_validation_macro_f1": best_validation if cfg.select_best_validation else None,
        "selected_prototype_fusion": best_fusion if cfg.select_validation_fusion else cfg.prototype_fusion,
        "checkpoint": str(checkpoint_path), "predictions": str(pred_path),
    }
    out = RESULTS / "models" / f"run_{stem}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
