from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score, log_loss


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(axis=1)
    present = np.unique(labels)
    recalls = [np.mean(pred[labels == c] == c) for c in present]
    return {
        "accuracy": float((pred == labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "nll": float(log_loss(labels, probs, labels=list(range(probs.shape[1])))),
        "ece": expected_calibration_error(probs, labels),
        "brier": brier_score(probs, labels),
    }


def aggregate_client_metrics(client_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key in client_metrics[0]:
        values = np.array([m[key] for m in client_metrics], dtype=float)
        result[key] = float(values.mean())
        result[f"{key}_std_clients"] = float(values.std(ddof=0))
        result[f"{key}_worst"] = float(values.min()) if key not in {"nll", "ece", "brier"} else float(values.max())
    accuracies = np.array([m["accuracy"] for m in client_metrics])
    result["accuracy_p10"] = float(np.quantile(accuracies, 0.10))
    result["accuracy_gap"] = float(accuracies.max() - accuracies.min())
    return result
