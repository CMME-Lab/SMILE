from __future__ import annotations

from typing import Sequence

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

__all__ = ["binary_metrics", "average", "KEYS"]

KEYS = ("accuracy", "precision", "recall", "f1-score")
_NAN = {key: float("nan") for key in KEYS}


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    if not len(y_true):
        return dict(_NAN)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"accuracy": float(accuracy), "precision": float(precision),
            "recall": float(recall), "f1-score": float(f1)}


def average(scores: Sequence[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in KEYS:
        values = [s[key] for s in scores if s[key] == s[key]]
        out[key] = float(sum(values) / len(values)) if values else float("nan")
    return out
