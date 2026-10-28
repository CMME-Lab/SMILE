from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = ["l2_normalize", "cosine", "aggregate_views", "compute"]

Encoder = Callable[[Sequence], "object"]


def l2_normalize(matrix):
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def cosine(image_embeddings, text_embeddings):
    images = l2_normalize(image_embeddings)
    texts = l2_normalize(text_embeddings)
    return np.sum(images * texts, axis=1)


def aggregate_views(per_view: Sequence[float], how: str = "mean") -> float:
    values = list(per_view)
    if not values:
        return float("nan")
    if how == "mean":
        return sum(values) / len(values)
    if how == "max":
        return max(values)


def compute(
    records: Sequence[dict],
    *,
    encode_image: Encoder,
    encode_text: Encoder,
    view_fields: Sequence[str],
    text_field: str,
    how: str = "mean",
) -> dict[str, object]:
    per_patient: list[float] = []
    for record in records:
        paths = [record[f] for f in view_fields if record.get(f)]
        if not paths:
            continue
        text = str(record.get(text_field) or "")
        if not text.strip():
            continue

        image_embeddings = np.asarray(encode_image(paths))
        text_embedding = np.asarray(encode_text([text]))
        repeated = np.repeat(text_embedding, len(image_embeddings), axis=0)
        per_patient.append(aggregate_views(cosine(image_embeddings, repeated), how))

    return {
        "cosine": float(np.mean(per_patient)) if per_patient else float("nan"),
        "n": len(per_patient),
        "aggregation": how,
    }
