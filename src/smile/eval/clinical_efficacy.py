from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Sequence

from smile.eval.binary import average, binary_metrics
from smile.eval.text import clean

__all__ = ["entities", "entity_names", "presence_matrix", "compute"]

_VOCABULARY = files(__package__) / "clinical_entities.json"


@lru_cache(maxsize=None)
def entities() -> tuple[tuple[str, tuple[str, ...]], ...]:
    blob = json.loads(_VOCABULARY.read_text(encoding="utf-8"))
    return tuple((e["canonical"], tuple(e["variants"])) for e in blob["entities"])


def entity_names() -> list[str]:
    return [canonical for canonical, _ in entities()]


def presence_matrix(texts: Sequence[str]) -> list[list[int]]:
    vocabulary = entities()
    return [
        [1 if any(variant and variant in clean(text) for variant in variants) else 0
         for _canonical, variants in vocabulary]
        for text in texts
    ]


def compute(references: Sequence[str], predictions: Sequence[str]) -> dict[str, object]:
    y_true = presence_matrix(references)
    y_pred = presence_matrix(predictions)

    micro = binary_metrics([v for row in y_true for v in row],
                           [v for row in y_pred for v in row])

    names = entity_names()
    per_entity: dict[str, dict[str, float]] = {}
    scores: list[dict[str, float]] = []
    for column, name in enumerate(names):
        score = binary_metrics([row[column] for row in y_true], [row[column] for row in y_pred])
        scores.append(score)
        per_entity[name] = score

    return {"micro": micro, "macro": average(scores), "per_entity": per_entity}
