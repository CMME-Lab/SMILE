from __future__ import annotations

from typing import Sequence

from smile.eval.binary import binary_metrics
from smile.eval.text import clean

__all__ = ["normalize_answer", "compute", "NOT_ASSESSABLE"]

NOT_ASSESSABLE = "not assessable"
_YES = frozenset({"yes", "y", "true", "1"})
_NO = frozenset({"no", "n", "false", "0"})


def normalize_answer(answer: object) -> str:
    value = clean(answer)
    if value in _YES:
        return "yes"
    if value in _NO:
        return "no"
    return NOT_ASSESSABLE


def compute(
    records: Sequence[dict],
    *,
    gt_key: str = "ground_truth_answer",
    pred_key: str = "agent_C_pred_answer",
    qa_key: str = "qa_pairs",
) -> dict[str, float]:
    y_true: list[int] = []
    y_pred: list[int] = []
    excluded = 0

    for record in records:
        pairs = record.get(qa_key, record.get("agentC_qa", []))
        if not isinstance(pairs, list):
            continue
        for qa in pairs:
            if not isinstance(qa, dict):
                continue
            truth = normalize_answer(qa.get(gt_key, qa.get("answer", "")))
            prediction = normalize_answer(qa.get(pred_key, qa.get("pred_answer", "")))
            if truth == NOT_ASSESSABLE or prediction == NOT_ASSESSABLE:
                excluded += 1
                continue
            y_true.append(1 if truth == "yes" else 0)
            y_pred.append(1 if prediction == "yes" else 0)

    return {**binary_metrics(y_true, y_pred), "n": len(y_true), "excluded": excluded}
