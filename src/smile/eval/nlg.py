from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from typing import Sequence

import bert_score.utils as bs_utils
import torch
from bert_score import score
from nltk.translate import meteor_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from smile.eval.text import clean, tokenize

__all__ = ["rouge_n_f1", "rouge_l_f1", "lcs_length", "compute", "METRIC_KEYS"]

METRIC_KEYS = ("bleu", "bleu_1", "bleu_2", "rouge-1", "rouge-L", "meteor", "bertscore")


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def rouge_n_f1(prediction: str, reference: str, n: int = 1) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    def ngrams(tokens: list[str]) -> Counter:
        return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))

    pred_ngrams, ref_ngrams = ngrams(pred_tokens), ngrams(ref_tokens)
    overlap = sum((pred_ngrams & ref_ngrams).values())
    if overlap == 0:
        return 0.0

    return _f1(overlap / max(1, sum(pred_ngrams.values())),
               overlap / max(1, sum(ref_ngrams.values())))


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ai = a[i - 1]
        row, prev = table[i], table[i - 1]
        for j in range(1, n + 1):
            row[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], row[j - 1])
    return table[m][n]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    return _f1(lcs / len(pred_tokens), lcs / len(ref_tokens))


def _bleu(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    smooth = SmoothingFunction().method4
    weights = {"bleu": None, "bleu_1": (1.0, 0.0, 0.0, 0.0), "bleu_2": (0.5, 0.5, 0.0, 0.0)}
    totals: dict[str, list[float]] = {key: [] for key in weights}

    for prediction, reference in zip(predictions, references):
        pred_tokens, ref_tokens = prediction.split(), reference.split()
        for key, weight in weights.items():
            if not pred_tokens and not ref_tokens:
                totals[key].append(1.0)
            elif not pred_tokens or not ref_tokens:
                totals[key].append(0.0)
            elif weight is None:
                totals[key].append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth))
            else:
                totals[key].append(
                    sentence_bleu([ref_tokens], pred_tokens, weights=weight, smoothing_function=smooth)
                )

    return {key: (sum(vals) / len(vals)) if vals else float("nan") for key, vals in totals.items()}


def _meteor(predictions: Sequence[str], references: Sequence[str]) -> float:
    scores: list[float] = []
    for prediction, reference in zip(predictions, references):
        pred_tokens, ref_tokens = prediction.split(), reference.split()
        if not pred_tokens and not ref_tokens:
            scores.append(1.0)
        elif not pred_tokens or not ref_tokens:
            scores.append(0.0)
        else:
            scores.append(meteor_score.meteor_score([ref_tokens], pred_tokens))
    return sum(scores) / len(scores) if scores else float("nan")


def _bertscore(predictions: Sequence[str], references: Sequence[str], batch_size: int = 32) -> float:
    original = bs_utils.sent_encode

    def patched(tokenizer, sent):
        if not tokenizer.tokenize(sent):
            return tokenizer.encode("", add_special_tokens=True)
        return original(tokenizer, sent)

    with ExitStack() as cleanup:
        cleanup.callback(setattr, bs_utils, "sent_encode", original)
        bs_utils.sent_encode = patched
        _precision, _recall, f1 = score(
            list(predictions), list(references), lang="en", verbose=False, batch_size=batch_size
        )

    if not isinstance(f1, torch.Tensor):
        f1 = torch.tensor(f1)
    f1 = torch.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0)
    mean = f1.mean()
    return float(mean.item())


def compute(
    references: Sequence[str],
    predictions: Sequence[str],
    *,
    metrics: Sequence[str] = METRIC_KEYS,
) -> dict[str, float]:
    references = [str(x or "") for x in references]
    predictions = [str(x or "") for x in predictions]


    out: dict[str, float] = {}

    if any(key.startswith("bleu") for key in metrics):
        out.update({k: v for k, v in _bleu(predictions, references).items() if k in metrics})

    if "rouge-1" in metrics:
        out["rouge-1"] = sum(rouge_n_f1(p, r, 1) for p, r in zip(predictions, references)) / len(references)
    if "rouge-L" in metrics:
        out["rouge-L"] = sum(rouge_l_f1(p, r) for p, r in zip(predictions, references)) / len(references)
    if "meteor" in metrics:
        out["meteor"] = _meteor(predictions, references)
    if "bertscore" in metrics:
        out["bertscore"] = _bertscore(predictions, references)

    return {key: out[key] for key in METRIC_KEYS if key in out}
