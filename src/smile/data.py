from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator, Sequence

from smile import prompts
from smile.config import load as load_config

__all__ = ["load_records", "view_fields", "reference_field", "answer_for", "vqa_examples",
           "load_split", "split_records", "split_summary", "check_split"]


_MAX_PATH = 255


def _looks_like_path(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_PATH
        and ("/" in value or "\\" in value)
        and "\n" not in value
    )


def _resolve_paths(record: dict[str, Any], base: Path, path_fields: Sequence[str] | None) -> None:
    keys = path_fields if path_fields is not None else list(record)
    for key in keys:
        value = record.get(key)
        if path_fields is None and not _looks_like_path(value):
            continue
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            continue
        candidate = base / value
        if candidate.exists():
            record[key] = str(candidate)


def load_records(path: str | Path, *, path_fields: Sequence[str] | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    blob = json.loads(path.read_text(encoding="utf-8"))
    records = blob["results"] if isinstance(blob, dict) and "results" in blob else blob

    base = path.parent
    for record in records:
        _resolve_paths(record, base, path_fields)
    return records


def view_fields(dataset_cfg: str | Path | dict) -> list[str]:
    cfg = dataset_cfg if isinstance(dataset_cfg, dict) else load_config(dataset_cfg)
    fields = (cfg.get("fields") or {}).get("views")
    return list(fields)


def reference_field(dataset_cfg: dict) -> str:
    return (dataset_cfg.get("fields") or {}).get("reference_caption", "caption")


def answer_for(record: dict, qid: str) -> str | None:
    for pair in (record.get("qa_pairs") or []):
        if isinstance(pair, dict) and pair.get("qid") == qid:
            answer = pair.get("ground_truth_answer")
            return str(answer) if answer is not None else None
    return None


def vqa_examples(records: Sequence[dict], dataset: str) -> Iterator[tuple[dict, dict, str | None]]:
    bank = prompts.question_bank(dataset)
    for record in records:
        for question in bank:
            yield record, question, answer_for(record, question["qid"])


SPLIT_NAMES = ("train", "val", "test")


def load_split(path: str | Path, *, path_fields: Sequence[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    path = Path(path)
    blob = json.loads(path.read_text(encoding="utf-8"))

    out: dict[str, list[dict[str, Any]]] = {}
    base = path.parent
    for name in SPLIT_NAMES:
        records = blob.get(name)
        if not isinstance(records, list):
            continue
        for record in records:
            _resolve_paths(record, base, path_fields)
        out[name] = records

    return out


def split_records(
    records: Sequence[dict[str, Any]],
    *,
    ratios: dict[str, float],
    seed: int = 42,
    group_by: str | None = "patient_id",
) -> dict[str, list[dict[str, Any]]]:

    if group_by:
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            key = str(record.get(group_by))
            groups.setdefault(key, []).append(record)
        units = [groups[k] for k in sorted(groups)]
    else:
        units = [[record] for record in records]

    rng = random.Random(seed)
    rng.shuffle(units)

    counts = {name: int(len(units) * ratios.get(name, 0.0)) for name in SPLIT_NAMES}
    counts["train"] += len(units) - sum(counts.values())

    out: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for name in SPLIT_NAMES:
        chunk = units[cursor: cursor + counts[name]]
        cursor += counts[name]
        out[name] = [record for unit in chunk for record in unit]
    return out


def split_summary(splits: dict[str, list[dict[str, Any]]], *, group_by: str = "patient_id") -> str:
    total = sum(len(v) for v in splits.values())
    lines = []
    for name in SPLIT_NAMES:
        records = splits.get(name) or []
        if not records:
            continue
        groups = len({str(r.get(group_by)) for r in records})
        share = len(records) / total if total else 0.0
        lines.append(f"  {name:<6}{len(records):>6} records  {groups:>6} {group_by}s  ratio {share:.4f}")
    lines.append(f"  {'total':<6}{total:>6} records")
    return "\n".join(lines)


def check_split(splits: dict[str, list[dict[str, Any]]], *, group_by: str = "patient_id") -> list[str]:
    problems: list[str] = []

    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = str(record.get(group_by))
            if key in seen and seen[key] != name:
                problems.append(f"{group_by} {key} appears in both {seen[key]} and {name}")
            seen[key] = name

    for name, records in splits.items():
        empty = sum(1 for r in records if not str(r.get("caption") or "").strip())
        if empty:
            problems.append(f"{name}: {empty} records have an empty caption")

    return sorted(set(problems))
