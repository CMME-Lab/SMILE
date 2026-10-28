from __future__ import annotations

import re

__all__ = ["clean", "tokenize"]


def clean(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", clean(text))
