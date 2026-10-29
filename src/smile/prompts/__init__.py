from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

__all__ = [
    "load",
    "digest",
    "agent_system",
    "agent_user",
    "agent_c_user",
    "judge_system",
    "judge_user",
    "judge_reference_prompts",
    "question_bank",
    "names",
]

_TEXT = files(__package__) / "text"

_REFERENCE_PROMPT_FILES = ("judge.reference_prompt_a", "judge.reference_prompt_b")


@lru_cache(maxsize=None)
def load(name: str) -> str:
    path = _TEXT / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def names() -> list[str]:
    return sorted(p.name[: -len(".txt")] for p in _TEXT.iterdir() if p.name.endswith(".txt"))


def digest(name: str) -> str:
    return hashlib.sha256(load(name).encode("utf-8")).hexdigest()


def agent_system(agent: str) -> str:
    return load(f"{agent}.system")


def agent_user(prompt: str) -> str:
    return load(f"{prompt}.user")


def agent_c_user(question: str) -> str:
    return load("agent_c.user.template").format(question=question)


def judge_system() -> str:
    return load("judge.system")


def judge_user(
    *,
    prompt_A: str,
    out_A: str,
    prompt_B: str,
    out_B: str,
    qa_text: str,
) -> str:
    return load("judge.user.template").format(
        prompt_A=prompt_A.strip(),
        out_A=out_A.strip(),
        prompt_B=prompt_B.strip(),
        out_B=out_B.strip(),
        qa_text=qa_text,
    )


def judge_reference_prompts() -> tuple[str, str]:
    return tuple(load(n) for n in _REFERENCE_PROMPT_FILES)


@lru_cache(maxsize=None)
def question_bank(dataset: str) -> tuple[dict[str, Any], ...]:
    path = _TEXT / f"question_bank.{dataset}.json"
    return tuple(json.loads(path.read_text(encoding="utf-8")))
