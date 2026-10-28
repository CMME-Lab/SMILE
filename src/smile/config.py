from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "AgentConfig", "JudgeConfig", "load", "load_agent", "load_judge",
           "resolve_dataset", "deep_merge", "apply_overrides"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(data: dict, overrides: list[str] | None) -> dict:
    out = copy.deepcopy(data)
    for item in overrides or []:
        dotted, _, raw = item.partition("=")
        node = out
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = _coerce(raw)
    return out


def _read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data


def load(path: str | Path, *, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    data = apply_overrides(_read_yaml(path), overrides)
    data["_source"] = str(path)
    return data


@dataclass
class Config:
    name: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _split(cls, data: dict) -> tuple[dict, dict]:
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        return ({k: v for k, v in data.items() if k in known},
                {k: v for k, v in data.items() if k not in known})

    @classmethod
    def from_dict(cls, data: dict):
        fields, extra = cls._split(data)
        return cls(**fields, extra=extra)


@dataclass
class AgentConfig(Config):
    role: str = ""
    prompt: str = ""
    base_model: str = ""
    weights: str = ""
    dataset: str = ""
    lora: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    generate: dict[str, Any] = field(default_factory=dict)

    def resolved_weights(self, *, required: bool = True) -> Path:
        path = Path(self.weights)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path


@dataclass
class JudgeConfig(Config):
    base_model: str = ""
    prompt: str = "judge"
    generate: dict[str, Any] = field(default_factory=dict)
    reference_prompts: dict[str, Any] = field(default_factory=dict)


def resolve_dataset(name: str) -> Path:
    directory = REPO_ROOT / "configs" / "datasets"
    direct = directory / f"{name}.yaml"
    if direct.is_file():
        return direct

    for candidate in sorted(directory.glob("*.yaml")):
        if _read_yaml(candidate).get("name") == name:
            return candidate


def load_agent(path: str | Path, **kw) -> AgentConfig:
    return AgentConfig.from_dict(load(path, **kw))


def load_judge(path: str | Path, **kw) -> JudgeConfig:
    return JudgeConfig.from_dict(load(path, **kw))
