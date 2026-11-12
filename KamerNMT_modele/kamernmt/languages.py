"""Language registry and language-tag tokens.

Codes follow ISO 639-3 where one exists. A language is only "supported" once a
direction has been trained AND evaluated (see `configs/languages.yaml`,
field `supported_directions`), never just because it is listed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    status: str  # "pivot" | "candidate" | "active" | "test-fixture"


DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "configs" / "languages.yaml"


def src_tag(code: str) -> str:
    return f"<src:{code}>"


def tgt_tag(code: str) -> str:
    return f"<tgt:{code}>"


def load_languages(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Language]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {
        code: Language(code=code, name=info["name"], status=info.get("status", "candidate"))
        for code, info in raw["languages"].items()
    }


def load_pivot(path: str | Path = DEFAULT_REGISTRY) -> str | None:
    """The pivot language (French by default); None means translate directly."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("pivot", "fr")


def all_tag_tokens(codes: list[str]) -> list[str]:
    """Every language-tag token the tokenizer must keep as a single piece."""
    tags: list[str] = []
    for c in sorted(codes):
        tags += [src_tag(c), tgt_tag(c)]
    return tags
