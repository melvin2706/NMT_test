"""Canonical record types. Every record says which source it comes from."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Provenance:
    source_id: str
    origin: str = ""         # free-text description of the source
    validated: bool = False  # checked by a competent speaker
    synthetic: bool = False  # produced by a model or rule, not by a human translator
    kind: str = "parallel"   # dictionary_headword | dictionary_example | parallel | backtranslation | worksheet


@dataclass
class LexEntry:
    """One dictionary entry in a local (or pivot) language."""

    language: str
    headword: str
    translations: dict[str, list[str]]           # {"fr": [...], "en": [...]}
    pos: str | None = None
    variants: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)  # [{"src": ..., "fr": ..., "en": ...}]
    morphology: str | None = None
    entry_id: str = ""
    provenance: Provenance | None = None


@dataclass
class Pair:
    """One aligned sentence or phrase pair (spec §6.2)."""

    source_language: str
    target_language: str
    source_text: str
    target_text: str
    group_id: str            # pairs sharing a group never cross train/valid/test
    provenance: Provenance


def pair_from_dict(d: dict) -> Pair:
    d = dict(d)
    d["provenance"] = Provenance(**d["provenance"])
    return Pair(**d)


def entry_from_dict(d: dict) -> LexEntry:
    d = dict(d)
    if d.get("provenance"):
        d["provenance"] = Provenance(**d["provenance"])
    return LexEntry(**d)


def write_jsonl(path: str | Path, records: Iterable) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(asdict(r) if not isinstance(r, dict) else r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
