"""Turn imported dictionaries + parallel files into a versioned training corpus.

Output directory layout:
    train.jsonl  valid.jsonl  test.jsonl   Pair records, both directions
    quarantine.jsonl                       unvalidated or rejected pairs (not trained on)
    manifest.json                          sources + hashes, counts per direction, issues
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from .importers import import_all
from .normalize import dedup_key
from .schema import LexEntry, Pair, Provenance, write_jsonl
from .validate import check_pair, is_rejected, summarize


def pairs_from_lexicon(entries: list[LexEntry]) -> list[Pair]:
    """Headwords, variants and dictionary examples become short parallel pairs.

    Headword pairs are lexical knowledge: they are always kept in `train`
    (see `split_pairs`), since the model is evaluated on sentences.
    """
    out: list[Pair] = []
    for e in entries:
        base = e.provenance
        gid = e.entry_id
        for pivot, glosses in e.translations.items():
            for g in glosses:
                for form in [e.headword, *e.variants]:
                    out.append(Pair(e.language, pivot, form, g, gid, replace(base, kind="dictionary_headword")))
        # Gloss-to-gloss fr<->en pairs are only taken from fully translated examples,
        # because two glosses of one word are not reliably translations of each other.
        for j, ex in enumerate(e.examples):
            egid = f"{gid}:ex{j}"
            prov = replace(base, kind="dictionary_example")
            for pivot in ("fr", "en"):
                if ex.get(pivot):
                    out.append(Pair(e.language, pivot, ex["src"], ex[pivot], egid, prov))
            if ex.get("fr") and ex.get("en"):
                out.append(Pair("fr", "en", ex["fr"], ex["en"], egid, prov))
    return out


def with_reverse(pairs: list[Pair]) -> list[Pair]:
    """Add the reverse direction. Synthetic pairs are NOT reversed: their machine-made
    side must only ever be a model input, never a training target."""
    out = []
    for p in pairs:
        out.append(p)
        if p.provenance.synthetic:
            continue
        out.append(Pair(p.target_language, p.source_language, p.target_text, p.source_text, p.group_id, p.provenance))
    return out


def dedup(pairs: list[Pair]) -> tuple[list[Pair], int]:
    seen: set[tuple] = set()
    out: list[Pair] = []
    for p in pairs:
        k = (p.source_language, p.target_language, dedup_key(p.source_text), dedup_key(p.target_text))
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out, len(pairs) - len(out)


def _bucket(group_id: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_pairs(pairs: list[Pair], valid_frac: float, test_frac: float, seed: int) -> dict[str, list[Pair]]:
    """Group-level split so a sentence and its reverse never land in different splits.

    Only validated, non-headword pairs can go to valid/test.
    """
    splits: dict[str, list[Pair]] = {"train": [], "valid": [], "test": []}
    for p in pairs:
        if p.provenance.kind == "dictionary_headword" or p.provenance.synthetic or not p.provenance.validated:
            splits["train"].append(p)
            continue
        b = _bucket(p.group_id, seed)
        if b < test_frac:
            splits["test"].append(p)
        elif b < test_frac + valid_frac:
            splits["valid"].append(p)
        else:
            splits["train"].append(p)
    return splits


def _leak_check(splits: dict[str, list[Pair]]) -> int:
    """Count eval source sentences that also appear (any direction) in train."""
    train_texts = {dedup_key(p.source_text) for p in splits["train"]}
    return sum(dedup_key(p.source_text) in train_texts for s in ("valid", "test") for p in splits[s])


def direction_counts(pairs: list[Pair]) -> dict[str, int]:
    c = Counter(f"{p.source_language}-{p.target_language}" for p in pairs)
    return dict(sorted(c.items()))


def build(
    sources_yaml: str,
    out_dir: str,
    known_langs: set[str],
    root: str = ".",
    valid_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 1234,
    allow_unvalidated: bool = False,
    extra_pairs: list[Pair] | None = None,
) -> dict:
    lex, parallel, src_manifest = import_all(sources_yaml, root)
    raw = pairs_from_lexicon(lex) + parallel + (extra_pairs or [])

    kept: list[Pair] = []
    quarantine: list[dict] = []
    all_issues: list[list[str]] = []
    for p in raw:
        issues = check_pair(p, known_langs)
        all_issues.append(issues)
        if is_rejected(issues):
            quarantine.append({**asdict(p), "reason": issues})
        elif not p.provenance.validated and not p.provenance.synthetic and not allow_unvalidated:
            quarantine.append({**asdict(p), "reason": ["unvalidated"]})
        else:
            kept.append(p)

    both = with_reverse(kept)
    both, n_dups = dedup(both)
    splits = split_pairs(both, valid_frac, test_frac, seed)
    leaks = _leak_check(splits)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, ps in splits.items():
        write_jsonl(out / f"{name}.jsonl", ps)
    write_jsonl(out / "quarantine.jsonl", quarantine)

    content_hash = hashlib.sha256()
    for name in ("train", "valid", "test"):
        content_hash.update((out / f"{name}.jsonl").read_bytes())

    manifest = {
        "corpus_version": out.name,
        "content_sha256": content_hash.hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": seed,
        "valid_frac": valid_frac,
        "test_frac": test_frac,
        "allow_unvalidated": allow_unvalidated,
        "sources": src_manifest,
        "lexicon_entries": len(lex),
        "raw_pairs": len(raw),
        "quarantined": len(quarantine),
        "duplicates_removed": n_dups,
        "eval_sentences_also_in_train": leaks,
        "issues": summarize(all_issues),
        "directions": {k: direction_counts(v) for k, v in splits.items()},
        "kinds": dict(Counter(p.provenance.kind for p in both)),
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def languages_in(corpus_dir: str) -> list[str]:
    m = json.loads((Path(corpus_dir) / "manifest.json").read_text(encoding="utf-8"))
    langs: set[str] = set()
    for counts in m["directions"].values():
        for d in counts:
            langs.update(d.split("-"))
    return sorted(langs)


def load_split(corpus_dir: str, split: str, include_synthetic: bool = True) -> list[Pair]:
    from .schema import pair_from_dict, read_jsonl

    path = Path(corpus_dir) / f"{split}.jsonl"
    pairs = [pair_from_dict(d) for d in read_jsonl(path)] if path.exists() else []
    if not include_synthetic:
        pairs = [p for p in pairs if not p.provenance.synthetic]
    return pairs


def group_by_direction(pairs: list[Pair]) -> dict[str, list[Pair]]:
    d: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        d[f"{p.source_language}-{p.target_language}"].append(p)
    return dict(d)


__all__ = ["build", "load_split", "group_by_direction", "languages_in", "Provenance"]
