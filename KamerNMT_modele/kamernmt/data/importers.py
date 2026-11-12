"""Import dictionaries and parallel files listed in configs/sources.yaml.

Required per source: id, path, format, kind (and language for dictionaries).
Optional: origin (free-text description), validated, synthetic, enabled (false = skip).
A source matching the blocklist (Cameroon-int8, spec §4.2) is refused.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml

from .normalize import normalize_text
from .schema import LexEntry, Pair, Provenance


class SourceRejected(Exception):
    pass


def _split_multi(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split("|") if v.strip()]


def load_sources(path: str | Path) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("sources") or [], [b.lower() for b in raw.get("blocklist") or []]


def check_source(src: dict, blocklist: list[str], root: Path) -> None:
    for key in ("id", "path", "format", "kind"):
        if not src.get(key):
            raise SourceRejected(f"source {src.get('id', '?')}: missing '{key}'")
    if src["kind"] == "dictionary" and not src.get("language"):
        raise SourceRejected(f"source {src['id']}: dictionaries need 'language'")
    haystack = " ".join(str(src.get(k, "")) for k in ("id", "path", "origin")).lower()
    for bad in blocklist:
        if bad in haystack:
            raise SourceRejected(f"source {src['id']}: matches blocklist entry '{bad}'")
    if not (root / src["path"]).exists():
        raise SourceRejected(f"source {src['id']}: file not found: {src['path']}")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prov(src: dict, kind: str) -> Provenance:
    return Provenance(
        source_id=src["id"],
        origin=src.get("origin", ""),
        validated=bool(src.get("validated", False)),
        synthetic=bool(src.get("synthetic", False)),
        kind=kind,
    )


def _read_rows(path: Path, fmt: str):
    if fmt in ("csv", "tsv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f, delimiter="," if fmt == "csv" else "\t")
    elif fmt == "jsonl":
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        raise ValueError(f"unsupported dictionary format: {fmt}")


def import_dictionary(src: dict, root: Path) -> list[LexEntry]:
    """Read a dictionary using its column mapping into LexEntry records."""
    lang = src["language"]
    cols = src.get("columns") or {}
    col = lambda row, key: row.get(cols.get(key, key))  # noqa: E731
    entries: list[LexEntry] = []
    for i, row in enumerate(_read_rows(root / src["path"], src["format"])):
        head = normalize_text(col(row, "headword"), lang)
        if not head:
            continue
        translations = {}
        for pivot in ("fr", "en"):
            vals = [normalize_text(v, pivot) for v in _split_multi(col(row, pivot))]
            if vals:
                translations[pivot] = vals
        examples = []
        ex_src = normalize_text(col(row, "example_src"), lang)
        if ex_src:
            ex = {"src": ex_src}
            for pivot in ("fr", "en"):
                v = normalize_text(col(row, f"example_{pivot}"), pivot)
                if v:
                    ex[pivot] = v
            examples.append(ex)
        entries.append(
            LexEntry(
                language=lang,
                headword=head,
                translations=translations,
                pos=(col(row, "pos") or None),
                variants=[normalize_text(v, lang) for v in _split_multi(col(row, "variants"))],
                synonyms=[normalize_text(v, lang) for v in _split_multi(col(row, "synonyms"))],
                examples=examples,
                morphology=(col(row, "morphology") or None),
                entry_id=f"{src['id']}:{i}",
                provenance=_prov(src, "dictionary_headword"),
            )
        )
    return entries


def import_parallel(src: dict, root: Path) -> list[Pair]:
    """Parallel TSV: columns src_lang, tgt_lang, src_text, tgt_text (header row required).

    Optional columns: group (keeps related rows in the same split),
    validated (true/false per row, overrides the source default).
    """
    pairs: list[Pair] = []
    path = root / src["path"]
    if src["format"] == "parallel_tsv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    elif src["format"] == "jsonl":
        rows = [r for r in _read_rows(path, "jsonl")]
    else:
        raise ValueError(f"unsupported parallel format: {src['format']}")
    for i, row in enumerate(rows):
        sl = row.get("src_lang") or row.get("source_language")
        tl = row.get("tgt_lang") or row.get("target_language")
        st = normalize_text(row.get("src_text") or row.get("source_text"), sl)
        tt = normalize_text(row.get("tgt_text") or row.get("target_text"), tl)
        if not (sl and tl and st and tt):
            continue
        prov = _prov(src, "backtranslation" if src.get("synthetic") else "parallel")
        if "validated" in row and row["validated"] not in (None, ""):
            prov.validated = str(row["validated"]).strip().lower() in ("1", "true", "yes", "oui")
        pairs.append(
            Pair(sl, tl, st, tt, group_id=f"{src['id']}:{row.get('group') or i}", provenance=prov)
        )
    return pairs


def import_all(sources_yaml: str | Path, root: str | Path = ".") -> tuple[list[LexEntry], list[Pair], list[dict]]:
    root = Path(root)
    sources, blocklist = load_sources(sources_yaml)
    lex: list[LexEntry] = []
    pairs: list[Pair] = []
    manifest: list[dict] = []
    for src in sources:
        if src.get("enabled") is False:
            manifest.append({"id": src["id"], "excluded": "enabled: false"})
            print(f"skipping source {src['id']}: enabled is false")
            continue
        check_source(src, blocklist, root)
        if src["kind"] == "dictionary" and src["format"] == "wiktionary_jsonl":
            from .wiktionary import to_entries as wikt_entries

            got = wikt_entries(root / src["path"], src, src["language"])
            lex += got
            n = len(got)
        elif src["kind"] == "dictionary" and src["format"] == "entries_jsonl":
            from .structured_dict import to_entries

            got = to_entries(root / src["path"], src, src["language"])
            lex += got
            n = len(got)
        elif src["kind"] == "dictionary":
            got = import_dictionary(src, root)
            lex += got
            n = len(got)
        elif src["kind"] == "parallel":
            got = import_parallel(src, root)
            pairs += got
            n = len(got)
        else:
            raise SourceRejected(f"source {src['id']}: unknown kind {src['kind']}")
        manifest.append(
            {
                "id": src["id"],
                "path": src["path"],
                "sha256": file_sha256(root / src["path"]),
                "origin": src.get("origin", ""),
                "records": n,
            }
        )
    return lex, pairs, manifest
