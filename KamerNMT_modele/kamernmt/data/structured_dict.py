"""Structured transcriptions of printed dictionaries (format `entries_jsonl`).

One JSON object per line, written by a human or a transcription pass:

  {"type": "entry", "pdf_page": .., "book_page": .., "headword": "déa", "forms": "déyédi",
   "pos": "...", "definition": "...", "senses": ["manger chez q."],
   "examples": [{"dua": "na ma~ ómboa dikó_m", "fr": "je mange chez un ami"}],
   "see": [...], "uncertain": [...]}
  {"type": "lexicon", "pdf_page": .., "fr": "herbe à éléphant", "dua": ["esosongo"]}
  {"type": "fragment", ...}   # continuation of an entry from the previous page; not used

Typing conventions (converted here):
  * `_` after a vowel = underlined vowel of the source orthography (e̱ = ɛ, o̱ = ɔ),
    written as U+0331 COMBINING MACRON BELOW.
  * `~` in an example stands for the headword, as in the printed book.

`senses` hold short glosses usable as translations. Descriptive text that is not
a translation (onomatopoeic expressions, definitions) goes into `definition`
and is NOT turned into a training pair.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .normalize import normalize_text
from .schema import LexEntry, Provenance

UNDERLINE = "̱"


def decode(text: str) -> str:
    """`e_` -> e̱, then NFC (puts the below-mark before tone accents)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFD", text).replace("_", UNDERLINE)
    return unicodedata.normalize("NFC", t)


def expand_tilde(example: str, headword: str) -> str:
    out = re.sub(r"\s*~\s*", f" {headword} ", example).strip()
    out = re.sub(r" '", "'", out)  # "~'a" -> "headword'a"
    return re.sub(r"\s+", " ", out)


def _strip_parens(s: str) -> str:
    """Optional parts in the lexicon, e.g. "(ó) idiba" -> "ó idiba"."""
    return re.sub(r"\s+", " ", s.replace("(", "").replace(")", "")).strip()


def load_records(path: str | Path) -> list[dict]:
    """Reads either a .jsonl of typed records, or a split dictionary .json
    ({"direction": "dua-fr" | "fr-dua", "entries": [...]}) as written by
    scripts/helmlinger.py (duala_fr.json, fr_duala.json)."""
    if str(path).endswith(".json"):
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        kind = "lexicon" if doc.get("direction", "").startswith(("fr-", "en-")) else "entry"
        return [{**r, "type": kind, "_line": i} for i, r in enumerate(doc["entries"], 1)]
    recs = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                r["_line"] = n
                recs.append(r)
    return recs


def to_entries(path: str | Path, src: dict, language: str = "dua") -> list[LexEntry]:
    prov = Provenance(source_id=src["id"], origin=src.get("origin", ""), validated=bool(src.get("validated", False)),
                      synthetic=bool(src.get("synthetic", False)), kind="dictionary_headword")
    entries: list[LexEntry] = []
    for r in load_records(path):
        rid = f"{src['id']}:p{r.get('book_page', r.get('pdf_page'))}:l{r['_line']}"
        if r["type"] == "entry":
            hw = normalize_text(decode(r["headword"]), language)
            examples = []
            for ex in r.get("examples", []):
                if not ex.get("fr"):
                    continue
                dua = normalize_text(expand_tilde(decode(ex["dua"]), hw), language)
                examples.append({"src": dua, "fr": normalize_text(ex["fr"], "fr")})
            senses = [normalize_text(s, "fr") for s in r.get("senses", []) if s.strip()]
            entries.append(LexEntry(
                language=language, headword=hw, translations={"fr": senses} if senses else {},
                pos=r.get("pos") or None, examples=examples, morphology=decode(r.get("forms", "")) or None,
                entry_id=rid, provenance=prov))
        elif r["type"] == "lexicon":
            fr = normalize_text(r["fr"], "fr")
            for d in r["dua"]:
                hw = normalize_text(_strip_parens(decode(d)), language)
                entries.append(LexEntry(language=language, headword=hw, translations={"fr": [fr]},
                                        entry_id=f"{rid}:{hw}", provenance=prov))
    return entries
