"""Compact Wiktionary-derived dictionaries (format `wiktionary_jsonl`).

One JSON object per line:
    {"": "manger", "p": ["noun", "verb"], "d": ["(transitive) to eat", "food, foodstuff"],
     "f": [inflected forms], "i": "/IPA/", "s": [synonyms], ...}

The headword is in the source's `language`; the `d` glosses are in `gloss_language`
(English for dictionary-fr.json). Only glosses that work as translations are kept:

  * dropped: "plural of X", "first-person … of X", "inflection of X:", "alternative form of X",
    "abbreviation of X", proper names (pos "name"), affixes ("-ci"), symbols;
  * cleaned: labels in parentheses "(informal) ", grammar notes in brackets "[with à …]";
  * split on ";" and "," into short alternatives (at most `max_words` words each).

A file whose glosses are definitions in the same language gives no translation pairs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .normalize import normalize_text
from .schema import LexEntry, Provenance

NOT_A_TRANSLATION = re.compile(  # "plural of X", "first-person singular … of X" — at the start only
    r"^[^,;]{0,50}?\b(plural|singular|feminine|masculine|form|forms|spelling|abbreviation|initialism|acronym|"
    r"misspelling|participle|inflection|conjugation|person|synonym|clipping|contraction|ellipsis|"
    r"diminutive|augmentative|superlative|comparative|imperative|indicative|subjunctive|conditional|"
    r"infinitive|gerund|variant|eye dialect|apocopic|aphetic|euphemistic|equivalent|agent noun|alternative|archaic|obsolete|dated)\b(\s+[^\s,;]+){0,4}?\s+of\b",
    re.I,
)
# checked after labels are removed
DESCRIPTIVE = re.compile(r"^(a|an|the) (male|female) given name|^a surname|^used |^indicates |^denotes |"
                         r"^in particular|^especially|^also\b|^e\.g\.|^i\.e\.|^\W", re.I)
FRAGMENT = re.compile(r"^(or|and|etc|especially|in particular|such as|e\.g|i\.e|also)\b", re.I)
LABEL = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
HEADWORD_OK = re.compile(r"^[^\W\d_][\w'’\- ]*$", re.UNICODE)


def clean_glosses(definitions: list[str], max_words: int = 4, max_senses: int = 3) -> list[str]:
    out: list[str] = []
    for d in definitions[:max_senses]:
        d = LABEL.sub(" ", d).replace("“", "").replace("”", "")
        d = re.sub(r"\s+", " ", d).strip(" .;:,")
        if not d or ":" in d or DESCRIPTIVE.search(d) or NOT_A_TRANSLATION.search(d):
            continue
        parts: list[str] = []
        for alt in d.split(";"):
            pieces = [p.strip(" .") for p in alt.split(",")]
            # "a, b, c" is a list of synonyms only if every piece is short;
            # otherwise it is a sentence: keep only what comes before the first comma
            if all(len(p.split()) <= max_words for p in pieces):
                parts += pieces
            else:
                parts.append(pieces[0])
        for part in parts:
            n = len(part.split())
            if FRAGMENT.search(part) or NOT_A_TRANSLATION.search(part):
                continue
            if 1 <= n <= max_words and re.search(r"[A-Za-z]", part) and part not in out:
                out.append(part)
    return out


def to_entries(path: str | Path, src: dict, language: str) -> list[LexEntry]:
    gloss_lang = src.get("gloss_language")
    if not gloss_lang or gloss_lang == language:
        return []  # monolingual definitions: no translation pairs
    prov = Provenance(source_id=src["id"], origin=src.get("origin", ""), validated=bool(src.get("validated", False)),
                      synthetic=bool(src.get("synthetic", False)), kind="dictionary_headword")
    skip_pos = set(src.get("skip_pos", ["name"]))
    entries: list[LexEntry] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            hw = normalize_text(r.get("", ""), language)
            pos = r.get("p", [])
            if not hw or not HEADWORD_OK.match(hw) or hw.startswith("-") or len(hw.split()) > 4:
                continue
            if pos and all(p in skip_pos for p in pos):
                continue
            glosses = [normalize_text(g, gloss_lang) for g in clean_glosses(r.get("d", []))]
            glosses = [g for g in glosses if g.casefold() != hw.casefold()]
            if glosses:
                entries.append(LexEntry(language=language, headword=hw, translations={gloss_lang: glosses},
                                        pos=",".join(pos) or None, entry_id=f"{src['id']}:{n}",
                                        provenance=prov))
    return entries
