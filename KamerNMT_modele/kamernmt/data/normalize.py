"""Text normalization that never removes linguistic information (spec §5.4).

What it does:
  * Unicode NFC (composes letters + combining accents/tones where a
    precomposed form exists; keeps combining marks otherwise). Never NFKC/NFD
    stripping, never ASCII folding.
  * Removes invisible control / zero-width / BOM characters.
  * Collapses runs of whitespace, trims.
  * Apostrophes: configurable per language. Many Cameroonian orthographies use
    the apostrophe as a letter (glottal stop, e.g. Ghomálá', Lamnso'), so the
    default for local languages is to unify the look-alikes to ONE chosen
    character rather than delete or split them.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that look like an apostrophe in source documents.
APOSTROPHE_LIKE = "'’‘ʼʹ´`ꞌ"

# Default target per language. fr/en use the ASCII apostrophe.
# Local languages use U+2019 by default; change to "ʼ" (modifier letter
# apostrophe) or "ꞌ" (saltillo) if the orthography committee prefers it.
APOSTROPHE_POLICY: dict[str, str] = {"fr": "'", "en": "'"}
DEFAULT_LOCAL_APOSTROPHE = "’"

_INVISIBLE = re.compile("[​‌‎‏⁠﻿­]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")
_APOS = re.compile(f"[{re.escape(APOSTROPHE_LIKE)}]")

# Common UTF-8-read-as-Latin-1 artefacts. Used by validation to flag, not to fix.
MOJIBAKE = re.compile("Ã[\u0080-¿]|â€|Ã©|Ã¨|Ã |Ã§|Æ|É›|É”")


def normalize_text(text: str, lang: str | None = None) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = _INVISIBLE.sub("", t)
    t = _CONTROL.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    target = APOSTROPHE_POLICY.get(lang or "", DEFAULT_LOCAL_APOSTROPHE)
    t = _APOS.sub(target, t)
    # Typographic quotes « » " are kept; only apostrophe look-alikes are unified.
    return t


def dedup_key(text: str) -> str:
    """Key for duplicate detection: case-folded, punctuation-light, accents KEPT."""
    t = unicodedata.normalize("NFC", text).casefold()
    t = re.sub(r"[\s.,;:!?¡¿\"«»()\[\]]+", " ", t)
    return t.strip()


def char_inventory(texts) -> dict[str, int]:
    """Character frequency table; used to check tokenizer coverage per language."""
    counts: dict[str, int] = {}
    for t in texts:
        for ch in t:
            counts[ch] = counts.get(ch, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
