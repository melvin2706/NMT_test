"""Per-pair checks (spec §6.4). Errors reject a pair; warnings are only reported."""

from __future__ import annotations

import unicodedata
from collections import Counter

from .normalize import MOJIBAKE
from .schema import Pair

ERRORS = {
    "empty",
    "same_language",
    "mojibake",
    "replacement_char",
    "no_provenance",
    "unknown_language",
}
WARNINGS = {"length_ratio", "identical_text", "too_long", "digits_mismatch"}


def check_pair(p: Pair, known_langs: set[str], max_chars: int = 1000) -> list[str]:
    issues: list[str] = []
    s, t = p.source_text, p.target_text
    if not s.strip() or not t.strip():
        issues.append("empty")
    if p.source_language == p.target_language:
        issues.append("same_language")
    if p.source_language not in known_langs or p.target_language not in known_langs:
        issues.append("unknown_language")
    if MOJIBAKE.search(s) or MOJIBAKE.search(t):
        issues.append("mojibake")
    if "�" in s or "�" in t:
        issues.append("replacement_char")
    if not (p.provenance and p.provenance.source_id):
        issues.append("no_provenance")
    if len(s) > max_chars or len(t) > max_chars:
        issues.append("too_long")
    ls, lt = max(len(s), 1), max(len(t), 1)
    if max(ls, lt) > 12 and max(ls / lt, lt / ls) > 3.0:
        issues.append("length_ratio")
    if s.casefold() == t.casefold() and len(s) > 3:
        issues.append("identical_text")
    ds = sorted(ch for ch in s if ch.isdigit())
    dt = sorted(ch for ch in t if ch.isdigit())
    if ds != dt:
        issues.append("digits_mismatch")
    return issues


def is_rejected(issues: list[str]) -> bool:
    return any(i in ERRORS for i in issues)


def non_nfc(text: str) -> bool:
    return unicodedata.normalize("NFC", text) != text


def summarize(issue_lists: list[list[str]]) -> dict[str, int]:
    c: Counter = Counter()
    for issues in issue_lists:
        c.update(issues)
    return dict(c.most_common())
