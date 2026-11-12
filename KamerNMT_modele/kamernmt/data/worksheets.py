"""Human-in-the-loop data creation (spec §6.3).

Claude (or anyone) may draft French/English *source* sentences — simple,
compound, questions, negations, variations — but translations into Cameroonian
languages must come from competent speakers. This module:

  1. exports a worksheet (TSV) of source sentences with an empty target column;
  2. imports filled worksheets back as a parallel source, keeping only rows a
     named validator has approved.

Machine output (back-translation, model drafts shown to translators) is always
flagged `synthetic=true` and never enters valid/test.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .normalize import normalize_text
from .schema import Pair, Provenance

WORKSHEET_COLUMNS = ["id", "src_lang", "tgt_lang", "src_text", "tgt_text", "translator", "validator", "status", "notes"]


def export_worksheet(sentences: list[str], src_lang: str, tgt_lang: str, out_path: str, prefix: str = "ws") -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(WORKSHEET_COLUMNS)
        n = 0
        for i, s in enumerate(sentences):
            s = normalize_text(s, src_lang)
            if s:
                w.writerow([f"{prefix}-{i:05d}", src_lang, tgt_lang, s, "", "", "", "todo", ""])
                n += 1
    return n


def import_worksheet(path: str, source_id: str, origin: str = "") -> tuple[list[Pair], dict[str, int]]:
    """Rows count only if status == 'validated' and a validator is named."""
    pairs: list[Pair] = []
    stats = {"validated": 0, "skipped_unvalidated": 0, "skipped_empty": 0}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sl, tl = row["src_lang"], row["tgt_lang"]
            st, tt = normalize_text(row["src_text"], sl), normalize_text(row["tgt_text"], tl)
            if not st or not tt:
                stats["skipped_empty"] += 1
                continue
            if row.get("status", "").strip().lower() != "validated" or not row.get("validator", "").strip():
                stats["skipped_unvalidated"] += 1
                continue
            prov = Provenance(source_id=source_id, origin=origin, validated=True, synthetic=False, kind="worksheet")
            pairs.append(Pair(sl, tl, st, tt, group_id=f"{source_id}:{row['id']}", provenance=prov))
            stats["validated"] += 1
    return pairs, stats
