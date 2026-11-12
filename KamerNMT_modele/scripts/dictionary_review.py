"""Review pack for a transcribed dictionary (entries_jsonl format).

Produces, in --out:
  review.html   each scanned page next to its transcription, for a speaker
  review.tsv    one row per item to check (ok / correction columns to fill)
  pairs.tsv     the training pairs this transcription would produce
  stats.json    counts and the character inventory of the local-language side

    python scripts/dictionary_review.py --source-id dua-helmlinger1972-dua-fr \
        --out work/review/duala-fr
    (add --pdf path/to/scan.pdf to show the scanned pages next to the transcription)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kamernmt.data.build_corpus import pairs_from_lexicon  # noqa: E402
from kamernmt.data.importers import load_sources  # noqa: E402
from kamernmt.data.structured_dict import decode, expand_tilde, load_records, to_entries  # noqa: E402


def render_pages(pdf: str, pages: list[int], out: Path, scale: float = 1.6) -> None:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf)
    for p in pages:
        target = out / f"page{p:03d}.png"
        if not target.exists():
            doc[p].render(scale=scale).to_pil().convert("L").save(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--sources", default="configs/sources.yaml")
    ap.add_argument("--pdf")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sources, _ = load_sources(ROOT / a.sources)
    src = next(s for s in sources if s["id"] == a.source_id)
    path = ROOT / src["path"]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    recs = load_records(path)
    entries = to_entries(path, src, src["language"])
    pairs = pairs_from_lexicon(entries)

    with open(out / "pairs.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["src_lang", "tgt_lang", "src_text", "tgt_text", "kind", "group"])
        for p in pairs:
            w.writerow([p.source_language, p.target_language, p.source_text, p.target_text, p.provenance.kind, p.group_id])

    rows = []
    for r in recs:
        page = r.get("book_page")
        if r["type"] == "entry":
            hw = decode(r["headword"])
            unc = "; ".join(decode(u) for u in r.get("uncertain", []))
            rows.append([page, "mot-vedette", hw, decode(r.get("forms", "")), "; ".join(r.get("senses", [])) or r.get("definition", ""), unc])
            for ex in r.get("examples", []):
                rows.append([page, "exemple", decode(ex["dua"]), expand_tilde(decode(ex["dua"]), hw), ex.get("fr", ""), ""])
        elif r["type"] == "lexicon":
            rows.append([page, "lexique fr→dua", "; ".join(decode(d) for d in r["dua"]), "", r["fr"], ""])
    with open(out / "review.tsv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["page_livre", "type", "duala_transcrit", "duala_~_développé", "français", "incertitudes",
                    "correct (oui/non)", "correction"])
        w.writerows(rows)

    chars = Counter(ch for p in pairs for t in (p.source_text if p.source_language == src["language"] else p.target_text,) for ch in t)
    stats = {
        "records": Counter(r["type"] for r in recs),
        "entries": len(entries),
        "examples_with_translation": sum(len(e.examples) for e in entries),
        "pairs_both_directions_before_dedup": len(pairs) * 2,
        "pairs_by_kind": Counter(p.provenance.kind for p in pairs),
        "items_flagged_uncertain": sum(len(r.get("uncertain", [])) for r in recs),
        "non_ascii_chars_local_side": {c: n for c, n in chars.most_common() if ord(c) > 127},
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    pdf_pages = sorted({r["pdf_page"] for r in recs})
    if a.pdf:
        render_pages(str(ROOT / a.pdf), pdf_pages, out)
    by_page: dict[int, list] = {}
    for r in recs:
        by_page.setdefault(r["pdf_page"], []).append(r)
    parts = ["<!doctype html><meta charset='utf-8'><title>Relecture</title><style>",
             "body{font-family:Georgia,serif;margin:16px;background:#fafafa;color:#222}",
             ".pg{display:flex;gap:16px;margin-bottom:32px;border-top:2px solid #999;padding-top:8px}",
             ".pg img{width:48%;max-width:620px;border:1px solid #ccc;align-self:flex-start;position:sticky;top:8px}",
             ".tx{flex:1;font-size:15px}.hw{font-weight:bold}.unc{color:#b00;font-size:13px}",
             ".ex{margin-left:18px}.dua{font-style:italic}.lx{margin:2px 0}</style>",
             f"<h1>Relecture — {html.escape(src.get('origin') or src['id'])}</h1>",
             "<p>À gauche le scan, à droite la transcription. Noter les erreurs dans review.tsv "
             "(colonnes « correct » et « correction »). En rouge : passages douteux.</p>"]
    for p in pdf_pages:
        items = []
        for r in by_page[p]:
            if r["type"] == "entry":
                items.append(f"<p><span class='hw'>{html.escape(decode(r['headword']))}</span>, "
                             f"{html.escape(decode(r.get('forms', '')))} <i>{html.escape(decode(r.get('pos', '')))}</i> — "
                             f"{html.escape('; '.join(r.get('senses', [])) or r.get('definition', ''))}")
                for ex in r.get("examples", []):
                    items.append(f"<div class='ex'><span class='dua'>{html.escape(decode(ex['dua']))}</span> "
                                 f"→ {html.escape(ex.get('fr', ''))}</div>")
                if r.get("uncertain"):
                    items.append(f"<div class='unc'>douteux : {html.escape('; '.join(decode(u) for u in r['uncertain']))}</div>")
                items.append("</p>")
            elif r["type"] == "lexicon":
                items.append(f"<div class='lx'>{html.escape(r['fr'])} — <span class='dua'>"
                             f"{html.escape(', '.join(decode(d) for d in r['dua']))}</span></div>")
            else:
                items.append(f"<p class='unc'>[fragment] {html.escape(r.get('note', ''))}</p>")
        img = ""
        if a.pdf:  # embedded so review.html is one self-contained file to send
            import base64

            b64 = base64.b64encode((out / f"page{p:03d}.png").read_bytes()).decode()
            img = f"<img src='data:image/png;base64,{b64}' alt='page {p}'>"
        parts.append(f"<div class='pg'>{img}<div class='tx'><h2>p. {by_page[p][0].get('book_page')}</h2>{''.join(items)}</div></div>")
    (out / "review.html").write_text("".join(parts), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
