"""Quality evaluation per direction (spec §11.1, §11.2, §11.4)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import sacrebleu

from .data.build_corpus import group_by_direction, load_split
from .inference import load_translator


def score(hyps: list[str], refs: list[str]) -> dict[str, float]:
    return {
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "chrf++": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
        "ter": round(sacrebleu.corpus_ter(hyps, [refs]).score, 2),
    }


def evaluate(model_path: str, corpus_dir: str, out_dir: str, split: str = "test", beam: int = 4,
             batch_size: int = 32, precision: str = "int8", max_per_direction: int | None = None) -> dict:
    tr = load_translator(model_path, precision=precision)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": model_path, "split": split, "beam": beam, "directions": {}}
    for d, pairs in sorted(group_by_direction(load_split(corpus_dir, split)).items()):
        pairs = pairs[:max_per_direction] if max_per_direction else pairs
        src_l, tgt_l = d.split("-")
        srcs = [p.source_text for p in pairs]
        refs = [p.target_text for p in pairs]
        hyps: list[str] = []
        for i in range(0, len(srcs), batch_size):
            hyps += tr.translate(srcs[i : i + batch_size], src_l, tgt_l, beam=beam)
        report["directions"][d] = {"sentences": len(pairs), **score(hyps, refs)}
        with open(out / f"hyp.{d}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["source", "reference", "hypothesis"])
            w.writerows(zip(srcs, refs, hyps))
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metrics.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def to_markdown(report: dict) -> str:
    lines = [f"Model: `{report['model']}` — split `{report['split']}`, beam {report['beam']}", "",
             "| Direction | Sentences | BLEU | chrF++ | TER |", "|---|---:|---:|---:|---:|"]
    for d, m in report["directions"].items():
        lines.append(f"| {d} | {m['sentences']} | {m['bleu']} | {m['chrf++']} | {m['ter']} |")
    return "\n".join(lines) + "\n"


def compare(old_metrics: str, new_metrics: str) -> str:
    """Side-by-side table of two metrics.json files, flagging regressions."""
    a = json.loads(Path(old_metrics).read_text(encoding="utf-8"))["directions"]
    b = json.loads(Path(new_metrics).read_text(encoding="utf-8"))["directions"]
    lines = ["| Direction | chrF++ old | chrF++ new | Δ | BLEU old | BLEU new | Δ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in sorted(set(a) | set(b)):
        ca, cb = a.get(d, {}).get("chrf++"), b.get(d, {}).get("chrf++")
        ba, bb = a.get(d, {}).get("bleu"), b.get(d, {}).get("bleu")
        dc = "" if ca is None or cb is None else f"{cb - ca:+.2f}" + (" ⚠" if cb < ca - 1.0 else "")
        db = "" if ba is None or bb is None else f"{bb - ba:+.2f}"
        lines.append(f"| {d} | {ca} | {cb} | {dc} | {ba} | {bb} | {db} |")
    return "\n".join(lines) + "\n"


HUMAN_CRITERIA = ["meaning_1to5", "accuracy_1to5", "grammar_1to5", "fluency_1to5", "terminology_1to5"]


def export_human_eval(eval_dir: str, out_path: str, per_direction: int = 100, seed: int = 0) -> int:
    """Sheet for speaker evaluation (spec §11.2). Rows are shuffled, model identity hidden."""
    import random

    rng = random.Random(seed)
    rows = []
    for f in sorted(Path(eval_dir).glob("hyp.*.tsv")):
        d = f.stem.split(".", 1)[1]
        with open(f, encoding="utf-8", newline="") as fh:
            items = list(csv.DictReader(fh, delimiter="\t"))
        for it in rng.sample(items, min(per_direction, len(items))):
            rows.append([d, it["source"], it["hypothesis"], it["reference"]] + [""] * len(HUMAN_CRITERIA) + [""])
    rng.shuffle(rows)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["direction", "source", "translation", "reference"] + HUMAN_CRITERIA + ["comment"])
        w.writerows(rows)
    return len(rows)
