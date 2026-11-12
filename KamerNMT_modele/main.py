"""Run the whole KamerNMT pipeline.

    python main.py                         # full pipeline with configs/small.yaml
    python main.py --config configs/tiny.yaml
    python main.py --steps train,evaluate  # only some steps
    python main.py --force                 # redo steps whose outputs already exist
    python main.py --demo                  # end-to-end run on toy data (no real data needed)

Steps, in order:
    sources    check that every source in configs/sources.yaml is declared correctly
    corpus     import, normalize, validate, deduplicate, split -> data.corpus_dir
    tokenizer  train SentencePiece -> data.tokenizer
    train      train the Transformer (resumes from last.pt if present)
    evaluate   BLEU / chrF++ / TER per direction on the test split
    export     CTranslate2 INT8 model for CPU
    evaluate_int8  same metrics on the INT8 model + comparison with FP32
    benchmark  CPU latency, throughput, RAM, size

Every source declared in configs/sources.yaml is used (unless `enabled: false`).
Pairs from sources not reviewed by a speaker (`validated: false`) are
quarantined unless --allow-unvalidated is given.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

STEPS = ["sources", "corpus", "tokenizer", "train", "evaluate", "export", "evaluate_int8", "benchmark"]


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}", flush=True)


def tokenizer_settings(tok_path: str) -> tuple[str, int]:
    """configs say e.g. artifacts/tokenizer/unigram-16000/spm.model -> ("unigram", 16000)."""
    m = re.search(r"(unigram|bpe)-(\d+)", tok_path)
    return (m.group(1), int(m.group(2))) if m else ("unigram", 16000)


def train_pair_count(corpus_dir: str) -> int:
    manifest = Path(corpus_dir) / "manifest.json"
    if not manifest.exists():
        return 0
    m = json.loads(manifest.read_text(encoding="utf-8"))
    return sum(m["directions"]["train"].values())


def run_demo(steps: int | None) -> None:
    banner("Demo: full pipeline on toy data (French/English + a fake test language)")
    cmd = [sys.executable, "scripts/smoke_test.py"] + (["--steps", str(steps)] if steps else [])
    sys.exit(subprocess.call(cmd))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--sources", default="configs/sources.yaml")
    ap.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset of: " + ", ".join(STEPS))
    ap.add_argument("--force", action="store_true", help="redo steps whose outputs already exist")
    ap.add_argument("--allow-unvalidated", action="store_true",
                    help="train on pairs not yet checked by a speaker (they are quarantined otherwise)")
    ap.add_argument("--quantization", default="int8")
    ap.add_argument("--demo", action="store_true", help="run the pipeline on toy data instead")
    ap.add_argument("--demo-steps", type=int, help="training steps for --demo (default from configs/smoke.yaml)")
    a = ap.parse_args()

    if a.demo:
        run_demo(a.demo_steps)

    from kamernmt.cli import main as cli
    from kamernmt.train import load_config

    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        ap.error(f"unknown step(s): {', '.join(sorted(unknown))}")

    cfg = load_config(a.config)
    exp = cfg["experiment"]
    corpus = cfg["data"]["corpus_dir"]
    tok_path = cfg["data"]["tokenizer"]
    tok_type, vocab = tokenizer_settings(tok_path)
    run_dir = Path(cfg["train"]["out_dir"])
    best = run_dir / "best.pt"
    reports = Path("reports") / exp
    # self-contained model folder: copy it anywhere and load it by path
    ct2_dir = Path("models") / f"kamernmt-{exp}-{a.quantization}"
    t0 = time.time()

    if "sources" in steps:
        banner("1/8  Checking sources")
        cli(["check-sources", "--sources", a.sources])

    if "corpus" in steps:
        banner(f"2/8  Building corpus -> {corpus}")
        args = ["build-corpus", "--sources", a.sources, "--out", corpus, "--seed", str(cfg["seed"])]
        cli(args + (["--allow-unvalidated"] if a.allow_unvalidated else []))

    n_train = train_pair_count(corpus)
    if any(s in steps for s in ("tokenizer", "train")) and n_train == 0:
        banner("Stopped: no training data")
        print(
            "The corpus has no training pairs: every source is disabled (enabled: false)\n"
            "or not yet reviewed by a speaker (validated: false).\n\n"
            "To train anyway on unreviewed data:  python main.py --allow-unvalidated\n\n"
            "To see the pipeline work end to end on toy data:  python main.py --demo"
        )
        sys.exit(1)

    if "tokenizer" in steps:
        banner(f"3/8  Training tokenizer ({tok_type}, {vocab}) -> {tok_path}")
        if Path(tok_path).exists() and not a.force:
            print("exists, skipping (use --force to retrain)")
        else:
            cli(["train-tokenizer", "--corpus", corpus, "--out", str(Path(tok_path).parent),
                 "--type", tok_type, "--vocab", str(vocab), "--seed", str(cfg["seed"])])

    if "train" in steps:
        banner(f"4/8  Training model ({exp}) -> {run_dir}")
        if best.exists() and not (run_dir / "last.pt").exists() and not a.force:
            print("best.pt exists, skipping (use --force to retrain)")
        else:
            cli(["train", "--config", a.config] + ([] if a.force else ["--resume"]))

    if "evaluate" in steps:
        banner(f"5/8  Evaluating FP32 model -> {reports / 'fp32'}")
        cli(["evaluate", "--model", str(best), "--corpus", corpus, "--out", str(reports / "fp32")])

    if "export" in steps:
        banner(f"6/8  Exporting CTranslate2 {a.quantization} -> {ct2_dir}")
        cli(["export-ct2", "--checkpoint", str(best), "--out", str(ct2_dir), "--quantization", a.quantization])

    if "evaluate_int8" in steps:
        banner(f"7/8  Evaluating {a.quantization} model -> {reports / a.quantization}")
        cli(["evaluate", "--model", str(ct2_dir), "--corpus", corpus, "--out", str(reports / a.quantization)])
        if (reports / "fp32" / "metrics.json").exists():
            print("FP32 vs quantized:")
            cli(["compare", str(reports / "fp32" / "metrics.json"), str(reports / a.quantization / "metrics.json")])

    if "benchmark" in steps:
        banner("8/8  CPU benchmark")
        from kamernmt.data.build_corpus import group_by_direction, load_split

        test = group_by_direction(load_split(corpus, "test") or load_split(corpus, "valid"))
        if not test:
            print("no test sentences, skipping")
        else:
            direction, pairs = max(test.items(), key=lambda kv: len(kv[1]))
            src, tgt = direction.split("-")
            sents = reports / f"bench_{direction}.txt"
            sents.parent.mkdir(parents=True, exist_ok=True)
            sents.write_text("\n".join(p.source_text for p in pairs[:200]), encoding="utf-8")
            cli(["benchmark", "--model", str(ct2_dir), "--input", str(sents), "--src", src, "--tgt", tgt,
                 "--out", str(reports / f"benchmark_{a.quantization}.json")])

    banner(f"Done in {(time.time() - t0) / 60:.1f} min")
    print(f"training checkpoint: {best}\nready-to-use model:  {ct2_dir}\nreports:             {reports}")
    print('\ntranslate with:  kamernmt --tgt dua "Je vais au marché."   (or: python translate.py)')


if __name__ == "__main__":
    main()
