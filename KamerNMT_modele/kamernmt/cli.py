"""Command line: python -m kamernmt.cli <command> --help"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _langs(registry: str) -> list[str]:
    from .languages import load_languages

    return list(load_languages(registry))


def cmd_check_sources(a):
    from .data.importers import check_source, load_sources

    sources, blocklist = load_sources(a.sources)
    ok = 0
    for s in sources:
        if s.get("enabled") is False:
            print(f"disabled  {s['id']}")
            continue
        check_source(s, blocklist, Path(a.root))
        status = "validated" if s.get("validated") else "not reviewed (needs --allow-unvalidated)"
        print(f"ok  {s['id']}  [{status}]")
        ok += 1
    print(f"{ok} source(s) ready, {len(sources) - ok} disabled")


def cmd_build_corpus(a):
    from .data.build_corpus import build

    m = build(a.sources, a.out, set(_langs(a.languages)), root=a.root, valid_frac=a.valid_frac,
              test_frac=a.test_frac, seed=a.seed, allow_unvalidated=a.allow_unvalidated)
    print(json.dumps({k: m[k] for k in ("raw_pairs", "quarantined", "duplicates_removed",
                                        "eval_sentences_also_in_train", "directions")}, indent=2))


def cmd_train_tokenizer(a):
    from .tokenizer import evaluate_tokenizer, train_tokenizer

    path = train_tokenizer(a.corpus, a.out, _langs(a.languages), a.type, a.vocab, a.temperature, seed=a.seed)
    rep = evaluate_tokenizer(str(path), a.corpus)
    (Path(a.out) / "segmentation_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))


def cmd_tokenizer_sweep(a):
    """Spec §7.2/7.3: BPE vs Unigram at several vocabulary sizes."""
    from .tokenizer import evaluate_tokenizer, train_tokenizer

    rows = []
    for t in a.types:
        for v in a.vocabs:
            out = Path(a.out) / f"{t}-{v}"
            path = train_tokenizer(a.corpus, str(out), _langs(a.languages), t, v, a.temperature, seed=a.seed)
            rep = evaluate_tokenizer(str(path), a.corpus)
            (out / "segmentation_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
            for lang, s in rep["languages"].items():
                rows.append((f"{t}-{v}", rep["vocab_size"], lang, s["tokens_per_word"], s["unk_rate"],
                             s["byte_fallback_rate"], s["roundtrip_failures"]))
    print("| tokenizer | vocab | lang | tok/word | unk | byte-fallback | roundtrip fails |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for r in rows:
        print("| " + " | ".join(map(str, r)) + " |")


def cmd_params(a):
    from .model import ModelConfig, Transformer
    from .train import load_config

    cfg = load_config(a.config)
    m = Transformer(ModelConfig(vocab_size=a.vocab, **cfg["model"]))
    n = m.num_parameters()
    print(f"{cfg['experiment']}: {n:,} parameters with vocab {a.vocab} "
          f"(~{n * 4 / 1e6:.0f} MB fp32, ~{n / 1e6:.0f} MB int8)")


def cmd_train(a):
    from .train import train

    print(f"best checkpoint: {train(a.config, resume=a.resume)}")


def cmd_export_ct2(a):
    from .export_ct2 import export

    print(f"written {export(a.checkpoint, a.out, a.quantization)}")


def cmd_export_onnx(a):
    from .export_onnx import export

    print(f"written {export(a.checkpoint, a.out, quantize=not a.no_quantize)}")


def cmd_translate(a):
    from .engine import TranslationEngine
    from .languages import load_pivot

    eng = TranslationEngine(a.model, pivot=load_pivot(), beam=a.beam, precision=a.precision)
    texts = a.text or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    for h in eng.translate_texts(texts, a.src, a.tgt):
        print(h)


def cmd_evaluate(a):
    from .evaluate import evaluate, to_markdown

    r = evaluate(a.model, a.corpus, a.out, split=a.split, beam=a.beam, precision=a.precision,
                 max_per_direction=a.max_per_direction)
    print(to_markdown(r))


def cmd_compare(a):
    from .evaluate import compare

    print(compare(a.old, a.new))


def cmd_human_sheet(a):
    from .evaluate import export_human_eval

    print(f"{export_human_eval(a.eval_dir, a.out, a.per_direction)} rows written to {a.out}")


def cmd_benchmark(a):
    from .benchmark import benchmark, save

    sents = [line.strip() for line in open(a.input, encoding="utf-8") if line.strip()][: a.n]
    r = benchmark(a.model, sents, a.src, a.tgt, beam=a.beam, threads=a.threads, precision=a.precision)
    if a.out:
        save(r, a.out)
    print(json.dumps(r, indent=2))


def cmd_worksheet_export(a):
    from .data.worksheets import export_worksheet

    sents = [line.strip() for line in open(a.input, encoding="utf-8") if line.strip()]
    print(f"{export_worksheet(sents, a.src, a.tgt, a.out, a.prefix)} rows written to {a.out}")


def cmd_worksheet_import(a):
    from .data.worksheets import import_worksheet

    pairs, stats = import_worksheet(a.input, a.source_id, a.origin)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["src_lang", "tgt_lang", "src_text", "tgt_text", "group", "validated"])
        for p in pairs:
            w.writerow([p.source_language, p.target_language, p.source_text, p.target_text, p.group_id, "true"])
    print(json.dumps(stats), f"-> {a.out}  (register it in configs/sources.yaml)")


def cmd_backtranslate(a):
    """Translate monolingual text in language `src` into `tgt`; the output pairs
    have a machine-made `tgt` side, so they are written as tgt->src with synthetic
    source text and genuine target text."""
    from .inference import load_translator

    tr = load_translator(a.model, precision=a.precision)
    mono = [line.strip() for line in open(a.input, encoding="utf-8") if line.strip()]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["src_lang", "tgt_lang", "src_text", "tgt_text", "group", "validated"])
        for i in range(0, len(mono), 32):
            chunk = mono[i : i + 32]
            for j, (orig, hyp) in enumerate(zip(chunk, tr.translate(chunk, a.src, a.tgt, beam=a.beam))):
                w.writerow([a.tgt, a.src, hyp, orig, f"bt{i + j}", "false"])
    print(f"{len(mono)} back-translated pairs -> {a.out}; register with synthetic: true")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kamernmt")
    p.add_argument("--languages", default="configs/languages.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check-sources", help="check that every source is declared correctly")
    s.add_argument("--sources", default="configs/sources.yaml"); s.add_argument("--root", default=".")
    s.set_defaults(fn=cmd_check_sources)

    s = sub.add_parser("build-corpus", help="import, normalize, validate, dedup, split")
    s.add_argument("--sources", default="configs/sources.yaml"); s.add_argument("--root", default=".")
    s.add_argument("--out", required=True); s.add_argument("--seed", type=int, default=1234)
    s.add_argument("--valid-frac", type=float, default=0.05); s.add_argument("--test-frac", type=float, default=0.05)
    s.add_argument("--allow-unvalidated", action="store_true")
    s.set_defaults(fn=cmd_build_corpus)

    s = sub.add_parser("train-tokenizer")
    s.add_argument("--corpus", required=True); s.add_argument("--out", required=True)
    s.add_argument("--type", default="unigram", choices=["unigram", "bpe"]); s.add_argument("--vocab", type=int, default=16000)
    s.add_argument("--temperature", type=float, default=3.0); s.add_argument("--seed", type=int, default=1234)
    s.set_defaults(fn=cmd_train_tokenizer)

    s = sub.add_parser("tokenizer-sweep")
    s.add_argument("--corpus", required=True); s.add_argument("--out", default="artifacts/tokenizer")
    s.add_argument("--types", nargs="+", default=["unigram", "bpe"])
    s.add_argument("--vocabs", nargs="+", type=int, default=[8000, 16000, 32000])
    s.add_argument("--temperature", type=float, default=3.0); s.add_argument("--seed", type=int, default=1234)
    s.set_defaults(fn=cmd_tokenizer_sweep)

    s = sub.add_parser("params", help="exact parameter count for a config")
    s.add_argument("--config", required=True); s.add_argument("--vocab", type=int, default=16000)
    s.set_defaults(fn=cmd_params)

    s = sub.add_parser("train"); s.add_argument("--config", required=True); s.add_argument("--resume", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("export-ct2"); s.add_argument("--checkpoint", required=True); s.add_argument("--out", required=True)
    s.add_argument("--quantization", default="int8",
                   choices=["float32", "float16", "bfloat16", "int16", "int8", "int8_float32", "int8_float16"])
    s.set_defaults(fn=cmd_export_ct2)

    s = sub.add_parser("export-onnx"); s.add_argument("--checkpoint", required=True); s.add_argument("--out", required=True)
    s.add_argument("--no-quantize", action="store_true")
    s.set_defaults(fn=cmd_export_onnx)

    def model_args(s):
        s.add_argument("--model", required=True, help=".pt checkpoint, CTranslate2 dir or ONNX dir")
        s.add_argument("--precision", default="int8", help="ONNX only: int8 | fp32")
        s.add_argument("--beam", type=int, default=4)

    s = sub.add_parser("translate"); model_args(s)
    s.add_argument("--src", required=True); s.add_argument("--tgt", required=True); s.add_argument("text", nargs="*")
    s.set_defaults(fn=cmd_translate)

    s = sub.add_parser("evaluate"); model_args(s)
    s.add_argument("--corpus", required=True); s.add_argument("--out", required=True)
    s.add_argument("--split", default="test"); s.add_argument("--max-per-direction", type=int)
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("compare"); s.add_argument("old"); s.add_argument("new"); s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("human-eval-sheet"); s.add_argument("--eval-dir", required=True); s.add_argument("--out", required=True)
    s.add_argument("--per-direction", type=int, default=100); s.set_defaults(fn=cmd_human_sheet)

    s = sub.add_parser("benchmark"); model_args(s)
    s.add_argument("--input", required=True); s.add_argument("--src", required=True); s.add_argument("--tgt", required=True)
    s.add_argument("--n", type=int, default=200); s.add_argument("--threads", type=int, default=0); s.add_argument("--out")
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("worksheet-export", help="source sentences -> TSV for human translators")
    s.add_argument("--input", required=True); s.add_argument("--src", required=True); s.add_argument("--tgt", required=True)
    s.add_argument("--out", required=True); s.add_argument("--prefix", default="ws")
    s.set_defaults(fn=cmd_worksheet_export)

    s = sub.add_parser("worksheet-import", help="filled worksheet -> parallel TSV (validated rows only)")
    s.add_argument("--input", required=True); s.add_argument("--out", required=True)
    s.add_argument("--source-id", required=True); s.add_argument("--origin", default="")
    s.set_defaults(fn=cmd_worksheet_import)

    s = sub.add_parser("backtranslate"); model_args(s)
    s.add_argument("--input", required=True, help="monolingual text, one sentence per line")
    s.add_argument("--src", required=True); s.add_argument("--tgt", required=True); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_backtranslate)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
