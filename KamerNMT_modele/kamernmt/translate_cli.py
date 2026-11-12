"""`kamernmt` command: translate phrases or JSON documents, or translate interactively.

    kamernmt                                          # interactive mode
    kamernmt --tgt dua "Je vais au marché."           # one or more phrases
    kamernmt --tgt dua --input doc.json               # JSON document -> screen
    kamernmt --tgt dua --input doc.json --output doc.dua.json
    type doc.json | kamernmt --tgt dua --input -      # JSON from standard input
    kamernmt --request req.json --output out.json     # request file (same format as the HTTP API)

(Inside the project folder, `python translate.py …` does exactly the same.)

Options:
    --src / --tgt      language codes (fr, dua, en, ...); default fr -> dua
    --model PATH       model folder (models/…), ONNX folder or .pt checkpoint;
                       default: KAMERNMT_MODEL, else the most recent folder in models/
    --skip-keys a,b    JSON keys whose values are never translated (e.g. id,url,code)
    --only-keys a,b    translate only the values under these JSON keys
    --no-pivot         translate directly instead of going through French
    --beam N           beam size (default 4; 1 = fastest)
    --device D         auto (GPU if available, CPU as backup) | cpu | cuda

The output keeps the input's format: a phrase gives a phrase; a JSON document gives
the same document (same keys, nesting and order) with its text values translated.
Placeholders ({name}, %s, <b>…</b>, URLs, e-mails) are kept as they are.
French is the pivot: en -> dua is translated en -> fr -> dua.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HELP = """commands:
  :src xx     change source language      :swap   swap source and target
  :tgt xx     change target language      :langs  list language codes
  :help       this help                   :q      quit (or Ctrl+C / empty line twice)"""


def language_names(codes: list[str]) -> dict[str, str]:
    try:
        from .languages import load_languages

        reg = load_languages()
    except (FileNotFoundError, OSError):
        reg = {}
    return {c: (reg[c].name if c in reg else c) for c in codes}


def interactive(engine, src: str, tgt: str, names: dict[str, str]) -> None:
    print("KamerNMT — interactive translation. Type a phrase and press Enter.")
    print(f"model: {engine.model_path}   pivot: {engine.pivot or 'none'}   device: {engine.device}")
    print(HELP)
    empty = 0
    while True:
        try:
            line = input(f"\n[{src} → {tgt}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            empty += 1
            if empty >= 2:
                return
            continue
        empty = 0
        if line.startswith(":"):
            cmd, _, arg = line[1:].partition(" ")
            arg = arg.strip()
            if cmd in ("q", "quit", "exit"):
                return
            if cmd == "help":
                print(HELP)
            elif cmd == "langs":
                trained = {c for d in engine.card.get("trained_directions", []) for c in d.split("-")}
                for code, name in names.items():
                    mark = "  (trained)" if code in trained else ("  (no training data yet)" if trained else "")
                    print(f"  {code:4} {name}{mark}")
            elif cmd == "swap":
                src, tgt = tgt, src
            elif cmd in ("src", "tgt") and arg:
                if arg not in names:
                    print(f"unknown language code {arg!r} (:langs to list them)")
                elif cmd == "src":
                    src = arg
                else:
                    tgt = arg
            else:
                print("unknown command (:help)")
            continue
        hops = engine.route(src, tgt)
        out = engine.translate_text(line, src, tgt)
        via = f"   (via {' → '.join([hops[0][0]] + [b for _, b in hops])})" if len(hops) > 1 else ""
        print(out + via)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="kamernmt", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="phrase(s) to translate; none = interactive mode")
    ap.add_argument("--src", default="fr")
    ap.add_argument("--tgt", default="dua")
    ap.add_argument("--model")
    ap.add_argument("--input", help="JSON document to translate ('-' = standard input)")
    ap.add_argument("--request", help="request file, same format as the HTTP API: "
                                      "{source_language, target_language, content, [skip_keys], [only_keys]}")
    ap.add_argument("--output", help="where to write the translated JSON (default: screen)")
    ap.add_argument("--skip-keys", default="")
    ap.add_argument("--only-keys", default="")
    ap.add_argument("--no-pivot", action="store_true")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto = GPU if available, with the CPU as backup (default)")
    ap.add_argument("--indent", type=int, default=2, help="indentation of the output JSON (0 = one line)")
    a = ap.parse_args(argv)

    from .engine import AUTO, TranslationEngine

    try:
        engine = TranslationEngine(a.model, pivot=None if a.no_pivot else AUTO, beam=a.beam, device=a.device)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"kamernmt: {e}")
    names = language_names(engine.languages)
    sys.stdout.reconfigure(encoding="utf-8")

    if a.request:  # same contract as POST /translate; output = the translated content
        from .server import RequestError, handle_request

        raw = sys.stdin.read() if a.request == "-" else Path(a.request).read_text(encoding="utf-8-sig")
        try:
            result = handle_request(engine, json.loads(raw))
        except (RequestError, json.JSONDecodeError) as e:
            sys.exit(f"kamernmt: invalid request: {e}")
        text = json.dumps(result["content"], ensure_ascii=False, indent=a.indent if a.indent > 0 else None) + "\n"
        if a.output:
            Path(a.output).write_text(text, encoding="utf-8")
            print(f"translated content ({result['target_language']}) written to {a.output}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        return

    for code in (a.src, a.tgt):
        if code not in names:
            ap.error(f"unknown language code {code!r}; this model knows: {', '.join(names)}")

    if a.input:
        raw = sys.stdin.read() if a.input == "-" else Path(a.input).read_text(encoding="utf-8-sig")
        split = lambda s: {k.strip() for k in s.split(",") if k.strip()}  # noqa: E731
        out = engine.translate_document(json.loads(raw), a.src, a.tgt,
                                        only_keys=split(a.only_keys) or None, skip_keys=split(a.skip_keys))
        text = json.dumps(out, ensure_ascii=False, indent=a.indent if a.indent > 0 else None) + "\n"
        if a.output:
            Path(a.output).write_text(text, encoding="utf-8")
            print(f"translated document written to {a.output}", file=sys.stderr)
        else:
            sys.stdout.write(text)
    elif a.text:
        for line in engine.translate_texts(a.text, a.src, a.tgt):
            print(line)
    else:
        interactive(engine, a.src, a.tgt, names)


if __name__ == "__main__":
    main()
