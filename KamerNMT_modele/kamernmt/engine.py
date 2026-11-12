"""Translation engine: phrases, lists and whole JSON documents.

    from kamernmt import Translator
    tr = Translator()                                   # finds the model by itself
    tr.translate("Je vais au marché.", src="fr", tgt="dua")
    tr.translate({"title": "Bonjour", "items": ["Merci", 3]}, src="fr", tgt="dua")

What it guarantees:
  * The output has exactly the input's format and structure. A phrase gives a phrase;
    a JSON document gives the same JSON document: same keys, same nesting, same order.
    Numbers, booleans and null are unchanged; only string values are translated.
  * Parts of a string that must not be translated are kept exactly as they are:
    placeholders ({name}, {{count}}, %s, %(user)s, ${var}), HTML/XML tags, URLs,
    e-mail addresses, line breaks, leading/trailing spaces, and strings with no letters.
  * Long texts are split into sentences, translated one by one, and rejoined with
    their original separators.
  * French is the pivot: if neither side is French, the text goes source -> French -> target.
  * Identical strings are translated once (cache), and everything is batched.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .inference import load_translator

# Things that are kept verbatim inside a string.
PROTECTED = re.compile(
    r"(\{\{[^{}]*\}\}"            # {{mustache}}
    r"|\$\{[^{}]*\}"              # ${var}
    r"|\{[^{}\s]*\}"              # {name}, {0}
    r"|%\([^)]+\)[sdifr]"         # %(name)s
    r"|%[sdifr]"                  # %s %d
    r"|<[^<>]+>"                  # <b>, </a>, <br/>
    r"|https?://\S+|www\.\S+"     # URLs
    r"|[\w.+-]+@[\w-]+\.[\w.-]+"  # e-mails
    r"|\r?\n|\t)"                 # line breaks, tabs
)
SENTENCE_END = re.compile(r"(?<=[.!?…])(\s+)")
HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
LEADING_PUNCT = re.compile(r"^[,;:.!?)\]»\-–—]+\s*")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTO = "auto"


def find_default_model() -> str:
    """Where to find a model when none is given, in this order:
    1. the KAMERNMT_MODEL environment variable;
    2. the most recent folder in ./models/ (current directory), then in the project's models/;
    3. the most recent checkpoint in the project's runs/*/best.pt.
    """
    env = os.environ.get("KAMERNMT_MODEL")
    if env:
        return env
    for base in dict.fromkeys([Path.cwd(), PROJECT_ROOT]):
        folders = [p for p in base.glob("models/*") if (p / "model.bin").exists() or any(p.glob("encoder.*.onnx"))]
        if folders:
            return str(max(folders, key=lambda p: p.stat().st_mtime))
    ckpts = list(PROJECT_ROOT.glob("runs/*/best.pt"))
    if ckpts:
        return str(max(ckpts, key=lambda p: p.stat().st_mtime))
    raise FileNotFoundError("no trained model found: run `python main.py` first, "
                            "pass a model folder, or set KAMERNMT_MODEL")


def read_model_card(model_path: str) -> dict:
    card = Path(model_path) / "kamernmt.json"
    if card.is_file():
        return json.loads(card.read_text(encoding="utf-8"))
    return {}


class TranslationEngine:
    """Load a model once, then translate phrases, lists or JSON documents.

    model_path: a model folder (models/…), an ONNX folder or a .pt checkpoint;
                None = find it automatically (see find_default_model).
    device:     "auto" = NVIDIA GPU if available (CPU takes over what does not fit), "cpu", "cuda".
    pivot:      "auto" = from the model folder (kamernmt.json) or configs/languages.yaml,
                default French; a language code; or None to translate directly.
    """

    def __init__(self, model_path: str | None = None, pivot: str | None = AUTO, beam: int = 4,
                 batch_size: int = 32, threads: int = 0, precision: str = "int8", device: str = "auto"):
        self.model_path = str(model_path or find_default_model())
        self.translator = load_translator(self.model_path, precision=precision, threads=threads, device=device)
        self.card = read_model_card(self.model_path)
        if pivot == AUTO:
            pivot = self.card.get("pivot")
            if pivot is None:
                try:
                    from .languages import load_pivot

                    pivot = load_pivot()
                except FileNotFoundError:
                    pivot = "fr"
        self.pivot = pivot
        self.beam = beam
        self.batch_size = batch_size
        self._cache: dict[tuple[str, str, str], str] = {}

    @property
    def device(self) -> str:
        return self.translator.device

    @property
    def languages(self) -> list[str]:
        """Language codes the model's tokenizer knows."""
        sp = self.translator.tok.sp
        return sorted({sp.id_to_piece(i)[5:-1] for i in range(sp.get_piece_size())
                       if sp.id_to_piece(i).startswith("<src:")})

    def translate(self, content: Any, src: str = "fr", tgt: str = "dua", **doc_options) -> Any:
        """Translate anything and give it back in the same form:
        a phrase -> a phrase; a list of phrases -> a list; a dict / JSON document -> the same document.
        doc_options (documents only): only_keys=set(...), skip_keys=set(...).
        """
        if isinstance(content, str):
            return self.translate_text(content, src, tgt)
        if isinstance(content, list) and all(isinstance(x, str) for x in content):
            return self.translate_texts(content, src, tgt)
        return self.translate_document(content, src, tgt, **doc_options)

    # ------------------------------------------------------------- routing
    def route(self, src: str, tgt: str) -> list[tuple[str, str]]:
        """Translation hops, e.g. [("en", "fr"), ("fr", "dua")]."""
        if src == tgt:
            return []
        if self.pivot and self.pivot not in (src, tgt):
            return [(src, self.pivot), (self.pivot, tgt)]
        return [(src, tgt)]

    def _direct(self, texts: list[str], src: str, tgt: str) -> list[str]:
        todo = [t for t in dict.fromkeys(texts) if (src, tgt, t) not in self._cache]
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i: i + self.batch_size]
            for t, out in zip(chunk, self.translator.translate(chunk, src, tgt, beam=self.beam)):
                self._cache[(src, tgt, t)] = out
        return [self._cache[(src, tgt, t)] for t in texts]

    def translate_segments(self, texts: list[str], src: str, tgt: str) -> list[str]:
        """Plain sentences in, plain sentences out (no protection, no splitting)."""
        out = list(texts)
        for a, b in self.route(src, tgt):
            out = self._direct(out, a, b)
        return out

    # ------------------------------------------------------- text structure
    @staticmethod
    def _plan(text: str) -> list[tuple[bool, str]]:
        """Split a string into (translate?, piece) parts; joining the pieces gives the string back."""
        plan: list[tuple[bool, str]] = []
        for i, part in enumerate(PROTECTED.split(text)):
            if not part:
                continue
            if i % 2 == 1:  # captured protected token
                plan.append((False, part))
                continue
            lead = part[: len(part) - len(part.lstrip())]
            trail = part[len(part.rstrip()):]
            core = part.strip()
            if lead:
                plan.append((False, lead))
            for j, piece in enumerate(SENTENCE_END.split(core)):
                if not piece:
                    continue
                if j % 2 == 1:  # whitespace between sentences
                    plan.append((False, piece))
                    continue
                if not HAS_LETTER.search(piece):
                    plan.append((False, piece))
                    continue
                # punctuation glued to a protected token (", bienvenue") stays where it is
                m = LEADING_PUNCT.match(piece)
                if m:
                    plan.append((False, m.group()))
                    piece = piece[m.end():]
                plan.append((True, piece))
            if trail and core:
                plan.append((False, trail))
        return plan

    def translate_texts(self, texts: list[str], src: str, tgt: str) -> list[str]:
        plans = [self._plan(t) for t in texts]
        to_do = [p for plan in plans for keep, p in plan if keep]
        done = dict(zip(to_do, self.translate_segments(to_do, src, tgt))) if to_do else {}
        return ["".join(done[p] if tr else p for tr, p in plan) for plan in plans]

    def translate_text(self, text: str, src: str, tgt: str) -> str:
        return self.translate_texts([text], src, tgt)[0]

    # ----------------------------------------------------------- documents
    def translate_document(self, doc: Any, src: str, tgt: str, only_keys: set[str] | None = None,
                           skip_keys: set[str] | None = None) -> Any:
        """Same structure back; string values translated.

        only_keys: translate only string values found under these keys (at any depth).
        skip_keys: never translate values under these keys (e.g. {"id", "url"}).
        """
        skip_keys = skip_keys or set()
        strings: list[str] = []

        def collect(node, key=None, allowed=only_keys is None):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in skip_keys:
                        continue
                    collect(v, k, allowed or (only_keys is not None and k in only_keys))
            elif isinstance(node, list):
                for v in node:
                    collect(v, key, allowed)
            elif isinstance(node, str) and allowed:
                strings.append(node)

        collect(doc)
        table = dict(zip(strings, self.translate_texts(strings, src, tgt))) if strings else {}

        def rebuild(node, allowed=only_keys is None):
            if isinstance(node, dict):
                return {k: (v if k in skip_keys else
                            rebuild(v, allowed or (only_keys is not None and k in only_keys)))
                        for k, v in node.items()}
            if isinstance(node, list):
                return [rebuild(v, allowed) for v in node]
            if isinstance(node, str) and allowed:
                return table[node]
            return node

        return rebuild(doc)
