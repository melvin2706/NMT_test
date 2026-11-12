"""Shared multilingual SentencePiece tokenizer (spec §7).

Choices that protect Cameroonian orthographies:
  * normalization_rule_name="identity": SentencePiece's default NFKC rule is
    disabled; text is already NFC-normalized by our own pipeline.
  * character_coverage=1.0: every character seen in training (ɛ, ɔ, ŋ, ə, ʉ,
    combining tone marks, apostrophe letters...) gets a piece.
  * byte_fallback=True: unseen characters become byte pieces instead of <unk>.
  * Language tags <src:xx>/<tgt:xx> for every registered language are
    user-defined symbols, so adding a language later does not change ids of
    existing tags.
  * Training text is sampled per language with a temperature so French and
    English do not crowd out the local languages in the vocabulary.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm

from .languages import all_tag_tokens, src_tag, tgt_tag

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def _texts_by_language(corpus_dir: str) -> dict[str, list[str]]:
    from .data.build_corpus import load_split

    by_lang: dict[str, set[str]] = defaultdict(set)
    for p in load_split(corpus_dir, "train"):
        by_lang[p.source_language].add(p.source_text)
        by_lang[p.target_language].add(p.target_text)
    return {k: sorted(v) for k, v in by_lang.items()}


def temperature_sample(by_lang: dict[str, list[str]], temperature: float, total: int, seed: int) -> list[str]:
    sizes = {k: len(v) for k, v in by_lang.items() if v}
    z = sum(n ** (1 / temperature) for n in sizes.values())
    rng = random.Random(seed)
    out: list[str] = []
    for lang, texts in by_lang.items():
        if not texts:
            continue
        k = max(1, round(total * (sizes[lang] ** (1 / temperature)) / z))
        if k <= len(texts):
            out += rng.sample(texts, k)
        else:  # upsample small languages
            out += texts * (k // len(texts)) + rng.sample(texts, k % len(texts))
    rng.shuffle(out)
    return out


def train_tokenizer(
    corpus_dir: str,
    out_dir: str,
    language_codes: list[str],
    model_type: str = "unigram",
    vocab_size: int = 16000,
    temperature: float = 3.0,
    sample_lines: int = 2_000_000,
    seed: int = 1234,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_lang = _texts_by_language(corpus_dir)
    lines = temperature_sample(by_lang, temperature, sample_lines, seed)
    train_txt = out / "spm_train.txt"
    train_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(train_txt),
        model_prefix=str(out / "spm"),
        model_type=model_type,
        vocab_size=vocab_size,
        character_coverage=1.0,
        normalization_rule_name="identity",
        byte_fallback=True,
        split_digits=True,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=all_tag_tokens(language_codes),
        input_sentence_size=sample_lines,
        shuffle_input_sentence=False,  # already shuffled with our seed -> reproducible
        hard_vocab_limit=False,
        num_threads=8,
        minloglevel=2,
    )
    meta = {
        "model_type": model_type,
        "vocab_size_requested": vocab_size,
        "temperature": temperature,
        "languages_in_training_text": {k: len(v) for k, v in by_lang.items()},
        "language_tags": all_tag_tokens(language_codes),
        "corpus_manifest": json.loads((Path(corpus_dir) / "manifest.json").read_text(encoding="utf-8"))["content_sha256"],
        "seed": seed,
    }
    (out / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    train_txt.unlink()
    return out / "spm.model"


class Tokenizer:
    """Thin wrapper that knows how to add language tags."""

    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.path = str(model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def tag_id(self, tag: str) -> int:
        i = self.sp.piece_to_id(tag)
        if i == UNK_ID:
            raise KeyError(f"language tag {tag} not in tokenizer vocabulary")
        return i

    def encode_source(self, text: str, src: str, tgt: str, max_len: int | None = None) -> list[int]:
        ids = self.sp.encode(text)
        if max_len:
            ids = ids[: max_len - 3]
        return [self.tag_id(src_tag(src)), self.tag_id(tgt_tag(tgt)), *ids, EOS_ID]

    def source_pieces(self, text: str, src: str, tgt: str) -> list[str]:
        """Token strings in the format CTranslate2 expects."""
        return [src_tag(src), tgt_tag(tgt), *self.sp.encode(text, out_type=str), "</s>"]

    def encode_target(self, text: str, max_len: int | None = None) -> list[int]:
        ids = self.sp.encode(text)
        if max_len:
            ids = ids[: max_len - 2]
        return [BOS_ID, *ids, EOS_ID]

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self.sp.decode(ids)

    def decode_pieces(self, pieces: list[str]) -> str:
        return self.sp.decode_pieces([p for p in pieces if p not in ("<s>", "</s>", "<pad>")])


def evaluate_tokenizer(model_path: str, corpus_dir: str, split: str = "valid") -> dict:
    """Per-language segmentation statistics (spec §7.4)."""
    from .data.build_corpus import load_split

    tok = Tokenizer(model_path)
    by_lang: dict[str, set[str]] = defaultdict(set)
    pairs = load_split(corpus_dir, split) or load_split(corpus_dir, "train")
    for p in pairs:
        by_lang[p.source_language].add(p.source_text)
    report = {"tokenizer": model_path, "vocab_size": tok.vocab_size, "languages": {}}
    for lang, texts in sorted(by_lang.items()):
        n_tok = n_words = n_unk = n_bytes = n_bad_roundtrip = n_chars = 0
        for t in texts:
            pieces = tok.sp.encode(t, out_type=str)
            ids = tok.sp.encode(t)
            n_tok += len(ids)
            n_words += max(1, len(t.split()))
            n_chars += len(t)
            n_unk += sum(i == UNK_ID for i in ids)
            n_bytes += sum(p.startswith("<0x") for p in pieces)
            if tok.sp.decode(ids) != t:
                n_bad_roundtrip += 1
        report["languages"][lang] = {
            "sentences": len(texts),
            "tokens_per_word": round(n_tok / max(n_words, 1), 3),
            "chars_per_token": round(n_chars / max(n_tok, 1), 3),
            "unk_rate": round(n_unk / max(n_tok, 1), 5),
            "byte_fallback_rate": round(n_bytes / max(n_tok, 1), 5),
            "roundtrip_failures": n_bad_roundtrip,
        }
    return report
