"""Convert a KamerNMT checkpoint to a CTranslate2 model (spec §10).

The weight names are mapped by hand onto ctranslate2.specs.TransformerSpec, so
no third-party model format is involved. Quantization options:
float32 | int8 | int8_float32 | int16 | float16 | bfloat16 (int8 is the target).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .model import Transformer, sinusoidal_table
from .tokenizer import Tokenizer


def _np(t) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _linear(spec, lin) -> None:
    spec.weight = _np(lin.weight)
    if lin.bias is not None:
        spec.bias = _np(lin.bias)


def _fused(spec, linears) -> None:
    spec.weight = np.concatenate([_np(l.weight) for l in linears], axis=0)
    spec.bias = np.concatenate([_np(l.bias) for l in linears], axis=0)


def _norm(spec, ln) -> None:
    spec.gamma = _np(ln.weight)
    spec.beta = _np(ln.bias)


def build_spec(model: Transformer, tok: Tokenizer):
    from ctranslate2.specs import common_spec, transformer_spec

    c = model.config
    spec = transformer_spec.TransformerSpec.from_config(
        (c.encoder_layers, c.decoder_layers),
        c.n_heads,
        pre_norm=True,
        activation=common_spec.Activation.RELU,
    )
    emb = _np(model.embed.weight)
    pos = _np(sinusoidal_table(max(1024, c.max_positions), c.d_model))

    enc, dec = spec.encoder, spec.decoder
    enc.embeddings[0].weight = emb
    enc.scale_embeddings = True
    enc.position_encodings.encodings = pos
    _norm(enc.layer_norm, model.encoder_norm)
    for ls, lm in zip(enc.layer, model.encoder_layers):
        a = lm.self_attn
        _norm(ls.self_attention.layer_norm, lm.self_attn_norm)
        _fused(ls.self_attention.linear[0], [a.q_proj, a.k_proj, a.v_proj])
        _linear(ls.self_attention.linear[1], a.out_proj)
        _norm(ls.ffn.layer_norm, lm.ffn_norm)
        _linear(ls.ffn.linear_0, lm.ffn.linear1)
        _linear(ls.ffn.linear_1, lm.ffn.linear2)

    dec.embeddings.weight = emb
    dec.scale_embeddings = True
    dec.position_encodings.encodings = pos
    _norm(dec.layer_norm, model.decoder_norm)
    dec.projection.weight = emb if model.output_proj is None else _np(model.output_proj.weight)
    for ls, lm in zip(dec.layer, model.decoder_layers):
        a = lm.self_attn
        _norm(ls.self_attention.layer_norm, lm.self_attn_norm)
        _fused(ls.self_attention.linear[0], [a.q_proj, a.k_proj, a.v_proj])
        _linear(ls.self_attention.linear[1], a.out_proj)
        x = lm.cross_attn
        _norm(ls.attention.layer_norm, lm.cross_attn_norm)
        _linear(ls.attention.linear[0], x.q_proj)
        _fused(ls.attention.linear[1], [x.k_proj, x.v_proj])
        _linear(ls.attention.linear[2], x.out_proj)
        _norm(ls.ffn.layer_norm, lm.ffn_norm)
        _linear(ls.ffn.linear_0, lm.ffn.linear1)
        _linear(ls.ffn.linear_1, lm.ffn.linear2)

    vocab = [tok.sp.id_to_piece(i) for i in range(tok.vocab_size)]
    spec.register_source_vocabulary(vocab)
    spec.register_target_vocabulary(vocab)
    spec.config.add_source_bos = False
    spec.config.add_source_eos = False  # </s> is added by Tokenizer.source_pieces
    spec.config.bos_token = "<s>"
    spec.config.eos_token = "</s>"
    spec.config.unk_token = "<unk>"
    spec.config.decoder_start_token = "<s>"
    return spec


def export(checkpoint: str, out_dir: str, quantization: str = "int8") -> Path:
    from ctranslate2.converters.converter import Converter

    from .train import load_checkpoint

    model, ckpt = load_checkpoint(checkpoint)
    ckpt.setdefault("run_dir", str(Path(checkpoint).parent))
    tok = Tokenizer(ckpt["tokenizer"])

    class _KamerConverter(Converter):
        def _load(self):
            return build_spec(model, tok)

    out = Path(out_dir)
    _KamerConverter().convert(str(out), quantization=None if quantization == "float32" else quantization, force=True)
    shutil.copy(ckpt["tokenizer"], out / "spm.model")
    write_model_card(out, ckpt, tok, quantization, runtime="ctranslate2")
    return out


def write_model_card(out: Path, ckpt: dict, tok: Tokenizer, quantization: str, runtime: str) -> None:
    """kamernmt.json: everything needed to use the folder on its own (copy it anywhere)."""
    import json
    from datetime import datetime, timezone

    from . import __version__
    from .languages import load_pivot

    langs = sorted({tok.sp.id_to_piece(i)[5:-1] for i in range(tok.vocab_size)
                    if tok.sp.id_to_piece(i).startswith("<src:")})
    try:
        pivot = load_pivot()
    except FileNotFoundError:
        pivot = "fr"
    try:  # directions the model was actually trained on
        run_info = json.loads((Path(ckpt.get("run_dir", "")) / "run_info.json").read_text(encoding="utf-8"))
        trained = sorted(run_info.get("train_directions", {}))
    except (OSError, ValueError):
        trained = []
    card = {
        "name": out.name,
        "kamernmt_version": __version__,
        "runtime": runtime,
        "quantization": quantization,
        "pivot": pivot,
        "languages": langs,
        "trained_directions": trained,
        "experiment": ckpt.get("experiment"),
        "training_step": ckpt.get("step"),
        "model_config": ckpt.get("model_config"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "kamernmt.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
