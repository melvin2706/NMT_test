"""ONNX export + dynamic INT8 quantization (alternative runtime, spec §10.3).

Two graphs are exported:
  encoder.onnx  src_ids -> memory
  decoder.onnx  (tgt_ids, memory, src_ids) -> logits of the last position

The decoder graph has no KV cache, so it recomputes the prefix at each step.
That is simpler and fine for short sentences; CTranslate2 is the faster path.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from .model import PAD_ID, Transformer


class _Encoder(torch.nn.Module):
    def __init__(self, m: Transformer):
        super().__init__()
        self.m = m

    def forward(self, src):
        return self.m.encode(src)[0]


class _Decoder(torch.nn.Module):
    def __init__(self, m: Transformer):
        super().__init__()
        self.m = m

    def forward(self, tgt, memory, src):
        mask = (src != PAD_ID)[:, None, None, :]
        return self.m.decode(tgt, memory, mask)[:, -1]


def export(checkpoint: str, out_dir: str, quantize: bool = True, opset: int = 17) -> Path:
    from .train import load_checkpoint

    model, ckpt = load_checkpoint(checkpoint)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = torch.tensor([[4, 5, 6, 7, 3]])
    tgt = torch.tensor([[2, 8, 9]])
    with torch.no_grad():
        mem = model.encode(src)[0]
    common = dict(opset_version=opset, do_constant_folding=True, dynamo=False)
    torch.onnx.export(_Encoder(model), (src,), out / "encoder.fp32.onnx", input_names=["src"],
                      output_names=["memory"], dynamic_axes={"src": {0: "b", 1: "s"}, "memory": {0: "b", 1: "s"}},
                      **common)
    torch.onnx.export(_Decoder(model), (tgt, mem, src), out / "decoder.fp32.onnx",
                      input_names=["tgt", "memory", "src"], output_names=["logits"],
                      dynamic_axes={"tgt": {0: "b", 1: "t"}, "memory": {0: "b", 1: "s"}, "src": {0: "b", 1: "s"},
                                    "logits": {0: "b"}}, **common)
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        for part in ("encoder", "decoder"):
            quantize_dynamic(str(out / f"{part}.fp32.onnx"), str(out / f"{part}.int8.onnx"),
                             weight_type=QuantType.QInt8)
    shutil.copy(ckpt["tokenizer"], out / "spm.model")
    (out / "model_config.json").write_text(__import__("json").dumps(ckpt["model_config"], indent=2))
    from .export_ct2 import write_model_card
    from .tokenizer import Tokenizer

    ckpt.setdefault("run_dir", str(Path(checkpoint).parent))
    write_model_card(out, ckpt, Tokenizer(ckpt["tokenizer"]), "int8" if quantize else "float32", runtime="onnxruntime")
    return out
