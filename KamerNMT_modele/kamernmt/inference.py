"""Model loading for translation, on GPU when available, always with the CPU as backup.

    from kamernmt.inference import load_translator
    tr = load_translator("models/kamernmt-small-v1-int8")             # device="auto"
    tr = load_translator("models/kamernmt-small-v1-int8", device="cpu")
    tr.translate(["Bonjour."], src="fr", tgt="en")

device:
    "auto"  use the NVIDIA GPU if the runtime can see one, otherwise the CPU (default)
    "cuda"  require the GPU (error if there is none)
    "cpu"   CPU only

On the GPU, translation is wrapped in GpuWithCpuFallback: if a batch does not fit in GPU
memory it is split in halves and retried; a sentence that still fails is translated on the
CPU (CPU model loaded on first need). After repeated GPU failures the CPU takes over for
good. So GPU memory is used when there is enough of it, and the CPU picks up the rest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .model import BOS_ID, EOS_ID, PAD_ID
from .tokenizer import Tokenizer


def gpu_available(runtime: str) -> bool:
    """Can this runtime use an NVIDIA GPU on this machine?"""
    try:
        if runtime == "ctranslate2":
            import ctranslate2

            return ctranslate2.get_cuda_device_count() > 0
        if runtime == "torch":
            import torch

            return torch.cuda.is_available()
        if runtime == "onnxruntime":
            import onnxruntime as ort

            return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False
    return False


def describe_hardware() -> dict:
    """What the three runtimes can see (shown by `kamernmt --device-info` and GET /health)."""
    info = {"ctranslate2_gpu": gpu_available("ctranslate2"), "torch_gpu": gpu_available("torch"),
            "onnxruntime_gpu": gpu_available("onnxruntime")}
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu_name"] = p.name
            info["gpu_memory_gb"] = round(p.total_memory / 1e9, 1)
    except Exception:
        pass
    return info


class BaseTranslator:
    tok: Tokenizer
    max_len: int = 256
    device: str = "cpu"

    def translate(self, texts: list[str], src: str, tgt: str, beam: int = 4) -> list[str]:
        raise NotImplementedError


class TorchTranslator(BaseTranslator):
    def __init__(self, checkpoint: str, threads: int | None = None, device: str = "cpu"):
        import torch

        from .train import load_checkpoint

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.model, ck = load_checkpoint(checkpoint, map_location=device)
        self.model.to(device)
        self.tok = Tokenizer(ck["tokenizer"])
        self.max_len = self.model.config.max_positions

    def translate(self, texts, src, tgt, beam=4):
        ids = [self.tok.encode_source(t, src, tgt, self.max_len) for t in texts]
        m = max(len(x) for x in ids)
        x = self.torch.tensor([s + [PAD_ID] * (m - len(s)) for s in ids], device=self.device)
        out = self.model.beam_search(x, beam=beam)
        return [self.tok.decode(o) for o in out]


class CT2Translator(BaseTranslator):
    def __init__(self, model_dir: str, threads: int = 0, device: str = "cpu"):
        import ctranslate2

        self.device = device
        if device == "cuda":
            # "auto" = fastest type the GPU supports (int8 / float16 mix)
            self.tr = ctranslate2.Translator(model_dir, device="cuda", compute_type="auto")
        else:
            self.tr = ctranslate2.Translator(model_dir, device="cpu", inter_threads=1,
                                             intra_threads=threads, compute_type="default")
        self.tok = Tokenizer(Path(model_dir) / "spm.model")

    def translate(self, texts, src, tgt, beam=4):
        batch = [self.tok.source_pieces(t, src, tgt)[: self.max_len] for t in texts]
        res = self.tr.translate_batch(batch, beam_size=beam, max_decoding_length=self.max_len,
                                      length_penalty=1.0)
        return [self.tok.decode_pieces(r.hypotheses[0]) for r in res]


class OnnxTranslator(BaseTranslator):
    """Greedy decoding only (no KV cache in the exported graph)."""

    def __init__(self, model_dir: str, precision: str = "int8", threads: int = 0, device: str = "cpu"):
        import onnxruntime as ort

        d = Path(model_dir)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.device = device
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda"
                     else ["CPUExecutionProvider"])
        self.enc = ort.InferenceSession(str(d / f"encoder.{precision}.onnx"), so, providers=providers)
        self.dec = ort.InferenceSession(str(d / f"decoder.{precision}.onnx"), so, providers=providers)
        self.tok = Tokenizer(d / "spm.model")
        self.max_len = json.loads((d / "model_config.json").read_text())["max_positions"]

    def translate(self, texts, src, tgt, beam=1):
        ids = [self.tok.encode_source(t, src, tgt, self.max_len) for t in texts]
        m = max(len(x) for x in ids)
        s = np.array([x + [PAD_ID] * (m - len(x)) for x in ids], dtype=np.int64)
        mem = self.enc.run(None, {"src": s})[0]
        b = len(texts)
        tgt_ids = np.full((b, 1), BOS_ID, dtype=np.int64)
        done = np.zeros(b, dtype=bool)
        limit = min(self.max_len - 1, 2 * m + 10)
        for _ in range(limit):
            logits = self.dec.run(None, {"tgt": tgt_ids, "memory": mem, "src": s})[0]
            nxt = logits.argmax(-1)
            nxt = np.where(done, PAD_ID, nxt)
            tgt_ids = np.concatenate([tgt_ids, nxt[:, None]], axis=1)
            done |= nxt == EOS_ID
            if done.all():
                break
        return [self.tok.decode(row[1:].tolist()) for row in tgt_ids]


class GpuWithCpuFallback(BaseTranslator):
    """GPU first; split batches that do not fit; hand what still fails to the CPU."""

    MAX_FAILURES = 3

    def __init__(self, gpu: BaseTranslator, make_cpu):
        self.gpu, self._make_cpu, self._cpu = gpu, make_cpu, None
        self.tok, self.max_len = gpu.tok, gpu.max_len
        self.failures = 0
        self.gpu_only_failed = False

    @property
    def device(self) -> str:
        return "cpu (GPU failed)" if self.gpu_only_failed else "cuda (+cpu fallback)"

    @property
    def cpu(self) -> BaseTranslator:
        if self._cpu is None:
            self._cpu = self._make_cpu()
        return self._cpu

    def _free_gpu(self) -> None:
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def translate(self, texts, src, tgt, beam=4):
        if self.gpu_only_failed:
            return self.cpu.translate(texts, src, tgt, beam)
        try:
            out = self.gpu.translate(texts, src, tgt, beam)
            self.failures = 0
            return out
        except Exception as e:  # out of memory, driver or kernel error
            self._free_gpu()
            if len(texts) > 1:  # smaller batches may fit in GPU memory
                half = len(texts) // 2
                return (self.translate(texts[:half], src, tgt, beam)
                        + self.translate(texts[half:], src, tgt, beam))
            self.failures += 1
            if self.failures >= self.MAX_FAILURES:
                self.gpu_only_failed = True
                print(f"[kamernmt] GPU keeps failing ({e}); switching to CPU.", file=sys.stderr)
            return self.cpu.translate(texts, src, tgt, beam)


def _runtime_of(p: Path) -> str:
    if p.suffix == ".pt":
        return "torch"
    if (p / "model.bin").exists():
        return "ctranslate2"
    if any(p.glob("encoder.*.onnx")):
        return "onnxruntime"
    raise ValueError(f"cannot tell what kind of model is at {p}")


def load_translator(path: str, precision: str = "int8", threads: int = 0, device: str = "auto") -> BaseTranslator:
    p = Path(path)
    runtime = _runtime_of(p)

    def build(dev: str) -> BaseTranslator:
        if runtime == "torch":
            return TorchTranslator(str(p), threads or None, device=dev)
        if runtime == "ctranslate2":
            return CT2Translator(str(p), threads, device=dev)
        return OnnxTranslator(str(p), precision, threads, device=dev)

    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto, cpu or cuda, not {device!r}")
    use_gpu = device == "cuda" or (device == "auto" and gpu_available(runtime))
    if device == "cuda" and not gpu_available(runtime):
        raise RuntimeError(f"no NVIDIA GPU usable by {runtime} on this machine (use --device auto or cpu)")
    if not use_gpu:
        return build("cpu")
    try:
        return GpuWithCpuFallback(build("cuda"), lambda: build("cpu"))
    except Exception as e:  # GPU present but unusable (driver, CUDA libraries…)
        if device == "cuda":
            raise
        print(f"[kamernmt] GPU found but not usable ({e}); using the CPU.", file=sys.stderr)
        return build("cpu")
