"""CPU performance benchmark (spec §11.3): latency mean/P50/P95, sentences/s,
tokens/s, RSS memory, CPU usage and model size on disk."""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import psutil

from .inference import load_translator


def dir_size_mb(path: str) -> float:
    p = Path(path)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return round(sum(f.stat().st_size for f in files) / 1e6, 2)


def benchmark(model_path: str, sentences: list[str], src: str, tgt: str, beam: int = 4, threads: int = 0,
              precision: str = "int8", batch_size: int = 16, warmup: int = 3) -> dict:
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    t_load = time.perf_counter()
    tr = load_translator(model_path, precision=precision, threads=threads)
    load_s = time.perf_counter() - t_load

    for s in sentences[:warmup]:
        tr.translate([s], src, tgt, beam=beam)

    # Batch size 1: interactive latency.
    lat, out_tokens = [], 0
    cpu0 = proc.cpu_times()
    t0 = time.perf_counter()
    for s in sentences:
        a = time.perf_counter()
        hyp = tr.translate([s], src, tgt, beam=beam)[0]
        lat.append((time.perf_counter() - a) * 1000)
        out_tokens += len(tr.tok.sp.encode(hyp)) + 1
    wall = time.perf_counter() - t0
    cpu1 = proc.cpu_times()

    # Batched throughput.
    t1 = time.perf_counter()
    for i in range(0, len(sentences), batch_size):
        tr.translate(sentences[i : i + batch_size], src, tgt, beam=beam)
    batched = time.perf_counter() - t1

    lat_sorted = sorted(lat)
    p = lambda q: lat_sorted[min(len(lat_sorted) - 1, int(q * len(lat_sorted)))]  # noqa: E731
    cpu_s = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    return {
        "model": model_path,
        "precision": precision if Path(model_path).suffix != ".pt" else "fp32 (pytorch)",
        "direction": f"{src}-{tgt}",
        "sentences": len(sentences),
        "beam": beam,
        "threads": threads or psutil.cpu_count(logical=False),
        "load_seconds": round(load_s, 3),
        "latency_ms_mean": round(statistics.mean(lat), 2),
        "latency_ms_p50": round(p(0.50), 2),
        "latency_ms_p95": round(p(0.95), 2),
        "sentences_per_s_batch1": round(len(sentences) / wall, 2),
        f"sentences_per_s_batch{batch_size}": round(len(sentences) / batched, 2),
        "generated_tokens_per_s_batch1": round(out_tokens / wall, 1),
        "cpu_utilisation_cores": round(cpu_s / wall, 2),
        "rss_mb_total": round(proc.memory_info().rss / 1e6, 1),
        "rss_mb_model_delta": round((proc.memory_info().rss - rss_before) / 1e6, 1),
        "model_size_mb": dir_size_mb(model_path),
        "machine": {"cpu": platform.processor(), "arch": platform.machine(), "os": platform.platform(),
                    "logical_cpus": psutil.cpu_count()},
    }


def save(result: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")
