"""Training loop (spec §9): temperature-sampled multilingual batches, inverse-sqrt
schedule, label smoothing, validation per direction, early stopping, resumable
checkpoints and a full reproducibility record."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .data.build_corpus import group_by_direction, load_split
from .model import PAD_ID, ModelConfig, Transformer, label_smoothed_loss
from .tokenizer import Tokenizer

Example = tuple[list[int], list[int]]


# ---------------------------------------------------------------- utilities
def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library_versions() -> dict:
    import sentencepiece

    v = {"python": platform.python_version(), "torch": torch.__version__, "sentencepiece": sentencepiece.__version__,
         "numpy": np.__version__, "platform": platform.platform()}
    try:
        import sacrebleu

        v["sacrebleu"] = sacrebleu.__version__
    except ImportError:
        pass
    return v


def build_model(cfg: dict, vocab_size: int) -> Transformer:
    return Transformer(ModelConfig(vocab_size=vocab_size, **cfg["model"]))


def load_checkpoint(path: str, map_location: str = "cpu") -> tuple[Transformer, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = Transformer(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# ------------------------------------------------------------------ data
def encode_pairs(pairs, tok: Tokenizer, max_len: int) -> dict[str, list[Example]]:
    out: dict[str, list[Example]] = defaultdict(list)
    for d, ps in group_by_direction(pairs).items():
        for p in ps:
            s = tok.encode_source(p.source_text, p.source_language, p.target_language, max_len)
            t = tok.encode_target(p.target_text, max_len)
            out[d].append((s, t))
    return dict(out)


def direction_probs(sizes: dict[str, int], temperature: float) -> dict[str, float]:
    w = {d: n ** (1.0 / temperature) for d, n in sizes.items() if n > 0}
    z = sum(w.values())
    return {d: x / z for d, x in w.items()}


class TemperatureBatcher:
    """Endless stream of token-budget batches, directions mixed by temperature sampling."""

    def __init__(self, data: dict[str, list[Example]], temperature: float, max_tokens: int, seed: int, pool: int = 50):
        self.data = data
        self.dirs = list(data)
        probs = direction_probs({d: len(v) for d, v in data.items()}, temperature)
        self.weights = [probs[d] for d in self.dirs]
        self.max_tokens = max_tokens
        self.rng = random.Random(seed)
        self.pool = pool

    def __iter__(self):
        while True:
            # Estimate how many examples fill `pool` batches, sample them, sort by length.
            n = max(64, self.pool * self.max_tokens // 40)
            dirs = self.rng.choices(self.dirs, weights=self.weights, k=n)
            exs = [self.rng.choice(self.data[d]) for d in dirs]
            exs.sort(key=lambda e: (len(e[0]), len(e[1])))
            batches, cur, cur_max = [], [], 0
            for e in exs:
                m = max(cur_max, len(e[0]), len(e[1]))
                if cur and m * (len(cur) + 1) > self.max_tokens:
                    batches.append(cur)
                    cur, m = [], max(len(e[0]), len(e[1]))
                cur.append(e)
                cur_max = m
            if cur:
                batches.append(cur)
            self.rng.shuffle(batches)
            yield from batches


def collate(batch: list[Example], device: torch.device):
    def pad(seqs):
        m = max(len(s) for s in seqs)
        return torch.tensor([s + [PAD_ID] * (m - len(s)) for s in seqs], dtype=torch.long, device=device)

    src = pad([e[0] for e in batch])
    tgt = pad([e[1] for e in batch])
    return src, tgt[:, :-1], tgt[:, 1:]


def fixed_batches(examples: list[Example], max_tokens: int) -> list[list[Example]]:
    exs = sorted(examples, key=lambda e: (len(e[0]), len(e[1])))
    out, cur, cur_max = [], [], 0
    for e in exs:
        m = max(cur_max, len(e[0]), len(e[1]))
        if cur and m * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, m = [], max(len(e[0]), len(e[1]))
        cur.append(e)
        cur_max = m
    if cur:
        out.append(cur)
    return out


# ------------------------------------------------------------- training
class InverseSqrt:
    def __init__(self, lr: float, warmup: int):
        self.lr, self.warmup = lr, max(1, warmup)

    def __call__(self, step: int) -> float:
        step = max(step, 1)
        return self.lr * min(step / self.warmup, math.sqrt(self.warmup / step))


@torch.no_grad()
def validate(model: Transformer, valid: dict[str, list[Example]], device, max_tokens: int) -> dict[str, float]:
    model.eval()
    per_dir = {}
    tot_loss = tot_tok = 0.0
    for d, exs in sorted(valid.items()):
        loss_sum = ntok = 0.0
        for b in fixed_batches(exs, max_tokens):
            src, tin, tout = collate(b, device)
            loss, n = label_smoothed_loss(model(src, tin), tout, 0.0)
            loss_sum += loss.item()
            ntok += n
        per_dir[d] = loss_sum / max(ntok, 1)
        tot_loss += loss_sum
        tot_tok += ntok
    model.train()
    return {"loss": tot_loss / max(tot_tok, 1), "per_direction": per_dir}


def train(config_path: str, resume: bool = False) -> Path:
    cfg = load_config(config_path)
    tc, dc = cfg["train"], cfg["data"]
    out = Path(tc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = pick_device(tc["device"])

    tok = Tokenizer(dc["tokenizer"])
    train_pairs = load_split(dc["corpus_dir"], "train", include_synthetic=dc.get("include_synthetic", False))
    valid_pairs = load_split(dc["corpus_dir"], "valid")
    train_data = encode_pairs(train_pairs, tok, dc["max_len"])
    valid_data = encode_pairs(valid_pairs, tok, dc["max_len"])
    if not train_data:
        raise SystemExit("no training data")
    if not valid_data:
        print("warning: empty validation split, using a slice of train for validation")
        valid_data = {d: v[:50] for d, v in train_data.items()}

    model = build_model(cfg, tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=(0.9, 0.98), eps=1e-9,
                            weight_decay=tc["weight_decay"])
    sched = InverseSqrt(tc["lr"], tc["warmup_steps"])
    use_amp = bool(tc.get("amp")) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    step, best, bad_evals = 0, float("inf"), 0
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best, bad_evals = ck["step"], ck["best_valid_loss"], ck["bad_evals"]
        random.setstate(ck["rng"]["python"])
        torch.set_rng_state(ck["rng"]["torch"])
        print(f"resumed from step {step}")

    run_info = {
        "experiment": cfg["experiment"],
        "config": cfg,
        "config_file": str(config_path),
        "parameters": model.num_parameters(),
        "vocab_size": tok.vocab_size,
        "tokenizer_sha256": sha256_file(dc["tokenizer"]),
        "corpus_manifest": json.loads((Path(dc["corpus_dir"]) / "manifest.json").read_text(encoding="utf-8")),
        "train_directions": {d: len(v) for d, v in train_data.items()},
        "sampling_probs": direction_probs({d: len(v) for d, v in train_data.items()}, dc["sampling_temperature"]),
        "device": str(device),
        "amp": str(amp_dtype) if use_amp else None,
        "versions": library_versions(),
    }
    run_info["corpus_manifest"].pop("issues", None)
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    gpu = ""
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        gpu = f" ({prop.name}, {prop.total_memory / 1e9:.1f} GB)"
    print(f"model parameters: {model.num_parameters():,}  device: {device}{gpu}"
          f"  CPU threads: {torch.get_num_threads()}")

    batch_source = TemperatureBatcher(train_data, dc["sampling_temperature"], tc["max_tokens_per_batch"],
                                      seed=cfg["seed"] + step)
    batcher = iter(batch_source)
    eval_tokens = tc["max_tokens_per_batch"]
    oom_error = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # faster matrix products on recent GPUs
        torch.backends.cudnn.allow_tf32 = True
    log = open(out / "metrics.jsonl", "a", encoding="utf-8")
    model.train()
    t0, run_loss, run_tok = time.time(), 0.0, 0

    def save(path: Path):
        torch.save({
            "model": model.state_dict(),
            "model_config": model.config.to_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "best_valid_loss": best,
            "bad_evals": bad_evals,
            "tokenizer": dc["tokenizer"],
            "tokenizer_sha256": run_info["tokenizer_sha256"],
            "experiment": cfg["experiment"],
            "rng": {"python": random.getstate(), "torch": torch.get_rng_state()},
        }, path)

    while step < tc["max_steps"]:
        for g in opt.param_groups:
            g["lr"] = sched(step + 1)
        opt.zero_grad(set_to_none=True)
        try:
            for _ in range(tc["grad_accum"]):
                src, tin, tout = collate(next(batcher), device)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    loss, n = label_smoothed_loss(model(src, tin), tout, tc["label_smoothing"])
                scaler.scale(loss / max(n, 1) / tc["grad_accum"]).backward()
                run_loss += loss.item()
                run_tok += n
        except oom_error:
            # GPU memory full: drop this step, halve the batch size, keep training.
            if device.type != "cuda" or batch_source.max_tokens <= 256:
                raise
            opt.zero_grad(set_to_none=True)
            src = tin = tout = loss = None  # release the tensors of the failed step
            torch.cuda.empty_cache()
            batch_source.max_tokens //= 2
            eval_tokens = min(eval_tokens, batch_source.max_tokens)
            batcher = iter(batch_source)
            print(f"GPU out of memory: batch size reduced to {batch_source.max_tokens} tokens "
                  f"(raise grad_accum in the config to keep the same effective batch)")
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_norm"])
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step:>7}  loss {run_loss / max(run_tok, 1):.4f}  lr {sched(step):.2e}  tok/s {run_tok / el:,.0f}")
            log.write(json.dumps({"step": step, "train_loss": run_loss / max(run_tok, 1), "lr": sched(step)}) + "\n")
            t0, run_loss, run_tok = time.time(), 0.0, 0

        if step % tc["eval_every"] == 0 or step == tc["max_steps"]:
            v = validate(model, valid_data, device, eval_tokens)
            improved = v["loss"] < best
            if improved:
                best, bad_evals = v["loss"], 0
            else:
                bad_evals += 1
            print(f"[valid] step {step}  loss {v['loss']:.4f}  best {best:.4f}  "
                  + "  ".join(f"{d}:{x:.3f}" for d, x in v["per_direction"].items()))
            log.write(json.dumps({"step": step, "valid": v, "best": best}) + "\n")
            log.flush()
            if improved:
                save(out / "best.pt")
            if bad_evals >= tc["early_stopping_patience"]:
                print(f"early stopping at step {step}")
                save(out / "last.pt")
                break

        if step % tc["save_every"] == 0:
            save(out / f"step{step}.pt")
            save(out / "last.pt")
            ckpts = sorted(out.glob("step*.pt"), key=lambda p: int(p.stem[4:]))
            for old in ckpts[: -tc["keep_checkpoints"]]:
                old.unlink()

    save(out / "last.pt")
    log.close()
    return out / "best.pt"
