"""Transformer encoder-decoder written from scratch (spec §8).

Design (chosen so the weights map 1:1 onto CTranslate2's TransformerSpec):
  * pre-norm layers + final LayerNorm on encoder and decoder
  * ReLU feed-forward
  * sinusoidal position encodings (table exported with the weights)
  * one embedding matrix shared by encoder input, decoder input and the
    output projection, scaled by sqrt(d_model)
  * separate q/k/v projections with bias (fused at export time)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID, BOS_ID, EOS_ID = 0, 2, 3


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    ffn_dim: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_positions: int = 256
    tie_embeddings: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def sinusoidal_table(n_positions: int, dim: int) -> torch.Tensor:
    """[sin | cos] layout (tensor2tensor / fairseq style)."""
    half = dim // 2
    freq = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / max(half - 1, 1)))
    pos = torch.arange(n_positions, dtype=torch.float32)[:, None] * freq[None, :]
    table = torch.cat([torch.sin(pos), torch.cos(pos)], dim=1)
    if dim % 2:
        table = F.pad(table, (0, 1))
    return table


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dk).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        cache: dict | None = None,
        static_kv: bool = False,
    ) -> torch.Tensor:
        """attn_mask: bool, broadcastable to (B, H, Tq, Tk); True = may attend."""
        q = self._split(self.q_proj(query))
        if cache is not None and static_kv and "k" in cache:
            k, v = cache["k"], cache["v"]
        else:
            k, v = self._split(self.k_proj(kv)), self._split(self.v_proj(kv))
            if cache is not None:
                if not static_kv and "k" in cache:
                    k = torch.cat([cache["k"], k], dim=2)
                    v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        b, _, t, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, t, self.h * self.dk))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(c.d_model)
        self.self_attn = MultiHeadAttention(c.d_model, c.n_heads, c.attention_dropout)
        self.ffn_norm = nn.LayerNorm(c.d_model)
        self.ffn = FeedForward(c.d_model, c.ffn_dim, c.dropout)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.self_attn_norm(x)
        x = x + self.dropout(self.self_attn(h, h, mask))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class DecoderLayer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(c.d_model)
        self.self_attn = MultiHeadAttention(c.d_model, c.n_heads, c.attention_dropout)
        self.cross_attn_norm = nn.LayerNorm(c.d_model)
        self.cross_attn = MultiHeadAttention(c.d_model, c.n_heads, c.attention_dropout)
        self.ffn_norm = nn.LayerNorm(c.d_model)
        self.ffn = FeedForward(c.d_model, c.ffn_dim, c.dropout)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x, memory, self_mask, cross_mask, cache: dict | None = None):
        h = self.self_attn_norm(x)
        x = x + self.dropout(self.self_attn(h, h, self_mask, cache=None if cache is None else cache["self"]))
        h = self.cross_attn_norm(x)
        x = x + self.dropout(
            self.cross_attn(h, memory, cross_mask, cache=None if cache is None else cache["cross"], static_kv=True)
        )
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class Transformer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model, padding_idx=PAD_ID)
        nn.init.normal_(self.embed.weight, mean=0.0, std=c.d_model**-0.5)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()
        self.embed_scale = math.sqrt(c.d_model)
        self.register_buffer("positions", sinusoidal_table(c.max_positions, c.d_model), persistent=False)
        self.dropout = nn.Dropout(c.dropout)
        self.encoder_layers = nn.ModuleList(EncoderLayer(c) for _ in range(c.encoder_layers))
        self.encoder_norm = nn.LayerNorm(c.d_model)
        self.decoder_layers = nn.ModuleList(DecoderLayer(c) for _ in range(c.decoder_layers))
        self.decoder_norm = nn.LayerNorm(c.d_model)
        if c.tie_embeddings:
            self.output_proj = None
        else:
            self.output_proj = nn.Linear(c.d_model, c.vocab_size, bias=False)
        for name, p in self.named_parameters():
            if p.dim() == 2 and "embed" not in name:
                nn.init.xavier_uniform_(p)

    # ----- building blocks -------------------------------------------------
    def _embed(self, ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        t = ids.size(1)
        if offset + t > self.positions.size(0):
            raise ValueError(f"sequence length {offset + t} exceeds max_positions {self.positions.size(0)}")
        x = self.embed(ids) * self.embed_scale + self.positions[offset : offset + t]
        return self.dropout(x)

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.output_proj is None:
            return F.linear(h, self.embed.weight)
        return self.output_proj(h)

    @staticmethod
    def padding_mask(ids: torch.Tensor) -> torch.Tensor:
        """(B, 1, 1, S) bool, True where a key is a real token."""
        return (ids != PAD_ID)[:, None, None, :]

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.padding_mask(src)
        x = self._embed(src)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.encoder_norm(x), mask

    def decode(self, tgt_in, memory, mem_mask, cache: list | None = None, offset: int = 0):
        t = tgt_in.size(1)
        if cache is None:
            causal = torch.ones(t, t, dtype=torch.bool, device=tgt_in.device).tril()
            self_mask = causal[None, None] & self.padding_mask(tgt_in)
        else:
            self_mask = None  # one new token attends to everything cached so far
        x = self._embed(tgt_in, offset)
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, memory, self_mask, mem_mask, None if cache is None else cache[i])
        return self._logits(self.decoder_norm(x))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, mem_mask = self.encode(src)
        return self.decode(tgt_in, memory, mem_mask)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ----- generation ------------------------------------------------------
    def _new_cache(self) -> list[dict]:
        return [{"self": {}, "cross": {}} for _ in self.decoder_layers]

    @staticmethod
    def _reorder_cache(cache: list[dict], index: torch.Tensor) -> None:
        for layer in cache:
            for part in ("self", "cross"):
                for k in ("k", "v"):
                    if k in layer[part]:
                        layer[part][k] = layer[part][k].index_select(0, index)

    def _max_len(self, src: torch.Tensor, max_len: int | None) -> int:
        limit = self.positions.size(0) - 1
        auto = int((src != PAD_ID).sum(1).max().item() * 2 + 10)
        return min(max_len or auto, limit)

    @torch.no_grad()
    def greedy(self, src: torch.Tensor, max_len: int | None = None) -> list[list[int]]:
        self.eval()
        max_len = self._max_len(src, max_len)
        memory, mem_mask = self.encode(src)
        b = src.size(0)
        cache = self._new_cache()
        tok = torch.full((b, 1), BOS_ID, dtype=torch.long, device=src.device)
        out = [[] for _ in range(b)]
        done = torch.zeros(b, dtype=torch.bool, device=src.device)
        for step in range(max_len):
            logits = self.decode(tok, memory, mem_mask, cache, offset=step)[:, -1]
            nxt = logits.argmax(-1)
            for i in range(b):
                if not done[i] and nxt[i].item() != EOS_ID:
                    out[i].append(nxt[i].item())
            done |= nxt == EOS_ID
            if done.all():
                break
            tok = nxt[:, None]
        return out

    @torch.no_grad()
    def beam_search(
        self, src: torch.Tensor, beam: int = 4, max_len: int | None = None, length_penalty: float = 1.0
    ) -> list[list[int]]:
        if beam <= 1:
            return self.greedy(src, max_len)
        self.eval()
        max_len = self._max_len(src, max_len)
        b, dev = src.size(0), src.device
        memory, mem_mask = self.encode(src)
        memory = memory.repeat_interleave(beam, 0)
        mem_mask = mem_mask.repeat_interleave(beam, 0)
        cache = self._new_cache()
        scores = torch.full((b, beam), float("-inf"), device=dev)
        scores[:, 0] = 0.0
        hyps: list[list[int]] = [[] for _ in range(b * beam)]
        tok = torch.full((b * beam, 1), BOS_ID, dtype=torch.long, device=dev)
        finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(b)]
        active = [True] * b

        for step in range(max_len):
            logp = F.log_softmax(self.decode(tok, memory, mem_mask, cache, offset=step)[:, -1].float(), -1)
            v = logp.size(-1)
            cand = (scores.view(-1, 1) + logp).view(b, beam * v)
            top_s, top_i = cand.topk(2 * beam, dim=1)
            new_scores = torch.full((b, beam), float("-inf"), device=dev)
            new_tok = torch.full((b * beam,), PAD_ID, dtype=torch.long, device=dev)
            origin = torch.arange(b * beam, device=dev)  # default: keep own row
            new_hyps: list[list[int]] = [[] for _ in range(b * beam)]
            for i in range(b):
                if not active[i]:
                    continue
                slot = 0
                for s, idx in zip(top_s[i].tolist(), top_i[i].tolist()):
                    k, t = divmod(idx, v)
                    src_row = i * beam + k
                    if s == float("-inf"):
                        break
                    if t == EOS_ID:
                        lp = ((5 + step + 1) / 6) ** length_penalty
                        finished[i].append((s / lp, hyps[src_row]))
                        continue
                    row = i * beam + slot
                    new_scores[i, slot] = s
                    new_tok[row] = t
                    origin[row] = src_row
                    new_hyps[row] = hyps[src_row] + [t]
                    slot += 1
                    if slot == beam:
                        break
                if slot == 0:
                    active[i] = False
                elif len(finished[i]) >= beam:
                    best_done = max(f[0] for f in finished[i])
                    best_alive = new_scores[i].max().item() / (((5 + max_len) / 6) ** length_penalty)
                    if best_done >= best_alive:
                        active[i] = False
            if not any(active):
                break
            scores, hyps = new_scores, new_hyps
            for i in range(b):
                if not active[i]:
                    scores[i] = float("-inf")
            self._reorder_cache(cache, origin)
            tok = new_tok[:, None]

        results = []
        for i in range(b):
            if not finished[i]:
                lp = ((5 + max_len) / 6) ** length_penalty
                finished[i] = [(scores[i, k].item() / lp, hyps[i * beam + k]) for k in range(beam)]
            results.append(max(finished[i], key=lambda f: f[0])[1])
        return results


def label_smoothed_loss(logits: torch.Tensor, target: torch.Tensor, eps: float) -> tuple[torch.Tensor, int]:
    """Sum of token losses and number of non-pad tokens."""
    logits = logits.float().view(-1, logits.size(-1))
    target = target.reshape(-1)
    loss = F.cross_entropy(logits, target, ignore_index=PAD_ID, label_smoothing=eps, reduction="sum")
    return loss, int((target != PAD_ID).sum().item())
