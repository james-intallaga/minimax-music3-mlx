from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def __call__(self, x: mx.array) -> mx.array:
        h = self.input_layernorm(x)
        batch, length, _ = h.shape
        q = self.to_q(h).reshape(batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.to_k(h).reshape(batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.to_v(h).reshape(batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(length).astype(q.dtype)
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask=mask)
        h = h.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        x = x + self.to_out(h)
        h = self.post_attention_layernorm(x)
        h = self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))
        return x + h


class RVQDepthDecoder(nn.Module):
    """MLX implementation of the c1..c7 MiniMax depth decoder."""

    def __init__(self, hidden_size=4096, num_heads=16, intermediate_size=6144, num_layers=4, vocab_size=1024):
        super().__init__()
        self.audio_embeddings = nn.Embedding(vocab_size * 7, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(16, hidden_size)
        self.layers = [DecoderBlock(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.audio_heads = [nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(7)]

    def __call__(self, seq: mx.array) -> mx.array:
        x = seq + self.pos_embedding(mx.arange(seq.shape[1]))[None]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def load_safetensors(self, path: Path, dtype=mx.bfloat16) -> None:
        source_to_target = {
            "audio_embeddings.weight": "audio_embeddings.weight",
            "projection.weight": "projection.weight",
            "pos_embedding.weight": "pos_embedding.weight",
            "norm.weight": "norm.weight",
        }
        for index in range(7):
            source_to_target[f"audio_heads.{index}.weight"] = f"audio_heads.{index}.weight"
        for index in range(len(self.layers)):
            source_to_target.update(
                {
                    f"layers.{index}.input_layernorm.weight": f"layers.{index}.input_layernorm.weight",
                    f"layers.{index}.post_attention_layernorm.weight": f"layers.{index}.post_attention_layernorm.weight",
                    f"layers.{index}.attn.to_q.weight": f"layers.{index}.to_q.weight",
                    f"layers.{index}.attn.to_k.weight": f"layers.{index}.to_k.weight",
                    f"layers.{index}.attn.to_v.weight": f"layers.{index}.to_v.weight",
                    f"layers.{index}.attn.to_out.weight": f"layers.{index}.to_out.weight",
                    f"layers.{index}.gate_proj.weight": f"layers.{index}.gate_proj.weight",
                    f"layers.{index}.up_proj.weight": f"layers.{index}.up_proj.weight",
                    f"layers.{index}.down_proj.weight": f"layers.{index}.down_proj.weight",
                }
            )
        weights = {}
        seen_sources = set()
        for source, tensor in mx.load(str(path)).items():
            target = source_to_target.get(source)
            if target is not None:
                weights[target] = tensor.astype(dtype)
                seen_sources.add(source)
        missing = sorted(set(source_to_target) - seen_sources)
        unexpected = sorted(seen_sources - set(source_to_target))
        if missing or unexpected:
            raise ValueError(f"RVQ checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.load_weights(list(weights.items()), strict=True)
        mx.eval(self.parameters())
