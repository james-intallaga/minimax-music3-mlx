from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def _partial_rope(x: mx.array, cos: mx.array, sin: mx.array, rotary_dim: int) -> mx.array:
    rotated = x[..., :rotary_dim]
    first, second = mx.split(rotated, 2, axis=-1)
    rotated_half = mx.concatenate((-second, first), axis=-1)
    rotated = rotated * cos[None, :, None, :].astype(x.dtype) + rotated_half * sin[None, :, None, :].astype(x.dtype)
    return mx.concatenate((rotated, x[..., rotary_dim:]), axis=-1)


class FlowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)
        self.heads = heads
        self.head_dim = head_dim

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, rotary_dim: int) -> mx.array:
        batch, length, _ = x.shape
        q = self.to_q(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.to_k(x).reshape(batch, length, self.heads, self.head_dim)
        v = self.to_v(x).reshape(batch, length, self.heads, self.head_dim)
        q = _partial_rope(q, cos, sin, rotary_dim).transpose(0, 2, 1, 3)
        k = _partial_rope(k, cos, sin, rotary_dim).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        return self.to_out(attended.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class FlowBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = FlowAttention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.ff_in = nn.Linear(dim, ff_inner_dim * 2)
        self.ff_out = nn.Linear(ff_inner_dim, dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, rotary_dim: int) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin, rotary_dim)
        gate_states, gate = mx.split(self.ff_in(self.norm2(x)), 2, axis=-1)
        return x + self.ff_out(gate_states * nn.silu(gate))


class FlowTransformer(nn.Module):
    """MiniMax Music 3's 2.4B flow-matching transformer in MLX."""

    def __init__(
        self,
        in_channels: int = 128,
        condition_dim: int = 2048,
        num_layers: int = 36,
        heads: int = 32,
        head_dim: int = 64,
        ff_inner_dim: int = 8192,
        rotary_dim: int = 32,
        fourier_dim: int = 256,
    ):
        super().__init__()
        inner = heads * head_dim
        concat_channels = 2 * in_channels + condition_dim
        self.time_proj_weight = mx.zeros((fourier_dim // 2, 1))
        self.time_linear_1 = nn.Linear(fourier_dim, inner)
        self.time_linear_2 = nn.Linear(inner, inner)
        self.preprocess_conv = nn.Conv1d(concat_channels, concat_channels, kernel_size=1, bias=False)
        self.proj_in = nn.Linear(concat_channels, inner, bias=False)
        self.transformer_blocks = [FlowBlock(inner, heads, head_dim, ff_inner_dim) for _ in range(num_layers)]
        self.proj_out = nn.Linear(inner, in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.rotary_dim = rotary_dim

    def _time_embedding(self, timestep: mx.array) -> mx.array:
        angles = 2.0 * math.pi * timestep[:, None].astype(mx.float32) * self.time_proj_weight[:, 0][None]
        fourier = mx.concatenate((mx.cos(angles), mx.sin(angles)), axis=-1).astype(
            self.time_linear_1.weight.dtype
        )
        return self.time_linear_2(nn.silu(self.time_linear_1(fourier)))

    def _rotary(self, length: int) -> tuple[mx.array, mx.array]:
        inv_freq = 1.0 / (10_000.0 ** (mx.arange(0, self.rotary_dim, 2).astype(mx.float32) / self.rotary_dim))
        freqs = mx.arange(length).astype(mx.float32)[:, None] * inv_freq[None]
        freqs = mx.concatenate((freqs, freqs), axis=-1)
        return mx.cos(freqs), mx.sin(freqs)

    def __call__(self, latents: mx.array, timestep: mx.array, condition: mx.array) -> mx.array:
        if latents.ndim != 3 or latents.shape[1] != 128:
            raise ValueError(f"latents must have shape [batch, 128, length], got {latents.shape}")
        latent_nlc = latents.transpose(0, 2, 1)
        if condition.shape[:2] != latent_nlc.shape[:2]:
            raise ValueError(f"condition timeline {condition.shape[:2]} does not match latents {latent_nlc.shape[:2]}")
        x = mx.concatenate((latent_nlc, mx.zeros_like(latent_nlc), condition), axis=-1)
        x = self.preprocess_conv(x) + x
        x = self.proj_in(x)
        x = mx.concatenate((self._time_embedding(timestep)[:, None], x), axis=1)
        cos, sin = self._rotary(x.shape[1])
        for block in self.transformer_blocks:
            x = block(x, cos, sin, self.rotary_dim)
        x = self.proj_out(x[:, 1:])
        x = self.postprocess_conv(x) + x
        return x.transpose(0, 2, 1)

    def load_safetensors(self, path: Path, dtype=mx.float32) -> None:
        files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"no safetensor shards found under {path}")
        source: dict[str, mx.array] = {}
        for file in files:
            shard = mx.load(str(file))
            duplicate = set(source).intersection(shard)
            if duplicate:
                raise ValueError(f"duplicate flow weights in {file}: {sorted(duplicate)}")
            source.update(shard)
        consumed: set[str] = set()
        weights: list[tuple[str, mx.array]] = []

        def direct(source_name: str, target_name: str | None = None) -> None:
            weights.append((target_name or source_name, source[source_name].astype(dtype)))
            consumed.add(source_name)

        def conv(source_name: str) -> None:
            direct(f"{source_name}.weight", f"{source_name}.weight")
            weights[-1] = (weights[-1][0], weights[-1][1].transpose(0, 2, 1))

        direct("time_proj.weight", "time_proj_weight")
        direct("time_embed.linear_1.weight", "time_linear_1.weight")
        direct("time_embed.linear_1.bias", "time_linear_1.bias")
        direct("time_embed.linear_2.weight", "time_linear_2.weight")
        direct("time_embed.linear_2.bias", "time_linear_2.bias")
        conv("preprocess_conv")
        direct("proj_in.weight")
        for index in range(len(self.transformer_blocks)):
            source_base = f"transformer_blocks.{index}"
            for name in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
                direct(f"{source_base}.{name}")
            for name in ("to_q.weight", "to_k.weight", "to_v.weight"):
                direct(f"{source_base}.attn.{name}")
            direct(f"{source_base}.attn.to_out.0.weight", f"{source_base}.attn.to_out.weight")
            for name in ("ff_in.weight", "ff_in.bias", "ff_out.weight", "ff_out.bias"):
                direct(f"{source_base}.{name}")
        direct("proj_out.weight")
        conv("postprocess_conv")
        if consumed != set(source):
            raise ValueError(
                f"flow checkpoint mismatch: missing={sorted(set(source)-consumed)}, "
                f"unexpected={sorted(consumed-set(source))}"
            )
        self.load_weights(weights, strict=True)
        mx.eval(self.parameters())
