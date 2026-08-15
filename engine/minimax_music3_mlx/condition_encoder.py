from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class ConditionEncoder(nn.Module):
    """Align MiniMax's eight AR hidden-state streams to the Flow-VAE timeline."""

    def __init__(self, hidden_dim: int = 4096, num_layers: int = 8, out_dim: int = 2048):
        super().__init__()
        self.layer_weight_logits = mx.zeros((num_layers,))
        self.layer_scale = mx.ones((1,))
        self.proj = nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch, frames, width = hidden_states.shape
        expected = self.num_layers * self.hidden_dim
        if width != expected:
            raise ValueError(f"expected hidden width {expected}, got {width}")
        streams = hidden_states.reshape(batch, frames, self.num_layers, self.hidden_dim)
        weights = mx.softmax(self.layer_weight_logits.astype(mx.float32)).astype(hidden_states.dtype)
        mixed = mx.sum(streams * weights[None, None, :, None], axis=2)
        projected = self.proj(mixed * self.layer_scale.astype(hidden_states.dtype))
        latent_length = max(1, int(frames * 44_100 / 24_000 * 960 / 512))
        indices = mx.floor(mx.arange(latent_length) * frames / latent_length).astype(mx.int32)
        return projected[:, indices]

    def load_safetensors(self, path: Path, dtype=mx.float32) -> None:
        source = mx.load(str(path))
        expected = {"layer_weight_logits", "layer_scale", "proj.weight", "proj.bias"}
        if set(source) != expected:
            raise ValueError(
                f"condition checkpoint mismatch: missing={sorted(expected-set(source))}, "
                f"unexpected={sorted(set(source)-expected)}"
            )
        weights = [
            ("layer_weight_logits", source["layer_weight_logits"].astype(dtype)),
            ("layer_scale", source["layer_scale"].astype(dtype)),
            ("proj.weight", source["proj.weight"].transpose(0, 2, 1).astype(dtype)),
            ("proj.bias", source["proj.bias"].astype(dtype)),
        ]
        self.load_weights(weights, strict=True)
        mx.eval(self.parameters())
