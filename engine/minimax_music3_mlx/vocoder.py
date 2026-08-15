from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha[None, None, :]
        return x + mx.square(mx.sin(alpha * x)) / (alpha + 1e-9)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.snake1 = Snake1d(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=3 * dilation)
        self.snake2 = Snake1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.conv2(self.snake2(self.conv1(self.snake1(x))))


class VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
        )
        self.res_unit1 = ResidualUnit(output_dim, 1)
        self.res_unit2 = ResidualUnit(output_dim, 3)
        self.res_unit3 = ResidualUnit(output_dim, 9)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_t1(self.snake1(x))
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        return self.res_unit3(x)


class Vocoder(nn.Module):
    """MLX port of the 44.1 kHz stereo MiniMax Flow-VAE decoder."""

    def __init__(self):
        super().__init__()
        self.dec_in_proj = nn.Conv1d(64, 1024, kernel_size=1)
        self.conv_in = nn.Conv1d(1024, 1536, kernel_size=7, padding=3)
        ratios = (8, 8, 4, 2)
        self.blocks = [
            VocoderBlock(1536 // (2**i), 1536 // (2 ** (i + 1)), stride) for i, stride in enumerate(ratios)
        ]
        self.snake_out = Snake1d(96)
        self.conv_out = nn.Conv1d(96, 1, kernel_size=7, padding=3)

    def __call__(self, latents: mx.array) -> mx.array:
        if latents.ndim != 3 or latents.shape[1] != 128:
            raise ValueError(f"latents must have shape [batch, 128, length], got {latents.shape}")
        batch, _, length = latents.shape
        x = latents.reshape(batch * 2, 64, length).transpose(0, 2, 1)
        x = self.conv_in(self.dec_in_proj(x))
        for block in self.blocks:
            x = block(x)
        waveform = mx.tanh(self.conv_out(self.snake_out(x)))
        return waveform.transpose(0, 2, 1).reshape(batch, 2, -1)

    def load_safetensors(self, path: Path, dtype=mx.float32) -> None:
        source = mx.load(str(path))
        consumed: set[str] = set()
        weights: list[tuple[str, mx.array]] = []

        def plain_conv(source_name: str, target_name: str, transpose: bool = False) -> None:
            weight = source[f"{source_name}.weight"]
            axes = (1, 2, 0) if transpose else (0, 2, 1)
            weights.append((f"{target_name}.weight", weight.transpose(*axes).astype(dtype)))
            consumed.add(f"{source_name}.weight")
            bias_name = f"{source_name}.bias"
            if bias_name in source:
                weights.append((f"{target_name}.bias", source[bias_name].astype(dtype)))
                consumed.add(bias_name)

        def norm_conv(source_name: str, target_name: str, transpose: bool = False) -> None:
            value = source[f"{source_name}.weight_v"].astype(mx.float32)
            gain = source[f"{source_name}.weight_g"].astype(mx.float32)
            normalized = value * gain / mx.maximum(mx.sqrt(mx.sum(mx.square(value), axis=(1, 2), keepdims=True)), 1e-12)
            axes = (1, 2, 0) if transpose else (0, 2, 1)
            weights.append((f"{target_name}.weight", normalized.transpose(*axes).astype(dtype)))
            consumed.update((f"{source_name}.weight_v", f"{source_name}.weight_g"))
            bias_name = f"{source_name}.bias"
            if bias_name in source:
                weights.append((f"{target_name}.bias", source[bias_name].astype(dtype)))
                consumed.add(bias_name)

        def snake(source_name: str, target_name: str) -> None:
            name = f"{source_name}.alpha"
            weights.append((f"{target_name}.alpha", source[name].reshape(-1).astype(dtype)))
            consumed.add(name)

        plain_conv("dec_in_proj", "dec_in_proj")
        norm_conv("conv_in", "conv_in")
        for block_index in range(4):
            base = f"blocks.{block_index}"
            snake(f"{base}.snake1", f"{base}.snake1")
            norm_conv(f"{base}.conv_t1", f"{base}.conv_t1", transpose=True)
            for unit_index in range(1, 4):
                unit = f"{base}.res_unit{unit_index}"
                snake(f"{unit}.snake1", f"{unit}.snake1")
                norm_conv(f"{unit}.conv1", f"{unit}.conv1")
                snake(f"{unit}.snake2", f"{unit}.snake2")
                norm_conv(f"{unit}.conv2", f"{unit}.conv2")
        snake("snake_out", "snake_out")
        norm_conv("conv_out", "conv_out")
        if consumed != set(source):
            raise ValueError(
                f"vocoder checkpoint mismatch: missing={sorted(set(source)-consumed)}, "
                f"unexpected={sorted(consumed-set(source))}"
            )
        self.load_weights(weights, strict=True)
        mx.eval(self.parameters())
