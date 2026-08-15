from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx

from .autoregressive import generate_frame_hiddens
from .condition_encoder import ConditionEncoder
from .flow_transformer import FlowTransformer
from .language_model import load_language_model
from .rvq import RVQDepthDecoder
from .tokenizer import load_tokenizer, tokenize_cfg_pair
from .vocoder import Vocoder

FRAME_RATE = 25
CHUNK_FRAMES = 200
CHUNK_HOP = 100
OVERLAP_LATENTS = 172
CROP_LEFT_LATENTS = 86
CROP_RIGHT_LATENTS = 258
SAMPLE_RATE = 44_100


@dataclass(frozen=True)
class GenerationResult:
    audio: mx.array
    sample_rate: int
    frame_codes: mx.array


class MiniMaxMusic3Pipeline:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        self.tokenizer = load_tokenizer(checkpoint / "tokenizer")
        self.language_model, _ = load_language_model(checkpoint / "language_model", lazy=True)
        self.depth_decoder = RVQDepthDecoder()
        self.depth_decoder.load_safetensors(
            checkpoint / "rvq_depth_decoder" / "diffusion_pytorch_model.safetensors"
        )
        self.condition_encoder = ConditionEncoder()
        self.condition_encoder.load_safetensors(
            checkpoint / "condition_encoder" / "diffusion_pytorch_model.safetensors"
        )
        self.flow_transformer = FlowTransformer()
        self.flow_transformer.load_safetensors(checkpoint / "transformer")
        self.vocoder = Vocoder()
        self.vocoder.load_safetensors(checkpoint / "vocoder" / "diffusion_pytorch_model.safetensors")

    @staticmethod
    def _flow_schedule(steps: int) -> tuple[list[float], list[float]]:
        if steps < 1:
            raise ValueError("steps must be positive")
        # Released scheduler config: invert_sigmas=true, num_train_timesteps=1.
        # Diffusers first creates [1, ..., 1/steps], then inversion changes both
        # scheduler sigmas *and model timesteps* to [0, ..., 1 - 1/steps].
        timesteps = [i / steps for i in range(steps)]
        scheduler_sigmas = timesteps + [1.0]
        return timesteps, scheduler_sigmas

    def _denoise_chunks(
        self,
        frame_hiddens: mx.array,
        *,
        steps: int,
        seed: int,
        progress: Callable[[str], None] | None,
    ) -> list[mx.array]:
        num_frames = frame_hiddens.shape[1]
        chunk_starts = [0] if num_frames <= CHUNK_FRAMES else list(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))
        timesteps, sigmas = self._flow_schedule(steps)
        random_key = mx.random.key(seed)
        latent_chunks: list[mx.array] = []
        previous_latent = None
        previous_condition = None

        for chunk_index, start in enumerate(chunk_starts):
            end = min(start + CHUNK_FRAMES, num_frames)
            condition = self.condition_encoder(frame_hiddens[:, start:end].astype(mx.float32)).astype(mx.float32)
            overlap = 0
            if previous_latent is not None:
                overlap = min(previous_latent.shape[-1], condition.shape[1])
                condition = mx.concatenate((previous_condition[:, :overlap], condition[:, overlap:]), axis=1)

            random_key, chunk_key = mx.random.split(random_key, 2)
            latents = mx.random.normal((1, 128, condition.shape[1]), key=chunk_key).astype(mx.float32)
            noise_prompt = latents[..., :overlap] if overlap else None

            for step_index, timestep_value in enumerate(timesteps):
                if overlap:
                    t = mx.array(timestep_value, dtype=latents.dtype)
                    blended = (1.0 - (1.0 - 1e-6) * t) * noise_prompt + t * previous_latent[..., :overlap]
                    latents = mx.concatenate((blended, latents[..., overlap:]), axis=-1)

                model_latents = mx.concatenate((latents, latents), axis=0)
                model_condition = mx.concatenate((condition, mx.zeros_like(condition)), axis=0)
                timestep = mx.full((2,), timestep_value, dtype=latents.dtype)
                prediction = self.flow_transformer(model_latents, timestep, model_condition)
                conditional, unconditional = prediction[:1], prediction[1:]
                velocity = unconditional + 1.7 * (conditional - unconditional)
                dt = sigmas[step_index + 1] - sigmas[step_index]
                latents = (latents.astype(mx.float32) + dt * velocity.astype(mx.float32)).astype(velocity.dtype)
                mx.eval(latents)
                if progress:
                    progress(f"denoise chunk {chunk_index + 1}/{len(chunk_starts)}, step {step_index + 1}/{steps}")

            if overlap:
                latents = mx.concatenate((previous_latent[..., :overlap], latents[..., overlap:]), axis=-1)
            overlap_start = max(0, latents.shape[-1] - 2 * OVERLAP_LATENTS)
            overlap_end = max(overlap_start, latents.shape[-1] - OVERLAP_LATENTS)
            previous_latent = latents[..., overlap_start:overlap_end]
            previous_condition = condition[:, overlap_start:overlap_end]
            latent_chunks.append(latents)
        return latent_chunks

    def _decode_chunks(self, latent_chunks: list[mx.array]) -> mx.array:
        waveforms = []
        for index, latents in enumerate(latent_chunks):
            waveform = self.vocoder(latents.astype(mx.float32))
            left = 0 if index == 0 else CROP_LEFT_LATENTS * 512
            right = 0 if index == len(latent_chunks) - 1 else CROP_RIGHT_LATENTS * 512
            waveforms.append(waveform[..., left : waveform.shape[-1] - right if right else None])
        audio = mx.concatenate(waveforms, axis=-1).astype(mx.float32)
        return mx.clip(audio, -1.0, 1.0)

    def generate(
        self,
        caption: str,
        lyrics: str,
        *,
        duration_seconds: float,
        steps: int = 30,
        seed: int = 0,
        progress: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        if not 0.04 <= duration_seconds <= 360:
            raise ValueError("duration_seconds must be between 0.04 and 360")
        max_frames = max(1, round(duration_seconds * FRAME_RATE))
        text_ids = mx.array(tokenize_cfg_pair(self.tokenizer, caption, lyrics))
        if progress:
            progress(f"generating {max_frames} music-token frames")
        ar = generate_frame_hiddens(
            self.language_model,
            self.depth_decoder,
            text_ids,
            max_frames=max_frames,
            seed=seed,
            progress=(lambda current, total: progress(f"compose frame {current}/{total}")) if progress else None,
        )
        if progress:
            progress("music tokens complete; starting flow denoising")
        chunks = self._denoise_chunks(ar.frame_hiddens, steps=steps, seed=seed + 1, progress=progress)
        audio = self._decode_chunks(chunks)
        mx.eval(audio)
        return GenerationResult(audio=audio, sample_rate=SAMPLE_RATE, frame_codes=ar.frame_codes)
