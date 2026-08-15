from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from .prompt import AUDIO_CODE_OFFSET
from .rvq import RVQDepthDecoder

AUDIO_END_TOKEN_ID = 151670
SEMANTIC_VOCAB_SIZE = 16_384
AR_CFG_SCALE = 1.5
AR_TOP_K = 50


@dataclass(frozen=True)
class AutoregressiveResult:
    frame_hiddens: mx.array
    frame_codes: mx.array
    stopped_early: bool


class SeededSampler:
    def __init__(self, seed: int):
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self._key = mx.random.key(seed)

    def next_key(self) -> mx.array:
        self._key, key = mx.random.split(self._key, 2)
        return key

    def top_k(self, logits: mx.array, k: int = AR_TOP_K) -> mx.array:
        k = min(k, logits.shape[-1])
        candidate_ids = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        candidate_logits = mx.take_along_axis(logits, candidate_ids, axis=-1)
        selected = mx.random.categorical(candidate_logits, key=self.next_key())
        return mx.take_along_axis(candidate_ids, selected[..., None], axis=-1).squeeze(-1)

    def candidates(self, logits: mx.array, candidate_ids: mx.array) -> mx.array:
        candidate_logits = mx.take_along_axis(logits, candidate_ids, axis=-1)
        selected = mx.random.categorical(candidate_logits, key=self.next_key())
        return mx.take_along_axis(candidate_ids, selected[..., None], axis=-1).squeeze(-1)


def _selected_lm_logits(language_model, hidden: mx.array, token_ids: mx.array) -> mx.array:
    """Project only the legal music-token rows of the untied language-model head."""
    weights = language_model.lm_head.weight[token_ids]
    return hidden @ weights.T


def _embed_audio_frame(language_model, depth_decoder: RVQDepthDecoder, codes: mx.array) -> mx.array:
    semantic = language_model.model.embed_tokens(codes[:, :1] + AUDIO_CODE_OFFSET)
    offsets = mx.arange(7, dtype=codes.dtype)[None, :] * 1024
    residual = depth_decoder.audio_embeddings(codes[:, 1:] + offsets).sum(axis=1, keepdims=True)
    return (semantic + residual.astype(semantic.dtype)) * (8**-0.5)


def _generate_depth_codes(
    language_model,
    depth_decoder: RVQDepthDecoder,
    last_hidden: mx.array,
    semantic_code: mx.array,
    sampler: SeededSampler,
) -> tuple[mx.array, mx.array]:
    sequence = [depth_decoder.projection(last_hidden)[:, None, :]]
    semantic_embed = language_model.model.embed_tokens(semantic_code + AUDIO_CODE_OFFSET)
    sequence.append(depth_decoder.projection(semantic_embed)[:, None, :])
    codes = [semantic_code]
    hidden_parts = []

    for index in range(1, 8):
        hidden = depth_decoder(mx.concatenate(sequence, axis=1))[:, -1]
        hidden_parts.append(hidden[:1])
        logits = depth_decoder.audio_heads[index - 1](hidden).astype(mx.float32)
        conditional, unconditional = logits[:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
        code = mx.repeat(sampler.top_k(guided), 2)
        codes.append(code)
        if index < 7:
            embed = depth_decoder.audio_embeddings(code + (index - 1) * 1024)
            sequence.append(depth_decoder.projection(embed)[:, None, :])

    return mx.stack(codes, axis=1), mx.concatenate(hidden_parts, axis=-1)


def generate_frame_hiddens(
    language_model,
    depth_decoder: RVQDepthDecoder,
    text_ids: mx.array,
    *,
    max_frames: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> AutoregressiveResult:
    """Run MiniMax Music 3's reference semantic/RVQ autoregressive stage."""
    if text_ids.ndim != 2 or text_ids.shape[0] != 2:
        raise ValueError(f"text_ids must have shape [2, sequence], got {text_ids.shape}")
    if max_frames < 1 or max_frames > 9_000:
        raise ValueError("max_frames must be between 1 and 9000")

    sampler = SeededSampler(seed)
    cache = make_prompt_cache(language_model)
    hidden = language_model.model(text_ids, cache=cache)
    last_hidden = hidden[:, -1]
    mx.eval(last_hidden)

    legal_token_ids = mx.concatenate(
        (
            mx.array([AUDIO_END_TOKEN_ID], dtype=mx.int32),
            mx.arange(AUDIO_CODE_OFFSET, AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE, dtype=mx.int32),
        )
    )
    emitted_hiddens = []
    emitted_codes = []
    stopped_early = False

    # The first sampled frame advances beyond <|audio_start|> but is not emitted.
    for frame_index in range(max_frames + 1):
        logits = _selected_lm_logits(language_model, last_hidden, legal_token_ids).astype(mx.float32)
        conditional, unconditional = logits[:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
        top_conditional = mx.argpartition(-conditional, kth=AR_TOP_K - 1, axis=-1)[..., :AR_TOP_K]
        sampled_column = sampler.candidates(guided, top_conditional)
        mx.eval(sampled_column)
        if int(sampled_column.item()) == 0:
            stopped_early = True
            break

        semantic_code = mx.repeat(sampled_column - 1, 2)
        codes, depth_hidden = _generate_depth_codes(
            language_model, depth_decoder, last_hidden, semantic_code, sampler
        )
        mx.eval(codes, depth_hidden)
        if frame_index > 0:
            emitted_hiddens.append(mx.concatenate((last_hidden[:1], depth_hidden), axis=-1))
            emitted_codes.append(codes[:1])
            if progress is not None:
                progress(len(emitted_hiddens), max_frames)
            if len(emitted_hiddens) >= max_frames:
                break

        feedback = _embed_audio_frame(language_model, depth_decoder, codes)
        dummy_ids = mx.zeros((2, 1), dtype=mx.int32)
        hidden = language_model.model(dummy_ids, cache=cache, input_embeddings=feedback)
        last_hidden = hidden[:, -1]
        mx.eval(last_hidden)

    if not emitted_hiddens:
        raise RuntimeError("MiniMax Music 3 generated zero audio frames")
    return AutoregressiveResult(
        frame_hiddens=mx.stack(emitted_hiddens, axis=1),
        frame_codes=mx.stack(emitted_codes, axis=1),
        stopped_early=stopped_early,
    )
