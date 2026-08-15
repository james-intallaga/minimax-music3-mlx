from __future__ import annotations

from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .prompt import SPECIAL_TOKEN_IDS, build_prompt

AUDIO_CFG_TOKEN_ID = 151654


def load_tokenizer(tokenizer_dir: Path):
    # This accepts only an already-downloaded local directory; there is no Hub fallback.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)  # nosec B615
    for token, expected in SPECIAL_TOKEN_IDS.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected:
            raise ValueError(f"tokenizer mismatch for {token}: expected {expected}, got {actual}")
    return tokenizer


def tokenize_cfg_pair(tokenizer, caption: str, lyrics: str) -> np.ndarray:
    """Return the reference conditional/unconditional prompt pair."""
    conditional = np.asarray(tokenizer(build_prompt(caption, lyrics))["input_ids"], dtype=np.int32)
    if conditional.size > 5_000:
        raise ValueError(f"assembled prompt has {conditional.size} tokens; maximum is 5000")
    unconditional = conditional.copy()
    unconditional[1:-2] = AUDIO_CFG_TOKEN_ID
    return np.stack((conditional, unconditional))
