from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlx_lm.utils import load_model


def normalize_qwen3_config(config: dict[str, Any]) -> dict[str, Any]:
    """Adapt the Transformers 5 Qwen3 config to MLX-LM's Qwen3 schema."""
    normalized = dict(config)
    rope_parameters = normalized.get("rope_parameters") or {}
    if "rope_theta" not in normalized and "rope_theta" in rope_parameters:
        normalized["rope_theta"] = rope_parameters["rope_theta"]
    if "rope_scaling" not in normalized:
        rope_type = rope_parameters.get("rope_type")
        normalized["rope_scaling"] = None if rope_type in (None, "default") else dict(rope_parameters)
    return normalized


def read_language_model_config(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError(f"expected qwen3 language model, got {config.get('model_type')!r}")
    return normalize_qwen3_config(config)


def load_language_model(model_dir: Path, *, lazy: bool = True):
    """Load MiniMax's customized Qwen3 checkpoint with MLX-LM."""
    config = read_language_model_config(model_dir)
    model, loaded_config = load_model(model_dir, lazy=lazy, strict=True, model_config=config)
    return model, loaded_config
