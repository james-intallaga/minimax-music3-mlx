from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointFile:
    path: Path
    size_bytes: int


def collect_checkpoint(root: Path) -> list[CheckpointFile]:
    """Collect local safetensors/PyTorch checkpoint files without loading them."""
    files: list[CheckpointFile] = []
    for pattern in ("*.safetensors", "*.safetensors.index.json", "*.pth"):
        for path in root.rglob(pattern):
            if path.is_file():
                files.append(CheckpointFile(path, path.stat().st_size))
    return sorted(files, key=lambda item: str(item.path))


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")
