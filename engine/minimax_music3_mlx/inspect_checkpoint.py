from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import collect_checkpoint, human_size, read_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect MiniMax Music 3 checkpoint files")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"checkpoint directory does not exist: {root}")

    files = collect_checkpoint(root)
    total = sum(item.size_bytes for item in files)
    print(f"checkpoint: {root}")
    print(f"files: {len(files)}")
    print(f"total: {human_size(total)}")
    for item in files:
        print(f"{human_size(item.size_bytes):>12}  {item.path.relative_to(root)}")

    print("\ncomponent configs:")
    for config in sorted(root.rglob("config.json")):
        data = read_json(config) or {}
        architecture = data.get("architectures") or data.get("_class_name") or data.get("model_type")
        print(f"- {config.relative_to(root)}: {architecture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
