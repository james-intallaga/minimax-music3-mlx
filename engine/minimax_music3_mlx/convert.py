from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx


def convert_file(source: Path, output: Path, dtype: str) -> int:
    """Convert one safetensors file to MLX .npz without loading all tensors at once."""
    output.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    arrays: dict[str, mx.array] = {}
    target_dtype = getattr(mx, dtype)
    for name, tensor in mx.load(str(source)).items():
        arrays[name] = tensor.astype(target_dtype)
        converted += 1
    mx.savez(str(output), **arrays)
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a MiniMax Music 3 safetensors file to MLX npz")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    args = parser.parse_args()
    if args.source.is_dir():
        files = sorted(args.source.glob("*.safetensors"))
        if not files:
            parser.error(f"no safetensors files found in {args.source}")
        manifest = []
        for source in files:
            destination = args.output / f"{source.stem}.npz"
            count = convert_file(source, destination, args.dtype)
            manifest.append({"source": str(source), "output": str(destination), "tensors": count})
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"converted {len(manifest)} files to {args.output}")
    else:
        count = convert_file(args.source, args.output, args.dtype)
        print(f"converted {count} tensors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
