from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from .pipeline import MiniMaxMusic3Pipeline


def write_wav(path: Path, audio, sample_rate: int) -> None:
    samples = np.asarray(audio[0]).T
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate music locally with MiniMax Music 3 on Apple MLX")
    parser.add_argument("--caption", required=True, help="Description of the music")
    parser.add_argument("--lyrics", required=True, help="Lyrics with section tags such as [verse]")
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint"))
    parser.add_argument("--duration", type=float, default=10.0, help="Requested duration in seconds")
    parser.add_argument("--steps", type=int, default=30, help="Flow denoising steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pipeline = MiniMaxMusic3Pipeline(args.checkpoint)
    result = pipeline.generate(
        args.caption,
        args.lyrics,
        duration_seconds=args.duration,
        steps=args.steps,
        seed=args.seed,
        progress=print,
    )
    write_wav(args.output, result.audio, result.sample_rate)
    print(f"saved {args.output} ({result.audio.shape[-1] / result.sample_rate:.2f}s, stereo {result.sample_rate} Hz)")


if __name__ == "__main__":
    main()
