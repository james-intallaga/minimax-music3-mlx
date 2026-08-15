from __future__ import annotations

import json
import hmac
import logging
import os
import platform
import re
from difflib import SequenceMatcher
# Only fixed, absolute macOS utility paths are executed.
import subprocess  # nosec B404
import sys
import threading
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field


APP_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = Path(os.environ.get("MINIMAX_MUSIC3_HOME", APP_ROOT / "engine")).resolve()
CHECKPOINT = Path(os.environ.get("MINIMAX_MUSIC3_CHECKPOINT", APP_ROOT / "model" / "MiniMax-Music3")).resolve()
WHISPER_CHECKPOINT = Path(os.environ.get("MINIMAX_MUSIC3_WHISPER_CHECKPOINT", APP_ROOT / "model" / "whisper-large-v3-turbo")).resolve()
DATA_ROOT = Path(os.environ.get("MINIMAX_MUSIC3_DATA", APP_ROOT / "data")).resolve()
SONGS_ROOT = DATA_ROOT / "songs"
INDEX_PATH = DATA_ROOT / "songs.json"
REQUIRED_COMPONENTS = (
    "tokenizer/tokenizer.json",
    "language_model/model.safetensors.index.json",
    "rvq_depth_decoder/diffusion_pytorch_model.safetensors",
    "condition_encoder/diffusion_pytorch_model.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vocoder/diffusion_pytorch_model.safetensors",
)
MUSIC_MODEL_REPO = "MiniMaxAI/MiniMax-Music3"
MUSIC_MODEL_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
WHISPER_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
WHISPER_MODEL_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
LOCAL_API_TOKEN = os.environ.get("MINIMAX_MUSIC3_LOCAL_TOKEN", "")
logger = logging.getLogger("minimax_music3.local")

DATA_ROOT.mkdir(parents=True, exist_ok=True)
SONGS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MiniMax Music 3 MLX", docs_url=None, redoc_url=None)


@app.middleware("http")
async def require_local_token(request: Request, call_next):
    """Keep the engine private even if a malicious website probes localhost."""
    if request.url.path == "/":
        return await call_next(request)
    if not LOCAL_API_TOKEN:
        return JSONResponse({"detail": "The local engine must be started with start-local.sh."}, status_code=503)
    supplied = request.headers.get("authorization", "")
    expected = f"Bearer {LOCAL_API_TOKEN}"
    if not hmac.compare_digest(supplied, expected):
        return JSONResponse({"detail": "Not authorized."}, status_code=401)
    return await call_next(request)


class GenerationRequest(BaseModel):
    caption: str = Field(min_length=3, max_length=4000)
    lyrics: str = Field(min_length=1, max_length=25_000)
    duration_seconds: int = Field(ge=60, le=300)
    seed: int = Field(default=7, ge=0, le=2_147_483_647)
    steps: int = Field(default=30, ge=1, le=50)


class CancelledByUser(RuntimeError):
    pass


state_lock = threading.RLock()
cancel_event = threading.Event()
current_job: dict[str, Any] | None = None
pipeline_instance = None
model_download: dict[str, Any] = {"running": False, "message": ""}
DOWNLOAD_TOTAL_GB = 28.1


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_ready() -> bool:
    return all((CHECKPOINT / relative).is_file() for relative in REQUIRED_COMPONENTS) and (WHISPER_CHECKPOINT / "config.json").is_file()


def memory_gb() -> float:
    try:
        # The executable and all arguments are fixed; no request data reaches this call.
        result = subprocess.run(  # nosec B603
            ["/usr/sbin/sysctl", "-n", "hw.memsize"], check=True, capture_output=True, text=True
        )
        return round(int(result.stdout.strip()) / 2**30, 1)
    except Exception:
        return 0.0


def model_size_gb() -> float:
    roots = [CHECKPOINT]
    if not WHISPER_CHECKPOINT.is_relative_to(CHECKPOINT):
        roots.append(WHISPER_CHECKPOINT)
    total = sum(path.stat().st_size for root in roots if root.exists() for path in root.rglob("*") if path.is_file())
    return round(total / 2**30, 1)


def installed_bytes() -> int:
    roots = [CHECKPOINT, WHISPER_CHECKPOINT]
    total = 0
    seen: set[tuple[int, int]] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or ".cache" in path.parts:
                continue
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += stat.st_size
    return total


def public_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if job is None:
        return None
    result = {key: value for key, value in job.items() if key not in {"request", "started_monotonic"}}
    started = job.get("started_monotonic")
    elapsed = max(0.0, time.monotonic() - started) if started else 0.0
    result["elapsed_seconds"] = round(elapsed, 1)
    progress = float(job.get("progress", 0.0))
    result["eta_seconds"] = round(elapsed * (1.0 - progress) / progress, 1) if progress > 0.025 else None
    return result


def load_songs() -> list[dict[str, Any]]:
    try:
        data = json.loads(INDEX_PATH.read_text())
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_songs(songs: list[dict[str, Any]]) -> None:
    temporary = INDEX_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(songs, indent=2))
    temporary.replace(INDEX_PATH)


def song_title(caption: str) -> str:
    cleaned = re.sub(r"^(global metadata|genre)\s*:\s*", "", caption.strip(), flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z0-9'’-]+", cleaned)
    title = " ".join(words[:6]).strip()
    return title[:1].upper() + title[1:] if title else "My local song"


def lyric_lines(lyrics: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for raw in lyrics.splitlines():
        text = raw.strip()
        if not text:
            continue
        lines.append({"index": len(lines), "text": text, "section": bool(re.fullmatch(r"\[[^]]+\]", text))})
    return lines


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def fallback_lyric_timing(lyrics: str, duration: float) -> list[dict[str, Any]]:
    lines = lyric_lines(lyrics)
    sung = [line for line in lines if not line["section"]]
    total_weight = sum(max(1, len(normalized_words(line["text"]))) for line in sung)
    cursor = min(4.0, duration * 0.08)
    usable = max(0.1, duration - cursor - min(2.0, duration * 0.04))
    for line in sung:
        weight = max(1, len(normalized_words(line["text"])))
        line["start_ms"] = round(cursor * 1000)
        cursor += usable * weight / max(total_weight, 1)
        line["end_ms"] = round(min(duration, cursor) * 1000)
    for position, line in enumerate(lines):
        if line["section"]:
            next_sung = next((item for item in lines[position + 1 :] if not item["section"]), None)
            line["start_ms"] = next_sung.get("start_ms", 0) if next_sung else 0
            line["end_ms"] = line["start_ms"]
    return lines


def align_lyrics(lyrics: str, wav_path: Path, duration: float) -> list[dict[str, Any]]:
    """Align supplied lyric lines to locally detected sung words, with a deterministic fallback."""
    fallback = fallback_lyric_timing(lyrics, duration)
    if not any(not line["section"] for line in fallback):
        return []
    try:
        import mlx_whisper
        import numpy as np
        from scipy.signal import resample_poly

        with wave.open(str(wav_path), "rb") as source:
            channels = source.getnchannels()
            rate = source.getframerate()
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        audio = samples.reshape(-1, channels).mean(axis=1) if channels > 1 else samples
        if rate != 16_000:
            divisor = __import__("math").gcd(rate, 16_000)
            audio = resample_poly(audio, 16_000 // divisor, rate // divisor).astype(np.float32)

        prompt = " ".join(line["text"] for line in fallback if not line["section"])
        transcript = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=str(WHISPER_CHECKPOINT),
            word_timestamps=True,
            initial_prompt=prompt[:1800],
            condition_on_previous_text=True,
        )
        heard = [word for segment in transcript.get("segments", []) for word in segment.get("words", [])]
        heard_tokens = [normalized_words(str(word.get("word", ""))) for word in heard]
        heard_tokens = [tokens[0] if tokens else "" for tokens in heard_tokens]
        cursor = 0
        for line in fallback:
            if line["section"]:
                continue
            wanted = normalized_words(line["text"])
            if not wanted:
                continue
            best: tuple[float, int, int] | None = None
            search_end = min(len(heard_tokens), cursor + max(80, len(wanted) * 5))
            for start in range(cursor, search_end):
                for size in range(max(1, len(wanted) - 2), len(wanted) + 3):
                    end = min(len(heard_tokens), start + size)
                    if end <= start:
                        continue
                    score = SequenceMatcher(None, wanted, heard_tokens[start:end]).ratio()
                    if best is None or score > best[0]:
                        best = (score, start, end)
            if best and best[0] >= 0.52:
                _, start, end = best
                start_seconds = float(heard[start].get("start", 0.0))
                end_seconds = float(heard[end - 1].get("end", start_seconds + 0.5))
                line["start_ms"] = round(max(0.0, start_seconds) * 1000)
                line["end_ms"] = round(min(duration, max(start_seconds + 0.1, end_seconds)) * 1000)
                line["detected"] = True
                cursor = end
        sung_lines = [line for line in fallback if not line["section"]]
        anchor_positions = [index for index, line in enumerate(sung_lines) if line.get("detected")]
        boundaries = [-1, *anchor_positions, len(sung_lines)]
        for boundary_index in range(len(boundaries) - 1):
            left = boundaries[boundary_index]
            right = boundaries[boundary_index + 1]
            missing = sung_lines[left + 1 : right]
            if not missing:
                continue
            start_ms = int(sung_lines[left]["end_ms"]) if left >= 0 else 0
            end_ms = int(sung_lines[right]["start_ms"]) if right < len(sung_lines) else round(duration * 1000)
            span = max(0, end_ms - start_ms)
            total = sum(max(1, len(normalized_words(line["text"]))) for line in missing)
            cursor_ms = start_ms
            for line in missing:
                weight = max(1, len(normalized_words(line["text"])))
                line["start_ms"] = cursor_ms
                cursor_ms += round(span * weight / max(total, 1))
                line["end_ms"] = min(end_ms, cursor_ms)
        for position, line in enumerate(fallback):
            if line["section"]:
                next_sung = next((item for item in fallback[position + 1 :] if not item["section"]), None)
                line["start_ms"] = next_sung.get("start_ms", line["start_ms"]) if next_sung else line["start_ms"]
                line["end_ms"] = line["start_ms"]
        return fallback
    except Exception:
        return fallback


def update_job(**changes: Any) -> None:
    global current_job
    with state_lock:
        if current_job is not None:
            current_job.update(changes)


def on_progress(message: str) -> None:
    if cancel_event.is_set():
        raise CancelledByUser("Generation stopped")

    compose = re.fullmatch(r"compose frame (\d+)/(\d+)", message)
    denoise = re.fullmatch(r"denoise chunk (\d+)/(\d+), step (\d+)/(\d+)", message)
    if compose:
        current, total = map(int, compose.groups())
        fraction = current / max(total, 1)
        update_job(phase="Composing", progress=0.03 + 0.70 * fraction, message=f"Writing music frame {current:,} of {total:,}")
    elif denoise:
        chunk, chunks, step, steps = map(int, denoise.groups())
        fraction = ((chunk - 1) * steps + step) / max(chunks * steps, 1)
        update_job(phase="Producing", progress=0.74 + 0.23 * fraction, message=f"Producing section {chunk} of {chunks} · pass {step} of {steps}")
    elif message.startswith("music tokens complete"):
        update_job(phase="Producing", progress=0.74, message="The composition is ready. Shaping the final sound…")
    elif message.startswith("generating"):
        update_job(phase="Composing", progress=0.03, message=message.capitalize())


def ensure_pipeline():
    global pipeline_instance
    if pipeline_instance is not None:
        return pipeline_instance
    if not model_ready():
        raise RuntimeError("MiniMax Music 3 is not installed yet")
    if not ENGINE_ROOT.is_dir():
        raise RuntimeError("The local MLX engine could not be found")
    if str(ENGINE_ROOT) not in sys.path:
        sys.path.insert(0, str(ENGINE_ROOT))
    update_job(status="loading", phase="Getting ready", progress=0.01, message="Loading MiniMax Music 3 into unified memory…")
    from minimax_music3_mlx.pipeline import MiniMaxMusic3Pipeline

    pipeline_instance = MiniMaxMusic3Pipeline(CHECKPOINT)
    return pipeline_instance


def run_generation(job_id: str, request: GenerationRequest) -> None:
    global current_job
    keep_awake = None
    try:
        try:
            # The executable and all arguments are fixed; no request data reaches this call.
            keep_awake = subprocess.Popen(  # nosec B603
                ["/usr/bin/caffeinate", "-dimsu"]
            )
        except OSError:
            keep_awake = None
        pipeline = ensure_pipeline()
        update_job(status="running", phase="Composing", progress=0.025, message="Planning the melody and arrangement…")
        result = pipeline.generate(
            request.caption,
            request.lyrics,
            duration_seconds=request.duration_seconds,
            steps=request.steps,
            seed=request.seed,
            progress=on_progress,
        )
        if cancel_event.is_set():
            raise CancelledByUser("Generation stopped")
        update_job(phase="Finishing", progress=0.985, message="Saving your stereo WAV…")
        from minimax_music3_mlx.cli import write_wav

        filename = f"{job_id}.wav"
        wav_path = SONGS_ROOT / filename
        write_wav(wav_path, result.audio, result.sample_rate)
        duration = round(result.audio.shape[-1] / result.sample_rate, 2)
        update_job(phase="Syncing lyrics", progress=0.99, message="Listening back and syncing each lyric line…")
        timing = align_lyrics(request.lyrics, wav_path, duration)
        song = {
            "id": job_id,
            "title": song_title(request.caption),
            "caption": request.caption,
            "duration_seconds": duration,
            "created_at": now_iso(),
            "audio_url": f"/media/{filename}",
            "lyrics": request.lyrics,
            "lyric_timing": timing,
        }
        with state_lock:
            songs = [song, *[item for item in load_songs() if item.get("id") != job_id]]
            save_songs(songs[:100])
            if current_job and current_job.get("id") == job_id:
                current_job.update(
                    status="completed",
                    phase="Ready",
                    progress=1.0,
                    message="Your song is ready.",
                    audio_url=song["audio_url"],
                    completed_at=now_iso(),
                )
    except CancelledByUser:
        update_job(status="cancelled", phase="Stopped", message="Generation stopped safely.", completed_at=now_iso())
    except Exception:
        logger.exception("Song generation failed")
        update_job(
            status="failed",
            phase="Could not finish",
            message="The local engine hit a problem.",
            error="Generation failed. Check the local Terminal window for details.",
            completed_at=now_iso(),
        )
    finally:
        if keep_awake is not None:
            keep_awake.terminate()


def run_model_download() -> None:
    model_download.update(running=True, message="Downloading MiniMax‑Music3…")
    try:
        from huggingface_hub import snapshot_download

        CHECKPOINT.mkdir(parents=True, exist_ok=True)
        snapshot_download(MUSIC_MODEL_REPO, revision=MUSIC_MODEL_REVISION, local_dir=CHECKPOINT)
        model_download.update(message="Adding fully local lyric sync…")
        WHISPER_CHECKPOINT.mkdir(parents=True, exist_ok=True)
        snapshot_download(WHISPER_MODEL_REPO, revision=WHISPER_MODEL_REVISION, local_dir=WHISPER_CHECKPOINT)
        model_download.update(running=False, message="Model download complete.")
    except Exception:
        logger.exception("Model download failed")
        model_download.update(running=False, message="Download stopped. Check your connection, then try again.")


@app.get("/api/status")
def status() -> dict[str, Any]:
    ready = model_ready()
    installed = installed_bytes()
    is_apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    if model_download["running"]:
        message = model_download["message"]
    elif ready:
        message = "MiniMax‑Music3 is installed and ready."
    elif not is_apple:
        message = "This MLX edition requires an Apple Silicon Mac."
    else:
        message = "The one-time local model download needs about 29 GB."
    return {
        "engine_ready": ENGINE_ROOT.is_dir() and is_apple,
        "model_ready": ready,
        "downloading": model_download["running"],
        "device": f"{platform.machine()} Mac" if platform.system() == "Darwin" else platform.platform(),
        "memory_gb": memory_gb(),
        "model_size_gb": model_size_gb(),
        "downloaded_gb": round(installed / 2**30, 1),
        "download_total_gb": DOWNLOAD_TOTAL_GB,
        "download_progress": 1.0 if ready else min(0.99, installed / (DOWNLOAD_TOTAL_GB * 2**30)),
        "message": message,
    }


@app.get("/api/songs")
def songs() -> list[dict[str, Any]]:
    return load_songs()


@app.get("/api/generations/current")
def generation_status() -> dict[str, Any] | None:
    with state_lock:
        return public_job(current_job)


@app.post("/api/generations")
def create_generation(request: GenerationRequest) -> dict[str, Any]:
    global current_job
    if not model_ready():
        raise HTTPException(status_code=409, detail="Install MiniMax‑Music3 before creating a song.")
    with state_lock:
        if current_job and current_job.get("status") in {"queued", "loading", "running"}:
            raise HTTPException(status_code=409, detail="A song is already being made on this Mac.")
        cancel_event.clear()
        job_id = uuid.uuid4().hex[:12]
        current_job = {
            "id": job_id,
            "status": "queued",
            "phase": "Queued",
            "progress": 0.0,
            "message": "Your song is next.",
            "created_at": now_iso(),
            "started_monotonic": time.monotonic(),
            "request": request.model_dump(),
        }
        thread = threading.Thread(target=run_generation, args=(job_id, request), daemon=True, name="music3-local-generation")
        thread.start()
        return public_job(current_job) or {}


@app.post("/api/generations/cancel")
def cancel_generation() -> dict[str, Any]:
    with state_lock:
        if not current_job or current_job.get("status") not in {"queued", "loading", "running"}:
            raise HTTPException(status_code=409, detail="There is no active song to stop.")
        cancel_event.set()
        current_job["message"] = "Stopping safely after the current music frame…"
        return public_job(current_job) or {}


@app.post("/api/model/download")
def download_model() -> dict[str, str]:
    if model_ready():
        return {"message": "The model is already installed."}
    if model_download["running"]:
        return {"message": model_download["message"]}
    threading.Thread(target=run_model_download, daemon=True, name="music3-model-download").start()
    return {"message": "Model download started."}


@app.get("/media/{filename}")
def media(filename: str):
    if not re.fullmatch(r"[a-f0-9]{12}\.wav", filename):
        raise HTTPException(status_code=404, detail="Song not found")
    path = SONGS_ROOT / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Song not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "MiniMax Music 3 MLX", "status": "ready"}
