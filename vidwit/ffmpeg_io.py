from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFmpegMissingError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FFmpegMissingError("ffmpeg + ffprobe required on PATH")


@dataclass(slots=True, frozen=True)
class VideoInfo:
    path: Path
    duration_s: float
    width: int
    height: int
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    require_ffmpeg()
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    duration = float(data.get("format", {}).get("duration", 0.0))
    width = height = 0
    has_audio = False
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and width == 0:
            width = int(s.get("width", 0))
            height = int(s.get("height", 0))
        if s.get("codec_type") == "audio":
            has_audio = True
    return VideoInfo(path=path, duration_s=duration, width=width, height=height, has_audio=has_audio)


def extract_audio(src: Path, dst: Path, sample_rate: int = 16000, threads: int = 0) -> Path:
    """Extract mono PCM WAV at sample_rate; whisper expects 16k mono."""
    require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(src),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            "-threads", str(threads),
            str(dst),
        ],
        check=True,
    )
    return dst


def extract_frames(
    src: Path,
    dst_dir: Path,
    fps: float,
    threads: int = 0,
    quality: int = 4,
    max_width: int | None = None,
    max_height: int | None = None,
) -> list[Path]:
    """Extract frames at given fps as JPEG. Returns paths sorted by index.

    If max_width and max_height are set, frames are downscaled to fit within
    that box while preserving aspect ratio (no padding). 0 / None disables.
    """
    require_ffmpeg()
    dst_dir.mkdir(parents=True, exist_ok=True)
    pattern = dst_dir / "f_%08d.jpg"
    vf = f"fps={fps}"
    if max_width and max_height:
        vf += f",scale={max_width}:{max_height}:force_original_aspect_ratio=decrease"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(src),
            "-vf", vf,
            "-q:v", str(quality),
            "-threads", str(threads),
            str(pattern),
        ],
        check=True,
    )
    return sorted(dst_dir.glob("f_*.jpg"))
