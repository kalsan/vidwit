from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class ScratchLayout:
    root: Path           # <scratch_base>/<hash>/
    audio_wav: Path
    frames_dir: Path
    chunks_dir: Path
    transcript_json: Path
    state_json: Path

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)


def video_hash(path: Path) -> str:
    """Cheap stable id: size + mtime + first/last 1MB sha256."""
    h = hashlib.sha256()
    st = path.stat()
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    with path.open("rb") as f:
        h.update(f.read(1024 * 1024))
        if st.st_size > 2 * 1024 * 1024:
            f.seek(-1024 * 1024, 2)
            h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]


def scratch_for(video: Path, temp_root: Path | None) -> ScratchLayout:
    """Sibling scratch by default, mirrors yt-dlp `--paths temp:` if temp_root set."""
    base = temp_root if temp_root is not None else video.parent / ".vidwit-tmp"
    root = base / video_hash(video)
    return ScratchLayout(
        root=root,
        audio_wav=root / "audio.wav",
        frames_dir=root / "frames",
        chunks_dir=root / "chunks",
        transcript_json=root / "transcript.json",
        state_json=root / "state.json",
    )


def output_path(
    video: Path,
    home_root: Path | None,
    override: Path | None = None,
) -> Path:
    """Final .md path.

    Precedence:
      - `override` (-o/--output) wins absolutely; relative paths resolved
        against cwd.
      - else `home_root` (yt-dlp-style `--paths home:`) sets the output dir.
      - else write next to the input video.
    """
    if override is not None:
        return override if override.is_absolute() else Path.cwd() / override
    stem = video.stem
    if home_root is None:
        return video.with_suffix(".md")
    return home_root / (stem + ".md")


def part_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".part")
