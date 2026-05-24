from __future__ import annotations

import logging
import shutil
from pathlib import Path

from . import ffmpeg_io, llm, scratch, transcribe
from .assembler import assemble
from .chunker import Window, windows
from .config import Config


log = logging.getLogger("vidwit.pipeline")

_DEFAULT_SYSTEM_PROMPT = """\
You are vidwit, a multimodal witness. Given frames and a word-level
transcript for a short window of a video, write a single markdown
block describing exactly what is seen and heard in that window.
The user message carries a `# Capture metadata` block telling you the
frame sampling rate, window length, overlap, resolution and transcript
model. Calibrate your description against those — visual events
shorter than the frame spacing may be missed in the frames and must be
inferred from the transcript instead.
If the transcript looks like gibberish (random words, broken syntax,
or wildly inconsistent with the visible context), whisper has likely
failed — common for dialects or low-resource languages such as Swiss
German. Mark the affected timecodes with `[⚠ transcript unreliable]`
and skip narration quotes for them. Burned-in subtitles, if present,
belong in the visual description with an attribution like
`subtitle reads: "Hello world"` — never promote them into the audio
stream.

Conventions:
- Block header: `### [MM:SS.mmm – MM:SS.mmm) — short title [TAG]`
- Tags: [FOOTAGE], [ANIM], [FOOTAGE + ANIM]
- Quote on-screen text verbatim (preserve case); attribute its source
  (title card, lower-third, subtitle, sign, infographic, label, etc.).
- Quote narration in `>` blockquotes with speaker + word-precise timecode.
- Bracket non-speech audio: [lion roars].
- Re-introduce background only on change.
- Flag graphic content with `[⚠ <reason>]` and a skip range if useful.
"""


def run_one(video: Path, cfg: Config) -> Path:
    """Process one video file. Returns the final .md path written."""
    ffmpeg_io.require_ffmpeg()
    info = ffmpeg_io.probe(video)
    log.info("video: %s  duration=%.2fs  %dx%d  audio=%s",
             video.name, info.duration_s, info.width, info.height, info.has_audio)

    out_path = scratch.output_path(video, cfg.paths_home)
    if out_path.exists() and not cfg.overwrite and not cfg.resume:
        raise FileExistsError(f"{out_path} exists (use --overwrite or --resume)")

    layout = scratch.scratch_for(video, cfg.paths_temp)
    layout.ensure()

    # 1. Audio + transcript.
    tx = _ensure_transcript(video, layout, cfg)

    # 2. Frames at fps.
    frames = _ensure_frames(video, layout, cfg)

    # 3. Windowed loop.
    plan = windows(info.duration_s, cfg.window, cfg.overlap)
    log.info("planned %d windows (%.1fs each, %.1fs overlap)",
             len(plan), cfg.window, cfg.overlap)

    provider = llm.build(cfg.llm)
    system_prompt = _read_prompt(cfg)
    meta = llm.CaptureMeta(
        fps=cfg.fps,
        window_s=cfg.window,
        overlap_s=cfg.overlap,
        width=info.width or None,
        height=info.height or None,
        whisper_model=cfg.whisper_model,
        duration_s=info.duration_s,
        detected_language=tx.language,
        detected_language_probability=tx.language_probability,
        audio_language_hint=cfg.audio_language,
        notes=cfg.notes,
    )
    tail: list[str] = []
    rolling_summary = ""

    for w in plan:
        chunk_path = layout.chunks_dir / f"{w.label}.md"
        if chunk_path.exists() and cfg.resume:
            log.info("resume: skip %s", chunk_path.name)
            tail = _push_tail(tail, chunk_path.read_text(encoding="utf-8"))
            continue
        body = _process_window(
            w, frames, tx, cfg, provider, system_prompt, tail, rolling_summary, meta,
        )
        chunk_path.write_text(body, encoding="utf-8")
        tail = _push_tail(tail, body)

    # 4. Assemble + atomic publish via .part.
    final_md = assemble(layout.chunks_dir, video.name)
    part = scratch.part_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(final_md, encoding="utf-8")
    part.replace(out_path)
    log.info("wrote %s", out_path)

    if not cfg.keep_scratch:
        shutil.rmtree(layout.root, ignore_errors=True)
    return out_path


def _ensure_transcript(video: Path, layout: scratch.ScratchLayout, cfg: Config) -> transcribe.Transcript:
    if cfg.resume and layout.transcript_json.exists():
        log.info("resume: load transcript")
        return transcribe.Transcript.from_json(layout.transcript_json.read_text(encoding="utf-8"))
    if not layout.audio_wav.exists():
        log.info("extract audio → %s", layout.audio_wav.name)
        ffmpeg_io.extract_audio(video, layout.audio_wav, threads=cfg.jobs)
    log.info("transcribe (%s, %s)", cfg.whisper_model, cfg.whisper_device)
    tx = transcribe.transcribe(
        layout.audio_wav,
        model_name=cfg.whisper_model,
        device=cfg.whisper_device,
        default_speaker=cfg.default_speaker,
        language=cfg.audio_language,
    )
    log.info(
        "whisper detected language=%s prob=%.2f",
        tx.language, tx.language_probability or 0.0,
    )
    layout.transcript_json.write_text(tx.to_json(), encoding="utf-8")
    return tx


def _ensure_frames(video: Path, layout: scratch.ScratchLayout, cfg: Config) -> list[Path]:
    existing = sorted(layout.frames_dir.glob("f_*.jpg"))
    if cfg.resume and existing:
        log.info("resume: %d existing frames", len(existing))
        return existing
    log.info("extract frames @ %g fps", cfg.fps)
    return ffmpeg_io.extract_frames(video, layout.frames_dir, fps=cfg.fps, threads=cfg.jobs)


def _process_window(
    w: Window,
    frames: list[Path],
    tx: transcribe.Transcript,
    cfg: Config,
    provider: llm.Provider,
    system_prompt: str,
    tail: list[str],
    rolling_summary: str,
    _meta: llm.CaptureMeta,
) -> str:
    # Frame index = floor(t * fps). Frames are 1-indexed by ffmpeg (f_00000001.jpg).
    first = max(1, int(w.start * cfg.fps) + 1)
    last = max(first, int(w.end * cfg.fps))
    window_frames = [f for f in frames if first <= _frame_index(f) <= last]

    words = tx.slice(w.start, w.end)
    transcript_lines = [
        f"{_fmt(word.start)}–{_fmt(word.end)} "
        f"{(word.speaker + ': ') if word.speaker else ''}{word.text}"
        for word in words
    ]

    req = llm.ChunkRequest(
        system=system_prompt,
        rolling_summary=rolling_summary,
        tail_chunks=tail,
        transcript_lines=transcript_lines,
        frames=window_frames,
        window_start_s=w.start,
        window_end_s=w.end,
        meta=_meta,
    )
    return provider.vision_chat(req, max_output_tokens=cfg.llm.max_output_tokens)


def _frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _push_tail(tail: list[str], body: str, max_keep: int = 2) -> list[str]:
    out = [*tail, body]
    return out[-max_keep:]


def _read_prompt(cfg: Config) -> str:
    if cfg.prompt_path and cfg.prompt_path.exists():
        return cfg.prompt_path.read_text(encoding="utf-8")
    return _DEFAULT_SYSTEM_PROMPT


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:06.3f}"
