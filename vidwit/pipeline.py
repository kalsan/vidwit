from __future__ import annotations

import logging
import re
import shutil
import time
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

Output format — strict:
- Emit ONE markdown block and nothing else.
- The first character of your output must be `#` (the block header).
- Do NOT write a preamble, plan, analysis section, or commentary
  before or after the block. No "Visual Analysis:", "Synthesis Plan:",
  "Execution:", "The user wants…", etc. The block itself is the
  entire answer.
- Do NOT emit a second `###` header in the same response.

Conventions:
- Block header: `### [MM:SS.mmm – MM:SS.mmm) — short title [TAG]`
- Use the full `MM:SS.mmm` form on both ends and a closing `)`, never
  square brackets, never abbreviations like `06.000`.
- Tags: [FOOTAGE], [ANIM], [FOOTAGE + ANIM]
- Quote on-screen text verbatim (preserve case); attribute its source
  (title card, lower-third, subtitle, sign, infographic, label, etc.).
- Quote narration verbatim from the speech segments in the `# Speech segments` section. Each line there is already formatted as `[start – end) Speaker: "..."`; copy the text into a `>` blockquote and keep the timecode range, e.g. `> "..." — Speaker, [00:00.000 – 00:02.940)`.
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

    out_path = scratch.output_path(video, cfg.paths_home, cfg.output_override)
    if out_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{out_path} exists (use --overwrite)")

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

    total = len(plan)
    run_start = time.monotonic()
    completed_this_run = 0
    for w in plan:
        chunk_path = layout.chunks_dir / f"{w.label}.md"
        if chunk_path.exists() and cfg.resume:
            log.info("resume: skip chunk %d/%d (%s)", w.index + 1, total, chunk_path.name)
            tail = _push_tail(tail, chunk_path.read_text(encoding="utf-8"))
            continue
        t0 = time.monotonic()
        log.info(
            "chunk %d/%d [%s – %s) start",
            w.index + 1, total, _fmt(w.start), _fmt(w.end),
        )
        body = _process_window(
            w, frames, tx, cfg, provider, system_prompt, tail, rolling_summary, meta,
        )
        body = _sanitise_chunk(body)
        chunk_path.write_text(body, encoding="utf-8")
        tail = _push_tail(tail, body)
        completed_this_run += 1
        dt = time.monotonic() - t0
        elapsed = time.monotonic() - run_start
        avg = elapsed / completed_this_run
        remaining = total - (w.index + 1)
        eta_s = avg * remaining
        log.info(
            "chunk %d/%d done in %.1fs (avg %.1fs/chunk, ETA %s for %d more)",
            w.index + 1, total, dt, avg, _fmt_eta(eta_s), remaining,
        )

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
    return ffmpeg_io.extract_frames(
        video, layout.frames_dir,
        fps=cfg.fps, threads=cfg.jobs,
        max_width=cfg.frame_width, max_height=cfg.frame_height,
    )


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
    transcript_lines = _aggregate_transcript(words)

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


_SENTENCE_TERMINATORS = (".", "?", "!", "…")
_GAP_THRESHOLD_S = 0.6


_PREAMBLE_PREFIXES = (
    "The user wants me to",
    "The user is asking me",
    "**Metadata Check",
    "**Plan:**",
    "**Execution:**",
    "**Visual Analysis",
    "**Visual description synthesis",
    "**Audio synchronization",
    "**Synthesis Plan",
    "I will structure the block",
    "I will construct the output",
    "I will now construct",
    "Okay, let's break down",
    "Let me analyze",
    "Based on the provided",
)


def _sanitise_chunk(body: str) -> str:
    """Strip pre-block noise, duplicate sibling blocks, and trailing JSON /
    fence residue from a model response.

    Small / reasoning-leaky models sometimes emit a planning preamble
    (`The user wants me to...`, `**Visual Analysis:**`, etc.) before
    the actual `### [...]` header, occasionally a second block (with
    `### ` or with the malformed `# [` form we have observed on
    Gemma 4 e4b, sometimes glued onto a closing markdown fence like
    ```` ```# [...] ````), and trailing fence / JSON debug residue.

    Keep only the content from the first `### [...]` header up to (but
    not including) the next block-header-shaped line, then strip
    trailing fence + JSON-noise + blank lines.
    """
    # Force header markers onto their own line. Small models sometimes
    # glue `### [` or `# [` to the end of a preamble or fence line; if
    # the marker never appears at line start we cannot detect the
    # block boundary. Inserting a newline before any such marker fixes
    # both the start-search and the duplicate-cut search below.
    # Skip backtick context — `## [foo]` quoted inside a `...` span is
    # the model literally citing the template format, not a header.
    body = re.sub(r"([^\n`])(### \[)", r"\1\n\2", body)
    # Exclude `#` from the lookbehind so we don't fragment a real
    # `### [` header into `##\n# [`.
    body = re.sub(r"([^\n`#])(# \[)", r"\1\n\2", body)
    lines = body.splitlines()
    start = None
    second = None
    for i, line in enumerate(lines):
        s = line.lstrip()
        if start is None:
            # Accept `# [` as a fallback start: small models occasionally
            # drop two of the three hashes. We normalise it back to
            # `### [` on output so the assembler TOC still catches it.
            if s.startswith("### ") or s.startswith("# ["):
                start = i
            continue
        # After the first block header: cut at the next block-header
        # shape OR at a planning-preamble marker OR a `---` rule.
        # Small reasoning-leaky models sometimes:
        #   - emit two windows in one response separated by `---`
        #   - glue a duplicate header to the end of a preamble line
        #     (`...required format.# [...]`)
        #   - prefix the duplicate with a markdown fence
        if s.startswith("### ") or s.startswith("# ["):
            second = i
            break
        if s.lstrip("`").startswith("# ["):
            second = i
            break
        if s.startswith("---"):
            second = i
            break
        if s.startswith(_PREAMBLE_PREFIXES):
            second = i
            break
    if start is None:
        # No block header found at all — return body as-is so the
        # operator can see what went wrong; assembler/TOC will skip it.
        return body.rstrip() + "\n"
    end = second if second is not None else len(lines)
    cleaned = lines[start:end]
    # Normalise `# [...]` start header to canonical `### [...]`.
    if cleaned and cleaned[0].lstrip().startswith("# ["):
        head = cleaned[0]
        i = head.index("# [")
        cleaned[0] = head[:i] + "### " + head[i + 2:]
    # Strip trailing markdown-fence, JSON debug and blank lines.
    while len(cleaned) > 1 and _is_trailing_junk(cleaned[-1]):
        cleaned.pop()
    return "\n".join(cleaned).rstrip() + "\n"


def _is_trailing_junk(line: str) -> bool:
    """Lines we are willing to strip from the tail of a chunk."""
    s = line.strip()
    if not s:
        return True
    if s.startswith("```"):
        return True
    if s in ("[", "]", "{", "}"):
        return True
    # Single-line JSON object dump, e.g. {"start": "...", ...}
    if s.startswith('{"') and (s.endswith("}") or s.endswith(",")):
        return True
    # Single-line JSON array dump
    if s.startswith("[{") and (s.endswith("}]") or s.endswith("},")):
        return True
    return False


def _aggregate_transcript(words: list, gap_threshold_s: float = _GAP_THRESHOLD_S) -> list[str]:
    """Group whisper Words into sentence-like segments and format them for the LLM.

    Sentence break = either:
      - The previous word ends with `.`, `?`, `!`, or `…`, or
      - The gap to the next word exceeds `gap_threshold_s`.

    Output lines look like:
      [MM:SS.mmm – MM:SS.mmm) Narrator: "joined sentence text"

    Small models can copy these verbatim into the `>` blockquote convention
    without having to re-synthesise sentences from word-level fragments.
    """
    if not words:
        return []
    segments: list[list] = [[words[0]]]
    for prev, word in zip(words, words[1:]):
        prev_text = prev.text or ""
        ends_sentence = prev_text and prev_text[-1] in _SENTENCE_TERMINATORS
        gap = max(0.0, word.start - prev.end)
        if ends_sentence or gap > gap_threshold_s:
            segments.append([word])
        else:
            segments[-1].append(word)
    lines: list[str] = []
    for seg in segments:
        text = " ".join((w.text or "").strip() for w in seg if (w.text or "").strip())
        if not text:
            continue
        speaker = seg[0].speaker or "Speaker"
        lines.append(
            f"[{_fmt(seg[0].start)} – {_fmt(seg[-1].end)}) {speaker}: \"{text}\""
        )
    return lines


def _fmt_eta(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:06.3f}"
