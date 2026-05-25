from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import LLMConfig


@dataclass(slots=True, frozen=True)
class CaptureMeta:
    """Conditions under which the input materials were produced. Forwarded
    to the LLM so it can interpret frame gaps + transcript precision."""
    fps: float                       # sampling rate of frames passed in this request
    window_s: float                  # window length
    overlap_s: float                 # overlap with adjacent windows
    source_fps: float | None = None  # original video fps, if probed
    width: int | None = None
    height: int | None = None
    whisper_model: str | None = None
    duration_s: float | None = None  # full video duration
    detected_language: str | None = None        # whisper-detected language code
    detected_language_probability: float | None = None
    audio_language_hint: str | None = None      # user-supplied --audio-language
    notes: str | None = None                    # free-text from --notes


@dataclass(slots=True, frozen=True)
class ChunkRequest:
    system: str
    rolling_summary: str
    tail_chunks: list[str]
    transcript_lines: list[str]
    frames: list[Path]
    window_start_s: float
    window_end_s: float
    meta: CaptureMeta | None = None


class Provider(Protocol):
    def vision_chat(self, req: ChunkRequest, max_output_tokens: int) -> str: ...
    def text_chat(self, system: str, user: str, max_output_tokens: int) -> str: ...


def build(cfg: LLMConfig) -> Provider:
    p = cfg.provider.lower()
    if p == "dummy":
        return DummyProvider()
    if p == "anthropic":
        return AnthropicProvider(model=cfg.model or "claude-sonnet-4-6", api_key=cfg.api_key)
    if p in ("openai", "lmstudio", "openai-compat"):
        return OpenAICompatProvider(
            model=cfg.model,
            base_url=cfg.base_url or "https://api.openai.com/v1",
            api_key=cfg.api_key or "lm-studio",
        )
    raise ValueError(f"unknown llm provider: {cfg.provider}")


def _b64_image(path: Path) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return mime, data


def _user_text(req: ChunkRequest) -> str:
    parts: list[str] = [f"Window: [{req.window_start_s:.3f}s – {req.window_end_s:.3f}s)"]
    if req.meta is not None:
        parts.append(_meta_block(req.meta, len(req.frames)))
    if req.rolling_summary:
        parts.append("\n# Story so far\n" + req.rolling_summary)
    if req.tail_chunks:
        parts.append("\n# Recent witness output (verbatim tail)\n" + "\n---\n".join(req.tail_chunks))
    if req.transcript_lines:
        parts.append("\n# Transcript (word-level, this window)\n" + "\n".join(req.transcript_lines))
    parts.append(
        "\n# Task\n"
        "Write the witness record for this window only, following the format conventions."
    )
    return "\n".join(parts)


def _meta_block(m: CaptureMeta, n_frames: int) -> str:
    spacing = (1.0 / m.fps) if m.fps else float("nan")
    lines = [
        "\n# Capture metadata",
        f"- frame sampling: {m.fps:g} fps (≈ {spacing:.3f} s between attached frames)",
        f"- frames attached: {n_frames} (sequential, in time order)",
        f"- window length: {m.window_s:g} s, overlap with neighbours: {m.overlap_s:g} s",
    ]
    if m.source_fps is not None:
        lines.append(f"- original video fps: {m.source_fps:g}")
    if m.width and m.height:
        lines.append(f"- video resolution: {m.width}x{m.height}")
    if m.whisper_model:
        lines.append(f"- transcript model: faster-whisper {m.whisper_model} (word-level timestamps)")
    if m.duration_s is not None:
        lines.append(f"- total video duration: {m.duration_s:.2f} s")
    if m.audio_language_hint:
        lines.append(f"- audio language (user-supplied hint): {m.audio_language_hint}")
    if m.detected_language:
        conf = (
            f" (confidence {m.detected_language_probability:.2f})"
            if m.detected_language_probability is not None else ""
        )
        lines.append(f"- audio language (whisper-detected): {m.detected_language}{conf}")
    if m.notes:
        lines.append(f"- notes: {m.notes}")
    lines.append(
        "Note: events shorter than the frame spacing may be missed visually; "
        "rely on transcript timing for sub-frame events."
    )
    return "\n".join(lines)


def _post_json(url: str, payload: dict, headers: dict, timeout: float = 180.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e


class DummyProvider:
    """No-network placeholder. Useful for scaffolding + offline tests."""

    def vision_chat(self, req: ChunkRequest, max_output_tokens: int) -> str:
        ts = f"[{_fmt(req.window_start_s)} – {_fmt(req.window_end_s)})"
        body = [
            f"### {ts} — (dummy chunk)",
            "",
            f"_DummyProvider: {len(req.frames)} frames, "
            f"{len(req.transcript_lines)} transcript lines._",
            "",
        ]
        if req.transcript_lines:
            body.append("> " + " ".join(req.transcript_lines))
        return "\n".join(body)

    def text_chat(self, system: str, user: str, max_output_tokens: int) -> str:
        return f"(dummy summary; {len(user)} chars of input)"


class AnthropicProvider:
    """Anthropic Messages API via raw HTTPS (no SDK)."""

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, model: str, api_key: str | None):
        if not api_key:
            raise RuntimeError("anthropic provider needs api_key (ANTHROPIC_API_KEY)")
        self.model = model
        self.api_key = api_key

    def vision_chat(self, req: ChunkRequest, max_output_tokens: int) -> str:
        content: list[dict] = []
        for f in req.frames:
            mime, data = _b64_image(f)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": data},
            })
        content.append({"type": "text", "text": _user_text(req)})
        payload = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "system": req.system,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
        }
        data = _post_json(self.API_URL, payload, headers)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def text_chat(self, system: str, user: str, max_output_tokens: int) -> str:
        payload = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": self.API_VERSION}
        data = _post_json(self.API_URL, payload, headers)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class OpenAICompatProvider:
    """OpenAI Chat Completions schema (works for OpenAI + LM Studio)."""

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def vision_chat(self, req: ChunkRequest, max_output_tokens: int) -> str:
        content: list[dict] = []
        for f in req.frames:
            mime, data = _b64_image(f)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            })
        content.append({"type": "text", "text": _user_text(req)})
        payload = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": content},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = _post_json(f"{self.base_url}/chat/completions", payload, headers)
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        return msg.get("content") or ""

    def text_chat(self, system: str, user: str, max_output_tokens: int) -> str:
        payload = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = _post_json(f"{self.base_url}/chat/completions", payload, headers)
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        return msg.get("content") or ""


_SUMMARY_SYSTEM = """\
You maintain a 'story so far' for a video witness pipeline. Given the
previous summary and the newest scene's witness markdown, return an
updated summary that preserves continuity for later scenes.

Keep:
- Named entities (people, places, animals, objects) with their stable identities.
- Active through-lines (what was introduced and is still in play).
- Currently-established background (so later scenes can reference it
  without re-describing).

Drop redundant detail and chronology trivia. Past tense, third person.
Plain prose, no bullet lists, no headings, no quotes. Keep it under
~{max_chars} characters.
"""


def summarize(
    provider: Provider,
    prev_summary: str,
    new_chunk_md: str,
    max_chars: int = 2000,
    max_output_tokens: int = 600,
) -> str:
    system = _SUMMARY_SYSTEM.format(max_chars=max_chars)
    user = (
        f"PREVIOUS SUMMARY:\n{prev_summary or '(none yet)'}\n\n"
        f"NEW SCENE:\n{new_chunk_md}"
    )
    out = provider.text_chat(system, user, max_output_tokens=max_output_tokens)
    return out.strip()


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:06.3f}"
