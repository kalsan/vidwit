from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path


DEFAULT_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi")

CONFIG_FILENAME = "vidwit.toml"


@dataclass(slots=True)
class LLMConfig:
    provider: str = "dummy"  # "anthropic" | "openai" | "lmstudio" | "dummy"
    model: str = ""
    base_url: str | None = None
    api_key: str | None = None
    max_output_tokens: int = 2048
    request_timeout: float = 600.0          # seconds; bumped for slow local models
    extra_body: dict = field(default_factory=dict)  # merged into chat-completions payload


@dataclass(slots=True)
class Config:
    fps: float = 1.0
    window: float = 10.0
    overlap: float = 1.0
    overwrite: bool = False
    resume: bool = True  # reuse cached transcript/frames/chunks from scratch dir
    keep_scratch: bool = False
    jobs: int = field(default_factory=lambda: max(1, os.cpu_count() or 1))
    paths_home: Path | None = None  # final .md output dir override
    paths_temp: Path | None = None  # scratch dir override
    video_exts: tuple[str, ...] = DEFAULT_VIDEO_EXTS
    default_speaker: str | None = None
    prompt_path: Path | None = None
    whisper_model: str = "small"
    whisper_device: str = "auto"  # "auto" | "cpu" | "cuda"
    audio_language: str | None = None  # ISO code, e.g. "de"; forces whisper language
    notes: str | None = None           # free-text forwarded to LLM capture metadata
    output_override: Path | None = None  # explicit -o/--output path; single-input only
    frame_width: int = 256             # downscale frames to fit within W x H (aspect preserved)
    frame_height: int = 144
    llm: LLMConfig = field(default_factory=LLMConfig)


def from_env(base: Config | None = None) -> Config:
    cfg = base or Config()
    env = os.environ
    return replace(
        cfg,
        whisper_model=env.get("VIDWIT_WHISPER_MODEL", cfg.whisper_model),
        whisper_device=env.get("VIDWIT_WHISPER_DEVICE", cfg.whisper_device),
        llm=LLMConfig(
            provider=env.get("VIDWIT_LLM_PROVIDER", cfg.llm.provider),
            model=env.get("VIDWIT_LLM_MODEL", cfg.llm.model),
            base_url=env.get("VIDWIT_LLM_BASE_URL", cfg.llm.base_url),
            api_key=env.get(
                "VIDWIT_LLM_API_KEY",
                env.get("ANTHROPIC_API_KEY") or env.get("OPENAI_API_KEY") or cfg.llm.api_key,
            ),
            max_output_tokens=int(
                env.get("VIDWIT_LLM_MAX_OUTPUT_TOKENS", cfg.llm.max_output_tokens)
            ),
        ),
    )


def from_file(path: Path, base: Config | None = None) -> Config:
    """Load TOML config. Unknown keys ignored. Empty/missing file → base unchanged."""
    cfg = base or Config()
    if not path.exists():
        return cfg
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    defaults = data.get("defaults", {}) or {}
    cfg = replace(
        cfg,
        fps=float(defaults.get("fps", cfg.fps)),
        window=float(defaults.get("window", cfg.window)),
        overlap=float(defaults.get("overlap", cfg.overlap)),
        jobs=int(defaults.get("jobs", cfg.jobs)),
        default_speaker=defaults.get("default_speaker", cfg.default_speaker),
        prompt_path=Path(defaults["prompt"]).expanduser() if "prompt" in defaults else cfg.prompt_path,
        frame_width=int(defaults.get("frame_width", cfg.frame_width)),
        frame_height=int(defaults.get("frame_height", cfg.frame_height)),
    )

    whisper = data.get("whisper", {}) or {}
    cfg = replace(
        cfg,
        whisper_model=whisper.get("model", cfg.whisper_model),
        whisper_device=whisper.get("device", cfg.whisper_device),
    )

    video = data.get("video", {}) or {}
    cfg = replace(
        cfg,
        audio_language=video.get("audio_language", cfg.audio_language),
        notes=video.get("notes", cfg.notes),
    )

    llm_data = data.get("llm", {}) or {}
    eb = llm_data.get("extra_body") or {}
    cfg = replace(
        cfg,
        llm=LLMConfig(
            provider=llm_data.get("provider", cfg.llm.provider),
            model=llm_data.get("model", cfg.llm.model),
            base_url=llm_data.get("base_url", cfg.llm.base_url),
            api_key=llm_data.get("api_key", cfg.llm.api_key),
            max_output_tokens=int(llm_data.get("max_output_tokens", cfg.llm.max_output_tokens)),
            request_timeout=float(llm_data.get("request_timeout", cfg.llm.request_timeout)),
            extra_body=dict(eb) if isinstance(eb, dict) else {},
        ),
    )

    paths = data.get("paths", {}) or {}
    cfg = replace(
        cfg,
        paths_home=Path(paths["home"]).expanduser() if "home" in paths else cfg.paths_home,
        paths_temp=Path(paths["temp"]).expanduser() if "temp" in paths else cfg.paths_temp,
    )
    return cfg


def find_config_file() -> Path | None:
    """Search standard locations for a vidwit.toml. First hit wins."""
    cwd = Path.cwd() / CONFIG_FILENAME
    xdg = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    home = xdg / "vidwit" / CONFIG_FILENAME
    for cand in (cwd, home):
        if cand.is_file():
            return cand
    return None
