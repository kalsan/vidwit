from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__, config as cfg_mod, pipeline


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _make_config(args)

    inputs = _expand_inputs(args.inputs, cfg.video_exts)
    if not inputs:
        print("vidwit: no video inputs", file=sys.stderr)
        return 2

    rc = 0
    for v in inputs:
        try:
            pipeline.run_one(v, cfg)
        except FileExistsError as e:
            print(f"vidwit: {e}", file=sys.stderr)
            rc = 1
        except Exception as e:
            logging.exception("failed: %s", v)
            print(f"vidwit: error on {v}: {e}", file=sys.stderr)
            rc = 1
    return rc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vidwit",
        description="Multimodal video witness — emit exhaustive markdown next to each video.",
    )
    p.add_argument("inputs", nargs="+", help="video file(s) or directory(ies)")
    p.add_argument("--config", type=Path, default=None,
                   help="path to vidwit.toml (default: ./vidwit.toml or ~/.config/vidwit/vidwit.toml)")
    p.add_argument("--fps", type=float, default=None, help="frame sampling rate (default 1.0)")
    p.add_argument("--window", type=float, default=None, help="window length seconds (default 10.0)")
    p.add_argument("--overlap", type=float, default=None, help="window overlap seconds (default 1.0)")
    p.add_argument("--overwrite", action="store_true", help="replace existing .md")
    p.add_argument("--resume", action="store_true", help="resume partial run from scratch dir")
    p.add_argument("--keep-scratch", action="store_true", help="don't delete scratch dir on success")
    p.add_argument("--jobs", type=int, default=None, help="threads for ffmpeg/whisper (default nproc)")
    p.add_argument(
        "--paths",
        action="append",
        default=[],
        metavar="KEY:VAL",
        help="yt-dlp style overrides: home:<dir> for final .md, temp:<dir> for scratch. Repeatable.",
    )
    p.add_argument("--ext", action="append", default=[], help="extra video extension (repeatable)")
    p.add_argument("--default-speaker", default=None, help="label transcribed words with this name")
    p.add_argument("--prompt", type=Path, default=None, help="path to system prompt markdown")
    p.add_argument("--audio-language", default=None,
                   help="ISO language hint for whisper (e.g. 'de'); skips auto-detect")
    p.add_argument("--notes", default=None,
                   help="free-text context forwarded to the LLM in every chunk")

    llm = p.add_argument_group("LLM")
    llm.add_argument("--llm", dest="llm_provider", default=None,
                     choices=["dummy", "anthropic", "openai", "lmstudio"])
    llm.add_argument("--model", dest="llm_model", default=None)
    llm.add_argument("--base-url", dest="llm_base_url", default=None,
                     help="OpenAI-compatible endpoint (e.g. http://localhost:1234/v1)")

    p.add_argument("--whisper-model", default=None,
                   help="whisper model name (tiny, base, small, medium, large-v3)")
    p.add_argument("--whisper-device", default=None, choices=["auto", "cpu", "cuda"])

    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"vidwit {__version__}")
    return p


def _make_config(args: argparse.Namespace) -> cfg_mod.Config:
    cfg = cfg_mod.Config()
    config_path = args.config or cfg_mod.find_config_file()
    if config_path is not None:
        if args.config is not None and not config_path.exists():
            raise SystemExit(f"vidwit: --config {config_path} not found")
        cfg = cfg_mod.from_file(config_path, cfg)
        logging.getLogger("vidwit").info("loaded config: %s", config_path)
    cfg = cfg_mod.from_env(cfg)

    if args.fps is not None: cfg = replace(cfg, fps=args.fps)
    if args.window is not None: cfg = replace(cfg, window=args.window)
    if args.overlap is not None: cfg = replace(cfg, overlap=args.overlap)
    if args.overwrite: cfg = replace(cfg, overwrite=True)
    if args.resume: cfg = replace(cfg, resume=True)
    if args.keep_scratch: cfg = replace(cfg, keep_scratch=True)
    if args.jobs is not None: cfg = replace(cfg, jobs=args.jobs)
    if args.default_speaker: cfg = replace(cfg, default_speaker=args.default_speaker)
    if args.prompt: cfg = replace(cfg, prompt_path=args.prompt)
    if args.audio_language: cfg = replace(cfg, audio_language=args.audio_language)
    if args.notes: cfg = replace(cfg, notes=args.notes)
    if args.whisper_model: cfg = replace(cfg, whisper_model=args.whisper_model)
    if args.whisper_device: cfg = replace(cfg, whisper_device=args.whisper_device)

    if args.ext:
        exts = tuple({*cfg.video_exts, *(_normalise_ext(e) for e in args.ext)})
        cfg = replace(cfg, video_exts=exts)

    home, temp = _parse_paths(args.paths)
    if home: cfg = replace(cfg, paths_home=home)
    if temp: cfg = replace(cfg, paths_temp=temp)

    llm = cfg.llm
    if args.llm_provider: llm = replace(llm, provider=args.llm_provider)
    if args.llm_model: llm = replace(llm, model=args.llm_model)
    if args.llm_base_url: llm = replace(llm, base_url=args.llm_base_url)
    cfg = replace(cfg, llm=llm)

    return cfg


def _parse_paths(items: list[str]) -> tuple[Path | None, Path | None]:
    home = temp = None
    for raw in items:
        if ":" not in raw:
            raise SystemExit(f"--paths expects KEY:VAL, got {raw!r}")
        key, val = raw.split(":", 1)
        key = key.strip().lower()
        path = Path(val).expanduser()
        if key == "home":
            home = path
        elif key == "temp":
            temp = path
        else:
            raise SystemExit(f"--paths unknown key {key!r} (want home or temp)")
    return home, temp


def _normalise_ext(ext: str) -> str:
    ext = ext.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"


def _expand_inputs(args: list[str], exts: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    ext_set = {e.lower() for e in exts}
    for raw in args:
        p = Path(raw).expanduser()
        if not p.exists():
            print(f"vidwit: not found: {p}", file=sys.stderr)
            continue
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in ext_set:
                    out.append(f)
        elif p.is_file():
            out.append(p)
    # Dedupe, keep order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in out:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(f)
    return unique


if __name__ == "__main__":
    raise SystemExit(main())
