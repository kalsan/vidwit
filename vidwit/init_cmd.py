"""`vidwit init` — drop the bundled sample config (and optional prompt)
into the current directory or `~/.config/vidwit/` so a fresh
`pip install vidwit` user can start configuring without cloning the repo.
"""
from __future__ import annotations

import argparse
import os
import sys
from importlib.resources import files
from pathlib import Path


def run(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="vidwit init",
        description="Write a starter vidwit.toml (and optionally a prompt template) "
                    "to the current directory or ~/.config/vidwit/.",
    )
    p.add_argument(
        "--user", action="store_true",
        help="write to ~/.config/vidwit/ instead of the current directory",
    )
    p.add_argument(
        "--prompt", action="store_true",
        help="also drop vidwit_prompt.md alongside vidwit.toml",
    )
    p.add_argument(
        "--force", "-f", action="store_true",
        help="overwrite existing files",
    )
    args = p.parse_args(argv)

    if args.user:
        xdg = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
        target_dir = xdg / "vidwit"
    else:
        target_dir = Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    skipped: list[Path] = []

    items = [("vidwit.toml", "vidwit.toml")]
    if args.prompt:
        items.append(("vidwit_prompt.md", "vidwit_prompt.md"))

    templates = files("vidwit.templates")
    for src_name, dst_name in items:
        dst = target_dir / dst_name
        if dst.exists() and not args.force:
            skipped.append(dst)
            continue
        dst.write_text(templates.joinpath(src_name).read_text(encoding="utf-8"),
                       encoding="utf-8")
        written.append(dst)

    for f in written:
        print(f"wrote {f}")
    for f in skipped:
        print(f"exists, skipped {f} (use --force to overwrite)", file=sys.stderr)

    if skipped and not written:
        return 1

    if written:
        toml = next((f for f in written if f.name == "vidwit.toml"), None)
        if toml is not None:
            print()
            print(f"Next: edit {toml} and set [llm] api_key.")
    return 0
