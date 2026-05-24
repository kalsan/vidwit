from __future__ import annotations

import re
from pathlib import Path


_WARNING_RE = re.compile(r"\[⚠[^\]]*\]")


def assemble(chunks_dir: Path, video_name: str) -> str:
    """Concatenate chunk_*.md into one witness document with TOC + warnings index."""
    files = sorted(chunks_dir.glob("chunk_*.md"))
    if not files:
        raise FileNotFoundError(f"no chunks under {chunks_dir}")

    bodies = [p.read_text(encoding="utf-8").rstrip() for p in files]
    full_body = "\n\n".join(bodies)

    warnings = sorted(set(_WARNING_RE.findall(full_body)))
    toc = _build_toc(bodies)

    header = [f"# vidwit — {video_name}", ""]
    if warnings:
        header += ["## Content warnings", ""]
        header += [f"- {w}" for w in warnings]
        header += [""]
    if toc:
        header += ["## Table of contents", ""]
        header += toc
        header += [""]
    header += ["---", ""]
    return "\n".join(header) + full_body + "\n"


def _build_toc(bodies: list[str]) -> list[str]:
    lines = []
    for b in bodies:
        for ln in b.splitlines():
            m = re.match(r"^###\s+(.*)$", ln)
            if m:
                lines.append(f"- {m.group(1).strip()}")
                break
    return lines
