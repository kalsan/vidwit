from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Window:
    index: int
    start: float  # seconds, inclusive
    end: float    # seconds, half-open

    @property
    def label(self) -> str:
        return f"chunk_{self.index:04d}"


def windows(duration_s: float, window_s: float, overlap_s: float) -> list[Window]:
    """Half-open windows [start, end) covering duration. Stride = window - overlap.

    Adjacent windows overlap by overlap_s. Final window is clamped to duration.
    """
    if window_s <= 0:
        raise ValueError("window_s must be > 0")
    if not 0 <= overlap_s < window_s:
        raise ValueError("overlap_s must satisfy 0 <= overlap < window")
    stride = window_s - overlap_s
    out: list[Window] = []
    i = 0
    start = 0.0
    while start < duration_s:
        end = min(start + window_s, duration_s)
        out.append(Window(index=i, start=start, end=end))
        if end >= duration_s:
            break
        start += stride
        i += 1
    return out
