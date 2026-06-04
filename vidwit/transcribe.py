from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class Word:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(slots=True, frozen=True)
class Transcript:
    words: tuple[Word, ...]
    language: str | None = None
    language_probability: float | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "language": self.language,
                "language_probability": self.language_probability,
                "words": [asdict(w) for w in self.words],
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> "Transcript":
        data = json.loads(payload)
        return cls(
            language=data.get("language"),
            language_probability=data.get("language_probability"),
            words=tuple(Word(**w) for w in data.get("words", [])),
        )

    def slice(self, start: float, end: float) -> list[Word]:
        """Words whose midpoint falls in [start, end)."""
        out = []
        for w in self.words:
            mid = 0.5 * (w.start + w.end)
            if start <= mid < end:
                out.append(w)
        return out


def load_external(
    path: Path,
    default_speaker: str | None = None,
    language: str | None = None,
) -> Transcript:
    """Load a pre-computed transcript JSON instead of running whisper.

    Accepts two shapes:

    * **vidwit-native** — a top-level ``"words"`` list of
      ``{start, end, text[, speaker]}`` (what :meth:`Transcript.to_json`
      emits). Used verbatim.
    * **faster-whisper / openai-whisper** — a ``"segments"`` list of
      ``{start, end, text}`` (e.g. the ``.json`` our process.sh pipeline
      writes). Each segment is expanded into evenly-timed pseudo-words so
      the windowing/slicing downstream behaves as if whisper had produced
      word timestamps.

    ``language`` is a fallback used only when the file has no ``language``
    key; ``default_speaker`` labels words that carry no speaker of their own.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lang = data.get("language", language)
    lang_prob = data.get("language_probability")

    words: list[Word] = []
    raw_words = data.get("words")
    if isinstance(raw_words, list) and raw_words:
        for w in raw_words:
            words.append(
                Word(
                    start=float(w["start"]),
                    end=float(w["end"]),
                    text=(w.get("text") or "").strip(),
                    speaker=w.get("speaker") or default_speaker,
                )
            )
    else:
        for seg in data.get("segments", []):
            words.extend(_segment_to_words(seg, default_speaker))

    return Transcript(
        words=tuple(words),
        language=lang,
        language_probability=float(lang_prob) if lang_prob is not None else None,
    )


def _segment_to_words(seg: dict, default_speaker: str | None) -> list[Word]:
    """Split a whisper segment into evenly-timed word tokens.

    We have no per-word timing in a segment-level transcript, so spread the
    tokens linearly across the segment's [start, end) span. This keeps the
    downstream window-slicing (by word midpoint) well-aligned with the frames.
    """
    start = float(seg["start"])
    end = float(seg["end"])
    toks = (seg.get("text") or "").split()
    if not toks:
        return []
    span = max(0.0, end - start)
    n = len(toks)
    out: list[Word] = []
    for i, tok in enumerate(toks):
        out.append(
            Word(
                start=start + span * (i / n),
                end=start + span * ((i + 1) / n),
                text=tok,
                speaker=default_speaker,
            )
        )
    return out


def transcribe(
    audio_path: Path,
    model_name: str = "small",
    device: str = "auto",
    default_speaker: str | None = None,
    language: str | None = None,
) -> Transcript:
    """Word-level transcript via faster-whisper.

    `language` is an optional ISO code (e.g. "de"). When None, whisper
    auto-detects. Passing a hint forces decoding in that language —
    useful when auto-detect picks the wrong one.
    """
    from faster_whisper import WhisperModel

    if device == "auto":
        device = "cuda" if _cuda_ok() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path), word_timestamps=True, language=language,
    )

    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            words.append(
                Word(
                    start=float(w.start),
                    end=float(w.end),
                    text=w.word.strip(),
                    speaker=default_speaker,
                )
            )
    return Transcript(
        words=tuple(words),
        language=info.language,
        language_probability=float(info.language_probability),
    )


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
