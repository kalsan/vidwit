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

    def to_json(self) -> str:
        return json.dumps(
            {
                "language": self.language,
                "words": [asdict(w) for w in self.words],
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> "Transcript":
        data = json.loads(payload)
        return cls(
            language=data.get("language"),
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


def transcribe(
    audio_path: Path,
    model_name: str = "small",
    device: str = "auto",
    default_speaker: str | None = None,
) -> Transcript:
    """Word-level transcript via faster-whisper."""
    from faster_whisper import WhisperModel

    if device == "auto":
        device = "cuda" if _cuda_ok() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), word_timestamps=True)

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
    return Transcript(words=tuple(words), language=info.language)


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
