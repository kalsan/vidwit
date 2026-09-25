# Changelog

All notable changes to vidwit are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-09-25

### Added
- `--skip-identical-frames` flag (`skip_identical_frames` under `[defaults]`):
  keep a frame only if it differs from the last kept one, so `--fps` becomes a
  maximum rate reached only during continuous movement. Aimed at screen
  recordings, where long still stretches otherwise cost tokens for identical
  frames. Detection uses ffmpeg's `mpdecimate` on the downscaled frames;
  `--mpdecimate OPTS` tunes it. Each frame sent to the LLM is labelled with its
  timestamp, the capture metadata tells the model that gaps are still pictures,
  and a window without a change at its start also receives the last kept frame
  before it. Requires ffmpeg ≥ 5.1.

### Changed
- Resume now re-extracts cached frames when fps, frame size or the
  identical-frame settings differ from the cached run.

## [1.1.0] - 2026-06-04

### Added
- `--transcript FILE` flag: use a pre-computed transcript JSON instead of
  running whisper. Accepts both the vidwit-native `{words:[...]}` shape and the
  segment-level faster-whisper / openai-whisper `{segments:[...]}` shape; each
  segment is expanded into evenly-timed pseudo-words so downstream
  window-slicing stays aligned. Single-input only — rejected with multiple
  inputs, since one transcript cannot describe a batch.

## [1.0.0] - 2026-05-26

First public release.

### Added
- Core pipeline: iterate fixed-length time windows over a video, sampling
  frames and whisper word-level transcript per window, and assemble the
  per-window LLM results into one Markdown document with verbatim transcript
  citations and speaker labels.
- LLM providers: `anthropic`, `openai`, `lmstudio`, `dummy`.
- LLM tuning: `--timeout`, `--extra-body` (repeatable, merged into the
  chat-completions payload), `--max-tokens` (cumulative per-video token cap;
  abort and assemble what is done).
- Prompt safeguards and output sanitizing tuned for less capable local models,
  including handling of unreliable transcripts and burned-in subtitles
  (`[⚠ transcript unreliable]`).
- Frame autoscaling (`--frame-width`, `--frame-height`) to optimize for vision
  LLMs.
- Resume by default: cached transcript, frames, and chunks reused from scratch;
  `--no-resume` to force re-run.
- Path overrides (`--paths home:DIR`, `--paths temp:DIR`), `-o/--output`,
  multi-input and directory recursion, custom `--prompt`, `--default-speaker`,
  `--audio-language`, `--notes`.
- CI and packaging for publishing to PyPI.

[1.2.0]: https://github.com/kalsan/vidwit/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/kalsan/vidwit/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/kalsan/vidwit/releases/tag/v1.0.0
