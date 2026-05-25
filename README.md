# vidwit

vidwit is a multimodal video witness. It reads a video and writes an
exhaustive Markdown record that combines verbatim transcript citations
with detailed visual descriptions, so that a reader of the Markdown
file knows everything a viewer of the video would have known.

This is mainly intended as a tool that allows LLMs to "watch" a video.
The idea is that you run vidwit first and feed the output to the LLM,
after which it should have most relevant information found in the video.

## What you get

For any input video, vidwit produces a single Markdown document that:

- tells the entire story of the video, in order, as a continuous
  narration;
- embeds transcript citations verbatim, with speaker labels when they
  are known;
- annotates each citation and scene with a description of what is
  visually shown at that moment, including subjects, actions, spatial
  relations, environment, and any on-screen text quoted verbatim;
- tags every block with a half-open timecode range
  `[MM:SS.mmm – MM:SS.mmm)` so that a reader can locate the
  corresponding slice in the source video;
- collects content warnings into an index at the top of the document,
  so a reader can decide which sections to skip — for instance graphic
  violence or NSFW imagery — without watching the video first.

The reader test for the output is straightforward: *if I read this
Markdown file, I should know every meaningful fact that a viewer of
the video would have learned.*

## Example output

```
### [02:14.300 – 02:31.800) — Tiger and cub at the riverbank [FOOTAGE]

> "The tiger has a young cub." — Narrator, 02:18.100

Visual: A large adult tiger sits on the left bank of a shallow river.
A small cub (about a quarter of the adult's size) sits to the tiger's
right and licks its own front paw. Background: dense green forest,
late afternoon light.

### [04:02.000 – 04:21.500) — Tiger hunts gazelle [FOOTAGE] [⚠ graphic]

> "Once it spots its prey…" — Narrator, 04:03.400

Visual: The adult tiger sprints across the savannah and takes down a
gazelle. A close-up shows the bite to the gazelle's neck; blood is
visible from 04:18. Skip 04:17 – 04:21 to avoid the blood.
```

## Installation

vidwit requires Python 3.11 or newer, along with `ffmpeg` and
`ffprobe` on the system `PATH`. Once those prerequisites are in place,
installation is a standard editable install:

```bash
git clone <repo-url> vidwit
cd vidwit
python3 -m venv .venv
.venv/bin/pip install -e .
```

This pulls `faster-whisper`, `ctranslate2`, and a few small supporting
libraries. PyTorch is not required, because `faster-whisper` runs
inference through `ctranslate2`. The Anthropic and OpenAI Python SDKs
are also not required, because vidwit uses the standard library's
`urllib` to call the HTTP endpoints directly.

The first run downloads the chosen whisper model into
`~/.cache/huggingface/hub/`. The default model is `small`, which is
roughly 480 MB on disk. For a smaller first download, choose `tiny`
(about 75 MB) or `base` (about 140 MB) via the `--whisper-model` flag
or the configuration file.

## Configuration

Copy the sample configuration file and paste in your LLM API key:

```bash
cp config/vidwit.toml.sample vidwit.toml
$EDITOR vidwit.toml
```

vidwit searches for its configuration file at `./vidwit.toml`, then at
`~/.config/vidwit/vidwit.toml`, or at an explicit path passed via
`--config <path>`. Settings are merged with the following precedence,
where higher entries win over lower ones: command-line flags,
environment variables, configuration file, and the built-in defaults.

Four LLM providers are supported out of the box:

| Provider    | Description                                                     |
|-------------|-----------------------------------------------------------------|
| `anthropic` | Anthropic Claude (Sonnet 4.6 by default, Opus 4.7 for highest fidelity). |
| `openai`    | OpenAI Chat Completions API.                                    |
| `lmstudio`  | Any OpenAI-compatible endpoint, such as LM Studio or vLLM.      |
| `dummy`     | A no-network placeholder useful for offline testing; it emits stub chunks instead of calling a real model. |

The real `vidwit.toml` file is excluded from version control via
`.gitignore`; only the `.sample` is committed.

## Usage

```bash
vidwit path/to/video.mp4
vidwit path/to/video1.mp4 path/to/video2.mkv
vidwit path/to/dir/    # recurses, picks up files by extension
```

For each input video `foo.mp4`, vidwit writes `foo.md` next to it in
the same directory and under the same basename. The default set of
video extensions is `.mp4`, `.mkv`, `.mov`, `.webm`, and `.avi`. You
can add further extensions with the `--ext` flag.

The most commonly used command-line flags are:

```
--fps FLOAT          frame sampling rate (default 1.0)
--window FLOAT       window length in seconds (default 10.0)
--overlap FLOAT      window overlap in seconds (default 1.0)
--overwrite          replace an existing .md output file
--no-resume          force re-run; ignore cached transcript/frames/chunks in scratch
--keep-scratch       keep the scratch directory after a successful run
--jobs N             threads for ffmpeg and whisper (default: nproc)
--paths home:DIR     write the final .md elsewhere (yt-dlp-style override)
--paths temp:DIR     place the scratch directory elsewhere
--default-speaker S  label transcribed words with the speaker name S
--prompt FILE        use a custom system prompt
--audio-language CODE   ISO language hint for whisper (e.g. "de"); skips auto-detect
--notes "TEXT"       free-text context forwarded to the LLM in every chunk
-o, --output PATH    explicit output file path (single-input only); relative or absolute
--frame-width N      downscale frames to fit within this width (default 256)
--frame-height N     downscale frames to fit within this height (default 144)
--llm PROVIDER       anthropic | openai | lmstudio | dummy
--model NAME         model identifier
--base-url URL       OpenAI-compatible endpoint URL
--no-summary         disable the rolling "story so far" loop (on by default)
--summary-llm PROV   override the LLM used for the rolling summary (defaults to primary)
--summary-model NAME
--summary-base-url URL
--whisper-model NAME tiny / base / small / medium / large-v3
```

## How it works

Because vision LLMs cannot see an arbitrarily long video in a single
call, vidwit iterates over fixed-length time windows and assembles the
results into one document. The core loop, written as pseudocode, is:

```
for window in windows(video, length=10s, overlap=1s):
    frames = ffmpeg frames within [window.start, window.end)
    words  = whisper word-level transcript within window
    chunk  = LLM(
        system  = vidwit_prompt,
        context = rolling summary + raw tail of recent chunks,
        input   = frames + words + window timecode + capture metadata,
    )
    write chunk to disk (resumable)
assemble chunks into final markdown (TOC + content warnings index)
write `<video>.md.part` and atomically rename it to `<video>.md`
```

The combination of fixed windows and a rolling context window is what
lets vidwit work on inputs that would never fit in a single LLM call.

### Why a rolling context is necessary

The rolling context allows the model to maintain continuity across
windows. Without it, the model would not remember introductions or
recurring entities. In particular:

- A narrator who refers to "the cub" several minutes after introducing
  it must still resolve correctly to the original animal.
- Re-appearing characters should be given the same name in every
  window where they appear.
- The model should not restate background information that has already
  been established in earlier windows.

vidwit ships with two complementary mechanisms for rolling context:

1. **Sliding tail**: the verbatim markdown of the last couple of
   chunks is prepended to each new chunk's prompt. Always on. Cheap
   and lossless for the immediately preceding context.
2. **Rolling summary**: a "story so far" string is maintained across
   all chunks and updated after each one. **On by default**, using
   the primary LLM for the summary call. Disable with `--no-summary`.
   You can also point the summary at a different (typically smaller
   and cheaper) text-only model by setting `[llm.summary]` in
   `vidwit.toml` or by passing `--summary-llm`, `--summary-model`, or
   `--summary-base-url` on the command line, which keeps the main
   vision LLM unburdened by the summary call.

The latest summary is persisted to `state.json` in the scratch
directory so that resumed runs continue with the same continuity
state.

### Capture metadata sent to the LLM

Every LLM call carries a `# Capture metadata` block that lists the
frame sampling rate, the window length, the overlap with neighbouring
windows, the source resolution, the whisper model used for
transcription, the total duration of the video, the language whisper
detected (with its confidence), and any user-supplied audio-language
hint or free-form notes. The model uses this metadata to calibrate
its description. In particular, visual events shorter than the frame
spacing may be missed in the supplied frames, and the model is
expected to infer those from the transcript instead.

### Unreliable transcripts and burned-in subtitles

Whisper is excellent for the languages it was trained on but can
produce gibberish for regional dialects or low-resource languages
(Swiss German is a common example). When the transcript reads as
gibberish, the affected timecodes are tagged with
`[⚠ transcript unreliable]` and the narration quote is skipped.

Burned-in subtitles are not promoted into the audio stream — they
are visual content. They stay in the visual description with a
source attribution, the same way any other on-screen text would
appear:

```
Visual: a man is speaking to camera; subtitle reads: "We are here."
```

Two flags help with these situations and can also be set under the
`[video]` table in `vidwit.toml`:

- `--audio-language CODE` forces whisper to decode in a specific
  language instead of auto-detecting; useful when auto-detect picks
  the wrong language for non-English speech.
- `--notes "TEXT"` accepts arbitrary free text — dialect, situation,
  burned-in subtitle presence — and forwards it verbatim to the LLM
  in every chunk's capture metadata.

### Resumability

Intermediate per-window outputs are stored under
`<video-dir>/.vidwit-tmp/<video-hash>/chunks/NNNN.md`. The whisper
transcript and the extracted frames are cached in the same scratch
directory. The final output is first written as `<video>.md.part`
and then atomically renamed to `<video>.md`.

**Resume is on by default.** If a run crashes, re-running the same
command picks up where it left off — cached transcript, frames, and
finished chunks are reused. To force a clean re-run from scratch, pass
`--no-resume`.

Scratch directories are cleaned up on success unless `--keep-scratch`
is passed.

Caveats:

- The scratch directory is keyed by video hash alone, not by model or
  by sampling settings. If you change `--fps`, `--window`,
  `--frame-width`, `--audio-language`, or the LLM model between runs,
  cached chunks and frames are no longer valid. Pass `--no-resume`
  for that run, or wipe the scratch directory first.

### Scratch and output paths

vidwit follows the yt-dlp convention for redirecting scratch and
output locations:

```
--paths home:/out          # write the final .md into /out
--paths temp:/fast-ssd     # place the scratch dir on /fast-ssd
```

By default, both the final output and the scratch directory live next
to the input video file.

## Project layout

```
vidwit/
  pyproject.toml
  vidwit/
    __init__.py
    __main__.py             # `python -m vidwit`
    cli.py                  # argument parsing, input expansion, per-file loop
    config.py               # defaults + environment variables + TOML file merge
    pipeline.py             # per-video orchestration
    ffmpeg_io.py            # probe, audio extraction, frame extraction
    transcribe.py           # faster-whisper wrapper with word-level timestamps
    chunker.py              # window iteration with overlap
    llm.py                  # Provider protocol + Anthropic + OpenAI-compatible + dummy
    assembler.py            # chunk merge, table of contents, content warnings
    scratch.py              # paths, hashing, resume support
  config/
    vidwit.toml.sample
    vidwit_prompt.md.sample
```

## Output format conventions

The system prompt enforces a strict block layout so that the output is
both readable and machine-parseable:

- Each block begins with a header of the form
  `### [MM:SS.mmm – MM:SS.mmm) — short title [TAG]`.
- Intervals are half-open (`[start, end)`) so that adjacent blocks
  never overlap.
- The tag indicates the kind of content in the block:
  - `[FOOTAGE]` for live-action camera shots,
  - `[ANIM]` for animation, infographics or title cards,
  - `[FOOTAGE + ANIM]` for live footage with overlaid text or graphics,
  - `[⚠ <reason>]` for content warnings, optionally followed by a
    recommended skip range underneath.
- On-screen text is quoted verbatim, with capitalisation preserved.
- Speech is rendered in `>` blockquotes, with a speaker label and a
  word-precise timecode taken from whisper.
- Non-speech audio is bracketed: `[lion roars]`, `[applause]`,
  `[wind]`.
- When speech crosses a window boundary, the quote ends with `…` and
  continues in the next block, starting with `…`.

## Roadmap

### v1 (current)

- Local-first CLI with no service and no database.
- ffmpeg, faster-whisper, and a vision LLM (Anthropic or any
  OpenAI-compatible endpoint).
- Iterative chunked loop with rolling context.
- Resumable runs; yt-dlp-style scratch and output paths.
- A cost ceiling is not yet implemented; when it is added it will most
  likely be token-based via a `--max-tokens` flag, so that it is
  provider-agnostic.

### v2 — speaker awareness (if possible)

- Diarisation through `pyannote-audio`, merged with whisper's
  word-level timings.
- Face detection and re-identification through `insightface`. Faces
  are clustered across the video so that the same person receives a
  stable identifier throughout the record.

### v2+ — cross-video corpus state

The goal of corpus state is that when vidwit processes video B, it
already knows what happened in video A and can link, deduplicate, and
reuse identities.

```bash
vidwit --corpus ./my-docs/ ./my-docs/ep01.mp4
vidwit --corpus ./my-docs/ ./my-docs/         # process the whole directory
```

Per-corpus state lives in `<corpus_dir>/.vidwit/`:

```
.vidwit/
  entities.json   # stable id → {names, aliases, descriptors, embeddings}
  index.json      # video path → {hash, mtime, md_path, processed_at}
  faces/          # representative crops per face id
  voices/         # speaker embeddings per voice id
```

Concrete benefits of corpus state include:

- **Recurring entities.** The same person, place, product, or animal
  appearing in multiple files gets the same stable name. The record
  for video B can then say "same speaker as in
  `2025-04-12_interview.md` (Anna)".
- **Series continuity.** Episode 3 of a documentary can be analysed
  with episodes 1 and 2 in mind ("picks up the river-bank tiger from
  episode 1, 11:02").
- **Skip-link reuse.** If a content warning was justified in video A,
  the same scene reused in a recap in video B can reuse the warning.

The `--corpus` flag is opt-in. Without it, vidwit is stateless on a
per-video basis: identical input yields identical output, with no
hidden surprises.

The following are out of scope by design:

- Bi-directional editing. Renaming an entity does **not** rewrite past
  Markdown files; past Markdown is a frozen artefact.
- Cross-corpus federation. One corpus directory is one world.

## Notice of copyright issues with produced output files

vidwit produces verbatim transcripts and descriptive derivatives of
its input videos. The legality of using those outputs is your
responsibility — please make sure you have the rights to the inputs
you feed into the tool.

## License

See [LICENSE](LICENSE).
