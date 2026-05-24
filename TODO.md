# TODO — vidwit

Multimodal video witness: read a video, write an exhaustive markdown
record that combines transcript citations with visual descriptions, so a
reader of the markdown knows everything a viewer of the video would
have known.

Name = **vid** + **wit(ness)**. Reader experience: the tool acts as a
witness who watched the video for you and now tells you what happened.

Sibling-in-spirit to `ai_video_analyzer` but a different tool:
local-first CLI, no download, no service. Reuses ideas (ffmpeg + whisper
+ vision LLM) but evolves independently.

## Invocation model

Local CLI. No HTTP server, no SQLite, no download step. Operates on
files already on disk.

```bash
vidwit path/to/video.mp4
vidwit path/to/video1.mp4 path/to/video2.mkv
vidwit path/to/dir/         # recurses, picks up video files by extension
```

For each input video `foo.mp4`, the tool writes `foo.md` next to it
(same directory, same basename). If `foo.md` already exists, fail by
default; `--overwrite` to replace, `--resume` to continue a partial run.

Allowed extensions configurable, default e.g. `.mp4 .mkv .mov .webm .avi`.
Directory inputs recurse depth-first; non-video files skipped silently.

## Goal

For any input video, emit a single large markdown document that:

- Tells the **entire story** of the video, in order, as a narration.
- Embeds **transcript citations** verbatim (with speaker, if known).
- Annotates each citation / scene with what is **visually shown** at that
  moment (subjects, actions, spatial relations, environment).
- Tags **every** statement with a **timecode range** `[MM:SS.mmm – MM:SS.mmm]`
  so a reader can locate the exact slice in the source video.
- Lets a reader **decide which parts to skip** (e.g. blood, violence,
  NSFW imagery) without watching first.

Reader test: "If I read this markdown, I should know every meaningful
fact a viewer of the video would have learned."

## Example output shape

```
### 02:14.300 – 02:31.800 — Tiger and cub at the riverbank

> "The tiger has a young cub." — Narrator, 02:18.100

Visual: A large adult tiger sits on the left bank of a shallow river.
A small cub (≈ 1/4 the adult's size) sits to the tiger's right and licks
its own front paw. Background: dense green forest, late afternoon light.

### 04:02.000 – 04:21.500 — Tiger hunts gazelle — ⚠ graphic

> "Once it spots its prey..." — Narrator, 04:03.400

Visual: Adult tiger sprints across savannah, takes down a gazelle.
Close-up of bite to the gazelle's neck; blood visible from 04:18.
**Skip 04:17 – 04:21 to avoid blood.**
```

## How it differs from `ai_video_analyzer`

| Aspect | `ai_video_analyzer` | `vidwit` |
|---|---|---|
| Surface | HTTP service, Docker | Local CLI, `pip install -e .` |
| Input | YouTube URL (yt-dlp) | Local file path or directory |
| Output | Short structured summary | Exhaustive narrative, full coverage |
| Frame rate | 1 frame / 5 s, subsampled | **1 fps default, configurable via `--fps`** |
| LLM call | Single shot, whole video | **Iterative**, chunked, rolling context |
| Length | Fits in context window | Will not fit — stream-built |
| Speaker labels | None | v2: diarization + face/voice ID |
| Output target | SQLite blob | `foo.md` next to `foo.mp4` |

## Pipeline (proposed)

1. **Resolve inputs**: expand directory args, filter by extension,
   produce a flat list of video paths.
2. **Per video, in a scratch dir**:
   1. **Transcript** (whisper) with **word-level timestamps**
      (`--word_timestamps True`) so citations can be precise to the word.
   2. **Frames** (ffmpeg) at `--fps` (default 1).
   3. *(Optional, v2)* **Speaker diarization** — `pyannote-audio`,
      merged with whisper word timings.
   4. *(Optional, v2)* **Face detection / re-id** — `insightface`,
      cluster faces across the video, assign stable IDs.
   5. **Chunked analysis loop** — see below.
   6. **Assemble** chunks into one markdown file (TOC + content warnings
      index at the top).
   7. Write `foo.md` next to `foo.mp4`. Wipe the scratch dir.
3. *(Optional)* **Update cross-video corpus state** — see below.

## Iterative analysis loop

Model cannot see the whole video at once. Iterate over fixed-length
**windows** (initial guess: 10 s, configurable):

```
for window in windows(video, length=10s):
    frames = ffmpeg frames within [window.start, window.end]
    words  = whisper word-level transcript within window
    chunk_md = LLM(
        system  = vidwit_prompt,
        context = compressed_summary_of_all_previous_chunks
                  + last N raw chunks (sliding tail),
        input   = frames + words + window timecode,
    )
    append chunk_md to output
    update compressed_summary (rolling)
```

Context strategy candidates (pick one — see open questions):

- **Sliding tail**: last N seconds of chunk output passed verbatim.
- **Rolling summary**: maintained "story so far", rewritten each step
  (cheap, lossy).
- **Hybrid** *(default guess)*: rolling summary + last 1–2 raw chunks.

Required so that:

- Narrator referring to "the cub" 5 minutes after introducing it still
  resolves correctly.
- Re-appearing characters get the same name.
- Model does not restate background info already established.

## Project structure

```
vidwit/
  pyproject.toml          # installable as `vidwit` command
  requirements.txt
  README.md
  vidwit/
    __init__.py
    cli.py                # arg parsing, input expansion, per-file loop
    config.py             # env + CLI defaults
    pipeline.py           # frames + transcript + chunked loop, per file
    chunker.py            # window iteration + context management
    assembler.py          # merge chunks → final markdown
    llm.py                # vision LLM client (Claude / GPT / Gemini)
    corpus.py             # OPTIONAL cross-video state — see below
    # diarize.py          # v2
    # faces.py            # v2
  config/
    vidwit_prompt.md.sample
```

Distribute as `pip install -e .` for now. Docker image can come later;
heavy deps (whisper torch) make a local venv nicer during dev.

## Cross-video referencing (optional, v2+)

Goal: when processing video B, the tool already knows what happened in
video A (and earlier videos) and can link / dedupe / reuse identities.

Concrete value:

- **Recurring entities**: same person, place, product or animal across
  multiple files gets the same stable name. The record for B can say
  "Same speaker as in `2025-04-12_interview.md` (Anna)".
- **Series continuity**: episode 3 of a documentary can be analysed
  with episodes 1 and 2 in mind ("picks up the river-bank tiger from
  ep 1, 11:02").
- **Skip-link reuse**: if a content warning was justified in video A,
  the same scene reused in a recap in B can reuse the warning.

### Sketch — corpus state file

Per-corpus state lives in `<corpus_dir>/.vidwit/`:

```
.vidwit/
  entities.json   # stable id → {names, aliases, descriptors, embeddings?}
  index.json      # video path → {hash, mtime, md_path, processed_at}
  faces/          # v2: representative crops per face id
  voices/         # v2: speaker embeddings per voice id
```

CLI:

```bash
vidwit --corpus ./my-docs/ ./my-docs/ep01.mp4
vidwit --corpus ./my-docs/ ./my-docs/         # process whole dir
```

`--corpus` is opt-in. Without it the tool is **stateless per video** —
identical input, identical output, no surprises.

### Open mechanism questions

- How are entities introduced? Probably: per-chunk LLM emits candidate
  entity refs (`<entity id="...">name</entity>`-ish); a second pass
  deduplicates against `entities.json` (LLM-mediated, or embedding
  similarity).
- Cross-video face/voice identity should be **fingerprint-based**
  (embeddings), not LLM-mediated. Cheap and reliable. Depends on v2
  diarization / face features.
- Do we re-process older videos when an entity is renamed? No — `.md`
  files are append-only artefacts; the corpus state is the authority for
  the next run.

### Out of scope (for now)

- Bi-directional editing: changing an entity name should **not**
  rewrite past `.md` files. Past markdown is a frozen artefact.
- Cross-corpus federation. One corpus dir = one world.

## Open questions (decide before coding)

- ~~**Frame rate**~~ — **Decided**: 1 fps default, override via `--fps` CLI flag.
- **Window length**: 10 s? 15 s? Depends on model context + visual rate.
- **Which LLM**: need a vision model with **large context** and good
  image-grounding. Candidates: Claude Opus / Sonnet 4.x (1M ctx),
  GPT-4.1, Gemini 2.5 Pro. Local (LM Studio) likely too weak for the
  fidelity we want — confirm.
- **How much overlap** between adjacent windows (so we don't cut a
  sentence / a visual event in half)? 1 s overlap?
- ~~**Speaker pipeline**~~ — **Decided**: skip in v1. Add pyannote +
  insightface in v2 once core loop works.
- ~~**Storage**~~ — **Decided**: write `foo.md` next to `foo.mp4`. No
  database. Optional corpus state in `<corpus>/.vidwit/`.
- **Resumability**: if the loop crashes at chunk 47/120, can we resume?
  Persist intermediate chunks under a `.vidwit-tmp/<video-hash>/` dir
  so `--resume` skips already-done chunks.
- **Cost ceiling**: a 30-min video at 1 fps = 1800 frames; chunked into
  10 s windows = 180 LLM calls. Budget per video?
- **Scratch dir location**: `$TMPDIR` (default) vs sibling
  `.vidwit-tmp/`. Sibling makes `--resume` discoverable; tmp makes
  clean-up automatic.
- **Concurrency**: multiple input files — process sequentially or in
  parallel? Whisper + LLM are I/O- and GPU-bound respectively, so naive
  parallelism is risky. Default sequential, expose `--jobs N` later.

## Test video — decided category

**Short nature documentary clip, 2–4 min, single narrator.** Matches
the tiger/cub example. Tests core audio↔visual correlation without
dragging in diarization. Concrete video file still to pick (must be
local on Sandro's laptop).

## Milestones

1. Pick local test video file.
2. Hand-write the witness record for the first 30 s — this is the
   spec.
3. Scaffold `vidwit` repo as a local CLI. Bare loop: transcript +
   frames + dummy chunker, writes `foo.md` next to input.
4. Add word-level whisper timestamps; verify alignment.
5. Build real chunked loop (window + rolling context) without
   speaker/face features. Compare to hand spec.
6. Tune frame rate, window length, context strategy.
7. Resumability — intermediate chunk persistence + `--resume`.
8. **v2**: diarization, face re-id.
9. **v2+**: cross-video corpus state (`--corpus` flag).
