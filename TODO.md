# TODO — native-video mode (feed video directly to a video model)

Idea: add an opt-in mode where vidwit sends the **video itself** to a
native video-capable model (e.g. Qwen3.6-27B on vLLM) and lets the model
sample frames internally, instead of vidwit extracting frames as still
images and sending them as `image_url` parts.

This doc captures everything learned standing up such an endpoint
(airig, 2026-07-03) and everything that will matter for the
implementation. Nothing here is built yet.

---

## 1. What actually changes (and what does NOT)

Current pipeline (`vidwit/pipeline.py :: run_one`):

1. `ffmpeg_io.probe` → duration/size/has_audio
2. `_ensure_transcript` — whisper → word-level transcript
3. `_ensure_frames` — **ffmpeg extracts stills at `cfg.fps`**, downscaled to
   `frame_width × frame_height` (default 256×144)
4. `windows()` — 10 s windows, 1 s overlap
5. per window: `llm.vision_chat(ChunkRequest{frames, transcript_lines, meta})`
   → vision LLM gets **base64 stills + transcript text** → one markdown block
6. `assemble` → stitch blocks

Native-video mode replaces **only step 3 + how frames reach the model in
step 5**: instead of ffmpeg stills sent as `image_url`, send a video clip
as an OpenAI `video_url` content part; the model does its own frame
sampling.

**Crucially unchanged — the transcript pipeline stays.** Qwen3.6-27B is a
**VL (vision) model, NOT Omni — it has no audio understanding.** The model
sees frames but cannot hear speech. vidwit's whole value is transcript +
visual, so whisper is still required and its output is still fed as text
alongside the video. Native mode swaps the *visual* channel from
"vidwit-sampled stills" to "model-sampled video", nothing else.

ffmpeg is still needed: `probe`, audio extraction for whisper, and (new)
**slicing the source into per-window clips** (see §4).

---

## 2. The endpoint (what we stood up on airig)

- **Model:** `cyankiwi/Qwen3.6-27B-AWQ-INT4` (community 4-bit; FP8 does not
  fit a 32 GB card — see the deployer repo `deployments/airig/ai/`).
  Vision retained (image + video). Served via vLLM ≥ 0.19.
- **Front:** LiteLLM proxy adds bearer auth; OpenAI-compatible.
- **URL (phase 1, VPN/LAN):** `http://192.168.2.70:4000/v1`
  (phase 2 will be `https://ai.kalsan.ch` — TLS via standalone traefik).
- **Client model name:** `qwen-vision`
- **Auth:** `Authorization: Bearer sk-...` — the key **must** carry the
  `sk-` prefix or LiteLLM treats it as a virtual key, hits the (absent)
  DB and returns `{"error":"No connected db."}`.
- **Limits:** `--max-model-len 32768`, `--limit-mm-per-prompt image=4,video=1`
  (one video per request). Note vLLM ≥0.19 wants that flag as JSON.
- vidwit config already supports this shape: `provider = "openai"`,
  `base_url`, `model`, `api_key`, plus `extra_body` merged into the
  chat-completions payload (see `LLMConfig`). A new provider is probably
  **not** needed — extend the OpenAI provider to emit `video_url`.

---

## 3. Video delivery to the model — two ways

Verified both against the live endpoint on 2026-07-03.

### a) base64 data URI (simplest, verified)
```json
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,<...>"}}
```
Works end-to-end through LiteLLM → vLLM. **Ceiling:** the whole clip rides
in the JSON body → base64 is +33% and LiteLLM/uvicorn body-size limits
will reject large payloads. Fine for ~10 s window clips (~1 MB → ~1.3 MB
b64). This is the recommended default for windowed mode.

### b) http(s) URL fetched server-side (for big/whole-video)
```json
{"type":"video_url","video_url":{"url":"http://<host>/clip.mp4"}}
```
vLLM fetches the URL **from airig**, so the URL must be reachable *from
the server*. Would require vidwit to serve the slice (local http server)
and airig to reach vidwit over the VPN (reverse reachability — not
guaranteed). More moving parts; only worth it if base64 size becomes a
problem.

---

## 4. Windowing vs whole-video

Token budget is the hard constraint. Measured: a **10 s clip ≈ 3666 prompt
tokens** at the model's default sampling. Context is **32768**. So:

- **Whole-video in one request** only works for short clips (~≤90 s at
  default sampling) — does not scale. Reject as the general path.
- **Keep windowing.** Slice the source into per-window clips with ffmpeg
  (`ffmpeg -ss <start> -t <len> -i in.mp4 -c copy win.mp4`, or re-encode
  if `-c copy` breaks keyframe alignment), send each clip as `video_url`
  plus that window's transcript lines + rolling summary + tail chunks.
  This preserves the existing context design (`ChunkRequest.rolling_summary`,
  `tail_chunks`) with minimal change.

Overlap: current design overlaps windows by `cfg.overlap`. Slicing must
honor the same start/end the `windows()` planner emits.

---

## 5. Frame-sampling control

In native mode vidwit no longer controls frame count via ffmpeg `-r fps`.
Instead the **model** samples. Qwen3-VL/3.6 accepts sampling hints
(`fps` / `nframes`, and a pixel budget) — pass them via the `video_url`
object and/or `extra_body` (`mm_processor_kwargs`). Implications:

- `cfg.fps` changes meaning: it becomes the *requested* sampling rate for
  the model, not an ffmpeg extraction rate. Wire it through.
- `CaptureMeta.fps` (forwarded to the prompt so the model calibrates
  against frame gaps) must reflect the **native** sampling actually used.
  The model may sample dynamically — document the resulting uncertainty in
  the prompt (see §6).
- Fewer frames = fewer tokens. For long static footage (property
  walkthroughs), request ~1–2 fps to stay under budget.
- Guardrail: log `prompt_tokens` from `usage`; if it approaches 32768,
  the window is over-sampled → lower fps or shorten the window.

---

## 6. Prompt changes

`_DEFAULT_SYSTEM_PROMPT` currently says *"Given frames and a word-level
transcript"* and tells the model to calibrate against *"the frame sampling
rate"*. For native mode:

- Reword to *"Given a short video clip and a transcript"*.
- The frame-gap calibration note still applies but is now about the
  model's own sampling density, not vidwit's stills. State the requested
  fps in the `# Capture metadata` block.
- Keep the strict single-markdown-block output contract unchanged.
- On-screen text / burned-in subtitles: a native video model may read
  these better than sparse stills — keep the "subtitle reads: …" rule.

### Thinking mode — must handle
Qwen3.6 is a **reasoning model**. The endpoint runs `--reasoning-parser
qwen3`, so responses carry `reasoning_content` and the model burns many
tokens "thinking" (observed ~200 completion tokens for a one-word reply).
vidwit needs a strict single block. Options:

- **Disable thinking** per request via `extra_body`
  (e.g. `chat_template_kwargs: {enable_thinking: false}` — confirm the
  exact key for this checkpoint), OR
- read `message.content` and **ignore** `reasoning_content`.

Do not let reasoning leak into the markdown block. This also cuts latency
and cost.

---

## 7. Config / API surface (proposed)

- New opt-in switch, e.g. `video_native = true` (TOML `[defaults]`) or
  a `--video-native` CLI flag. Default stays frame-extraction (backward
  compatible).
- Reuse existing `LLMConfig`: `provider="openai"`, `base_url`, `model`,
  `api_key`, `extra_body`.
- New knobs: video delivery method (`base64` | `served-url`), requested
  sampling `fps`/`nframes`, per-window clip length reuse of `cfg.window`.
- `llm.Provider.vision_chat` / `ChunkRequest`: add an optional
  `video_clip: Path | None`. When set, the OpenAI provider emits a
  `video_url` part instead of the `frames` `image_url` parts. Keep both
  paths so a provider that only does stills still works.

---

## 8. Performance / cost notes

- Native decode + frame sampling happens **on the server** (airig, CPU
  ffmpeg/decord), not in vidwit. Offloads work but adds server latency —
  observed ~34 s wall for a 10 s clip *including* thinking + fetch.
  Disabling thinking should cut this substantially.
- Per-window base64 upload cost scales with clip size — trim/downscale
  before sending (see §9).
- GPU is bandwidth-bound single-stream; concurrency (multiple windows in
  flight) would raise throughput. vidwit already has `jobs` parallelism —
  but mind `--limit-mm-per-prompt video=1` is per-request, and total
  concurrent requests share the one GPU.

---

## 9. Recommended pre-send preprocessing (per window clip)

Even in "native" mode, cheap conditioning of each clip keeps tokens and
payload predictable:

- **Trim** to the window (already implied by slicing).
- **Strip audio** (`-an`) — the VL model can't hear it; whisper handles
  audio separately. Shrinks the base64 payload for free.
- **Downscale** to ≤720p (`-vf scale=-2:720`) — fewer tokens/frame.
- **Re-encode to h264 mp4** if the source is exotic (HEVC/AV1/odd
  container) so server-side decode is reliable.

One-liner:
`ffmpeg -ss S -t L -i in.mp4 -an -vf "scale=-2:720,fps=2" -c:v libx264 -crf 28 win.mp4`

---

## 10. Open questions / to verify during implementation

- Exact `extra_body` key to disable Qwen3.6 thinking on this vLLM build.
- Exact per-request frame-sampling params the vLLM OpenAI server honors
  for `video_url` (fps vs nframes vs pixel budget) and their defaults.
- Whether `-c copy` slicing yields clips the server can decode cleanly, or
  re-encode is mandatory.
- LiteLLM/uvicorn max request-body size for base64 clips → the real size
  ceiling for delivery method (a).
- Quality A/B: does native-video mode produce a *better* witness record
  than the current still-frames + transcript pipeline? This is the point
  of the eval — compare on real vidwit inputs (property videos etc.).
- Does the model read burned-in on-screen text better from native video
  than from sparse stills? (Potential quality win.)

---

## 11. Reference

- Endpoint deployment: deployer repo `deployments/airig/ai/`
  (docker-compose: vllm + litellm + a small Gradio `webui/` that already
  does image+video upload via base64 — a working reference for the
  `video_url` request shape).
- Verified request shapes (text / image_url / video_url, base64 + http)
  all return correct output as of 2026-07-03.

---
---

# TODO — skip-identical-frames mode (change-driven sampling)

Idea: an opt-in option (e.g. `--skip-identical-frames` / TOML
`skip_identical_frames = true`) where frames are **not** taken at a fixed
rate. Instead, a frame is kept only if it differs from the **last kept
frame**. `fps` then becomes a **ceiling**: the maximum number of frames per
second, reached only while there is continuous movement.

Target use case: screen recordings, where long stretches are static (user
reads, thinks, talks) and sending identical stills wastes tokens and time.

## Example

`--fps 4 --skip-identical-frames` on a 20 s screen recording:

| time      | on screen          | frames kept |
|-----------|--------------------|-------------|
| 0–10 s    | user works         | 40 (4/s cap)|
| 10–15 s   | user pauses        | 0           |
| 15–20 s   | user works again   | 20          |

Total 60 frames instead of 80.

## Behaviour

- Sample candidates at `fps` as today, then drop each candidate that is
  "identical" to the last **kept** frame. Compare against the last kept
  frame, not the previous candidate, so slow drift (e.g. a slowly scrolling
  page) still gets captured once it has accumulated.
- "Identical" needs a tolerance, not byte equality: compression noise,
  a blinking cursor or a ticking clock in the taskbar must not count as
  change. Expose a threshold knob (e.g. `change_threshold`) with a sane
  default tuned on real screen recordings.
- Default off: fixed-rate sampling stays the default (backward compatible).

## Implementation notes

- **Detection tool:** the idea was OpenCV (e.g. `absdiff` on grayscale,
  downscaled frames, then fraction of changed pixels > threshold). Worth
  checking ffmpeg's `mpdecimate` filter first
  (`-vf "fps=4,mpdecimate=hi=…:lo=…:frac=…,scale=…" -fps_mode vfr`): it
  drops near-duplicates vs. the last kept frame, needs no new dependency
  (vidwit currently only depends on faster-whisper) and stays a single
  ffmpeg pass in `ffmpeg_io.extract_frames`. Use OpenCV only if mpdecimate's
  thresholds prove too coarse; if so make it an optional extra
  (`vidwit[opencv]`), not a hard dependency.
- **Timestamps per frame are now required.** `pipeline._process_window`
  maps frames to windows via `index = floor(t * fps)` (`pipeline.py`,
  "Frame index = floor(t * fps)"). That breaks once frames are skipped.
  Record each kept frame's timestamp (e.g. `-frame_pts 1` / `showinfo`
  parsing with ffmpeg, or write them from the OpenCV loop) into a sidecar
  (`frames.json`) and select window frames by timestamp.
- **Prompt metadata must change.** `llm._meta_block` tells the model
  "≈ 1/fps s between attached frames" and the system prompt tells it to
  calibrate against the sampling rate. With skipping, spacing is irregular:
  attach each frame's timestamp (e.g. a text part `t=12.25 s` before each
  image) and state in the metadata that frames are change-driven, that
  `fps` is only the maximum, and that a gap means "screen unchanged".
- **Windows with zero frames** (pause spans the whole window): still send
  the transcript. Consider attaching the last kept frame from before the
  window as a reference ("screen as it has been since t=…"), otherwise the
  model has no visual context for what the user is talking about.
- **Resume:** `_ensure_frames` reuses any existing `f_*.jpg` on `--resume`.
  Store the sampling mode/threshold alongside the frames and re-extract if
  they differ, so a fixed-rate frame set is not reused in skip mode or
  vice versa.
- Log how many candidates were dropped (e.g. "kept 60/80 frames") so the
  threshold can be tuned.
