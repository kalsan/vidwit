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
