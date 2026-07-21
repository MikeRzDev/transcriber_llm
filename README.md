# transcribe-stt

A terminal speech-to-text client for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) voice models with Metal GPU acceleration on Apple Silicon, plus an **MLX engine** that runs directory-shaped Apple-Silicon models (NVIDIA Parakeet, Qwen3-ASR, Canary, Whisper-MLX, …) via [mlx-audio](https://github.com/Blaizzy/mlx-audio) (installs as the **Transcribe Speech** app on macOS). Browse or drag-and-drop audio *and video* files, watch segments stream in live, and export transcripts in an LLM-optimized format. Transcription runs fully locally; the network is only touched when you download models from Hugging Face in the built-in model manager.

```
 transcribe-stt   ggml-large-v3.bin (3.0 GB)  •  Metal GPU
┌ ~/Documents/AI/transcribe ─┐┌ demo.wav ──────────────────────────────────┐
│ ../                        ││ [00:00 → 00:05] Hello, this is a demo of   │
│ samples/                   ││                 the whisper terminal…      │
│  demo.wav                  ││ [00:05 → 00:11] It transcribes audio files │
│                            ││                 locally on Apple Silicon…  │
└────────────────────────────┘└─────────────────────────────────────────────┘
 ▓▓▓▓▓▓▓▓░░░░ 64%  Transcribing 126s of audio…
 Tab switch  Enter transcribe  m models  e export  c cancel  q quit
```

## Requirements

- macOS on Apple Silicon (Metal), Xcode command line tools
- Rust toolchain and cmake (`brew install rust cmake`)
- ffmpeg for video files and exotic audio codecs (`brew install ffmpeg`); pure-audio formats work without it
- for MLX models (Parakeet, Qwen3-ASR, …): the mlx-audio runtime — set up automatically, see below (needs any Python ≥ 3.10 to bootstrap from)

## Setup

```sh
# check deps (Xcode CLT, Homebrew, Rust, cmake, ffmpeg), install anything
# missing, then compile and install the transcribe-stt binary into ~/.cargo/bin
./scripts/install.sh

# optional: package the app as "Transcribe Speech" in /Applications, so it can
# be launched from Launchpad/Spotlight (opens in a Terminal window);
# re-run after rebuilding to refresh the bundled binary
./scripts/make-app.sh
```

Models are downloaded from inside the app: press `s` → **Model management** (see below). For a quick smoke test without the TUI, `transcribe-stt --download-test-model` fetches the smallest whisper model (ggml-tiny, ~75 MB) into the models folder.

## Usage

```sh
# browse your home folder
transcribe-stt

# start in a directory, or jump straight into transcribing a file
transcribe-stt ~/Podcasts
transcribe-stt interview.m4a

# no TUI — print segments to stdout (good for scripting)
transcribe-stt --headless -m models/ggml-large-v3.bin audio.mp3

# two-person conversation with speaker labels, language pinned to English
transcribe-stt --headless --diarize -l en call.m4a

# grab the smallest whisper model (~75 MB) for testing
transcribe-stt --download-test-model
```

Supported audio: wav, mp3, m4a (AAC/ALAC), flac, ogg/vorbis, opus, aiff, caf, wma, mka, weba, amr, ac3, dts, ape, wv, au, mp2, spx, tta, mpc, ra, gsm, w64 (formats symphonia can't decode natively go through ffmpeg).
Supported video (audio track is ripped automatically via ffmpeg, with a live progress gauge; cancellable like any job): mp4, mov, m4v, mkv, webm, avi, ts, mts, m2ts, 3gp, 3g2, flv, wmv, mpg, mpeg, m2v, ogv, vob, asf, f4v, divx, rm, rmvb.

### Drag & drop

Drop any audio or video file from Finder onto the terminal window and the start-confirmation dialog opens (drop a folder to browse it). Works via bracketed paste, with a key-burst fallback for terminals that don't support it.

### Keys

| Key | Action |
| --- | --- |
| `Tab` | Switch between file browser and transcript |
| `Enter` | Open directory / transcribe selected file (a confirmation dialog shows the file, model, and export formats before every job) |
| `m` | Model picker (lists `.bin`/`.gguf` files and MLX model folders in the models folder) |
| `s` | Settings: default model, models & output folders (via a built-in directory browser), export formats, model management, diarization, split mode, language (persisted) |
| `d` | Cycle the diarization strategy (off → auto → tinydiarize → embeddings) |
| `n` | Name the detected speakers (voice sample per speaker, re-exports on close) |
| `l` | Toggle the right pane between the **job log** (the default view) and the transcript. The log is a timestamped record of everything since the first file was loaded: file selection, model load, audio extraction, engine output, every segment, exports, errors (10k-line ring buffer, virtualized rendering, `j k`/`g`/`G` scroll with live follow) |
| `e` | Export the job log to `<output folder>/logs/log_<YYYYMMDD_HHMMSS>.log` — works at any time, including mid-job |
| `x` | Clear the job log |
| `c` | Cancel the running transcription (shown in the key bar only while a job is running) |
| `↑↓` / `j k`, `PgUp/PgDn`, `g`/`G` | Navigate / scroll (G re-enables follow) |
| `q` / `Ctrl-C` | Quit |

## LLM-ready export

After every successful transcription, the selected formats are written automatically into `<output folder>/<source-stem>_<YYYYMMDD_HHMMSS>/`. Which formats get written is chosen in settings → *Export formats* — a multi-select of md, json, txt, and srt (all on by default; at least one always stays selected). The output folder is set in settings and defaults to `~/Documents/llm_transcribe/output` on macOS (`~/llm_transcribe/output` on Linux). The primary output is `<name>.llm.md` — a transcript formatted for an LLM to reason over and act on:

- YAML frontmatter with machine-readable metadata: source file, duration, auto-detected language, model, segment count, and an explicit `diarization: none` marker
- A short preamble telling the model how to interpret the document
- Raw ASR segments merged into readable paragraphs, split on speech pauses (which usually track speaker turns), each anchored with a uniform `[HH:MM:SS]` timestamp into the source media

`<name>.segments.json` carries the same content at segment granularity for programmatic pipelines, plus plain `.txt` and `.srt`.

## Speaker diarization (strategy-based)

Diarization is a pluggable strategy, cycled with `d` (or in settings; persisted): **Off → Auto → TinyDiarize → Speaker embeddings**. `Auto` picks the recommended strategy for the model in use — a tdrz model brings its own turn tokens, everything else gets the embedding pipeline. Whatever a strategy still needs is spelled out — with sizes — in the status line when you select it, and again in the pre-transcription confirmation dialog; nothing downloads silently.

- **TinyDiarize** — whisper.cpp's native [tinydiarize](https://github.com/akashmjn/tinydiarize) turn detection: transcribes with the `ggml-small.en-tdrz.bin` model (488 MB, English only, available under Model management's suggestions since its repo is untagged on HF search). Detected turns alternate **Speaker A / Speaker B** — the right assumption for two-person conversations. A/B are consistent but arbitrary (A = whoever speaks first), missed turns merge speakers, and turn detection is tuned for real conversational speech — synthetic TTS audio often yields no turns.
- **Speaker embeddings** — engine-agnostic, works with **any** model (whisper.cpp and MLX alike) and any number of speakers: pyannote-style segmentation plus speaker-embedding clustering, run fully offline by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — no PyTorch, no account. It runs as a post-pass in the worker: the engine transcribes as usual, then the diarizer decodes the same audio, finds speaker turns, and labels each segment by overlap. Self-provisioning on first use (like the MLX runtime): the sherpa-onnx wheel pip-installs into the app venv (~25 MB) and the active ONNX models download into `<models>/diarization/` with live progress. Cancel or failure downgrades gracefully — the finished transcript is kept, just unlabeled.
- Models that label speakers natively (some MLX models emit `speaker_id`) keep their own labels; the post-pass steps aside.

**Diarization model catalog** — Model management (`s` → Model management) has a *diarization* section listing the interchangeable pipeline components, downloaded into `<models>/diarization/`: segmentation (pyannote segmentation-3.0 6 MB — default; Rev reverb-diarization-v1 10 MB, English, non-commercial license) and speaker embeddings (NeMo TitaNet-S 40 MB — default; TitaNet-L 101 MB higher quality; WeSpeaker CAM++ 29 MB; 3D-Speaker CAM++ zh+en 28 MB for Chinese/English). Enter downloads a missing component or makes an installed one the active choice for its role (✓); Del removes it. A freshly downloaded component becomes active automatically.

**Speaker count** — Settings → *Speakers*: auto-detect by default, or enter the known number of speakers (1–26) to pin the clustering to exactly that count — when you know it, labels get noticeably better. Headless: `--speakers N`.

**Naming speakers** — after a diarized transcription, press `n`: every detected speaker is listed with a voice sample (their longest utterance, `p` plays it via afplay), and Enter assigns a real name — George, Marco, Sarah. Names replace the anonymous letters in the TUI and in every export (with a `speakers: Speaker A = George, …` frontmatter line in `llm.md`); closing the dialog re-exports automatically if anything changed.

Labels flow into the TUI (colored per speaker), `llm.md` (speaker-prefixed paragraphs and honest `diarization:` metadata naming the method for the LLM), JSON, SRT and TXT. Headless: `--diarize [off|auto|tdrz|embedding]` (bare `--diarize` = auto).

## Long audio & split mode

Settings → **Split mode** decides how long recordings are fed to the engine:

- **auto** (default) — the whole file goes to whisper.cpp in one call; it windows 30 s at a time internally and seeks back to the last completed segment between windows, so words are never cut. Right for whisper models of any length.
- **silence** — the app chunks client-side at ~60 s, placing each cut at the quietest speech pause (25 ms RMS scan) within the 15 s before the target boundary. If speech is continuous, it falls back to a 0.5 s overlap and de-duplicates segments at the seam.
- **fixed** — plain 60 s chunks with the 0.5 s overlap + dedup, no pause hunting.

All chunking happens on the decoded 16 kHz f32 samples in memory — nothing is re-encoded, so splitting never costs quality. Timestamps are re-based so exports read as one continuous transcript, and diarization speaker labels carry across chunk boundaries. Engine capabilities are declared per engine (`split::EngineCaps`): whisper.cpp accepts unlimited length, and a future engine with a hard input window will force chunking automatically regardless of mode. Split mode applies to the whisper.cpp engine only — mlx-audio ingests the whole file and windows internally, so the MLX engine bypasses the planner.

## Language

Whisper auto-detects the language from the first 30 seconds by default. If you know it in advance, set it in settings (`s` → Language, ISO 639-1 code like `en`/`es`/`de`) or pass `-l <code>` — this skips detection, avoids misdetection on short or noisy clips, and steers decoding from the first token. English-only models (`.en`) always use `en`.

## Models & settings

Models live in `~/Documents/llm_transcribe/models` by default on macOS (`~/llm_transcribe/models` on Linux), with transcript exports next door in `…/llm_transcribe/output`. Press `s` to change the models folder, output folder (both open the built-in directory browser — navigate with `↑↓`/`Enter`, pick with "use this folder"), default model, diarization and language; all persist to `~/.config/transcribe-stt/config.toml` (`TRANSCRIBE_STT_MODELS` overrides the models folder). Changing the models folder rescans it: every supported model already inside joins the picker, and if none was selected the best available one (configured default, then large-v3, then first) is adopted automatically. If the previous folder still holds models, a dialog offers to move them into the newly selected folder (same-volume moves are instant renames; already-present entries are skipped; directory models move whole; the move runs in the background). The picker lists whisper.cpp GGML `.bin` files, `.gguf` files, and MLX model **folders** (a directory holding `config.json` plus safetensors weights), so any of them can also be dropped in by hand. Models load lazily: opening the app loads nothing (the header just shows file metadata), the selected model is loaded when the first transcription starts, stays resident for instant follow-up jobs, and is released as soon as you select a different model. Loading shows a real progress bar (byte-level file read, then ggml/Metal init) and can be cancelled with `c`; unloading reports start and finish in the status line. The demo model is **whisper large-v3**; on an M4 Pro it transcribes at ~0.2× realtime with Metal.

## Model management (in-app downloads)

`s` → **Model management** opens the built-in Hugging Face browser:

- A curated **suggested list** (from `assets/suggested_models.json`, embedded at build time) shows first: whisper-large-v3 (GGML) plus MLX directory models (NVIDIA Parakeet, Qwen3-ASR). On Apple Silicon every entry downloads with one Enter; MLX entries are greyed out on other machines.
- **Type to search** Hugging Face (debounced live search, most-downloaded first), restricted to the **speech-to-text** category (`automatic-speech-recognition`). Enter on a repo lists what it offers: MLX model **folders** first (the repo root, or per-variant subfolders — e.g. quantizations — each holding its own `config.json` + weights, each downloadable as an individual model), then its GGML/GGUF files with sizes. Enter downloads the selected entry into the models folder with a progress bar. Esc goes back a level, cancels a running download, or closes the modal. Note: conversion repos that never set a pipeline tag (e.g. tinydiarize) won't appear — download those manually into the models folder.
- Single files stream to `<name>.part` and are renamed into place only when complete; directory models stream file-by-file into `<name>.part/` (aggregated progress, pause/resume across files) and the folder is renamed into place only when every file is complete — an interrupted download never shows up as a usable model. Variant subfolders land as `<repo>-<variant>/`.
- Every downloaded model folder carries an integrity manifest (`.manifest.json`: the exact file list with sizes). A folder that later fails the check — a file deleted or truncated — is pulled from the library, and re-downloading it fetches **only the missing files** before restoring it.
- On Apple Silicon the header shows **Metal GPU** plus whether mlx-audio is installed; runnable files are badged `Metal ✓`, model folders `MLX`. There is no separate "Metal model" file: the Metal backend accelerates the *same* GGML file that runs on CPU elsewhere.

## MLX engine (Parakeet, Qwen3-ASR, Canary, Whisper-MLX, …)

Directory models run on the MLX engine, which shells out to [mlx-audio](https://github.com/Blaizzy/mlx-audio)'s STT CLI (`python -m mlx_audio.stt.generate`) — one runner for every MLX model family, so new families need no new engine code. Requirements: Apple Silicon; the runtime is **self-provisioning** — `scripts/install.sh` sets it up ahead of time, and if it's missing the first MLX job installs mlx-audio into an app-managed venv (`~/Library/Application Support/transcribe-stt/mlx-venv`, so it never fights the system/Homebrew Python) with pip's own progress streamed live into the status line, cancellable like any job. A Python where mlx-audio is already installed is used as-is. Detection is a fast import probe (no model load); Model management shows the runtime status in its header. Audio is decoded by the same in-app pipeline as whisper (so video and exotic codecs work identically) and handed over as a temp 16 kHz WAV; the subprocess loads the model per job (cold start); every line the subprocess prints streams live into the TUI status line (sanitized, `\r` progress bars included) so long silent phases always show what is happening; cancel kills the subprocess, and its JSON output (`segments` or `sentences`, with optional `speaker_id`) is mapped back into the same streamed segment/export pipeline. The RAM/CPU readout counts the whole process tree, so the Python child holding the MLX model weights is included. The whisper.cpp path is untouched: tdrz diarization, split modes, and the resident-context fast path remain whisper-only.

## Testing

```sh
cargo test                    # 94 unit tests, fast and hermetic
cargo test -- --ignored       # 3 transcription e2e tests (real model, Metal, ffmpeg,
                              # diarization) + 2 live Hugging Face tests (single-file
                              # download, whole-folder MLX download with manifest)
```

Unit tests cover audio decode/downmix/resampling (incl. the WAV writer round trip), model discovery (files and MLX folders, manifest integrity), config parsing and platform defaults, drop-path parsing (quotes, backslash escapes, `file://` URLs), the app state machine including auto-export, the directory picker, the model-management modal (search input, navigation, download events, variant folders, platform gating), Hub API response parsing (file listing, variant detection), mlx-audio JSON output mapping, and every export format. The e2e tests run the actual binary headless against the demo WAV and MP4. Live MLX inference needs mlx-audio and a downloaded model, so it stays a manual test.

## Architecture

The crate is a library (`transcribe_stt`) plus a thin binary: `src/main.rs` only parses the CLI (clap), dispatches through `lib::run`, and works around the ggml Metal atexit hang with `libc::_exit`.

- `src/cli.rs` / `src/headless.rs` / `src/tui.rs` — the three entry paths: clap argument parsing, `--headless` transcription + `--download-test-model` (stderr progress, stdout segments), and the terminal lifecycle + render/event loop
- `src/app.rs` + `src/app/` — application state and update logic (Elm-style). `App` is composed of per-concern sub-state, each with its logic and key handling beside it:
  - `browser.rs` (`FileBrowser` — directory listing/navigation), `library.rs` (`ModelLibrary` + `ModelPicker`), `settings.rs` (`SettingsUi`, `SettingsRow`, directory picker, move-models prompt), `transcript.rs` (`TranscriptState` — segments, scroll/follow), `hub_state.rs` (`HubState` — the Model management modal), `events.rs` (worker-event handling incl. auto-export), `drop.rs` (drag-and-drop path parsing + the `DropDetector` key-burst fallback), `keys.rs` (the modal-priority key router)
- `src/ui.rs` + `src/ui/` — rendering, one file per widget/modal, all reading `&App` (scroll clamping runs in the update phase, never during draw); `theme.rs` holds the shared colors/spinner/highlight style, `layout.rs` the screen regions
- `src/transcribe.rs` + `src/transcribe/` — `worker.rs`: the worker thread executing jobs; `backend.rs`: the common `Engine` interface every backend implements, plus the model-shape routing (`Backends::for_model` — a directory routes to MLX, a file to whisper); `backend/whisper_metal.rs`: whisper.cpp via whisper-rs, resident context between jobs, cancellation via whisper's abort callback; `backend/mlx.rs`: the mlx-audio subprocess runner (runtime probe, temp-WAV hand-off, JSON parsing, kill-on-cancel). New backends slot in as new submodules behind the same trait
- `src/audio.rs` — symphonia decode (audio) → mono downmix → rubato sinc resample to 16 kHz; video and unsupported codecs go through ffmpeg, which streams raw 16 kHz mono f32 over stdout (no temp files) with byte-accurate extraction progress against the ffprobe duration; both paths are cancellable mid-decode; plus the 16 kHz WAV writer used to hand audio to subprocess engines
- `src/hub.rs` + `src/hub/` — Model management backend: `api.rs` (Hugging Face search, repo file listing, directory-variant detection, serde-typed responses) and `download.rs` (streaming `.part` downloads with retry; whole-folder downloads with aggregated progress, integrity manifest, and missing-file repair), each on worker threads; the suggested list is embedded from `assets/suggested_models.json`
- `src/split.rs` — long-audio chunk planner (engine capability descriptor, silence-aware cut placement, overlap fallback)
- `src/export.rs` — all transcript output formats (llm.md, JSON via serde_json, txt, srt)
- `src/config.rs` — persisted settings (`~/.config/transcribe-stt/config.toml`), strict TOML via serde with a lenient fallback parser for legacy hand-edited files
- `src/models.rs` / `src/format.rs` / `src/stats.rs` — local model scanning (files and MLX folders, integrity-manifest checks) + default-model selection, shared time/size formatting, live process stats
- whisper.cpp log output is routed through the `log` crate so it can't corrupt the TUI
