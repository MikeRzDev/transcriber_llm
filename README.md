# transcribe-stt

A terminal speech-to-text client for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) voice models with Metal GPU acceleration on Apple Silicon (installs as the **Transcribe Speech** app on macOS). Browse or drag-and-drop audio *and video* files, watch segments stream in live, and export transcripts in an LLM-optimized format. Transcription runs fully locally; the network is only touched when you download models from Hugging Face in the built-in model manager.

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
# browse the current directory
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

Supported audio: wav, mp3, m4a (AAC/ALAC), flac, ogg/vorbis, opus, aiff, caf, wma.
Supported video (audio track is extracted automatically via ffmpeg): mp4, mov, m4v, mkv, webm, avi, ts, 3gp, flv, wmv.

### Drag & drop

Drop any audio or video file from Finder onto the terminal window and transcription starts immediately (drop a folder to browse it). Works via bracketed paste, with a key-burst fallback for terminals that don't support it.

### Keys

| Key | Action |
| --- | --- |
| `Tab` | Switch between file browser and transcript |
| `Enter` | Open directory / transcribe selected file |
| `m` | Model picker (lists `.bin` and `.gguf` in the models folder) |
| `s` | Settings: default model, models & output folders (via a built-in directory browser), model management, diarization, split mode, language (persisted) |
| `d` | Toggle speaker diarization for the next transcription |
| `e` | Re-export the transcript (exports also run automatically after every transcription) |
| `c` | Cancel the running transcription |
| `↑↓` / `j k`, `PgUp/PgDn`, `g`/`G` | Navigate / scroll (G re-enables follow) |
| `q` / `Ctrl-C` | Quit |

## LLM-ready export

After every successful transcription, all formats are written automatically into `<output folder>/<source-stem>_<YYYYMMDD_HHMMSS>/`; `e` re-exports on demand. The output folder is set in settings and defaults to `~/Documents/llm_transcribe/output` on macOS (`~/llm_transcribe/output` on Linux). The primary output is `<name>.llm.md` — a transcript formatted for an LLM to reason over and act on:

- YAML frontmatter with machine-readable metadata: source file, duration, auto-detected language, model, segment count, and an explicit `diarization: none` marker
- A short preamble telling the model how to interpret the document
- Raw ASR segments merged into readable paragraphs, split on speech pauses (which usually track speaker turns), each anchored with a uniform `[HH:MM:SS]` timestamp into the source media

`<name>.segments.json` carries the same content at segment granularity for programmatic pipelines, plus plain `.txt` and `.srt`.

## Speaker diarization (2-person conversations)

Optional and decided **before** transcription: toggle with `d` (or in settings; persisted). It uses whisper.cpp's native [tinydiarize](https://github.com/akashmjn/tinydiarize) turn detection via the `ggml-small.en-tdrz.bin` model (English only) — download it manually from [huggingface.co/akashmjn/tinydiarize-whisper.cpp](https://huggingface.co/akashmjn/tinydiarize-whisper.cpp) into the models folder; its repo is untagged on Hugging Face, so the in-app speech-to-text search won't list it. When on, the run automatically switches to the tdrz model, and detected turns alternate **Speaker A / Speaker B** labels — the right assumption for two-person conversations. Labels flow into the TUI (colored), `llm.md` (speaker-prefixed paragraphs and honest `diarization:` metadata for the LLM), JSON, SRT and TXT.

Caveats, stated in the export too: A/B are consistent but arbitrary (A = whoever speaks first), missed turns merge speakers, and turn detection is tuned for real conversational speech — synthetic TTS audio often yields no turns (verified identical to reference whisper.cpp behavior).

## Long audio & split mode

Settings → **Split mode** decides how long recordings are fed to the engine:

- **auto** (default) — the whole file goes to whisper.cpp in one call; it windows 30 s at a time internally and seeks back to the last completed segment between windows, so words are never cut. Right for whisper models of any length.
- **silence** — the app chunks client-side at ~60 s, placing each cut at the quietest speech pause (25 ms RMS scan) within the 15 s before the target boundary. If speech is continuous, it falls back to a 0.5 s overlap and de-duplicates segments at the seam.
- **fixed** — plain 60 s chunks with the 0.5 s overlap + dedup, no pause hunting.

All chunking happens on the decoded 16 kHz f32 samples in memory — nothing is re-encoded, so splitting never costs quality. Timestamps are re-based so exports read as one continuous transcript, and diarization speaker labels carry across chunk boundaries. Engine capabilities are declared per engine (`split::EngineCaps`): whisper.cpp accepts unlimited length, and a future engine with a hard input window (e.g. MLX Canary) will force chunking automatically regardless of mode.

## Language

Whisper auto-detects the language from the first 30 seconds by default. If you know it in advance, set it in settings (`s` → Language, ISO 639-1 code like `en`/`es`/`de`) or pass `-l <code>` — this skips detection, avoids misdetection on short or noisy clips, and steers decoding from the first token. English-only models (`.en`) always use `en`.

## Models & settings

Models live in `~/Documents/llm_transcribe/models` by default on macOS (`~/llm_transcribe/models` on Linux), with transcript exports next door in `…/llm_transcribe/output`. Press `s` to change the models folder, output folder (both open the built-in directory browser — navigate with `↑↓`/`Enter`, pick with "use this folder"), default model, diarization and language; all persist to `~/.config/transcribe-stt/config.toml` (`TRANSCRIBE_STT_MODELS` overrides the models folder). Changing the models folder rescans it: every supported model already inside joins the picker, and if none was selected the best available one (configured default, then large-v3, then first) is adopted automatically. If the previous folder still holds models, a dialog offers to move them into the newly selected folder (same-volume moves are instant renames; already-present files are skipped; the move runs in the background). The picker lists both whisper.cpp GGML `.bin` files and `.gguf` files, so other ggml-family voice models can be dropped in alongside whisper. Models load lazily: opening the app loads nothing (the header just shows file metadata), the selected model is loaded when the first transcription starts, stays resident for instant follow-up jobs, and is released as soon as you select a different model. Loading shows a real progress bar (byte-level file read, then ggml/Metal init) and can be cancelled with `c`; unloading reports start and finish in the status line. The demo model is **whisper large-v3**; on an M4 Pro it transcribes at ~0.2× realtime with Metal.

## Model management (in-app downloads)

`s` → **Model management** opens the built-in Hugging Face browser:

- A curated **suggested list** (from `assets/suggested_models.json`, embedded at build time) shows first. Entries the whisper.cpp engine can run download with one Enter; MLX-only entries (parakeet, canary) are greyed out until the app grows an MLX engine — there is no GGML build of them.
- **Type to search** Hugging Face (debounced live search, most-downloaded first), restricted to the **speech-to-text** category (`automatic-speech-recognition`). Enter on a repo lists its GGML/GGUF files with sizes; Enter on a file downloads it into the models folder with a progress bar. Esc goes back a level, cancels a running download, or closes the modal. Note: conversion repos that never set a pipeline tag (e.g. tinydiarize) won't appear — download those manually into the models folder.
- Downloads stream to `<name>.part` and are renamed into place only when complete, so an interrupted download never shows up as a usable model.
- On Apple Silicon the header shows **Metal GPU** and runnable files are badged `Metal ✓`. There is no separate "Metal model" file: the Metal backend accelerates the *same* GGML file that runs on CPU elsewhere.

## Testing

```sh
cargo test                    # 71 unit tests, fast and hermetic
cargo test -- --ignored       # 3 transcription e2e tests (real model, Metal, ffmpeg,
                              # diarization) + 1 live Hugging Face search/download test
```

Unit tests cover audio decode/downmix/resampling, model discovery, config parsing and platform defaults, drop-path parsing (quotes, backslash escapes, `file://` URLs), the app state machine including auto-export, the directory picker, the model-management modal (search input, navigation, download events, MLX gating), Hub API response parsing, and every export format. The e2e tests run the actual binary headless against the demo WAV and MP4.

## Architecture

The crate is a library (`transcribe_stt`) plus a thin binary: `src/main.rs` only parses the CLI (clap), dispatches through `lib::run`, and works around the ggml Metal atexit hang with `libc::_exit`.

- `src/cli.rs` / `src/headless.rs` / `src/tui.rs` — the three entry paths: clap argument parsing, `--headless` transcription + `--download-test-model` (stderr progress, stdout segments), and the terminal lifecycle + render/event loop
- `src/app.rs` + `src/app/` — application state and update logic (Elm-style). `App` is composed of per-concern sub-state, each with its logic and key handling beside it:
  - `browser.rs` (`FileBrowser` — directory listing/navigation), `library.rs` (`ModelLibrary` + `ModelPicker`), `settings.rs` (`SettingsUi`, `SettingsRow`, directory picker, move-models prompt), `transcript.rs` (`TranscriptState` — segments, scroll/follow), `hub_state.rs` (`HubState` — the Model management modal), `events.rs` (worker-event handling incl. auto-export), `drop.rs` (drag-and-drop path parsing + the `DropDetector` key-burst fallback), `keys.rs` (the modal-priority key router)
- `src/ui.rs` + `src/ui/` — rendering, one file per widget/modal, all reading `&App` (scroll clamping runs in the update phase, never during draw); `theme.rs` holds the shared colors/spinner/highlight style, `layout.rs` the screen regions
- `src/transcribe.rs` + `src/transcribe/` — `worker.rs`: the worker thread owning the `WhisperContext`; models load lazily on the first job, stay resident between jobs, and unload when the user switches models; `run.rs`: one job start-to-finish — segments/progress stream back over a channel, cancellation via whisper's abort callback
- `src/audio.rs` — symphonia decode (audio) → mono downmix → rubato sinc resample to 16 kHz; video and unsupported codecs go through ffmpeg, which streams raw 16 kHz mono f32 over stdout (no temp files)
- `src/hub.rs` + `src/hub/` — Model management backend: `api.rs` (Hugging Face search + repo file listing, serde-typed responses) and `download.rs` (streaming `.part` downloads with retry), each on worker threads; the suggested list is embedded from `assets/suggested_models.json`
- `src/split.rs` — long-audio chunk planner (engine capability descriptor, silence-aware cut placement, overlap fallback)
- `src/export.rs` — all transcript output formats (llm.md, JSON via serde_json, txt, srt)
- `src/config.rs` — persisted settings (`~/.config/transcribe-stt/config.toml`), strict TOML via serde with a lenient fallback parser for legacy hand-edited files
- `src/models.rs` / `src/format.rs` / `src/stats.rs` — local model scanning + default-model selection, shared time/size formatting, live process stats
- whisper.cpp log output is routed through the `log` crate so it can't corrupt the TUI
