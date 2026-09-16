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

To build an optimized executable into `binary/` at the project root:

```sh
./build.sh
./binary/transcribe-stt --help
```

The build script requires Rust, CMake, and the Xcode command line tools. It also recognizes CMake in this checkout's `.venv-realtime` environment. The existing installation script below can set up missing dependencies. MLX models still use the Python runtime described below.

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

### Live microphone transcription

Press **Shift-R** (`R`) to start recording with the selected MLX speech model, or launch directly:

```sh
# Use models already downloaded on an external drive; m switches models.
./scripts/run-realtime.sh /Volumes/MikeExternal/ai_models/speech-to-text-realtime

# Equivalent flags for an installed binary:
transcribe-stt --realtime --models-dir /path/to/models -l es
transcribe-stt --realtime -m /path/to/Qwen3-ASR-0.6B-8bit -l es
```

The left pane becomes a microphone monitor: a waveform of the actual input amplitude over the last 12 seconds, recording indicator, elapsed time, dBFS level, clipping/quiet indicator, input device and pending audio time. The right pane switches to the **live transcript**, showing partial text in yellow as the model produces it. New text scrolls into view automatically. `↑↓`, `j k`, or PageUp/PageDown pause following; **G** follows the newest text again. `l` opens the engine log.

**R** stops the microphone immediately, finishes queued speech (including the final partial audio packet), and automatically exports the transcript using the configured formats. **q** stops, exports, then quits. **c** cancels inference; **Ctrl-C** quits immediately. Cancellation/failure keeps the visible text but does not auto-export unfinished results. After stopping, **r** returns to the file browser; **R** starts a fresh session. Microphone audio stays in bounded memory and is not saved as an audio file. Live diarization is off; exported timestamps are approximate spans of microphone time rather than word alignment.

Live model support:

| Local model | Live behavior |
| --- | --- |
| Voxtral-Mini-4B-Realtime-2602, mlx-audio 4-bit / fp16 | Continuous audio streaming when the runtime provides `create_streaming_session`; otherwise short speech windows with streamed text |
| Qwen3-ASR, including 0.6B / 1.7B MLX variants | Model stays loaded; text streams from speech windows of up to 4 seconds, ending earlier on a pause |
| Other mlx-audio ASR folders | Same speech-window path; token updates when the model exposes `generate(stream=True)`, otherwise completed windows |
| Voxtral-Mini-4B-Realtime-6bit converted with voxmlx | Incompatible conversion; select an mlx-audio Voxtral folder instead |
| Qwen3-ForcedAligner | Requires existing text; cannot transcribe live audio |
| GGML/GGUF Whisper | File transcription only; select an MLX folder for live mode |

The model picker marks compatible MLX folders with `live ✓`. `--models-dir` selects the library for the current session without moving or downloading weights. A model is loaded **once per recording session**, before the microphone starts. Local inference runs with Hugging Face offline mode enabled. Missing MLX runtime packages are provisioned through the existing app venv; model weights are never downloaded by live mode. macOS may ask you to allow your terminal under **Privacy & Security → Microphone**. Press **a** to choose the built-in, USB, or a connected Bluetooth microphone; **r** refreshes the device list after connecting a headset. **System default** follows the input selected in macOS Sound settings when recording starts. The chosen input is retained between recordings in the current app session, and its actual name appears beside the waveform. Stop with **R** before switching inputs. If an explicitly chosen Bluetooth microphone disconnects, recording reports an error rather than switching to another microphone.

To list inputs without recording, run `transcribe-stt --list-input-devices` (or `./scripts/run-realtime.sh --list-input-devices` from this checkout). You can also select an exact listed name with `--input-device "AirPods Microphone"`. Pair and connect Bluetooth microphones in macOS first.

Latency depends on the model and hardware. The microphone monitor displays measured inference seconds per audio second (a rolling average of the last eight requests); values above 1 mean processing is slower than realtime. Native models automatically use the saved benchmark feed size when the hardware and model files match (1 second for the benchmarked Voxtral 4-bit model). The microphone monitor and engine log show the applied setting. Without a valid saved result, native feed sizes adapt between 0.5 and 4 seconds based on observed inference time. Qwen and Parakeet retain the tested speech windows of up to four seconds. Legacy Voxtral 4-bit checkpoints are optimized once in memory at load time: remaining large embedding/adapter matrices use the existing 4-bit format, and floating parameters/scales use BF16. The on-disk weights are unchanged; fp16 and other model variants keep their selected precision. [Sustained validation](benchmarks/voxtral-optimization.md) measured 0.44 processing seconds per audio second over five minutes. Native streaming context is flushed at a quiet boundary after 12 seconds or at a hard limit of 20 seconds, then renewed using the same loaded model weights. Pending text is drained first; no input samples are skipped or replayed. A forced boundary can affect recognition around that boundary. This bounds context growth but cannot guarantee realtime speed. The in-memory queue tolerates up to 120 seconds of backlog (about 23 MB at 48 kHz mono). If it fills, recording stops gracefully, all captured packets are transcribed, and the result is exported. This limit is a memory safeguard, not a model-specific speed estimate; no finite buffer can compensate for sustained inference slower than realtime. The speech-window fallback uses a simple input-level silence detector and can cut continuous speech at a window boundary. Pin Qwen's language with `-l es` or the TUI language setting; native Voxtral detects its own language.

The **m** model selector shows measured live suitability for this Mac: **good fit** (at least 20% compute headroom), **borderline** (keeps up with little headroom), or **too slow for this Mac** (a tested repetition fell behind). Each rating shows processing seconds per audio second and its tested feed size. Incompatible models say **live unavailable**; models without matching measurements say **not benchmarked**. These labels concern live transcription; models remain selectable for file transcription. Ratings are stored beside the app config in `live-benchmarks.json`, checked against the Mac's hardware and model file sizes/modification times, and refreshed whenever you press **m**. Starting a live session revalidates and automatically applies the saved native feed size, including when starting directly with `--realtime`.

To benchmark every model in a local library with the same speech recording:

```sh
.venv-realtime/bin/python scripts/benchmark-realtime.py \
  --models-dir /path/to/models --audio speech-16khz-mono.wav --language es \
  --seconds 24 --repeats 2 --output benchmarks/realtime.json
```

Use a 16 kHz mono PCM16 WAV containing more than four seconds of speech. The benchmark runs models sequentially through the actual live bridge, sweeps native feed sizes of 0.5/1/2/4 seconds, and uses four-second windows for other ASR models. It records load time, first-request latency, sustained real-time factor (RTF), finalization time, transcripts, and estimated capture backlog in JSON and Markdown. The first four seconds are excluded from sustained RTF. Recommendations favor the smallest feed with 20% measured compute headroom in every repetition. Forced aligners and incompatible conversions are reported as skipped. These are local speed measurements, not recognition-accuracy scores or guarantees for longer sessions. `--model FOLDER_NAME` selects one model; `--batches 1 2` narrows the native sweep. Models whose baseline runs all exceed 2 RTF skip further feed sizes by default; `--max-sweep-rtf 0` enables a full sweep. `--resume` continues matching saved results. Add `--install` to update the model selector ratings. See [the sustained streaming check](benchmarks/live-duration.md), [the local benchmark results](benchmarks/realtime.md) and [reproduction instructions](benchmarks/README.md).

The streaming bridge follows the [mlx-audio Voxtral session API](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/voxtral_realtime/streaming.py) and [Qwen streaming interface](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/qwen3_asr/qwen3_asr.py).

### Stream text to another app

Press **v** to toggle the local text service, or add `--serve` (`--serve=9000` for another port):

```sh
./scripts/run-realtime.sh /Volumes/MikeExternal/ai_models/speech-to-text-realtime --serve
# In another terminal:
curl -N http://127.0.0.1:8765/events
```

SSE is available at `http://127.0.0.1:8765/events`, WebSocket at `ws://127.0.0.1:8765/ws`, and the current transcript snapshot at `/transcript`. Both streams carry the same JSON messages for partial text, committed segments, session boundaries and completion. New clients receive the current state immediately. Updates leave the worker before the TUI polls them; slow subscribers do not block inference. The service also works with headless file transcription and stays local to this Mac.

See [the service protocol and client examples](docs/text-stream-service.md) for connection code, reconnect handling, transport choices and retention limits.

### Drag & drop

Drop any audio or video file from Finder onto the terminal window and the start-confirmation dialog opens (drop a folder to browse it). Works via bracketed paste, with a key-burst fallback for terminals that don't support it.

### Keys

| Key | Action |
| --- | --- |
| `Tab` | Switch between file browser and transcript |
| `Enter` | Open directory / transcribe selected file (a confirmation dialog shows the file, model, and export formats before every job) |
| `R` | Start live microphone transcription / stop and export |
| `a` | Select microphone input (built-in, USB, or connected Bluetooth) |
| `v` | Start/stop the local SSE and WebSocket text service |
| `m` | Model picker (lists `.bin`/`.gguf` files and MLX model folders in the models folder) |
| `s` | Settings: default model, models & output folders (via a built-in directory browser), export formats, model management, diarization, split mode, language, Hugging Face token (persisted) |
| `d` | Cycle the diarization strategy (off → auto → tinydiarize → embeddings → pyannote) |
| `p` | Set the known number of speakers (shown while a clustering diarization strategy — embeddings/pyannote — is selected) |
| `n` | Name the detected speakers (voice sample per speaker, re-exports on close; shown while diarization is on) |
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

Diarization is a pluggable strategy, cycled with `d` (or in settings; persisted): **Off → Auto → TinyDiarize → Speaker embeddings → Pyannote community-1**. `Auto` picks the recommended strategy for the model in use — a tdrz model brings its own turn tokens, everything else gets the embedding pipeline. Whatever a strategy still needs is spelled out — with sizes — in the status line when you select it, and again in the pre-transcription confirmation dialog; nothing downloads silently.

- **TinyDiarize** — whisper.cpp's native [tinydiarize](https://github.com/akashmjn/tinydiarize) turn detection: transcribes with the `ggml-small.en-tdrz.bin` model (488 MB, English only, listed in Model management's *diarization* section — Enter downloads it, or selects the TinyDiarize strategy once installed). Detected turns alternate **Speaker A / Speaker B** — the right assumption for two-person conversations. A/B are consistent but arbitrary (A = whoever speaks first), missed turns merge speakers, and turn detection is tuned for real conversational speech — synthetic TTS audio often yields no turns.
- **Speaker embeddings** — engine-agnostic, works with **any** model (whisper.cpp and MLX alike) and any number of speakers: pyannote-style segmentation plus speaker-embedding clustering, run fully offline by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — no PyTorch, no account. It runs as a post-pass in the worker: the engine transcribes as usual, then the diarizer decodes the same audio, finds speaker turns, and labels each segment by overlap. Self-provisioning on first use (like the MLX runtime): the sherpa-onnx wheel pip-installs into the app venv (~25 MB) and the active ONNX models download into `<models>/diarization/` with live progress. Cancel or failure downgrades gracefully — the finished transcript is kept, just unlabeled.
- **Pyannote community-1** — the open-accuracy SOTA (`pyannote/speaker-diarization-community-1` via [pyannote.audio](https://github.com/pyannote/pyannote-audio) 4.x), for when label quality matters more than footprint. Same engine-agnostic post-pass, honors the known speaker count, runs on MPS when available. Costs the others avoid: a one-time ~2 GB PyTorch install into the app venv, and the model is **gated** — accept its terms at [hf.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and store your token in Settings → *HF token* (`hf auth login` or `HF_TOKEN` work too); both are reported as requirements before anything is set up, and the pipeline weights land in the standard HF cache on first run. Starting a pyannote job without a token opens the consent page in your browser instead of failing later — the same happens when any gated hub download is refused with a 401/403.
- Models that label speakers natively (some MLX models emit `speaker_id`) keep their own labels; the post-pass steps aside.

**Diarization model catalog** — Model management (`s` → Model management) has a *diarization* section headed by the tdrz whisper build (Enter downloads it, or makes TinyDiarize the strategy once installed) followed by the interchangeable pipeline components, downloaded into `<models>/diarization/`: segmentation (pyannote segmentation-3.0 6 MB — default; Rev reverb-diarization-v1 10 MB, English, non-commercial license) and speaker embeddings (NeMo TitaNet-S 40 MB — default; TitaNet-L 101 MB higher quality; WeSpeaker CAM++ 29 MB; 3D-Speaker CAM++ zh+en 28 MB for Chinese/English). Enter downloads a missing component or makes an installed one the active choice for its role (✓); Del removes it. A freshly downloaded component becomes active automatically.

**Speaker count** — press `p` (in the key bar while a clustering strategy — embeddings/pyannote — is selected): auto-detect by default, or enter the known number of speakers (1–26) to pin the clustering to exactly that count — when you know it, labels get noticeably better. Headless: `--speakers N`.

**Naming speakers** — after a diarized transcription, press `n`: every detected speaker is listed with a voice sample (their longest utterance, `p` plays it via afplay), and Enter assigns a real name — George, Marco, Sarah. Names replace the anonymous letters in the TUI and in every export (with a `speakers: Speaker A = George, …` frontmatter line in `llm.md`); closing the dialog re-exports automatically if anything changed.

Labels flow into the TUI (colored per speaker), `llm.md` (speaker-prefixed paragraphs and honest `diarization:` metadata naming the method for the LLM), JSON, SRT and TXT. Headless: `--diarize [off|auto|tdrz|embedding]` (bare `--diarize` = auto).

## Long audio & split mode

Settings → **Split mode** decides how long recordings are fed to the engine:

- **auto** (default) — the whole file goes to whisper.cpp in one call; it windows 30 s at a time internally and seeks back to the last completed segment between windows, so words are never cut. Right for whisper models of any length.
- **silence** — the app chunks client-side at ~60 s, placing each cut at the quietest speech pause (25 ms RMS scan) within the 15 s before the target boundary. If speech is continuous, it falls back to a 0.5 s overlap and de-duplicates segments at the seam.
- **fixed** — plain 60 s chunks with the 0.5 s overlap + dedup, no pause hunting.

All chunking happens on the decoded 16 kHz f32 samples in memory — nothing is re-encoded, so splitting never costs quality. Timestamps are re-based so exports read as one continuous transcript, and diarization speaker labels carry across chunk boundaries. Engine capabilities are declared per engine (`split::EngineCaps`): whisper.cpp accepts unlimited length, and a future engine with a hard input window will force chunking automatically regardless of mode. Split mode applies to the whisper.cpp engine only — mlx-audio ingests the whole file and windows internally, so the MLX engine bypasses the planner.

## Language

Press `s` → **Input language** to choose **Auto, English, Spanish, German, Italian, or French** with `↑↓` and `Enter` (`Esc` cancels). The selection is saved and used for subsequent file and microphone transcriptions, including after switching models or restarting the app. Auto lets the model detect the input language; a specific language skips detection on models that support it. Whisper receives the language code and Qwen receives the full language name. English-only models (`.en`) always use English; native Voxtral streaming continues to detect its own language because its API has no language override. Changing this setting during a transcription applies to the next job or microphone session.

For other languages, use the `i` shortcut to enter an ISO 639-1 code, or pass `-l <code>`. CLI language arguments override the saved preference for that run; `-l auto` restores automatic detection. Headless transcription also uses the saved preference when no CLI language is supplied.

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
cargo test                    # unit tests, no model downloads
cargo test -- --ignored       # 3 transcription e2e tests (real model, Metal, ffmpeg,
                              # diarization) + 2 live Hugging Face tests (single-file
                              # download, whole-folder MLX download with manifest)
```

Live checks (no microphone or model downloads for the unit tests):

```sh
python3 -m unittest discover -s tests -p test_realtime.py
# Optional real-model smoke test; WAV must contain speech, PCM16 mono, 16 kHz.
# Use an interpreter with mlx-audio installed (e.g. the app-managed venv).
python scripts/verify-realtime.py --model /path/to/model --audio speech.wav
```

Live tests cover signal levels, continuous resampling and final audio tails, silence windows, partial text and both MLX output shapes, narrow terminal rendering, follow/pause/resume, cancellation/error state and exactly-once export of committed text. Physical microphone permission and device behavior require a manual recording session.

Unit tests cover audio decode/downmix/resampling (incl. the WAV writer round trip), model discovery (files and MLX folders, manifest integrity), config parsing and platform defaults, drop-path parsing (quotes, backslash escapes, `file://` URLs), the app state machine including auto-export, the directory picker, the model-management modal (search input, navigation, download events, variant folders, platform gating), Hub API response parsing (file listing, variant detection), mlx-audio JSON output mapping, and every export format. The e2e tests run the actual binary headless against the demo WAV and MP4. Live MLX inference needs mlx-audio and a downloaded model, so it stays a manual test.

## Architecture

The crate is a library (`transcribe_stt`) plus a thin binary: `src/main.rs` only parses the CLI (clap), dispatches through `lib::run`, and works around the ggml Metal atexit hang with `libc::_exit`.

- `src/cli.rs` / `src/headless.rs` / `src/tui.rs` — the three entry paths: clap argument parsing, `--headless` transcription + `--download-test-model` (stderr progress, stdout segments), and the terminal lifecycle + render/event loop
- `src/app.rs` + `src/app/` — application state and update logic (Elm-style). `App` is composed of per-concern sub-state, each with its logic and key handling beside it:
  - `browser.rs` (`FileBrowser` — directory listing/navigation), `library.rs` (`ModelLibrary` + `ModelPicker`), `settings.rs` (`SettingsUi`, `SettingsRow`, directory picker, move-models prompt), `transcript.rs` (`TranscriptState` — segments, scroll/follow), `hub_state.rs` (`HubState` — the Model management modal), `events.rs` (worker-event handling incl. auto-export), `drop.rs` (drag-and-drop path parsing + the `DropDetector` key-burst fallback), `keys.rs` (the modal-priority key router)
- `src/ui.rs` + `src/ui/` — rendering, one file per widget/modal, all reading `&App` (scroll clamping runs in the update phase, never during draw); `theme.rs` holds the shared colors/spinner/highlight style, `layout.rs` the screen regions
- `src/transcribe.rs` + `src/transcribe/` — `worker.rs`: the worker thread executing jobs; `backend.rs`: the common `Engine` interface every backend implements, plus the model-shape routing (`Backends::for_model` — a directory routes to MLX, a file to whisper); `backend/whisper_metal.rs`: whisper.cpp via whisper-rs, resident context between jobs, cancellation via whisper's abort callback; `backend/mlx.rs`: the mlx-audio subprocess runner (runtime probe, temp-WAV hand-off, JSON parsing, kill-on-cancel). New backends slot in as new submodules behind the same trait
- `src/stream_service.rs` + `src/stream_service/` — optional loopback HTTP/SSE/WebSocket runtime, shared transcript snapshots and bounded subscriptions; the worker publishes before UI consumption
- `src/audio/capture.rs` / `src/transcribe/realtime.rs` / `src/transcribe/realtime.py` — independent CPAL microphone capture, continuous sinc resampling, bounded buffering, persistent MLX process and JSON-lines streaming protocol; `src/app/realtime.rs` / `src/ui/realtime.rs` own the recording state and waveform monitor
- `src/audio.rs` — symphonia decode (audio) → mono downmix → rubato sinc resample to 16 kHz; video and unsupported codecs go through ffmpeg, which streams raw 16 kHz mono f32 over stdout (no temp files) with byte-accurate extraction progress against the ffprobe duration; both paths are cancellable mid-decode; plus the 16 kHz WAV writer used to hand audio to subprocess engines
- `src/hub.rs` + `src/hub/` — Model management backend: `api.rs` (Hugging Face search, repo file listing, directory-variant detection, serde-typed responses) and `download.rs` (streaming `.part` downloads with retry; whole-folder downloads with aggregated progress, integrity manifest, and missing-file repair), each on worker threads; the suggested list is embedded from `assets/suggested_models.json`
- `src/split.rs` — long-audio chunk planner (engine capability descriptor, silence-aware cut placement, overlap fallback)
- `src/export.rs` — all transcript output formats (llm.md, JSON via serde_json, txt, srt)
- `src/config.rs` — persisted settings (`~/.config/transcribe-stt/config.toml`), strict TOML via serde with a lenient fallback parser for legacy hand-edited files
- `src/models.rs` / `src/format.rs` / `src/stats.rs` — local model scanning (files and MLX folders, integrity-manifest checks) + default-model selection, shared time/size formatting, live process stats
- whisper.cpp log output is routed through the `log` crate so it can't corrupt the TUI
