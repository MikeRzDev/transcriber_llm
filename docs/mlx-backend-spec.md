# MLX backend — implementation spec

Status: **implemented** (2026-07-21). Deviations from this draft, found
during build against the real mlx-audio source and Hugging Face repos:

- **JSON contract**: mlx-audio writes `{output_path}.json` (extension
  appended); segment keys are `start`/`end` (not `start_time`), in two
  shapes — `segments` (Whisper-family) or `sentences` (Parakeet-family) —
  and a model returning no segments falls back to a plain `.txt` file.
  The parser handles all three (plus legacy `start_time` keys).
- **D1 resolved**: mlx-audio documents local model paths — `--model`
  gets the downloaded folder. D2 = cold subprocess, D3 = temp WAV,
  D4 = spinner, D5 = hint-only (`--language` passed only when set),
  D6 = detect-and-instruct, as drafted.
- **Canary**: `mlx-community/canary-1b-v2` does not exist; the curated
  list ships Qwen3-ASR (`mlx-community/Qwen3-ASR-1.7B-8bit`) alongside
  Parakeet instead. Canary works via search (mlx-audio's README endorses
  `Mediform/canary-1b-v2-mlx-q8`).
- **Detection**: MLX dirs are `config.json` + `*.safetensors` only —
  every mlx-audio loader globs safetensors; legacy npz Whisper-MLX
  conversions cannot be run, so they are not offered (quantized
  safetensors whispers like `mlx-community/whisper-tiny-4bit` work).
- **D6 upgraded**: the app self-provisions the runtime — if no Python
  with mlx_audio is found, the first MLX job creates an app-managed venv
  (`~/Library/Application Support/transcribe-stt/mlx-venv`) and pip
  installs mlx-audio into it (cancellable, spinner + status);
  `scripts/install.sh` pre-installs the same venv.
- **Beyond the draft**: engines live behind a common `Engine` trait in
  `transcribe/backend/` (`whisper_metal`, `mlx`); repo subfolder
  *variants* (own `config.json` + weights) are listed and downloaded as
  individual models; every folder download writes a `.manifest.json`
  integrity manifest — a folder failing it leaves the library and a
  re-download fetches only the missing files.
- The `engine`/`kind` JSON keys sketched in §6 stayed `format` — the
  existing key already carries the engine distinction.

---

Original draft below.
Scope: add a second transcription engine so the app can run Apple-Silicon **MLX**
speech-to-text models (NVIDIA Parakeet, NVIDIA Canary, and other families),
acquired through the existing model-management (Hugging Face) flow. The
whisper.cpp path is unchanged and stays the default.

---

## 1. Goal

- Run MLX ASR models locally on Apple Silicon, alongside the existing
  whisper.cpp engine.
- Cover **multiple model families with one backend**, not a per-model tool:
  Parakeet-TDT and Canary today, others (Whisper-MLX, Qwen3-ASR, Voxtral)
  for free later.
- Obtain MLX models **only through model management** (the hub) — no manual
  placement assumed.
- Reuse the existing engine-agnostic types (`Job`, `Segment`, `Event`) and audio
  pipeline; keep the whisper.cpp experience identical.

## 2. Runtime: `mlx-audio` as the MLX engine

The MLX engine is a **subprocess call to [`mlx-audio`](https://github.com/Blaizzy/mlx-audio)**
(Blaizzy/mlx-audio), a single MLX library whose STT module runs Whisper,
Parakeet, **Canary**, Qwen3-ASR, Voxtral and more. One runner covers every
family, so "support other models like canary" is a data change (add a suggested
model), not new engine code.

Why not `parakeet-mlx`? It only runs Parakeet — it could never run Canary.
`mlx-audio` is the generic tool.

Why subprocess, not native? MLX is a Python/C++/Swift framework; there is no
mature Rust MLX ASR. The app already shells out to `ffmpeg`, so a managed
subprocess is a precedented, honest boundary.

**CLI contract** (verified):
```
python -m mlx_audio.stt.generate \
    --model  <local_model_dir | hf_repo_id> \
    --audio  <audio_file> \
    --output-path <basename> \
    --format json \
    --verbose
```
- Writes a **JSON file** (no stdout streaming), fields per segment include
  `start_time`, `end_time`, `text`, `speaker_id` (seconds, float).
- `pip install mlx-audio`, **Python ≥ 3.10**, **Apple Silicon only**.
- `--model` documented with HF repo ids; local-dir support is **unverified**
  (see Open Decision D1).

## 3. Models (via model management)

MLX models are **multi-file directories** downloaded from HF, e.g.
`mlx-community/parakeet-tdt-0.6b-v3` (~2.5 GB):

```
parakeet-tdt-0.6b-v3/
  config.json
  model.safetensors        # ~2.5 GB
  tokenizer.model
  tokenizer.vocab
  vocab.txt
```

Canary is the same shape (`mlx-community/canary-1b-v2` or the id mlx-audio's
Canary README recommends — confirm during build). This is the crux of the
model-management work: **everything today assumes a single `.bin`/`.gguf`
file**, and MLX models are directories.

## 4. Architecture

### 4.1 Engine abstraction (`src/transcribe/`)

Introduce an engine seam and move the current whisper logic behind it. Two
concrete engines:

- `WhisperEngine` — today's `run.rs` logic (whisper-rs, tdrz, chunking).
- `MlxEngine` — spawns `mlx-audio`, parses JSON → `Segment`s.

Sketch:
```rust
trait Engine {
    // Ensure the model named by `job.model` is resident/ready, then stream
    // Events (Segment/Progress/Done) over `events`, honouring `cancel`.
    fn run(&mut self, job: &Job, events: &Sender<Event>, cancel: &Arc<AtomicBool>)
        -> anyhow::Result<()>;
}
```
`worker.rs` holds `Option<(PathBuf, Box<dyn Engine>)>` instead of
`Option<(PathBuf, WhisperContext)>`; lazy-load / keep-resident / unload
semantics are preserved. whisper.cpp keeps its resident `WhisperContext`; the
MLX engine is effectively stateless between jobs (the model lives on disk and
mlx-audio loads it per run — see Open Decision D2 on warm processes).

### 4.2 Engine selection — by model *shape*

`Job.model` is a path. Decide the engine from what's there:

| Model on disk | Engine |
|---|---|
| single file `*.bin` / `*.gguf` | whisper.cpp |
| directory containing `config.json` + `*.safetensors` | MLX |

This keeps selection data-driven; no per-model registry in code.

### 4.3 MLX job flow (`MlxEngine::run`)

1. **Preflight**: is `mlx-audio` runnable (`python -m mlx_audio --version` or an
   import probe) and are we on Apple Silicon? If not → `Event::Error` with the
   exact install hint. (Detection cached.)
2. **Audio in**: reuse `audio::load_media` → 16 kHz mono f32, write a temp
   `.wav`, pass it as `--audio` (Open Decision D3). This makes video / exotic
   codecs "just work" via the pipeline that already handles them.
3. **Spawn** `python -m mlx_audio.stt.generate --model <dir> --audio <tmp.wav>
   --output-path <tmp> --format json`. Emit `LoadingModel` / `Decoding`;
   progress is **indeterminate** (spinner) unless we parse `--verbose` tqdm on
   stderr (Open Decision D4).
4. **Cancel**: `cancel` flag → kill the child, clean temp files, `Event::Cancelled`.
5. **Parse** the JSON file → map each segment: `start_ms = round(start_time*1000)`,
   `end_ms = round(end_time*1000)`, `text`, `speaker = speaker_id` (if present).
   Emit `Event::Segment` per row (or `SegmentsFinal`), then `Event::Done`.

### 4.4 EngineCaps / chunking

mlx-audio ingests a whole file and chunks internally, so the MLX path does
**not** use `split::plan_chunks`. `EngineCaps` stays a whisper-only concern; the
MLX engine bypasses it. (If a family needs a hard window later, add
`EngineCaps::mlx()` then.)

## 5. Model management for directory models (the ripple)

This is the largest slice and it touches code from recent work.

- **`models.rs`**
  - `ModelEntry` gains a kind: `File(PathBuf)` (whisper) vs `Dir(PathBuf)` (MLX).
    Or keep `ModelFile` and add `is_dir` + summed `size_bytes`.
  - `scan_models` also yields **directories** that look like MLX models
    (`config.json` + a `*.safetensors`). `size_bytes` = sum of the dir's files.
  - `is_english_only` / `is_tdrz` return false for dir models.
- **hub download** (`hub/download.rs`, `hub_state.rs`)
  - New "download a whole model": list the repo's files (HF API) and stream each
    into `<models_dir>/<repo-basename>/`. Progress = bytes across all files.
    Resume/pause/cancel operate on the set (cancel removes the partial dir).
- **delete** (`hub_state.rs`)
  - Deleting a dir model removes the directory (existing confirm dialog; show
    summed size).
- **suggested list** (`assets/suggested_models.json`, `ui/hub.rs`)
  - Entries carry `engine: "mlx"` and a repo; on Apple Silicon they are
    **supported** (downloadable), greyed elsewhere. Rendering already keys the
    checkmark/size on the library scan, so dir models slot in.
- **picker / selection** (`app/library.rs`)
  - `pick_default`, `choose_model`, checkmark all key on name/path — dir models
    work once `scan_models` returns them.

## 6. `suggested_models.json` changes

Replace the two greyed MLX stubs with real, mlx-audio-runnable, directory
models:
```jsonc
{ "name": "parakeet-tdt-0.6b-v3", "repo": "mlx-community/parakeet-tdt-0.6b-v3",
  "engine": "mlx", "kind": "dir", "size": "2.5 GB",
  "note": "NVIDIA Parakeet · MLX · 25 languages · needs mlx-audio" }
{ "name": "canary-1b-v2", "repo": "mlx-community/canary-1b-v2",
  "engine": "mlx", "kind": "dir", "size": "~2 GB",
  "note": "NVIDIA Canary · MLX · ASR+translation · needs mlx-audio" }
```
`SuggestedModel::supported()` becomes: ggml/gguf always; `engine=="mlx"` only on
Apple Silicon **and** when `mlx-audio` is detected.

## 7. UX / edge cases

- **mlx-audio missing** → status + hub info: `pip install mlx-audio` (and note
  Apple-Silicon-only). Downloading the model is still allowed; running it errors
  clearly until the tool is present.
- **Diarization**: the whisper tdrz toggle does not apply. If a family returns
  `speaker_id`, surface it; otherwise no speaker labels.
- **Language**: Parakeet/Canary auto-detect; Canary can translate. Language
  hint mapping is out of scope for v1 (Open Decision D5).
- **Progress** is a spinner for MLX (no fine-grained callback).
- **Model load time**: mlx-audio loads the model each run (cold). Acceptable for
  v1; warm-process reuse is a later optimization (D2).

## 8. Testing

Unit-testable without a Mac/model/network:
- JSON → `Segment` mapping (fixture of a real mlx-audio json).
- Engine selection by model shape (temp dirs/files).
- Dir-model scanning + summed size.
- Multi-file download planning (given a file list).
- `supported()` platform gating.

Not automatable here (manual, on-device):
- Actual MLX inference quality/speed end-to-end.

## 9. Phasing

1. **Engine trait refactor** — whisper.cpp moved behind `Engine`, zero behavior
   change. Fully testable. *(low risk)*
2. **Directory-model model management** — scan/size/download/delete/select for
   dir models. Testable. *(the real cost)*
3. **MLX engine** — subprocess + JSON parse + cancel + preflight. Parse
   unit-tested; live run manual. *(medium)*
4. **Wire suggested MLX models + platform gating + README/docs.** *(low)*

Recommended: land 1 and 2 first (safe, tested), then 3–4.

## 10. Open decisions (need your call)

- **D1 — local model path**: confirm `mlx-audio --model <local_dir>` works
  (vs. only HF ids). If it only takes ids, we point `--model` at the HF id and
  let mlx-audio use its own HF cache — but then "downloaded via model management"
  is cosmetic. **Preference?**
- **D2 — process model**: cold `mlx-audio` per job (simple) vs. a persistent
  warm Python worker (faster, complex). v1 = cold.
- **D3 — audio input**: temp 16 kHz WAV from our pipeline (uniform, handles
  video; needs a tiny WAV writer) vs. hand mlx-audio the source file (simpler,
  but its own decoding/limits). v1 = temp WAV.
- **D4 — progress**: spinner only vs. parse `--verbose` tqdm from stderr.
- **D5 — Canary translation / language hints**: expose or ignore in v1.
- **D6 — dependency install**: detect-and-instruct only vs. offer to run
  `pip install mlx-audio` for the user.

## Sources
- mlx-audio: https://github.com/Blaizzy/mlx-audio
- Canary in mlx-audio: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/canary/README.md
- parakeet-mlx (parakeet-only alt): https://github.com/senstella/parakeet-mlx
- MLX Parakeet model: https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3
