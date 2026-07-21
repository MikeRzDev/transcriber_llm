# 01 — Core: workspace split & reusable Rust API

The foundation every binding builds on. Two parts: (A) split the single crate into a
workspace so the core is a publishable library, (B) evolve the core API to be
binding-friendly (callback sink, blocking convenience API, typed errors, feature gating,
serde wire types).

## A. Workspace split

### Crate layout

```
crates/transcribe-core/           # package name "transcribe-stt-core" (verify availability
  Cargo.toml                      #   with `cargo search`; fallbacks: "transcriber-core")
  assets/suggested_models.json    # moved here so include_str! and `cargo package` both work
  examples/transcribe_file.rs     # NEW minimal blocking-API example
  src/
    lib.rs                        # module list + curated public re-exports
    api.rs                        # NEW: Session, TranscribeOptions, TranscriptionResult, CancelToken
    error.rs                      # NEW: Error (thiserror) + ErrorCode
    sink.rs                       # NEW: Sink<T> trait + adapters
    transcribe.rs  transcribe/worker.rs  transcribe/backend.rs
    transcribe/backend/whisper.rs # git mv from whisper_metal.rs (no longer Metal-specific)
    transcribe/backend/mlx.rs
    audio.rs  split.rs  models.rs  export.rs  config.rs  format.rs
    hub.rs  hub/api.rs  hub/download.rs

crates/transcribe-tui/            # package name stays "transcribe-stt" → binary name, install.sh,
  Cargo.toml                      #   and make-app.sh keep working; publish = false for now
  tests/e2e.rs                    # moved; resolve model/sample paths via env!("CARGO_MANIFEST_DIR")/../..
  src/
    main.rs                       # keeps libc::_exit — a *binary-only* policy, never in core
    lib.rs                        # run() dispatch (download-test-model / headless / tui)
    cli.rs  tui.rs  headless.rs  stats.rs
    app.rs + app/*  ui.rs + ui/*
```

Module assignment rationale (import-verified): core gets everything with no
ratatui/crossterm/clap/sysinfo dependency. `format.rs` is core because `export.rs` uses its
`srt_time`/`llm_time`. `headless.rs` stays with the TUI binary (it implements `--headless`
of the installed executable and depends on `cli::Args`) but is **rewritten in step 5** as
the first consumer of the new blocking API; a trimmed copy becomes
`crates/transcribe-core/examples/transcribe_file.rs`.

Root `Cargo.toml` becomes a virtual workspace: `members = ["crates/*", "bindings/python",
"bindings/node"]`, `exclude = ["bindings/go", "bindings/java"]`, `resolver = "2"`,
`[workspace.package]` carrying `version`, `edition`, `license = "MIT OR Apache-2.0"`,
`repository`; `[workspace.dependencies]` for shared deps. `Cargo.lock` stays at root.

Core publish metadata: `description`, `license`, `repository`, `readme`,
`keywords = ["whisper", "speech-to-text", "transcription", "mlx"]`, `categories`.

### Git history

Two commits: (1) **pure `git mv`** of every file, unmodified; (2) mechanical edits
(manifests, `crate::` → `transcribe_stt_core::` in TUI files, `include_str!` path already
correct once `assets/` sits in the core crate, e2e paths, `scripts/install.sh` →
`cargo install --path crates/transcribe-tui`). Rename detection and `git log --follow`
survive both.

## B. API evolution

### Sink abstraction (`src/sink.rs`)

The hard-coded `std::sync::mpsc::Sender<Event>` becomes a trait so FFI bridges can be
callbacks instead of channels:

```rust
pub trait Sink<T>: Send + Sync + 'static {
    fn emit(&self, item: T);
}
impl<T: Send + 'static> Sink<T> for std::sync::mpsc::Sender<T> { /* ignore send error */ }
impl<T, F: Fn(T) + Send + Sync + 'static> Sink<T> for F { /* call self */ }
```

- `worker::spawn` (today `src/transcribe/worker.rs:57`) becomes
  `pub fn spawn(sink: impl Sink<Event>) -> Transcriber`, internally storing
  `Arc<dyn Sink<Event>>`. The TUI and headless keep compiling **unchanged** — their
  `Sender` implements the trait.
- `Engine::run` (`src/transcribe/backend.rs`) changes `events: &Sender<Event>` →
  `events: &Arc<dyn Sink<Event>>`. It must be `Arc`, not a generic parameter: `Engine` is
  used as `dyn`, and the whisper progress/segment callbacks clone the sender into
  `'static` closures — they clone the `Arc` instead. Same substitution in `mlx.rs`.
- `hub::{search, list_files, download, download_dir}` take `impl Sink<HubEvent>` the same
  way (TUI's `Sender<HubEvent>` still works).
- **Preserve verbatim** the whisper-rs 0.16 abort-callback double-box workaround at
  `src/transcribe/backend/whisper_metal.rs:204-212` — it papers over an FFI layout bug in
  `set_abort_callback_safe` and is load-bearing.

### Blocking convenience API (`src/api.rs`)

Built **on top of the existing worker** (wrap, don't bypass): reuses model residency,
cancellation, both engines, and the event protocol.

```rust
#[derive(Clone)]
pub struct CancelToken(Arc<AtomicBool>);        // .cancel(), .is_cancelled(); Clone+Send+Sync

#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TranscribeOptions {                  // ::new(model) + with_* builder setters
    pub model: PathBuf,
    pub language: Option<String>,               // ISO 639-1 hint; None = auto-detect
    pub diarize: bool,
    pub split_mode: SplitMode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub segments: Vec<Segment>,
    pub language: Option<String>,
    pub duration_secs: f32,                     // audio duration
    pub elapsed_secs: f32,                      // wall-clock transcription time
}

pub struct Session { /* owns Transcriber + a private channel */ }
impl Session {
    pub fn new() -> Session;                    // model stays resident across calls
    pub fn transcribe(&mut self, audio: &Path, opts: &TranscribeOptions)
        -> Result<TranscriptionResult, Error>;
    pub fn transcribe_with(&mut self, audio: &Path, opts: &TranscribeOptions,
        observer: impl FnMut(&Event)) -> Result<TranscriptionResult, Error>;
    pub fn cancel_token(&self) -> CancelToken;  // wraps the worker's cancel Arc
    pub fn unload(&mut self);                   // drop resident model, keep session
    pub fn shutdown(&mut self);                 // join worker, drop WhisperContext; also in Drop
}

/// One-shot headline API: Session::new → transcribe → shutdown.
/// Per-call model load; documented as such (use Session for batches).
pub fn transcribe(audio: &Path, opts: &TranscribeOptions)
    -> Result<TranscriptionResult, Error>;
```

Aggregation loop, lifted from `run_headless_loop` (`src/headless.rs:125`):

- collect `Event::Segment`s;
- `Event::SegmentsFinal` **replaces** the collected list (the diarized re-issue after a
  tinydiarize run — streamed segments carry no speaker);
- `Event::Done { elapsed_secs, audio_secs, language }` → build the result;
- `Event::Cancelled` → `Err(Error::Cancelled)`;
- `Event::Error { code, message }` → typed `Error`;
- **every** event is forwarded to the observer first, so callers still see streaming
  segments/progress. Callback and cancel are separate parameters — `TranscribeOptions`
  stays pure data (serde-able, FFI-flat).

Also: `impl Drop for Transcriber` calls the (idempotent — `jobs`/`handle` are already
`Option`s) `shutdown()`, so hosts that forget explicit shutdown still join the worker
before process exit.

### Typed errors (`src/error.rs`)

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("model not found: {0}")]      ModelNotFound(PathBuf),
    #[error("audio decode failed: {0}")]  AudioDecode(String),
    #[error("engine failure: {0}")]       Engine(String),
    #[error("cancelled")]                 Cancelled,
    #[error("MLX runtime missing: {0}")]  MlxRuntimeMissing(String),
    #[error("unsupported platform: {0}")] UnsupportedPlatform(String),
    #[error("network failure: {0}")]      Network(String),
    #[error("invalid argument: {0}")]     InvalidArg(String),
    #[error(transparent)]                 Io(#[from] std::io::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorCode { ModelNotFound, AudioDecode, Engine, Cancelled, MlxRuntimeMissing,
                     UnsupportedPlatform, Network, Io, InvalidArg }
impl Error { pub fn code(&self) -> ErrorCode; }
```

The snake_case serde names are the **stable wire codes** used identically in the C ABI,
Python exceptions, Node errors, Go, Java, and MCP: `model_not_found`, `audio_decode`,
`engine`, `cancelled`, `mlx_runtime_missing`, `unsupported_platform`, `network`, `io`,
`invalid_arg`. The FFI layer adds one extra code of its own: `panic`
([02-c-abi](02-c-abi.md)).

- Internals **keep anyhow** (minimal churn). Typed `Error`s are constructed only at
  categorized failure points and wrapped into `anyhow::Error`: the MLX platform `ensure!`
  (`src/transcribe/backend/mlx.rs:180`) → `UnsupportedPlatform`; venv/pip provisioning
  failures → `MlxRuntimeMissing`; model-file open → `ModelNotFound`; decode failures in
  `audio::load_media_with` → `AudioDecode`; hub HTTP failures → `Network`.
- The worker boundary changes `Event::Error(String)` → `Event::Error { code: ErrorCode,
  message: String }`, where `code = e.downcast_ref::<Error>().map(Error::code)
  .unwrap_or(ErrorCode::Engine)` and message keeps the full anyhow chain. Clonable,
  serde-able, flat. `Session` reconstructs a typed `Error` from it.
- `HubEvent::Failed { file, error }` gains `code: ErrorCode` the same way
  (`Network`, `Io`, `Cancelled`).
- TUI churn: exactly two match arms gain a field (`app/events.rs` `Event::Error`, hub
  state `Failed`).

### Feature gating

Verified: whisper-rs 0.16's default feature set is **empty** (plain CPU build), with
`metal`, `cuda`, `vulkan`, `coreml`, `hipblas`, `openblas`, `openmp`, `log_backend`
available.

```toml
# crates/transcribe-core/Cargo.toml
[dependencies]
whisper-rs = { version = "0.16", features = ["log_backend"] }   # log_backend always on

[features]
default  = []                       # CPU everywhere
metal    = ["whisper-rs/metal"]
coreml   = ["whisper-rs/coreml"]
cuda     = ["whisper-rs/cuda"]
vulkan   = ["whisper-rs/vulkan"]
openblas = ["whisper-rs/openblas"]
```

- **Default must be empty** (cargo features cannot be per-platform). The TUI turns Metal on
  only for Apple Silicon via a target-specific dependency — with resolver 2, target-gated
  features do not unify into other targets' builds:

  ```toml
  # crates/transcribe-tui/Cargo.toml
  [dependencies]
  transcribe-stt-core = { path = "../transcribe-core" }
  [target.'cfg(all(target_os = "macos", target_arch = "aarch64"))'.dependencies]
  transcribe-stt-core = { path = "../transcribe-core", features = ["metal"] }
  ```
- **MLX module: compiles everywhere, runtime-gated** (its current design — pure
  `std::process` code with a `cfg!` check). This keeps `Backends` uniform and lets a
  Linux-built MCP server return a clean `unsupported_platform` error instead of a link
  failure. Additionally the engine gains an `mlx` mode setting (`auto` | `off` | `require`)
  plumbed from `Session`/engine config + env `TRANSCRIBE_STT_MLX=off`; **venv
  auto-provisioning (pip install) never runs unless mode is `auto`/`require`** — a library
  must not surprise its host with a multi-minute pip install (risk #4 in
  [00-overview](00-overview.md)).
- Rename `whisper_metal.rs` → `whisper.rs` (git mv). The "uploading weights to Metal
  (GPU)" log line and `use_gpu(true)` must reflect the actually-compiled accel
  (`cfg!(feature = "metal")` etc. — `use_gpu(true)` is harmless on CPU builds but the log
  must not lie). Rework `hub::metal_available()`/`backend_label()` into a feature-driven
  `accel_label()` (compile-time accel + the runtime Apple-Silicon check for the MLX badge).

### Lifecycle contract

The ggml/Metal atexit hazard, stated as the core's contract (rustdoc on `Session` and
`Transcriber`):

1. `shutdown()` (and `Drop`) deterministically joins the worker and drops the
   `WhisperContext`. **No GPU context may be alive at process exit.**
2. Core never installs atexit handlers and never calls `_exit`.
3. ggml's own static destructors are outside core's control; binaries (TUI `src/main.rs:19`,
   MCP server) may apply the `libc::_exit`-after-clean-drop policy. Binding crates for hosts
   that still hang at exit may do the same at their layer — the library itself never does.
4. `whisper_rs::install_logging_hooks()` (today called on every spawn,
   `src/transcribe/worker.rs:65`; it is process-global) moves behind `std::sync::Once` —
   multiple concurrent `Session`s per process become legal and documented.

### Serde & API stability

- Add `Serialize`/`Deserialize` (+ `Clone` where missing): `Segment`, `Event`,
  `TranscriptionResult`, `TranscribeOptions`, `ErrorCode`, `SplitMode`, `ExportFormat`,
  `HubEvent`, `RepoHit`, `HubFile`, `SuggestedModel`.
- `Event` is **internally tagged, snake_case**: `#[serde(tag = "type",
  rename_all = "snake_case")]` — producing e.g. `{"type":"segment", "start_ms":0,
  "end_ms":1500, "text":"…", "speaker":0}`, `{"type":"progress","pct":42}`. This JSON **is**
  the FFI wire format ([02-c-abi](02-c-abi.md#events)); tuple variants are restructured to
  named-field variants where needed (`LoadProgress(i32)` → `{ pct: i32 }`, etc.). Negative
  `pct` keeps today's meaning: indeterminate.
- `#[non_exhaustive]` on `Event`, `HubEvent`, `Error`, `ErrorCode`, `TranscribeOptions`.
  TUI gains one wildcard arm per match; the wildcard logs at debug level so future events
  aren't silently dropped.
- **Stays private**: `worker::WorkerMsg`, `Backends`, the `Engine` trait, split-mode
  internals, `hub::api` internals.
- Public surface via curated `lib.rs` re-exports: `transcribe()`, `Session`,
  `TranscribeOptions`, `TranscriptionResult`, `CancelToken`, `Error`, `ErrorCode`, `Event`,
  `Segment`, `Sink`, `Transcriber`, `spawn`, `SplitMode`, `ExportFormat`, `TranscriptDoc`,
  and the `config`, `models`, `hub`, `audio` (extension lists + `DecodedAudio`), `format`
  modules.
- `#![warn(missing_docs)]`; every public item documented. Publish gates:
  `cargo doc --no-deps`, `cargo publish --dry-run`, `cargo search` name check.
- schemars derives (behind a `schemas` feature or an xtask) generate `schemas/*.json` for
  `Event`, `TranscribeOptions`, `TranscriptionResult`, `HubEvent`, capability/config
  objects — golden files consumed by every binding's tests.

## Migration steps (each step verified before the next)

| # | Step | Verification |
|---|---|---|
| 1 | Baseline: build, test, clippy; run TUI once; `--headless` with the tiny test model | all green, recorded |
| 2 | Workspace split (2 commits: pure `git mv`, then mechanical fixes). Core temporarily keeps `features = ["metal", "log_backend"]` so behavior is bit-identical | `cargo build/test --workspace`; TUI runs; `--headless` runs; `target/release/transcribe-stt` still produced for `make-app.sh` |
| 3 | `Sink<T>` abstraction (core-only edit) | TUI/headless recompile unchanged; existing `unload_is_safe_with_no_model_loaded` passes; new closure-sink test |
| 4 | Typed errors (`error.rs`, `Event::Error { code, message }`, `HubEvent` code, engine tagging, worker downcast) | TUI error display; `--headless` with a bad model path → `model_not_found` |
| 5 | Blocking API (`api.rs`; `Drop for Transcriber`; `Once` for logging hooks). **Rewrite `headless.rs` on `Session::transcribe_with`** — the proof the API is sufficient. Add `examples/transcribe_file.rs` + an `#[ignore]`d integration test using the tiny model | e2e headless test; example runs |
| 6 | Feature gating (features table, TUI target-specific dep, `whisper.rs` rename, `accel_label`, MLX mode setting) | `cargo build --workspace` (metal path unchanged); `cargo check -p transcribe-stt-core --no-default-features` (CPU whisper.cpp build proof); Linux/Windows proof deferred to CI. Never `--all-features` |
| 7 | Serde derives, `#[non_exhaustive]`, schemas generation, missing_docs, core README, LICENSE files, publish dry-run | full re-verification: build, tests, TUI session, headless, example |

## Reuse, not rewrite

- `src/headless.rs:125` drain loop → becomes `Session`'s aggregation loop (then headless
  consumes `Session`).
- `src/export.rs` `TranscriptDoc` + string renderers (`llm_markdown()`, `json()`,
  `plain_text()`, `srt()`) → already ideal for bindings (bytes without filesystem); only
  serde derives added.
- `src/transcribe/worker.rs` residency/cancel/shutdown machinery → wrapped, not replaced.
- `src/hub.rs` channel pattern → same `Sink<HubEvent>` generalization; download logic
  (Range resume, pause, `.part`, retry, dir manifests) untouched.
- `src/config.rs` platform dirs + `TRANSCRIBE_STT_MODELS`/`TRANSCRIBE_STT_CONFIG` env
  overrides → unchanged; the MCP server reuses them.
