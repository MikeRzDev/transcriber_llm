# 03 — Python binding: `bindings/python`

First-class **PyO3** binding over the Rust core directly (not via the C ABI), packaged
with **maturin**. The crate is a cargo workspace member so it shares the lockfile and core
version.

## Packaging

- PyO3 with `abi3-py39` → one wheel per platform covers all CPython ≥ 3.9.
- PyPI package **`transcribe-stt`**, import module **`transcribe_stt`**
  (verify PyPI name availability before first publish).
- maturin drives cargo, which builds whisper.cpp via cmake — wheels are the supported
  install path; the sdist works but is documented as requiring cmake + a C++ toolchain.
- Wheel matrix v1: `macosx_11_0_arm64` (metal feature), `manylinux_2_28_x86_64`,
  `manylinux_2_28_aarch64`, `win_amd64` (CPU). Distributed wheels use the portable ISA
  baseline (risk #6). CUDA wheels deferred to P3 as a separate `transcribe-stt-cuda`
  package.

## API

```python
import transcribe_stt as ts

class Transcriber:                       # context manager
    def __init__(self, models_dir=None, mlx="auto"): ...   # mlx: "auto"|"off"|"require"
    def transcribe(self, path, *, model=None, language=None, diarize=False,
                   split="auto",
                   on_event=None,        # fn(dict) -> None; full event stream
                   on_segment=None,      # fn(Segment) -> None
                   on_progress=None,     # fn(int) -> None  (-1 = indeterminate)
                   ) -> Transcription: ...
    def unload(self): ...                # drop resident model, keep worker
    def close(self): ...                 # join worker; also on __exit__

class Transcription:
    segments: list[Segment]              # Segment: start_ms, end_ms, text, speaker
    language: str | None
    duration_secs: float
    elapsed_secs: float
    def to_srt(self) -> str: ...         # delegate to core's renderers
    def to_txt(self) -> str: ...
    def to_llm_md(self) -> str: ...
    def to_dict(self) -> dict: ...

# module level
def list_models(dir=None) -> list[ModelInfo]: ...
def search_hub(query) -> list[RepoHit]: ...
def list_hub_files(repo) -> list[HubFile]: ...
def download_model(repo, file=None, dest=None, on_progress=None) -> str: ...
def capabilities() -> dict: ...
```

## Threading, GIL, and signals

- `transcribe()` runs the blocking core call under `py.allow_threads` — the GIL is
  released for the whole job.
- Callbacks are invoked from the worker thread via `Python::with_gil`.
- A callback that **raises** (or an `on_event` that returns `False`) sets the cancel
  token; the original exception is stashed and re-raised from `transcribe()` — user
  exceptions are never swallowed.
- The event pump calls `Python::check_signals` so **Ctrl+C cancels the job cleanly** and
  raises `KeyboardInterrupt`.
- Deadlock rule: `close()` releases the GIL **before** joining the worker (a callback
  waiting for the GIL while `close()` holds it would deadlock).

## Errors

Exception hierarchy mapped from the stable error codes
([01-core-api](01-core-api.md#typed-errors-srcerrorrs)):

```
TranscribeError(Exception)      # .code: str — always the wire code
├── ModelNotFoundError          # model_not_found
├── AudioDecodeError            # audio_decode
├── EngineError                 # engine
├── MlxRuntimeMissingError      # mlx_runtime_missing
├── UnsupportedPlatformError    # unsupported_platform (MLX off Apple Silicon)
├── NetworkError                # network
└── Cancelled                   # cancelled
```

## Interpreter-exit hazard (risk #1)

- The contract is `with Transcriber(...) as t:` or explicit `close()`.
- Best-effort backstops: the module registers one `atexit` hook that closes all live
  `Transcriber`s (tracked in a weakref registry); `__del__` calls `close()` best-effort.
- Documented prominently in the README: an unclosed Transcriber can hang interpreter
  shutdown in ggml's static destructors.

## Tests (run against the built wheel in CI)

pytest suite: transcribe a fixture with the tiny model; segment/progress callbacks fire;
cancellation via callback-exception and via Ctrl+C simulation; error mapping
(`model_not_found` on a bogus path); golden-file check of `to_dict()` against
`schemas/`; `with`-block and atexit cleanup smoke.
