# 00 — Overview: transcribe-stt as a multi-language library

Status: **specification** — no code in this directory's scope has been implemented yet.
Companion specs: [01-core-api](01-core-api.md), [02-c-abi](02-c-abi.md), [03-python](03-python.md),
[04-node](04-node.md), [05-go](05-go.md), [06-java](06-java.md), [07-mcp](07-mcp.md),
[08-ci-release](08-ci-release.md).

## Goals

- Make the transcription core consumable as a **dependency/library** from **Rust, Go, Python,
  Java, and JS/Node**, and by **agents** via an MCP server. The TUI remains one consumer among
  many — for humans; its user-visible behavior does not change.
- **Blocking-first API** in every language: `transcribe(file, options) -> result`, with an
  optional progress/segment callback and cancellation. The full streaming event API stays
  available in Rust for advanced consumers (the TUI itself).
- **Cross-platform from day one**: macOS arm64 (Metal), Linux x86_64/aarch64 (CPU, opt-in
  CUDA artifact), Windows x86_64 (CPU). The MLX engine remains Apple-Silicon-only at runtime
  but must fail with a clean, typed error elsewhere — never a missing symbol or crash.

## Non-goals (v1)

- Streaming microphone/live input (file-based only, as today).
- CUDA variants of the language packages (a CUDA C-ABI artifact ships; CUDA wheels/npm
  packages are P3).
- Non-CPython Python runtimes; browser JS (Node only — whisper.cpp is native).
- Windows arm64.

## Architecture

```
                    ┌────────────────────┐
                    │  transcribe-core    │  Rust lib (crates.io)
                    │  (whisper.cpp, MLX, │
                    │  audio, hub, export)│
                    └─────────┬──────────┘
        ┌───────────┬─────────┼──────────┬───────────────┐
        │           │         │          │               │
  transcribe-tui  PyO3     napi-rs  transcribe-ffi  transcribe-mcp
  (existing app) (Python)  (Node)   (C ABI cdylib)  (stdio MCP, agents)
                                        │
                                  ┌─────┴─────┐
                                  Go (cgo)   Java (Panama)
```

Two engines, two linkage models, unchanged from today:

- **whisper.cpp** — statically linked in-process via `whisper-rs` (compiled from source by
  `whisper-rs-sys`; needs cmake + C++ toolchain at build time). GPU acceleration is a cargo
  feature (`metal`, `cuda`, …) — see [01-core-api](01-core-api.md#feature-gating).
- **MLX** — out-of-process Python subprocess (`mlx-audio`), self-provisioned venv, Apple
  Silicon only. Provisioning becomes opt-in for library embedders — see the risk register.

## Target repository layout

```
Cargo.toml                      # virtual workspace: members = ["crates/*", "bindings/python", "bindings/node"],
                                # resolver = "2", [workspace.package] version/license/repository,
                                # [workspace.dependencies]; workspace.exclude = ["bindings/go", "bindings/java"]
crates/
  transcribe-core/              # package "transcribe-stt-core" — the only crate bindings depend on
  transcribe-tui/               # package "transcribe-stt" (binary name unchanged), publish = false
  transcribe-ffi/               # C ABI cdylib "tstt" + committed include/tstt.h
  transcribe-mcp/               # stdio MCP server binary (tokio + rmcp)
bindings/
  python/                       # PyO3 + maturin (workspace member)
  node/                         # napi-rs (workspace member)
  go/                           # own go.mod (NOT a cargo workspace member)
  java/                         # Gradle (NOT a cargo workspace member)
examples/{c,python,node,go,java,mcp}/   # runnable samples; doubled as CI smoke tests
schemas/                        # schemars-generated JSON Schemas — the cross-binding wire contract
.github/workflows/{ci.yml,release.yml}
docs/specs/                     # this specification
```

## Versioning & licensing

- **One version for everything**, sourced from `workspace.package.version` in the root
  `Cargo.toml`. Wheels, npm packages, the jar, the Go tag, MCP binaries, and C-ABI tarballs
  all carry the same `X.Y.Z`. Rationale: the bindings are thin shims over one core;
  independent versions only create a "which binding works with which lib" support matrix.
- The C ABI has a separate stability signal: `tstt_abi_version()` returns an integer bumped
  **only** on an ABI break ([02-c-abi](02-c-abi.md)). Adding JSON fields is not a break.
- **License: `MIT OR Apache-2.0`** (user-confirmed). The crate currently has **no license
  field** — it must be added (plus `LICENSE-MIT`/`LICENSE-APACHE` files) before any registry
  publish. Compatible with whisper.cpp (MIT).
- Naming placeholders used throughout: `<owner>` (GitHub org/user), `<scope>` (npm scope),
  `io.github.<owner>` (Maven group). **Verify at implementation time**: crates.io name
  `transcribe-stt-core` (`cargo search`), PyPI/npm name `transcribe-stt` availability.

## Phasing

| Phase | Contents | Gate |
|---|---|---|
| **P0** | Workspace split + core refactor ([01-core-api](01-core-api.md)): Sink abstraction, blocking `Session` API, typed errors, feature gating, serde wire types | TUI + headless behave identically; CPU build proven via `--no-default-features` |
| **P1 (v1)** | `transcribe-ffi` + header + C smoke test; Python wheels; Node prebuilds; `transcribe-mcp` binaries; CI matrix (mac-arm64 / linux-x64 / win-x64 full, linux-arm64 build-only); single-version release pipeline | first tagged release |
| **P2** | Go (pkg-config + fetch-lib) and Java (Panama jar, Maven Central); linux-arm64 fully tested; Homebrew formula for `libtstt`; CUDA ffi artifact | |
| **P3** | npx wrapper for MCP; CUDA Python/Node variants; docs site; per-language cookbooks wired into CI | |

P1 deliberately front-loads the highest-leverage consumers (scripting languages + agents)
that require **zero native toolchain** on the user's machine; Go and Java consumers are
developers who can handle a pkg-config/jar workflow, so they follow in P2.

## Consolidated risk register

| # | Risk | Mitigation (spec section) |
|---|---|---|
| 1 | **ggml atexit Metal teardown** can hang/assert at process exit while a whisper context is alive (why `src/main.rs:19` calls `libc::_exit`). Host processes (Python interpreter, JVM, Node) control their own exit. | Lifecycle contract in [01-core-api](01-core-api.md#lifecycle-contract): explicit `shutdown()`/`close()` everywhere; `Drop` joins the worker; best-effort per-host hooks (Python `atexit`, napi env-cleanup, JVM shutdown hook, Go `runtime.AddCleanup`). Only **binaries** (TUI, MCP) may `_exit`; the library never does. |
| 2 | **Callback re-entrancy/deadlock**: callbacks run on the worker thread while the engine mutex is held. | Contract in every binding + `tstt.h` docs: never call engine functions from a callback; only cancel-request is callback-safe. |
| 3 | **Cargo feature unification**: any `--all-features` build (or an unconditional `metal` dependency) breaks Linux/Windows. | Default features empty; TUI enables `metal` via a target-specific dependency; CI uses explicit per-target feature lists, `--all-features` is forbidden ([08-ci-release](08-ci-release.md)). |
| 4 | **MLX when embedded**: a *library* silently running `pip install`, or failing on minimal `PATH` in GUI/JVM hosts, is unacceptable. | Engine config `mlx: "auto" \| "off" \| "require"` + env `TRANSCRIBE_STT_MLX=off`; auto-provisioning of the venv never happens unless explicitly configured; off-Apple-Silicon calls return `unsupported_platform`. Existing `TRANSCRIBE_STT_MLX_VENV` override kept (`src/transcribe/backend/mlx.rs:204`). |
| 5 | **Symbol leakage**: the cdylib statically links ggml/whisper; a host process may load another whisper build (e.g. Python `pywhispercpp`). | Exports restricted to `tstt_*` via version script / exported-symbols list; CI `nm` check ([02-c-abi](02-c-abi.md#symbols)). |
| 6 | **Illegal-instruction crashes** from native-arch whisper.cpp builds inside portable wheels/prebuilds. | All distributed artifacts built with a portable ISA baseline (`GGML_NATIVE=OFF` equivalent). **Verify early in P1** that whisper-rs-sys passes this through its cmake invocation. |
| 7 | **JSON contract drift** across five bindings. | serde types in core are the single source of truth; schemars generates `schemas/*.json`; golden-file tests run in every binding's CI job. |
| 8 | **Codec coverage off-macOS**: symphonia handles wav/flac/mp3/aac/mp4/alac/aiff/ogg-vorbis; opus, wma, and video containers rely on a system `ffmpeg`. | Document ffmpeg as a soft runtime requirement; failures surface as `audio_decode` errors with an install hint. |
| 9 | **whisper-rs abort-callback workaround** (`src/transcribe/backend/whisper_metal.rs:204-212`, double-boxed `Box<dyn FnMut() -> bool>` papering over a whisper-rs 0.16 FFI layout bug) is load-bearing. | Preserve verbatim through every refactor; regression-test cancellation mid-transcription. |
| 10 | **Panics across FFI** are undefined behavior. | Every `extern "C"` function body wrapped in `catch_unwind` → error code `panic`; the ffi crate must NOT set `panic = "abort"`. |
