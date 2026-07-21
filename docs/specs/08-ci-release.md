# 08 — CI & release

GitHub Actions; two workflows. All jobs use explicit per-target feature lists —
**`--all-features` is forbidden** (it would force `metal` onto Linux/Windows; risk #3).

## Platform matrix

| Runner | Target | Accel | v1 status |
|---|---|---|---|
| `macos-14` | aarch64-apple-darwin | metal | full |
| `ubuntu-22.04` | x86_64-unknown-linux-gnu | CPU (+ opt-in CUDA artifact) | full |
| `ubuntu-24.04-arm` | aarch64-unknown-linux-gnu | CPU | build-only in P1, full in P2 |
| `windows-2022` | x86_64-pc-windows-msvc | CPU | full |

Build prerequisites: cmake + C++ toolchain everywhere (whisper-rs-sys compiles
whisper.cpp from source); Metal frameworks only on macOS; MSVC on Windows. **Verify at
implementation time** that cmake is preinstalled on the arm64 runner image. Distributed
artifacts are built with the portable ISA baseline (`GGML_NATIVE=OFF` equivalent — verify
whisper-rs-sys env/flag passthrough as the first task of P1; risk #6).

A tiny model (`ggml-tiny.bin`) and a short audio fixture are cached
(`actions/cache`) for every job that actually transcribes.

## `ci.yml` (PRs + main)

| Job | What it does |
|---|---|
| `test-core` | 4-target matrix: `cargo test -p transcribe-stt-core -p transcribe-ffi` with per-target features (`metal` on mac only); clippy + fmt |
| `build-ffi` | builds the cdylib per target; **header check** (regenerate with cbindgen, diff against committed `include/tstt.h`); **symbol check** (`nm -gD`/equivalent: no exported non-`tstt_*` symbols); uploads `tstt-<target>.tar.gz` = lib + `tstt.h` + `tstt.pc` |
| `test-ffi-c` | compiles `tests/smoke.c` against the artifact and runs it with the cached tiny model |
| `wheels` | maturin-action matrix (4 wheel targets from [03-python](03-python.md)); pytest runs against the **built wheel**, not the source tree |
| `napi` | napi-rs generated matrix; node tests against the built prebuild; `tsc --noEmit` on the TS example |
| `test-go` | downloads the `build-ffi` artifact, sets `PKG_CONFIG_PATH`, `go test ./bindings/go/...` (P2) |
| `build-java` | aggregates all 4 ffi artifacts into the jar; JUnit per platform (P2) |
| `build-mcp` | builds `transcribe-mcp` binaries; JSON-RPC stdio smoke test ([07-mcp](07-mcp.md#ci-validation)) |
| `schemas` | regenerates `schemas/*.json`, diffs against committed — golden files can't drift (risk #7) |
| `examples` | runs `examples/<lang>/` samples as smoke tests so documentation can't rot |

CUDA: opt-in `build-ffi-cuda` job (linux-x64, whisper-rs `cuda` feature, CUDA toolkit
container) producing `tstt-linux-x64-cuda.tar.gz`; runs on a schedule/label, **never
blocks** the main matrix.

## `release.yml` (on tag `vX.Y.Z`)

1. Re-run the full `ci.yml` matrix (release profile).
2. Publish:
   - **PyPI**: wheels via trusted publishing (OIDC, no long-lived token).
   - **npm**: main + platform packages via the napi artifact flow, with provenance.
   - **crates.io**: `transcribe-stt-core`, then `transcribe-mcp`.
   - **GitHub Release**: ffi tarballs (all targets + cuda), `transcribe-mcp` binaries
     (cargo-binstall naming), the jar.
   - **P2**: Maven Central (Sonatype), Go (the same repo tag is the Go module version);
     Homebrew formula bump for `libtstt`.

## Versioning

- Single source of truth: `workspace.package.version` in the root `Cargo.toml`
  ([00-overview](00-overview.md#versioning--licensing)). maturin reads it natively;
  `scripts/set-version.sh` syncs `bindings/node/package.json` + platform packages and
  the Gradle version, and CI **fails if they drift**.
- Tag scheme: `vX.Y.Z`, one tag releases every package.
- `tstt_abi_version()` (integer) changes only on C ABI breaks and is asserted by the Go
  and Java loaders at runtime.

## Docs & examples layout

```
examples/
  c/transcribe.c            # against tstt.h; also the doc example in 02-c-abi
  python/transcribe.py
  node/transcribe.mjs
  go/main.go
  java/Transcribe.java
  mcp/README.md             # client config snippets (claude mcp add, Claude Desktop)
```

Each is a minimal "transcribe this file with the tiny model, print segments + progress"
sample, executed by the `examples` CI job. Per-binding READMEs live in their
`bindings/<lang>/` directory; the root README gains a "Use as a library" section linking
here.
