# 07 — MCP server: `crates/transcribe-mcp`

The agent-facing surface: a **stdio MCP server** so Claude (Desktop, Code) and other
MCP clients can transcribe files as a tool. Ships in **P1**.

## Crate

- `crates/transcribe-mcp`, binary **`transcribe-mcp`**, depending on `transcribe-stt-core`.
- Built on **rmcp** — the official Rust MCP SDK
  (github.com/modelcontextprotocol/rust-sdk), stdio transport, `#[tool]` proc-macros.
- rmcp is tokio-based; the core is blocking → jobs run in `tokio::task::spawn_blocking`,
  with core events piped over an mpsc into MCP **`notifications/progress`** against the
  request's progress token (load → decode → transcribe phases; segment counts in the
  progress message).
- Kept as a **separate crate** so the TUI gains no tokio dependency.

## Tools

Input schemas are generated from the same serde types as everything else
(schemars → consistent with `schemas/`).

| Tool | Input | Output |
|---|---|---|
| `transcribe_file` | `{path, model?, language?, diarize?, output_format? = "llm.md"\|"txt"\|"srt"\|"segments", save_to?}` | rendered transcript as text content (plus structured segments when `output_format = "segments"`); if `save_to` set, writes the file and returns the path |
| `list_models` | `{}` | local models (name, path, size, kind) from the resolved models dir |
| `search_models` | `{query}` | HF hub hits (curated suggestions included) |
| `list_model_files` | `{repo}` | downloadable files/variants for a repo |
| `download_model` | `{repo, file?, subdir?}` | downloads into the models dir with progress notifications; returns the local path |

- Capabilities (accel, MLX availability, ffmpeg presence) exposed as an MCP **resource**,
  not a tool.
- Errors map the stable wire codes ([01-core-api](01-core-api.md#typed-errors-srcerrorrs))
  into tool-call errors with actionable messages (e.g. `unsupported_platform` explains
  MLX needs Apple Silicon; `mlx_runtime_missing` explains the provisioning setting).
- Config: reuses `config.rs` resolution (`TRANSCRIBE_STT_MODELS`, config.toml) plus flags
  `--models-dir`, `--mlx auto|off|require`.

## Lifecycle

On transport close (client disconnect): finish/cancel the in-flight job, drop the engine
explicitly, then the **binary-only** `libc::_exit` pattern from `src/main.rs:19` is
permitted (a binary may do this; the library layers never do — risk #1).

## Distribution

- **P1**: prebuilt per-platform binaries on GitHub Releases with cargo-binstall-compatible
  naming, plus `cargo install transcribe-mcp` for source builds.
- Docs include ready-to-paste client config:
  - `claude mcp add transcribe -- transcribe-mcp`
  - Claude Desktop `mcpServers` JSON snippet.
- **P3**: `@<scope>/transcribe-mcp` npm wrapper that downloads the platform binary so
  `npx` works.

## CI validation

JSON-RPC stdio smoke test in CI (per the mcp-inspector approach): `initialize` →
`tools/list` (assert the 5 tools + schemas) → `transcribe_file` on a fixture with the
tiny model → assert transcript content and at least one progress notification.
