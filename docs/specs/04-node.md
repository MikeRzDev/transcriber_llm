# 04 — Node binding: `bindings/node`

First-class **napi-rs v3** binding over the Rust core directly (not via the C ABI).
Workspace member; TypeScript definitions generated.

## Packaging

- Main npm package **`transcribe-stt`** with per-platform prebuilds delivered as
  `optionalDependencies` (napi-rs's standard artifact workflow generates the packages,
  publish pipeline, and `index.d.ts`):
  `@<scope>/transcribe-stt-darwin-arm64`, `-linux-x64-gnu`, `-linux-arm64-gnu`,
  `-win32-x64-msvc`.
- Node ≥ 18. Verify npm name/scope availability before first publish.

## API

```ts
import { Transcriber, listModels, searchHub, downloadModel,
         renderExport, capabilities } from "transcribe-stt";

const t = new Transcriber({ modelsDir?, mlx?: "auto" | "off" | "require" });

const result = await t.transcribe(path, {
  model?, language?, diarize?, split?,
  onEvent?:    (e: TranscribeEvent) => void,   // full stream
  onSegment?:  (s: Segment) => void,
  onProgress?: (pct: number) => void,          // -1 = indeterminate
  signal?:     AbortSignal,                    // standard cancellation
});
// result: { segments, language, durationSecs, elapsedSecs,
//           toSrt(), toTxt(), toLlmMd() }

t.unload();      // drop resident model
t.close();       // REQUIRED before exit (ggml hazard)

await downloadModel({ repo, file?, dest?, onProgress?, signal? });
```

## Async model

- `transcribe()` returns a `Promise`, executing the blocking core call on a **dedicated
  thread — not the libuv pool** (jobs can run for minutes and must not starve the pool).
- Progress/segment callbacks cross to the JS thread via `ThreadsafeFunction`
  (NonBlocking mode — events are droppable UI signals; segments use a bounded queue so
  none are lost).
- Cancellation: standard `AbortSignal` (napi-rs supports it) mapped to the core
  `CancelToken`; an aborted job rejects with code `cancelled`.
- Errors reject with `Error` objects carrying `.code` = the stable wire code
  ([01-core-api](01-core-api.md#typed-errors-srcerrorrs)).

## Exit hazard (risk #1)

- Contract: call `close()` when done.
- Backstop: register `Env::add_env_cleanup_hook` to drop live engines when the addon's
  env tears down.
- README documents that an unclosed engine can hang `process.exit` in ggml's static
  destructors.

## Tests (run against the built prebuild in CI)

Node test suite: transcribe fixture with tiny model; onSegment/onProgress fire;
AbortSignal cancels; `.code` mapping; golden-file check of the result JSON against
`schemas/`; TypeScript types compile (`tsc --noEmit` on an example).
