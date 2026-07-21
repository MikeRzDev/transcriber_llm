# 06 — Java binding: `bindings/java`

**Panama (`java.lang.foreign`, JDK 22+) over the C ABI** ([02-c-abi](02-c-abi.md)).
Not a cargo workspace member — Gradle build. Ships in **P2**.

## Why Panama over JNA

- Zero runtime dependencies (JNA is a ~2 MB jar + its own native dispatch lib).
- Upcall stubs are far cheaper and safer than JNA callback proxies for a per-segment
  callback.
- `jextract` bootstraps bindings straight from the committed `tstt.h`.
- By 2026, JDK 21 is the *old* LTS; a new native-heavy library can require 22+.

Trade-off documented: if JDK-17 demand appears, a JNA shim over the same C ABI is a
contained, additive fallback — the public Java API would not change.

## Packaging

- Maven coordinates `io.github.<owner>:transcribe-stt`, published via Gradle + Sonatype
  Central portal, same version as everything else.
- Single jar embedding
  `natives/{darwin-aarch64,linux-x86_64,linux-aarch64,windows-x86_64}/libtstt.*`.
- `NativeLoader` extracts the matching lib to
  `~/.cache/transcribe-stt/native/<version>/` (temp-dir fallback) and `System.load`s it
  at class init; verifies `tstt_abi_version()`.
- If the fat jar exceeds ~60 MB, switch to LWJGL-style per-platform classifier jars —
  decide once real artifact sizes are known.

## API

```java
try (var t = new Transcriber(Config.builder().modelsDir(p).mlx(Mlx.AUTO).build())) {
    TranscribeRequest req = TranscribeRequest.builder()
        .audio(path).model(model).language("en").diarize(true).build();

    Transcription r = t.transcribe(req, new TranscribeListener() {
        @Override public boolean onEvent(Event e) { return true; } // false = cancel
        @Override public void onSegment(Segment s) { ... }         // default no-op
        @Override public void onProgress(int pct) { ... }          // default no-op
    });
    r.segments(); r.language(); r.durationSecs(); r.elapsedSecs();
    r.toSrt(); r.toTxt(); r.toLlmMd();
}

// static: Models.list(dir), Hub.search(q), Hub.download(req, listener), Export.render(doc, fmt)
```

- `Transcriber implements AutoCloseable`; **try-with-resources is the contract**
  (`close()` → `tstt_engine_free`). Best-effort backstop: one JVM shutdown hook closes
  live transcribers (risk #1). A `Cleaner` registration guards against leaked instances.
- Events arrive as JSON per the ABI; parsed with a small vendored parser or
  `java.util.Optional`-friendly records generated to match `schemas/` — **no Jackson/Gson
  dependency**. (Alternative kept open: expose `String onEventJson(String)` for users who
  want raw JSON.)
- Errors: `TranscribeException` with `code()` returning the stable wire code; subclasses
  for `ModelNotFoundException`, `UnsupportedPlatformException`, `CancelledException`, etc.

## FFI mechanics

- Downcalls/upcalls generated with `jextract` from `tstt.h`, checked into
  `bindings/java` (regenerated + diffed in CI when the header changes).
- The upcall stub for `tstt_event_cb` is allocated per call in a **confined `Arena`**
  scoped to the `transcribe()` invocation; documented: the callback runs on the worker
  thread, and must not touch the Transcriber (ABI re-entrancy rule) — cancellation is via
  returning `false` or `CancelHandle.request()`.

## Tests

JUnit, run per platform in CI against the aggregated jar: transcribe fixture with tiny
model; listener callbacks fire; cancellation via `false` return; exception mapping;
golden-file JSON check against `schemas/`; loader extraction smoke.
