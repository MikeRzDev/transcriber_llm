# Speech-to-render latency after a quiet wait

Target: meaningful text rendered within 0.9 seconds of speech onset after more than seven minutes of silence.

| Check | Quiet interval | Measured time | Endpoint |
|---|---:|---:|---|
| Bridge validation | 480s | 0.831s | First non-empty meaningful partial received |
| Short software-path check | 1s | 0.841s | Terminal test-backend draw completed |
| Long software-path check | 480s | 0.836s | Terminal test-backend draw completed |

The long check rendered `Buenos`. It passed the 0.9-second threshold. The input was a paced 48 kHz mono fixture: eight minutes of zeros, followed by the Spanish speech fixture upsampled from 16 kHz. It passed through the production capture receiver, Rust resampler, live worker/IPC, MLX Voxtral, application events, transcript layout, and ratatui terminal test-backend drawing. Resource sampling was included. The model was loaded/warmed once; it performed no inference during the quiet interval.

This verifies the software path under these test conditions. It does not include a physical microphone/driver or a real terminal compositor/display refresh. It is not a universal deadline guarantee for all speech, devices, noise, or competing workloads. The app now measures its own first-text time through the completed draw so an actual live session can be checked directly.

Changes: native audio is sent every 100 ms, including the initial short resampler output; Voxtral is prewarmed before capture and uses its supported 80 ms delay; 250 ms of quiet pre-roll preserves onset; inference sleeps through quiet input while weights remain loaded; incoming events are applied before drawing; live polling is 20 ms; first-text timing finishes after the draw. Lower model delay prioritizes latency and can reduce recognition accuracy versus 480 ms.

[Mistral delay settings](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602#recommended-settings) · [Long render-path raw result](voxtral-render-480s.json) · [Bridge raw result](voxtral-latency-after-idle.json)

Reproduce the full software-path check with the app closed:

```sh
TRANSCRIBE_TEST_MODEL=/path/to/Voxtral-Mini-4B-Realtime-2602-4bit \
TRANSCRIBE_TEST_AUDIO=samples/benchmark-es.wav \
TRANSCRIBE_TEST_IDLE_SECS=480 \
TRANSCRIBE_TEST_REPORT=benchmarks/voxtral-render-480s.json \
cargo test --release --locked --lib paced_idle_first_render -- --ignored --nocapture --test-threads=1
```


## Session-length growth checks

The transcript UI previously wrapped the entire history twice per frame. It now caches committed lines and wraps only appended segments or changed partial text; a regression test with 10,000 segments verifies that repeated partial updates never re-wrap that history. Width or speaker-label changes invalidate layout once. Resource sampling now runs on a worker with nonblocking UI requests/results.

A local release-build measurement found a full-history wrap at 10,000 segments took 31.396 ms, while a cached partial update plus visible-line copying took about 0.004 ms. At 100 segments, the cached work was about 0.007 ms. These are component timings, not complete microphone-to-display latency. [Raw render-cost measurement](transcript-render-cost.json).

The final short software-path recheck after these changes rendered at 0.841 seconds. The eight-minute idle result above was measured before these additional UI improvements. The app explicitly displays and logs an observed value above the 0.9-second target; it does not hide misses, discard late speech, or imply a hard timing guarantee under arbitrary OS/GPU load.
