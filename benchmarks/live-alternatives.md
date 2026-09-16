# Local live transcription benchmark

Run: 2026-09-16T05:13:00.983659+00:00

Hardware: Apple M4; RAM: 24 GiB. Input: 300s; repetitions: 1.

Same audio for every model. RTF = inference seconds / audio seconds; below 1 keeps up. Sustained RTF excludes the first four seconds. Queue figures simulate continuous capture from measured request times. Model loading happens before recording and is excluded from RTF.

| Model | Selected feed | Sustained RTF | Peak queue | Result |
|---|---:|---:|---:|---|
| Qwen3-ASR-1.7B-8bit | 4s | 0.134 | 0.65s | Keeps up with 20% headroom |
| parakeet-tdt-0.6b-v3 | 4s | 0.027 | 0.14s | Keeps up with 20% headroom |

These are measurements for this machine, clip, and workload, not exact per-model constants or an accuracy evaluation. A sustained RTF above 1 cannot be fixed with any finite buffer. An observed buffer allowance in JSON is only a 25% margin over this clip's peak plus one second. Longer recordings, noise, contention, and thermal changes need separate testing.

## Trials

| Model | Feed | Load | First request | Sustained RTF | Total RTF incl. flush | Peak queue |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-ASR-1.7B-8bit | 4s | 3.64s | 0.65s | 0.134 | 0.134 | 0.65s |
| parakeet-tdt-0.6b-v3 | 4s | 3.28s | 0.14s | 0.027 | 0.027 | 0.14s |
