# Local live transcription benchmark

Run: 2026-09-16T05:17:06.182376+00:00

Hardware: Apple M4; RAM: 24 GiB. Input: 120s; repetitions: 1.

Same audio for every model. RTF = inference seconds / audio seconds; below 1 keeps up. Sustained RTF excludes the first four seconds. Queue figures simulate continuous capture from measured request times. Model loading happens before recording and is excluded from RTF.

| Model | Selected feed | Sustained RTF | Peak queue | Result |
|---|---:|---:|---:|---|
| Voxtral-Mini-4B-Realtime-2602-4bit | 1s | 0.402 | 0.96s | Keeps up with 20% headroom |

These are measurements for this machine, clip, and workload, not exact per-model constants or an accuracy evaluation. A sustained RTF above 1 cannot be fixed with any finite buffer. An observed buffer allowance in JSON is only a 25% margin over this clip's peak plus one second. Longer recordings, noise, contention, and thermal changes need separate testing.

## Trials

| Model | Feed | Load | First request | Sustained RTF | Total RTF incl. flush | Peak queue |
|---|---:|---:|---:|---:|---:|---:|
| Voxtral-Mini-4B-Realtime-2602-4bit | 1s | 4.42s | 0.63s | 0.402 | 0.402 | 0.96s |
