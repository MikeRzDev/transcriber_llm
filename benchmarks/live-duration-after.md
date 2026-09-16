# Local live transcription benchmark

Run: 2026-09-16T05:03:19.553988+00:00

Hardware: Apple M4; RAM: 24 GiB. Input: 120s; repetitions: 1.

Same audio for every model. RTF = inference seconds / audio seconds; below 1 keeps up. Sustained RTF excludes the first four seconds. Queue figures simulate continuous capture from measured request times. Model loading happens before recording and is excluded from RTF.

| Model | Selected feed | Sustained RTF | Peak queue | Result |
|---|---:|---:|---:|---|
| Voxtral-Mini-4B-Realtime-2602-4bit | 1s | 1.391 | 30.00s | No tested feed has 20% headroom |

These are measurements for this machine, clip, and workload, not exact per-model constants or an accuracy evaluation. A sustained RTF above 1 cannot be fixed with any finite buffer. An observed buffer allowance in JSON is only a 25% margin over this clip's peak plus one second. Longer recordings, noise, contention, and thermal changes need separate testing.

## Trials

| Model | Feed | Load | First request | Sustained RTF | Total RTF incl. flush | Peak queue |
|---|---:|---:|---:|---:|---:|---:|
| Voxtral-Mini-4B-Realtime-2602-4bit | 1s | 6.56s | 1.06s | 1.391 | 1.379 | 30.00s |
