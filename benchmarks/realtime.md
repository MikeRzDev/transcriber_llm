# Local live transcription benchmark

Run: 2026-09-16T04:28:37.699237+00:00

Hardware: Apple M4; RAM: 24 GiB. Input: 24s; repetitions: 2.

Same audio for every model. RTF = inference seconds / audio seconds; below 1 keeps up. Sustained RTF excludes the first four seconds. Queue figures simulate continuous capture from measured request times. Model loading happens before recording and is excluded from RTF.

| Model | Selected feed | Sustained RTF | Peak queue | Result |
|---|---:|---:|---:|---|
| Qwen3-ASR-0.6B-8bit | 4s | 0.058 | 0.31s | Keeps up with 20% headroom |
| Qwen3-ASR-1.7B-8bit | 4s | 0.129 | 0.67s | Keeps up with 20% headroom |
| Qwen3-ASR-1.7B-bf16 | 4s | 0.197 | 1.34s | Keeps up with 20% headroom |
| Qwen3-ForcedAligner-0.6B-8bit | — | — | — | Forced aligner requires a transcript; not a live ASR model |
| Voxtral-Mini-4B-Realtime-2602-4bit | 1s | 0.420 | 1.84s | Keeps up with 20% headroom |
| Voxtral-Mini-4B-Realtime-2602-fp16 | 0.5s | 6.162 | 19.70s | No tested feed has 20% headroom |
| Voxtral-Mini-4B-Realtime-6bit | — | — | — | voxmlx conversion is incompatible with the app's mlx-audio live bridge |
| parakeet-tdt-0.6b-v3 | 4s | 0.025 | 1.59s | Keeps up with 20% headroom |

These are measurements for this machine, clip, and workload, not exact per-model constants or an accuracy evaluation. A sustained RTF above 1 cannot be fixed with any finite buffer. An observed buffer allowance in JSON is only a 25% margin over this clip's peak plus one second. Longer recordings, noise, contention, and thermal changes need separate testing.

## Trials

| Model | Feed | Load | First request | Sustained RTF | Total RTF incl. flush | Peak queue |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-ASR-0.6B-8bit | 4s | 2.52s | 0.31s | 0.058 | 0.061 | 0.31s |
| Qwen3-ASR-0.6B-8bit | 4s | 0.82s | 0.30s | 0.059 | 0.061 | 0.30s |
| Qwen3-ASR-1.7B-8bit | 4s | 3.35s | 0.67s | 0.130 | 0.136 | 0.67s |
| Qwen3-ASR-1.7B-8bit | 4s | 0.94s | 0.64s | 0.129 | 0.134 | 0.64s |
| Qwen3-ASR-1.7B-bf16 | 4s | 4.95s | 1.34s | 0.199 | 0.222 | 1.34s |
| Qwen3-ASR-1.7B-bf16 | 4s | 0.97s | 1.01s | 0.196 | 0.206 | 1.01s |
| Voxtral-Mini-4B-Realtime-2602-4bit | 0.5s | 4.02s | 0.20s | 1.031 | 1.074 | 1.17s |
| Voxtral-Mini-4B-Realtime-2602-4bit | 0.5s | 4.37s | 0.20s | 0.994 | 1.040 | 0.58s |
| Voxtral-Mini-4B-Realtime-2602-4bit | 1s | 4.01s | 0.75s | 0.956 | 0.999 | 1.01s |
| Voxtral-Mini-4B-Realtime-2602-4bit | 1s | 2.23s | 0.75s | 0.952 | 0.999 | 1.01s |
| Voxtral-Mini-4B-Realtime-2602-4bit | 2s | 2.28s | 1.70s | 0.977 | 1.023 | 2.11s |
| Voxtral-Mini-4B-Realtime-2602-4bit | 2s | 1.07s | 1.69s | 1.034 | 1.076 | 2.48s |
| Voxtral-Mini-4B-Realtime-2602-4bit | 4s | 1.38s | 3.70s | 1.159 | 1.191 | 5.82s |
| Voxtral-Mini-4B-Realtime-2602-4bit | 4s | 1.06s | 4.36s | 1.177 | 1.238 | 6.50s |
| Voxtral-Mini-4B-Realtime-2602-fp16 | 0.5s | 10.07s | 0.76s | 6.252 | 6.501 | 19.70s |
| Voxtral-Mini-4B-Realtime-2602-fp16 | 0.5s | 10.56s | 0.46s | 6.072 | 6.144 | 19.00s |
| parakeet-tdt-0.6b-v3 | 4s | 3.81s | 1.59s | 0.025 | 0.087 | 1.59s |
| parakeet-tdt-0.6b-v3 | 4s | 0.79s | 0.13s | 0.026 | 0.027 | 0.13s |

Voxtral-Mini-4B-Realtime-2602-4bit: summary rating uses a 300s sustained validation with bounded context (0.438 RTF, 15 context flushes, model weights kept loaded). See [Voxtral optimization results](voxtral-optimization.md).

Voxtral-Mini-4B-Realtime-2602-fp16: Additional feed sizes not tested: every baseline run exceeded 2 RTF. Use --max-sweep-rtf 0 --resume for a full sweep.
