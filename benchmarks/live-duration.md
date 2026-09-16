# Sustained Voxtral streaming check

Historical results before the checkpoint-layout fix. [The later Voxtral optimization](voxtral-optimization.md) restored faster-than-realtime processing and supersedes this rating.

Apple M4 MacBook Air, 24 GB RAM. Same 120-second synthetic Spanish input, 1-second feeds, one run per version. The model was loaded once per run. These are throughput measurements, not an accuracy evaluation.

| Measurement | Original context | Bounded context |
|---|---:|---:|
| Overall RTF, including final flush | 1.455 | 1.379 |
| Audio 4–24s: processing s/audio s | 1.268 | 1.196 |
| Audio 30–60s: processing s/audio s | 1.307 | 1.369 |
| Audio 60–90s: processing s/audio s | 1.534 | 1.434 |
| Audio 100–120s: processing s/audio s | 1.613 | 1.513 |

The new bridge drained and reset streaming context six times, with at most 20 seconds of audio per session. It kept the same model weights loaded. The final incomplete words were flushed before each reset, and each input sample was fed once. Unit tests cover split requests, delayed-text flushing, absolute timestamps, and repeated resets.

Context resets reduced overall processing time modestly but did not eliminate slowdown or achieve realtime throughput on this machine in this test. The remaining cause was not conclusively identified; macOS reported no recorded thermal warning. At this stage, the model picker marked Voxtral 4-bit as too slow based on the longer run, instead of treating its earlier 24-second result as sufficient evidence.

Context resets can affect recognition near hard boundaries. The bridge prefers a quiet boundary after 12 seconds and forces a flush at 20 seconds. Qwen and Parakeet already process separate speech windows and do not use this native streaming history.

[Raw original measurements](live-duration-before.json) · [Raw bounded-context measurements](live-duration-after.json)
