# Reproducing the local benchmark

The recorded results were measured on an Apple M4 with 24 GB RAM. They use 24 seconds of synthetic Spanish speech from macOS's Paulina voice, at 155 words/minute. This isolates inference throughput; it does not measure transcription accuracy on real conversations. Each setting runs twice in a fresh model process. Runs are sequential so models do not compete with one another for the GPU.

Generate the input on macOS:

```sh
mkdir -p samples
say -v Paulina -r 155 -f assets/benchmark-es.txt -o samples/benchmark-es.aiff
ffmpeg -v error -y -i samples/benchmark-es.aiff -ar 16000 -ac 1 -c:a pcm_s16le samples/benchmark-es.wav
```

Run with the same MLX environment as the app:

```sh
.venv-realtime/bin/python scripts/benchmark-realtime.py \
  --models-dir /Volumes/MikeExternal/ai_models/speech-to-text-realtime \
  --audio samples/benchmark-es.wav --language es --seconds 24 --repeats 2
```

[Results](realtime.md) summarize [raw measurements](realtime.json), which include every request duration and the returned transcripts. The input checksum and OS/Python details are recorded in JSON. Native Voxtral feeds are swept at 0.5, 1, 2, and 4 seconds. Windowed ASR uses the app's four-second continuous-speech limit.

RTF below 1 means faster than realtime. The recommendation selects the shortest tested feed whose worst repetition is at most 0.8 RTF, leaving 20% measured compute headroom. If none qualifies, it reports the fastest tested feed without claiming sufficient headroom. Queue peaks are simulated from measured inference durations and audio arrival times; microphone capture itself is not exercised. The optional buffer allowance is just the observed peak plus a margin, not a guarantee or a model-specific constant. Sustained RTF above 1 has no finite safe buffer for an unlimited recording.

Use a longer WAV and `--seconds 120` (or more) to evaluate drift during longer sessions. Use `--output benchmarks/custom.json` to preserve these baseline results. The benchmark never changes your selected model or application settings.

By default, if every baseline repetition exceeds 2 RTF, the benchmark skips additional feed sizes for that model and records the reason. Voxtral fp16 hit this cutoff in this run. Use `--max-sweep-rtf 0 --resume` to complete all feed sizes; `--resume` reuses matching saved measurements and checks the input checksum, duration, language, and repetition count.

To install these results in the **m** model selector, append `--resume --install` to the benchmark command. The app reads `~/.config/transcribe-stt/live-benchmarks.json` and shows ratings only when the hardware and model-file metadata still match. Ratings do not disable selection. Starting a live session automatically applies the saved feed size for native streaming models; models without a matching result use adaptive batching. Qwen and Parakeet use the tested speech windows of up to four seconds.

A subsequent [120-second streaming check](live-duration.md) supersedes the short-run rating for Voxtral 4-bit. Its streaming context is now renewed every 12–20 seconds without reloading weights, but sustained throughput in that test was still slower than realtime.

The later [Voxtral checkpoint-layout optimization](voxtral-optimization.md) fixes the legacy conversion in memory and supersedes the earlier too-slow rating. The integrated bridge averaged 0.438 processing seconds per audio second over five minutes; the installed rating is good fit again.


For a long-session check with audio arriving at normal microphone speed:

```sh
.venv-realtime/bin/python scripts/benchmark-realtime.py \
  --models-dir /Volumes/MikeExternal/ai_models/speech-to-text-realtime \
  --audio samples/benchmark-es.wav --seconds 3600 --loop-audio \
  --paced --diagnostics --batches 1 --repeats 1 \
  --model Voxtral-Mini-4B-Realtime-2602-4bit \
  --output benchmarks/voxtral-hour-paced.json
```

This takes one hour of wall time after model loading. The short fixture is repeated using bounded memory. Each request waits until its audio would have arrived from a microphone, using absolute deadlines so a slow request does not move later capture deadlines. The default benchmark runs as fast as possible and is a throughput stress test; it does not reproduce the GPU idle time available during speech. These modes must not be mixed when resuming a report.

`--diagnostics` records active/cached GPU memory, streaming-context age, and completed context flushes. A `.progress.json` file updates every minute with speed and backlog. Diagnostics are opt-in and do not record microphone audio. The final report retains every request measurement.
