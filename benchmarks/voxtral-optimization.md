# Voxtral 4-bit loading optimization

Hardware: Apple M4 MacBook Air, 24 GB RAM. MLX 0.32.2 and mlx-audio 0.5.3. Installed checkpoint: Voxtral-Mini-4B-Realtime-2602-4bit. Tests used the same synthetic Spanish source repeated to the stated duration, one-second feeds, and bounded streaming context.

The legacy checkpoint contains an 805,306,368-byte FP16 tied token/output embedding, unquantized audio adapters, and FP32 parameters. The current MLX runtime supports a faster layout but preserves old checkpoint precision at load time. The bridge now quantizes only the remaining large embedding/adapter modules to the existing 4-bit affine format and normalizes floating parameters/scales to BF16 in memory. Already-packed uint32 weights remain identical. The checkpoint files are unchanged.

This follows the layout described in [mlx-audio PR #661](https://github.com/Blaizzy/mlx-audio/pull/661). The app applies it only to supported 4-bit affine Voxtral models; fp16, 8-bit, and other model families remain unchanged. Dtype-dependent cached scales are refreshed once, and model weights stay loaded across context resets.

| Run | Audio duration | Sustained processing s/audio s | Final interval |
|---|---:|---:|---:|
| Legacy layout, bounded context | 120s | 1.391 | 1.513 (last 20s) |
| Optimized prototype | 120s | 0.402 | 0.422 (last 20s) |
| Integrated app bridge | 300s | 0.438 | 0.657 (last minute) |

The two-minute improvement was 3.46×. The five-minute run remained below realtime in each minute, including the final minute. It completed 15 context flushes with one model load. Individual flushes can cause brief latency spikes; these results do not guarantee identical speed under every workload.

The original and optimized two-minute runs produced the same 285-word sequence after ignoring case and punctuation. This is a regression check on one synthetic sample, not a general accuracy evaluation. Changing remaining full-precision matrices to 4-bit can affect recognition on other inputs.

Real-MLX regression tests verify that packed weights are unchanged, all floating parameters use BF16, the optimized projection runs, reapplying the helper is a no-op, and fp16/8-bit/other model families are excluded.

[Original raw timings](live-duration-after.json) · [Prototype raw timings](voxtral-fast-layout.json) · [Integrated five-minute timings](voxtral-fast-confirmation.json)
