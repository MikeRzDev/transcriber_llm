# Spanish real-time ASR — model download manifest

Purpose: open-source speech-to-text models for real-time Spanish (and Spanish/English code-switching).
Two targets: a dual RTX 3090 Linux box (CUDA / vLLM) and a MacBook Air M4 24GB (MLX).

Prereq for all Hugging Face downloads:

```bash
pip install -U "huggingface_hub[cli]"
# optional, for gated/faster downloads:
# huggingface-cli login
```

---

## 1. Qwen3-ASR (Alibaba, Apache 2.0) — primary pick

Language ID + ASR for 52 languages/dialects incl. Spanish and English. Streaming + offline in one model.

| Model | Hugging Face URL | Notes |
|---|---|---|
| Qwen3-ASR-1.7B | https://huggingface.co/Qwen/Qwen3-ASR-1.7B | Best accuracy. Native qwen-asr / vLLM format. |
| Qwen3-ASR-0.6B | https://huggingface.co/Qwen/Qwen3-ASR-0.6B | Fast, lower accuracy. |
| Qwen3-ASR-1.7B-hf | https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf | Plain `transformers` format. |
| Qwen3-ForcedAligner-0.6B | https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B | Word/segment timestamps, up to 5 min audio. |

Code / toolkit: https://github.com/QwenLM/Qwen3-ASR

```bash
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./Qwen3-ForcedAligner-0.6B
pip install -U qwen-asr          # CUDA inference toolkit (vLLM batch, async serving, streaming)
```

### Apple Silicon (MLX)

- Package: `pip install mlx-qwen3-asr` — https://github.com/moona3k/mlx-qwen3-asr (built-in HTTP server, downloads weights on first use)
- Alt package: `pip install qwen3-asr-mlx` — https://pypi.org/project/qwen3-asr-mlx/
- Forced aligner MLX weights: https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit

---

## 2. Voxtral Realtime (Mistral, Apache 2.0) — streaming-native alternative

4B streaming ASR, ~13 languages incl. Spanish. Best raw accuracy of the group on FLEURS.

| Model | Hugging Face URL | Notes |
|---|---|---|
| Voxtral-Mini-4B-Realtime-2602 | https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602 | Official weights, served with vLLM. ~9GB safetensors. |

Collection (all Voxtral models): https://huggingface.co/collections/mistralai/voxtral

```bash
huggingface-cli download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir ./Voxtral-Mini-4B-Realtime-2602
pip install -U vllm mistral-common
```

### Apple Silicon (MLX)

- 4-bit MLX conversion (~2.3GB): https://huggingface.co/T0mSIlver/Voxtral-Mini-4B-Realtime-2602-MLX-4bit
  - `pip install voxmlx` then `voxmlx --model T0mSIlver/Voxtral-Mini-4B-Realtime-2602-MLX-4bit`
- `mlx-audio` also ships 4-bit and fp16 Voxtral Realtime conversions under the `mlx-community` org — check https://huggingface.co/mlx-community?search=voxtral for the exact repo name.

---

## 3. NVIDIA Parakeet-TDT-0.6B-v3 (CC-BY-4.0) — lowest latency

600M params, 25 European languages incl. Spanish, auto language detection. Highest throughput, weaker accuracy than the two above.

| Model | Hugging Face URL | Notes |
|---|---|---|
| parakeet-tdt-0.6b-v3 | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 | NeMo `.nemo` checkpoint. |

```bash
huggingface-cli download nvidia/parakeet-tdt-0.6b-v3 --local-dir ./parakeet-tdt-0.6b-v3
pip install -U "nemo_toolkit[asr]"
```

### Apple Silicon (MLX)

- MLX weights: https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3
- Streaming runtime: https://github.com/elifuzz/parakeet-mlx (`pip install parakeet-mlx`)

---

## 4. Unified Apple Silicon runtime (all three models)

- Python: `pip install mlx-audio` — https://github.com/Blaizzy/mlx-audio
  - One STT API for Qwen3-ASR, Voxtral Realtime, Parakeet; OpenAI-compatible REST server.
- Swift (native macOS/iOS): https://github.com/Blaizzy/mlx-audio-swift

---

## 5. Reference only — Qwen3-Omni (not recommended for pure ASR)

30B-A3B omni-modal chat model that Qwen3-ASR was distilled from. Overkill for transcription; needs 4-bit to fit on a 24GB 3090.

- https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- 4-bit AWQ: https://huggingface.co/cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
- Code: https://github.com/QwenLM/Qwen3-Omni

---

## Baseline

- Whisper large-v3: https://huggingface.co/openai/whisper-large-v3 (MIT) — keep as the comparison baseline only.
