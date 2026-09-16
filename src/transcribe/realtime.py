"""Private JSON-lines bridge. One model load per microphone session.

stdout is exclusively the protocol; library output is redirected to stderr.
Voxtral's continuous session API is used when available. Other mlx-audio
versions/models receive short speech windows and stream their text deltas.
"""
import contextlib
import inspect
import json
import os
import re
import sys
import time


def clean_text(text):
    return re.sub(r"<\|[^>]*\|>|</?asr_text>", "", text)


def output_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text", "")
    return getattr(item, "text", "")


class Transcript:
    def __init__(self, emit):
        self.emit = emit
        self.text = ""
        self.start = 0.0
        self.end = 0.0
        self.language = None

    def delta(self, text):
        self.text += text
        self.emit("partial", text=clean_text(self.text), start=self.start, end=self.end)

    def commit(self):
        text = clean_text(self.text).strip()
        if text:
            self.emit("segment", text=text, start=self.start, end=self.end)
        self.emit("partial", text="", start=self.end, end=self.end)
        self.text = ""
        self.start = self.end


def transcribe_window(model, audio, language, transcript):
    params = inspect.signature(model.generate).parameters
    options = {"verbose": False, "max_tokens": 1024}
    if language and "language" in params:
        options["language"] = language
    streaming = "stream" in params
    if streaming:
        options["stream"] = True
    options = {key: value for key, value in options.items() if key in params}
    result = model.generate(audio, **options)
    if streaming:
        for item in result:
            transcript.delta(output_text(item))
            transcript.language = getattr(item, "language", None) or transcript.language
    else:
        transcript.delta(output_text(result))
        transcript.language = getattr(result, "language", None) or transcript.language
    transcript.commit()


def optimize_voxtral_live(model, model_path):
    """Upgrade legacy 4-bit Voxtral layout in RAM; never rewrite a checkpoint.

    Older conversions left the tied output embedding and audio adapters in
    fp16 and norms in fp32. Match the fast layout described by mlx-audio #661:
    quantized big projections plus bf16 floating parameters/scales. Preserve
    already-quantized integer weights and leave fp16/other model variants alone.
    """
    if getattr(getattr(model, "config", None), "model_type", None) != "voxtral_realtime":
        return None
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from pathlib import Path

    projection = model.decoder.layers[0].attention.wq
    if not isinstance(projection, nn.QuantizedLinear) or projection.bits != 4 or projection.mode != "affine":
        return None
    targets = {"decoder.tok_embeddings", "encoder.audio_language_projection_0", "encoder.audio_language_projection_2"}
    changed = []
    def should_quantize(path, module):
        eligible = path in targets and isinstance(module, (nn.Embedding, nn.Linear))
        if eligible:
            changed.append(path)
        return eligible

    float_change = any(mx.issubdtype(value.dtype, mx.floating) and value.dtype != mx.bfloat16
                       for _, value in tree_flatten(model.parameters()))
    # set_dtype's default predicate preserves uint32 packed quantized weights.
    if float_change:
        model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=projection.group_size, bits=4, class_predicate=should_quantize)
    if float_change or changed:
        # Refresh dtype-dependent precomputed scales through the model's loading
        # hook. It initializes metadata/tokenizer/scales; it does not load weights.
        type(model).post_load_hook(model, Path(model_path))
        mx.eval(model.parameters())
        mx.clear_cache()
        print("Optimized legacy Voxtral 4-bit layout in memory; checkpoint files unchanged", file=sys.stderr)
    return {"layout": "voxtral-4bit-bf16-v1", "converted_layers": changed,
            "dtype_changed": float_change}


class NativeStream:
    """Bound streaming history without reloading weights or discarding audio.

    Prefer a quiet boundary after 12 seconds; force a flush at 20 seconds.
    A flush drains delayed text before releasing the session. Each input sample
    is fed exactly once, even if a request straddles the session boundary.
    """
    SAMPLE_RATE = 16000

    def __init__(self, model, transcript, max_seconds=20, pause_after=12, on_reset=None):
        self.model = model
        self.transcript = transcript
        self.max_samples = round(max_seconds * self.SAMPLE_RATE)
        if self.max_samples < 1:
            raise ValueError("Native session duration must be positive")
        self.pause_samples = round(pause_after * self.SAMPLE_RATE)
        self.on_reset = on_reset
        self.session = None
        self.samples = 0
        self.closed_sessions = 0
        self._open()

    def _open(self):
        self.session = self.model.create_streaming_session(max_tokens=4096)
        if getattr(self.session, "input_sample_rate", self.SAMPLE_RATE) != self.SAMPLE_RATE:
            raise ValueError("This streaming model needs an unsupported sample rate")
        self.samples = 0

    def _step(self):
        for delta in self.session.step(max_decode_tokens=64):
            self.transcript.delta(delta)

    def _release(self):
        self.transcript.commit()
        self.session = None
        self.samples = 0
        self.closed_sessions += 1
        if self.on_reset is not None:
            self.on_reset()

    def _finish(self):
        self.session.close()
        while not self.session.done:
            self._step()
        self._release()

    @staticmethod
    def _quiet_tail(samples):
        tail = samples[-8000:]
        return len(tail) == 8000 and sum(float(s) ** 2 for s in tail) / len(tail) < 0.003 ** 2

    def feed(self, samples, start, end, finish=False):
        offset = 0
        while offset < len(samples):
            if self.session is None:
                self._open()
            count = min(len(samples) - offset, self.max_samples - self.samples)
            piece = samples[offset:offset + count]
            self.session.feed(piece)
            self.samples += count
            offset += count
            self.transcript.end = end if offset == len(samples) else start + offset / self.SAMPLE_RATE
            close = (self.samples >= self.max_samples
                     or (self.samples >= self.pause_samples and self._quiet_tail(piece))
                     or (finish and offset == len(samples)))
            if close:
                self._finish()
            else:
                self._step()
                if self.session.done:
                    self._release()
                elif (self.transcript.text.rstrip().endswith((".", "?", "!", "。", "？", "！"))
                      or len(self.transcript.text) >= 240):
                    self.transcript.commit()
        if finish and self.session is not None:
            self.transcript.end = end
            self._finish()


def run(model_path, language, emit):
    import mlx.core as mx
    import numpy as np
    from mlx_audio.stt.utils import load_model

    diagnostics = os.environ.get("TRANSCRIBE_STT_LIVE_METRICS") == "1"
    started = time.monotonic()
    model = load_model(model_path)
    optimization = optimize_voxtral_live(model, model_path)
    native = callable(getattr(model, "create_streaming_session", None))
    transcript = Transcript(emit)
    stream = NativeStream(model, transcript, on_reset=mx.clear_cache) if native else None
    emit("ready", native=native, load_secs=time.monotonic() - started,
         optimization=optimization,
         session_limit_seconds=stream.max_samples / stream.SAMPLE_RATE if native else None)
    for line in sys.stdin:
        request = json.loads(line)
        samples = np.asarray(request.get("samples", []), dtype=np.float32)
        if native:
            stream.feed(samples, request["start"], request["end"], request.get("finish", False))
        elif samples.size:
            transcript.start = request["start"]
            transcript.end = request["end"]
            transcribe_window(model, mx.array(samples), language, transcript)
        metrics = {"active_gpu_bytes": mx.get_active_memory(),
                   "cached_gpu_bytes": mx.get_cache_memory()} if diagnostics else {}
        emit("ack", language=transcript.language, **metrics,
             session_seconds=stream.samples / stream.SAMPLE_RATE if native else None,
             closed_sessions=stream.closed_sessions if native else None)
        if request.get("finish"):
            break


def main():
    protocol = sys.stdout

    def emit(kind, **values):
        protocol.write(json.dumps({"type": kind, **values}, ensure_ascii=False) + "\n")
        protocol.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, emit)
    except Exception as error:
        emit("error", message=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
