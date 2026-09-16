"""Private JSON-lines bridge. One model load per microphone session.

stdout is exclusively the protocol; library output is redirected to stderr.
Voxtral's continuous session API is used when available. Other mlx-audio
versions/models receive short speech windows and stream their text deltas.
"""
import contextlib
from collections import deque
import inspect
import json
import os
import re
import sys
import time


class LivePriority:
    """Interactive transcription takes priority over our background benchmarks.

    Live sessions hold a shared advisory lock. Benchmarks only probe for an
    exclusive lock, releasing it immediately; they never reserve the GPU ahead
    of the user. OS cleanup releases live locks even after process termination.
    """
    def __init__(self, benchmark=False, path=None):
        self.benchmark = benchmark
        self.path = path
        self.fd = None

    def __enter__(self):
        import fcntl
        import tempfile
        from pathlib import Path
        path = self.path or Path(tempfile.gettempdir()) / f"transcribe-stt-live-{os.getuid()}.lock"
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        if not self.benchmark:
            fcntl.flock(self.fd, fcntl.LOCK_SH)
        return self

    def check(self):
        if not self.benchmark:
            return
        import fcntl
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Benchmark stopped: interactive live transcription has GPU priority") from None
        fcntl.flock(self.fd, fcntl.LOCK_UN)

    def __exit__(self, *args):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


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


def native_session(model, delay_ms=None):
    options = {"max_tokens": 4096}
    if delay_ms is not None:
        options["transcription_delay_ms"] = delay_ms
    session = model.create_streaming_session(**options)
    if getattr(session, "input_sample_rate", 16000) != 16000:
        raise ValueError("This streaming model needs an unsupported sample rate")
    return session


def warm_native(model, silence, delay_ms=None):
    # Compile/evaluate the streaming path before the microphone starts. Discard
    # warm-up output and context; retain the same weights and compiled kernels.
    session = native_session(model, delay_ms)
    session.feed(silence)
    session.close()
    while not session.done:
        session.step(max_decode_tokens=64)


class NativeStream:
    """Keep weights resident, suspend decoding during quiet microphone periods.

    Detect input energy in 20 ms frames using the microphone monitor's quiet
    threshold. Retain bounded pre-roll to preserve speech onset. Once an
    active utterance has a one-second quiet tail, flush its pending text and
    stop inference until signal returns. Active context remains bounded.
    """
    SAMPLE_RATE = 16000

    def __init__(self, model, transcript, max_seconds=20, pause_after=12, on_reset=None,
                 delay_ms=None, preroll_seconds=1):
        self.model = model
        self.transcript = transcript
        self.max_samples = round(max_seconds * self.SAMPLE_RATE)
        if self.max_samples < 1:
            raise ValueError("Native session duration must be positive")
        self.pause_samples = round(pause_after * self.SAMPLE_RATE)
        self.on_reset = on_reset
        self.delay_ms = delay_ms
        self.session = None
        self.samples = 0
        self.closed_sessions = 0
        self.quiet_samples = 0
        self.preroll = deque(maxlen=round(preroll_seconds * self.SAMPLE_RATE))
        self.inference_active = False
        self.waiting_for_speech = True

    @property
    def idle(self):
        return self.session is None

    @classmethod
    def _signal_bounds(cls, samples):
        # A short onset should not disappear in the RMS of a whole 1s packet.
        first, last = None, 0
        for start in range(0, len(samples), 320):
            frame = samples[start:start + 320]
            if sum(float(s) ** 2 for s in frame) >= len(frame) * 0.003 ** 2:
                if first is None:
                    first = start
                last = start + len(frame)
        return first, last

    def feed(self, samples, start, end, finish=False):
        self.inference_active = False
        was_waiting = self.waiting_for_speech
        first_signal, last_signal = self._signal_bounds(samples)
        self.waiting_for_speech = self.idle and not last_signal
        if self.idle and not last_signal:
            self.preroll.extend(samples)
            if finish:
                self.preroll.clear()
                self.transcript.end = end
            return
        if self.idle:
            if was_waiting:
                self.transcript.emit("speech_start", start=start + first_signal / self.SAMPLE_RATE)
            if self.preroll:
                prefix = list(self.preroll)
                samples = prefix + list(samples)
                start -= len(prefix) / self.SAMPLE_RATE
                last_signal += len(prefix)
                self.preroll.clear()
            self.transcript.start = start
            self.quiet_samples = 0
        self.quiet_samples = len(samples) - last_signal if last_signal else self.quiet_samples + len(samples)
        self.inference_active = True
        self._feed_active(samples, start, end, finish or self.quiet_samples >= self.SAMPLE_RATE)
        self.waiting_for_speech = self.idle and self.quiet_samples >= self.SAMPLE_RATE

    def _open(self):
        self.session = native_session(self.model, self.delay_ms)
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

    def _feed_active(self, samples, start, end, finish=False):
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
    with LivePriority(benchmark=os.environ.get("TRANSCRIBE_STT_BENCHMARK") == "1") as priority:
        priority.check()
        return run_model(model_path, language, emit, priority)


def run_model(model_path, language, emit, priority):
    import mlx.core as mx
    import numpy as np
    from mlx_audio.stt.utils import load_model

    diagnostics = os.environ.get("TRANSCRIBE_STT_LIVE_METRICS") == "1"
    from importlib.metadata import PackageNotFoundError, version
    runtime = {"python": sys.executable, "bridge": "idle-aware-2"}
    for package in ("mlx", "mlx-audio"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = "unknown"
    started = time.monotonic()
    model = load_model(model_path)
    priority.check()
    optimization = optimize_voxtral_live(model, model_path)
    native = callable(getattr(model, "create_streaming_session", None))
    transcript = Transcript(emit)
    delay_ms = 80 if native and getattr(model.config, "model_type", None) == "voxtral_realtime" else None
    if native:
        warm_native(model, np.zeros(16000, dtype=np.float32), delay_ms)
        mx.clear_cache()
    stream = NativeStream(model, transcript, on_reset=mx.clear_cache,
                          delay_ms=delay_ms, preroll_seconds=0.25) if native else None
    emit("ready", native=native, load_secs=time.monotonic() - started,
         optimization=optimization, runtime=runtime,
         transcription_delay_ms=delay_ms, live_feed_seconds=0.1 if native else None,
         session_limit_seconds=stream.max_samples / stream.SAMPLE_RATE if native else None)
    for line in sys.stdin:
        priority.check()
        request = json.loads(line)
        samples = np.asarray(request.get("samples", []), dtype=np.float32)
        inference_started = time.perf_counter()
        if native:
            stream.feed(samples, request["start"], request["end"], request.get("finish", False))
        elif samples.size:
            transcript.start = request["start"]
            transcript.end = request["end"]
            transcribe_window(model, mx.array(samples), language, transcript)
        metrics = {"active_gpu_bytes": mx.get_active_memory(),
                   "cached_gpu_bytes": mx.get_cache_memory()} if diagnostics else {}
        emit("ack", language=transcript.language, **metrics,
             idle=stream.waiting_for_speech if native else False,
             inference_active=stream.inference_active if native else bool(samples.size),
             inference_seconds=time.perf_counter() - inference_started,
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
