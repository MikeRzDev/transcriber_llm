"""Private JSON-lines bridge. One model load per microphone session.

stdout is exclusively the protocol; library output is redirected to stderr.
Voxtral's continuous session API is used when available. Other mlx-audio
versions/models receive short speech windows and stream their text deltas.
"""
import contextlib
import inspect
import json
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


def run(model_path, language, emit):
    import mlx.core as mx
    import numpy as np
    from mlx_audio.stt.utils import load_model

    started = time.monotonic()
    model = load_model(model_path)
    native = callable(getattr(model, "create_streaming_session", None))
    session = model.create_streaming_session(max_tokens=1_000_000) if native else None
    if session is not None and getattr(session, "input_sample_rate", 16000) != 16000:
        raise ValueError("This streaming model needs an unsupported sample rate")
    transcript = Transcript(emit)
    emit("ready", native=native, load_secs=time.monotonic() - started)
    for line in sys.stdin:
        request = json.loads(line)
        samples = np.asarray(request.get("samples", []), dtype=np.float32)
        transcript.end = request["end"]
        if native:
            session.feed(samples)
            if request.get("finish"):
                session.close()
                while not session.done:
                    for delta in session.step(max_decode_tokens=64):
                        transcript.delta(delta)
                transcript.commit()
            else:
                for delta in session.step(max_decode_tokens=64):
                    transcript.delta(delta)
                # Break the displayed transcript into readable, committed rows.
                # Native API emits deltas without word timestamps; these are
                # approximate microphone-time spans, never forced alignment.
                if transcript.text.rstrip().endswith((".", "?", "!", "。", "？", "！")) or len(transcript.text) >= 240:
                    transcript.commit()
                if session.done:
                    transcript.commit()
                    session = model.create_streaming_session(max_tokens=1_000_000)
        elif samples.size:
            transcript.start = request["start"]
            transcribe_window(model, mx.array(samples), language, transcript)
        emit("ack", language=transcript.language)
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
