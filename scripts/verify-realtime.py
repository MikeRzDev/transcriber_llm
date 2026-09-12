"""Exercise the actual live bridge with a 16 kHz mono PCM16 WAV, no microphone.

Run with the Python environment that contains mlx-audio:
  python scripts/verify-realtime.py --model /path/to/model --audio speech.wav
"""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import wave
from array import array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()
    with wave.open(args.audio) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        data = array("h", wav.readframes(wav.getnframes()))
        if sys.byteorder != "little":
            data.byteswap()
        samples = [sample / 32768 for sample in data]
    assert len(samples) >= 16000 and max(map(abs, samples)) > 0.01, "Smoke test needs at least one second of audible speech"
    child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve().parents[1] / "src/transcribe/realtime.py"), args.model, "es"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    responses = queue.Queue()
    errors = []
    def read_output():
        for line in child.stdout:
            responses.put(json.loads(line))
        responses.put({"type": "error", "message": "bridge exited: " + "\n".join(errors[-5:])})
    def read_errors():
        for line in child.stderr:
            errors.append(line.rstrip())
    readers = [threading.Thread(target=read_output, daemon=True), threading.Thread(target=read_errors, daemon=True)]
    for reader in readers:
        reader.start()
    segments = []
    partials = []
    def wait(kind):
        while True:
            item = responses.get(timeout=180)
            if item["type"] == "error":
                raise RuntimeError(item["message"])
            if item["type"] == "segment":
                segments.append(item)
            if item["type"] == "partial" and item["text"]:
                partials.append(item["text"])
            if item["type"] == kind:
                return item
    try:
        ready = wait("ready")
        print("ready:", ready, flush=True)
        stride = 8000 if ready["native"] else 64000
        started = time.monotonic()
        for offset in range(0, len(samples), stride):
            chunk = samples[offset:offset + stride]
            child.stdin.write(json.dumps({"samples": chunk, "start": offset / 16000, "end": (offset + len(chunk)) / 16000}) + "\n")
            child.stdin.flush()
            wait("ack")
        assert partials, "No text arrived while audio was still being fed"
        child.stdin.write(json.dumps({"finish": True, "start": len(samples) / 16000, "end": len(samples) / 16000}) + "\n")
        child.stdin.flush()
        wait("ack")
        child.stdin.close()
        assert child.wait(timeout=10) == 0
        assert segments, "No committed transcript after finish"
        assert all(s["end"] >= s["start"] >= 0 for s in segments)
        print(json.dumps({"audio_secs": len(samples)/16000, "inference_secs": round(time.monotonic()-started, 2), "partial_updates": len(partials), "segments": segments}, ensure_ascii=False, indent=2))
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        for reader in readers:
            reader.join(timeout=5)


if __name__ == "__main__":
    main()
