"""Verify silence-then-speech through the real live bridge, without a microphone.

Paces a quiet interval followed by a speech WAV at capture speed. The test
refuses to start alongside a running app, yields to newer live sessions, has a
wall-time alarm, saves partial results, and never installs a model-fit rating.
"""
import argparse
from array import array
import importlib.util
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/benchmark-realtime.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class SilenceThenSpeech:
    def __init__(self, quiet_seconds, speech):
        self.quiet_samples = round(quiet_seconds * 16000)
        self.speech = speech

    def __len__(self):
        return self.quiet_samples + len(self.speech)

    def __getitem__(self, index):
        start, stop, step = index.indices(len(self))
        assert step == 1
        quiet = [0.0] * max(0, min(stop, self.quiet_samples) - start)
        return quiet + self.speech[max(0, start - self.quiet_samples):max(0, stop - self.quiet_samples)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--quiet-seconds", type=int, default=600)
    parser.add_argument("--speech-seconds", type=int, default=24)
    parser.add_argument("--batch-seconds", type=float, default=0.1)
    parser.add_argument("--max-first-text-latency", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/voxtral-idle-resume.json")
    args = parser.parse_args()
    if args.quiet_seconds < 1 or args.speech_seconds < 1:
        parser.error("Durations must be positive")
    running = subprocess.run(["pgrep", "-x", "transcribe-stt"], capture_output=True)
    if running.returncode == 0:
        parser.error("The live app is running; close it before this GPU check")
    if running.returncode != 1:
        parser.error("Cannot verify that the live app is closed")
    with wave.open(str(args.audio)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        pcm = array("h", wav.readframes(args.speech_seconds * 16000))
    if sys.byteorder != "little":
        pcm.byteswap()
    speech = [s / 32768 for s in pcm]
    assert speech and max(map(abs, speech)) > 0.01
    source = SilenceThenSpeech(args.quiet_seconds, speech)
    onset = next(start / 16000 for start in range(0, len(speech), 320)
                 if sum(s*s for s in speech[start:start+320]) >= len(speech[start:start+320]) * 0.003**2)
    speech_onset = args.quiet_seconds + onset
    def deadline(*_):
        raise TimeoutError("Idle-resume test exceeded its wall-time limit")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.quiet_seconds + args.speech_seconds + 120)
    def progress(snapshot):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".progress.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        print(f"{snapshot['audio_seconds']:g}s: idle={snapshot.get('idle')} inference_active={snapshot.get('inference_active')} resets={snapshot.get('closed_sessions')} active GPU={snapshot.get('active_gpu_bytes', 0)/1024**2:.0f} MiB", flush=True)
    print(f"Checking {args.quiet_seconds}s of paced silence, then {len(speech)/16000:g}s speech; no microphone", flush=True)
    started = time.monotonic()
    result = bench.trial(args.model, source, "es", args.batch_seconds, 90, paced=True, diagnostics=True, progress=progress)
    signal.alarm(0)
    report = {"test": "paced silence then speech", "quiet_seconds": args.quiet_seconds,
              "speech_seconds": len(speech)/16000, "wall_seconds": time.monotonic()-started,
              "result": result}
    if result["status"] == "ok":
        quiet = [c for c in result["chunks"] if c["end"] <= args.quiet_seconds]
        voiced = [c for c in result["chunks"] if c["start"] >= args.quiet_seconds and c.get("inference_active")]
        words = re.findall(r"\w+", " ".join(result["segments"]).casefold())
        first_text = next((p for p in result.get("partials", []) if p["received_seconds"] >= speech_onset), None)
        latency = first_text["received_seconds"] - speech_onset if first_text else None
        report["speech_onset_seconds"] = speech_onset
        report["first_text_latency_seconds"] = latency
        report["first_text"] = first_text["text"] if first_text else None
        report["checks"] = {
            "first_text_within_budget": latency is not None and latency < args.max_first_text_latency,
            "no_quiet_inference": all(c.get("inference_active") is False for c in quiet),
            "no_quiet_decoder_sessions": all(c.get("closed_sessions") == 0 and c.get("session_seconds") == 0 for c in quiet),
            "speech_resumed": bool(voiced),
            "opening_words_preserved": words[:2] == ["buenos", "días"],
        }
        report["first_speech_request_seconds"] = voiced[0]["elapsed"] if voiced else None
        report["quiet_active_gpu_range"] = [min(c["active_gpu_bytes"] for c in quiet), max(c["active_gpu_bytes"] for c in quiet)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "result"}, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved: {args.output}", flush=True)
    return 0 if result["status"] == "ok" and all(report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
