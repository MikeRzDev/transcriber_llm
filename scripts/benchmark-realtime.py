"""Benchmark the installed live bridge without microphone capture or downloads.

Use the Python environment containing mlx-audio. Example:
  .venv-realtime/bin/python scripts/benchmark-realtime.py \
    --models-dir /path/to/models --audio samples/benchmark-es.wav --language es

Runs models sequentially. Native models sweep feed sizes; windowed models use
4 seconds, matching continuous speech in the app. JSON retains per-request
measurements. Backlog is simulated from measured service times, not measured
from a real microphone. This is a throughput test, not an accuracy evaluation.
"""
import argparse
from array import array
from collections import deque
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
import os
from pathlib import Path
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
import wave

BRIDGE = Path(__file__).resolve().parents[1] / "src/transcribe/realtime.py"


class LoopedAudio:
    """Repeat a short fixture without allocating the full soak-test duration."""
    def __init__(self, samples, length):
        self.samples = samples
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            return [self.samples[i % len(self.samples)] for i in range(start, stop, step)]
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.samples[index % len(self.samples)]


def wait_for_audio(started, audio_end, clock=time.perf_counter, sleep=time.sleep):
    """Use absolute capture deadlines; inference delays must not shift capture."""
    delay = started + audio_end - clock()
    if delay > 0:
        sleep(delay)


def fingerprint(model):
    files = [model / "config.json", *model.rglob("*.safetensors")]
    return [{"name": str(path.relative_to(model)), "size": path.stat().st_size,
             "modified_ns": str(path.stat().st_mtime_ns)} for path in sorted(files)]


def install_report(report):
    configured = os.environ.get("TRANSCRIBE_STT_CONFIG")
    folder = Path(configured).parent if configured else Path.home() / ".config/transcribe-stt"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "live-benchmarks.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(target)
    return target


def unavailable(model):
    config = json.loads((model / "config.json").read_text())
    if "forcedaligner" in model.name.lower() or config.get("model_type") == "qwen3_forced_aligner":
        return "Forced aligner requires a transcript; not a live ASR model"
    if not config.get("model_type") and "multimodal" in config:
        return "voxmlx conversion is incompatible with the app's mlx-audio live bridge"
    return None


def summarize(chunks, finish_seconds):
    # Exclude the first four seconds from sustained speed to separate initial
    # prefill/compilation. Keep startup in end-to-end timing and queue estimates.
    warm = [c for c in chunks if c["start"] >= 4.0]
    if not warm:
        raise ValueError("Need more than four seconds of audio for sustained speed")
    total_audio = chunks[-1]["end"]
    inference = sum(c["elapsed"] for c in chunks)
    rtf = sum(c["elapsed"] for c in warm) / sum(c["end"] - c["start"] for c in warm)
    clock = peak = 0.0
    for chunk in chunks:
        clock = max(clock, chunk["end"]) + chunk["elapsed"]
        peak = max(peak, max(0.0, min(clock, total_audio) - chunk["end"]))
    return {
        "audio_seconds": total_audio,
        "first_chunk_seconds": chunks[0]["elapsed"],
        "sustained_rtf": rtf,
        "end_to_end_rtf": (inference + finish_seconds) / total_audio,
        "finish_seconds": finish_seconds,
        "max_request_seconds": max(c["elapsed"] for c in chunks),
        "simulated_peak_queue_seconds": peak,
        "simulated_drain_after_stop_seconds": max(0.0, clock + finish_seconds - total_audio),
        "queue_growth_seconds_per_wall_minute": max(0.0, 1.0 - 1.0 / rtf) * 60 if rtf else 0.0,
        # Only a measured allowance for THIS clip; not a universal safe limit.
        "observed_buffer_allowance_seconds": math.ceil(peak * 1.25 + 1) if rtf < 1 else None,
    }


def trial(model, samples, language, batch, timeout, paced=False, diagnostics=False, progress=None):
    cmd = [sys.executable, "-u", str(BRIDGE), str(model)]
    # Match the Rust mlx_language mapping for Qwen.
    hint = "Spanish" if language == "es" and "qwen" in model.name.lower() else language
    if hint:
        cmd.append(hint)
    messages = queue.Queue()
    errors = deque(maxlen=20)
    started = time.perf_counter()
    child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True,
                             env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                                  "TRANSCRIBE_STT_LIVE_METRICS": "1" if diagnostics else "0",
                                  "TRANSCRIBE_STT_BENCHMARK": "1"})
    def read_stdout():
        try:
            for line in child.stdout:
                item = json.loads(line)
                item["_received_at"] = time.perf_counter()
                messages.put(item)
        except Exception as error:
            messages.put({"type": "error", "message": str(error)})
        finally:
            messages.put({"type": "error", "message": "Bridge exited unexpectedly"})
    def read_stderr():
        for line in child.stderr:
            errors.append(line.rstrip())
    threads = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
    for thread in threads:
        thread.start()
    segments = []
    partials = []
    last_ack = {}
    def wait(kind):
        deadline = time.monotonic() + timeout
        while True:
            try:
                item = messages.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty:
                raise TimeoutError(f"No {kind} within {timeout}s") from None
            if item["type"] == "error":
                raise RuntimeError(item["message"])
            if item["type"] == "segment":
                segments.append(item["text"])
            if item["type"] == "partial" and any(c.isalnum() for c in item.get("text", "")):
                partials.append({"text": item["text"], "received_at": item["_received_at"]})
            if item["type"] == kind:
                return item
            if time.monotonic() >= deadline:
                raise TimeoutError(f"No {kind} within {timeout}s")
    def request(payload):
        nonlocal last_ack
        started = time.perf_counter()
        child.stdin.write(json.dumps(payload) + "\n")
        child.stdin.flush()
        last_ack = wait("ack")
        return time.perf_counter() - started
    chunks = []
    try:
        ready = wait("ready")
        load_seconds = time.perf_counter() - started
        native = ready["native"]
        feed = batch if native else 4.0
        stride = round(feed * 16000)
        chunks = []
        capture_started = time.perf_counter()
        next_progress = 60.0
        for offset in range(0, len(samples), stride):
            chunk = samples[offset:offset + stride]
            start, end = offset / 16000, (offset + len(chunk)) / 16000
            if paced:
                wait_for_audio(capture_started, end)
            elapsed = request({"samples": chunk, "start": start, "end": end})
            chunks.append({"start": start, "end": end, "elapsed": elapsed,
                           **{k: last_ack[k] for k in ("session_seconds", "closed_sessions", "active_gpu_bytes", "cached_gpu_bytes", "idle", "inference_active", "inference_seconds") if k in last_ack}})
            if paced:
                chunks[-1]["queue_after_ack_seconds"] = max(0.0, min(time.perf_counter() - capture_started, len(samples) / 16000) - end)
            if progress is not None and end >= next_progress:
                recent = [c for c in chunks if c["end"] > end - 60]
                snapshot = {"audio_seconds": end,
                            "wall_seconds": time.perf_counter() - capture_started,
                            "last_minute_rtf": sum(c["elapsed"] for c in recent) / sum(c["end"] - c["start"] for c in recent),
                            "peak_queue_seconds": max(c.get("queue_after_ack_seconds", 0.0) for c in chunks),
                            **{k: last_ack[k] for k in ("session_seconds", "closed_sessions", "active_gpu_bytes", "cached_gpu_bytes", "idle", "inference_active", "inference_seconds") if k in last_ack}}
                progress(snapshot)
                next_progress = end + 60
        end = len(samples) / 16000
        finish = request({"samples": [], "start": end, "end": end, "finish": True})
        child.stdin.close()
        if child.wait(timeout=10) != 0:
            raise RuntimeError("Bridge exited with a nonzero status")
        return {"status": "ok", "native": native, "batch_seconds": feed,
                "load_seconds": load_seconds, "paced": paced,
                "wall_seconds": time.perf_counter() - capture_started,
                "observed_peak_queue_seconds": max((c.get("queue_after_ack_seconds", 0.0) for c in chunks), default=0.0) if paced else None,
                "session_limit_seconds": ready.get("session_limit_seconds"),
                "optimization": ready.get("optimization"),
                "transcription_delay_ms": ready.get("transcription_delay_ms"),
                "runtime": ready.get("runtime"),
                "closed_sessions": last_ack.get("closed_sessions"),
                **summarize(chunks, finish),
                "segments": segments, "chunks": chunks,
                "partials": [{"text": p["text"], "received_seconds": p["received_at"] - capture_started} for p in partials],
                "warning": None if segments else "No transcript produced; throughput alone does not establish usability"}
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt) or "interactive live transcription has GPU priority" in str(error)
        return {"status": "interrupted" if interrupted else "error", "batch_seconds": batch,
                "error": "Benchmark interrupted" if isinstance(error, KeyboardInterrupt) else str(error),
                "stderr_tail": list(errors), "chunks": chunks,
                "audio_seconds": chunks[-1]["end"] if chunks else 0.0}
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        for thread in threads:
            thread.join(timeout=5)
        child.stdout.close()
        child.stderr.close()
        if not child.stdin.closed:
            child.stdin.close()


def recommend(trials):
    groups = {}
    for item in trials:
        if item["status"] == "ok" and item["segments"]:
            groups.setdefault(item["batch_seconds"], []).append(item)
    if not groups:
        return None
    choices = [{"batch_seconds": batch,
                "median_sustained_rtf": statistics.median(t["sustained_rtf"] for t in group),
                "worst_sustained_rtf": max(t["sustained_rtf"] for t in group),
                "max_observed_queue_seconds": max(t["simulated_peak_queue_seconds"] for t in group)}
               for batch, group in groups.items()]
    # Prefer lowest latency that leaves at least 20% compute headroom on every
    # repetition. Otherwise report best throughput, explicitly without safety.
    sustainable = [c for c in choices if c["worst_sustained_rtf"] <= 0.8]
    best = min(sustainable, key=lambda c: c["batch_seconds"]) if sustainable else min(choices, key=lambda c: c["median_sustained_rtf"])
    best["keeps_up_with_headroom"] = bool(sustainable)
    return best


def write_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    rows = ["# Local live transcription benchmark", "", f"Run: {report['timestamp']}", "",
            f"Hardware: {report.get('hardware', {}).get('machdep.cpu.brand_string', report['machine'])}; RAM: {int(report.get('hardware', {}).get('hw.memsize', 0)) / 1024**3:g} GiB. Input: {report['audio_seconds']:g}s; repetitions: {report['repeats']}.", "",
            f"Feed mode: {'paced at microphone speed' if report.get('paced') else 'unpaced throughput stress test'}.", "",
            "Same audio for every model. RTF = inference seconds / audio seconds; below 1 keeps up. Sustained RTF excludes the first four seconds. Queue figures simulate continuous capture from measured request times. Model loading happens before recording and is excluded from RTF.", "",
            "| Model | Selected feed | Sustained RTF | Peak queue | Result |", "|---|---:|---:|---:|---|"]
    for model in report["models"]:
        best = model.get("recommendation")
        if best:
            result = "Keeps up with 20% headroom" if best["keeps_up_with_headroom"] else "No tested feed has 20% headroom"
            rows.append(f"| {model['model']} | {best['batch_seconds']:g}s | {best['median_sustained_rtf']:.3f} | {best['max_observed_queue_seconds']:.2f}s | {result} |")
        else:
            reason = model.get("reason") or next((t["error"] for t in model.get("trials", []) if t["status"] != "ok"), "Running / no usable transcript")
            rows.append(f"| {model['model']} | — | — | — | {reason.replace('|', '/').replace(chr(10), ' ')} |")
    rows += ["", "These are measurements for this machine, clip, and workload, not exact per-model constants or an accuracy evaluation. A sustained RTF above 1 cannot be fixed with any finite buffer. An observed buffer allowance in JSON is only a 25% margin over this clip's peak plus one second. Longer recordings, noise, contention, and thermal changes need separate testing.", "", "## Trials", "", "| Model | Feed | Load | First request | Sustained RTF | Total RTF incl. flush | Peak queue |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in report["models"]:
        for t in model.get("trials", []):
            if t["status"] == "ok":
                rows.append(f"| {model['model']} | {t['batch_seconds']:g}s | {t['load_seconds']:.2f}s | {t['first_chunk_seconds']:.2f}s | {t['sustained_rtf']:.3f} | {t['end_to_end_rtf']:.3f} | {t['simulated_peak_queue_seconds']:.2f}s |")
    for model in report["models"]:
        validation = model.get("long_run_validation")
        if validation:
            rows += ["", f"{model['model']}: summary rating uses a {validation['audio_seconds']:g}s sustained validation with bounded context ({validation['sustained_rtf']:.3f} RTF, {validation['closed_sessions']} context flushes, model weights kept loaded). See [Voxtral optimization results](voxtral-optimization.md)."]
        if model.get("sweep_note"):
            rows += ["", f"{model['model']}: {model['sweep_note']}"]
    output.with_suffix(".md").write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--language", default="es")
    parser.add_argument("--seconds", type=float, default=24)
    parser.add_argument("--batches", type=float, nargs="+", default=[0.5, 1, 2, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--paced", action="store_true", help="Feed audio at microphone speed instead of saturating the GPU")
    parser.add_argument("--loop-audio", action="store_true", help="Repeat the fixture up to --seconds using bounded memory")
    parser.add_argument("--diagnostics", action="store_true", help="Record active/cached GPU memory and streaming-context age")
    parser.add_argument("--install", action="store_true", help="Install measured live-fit labels for the app model picker")
    parser.add_argument("--resume", action="store_true", help="Resume matching saved measurements")
    parser.add_argument("--max-sweep-rtf", type=float, default=2.0,
                        help="Skip more feed sizes when every baseline run exceeds this RTF; 0 means full sweep")
    parser.add_argument("--model", action="append", help="Optional exact folder name; repeat to select multiple")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/realtime.json"))
    args = parser.parse_args()
    if args.seconds <= 4 or args.repeats < 1 or any(not 0.1 <= b <= 4 for b in args.batches):
        parser.error("Use >4 seconds of audio, >=1 repeat, and batches between 0.1 and 4 seconds")
    with wave.open(str(args.audio)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            parser.error("Audio must be 16 kHz mono PCM16 WAV")
        pcm = array("h", wav.readframes(min(wav.getnframes(), round(args.seconds * 16000))))
    if sys.byteorder != "little":
        pcm.byteswap()
    samples = [value / 32768 for value in pcm]
    if len(samples) <= 64000 or max(map(abs, samples)) < 0.01:
        parser.error("Need audible speech longer than four seconds")
    if args.loop_audio:
        samples = LoopedAudio(samples, round(args.seconds * 16000))
    runtime = {}
    for package in ("mlx", "mlx-audio", "numpy"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = "not installed"
    hardware = {}
    if sys.platform == "darwin":
        for key in ("hw.memsize", "hw.model", "machdep.cpu.brand_string"):
            probe = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            if probe.returncode == 0:
                hardware[key] = probe.stdout.strip()
    report = {"paced": args.paced, "loop_audio": args.loop_audio, "diagnostics": args.diagnostics,
              "runtime": runtime, "hardware": hardware, "timestamp": datetime.now(timezone.utc).isoformat(), "machine": platform.platform(),
              "python": sys.version, "audio": str(args.audio.resolve()),
              "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
              "audio_seconds": len(samples) / 16000, "language": args.language,
              "repeats": args.repeats, "models": []}
    if args.resume and args.output.exists():
        saved = json.loads(args.output.read_text())
        for key in ("audio_sha256", "audio_seconds", "language", "repeats", "hardware", "runtime"):
            if saved.get(key) != report[key]:
                parser.error(f"Cannot resume: {key} differs from saved report")
        for key in ("paced", "loop_audio", "diagnostics"):
            if saved.get(key, False) != report[key]:
                parser.error(f"Cannot resume: {key} differs from saved report")
        report["models"] = saved["models"]
        report["timestamp"] = saved["timestamp"]
        report["resumed_at"] = datetime.now(timezone.utc).isoformat()
    models = sorted(p for p in args.models_dir.iterdir() if (p / "config.json").is_file() and (not args.model or p.name in args.model))
    if not models:
        parser.error("No matching model folders with config.json")
    for model in models:
        item = next((m for m in report["models"] if m["path"] == str(model.resolve())), None)
        if item is None:
            item = {"model": model.name, "path": str(model.resolve()), "trials": []}
            report["models"].append(item)
        current_fingerprint = fingerprint(model)
        if item.get("fingerprint") and item["fingerprint"] != current_fingerprint:
            item["trials"] = []
            item.pop("recommendation", None)
            item.pop("sweep_note", None)
            item.pop("long_run_validation", None)
        item["fingerprint"] = current_fingerprint
        try:
            reason = unavailable(model)
        except (OSError, ValueError) as error:
            reason = f"Cannot read model config: {error}"
        if reason:
            item.update(status="skipped", reason=reason)
            print(f"SKIP {model.name}: {reason}", flush=True)
        else:
            item.pop("sweep_note", None)
            for batch in args.batches:
                baseline = [t for t in item["trials"] if t["status"] == "ok" and t["batch_seconds"] == args.batches[0]]
                if (batch != args.batches[0] and args.max_sweep_rtf > 0
                        and len(baseline) >= args.repeats
                        and all(t["sustained_rtf"] > args.max_sweep_rtf for t in baseline)):
                    item["sweep_note"] = f"Additional feed sizes not tested: every baseline run exceeded {args.max_sweep_rtf:g} RTF. Use --max-sweep-rtf 0 --resume for a full sweep."
                    print(f"  {item['sweep_note']}", flush=True)
                    break
                existing = [t for t in item["trials"] if t["status"] == "ok" and (t["batch_seconds"] == batch or not t["native"])]
                for repeat in range(args.repeats):
                    if repeat < len(existing):
                        result = existing[repeat]
                        continue
                    print(f"RUN {model.name}: feed {batch:g}s, repetition {repeat + 1}", flush=True)
                    def progress(snapshot):
                        status = {"model": model.name, "paced": args.paced, **snapshot}
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.with_suffix(".progress.json").write_text(json.dumps(status, indent=2) + "\n")
                        print(f"  {snapshot['audio_seconds']:g}s audio · last-minute RTF {snapshot['last_minute_rtf']:.3f} · queue peak {snapshot['peak_queue_seconds']:.2f}s · active GPU {snapshot.get('active_gpu_bytes', 0) / 1024**2:.0f} MiB", flush=True)
                    result = trial(model, samples, args.language, batch, args.timeout,
                                   paced=args.paced, diagnostics=args.diagnostics, progress=progress)
                    item["trials"].append(result)
                    if result["status"] == "interrupted":
                        item["reason"] = result["error"]
                        write_report(report, args.output)
                        print(f"STOPPED: {result['error']}; partial measurements saved to {args.output}", flush=True)
                        return
                    item["recommendation"] = recommend(item["trials"])
                    write_report(report, args.output)
                    if result["status"] == "ok":
                        print(f"  feed={result['batch_seconds']:g}s RTF={result['sustained_rtf']:.3f} peak_queue={result['simulated_peak_queue_seconds']:.2f}s", flush=True)
                    else:
                        print(f"  ERROR: {result['error']} {' '.join(result['stderr_tail'][-2:])}", flush=True)
                # Windowed models have one app feed policy; do not duplicate it.
                if result["status"] != "ok" or not result["native"]:
                    break
        write_report(report, args.output)
    if args.install:
        print(f"Installed model-picker ratings: {install_report(report)}", flush=True)
    print(f"Reports: {args.output} and {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
