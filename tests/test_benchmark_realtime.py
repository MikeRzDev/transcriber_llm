"""Check throughput/backlog calculations without loading model weights."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).parents[1] / "scripts/benchmark-realtime.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkTests(unittest.TestCase):
    def chunks(self, durations):
        return [{"start": i * 4, "end": (i + 1) * 4, "elapsed": t} for i, t in enumerate(durations)]

    def test_slow_model_has_no_finite_buffer_recommendation(self):
        result = bench.summarize(self.chunks([6, 6, 6, 6, 6, 6]), 0)
        self.assertEqual(result["sustained_rtf"], 1.5)
        self.assertAlmostEqual(result["queue_growth_seconds_per_wall_minute"], 20)
        self.assertIsNone(result["observed_buffer_allowance_seconds"])
        self.assertEqual(result["simulated_peak_queue_seconds"], 10)
        self.assertEqual(result["simulated_drain_after_stop_seconds"], 16)

    def test_startup_stall_is_in_backlog_but_not_sustained_speed(self):
        result = bench.summarize(self.chunks([16, 1, 1, 1, 1, 1]), 2)
        self.assertEqual(result["sustained_rtf"], 0.25)
        self.assertEqual(result["simulated_peak_queue_seconds"], 16)
        self.assertEqual(result["observed_buffer_allowance_seconds"], 21)
        self.assertAlmostEqual(result["end_to_end_rtf"], 23 / 24)

    def test_select_lowest_latency_with_headroom_on_every_repeat(self):
        trials = [{"status": "ok", "segments": ["speech"], "batch_seconds": batch,
                   "sustained_rtf": rtf, "simulated_peak_queue_seconds": 2}
                  for batch, rtf in [(0.5, 0.7), (0.5, 1.1), (1, 0.7), (1, 0.75), (2, 0.4)]]
        best = bench.recommend(trials)
        self.assertEqual(best["batch_seconds"], 1)
        self.assertTrue(best["keeps_up_with_headroom"])

    def test_real_subprocess_protocol_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = Path(directory) / "fake.py"
            bridge.write_text("""import json, sys
print(json.dumps({"type": "ready", "native": False}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"type": "segment", "text": "test speech"}), flush=True)
    print(json.dumps({"type": "ack"}), flush=True)
    if request.get("finish"):
        break
""")
            with patch.object(bench, "BRIDGE", bridge):
                result = bench.trial(Path("fake-model"), [0.1] * 128000, "es", 0.5, 5)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["batch_seconds"], 4)
            self.assertEqual(len(result["chunks"]), 2)
            self.assertTrue(result["segments"])

    def test_stalled_engine_times_out_and_is_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = Path(directory) / "fake.py"
            bridge.write_text("import time; time.sleep(60)")
            with patch.object(bench, "BRIDGE", bridge):
                result = bench.trial(Path("fake-model"), [], "es", 0.5, 0.1)
            self.assertEqual(result["status"], "error")
            self.assertIn("No ready", result["error"])

    def test_pacing_uses_capture_deadlines_without_accumulating_inference_delay(self):
        sleeps = []
        bench.wait_for_audio(100, 1, clock=lambda: 100.7, sleep=sleeps.append)
        self.assertAlmostEqual(sleeps[-1], 0.3)
        bench.wait_for_audio(100, 2, clock=lambda: 102.5, sleep=sleeps.append)
        self.assertEqual(len(sleeps), 1, "a late model must catch up without extra waiting")
        bench.wait_for_audio(100, 3, clock=lambda: 102.8, sleep=sleeps.append)
        self.assertAlmostEqual(sleeps[-1], 0.2)

    def test_hour_audio_repeats_without_allocating_an_hour_of_samples(self):
        audio = bench.LoopedAudio([0.1, 0.2, 0.3], 3600 * 16000)
        self.assertEqual(len(audio), 57600000)
        self.assertEqual(len(audio.samples), 3)
        self.assertEqual(audio[2:7], [0.3, 0.1, 0.2, 0.3, 0.1])
        self.assertEqual(audio[-1], 0.3)
        self.assertEqual(audio[len(audio)-2:len(audio)+10], [0.2, 0.3])
        with self.assertRaises(IndexError):
            _ = audio[len(audio)]

    def test_empty_transcript_cannot_be_recommended(self):
        self.assertIsNone(bench.recommend([{"status": "ok", "segments": []}]))


if __name__ == "__main__":
    unittest.main()
