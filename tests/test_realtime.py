"""Hermetic protocol tests: no microphone, model weights, GPU or downloads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile

spec = importlib.util.spec_from_file_location("realtime", Path(__file__).parents[1] / "src/transcribe/realtime.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


class BridgeTests(unittest.TestCase):
    def transcript(self):
        events = []
        return rt.Transcript(lambda kind, **values: events.append((kind, values))), events

    def test_string_deltas_become_one_committed_span(self):
        class Voxtral:
            def generate(self, audio, stream=False, verbose=False, max_tokens=0):
                assert stream
                yield "Hola"
                yield ", mundo."
        transcript, events = self.transcript()
        transcript.start, transcript.end = 1.5, 4.0
        rt.transcribe_window(Voxtral(), [], None, transcript)
        partials = [v["text"] for k, v in events if k == "partial"]
        self.assertEqual(partials, ["Hola", "Hola, mundo.", ""])
        self.assertEqual([v for k, v in events if k == "segment"], [{"text": "Hola, mundo.", "start": 1.5, "end": 4.0}])

    def test_qwen_structured_deltas_and_language(self):
        class Qwen:
            def generate(self, audio, stream=False, language=None):
                assert language == "es"
                yield SimpleNamespace(text="Buenos ", language="Spanish")
                yield SimpleNamespace(text="días", language="Spanish")
                yield SimpleNamespace(text="", language="Spanish", is_final=True)
        transcript, events = self.transcript()
        rt.transcribe_window(Qwen(), [], "es", transcript)
        self.assertEqual([v["text"] for k, v in events if k == "segment"], ["Buenos días"])
        self.assertEqual(transcript.language, "Spanish")

    def test_nonstreaming_model_and_empty_transcript(self):
        class Offline:
            def generate(self, audio):
                return SimpleNamespace(text="", language=None)
        transcript, events = self.transcript()
        rt.transcribe_window(Offline(), [], "es", transcript)
        self.assertFalse(any(k == "segment" for k, _ in events))

    def test_controls_are_removed_without_losing_unicode(self):
        self.assertEqual(rt.clean_text("<asr_text>¿Qué tal? 中文<|endoftext|>"), "¿Qué tal? 中文")


class FakeNativeSession:
    input_sample_rate = 16000

    def __init__(self):
        self.done = False
        self.closed = False
        self.chunks = []
        self.flush_steps = 0

    def feed(self, samples):
        self.chunks.append(list(samples))

    def close(self):
        self.closed = True

    def step(self, max_decode_tokens):
        if self.closed:
            self.flush_steps += 1
            if self.flush_steps == 2:
                self.done = True
                return ["complete."]
            return ["delayed "]
        return ["speech "]


class NativeStreamTests(unittest.TestCase):
    def stream(self, **kwargs):
        sessions, events = [], []
        class Model:
            def create_streaming_session(self, max_tokens):
                assert max_tokens == 4096
                session = FakeNativeSession()
                sessions.append(session)
                return session
        transcript = rt.Transcript(lambda kind, **values: events.append((kind, values)))
        return rt.NativeStream(Model(), transcript, **kwargs), sessions, events

    def test_boundary_splits_preserve_every_sample_and_flush_delayed_words(self):
        stream, sessions, events = self.stream(max_seconds=2)
        samples = list(range(80005))
        end = len(samples) / 16000
        stream.feed(samples, 0, end, finish=True)
        self.assertEqual(len(sessions), 3)
        actual = [s for session in sessions for chunk in session.chunks for s in chunk]
        self.assertEqual(actual, samples)
        self.assertTrue(all(sum(map(len, session.chunks)) <= 32000 for session in sessions))
        self.assertTrue(all(session.done and session.flush_steps == 2 for session in sessions))
        segments = [values for kind, values in events if kind == "segment"]
        self.assertEqual([(s["start"], s["end"]) for s in segments], [(0, 2), (2, 4), (4, end)])
        self.assertTrue(all(s["text"].endswith("delayed complete.") for s in segments))
        self.assertIsNone(stream.session)
        stream.feed([], end, end, finish=True)
        self.assertEqual(len(sessions), 3, "finishing must not open another session")

    def test_long_stream_releases_history_and_reuses_model(self):
        released = []
        stream, sessions, _ = self.stream(max_seconds=2, on_reset=lambda: released.append(True))
        for i in range(80):
            stream.feed([0.1] * 8000, i / 2, (i + 1) / 2)
            self.assertLess(stream.samples, 32000)
        self.assertEqual(stream.closed_sessions, 20)
        self.assertEqual(len(sessions), 20)
        self.assertEqual(len(released), 20)
        self.assertIsNone(stream.session)

    def test_ten_minutes_of_quiet_does_no_inference_and_retains_onset(self):
        stream, sessions, events = self.stream()
        quiet = [0.0] * 16000
        for second in range(600):
            stream.feed(quiet, second, second + 1)
            self.assertFalse(stream.inference_active)
            self.assertTrue(stream.waiting_for_speech)
        self.assertEqual(sessions, [], "silence must never open a decoder session")
        self.assertEqual(events, [])
        self.assertEqual(len(stream.preroll), 16000)
        onset = [0.0001] * 4000 + [0.05] * 12000
        stream.feed(onset, 600, 601)
        self.assertTrue(stream.inference_active)
        self.assertFalse(stream.waiting_for_speech)
        self.assertEqual(len(sessions), 1)
        fed = [value for chunk in sessions[0].chunks for value in chunk]
        self.assertEqual(fed, quiet + onset)
        self.assertEqual(stream.transcript.start, 599)
        self.assertEqual(stream.transcript.end, 601)

    def test_quiet_after_speech_flushes_once_and_next_utterance_keeps_absolute_time(self):
        stream, sessions, events = self.stream()
        stream.feed([0.05] * 16000, 0, 1)
        stream.feed([0.0] * 8000, 1, 1.5)
        self.assertFalse(stream.idle)
        stream.feed([0.0] * 8000, 1.5, 2)
        self.assertTrue(sessions[0].done)
        self.assertTrue(stream.waiting_for_speech)
        segments = [v for k, v in events if k == "segment"]
        self.assertTrue(segments[-1]["text"].endswith("delayed complete."))
        flushes = stream.closed_sessions
        for second in range(2, 602):
            stream.feed([0.0] * 16000, second, second + 1)
        self.assertEqual(stream.closed_sessions, flushes)
        self.assertEqual(len(sessions), 1)
        stream.feed([0.05] * 16000, 602, 603)
        self.assertEqual(len(sessions), 2)
        self.assertEqual(stream.transcript.start, 601)
        self.assertEqual(sum(map(len, sessions[1].chunks)), 32000)
        stream.feed([], 603, 603, finish=True)
        self.assertTrue(sessions[1].done)

    def test_latency_profile_preroll_preserves_onset_and_emits_detected_start(self):
        stream, sessions, events = self.stream(preroll_seconds=0.25)
        for i in range(10):
            stream.feed([0.0] * 1600, i / 10, (i + 1) / 10)
        self.assertEqual(len(stream.preroll), 4000)
        onset = [0.0] * 320 + [0.05] * 1280
        stream.feed(onset, 1, 1.1)
        starts = [v["start"] for k, v in events if k == "speech_start"]
        self.assertEqual(starts, [1.02])
        self.assertEqual(sessions[0].chunks[0], [0.0] * 4000 + onset)
        self.assertEqual(stream.transcript.start, 0.75)
        self.assertLess(next(i for i, (k, _) in enumerate(events) if k == "speech_start"),
                        next(i for i, (k, _) in enumerate(events) if k == "partial"))

    def test_warmup_finishes_temporary_context_without_reloading_model(self):
        sessions, options = [], []
        class Model:
            def create_streaming_session(self, **kwargs):
                options.append(kwargs)
                session = FakeNativeSession()
                sessions.append(session)
                return session
        model = Model()
        rt.warm_native(model, [0.0] * 16000, 80)
        self.assertEqual(options, [{"max_tokens": 4096, "transcription_delay_ms": 80}])
        self.assertTrue(sessions[0].done)
        self.assertEqual(sessions[0].flush_steps, 2)

    def test_low_level_background_noise_stays_idle(self):
        stream, sessions, events = self.stream()
        noise = [0.001, -0.001] * 8000
        for second in range(10):
            stream.feed(noise, second, second + 1)
        self.assertEqual(sessions, [])
        self.assertEqual(events, [])
        self.assertTrue(stream.waiting_for_speech)
        self.assertLessEqual(len(stream.preroll), 16000)

    def test_short_signal_is_not_diluted_by_a_long_quiet_packet(self):
        stream, sessions, _ = self.stream()
        samples = [0.0] * 15360 + [0.004] * 320 + [0.0] * 320
        stream.feed(samples, 0, 1)
        self.assertTrue(stream.inference_active)
        self.assertEqual(sessions[0].chunks[0], samples)

    def test_quiet_boundary_is_preferred_after_minimum_duration(self):
        stream, sessions, events = self.stream(max_seconds=20, pause_after=1)
        stream.feed([0.1] * 8000, 0, 0.5)
        self.assertFalse(sessions[0].closed)
        stream.feed([0.0] * 8000, 0.5, 1.0)
        self.assertTrue(sessions[0].closed)
        self.assertEqual(stream.closed_sessions, 1)
        self.assertEqual([v["end"] for k, v in events if k == "segment"], [1.0])


class LivePriorityTests(unittest.TestCase):
    def test_benchmark_yields_to_live_sessions_without_blocking_them(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "live.lock"
            with rt.LivePriority(benchmark=True, path=path) as benchmark:
                benchmark.check()
                with rt.LivePriority(path=path) as first:
                    first.check()
                    with rt.LivePriority(path=path) as second:
                        second.check()
                        with self.assertRaisesRegex(RuntimeError, "interactive live transcription"):
                            benchmark.check()
                    with self.assertRaises(RuntimeError):
                        benchmark.check()
                benchmark.check()
            self.assertIsNone(benchmark.fd)


if __name__ == "__main__":
    unittest.main()
