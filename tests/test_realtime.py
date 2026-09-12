"""Hermetic protocol tests: no microphone, model weights, GPU or downloads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

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


if __name__ == "__main__":
    unittest.main()
