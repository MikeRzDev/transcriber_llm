"""Small real-MLX checks; opt in with TRANSCRIBE_TEST_MLX=1 on a GPU-capable host."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("rt", Path(__file__).parents[1] / "src/transcribe/realtime.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


class LayoutScopeTests(unittest.TestCase):
    def test_other_model_families_are_untouched_without_importing_mlx(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_asr"))
        self.assertIsNone(rt.optimize_voxtral_live(model, "/unused"))


@unittest.skipUnless(os.environ.get("TRANSCRIBE_TEST_MLX") == "1", "opt-in real MLX test")
class LayoutMLXTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        self.mx, self.nn, self.flatten = mx, nn, tree_flatten

    def model(self, bits):
        nn = self.nn
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(model_type="voxtral_realtime")
                self.decoder = nn.Module()
                layer = nn.Module()
                layer.attention = nn.Module()
                layer.attention.wq = nn.Linear(64, 64) if bits is None else nn.QuantizedLinear(64, 64, bits=bits)
                self.decoder.layers = [layer]
                self.decoder.tok_embeddings = nn.Embedding(64, 64)
                self.encoder = nn.Module()
                self.encoder.audio_language_projection_0 = nn.Linear(64, 64)
                self.encoder.audio_language_projection_2 = nn.Linear(64, 64)
                self.encoder.norm = nn.RMSNorm(64)
                self.hook_calls = 0

            @classmethod
            def post_load_hook(cls, model, path):
                model.hook_calls += 1
                return model
        return Model()

    def test_legacy_conversion_preserves_packed_weights_and_is_idempotent(self):
        import numpy as np
        model = self.model(4)
        packed = np.array(model.decoder.layers[0].attention.wq.weight)
        result = rt.optimize_voxtral_live(model, "/unused")
        self.assertEqual(len(result["converted_layers"]), 3)
        self.assertTrue(result["dtype_changed"])
        self.assertEqual(model.hook_calls, 1)
        self.assertIsInstance(model.decoder.tok_embeddings, self.nn.QuantizedEmbedding)
        np.testing.assert_array_equal(packed, np.array(model.decoder.layers[0].attention.wq.weight))
        for _, value in self.flatten(model.parameters()):
            if self.mx.issubdtype(value.dtype, self.mx.floating):
                self.assertEqual(value.dtype, self.mx.bfloat16)
        logits = model.decoder.tok_embeddings.as_linear(self.mx.ones((1, 64), dtype=self.mx.bfloat16))
        self.mx.eval(logits)
        self.assertTrue(np.isfinite(np.array(logits.astype(self.mx.float32))).all())
        again = rt.optimize_voxtral_live(model, "/unused")
        self.assertEqual(again["converted_layers"], [])
        self.assertFalse(again["dtype_changed"])
        self.assertEqual(model.hook_calls, 1)

    def test_full_precision_and_eight_bit_models_are_not_converted(self):
        for bits in (None, 8):
            model = self.model(bits)
            original = model.decoder.tok_embeddings
            self.assertIsNone(rt.optimize_voxtral_live(model, "/unused"))
            self.assertIs(model.decoder.tok_embeddings, original)
            self.assertEqual(model.hook_calls, 0)


if __name__ == "__main__":
    unittest.main()
