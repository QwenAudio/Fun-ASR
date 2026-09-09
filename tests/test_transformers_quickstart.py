"""Offline contract tests; speech quality is verified separately on real audio."""
import importlib.util
import ast
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_example", ROOT / "examples/transformers/transcribe.py")
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class NativeExampleTests(unittest.TestCase):
    def test_fixed_native_artifact(self):
        self.assertEqual(native.MODEL_ID, "FunAudioLLM/Fun-ASR-Nano-2512-hf")
        self.assertEqual(native.REVISION, "d93b302ee7fd505e1b3576120fc142fc6f7820e1")

    def test_audio_is_resampled_and_downmixed_explicitly(self):
        out = native.prepare_audio(np.ones((48000, 2), dtype=np.float32), 48000)
        self.assertEqual(out.shape, (16000,))
        self.assertEqual(out.dtype, np.float32)

    def test_invalid_audio_is_rejected(self):
        for audio in [np.array([]), np.array([np.nan]), np.array([np.inf]), np.zeros((2, 2, 2))]:
            with self.subTest(shape=audio.shape), self.assertRaises(ValueError):
                native.prepare_audio(audio, 16000)
        with self.assertRaises(ValueError):
            native.prepare_audio(np.zeros(61 * 16000), 16000)
        with self.assertRaises(ValueError):
            native.prepare_audio(np.ones(10), 0)

    def test_language_contract(self):
        self.assertEqual(native.languages_for_batch(["en"], 2), ["en", "en"])
        self.assertEqual(native.languages_for_batch(["zh", "en"], 2), ["zh", "en"])
        for languages, count in [([], 1), (["en"], 0), (["zh", "en"], 3), (["ko"], 1)]:
            with self.subTest(languages=languages), self.assertRaises(ValueError):
                native.languages_for_batch(languages, count)

    def test_readmes_offer_native_before_toolkit_install(self):
        for suffix in ["", "_zh", "_ja", "_ko"]:
            text = (ROOT / f"README{suffix}.md").read_text()
            self.assertIn("examples/transformers/", text)
            self.assertLess(text.index("transformers==5.17.0"), text.index("pip install -r requirements.txt"))

    def test_notebook_is_unexecuted_and_python_cells_parse(self):
        notebook = json.loads((ROOT / "examples/colab/fun_asr_nano_transformers.ipynb").read_text())
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            if not source.startswith("%pip"):
                ast.parse(source)

    def test_clone_directory_and_legacy_anchors_remain_clear(self):
        guide = (ROOT / "examples/transformers/README.md").read_text()
        self.assertIn("git clone https://github.com/QwenAudio/Fun-ASR.git", guide)
        self.assertIn("cd Fun-ASR", guide)
        for suffix, anchor, heading in [("_ja", "主要機能", "主要機能"), ("_ko", "주요-기능", "주요 기능")]:
            text = (ROOT / f"README{suffix}.md").read_text()
            self.assertIn(f'<a name="{anchor}"></a>\n\n# {heading}', text)


if __name__ == "__main__":
    unittest.main()
