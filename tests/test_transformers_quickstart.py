"""Offline contract tests; speech quality is verified separately on real audio."""
import importlib.util
import ast
import json
from pathlib import Path
import re
import unittest

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_example", ROOT / "examples/transformers/transcribe.py")
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class NativeExampleTests(unittest.TestCase):
    def test_runtime_choices_fail_closed_without_silent_cpu_fallback(self):
        validate = getattr(native, 'validate_runtime', None)
        self.assertTrue(callable(validate), 'Explicit runtime validation is missing')
        validate('cpu', 'float32')
        validate('cuda', 'float32', cuda_available=True)
        validate('cuda', 'bfloat16', cuda_available=True, bf16_supported=True)
        for device, dtype, available, bf16 in [
            ('cuda', 'float32', False, False),
            ('cuda', 'bfloat16', True, False),
            ('cpu', 'bfloat16', False, False),
            ('auto', 'float32', False, False),
            ('cuda', 'float16', True, True),
        ]:
            with self.subTest(device=device, dtype=dtype), self.assertRaises(ValueError):
                validate(device, dtype, cuda_available=available, bf16_supported=bf16)

    def test_gpu_recipe_keeps_cpu_requirements_separate(self):
        gpu = ROOT / 'examples/transformers/requirements-gpu.txt'
        self.assertTrue(gpu.is_file())
        requirements = gpu.read_text()
        self.assertIn('torch==2.11.0+cu128', requirements)
        self.assertIn('torchaudio==2.11.0+cu128', requirements)
        self.assertIn('transformers==5.17.0', requirements)
        self.assertIn('torch==2.10.0', (ROOT / 'examples/transformers/requirements.txt').read_text())
        guide = (ROOT / 'examples/transformers/README.md').read_text()
        for marker in ('--device cuda --dtype bfloat16', 'requirements-gpu.txt', 'H100', '550.127.08'):
            self.assertIn(marker, guide)

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


@pytest.mark.parametrize("suffix", ["", "_zh", "_ja", "_ko"])
@pytest.mark.parametrize("entry", ["directory", "table"])
def test_model_download_entries_distinguish_native_and_toolkit(suffix, entry):
    text = (ROOT / f"README{suffix}.md").read_text()
    if entry == "directory":
        line = next(line for line in text.splitlines()
                    if "**Fun-ASR-Nano**" in line and "**Fun-ASR-MLT-Nano**" in line)
        nano, mlt = line.split("**Fun-ASR-MLT-Nano**", 1)
    else:
        nano = next(line for line in text.splitlines()
                    if line.startswith("|") and "Fun-ASR-Nano <br>" in line)
        mlt = next(line for line in text.splitlines()
                   if line.startswith("|") and "Fun-ASR-MLT-Nano" in line)
    links = dict(re.findall(r"\[([^\]]+)\]\(([^)\s]+)\)", nano))
    assert links.get("HF / Transformers") == "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf"
    assert links.get("HF / FunASR") == "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512"
    assert "https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512" in mlt
    assert "Fun-ASR-Nano-2512-hf" not in mlt


if __name__ == "__main__":
    unittest.main()
