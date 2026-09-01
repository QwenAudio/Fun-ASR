import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import compute_ja_cer, whisper_mix_normalize


class JapaneseNormalizerTest(unittest.TestCase):
    def test_kana_normalization_outputs_character_sequence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.txt"
            output = root / "output.txt"
            source.write_text("utt-1 今日は晴れです\n", encoding="utf-8")

            with mock.patch.object(
                whisper_mix_normalize.pyopenjtalk,
                "g2p",
                return_value="キョーワハレデス",
            ):
                whisper_mix_normalize.normalize_text(source, output, kana=True)

            self.assertEqual(
                "utt-1\tキ ョ ー ワ ハ レ デ ス\n",
                output.read_text(encoding="utf-8"),
            )

    def test_openjtalk_failure_is_not_silently_ignored(self):
        with (
            mock.patch.object(
                whisper_mix_normalize.pyopenjtalk,
                "g2p",
                side_effect=ValueError("invalid dictionary"),
            ),
            self.assertRaisesRegex(RuntimeError, "OpenJTalk failed"),
        ):
            whisper_mix_normalize.safe_ja_g2p("今日は晴れです")

    def test_path_id_text_format_excludes_path_and_id_from_g2p(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.txt"
            output = root / "output.txt"
            source.write_text("audio.wav utt-ja-1 今日は晴れです\n", encoding="utf-8")

            with mock.patch.object(
                whisper_mix_normalize.pyopenjtalk,
                "g2p",
                return_value="キョーワハレデス",
            ) as g2p:
                whisper_mix_normalize.normalize_text(
                    source,
                    output,
                    kana=True,
                    input_format="path-id-text",
                )

            g2p.assert_called_once_with("今日は晴れです", kana=True)
            self.assertEqual(
                "utt-ja-1\tキ ョ ー ワ ハ レ デ ス\n",
                output.read_text(encoding="utf-8"),
            )

    def test_explicit_dictionary_requires_sys_dic(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            self.assertRaisesRegex(RuntimeError, "missing sys.dic"),
        ):
            whisper_mix_normalize.configure_open_jtalk_dict(temp_dir)


class JapaneseCerTest(unittest.TestCase):
    def test_runs_compute_wer_on_normalized_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.txt"
            hypothesis = root / "hypothesis.txt"
            report = root / "cer.txt"
            normalized = root / "normalized"
            reference.write_text("utt-1 今日は晴れです\n", encoding="utf-8")
            hypothesis.write_text("utt-1 きょうは晴れです\n", encoding="utf-8")

            def fake_normalize(source, destination, **kwargs):
                del source, kwargs
                Path(destination).write_text(
                    "utt-1\tキ ョ ー ワ ハ レ デ ス\n", encoding="utf-8"
                )

            with (
                mock.patch.object(
                    compute_ja_cer, "normalize_text", side_effect=fake_normalize
                ) as normalize,
                mock.patch.object(
                    compute_ja_cer.shutil, "which", return_value="/usr/bin/compute-wer"
                ),
                mock.patch.object(compute_ja_cer.subprocess, "run") as run,
            ):
                result = compute_ja_cer.compute_ja_cer(
                    reference,
                    hypothesis,
                    report,
                    normalized_dir=normalized,
                )

            self.assertEqual(report, result)
            self.assertEqual(2, normalize.call_count)
            run.assert_called_once_with(
                [
                    "/usr/bin/compute-wer",
                    str(normalized / "reference.kana.txt"),
                    str(normalized / "hypothesis.kana.txt"),
                    str(report),
                ],
                check=True,
            )

    @unittest.skipUnless(
        shutil.which("compute-wer") is not None,
        "compute-wer is not installed",
    )
    def test_real_japanese_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.txt"
            hypothesis = root / "hypothesis.txt"
            report = root / "cer.txt"
            reference.write_text("utt-1 今日は晴れです\n", encoding="utf-8")
            hypothesis.write_text("utt-1 きょうは晴れです\n", encoding="utf-8")

            compute_ja_cer.compute_ja_cer(reference, hypothesis, report)

            self.assertIn(
                "Overall -> 0.00 %",
                report.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
