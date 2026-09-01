#!/usr/bin/env python3
"""Compute Japanese CER after OpenJTalk kana normalization."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

if __package__:
    from .whisper_mix_normalize import normalize_text
else:  # Support ``python tools/compute_ja_cer.py``.
    from whisper_mix_normalize import normalize_text


def compute_ja_cer(
    reference,
    hypothesis,
    output_file,
    normalized_dir=None,
    open_jtalk_dict=None,
    input_format="id-text",
):
    """Normalize reference/hypothesis to katakana and run ``compute-wer``."""
    compute_wer = shutil.which("compute-wer")
    if compute_wer is None:
        raise RuntimeError(
            "compute-wer is not installed; run `pip install compute-wer` first"
        )

    reference = Path(reference)
    hypothesis = Path(hypothesis)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if normalized_dir is not None:
        normalized_path = Path(normalized_dir)
        normalized_path.mkdir(parents=True, exist_ok=True)
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="fun_asr_ja_cer_")
        normalized_path = Path(temp_context.name)

    try:
        reference_norm = normalized_path / "reference.kana.txt"
        hypothesis_norm = normalized_path / "hypothesis.kana.txt"
        normalize_text(
            reference,
            reference_norm,
            kana=True,
            open_jtalk_dict=open_jtalk_dict,
            input_format=input_format,
        )
        normalize_text(
            hypothesis,
            hypothesis_norm,
            kana=True,
            open_jtalk_dict=open_jtalk_dict,
            input_format=input_format,
        )
        subprocess.run(
            [compute_wer, str(reference_norm), str(hypothesis_norm), str(output_file)],
            check=True,
        )
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Japanese reference and hypothesis text to katakana with "
            "OpenJTalk, then compute character error rate with compute-wer."
        )
    )
    parser.add_argument("reference", help="reference: utterance-id followed by text")
    parser.add_argument("hypothesis", help="hypothesis: utterance-id followed by text")
    parser.add_argument("output_file", help="alignment and CER report")
    parser.add_argument(
        "--normalized-dir",
        help="keep normalized reference.kana.txt and hypothesis.kana.txt here",
    )
    parser.add_argument(
        "--open-jtalk-dict",
        help="optional OpenJTalk dictionary directory containing sys.dic",
    )
    parser.add_argument(
        "--input-format",
        choices=("id-text", "path-id-text"),
        default="id-text",
        help="input columns: 'utterance-id text' or 'audio-path utterance-id text'",
    )
    args = parser.parse_args()

    compute_ja_cer(
        args.reference,
        args.hypothesis,
        args.output_file,
        normalized_dir=args.normalized_dir,
        open_jtalk_dict=args.open_jtalk_dict,
        input_format=args.input_format,
    )


if __name__ == "__main__":
    main()
