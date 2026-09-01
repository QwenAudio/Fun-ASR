#!/usr/bin/env python3
"""Normalize ASR transcripts before WER/CER scoring."""

import argparse
import os
import re
from pathlib import Path

import pyopenjtalk
import zhconv
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

if __package__:
    from . import cn_tn
    from . import format5res as cn_itn
else:  # Support ``python tools/whisper_mix_normalize.py``.
    import cn_tn
    import format5res as cn_itn

basic_normalizer = BasicTextNormalizer()
english_normalizer = EnglishTextNormalizer()


def is_only_chinese_and_english(s):
    # 定义正则表达式模式，匹配中文字符范围和英文字母（包括大小写）
    pattern = r"^[\u4e00-\u9fa5A-Za-z0-9,\.!\?:;，。！？：；、%\'\s\-\~]+$"
    # 使用正则表达式进行匹配
    return re.match(pattern, s) is not None


def is_only_english(s):
    # 定义正则表达式模式，匹配中文字符范围和英文字母（包括大小写）
    pattern = r"^[A-Za-z0-9,\.!\?:;，。！？：；、%\'\s\-\~]+$"
    # 使用正则表达式进行匹配
    return re.match(pattern, s) is not None


def is_number(s):
    # 定义正则表达式模式，匹配中文字符范围和英文字母（包括大小写）
    pattern = r"^[0-9,\.!\?:;，。！？：；、%\'\s]+$"
    # 使用正则表达式进行匹配
    return re.match(pattern, s) is not None


def configure_open_jtalk_dict(dict_dir):
    """Configure an explicit OpenJTalk system dictionary directory."""
    if dict_dir is None:
        return

    dict_path = Path(dict_dir).expanduser().resolve()
    if not (dict_path / "sys.dic").is_file():
        raise RuntimeError(f"OpenJTalk dictionary is missing sys.dic: {dict_path}")

    # pyopenjtalk reads this value when its global frontend is initialized.
    os.environ["OPEN_JTALK_DICT_DIR"] = str(dict_path)
    pyopenjtalk.OPEN_JTALK_DICT_DIR = str(dict_path).encode("utf-8")


def safe_ja_g2p(text, kana=True, max_length=100):
    """Convert Japanese text with OpenJTalk and fail on conversion errors."""
    if len(text) > max_length:
        parts = []
        for i in range(0, len(text), max_length):
            part = text[i : i + max_length]
            try:
                converted = pyopenjtalk.g2p(part, kana=kana)
                parts.append(converted)
            except Exception as exc:
                raise RuntimeError(
                    f"OpenJTalk failed to normalize Japanese text: {part[:80]!r}"
                ) from exc
        return " ".join(parts)

    try:
        return pyopenjtalk.g2p(text, kana=kana)
    except Exception as exc:
        raise RuntimeError(
            f"OpenJTalk failed to normalize Japanese text: {text[:80]!r}"
        ) from exc


def normalize_text(
    srcfn,
    dstfn,
    kana=False,
    open_jtalk_dict=None,
    input_format="id-text",
):
    """Normalize an utterance transcript file for ASR scoring."""
    if kana:
        configure_open_jtalk_dict(open_jtalk_dict)
    if input_format not in {"id-text", "path-id-text"}:
        raise ValueError(f"unsupported input format: {input_format}")

    with (
        open(srcfn, "r", encoding="utf-8") as f_read,
        open(dstfn, "w", encoding="utf-8") as f_write,
    ):
        all_lines = f_read.readlines()
        for line_number, line in enumerate(all_lines, start=1):
            line = line.strip()
            if not line:
                continue

            if input_format == "path-id-text":
                fields = line.split(maxsplit=2)
                if len(fields) < 2:
                    raise ValueError(
                        f"line {line_number} requires audio-path and utterance-id"
                    )
                key = fields[1]
                text = fields[2] if len(fields) == 3 else ""
            else:
                fields = line.split(maxsplit=1)
                key = fields[0]
                text = fields[1] if len(fields) == 2 else ""

            text = re.sub(r"=", " ", text)
            text = re.sub(r"\(", " ", text)
            text = re.sub(r"\)", " ", text)
            # From Chongjia Ni
            if kana:
                text = safe_ja_g2p(text, kana=True, max_length=100)

            line_arr = f"{key}\t{text}".split()
            conts = []
            language_bak = ""
            part = []
            for i in range(1, len(line_arr)):
                out_part = ""
                chn_eng_bool = is_only_chinese_and_english(line_arr[i])
                eng_bool = is_only_english(line_arr[i])
                num_bool = is_number(line_arr[i])
                if eng_bool and not num_bool:
                    language = "en"
                elif chn_eng_bool:
                    language = "chn_en"
                else:
                    language = "not_chn_en"
                if language == language_bak or language_bak == "":
                    part.append(line_arr[i])
                    language_bak = language
                else:
                    if language_bak == "en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part = cn_itn.scoreformat("", out_part1)
                    elif language_bak == "chn_en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    else:
                        out_part1 = basic_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    conts.append(out_part)
                    language_bak = language
                    part = []
                    part.append(line_arr[i])
                if i == len(line_arr) - 1:
                    if language == "en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part = cn_itn.scoreformat("", out_part1)
                    elif language == "chn_en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    else:
                        out_part1 = basic_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    conts.append(out_part)

            f_write.write("{}\t{}\n".format(key, " ".join(conts).strip()))


def main():
    parser = argparse.ArgumentParser(
        description="Normalize utterance-id/text files before ASR scoring."
    )
    parser.add_argument("srcfn", help="input file: utterance-id followed by text")
    parser.add_argument("dstfn", help="normalized output file")
    parser.add_argument(
        "legacy_ja_norm",
        nargs="?",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--language",
        choices=("auto", "ja"),
        default="auto",
        help="use OpenJTalk kana normalization for Japanese",
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

    kana = args.language == "ja" or args.legacy_ja_norm is not None
    if args.open_jtalk_dict and not kana:
        parser.error("--open-jtalk-dict requires --language ja")
    normalize_text(
        args.srcfn,
        args.dstfn,
        kana=kana,
        open_jtalk_dict=args.open_jtalk_dict,
        input_format=args.input_format,
    )


if __name__ == "__main__":
    main()
