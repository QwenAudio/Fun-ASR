from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOSS_URL = "https://www.funasr.com/deploy/moss-transcribe-diarize.html"


def test_readmes_distinguish_nano_from_one_pass_moss_diarization():
    for name in ("README.md", "README_zh.md"):
        text = (ROOT / name).read_text(encoding="utf-8")

        assert "MOSS-Transcribe-Diarize" in text, name
        assert MOSS_URL in text, name

    assert "do not emit speaker labels by themselves" in (
        ROOT / "README.md"
    ).read_text(encoding="utf-8")
    assert "本身不输出说话人标签" in (ROOT / "README_zh.md").read_text(
        encoding="utf-8"
    )
