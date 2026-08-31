from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_readmes_document_hub_specific_timestamp_checkpoint_state():
    for name in ("README.md", "README_zh.md", "README_ja.md", "README_ko.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "ModelScope" in text, name
        assert "Hugging Face" in text, name
        assert "272c57b82523ada6fd87095e955f8e29100979ab" in text, name
        assert "81fec8616083c69377f3ceef36aba3655660ee0ca69a5d4a1e9810cd340ca499" in text, name
        assert "55ae0d2fee369f0f11cce0795f6927934ad17cf11b278a7e56a51272074160bb" in text, name


def test_readme_does_not_claim_every_released_checkpoint_lacks_ctc_weights():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "The released Fun-ASR-Nano `model.pt` checkpoint does not include" not in text
