# Japanese CER evaluation

Fun-ASR evaluates Japanese transcripts after converting both the reference and
the hypothesis to katakana with OpenJTalk. This makes the score depend on the
pronunciation rather than the original mix of kanji, hiragana, and katakana.
The normalized character sequences are then aligned by `compute-wer`.

## Installation

The repository requirements already include `pyopenjtalk-plus` and
`compute-wer`:

```bash
pip install -r requirements.txt
```

`pyopenjtalk-plus` ships with an OpenJTalk-compatible dictionary. To reproduce
a benchmark that used a particular OpenJTalk 1.11 dictionary, pass that
dictionary directory explicitly with `--open-jtalk-dict`.

## Input format

Reference and hypothesis files use one utterance per line. The first
whitespace-separated field is the utterance ID and the remaining text is the
transcript. IDs must match between the two files.

```text
utt-001 今日は晴れです
utt-002 音声認識のテストです
```

Some evaluation pipelines write three columns: audio path, utterance ID, and
text. Select that format explicitly so the path and ID are not passed to
OpenJTalk:

```text
/data/001.wav utt-001 今日は晴れです
```

```bash
python tools/compute_ja_cer.py \
  label_domain \
  text_domain \
  japanese_cer.txt \
  --input-format path-id-text
```

## One-command evaluation

```bash
python tools/compute_ja_cer.py \
  reference.txt \
  hypothesis.txt \
  japanese_cer.txt
```

Keep the katakana inputs used for scoring when an audit or a regression test
needs them:

```bash
python tools/compute_ja_cer.py \
  reference.txt \
  hypothesis.txt \
  japanese_cer.txt \
  --normalized-dir normalized
```

Use an exact OpenJTalk dictionary for reproducible comparisons with an
existing benchmark:

```bash
python tools/compute_ja_cer.py \
  reference.txt \
  hypothesis.txt \
  japanese_cer.txt \
  --open-jtalk-dict /path/to/open_jtalk_dic_utf_8-1.11
```

For example, `今日は晴れです` and `きょうは晴れです` both normalize to:

```text
キ ョ ー ワ ハ レ デ ス
```

OpenJTalk predicts readings from context. Orthographic variants are only
treated as equal when their predicted katakana sequences are equal. For
example, `良い` may be read as `ヨイ`, while `いい` is read as `イイ`.

## Normalization only

The underlying normalizer remains available independently:

```bash
python tools/whisper_mix_normalize.py \
  reference.txt \
  reference.kana.txt \
  --language ja
```

Japanese conversion failures are fatal. The tool does not silently score the
original unnormalized text when OpenJTalk or its dictionary fails.
