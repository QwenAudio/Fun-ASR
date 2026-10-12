# Speaker-attributed ASR evaluation demo

This self-contained demo includes two scripts and sample annotations:

- `split_chunks_by_dur_sdr.py`: split audio and annotations into duration-based chunks, with `--skip-audio` for a metadata-only demo.
- `score_sdr.py`: compute DER, JER, WER, cpWER, and tcpWER.
- `jsonls/`: sample inputs, chunked annotations, mock predictions, and scoring results.

## Run

Requires Python 3.10+. Open a terminal in `sdr_eval_demo/` and run:

```bash
python3 -m pip install numpy scipy
bash demo.sh
```

All JSONL inputs and JSON/JSONL outputs are in `jsonls/`. No files outside this folder are needed for the default demo. From the parent directory, you can also run `bash sdr_eval_demo/demo.sh`.

`demo.sh` shows the three steps: chunking, attaching inference results, and scoring.

```text
example_in.jsonl
  -> example_in_chunked4min.jsonl
  -> example_in_chunked4min_modelInferResult.jsonl
  -> example_in_chunked4min_modelInferResult_resolved.jsonl
  -> example_in_chunked4min_modelInferResult_resolved.json
```

`jsonls/example_in.jsonl` contains annotations for two AISHELL-4 recordings. The corresponding WAV files are not included. By default, the demo writes chunk metadata without slicing audio, then copies reference text into `model_infer_result` as **mock predictions**. Scores from this run do not represent model performance; timestamp rounding can also introduce small nonzero errors.

For real audio, put the local WAV files at the `audio_path` locations relative to `sdr_eval_demo/` (for example, `sdr_eval_demo/audios/`), install ffmpeg, and remove `--skip-audio` from `demo.sh`. Audio chunks are written to `audio_chunks/`, which is created automatically.

Run model inference on each audio chunk and store the model's raw decoded text in the `model_infer_result` field of the corresponding record. Save these records to `jsonls/example_in_chunked4min_modelInferResult.jsonl` for scoring. When using the messages, send only system/user messages to the model; assistant messages contain the reference labels. To score your inference results without replacing them with mock predictions:

```bash
MOCK_INFERENCE=0 bash demo.sh
```

## Data format

Input records contain `audio_path`, `duration`, and `segments`. Each segment contains `start_time`, `end_time`, `speaker_id`, and `raw_transcript`.

Chunk records retain `segments_ref` and `chunk_range` in **source-audio time**. Reference text in assistant messages and model predictions use **chunk-relative time**, in seconds:

```text
[SPK00|0.00-1.25]Hello.
[SPK01|1.50-2.80]Good morning.
```

The scorer parses predictions into `segments_hyp` and adds the chunk start to their timestamps.

## Evaluation conventions

- Chunking greedily groups complete segments up to 240 seconds; a single long segment can exceed this limit. This is audio chunking, not a train/test split.
- Speaker matching is independent for each chunk. Speaker names may differ between reference and prediction.
- DER includes overlapping speech and is reported separately for each collar. JER is frame-based and does not use the DER collar.
- `mixed` tokenization uses CJK characters and other alphanumeric words, lowercases text, and discards punctuation. tcpWER uses interpolated token timestamps.
- `WER`, `cpWER`, and `tcpWER` use samples with hypothesis/reference token ratios in `[0.5, 2.0]`. The `*_all` fields include all samples with reference tokens. Check sample counts when interpreting scores.
- Numerical equivalence to external scoring tools has not been comprehensively validated.

The score JSON stores shared metrics under `results.metrics` and DER results under `results.collars`.
