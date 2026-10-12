#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Requires Python 3.10+ and: pip install numpy scipy
# jsonls/example_in.jsonl contains annotations for two AISHELL-4 recordings.

# 1. Split into 4-minute chunks. Remove --skip-audio to slice real audio.
# Real slicing requires ffmpeg and the WAV files under audios/.
python3 split_chunks_by_dur_sdr.py \
  --input-jsonl jsonls/example_in.jsonl \
  --output-jsonl jsonls/example_in_chunked4min.jsonl \
  --cache-dir audio_chunks \
  --max-duration 240 \
  --skip-audio

# 2. Mock inference: copy reference text to demonstrate the result format.
# Replace this block with your model call for a real evaluation.
# Send only system/user messages to the model; assistant is the reference.
# MOCK_INFERENCE=0 bash demo.sh scores an existing model result file.
if [[ "${MOCK_INFERENCE:-1}" == "1" ]]; then
  python3 - <<'PY'
import json

with open("jsonls/example_in_chunked4min.jsonl", encoding="utf-8") as src, \
     open("jsonls/example_in_chunked4min_modelInferResult.jsonl", "w", encoding="utf-8") as dst:
    for line in src:
        sample = json.loads(line)
        sample["model_infer_result"] = "\n".join(
            part["text"]
            for message in sample["messages"] if message["role"] == "assistant"
            for part in message["content"] if part["type"] == "text"
        )
        dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
print("Mock predictions written (reference copies, not model outputs).")
PY
fi

# 3. Score predictions. Model timestamps must be relative to each chunk.
python3 score_sdr.py \
  --input jsonls/example_in_chunked4min_modelInferResult.jsonl \
  --collar 0.0 0.25 \
  --tcp-collar 5.0 \
  --tokenize mixed
