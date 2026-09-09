# Fun-ASR-Nano with Transformers

Transcribe Chinese, English or Japanese with the native Hugging Face processor
and generation API. No FunASR toolkit, repository-local `model.py`, or remote
Python code is needed. The first model load downloads about 1.66 GB of weights.

[Online demo](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano) ·
[Notebook](../colab/fun_asr_nano_transformers.ipynb) ·
[中文指南](https://www.funasr.com/docs/native-transformers.html)

## First transcript

Clone the **QwenAudio/Fun-ASR code repository**, not a Hugging Face weight
repository. Run all commands below from this repository's root:

```bash
git clone https://github.com/QwenAudio/Fun-ASR.git
cd Fun-ASR
```

Use an isolated environment. This reproducible recipe was exercised on Linux
x86-64, Python 3.12 and CPU; it does not upgrade your existing serving environment.

```bash
python3.12 -m venv .venv-native
. .venv-native/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu' 'torchaudio==2.10.0+cpu'
python -m pip install -r examples/transformers/requirements.txt
python -m pip check
python examples/transformers/transcribe.py
```

The default input is the pinned official English sample. A successful result
contains `text` and `reached_eos: true`. No access token is required for this
public checkpoint. The model itself also runs through the short, no-clone
[Python recipe](https://www.funasr.com/en/docs/native-transformers.html).

## Your recordings and batches

```bash
python examples/transformers/transcribe.py recording.wav --language zh --keywords 开放时间
python examples/transformers/transcribe.py chinese.wav english.wav --language zh en
python examples/transformers/transcribe.py english.wav --language en --prompt 'A conversation about opening hours.'
```

Files are read as float32, stereo/multichannel audio is averaged to mono, and
non-16 kHz audio is explicitly resampled with `soxr_hq`. Original files are not
modified. This example accepts 1-4 files and at most 60 seconds total. Longer
recordings are rejected, not silently trimmed. Segment long audio deliberately
or choose a [serving recipe](https://www.funasr.com/en/deploy/).

Output order matches input order. One language applies to all files, or give one
per file. `keywords` and `prompt` are hints, not enforced vocabulary. Missing EOS
or empty text produces a nonzero exit after printing the diagnostic result.
EOS is a generation boundary, not proof that every word was recognized.

## Select a backend, not just a suffix

| Need | Artifact and entry point |
| --- | --- |
| Native Transformers | `FunAudioLLM/Fun-ASR-Nano-2512-hf`, `AutoProcessor` + `AutoModelForSpeechSeq2Seq` |
| FunASR pipelines and existing services | `FunAudioLLM/Fun-ASR-Nano-2512`, `funasr.AutoModel` |
| Native vLLM serving | `FunAudioLLM/Fun-ASR-Nano-2512-vllm`, [native vLLM guide](https://www.funasr.com/en/docs/official-native-vllm.html) |
| C++ / edge | Converted GGUF, [llama.cpp guide](https://www.funasr.com/en/llama-cpp.html) |

The native `-hf` export produces transcription text. It does not add word
timestamps, speaker identities, a streaming protocol or an HTTP server. The
31-language MLT checkpoint is a separate model, not an alternate name for this
zh/en/ja export. GPU dtype, attention kernels and throughput require separate
hardware validation; the CPU sample is not a GPU benchmark.

## Verification

Transformers **5.17.0** is a released package containing `fun_asr_nano`.
The examples pin native checkpoint revision
`d93b302ee7fd505e1b3576120fc142fc6f7820e1` and set
`trust_remote_code=False`. Chinese/English public-sample inference, keywords
and padded batching were functionally exercised on CPU on 2026-09-09; these
checks are not CER/WER or capacity evaluation.

[Upstream model documentation](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/fun_asr_nano)
and [model card](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf)
describe the native interface. Keep the runtime, model revision and raw output
with your own evaluation. Never publish private recordings in a bug report.
