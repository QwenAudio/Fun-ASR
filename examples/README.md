# Runnable examples

## Native Transformers (start here for the Hugging Face API)

Use the [native quickstart](transformers/) for the released `transformers==5.17.0`
package and the official `Fun-ASR-Nano-2512-hf` checkpoint. It includes a tested
CPU script, local audio, batching, keywords and an [upload notebook](colab/fun_asr_nano_transformers.ipynb).
This path does not require the FunASR toolkit or repository-local remote code.

## FunASR toolkit examples

The remaining scripts use the original toolkit checkpoint and a separate
environment. Install the base requirements for these scripts only:

```bash
pip install -r requirements.txt
```

Run commands from the repository root. Each script defaults to the bundled
`example/zh.mp3` audio inside the downloaded model and accepts a custom audio
path when you want to test your own file.

## Quickstart with `funasr.AutoModel`

```bash
python examples/quickstart.py --hub hf --device cuda:0
python examples/quickstart.py path/to/audio.wav --language 中文 --hotwords 开放时间
```

## Direct model inference

Use this when you want to call `model.py` directly without the high-level
`AutoModel` wrapper.

```bash
python examples/direct_inference.py --device cuda:0
python examples/direct_inference.py path/to/audio.wav
```

## Speaker diarization

This example enables VAD, CAM++ speaker labels, and punctuation restoration.

```bash
python examples/speaker_diarization.py --hub hf --device cuda:0
python examples/speaker_diarization.py path/to/meeting.wav
```

## vLLM offline batch inference

Install vLLM before running this example:

```bash
pip install "funasr>=1.3.26" "vllm>=0.12.0"
python examples/vllm_batch.py --tensor-parallel-size 1
python examples/vllm_batch.py audio1.wav audio2.wav --hotwords 张三 北京
```

`AutoModelVLLM` decodes each input in a single pass. For long meetings, segment
the audio first or use the `AutoModel(..., vad_model="fsmn-vad")` path.

## Streaming SDK

Install vLLM before running this example:

```bash
pip install "funasr>=1.3.26" "vllm>=0.12.0"
python examples/streaming_sdk.py --hub hf --chunk-ms 720
python examples/streaming_sdk.py path/to/audio.wav
```
