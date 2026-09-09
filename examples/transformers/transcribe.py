"""Native Transformers CPU example for short, authorized recordings."""
import argparse
import json
from pathlib import Path

import numpy as np

MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
REVISION = "d93b302ee7fd505e1b3576120fc142fc6f7820e1"
SAMPLE_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
SAMPLE_REVISION = "272c57b82523ada6fd87095e955f8e29100979ab"
SAMPLE_RATE = 16000
MAX_SECONDS = 60


def prepare_audio(audio, sample_rate):
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate <= 0 or audio.ndim not in (1, 2) or not audio.size:
        raise ValueError("Audio must be non-empty mono or frames-by-channels audio")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    if len(audio) / sample_rate > MAX_SECONDS:
        raise ValueError("This short-file example accepts at most 60 seconds; no silent trimming")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    return np.ascontiguousarray(audio, dtype=np.float32)


def languages_for_batch(languages, count):
    if count < 1 or count > 4 or len(languages) not in (1, count):
        raise ValueError("Use 1-4 files and one language or one language per file")
    if any(language not in ("zh", "en", "ja") for language in languages):
        raise ValueError("Base Nano supports zh, en and ja; MLT is a separate checkpoint")
    return languages * count if len(languages) == 1 else languages


def load_audio(path):
    import soundfile as sf

    info = sf.info(path)
    if not info.frames or not 0 < info.duration <= MAX_SECONDS:
        raise ValueError("Use a non-empty recording of at most 60 seconds")
    audio, sample_rate = sf.read(path, dtype="float32")
    return prepare_audio(audio, sample_rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="*", type=Path, help="Local audio files; default: pinned official English sample")
    parser.add_argument("--language", nargs="+", default=["en"], choices=["zh", "en", "ja"])
    parser.add_argument("--keywords", nargs="*", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 1024:
        parser.error("--max-new-tokens must be between 1 and 1024")
    languages = languages_for_batch(args.language, len(args.audio) or 1)
    if not args.audio:
        from huggingface_hub import hf_hub_download

        args.audio = [Path(hf_hub_download(SAMPLE_MODEL, "example/en.mp3", revision=SAMPLE_REVISION, token=False))]
    audio = [load_audio(path) for path in args.audio]
    if sum(len(item) for item in audio) > MAX_SECONDS * SAMPLE_RATE:
        raise ValueError("Keep total batch audio within 60 seconds for this example")

    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False, token=False)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=False, token=False,
        dtype=torch.float32,
    ).to("cpu").eval()
    inputs = processor.apply_transcription_request(
        audio=audio, language=languages, keywords=args.keywords, prompt=args.prompt,
        processor_kwargs={"return_tensors": "pt", "audio_kwargs": {"sampling_rate": SAMPLE_RATE},
                          "text_kwargs": {"padding": True}},
    )
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    new_tokens = generated[:, inputs.input_ids.shape[1]:]
    texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    results = []
    for path, language, text, tokens in zip(args.audio, languages, texts, new_tokens.tolist()):
        results.append({"file": str(path), "language": language, "text": text,
                        "reached_eos": any(token in eos_ids for token in tokens)})
    print(json.dumps({"model": MODEL_ID, "revision": REVISION, "transformers": transformers.__version__,
                      "device": "cpu", "results": results}, ensure_ascii=False, indent=2))
    if not all(row["reached_eos"] and row["text"].strip() for row in results):
        raise SystemExit("Incomplete generation: inspect empty text or missing EOS; do not treat this as a complete transcript")


if __name__ == "__main__":
    main()
