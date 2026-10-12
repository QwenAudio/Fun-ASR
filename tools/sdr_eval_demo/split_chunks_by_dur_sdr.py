#!/usr/bin/env python3
"""Split annotated audio into duration-based chunks and write messages JSONL."""


import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_oss_path(oss_path: str):
    assert oss_path.startswith("oss://"), f"Invalid OSS path: {oss_path}"
    path = oss_path[len("oss://"):]
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def generate_oss_signed_url(oss_path: str, ak: str, sk: str, endpoint: str, expires: int = 3600) -> str:
    import oss2
    bucket_name, object_key = parse_oss_path(oss_path)
    auth = oss2.Auth(ak, sk)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    return bucket.sign_url('GET', object_key, expires)


def slice_audio_ffmpeg(source: str, output_path: str,
                       start_time: float, end_time: float) -> bool:
    tmp_path = output_path + ".tmp"
    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-i", source,
        "-ar", "16000", "-ac", "1", "-f", "wav",
        tmp_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  [ERROR] ffmpeg failed: {result.stderr[:300]}", file=sys.stderr)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
        os.rename(tmp_path, output_path)
        return True
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] ffmpeg timed out", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
    except Exception as e:
        print(f"  [ERROR] ffmpeg error: {e}", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def resolve_audio_source(audio_path: str, oss_ak: str, oss_sk: str, oss_endpoint: str) -> str:
    if audio_path.startswith("oss://"):
        return generate_oss_signed_url(audio_path, oss_ak, oss_sk, oss_endpoint)
    return audio_path


def upload_to_oss(local_path: str, oss_dest: str, ak: str, sk: str, endpoint: str) -> bool:
    import oss2
    bucket_name, object_key = parse_oss_path(oss_dest)
    auth = oss2.Auth(ak, sk)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    try:
        bucket.put_object_from_file(object_key, local_path)
        return True
    except Exception as e:
        print(f"  [ERROR] OSS upload failed: {e}", file=sys.stderr)
        return False


def oss_object_exists(oss_path: str, ak: str, sk: str, endpoint: str) -> bool:
    import oss2
    bucket_name, object_key = parse_oss_path(oss_path)
    auth = oss2.Auth(ak, sk)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    return bucket.object_exists(object_key)


def group_segments_into_chunks(segments: list, max_duration: float) -> list[list]:
    # Preserve complete segments, including segments longer than the limit.
    sorted_segs = sorted(segments, key=lambda s: s.get("start_time", 0))
    chunks = []
    current_chunk = []
    chunk_start = None

    for seg in sorted_segs:
        seg_start = seg.get("start_time", 0)
        seg_end = seg.get("end_time", 0)

        if chunk_start is None:
            chunk_start = seg_start
            current_chunk.append(seg)
        elif (seg_end - chunk_start) <= max_duration:
            current_chunk.append(seg)
        else:
            chunks.append(current_chunk)
            current_chunk = [seg]
            chunk_start = seg_start

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_whole_audio(duration: float, max_duration: float) -> list[tuple[float, float]]:
    chunks = []
    start = 0.0
    while start < duration:
        end = min(start + max_duration, duration)
        chunks.append((start, end))
        start = end
    return chunks


def build_sd_ref_text(segments: list, chunk_start: float) -> str:
    lines = []
    for seg in segments:
        spk = seg.get("speaker_id", "SPK00")
        st = seg.get("start_time", 0) - chunk_start
        et = seg.get("end_time", 0) - chunk_start
        text = seg.get("raw_transcript", "")
        lines.append(f"[{spk}|{st:.2f}-{et:.2f}]{text}")
    return "\n".join(lines)


def build_output_entry(entry_id: str, chunk_audio_path: str,
                       chunk_start: float, chunk_end: float,
                       chunk_duration: float,
                       source_audio: str, source_audio_duration: float,
                       segments_ref: list,
                       system_prompt: str, user_prompt: str,
                       skip_think: bool, ref_format: str) -> dict:

    user_content = [
        {
            "type": "input_audio",
            "input_audio": {
                "data": chunk_audio_path,
                "format": "wav",
                "start_time": 0.0,
                "end_time": round(chunk_duration, 3),
                "duration": round(chunk_duration, 3),
                "raw_audio_duration": round(chunk_duration, 3),
            },
        },
    ]
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_content})

    if segments_ref:
        if ref_format == "sd":
            ref_text = build_sd_ref_text(segments_ref, chunk_start)
        else:
            ref_text = "\n".join(
                seg.get("raw_transcript", "") for seg in segments_ref
            )

        assistant_content = []
        if skip_think:
            assistant_content.append({
                "type": "thought", "loss": False,
                "thought": {"text": ""},
            })
        assistant_content.append({
            "type": "text", "loss": True,
            "text": ref_text,
        })
        messages.append({"role": "assistant", "content": assistant_content})

    return {
        "id": entry_id,
        "messages": messages,
        "source_audio": source_audio,
        "source_audio_duration": round(source_audio_duration, 3),
        "chunk_range": [round(chunk_start, 3), round(chunk_end, 3)],
        "segments_ref": segments_ref,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Split annotated audio by duration and write messages JSONL")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--skip-audio", action="store_true",
                        help="Write chunk metadata only; do not read, slice, or upload audio")
    parser.add_argument("--max-duration", type=float, default=240.0)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--user-prompt", default="")
    parser.add_argument("--skip-think", action="store_true", default=True)
    parser.add_argument("--no-skip-think", dest="skip_think", action="store_false")
    parser.add_argument("--ref-format", choices=["sd", "plain"], default="sd")
    parser.add_argument("--oss-upload-prefix", default=None)
    parser.add_argument("--oss-ak", default=None)
    parser.add_argument("--oss-sk", default=None)
    parser.add_argument("--oss-endpoint", default=None)
    args = parser.parse_args()

    if args.skip_audio and args.oss_upload_prefix:
        parser.error("--skip-audio cannot be combined with --oss-upload-prefix")

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    cache_dir = Path(args.cache_dir)

    oss_ak = args.oss_ak or os.environ.get("OSS_ACCESS_KEY_ID", "")
    oss_sk = args.oss_sk or os.environ.get("OSS_ACCESS_KEY_SECRET", "")
    oss_endpoint = args.oss_endpoint or os.environ.get("OSS_ENDPOINT", "")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print("=" * 60)
    print("SDR audio chunking")
    print(f"  Input: {input_path}  ({len(entries)} records)")
    print(f"  Output: {output_path}")
    print(f"  Cache: {cache_dir}")
    print(f"  Maximum duration: {args.max_duration}s")
    print(f"  skip_think: {args.skip_think}")
    print(f"  ref_format: {args.ref_format}")
    if args.system_prompt:
        print(f"  system_prompt: {args.system_prompt[:80]}...")
    if args.oss_upload_prefix:
        print(f"  OSS upload prefix: {args.oss_upload_prefix}")
    print("=" * 60)

    if args.skip_audio:
        print("[INFO] Metadata-only mode: audio paths are placeholders.")

    total_chunks = 0
    skipped_slices = 0
    new_slices = 0
    uploaded = 0
    upload_skipped = 0

    oss_prefix = args.oss_upload_prefix
    if oss_prefix and not oss_prefix.endswith("/"):
        oss_prefix += "/"

    with open(output_path, "w", encoding="utf-8") as f_out:
        for ei, entry in enumerate(entries, 1):
            audio_path = entry.get("audio_path", "")
            duration = entry.get("duration", 0)
            segments = entry.get("segments", [])

            if audio_path.startswith("oss://"):
                _, obj_key = parse_oss_path(audio_path)
                audio_stem = Path(obj_key).stem
            else:
                audio_stem = Path(audio_path).stem

            print(f"\n[{ei}/{len(entries)}] {audio_stem}  "
                  f"(dur={duration:.1f}s, segs={len(segments)})")

            is_oss = audio_path.startswith("oss://")
            if is_oss and not args.skip_audio:
                if not oss_ak or not oss_sk or not oss_endpoint:
                    print(f"  [ERROR] OSS paths require AK/SK/Endpoint", file=sys.stderr)
                    continue
                audio_source = resolve_audio_source(audio_path, oss_ak, oss_sk, oss_endpoint)
            else:
                audio_source = audio_path

            if segments:
                seg_chunks = group_segments_into_chunks(segments, args.max_duration)
                chunk_ranges = []
                for chunk_segs in seg_chunks:
                    cs = min(s["start_time"] for s in chunk_segs)
                    ce = max(s["end_time"] for s in chunk_segs)
                    chunk_ranges.append((cs, ce, chunk_segs))
            else:
                time_chunks = chunk_whole_audio(duration, args.max_duration)
                chunk_ranges = [(s, e, []) for s, e in time_chunks]

            print(f"  Chunks: {len(chunk_ranges)}")

            for ci, (chunk_start, chunk_end, chunk_segs) in enumerate(chunk_ranges, 1):
                chunk_duration = chunk_end - chunk_start

                cache_name = f"{audio_stem}_{chunk_start:.3f}_{chunk_end:.3f}.wav"
                cache_path = str(cache_dir / cache_name)

                if args.skip_audio:
                    pass
                elif os.path.exists(cache_path):
                    skipped_slices += 1
                else:
                    ok = slice_audio_ffmpeg(audio_source, cache_path,
                                           chunk_start, chunk_end)
                    if not ok:
                        print(f"  [ERROR] chunk {ci} slicing failed; skipping", file=sys.stderr)
                        continue
                    new_slices += 1

                audio_path_for_jsonl = cache_path
                if oss_prefix:
                    oss_dest = f"{oss_prefix}{cache_name}"
                    if oss_object_exists(oss_dest, oss_ak, oss_sk, oss_endpoint):
                        upload_skipped += 1
                    else:
                        ok = upload_to_oss(cache_path, oss_dest, oss_ak, oss_sk, oss_endpoint)
                        if ok:
                            uploaded += 1
                        else:
                            print(f"  [WARN] Upload failed; using the local path in JSONL", file=sys.stderr)
                            oss_dest = None
                    if oss_dest:
                        audio_path_for_jsonl = oss_dest

                entry_id = f"{audio_stem}_chunk{ci:03d}"
                out_entry = build_output_entry(
                    entry_id=entry_id,
                    chunk_audio_path=audio_path_for_jsonl,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    chunk_duration=chunk_duration,
                    source_audio=audio_path,
                    source_audio_duration=duration,
                    segments_ref=chunk_segs,
                    system_prompt=args.system_prompt,
                    user_prompt=args.user_prompt,
                    skip_think=args.skip_think,
                    ref_format=args.ref_format,
                )

                f_out.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                total_chunks += 1

            print(f"  Finished {len(chunk_ranges)} chunks")

    print(f"\n{'=' * 60}")
    print(f"Chunking complete")
    print(f"  Total chunks written: {total_chunks}")
    print(f"  New slices: {new_slices}, Cached slices: {skipped_slices}")
    if oss_prefix:
        print(f"  OSS uploads: {uploaded}, Existing OSS objects skipped: {upload_skipped}")
    print(f"  Output file: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
