#!/usr/bin/env python3
"""align_streaming_vad.py — cross-check the C++ streaming VAD against the Python one.

Feeds a wav through FunASR's DynamicStreamingVAD (the same wrapper used by
serve_realtime_ws.py) in 60ms chunks and compares the confirmed segment
boundaries against the LOCKED timestamps that `llama-funasr-cli --stream`
logs on stderr ("[stream] locked [START,ENDms] ...").

  align_streaming_vad.py sample.wav                     # just print python segments
  align_streaming_vad.py sample.wav --cpp locked.txt    # pass/fail vs C++ (tolerance 100ms)

Requires the funasr python package + torch (CPU is fine); fsmn-vad is tiny.
"""
import argparse
import sys
import wave

TOLERANCE_MS = 100


def python_segments(wav_path):
    import numpy as np
    import torch
    from funasr import AutoModel
    from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

    with wave.open(wav_path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            sys.exit("expected 16 kHz mono s16 wav")
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    vad = DynamicStreamingVAD(AutoModel(model="fsmn-vad", device="cpu", disable_update=True))
    segs = []
    chunk = 960  # 60ms
    for off in range(0, len(audio), chunk):
        segs.extend(vad.feed(torch.from_numpy(audio[off:off + chunk]).float(), is_final=False))
    segs.extend(vad.finalize())
    return segs


def cpp_segments(path):
    segs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("[stream] locked ["):
            span = line.split("[", 2)[2].split("]")[0].replace("ms", "")
            start, end = span.split(",")
            segs.append([int(start), int(end)])
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--cpp", help="llama-funasr-cli --stream stderr log with locked lines")
    args = ap.parse_args()

    py_segs = python_segments(args.wav)
    for s, e in py_segs:
        print(f"python [{s},{e}ms]")

    if not args.cpp:
        return
    cpp = cpp_segments(args.cpp)
    if len(py_segs) != len(cpp):
        sys.exit(f"FAIL: segment count differs: python={len(py_segs)} cpp={len(cpp)}")
    for (ps, pe), (cs, ce) in zip(py_segs, cpp):
        if abs(ps - cs) > TOLERANCE_MS or abs(pe - ce) > TOLERANCE_MS:
            sys.exit(f"FAIL: python [{ps},{pe}] vs cpp [{cs},{ce}] (tolerance {TOLERANCE_MS}ms)")
    print(f"PASS: {len(cpp)} segment(s) within {TOLERANCE_MS}ms of python DynamicStreamingVAD")


if __name__ == "__main__":
    main()
