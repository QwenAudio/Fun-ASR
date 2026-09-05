#!/usr/bin/env python3
"""wav2pcm.py — dump a 16 kHz mono s16 WAV as raw little-endian PCM on stdout.

Used by the llama-funasr-cli --stream regression test to simulate a PCM input
stream. Stdlib only.

  wav2pcm.py sample.wav                 # whole file at once
  wav2pcm.py sample.wav --chunk-ms 60   # 60ms chunks
  wav2pcm.py sample.wav --chunk-ms 960 --realtime  # also sleep between chunks
"""
import argparse
import sys
import time
import wave


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--chunk-ms", type=float, default=0,
                    help="emit in N-ms chunks (default: whole file)")
    ap.add_argument("--realtime", action="store_true",
                    help="sleep one chunk between writes (realtime simulation)")
    args = ap.parse_args()

    with wave.open(args.wav, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            sys.exit("wav2pcm: expected 16 kHz mono s16 wav, got "
                     f"{w.getframerate()} Hz x {w.getnchannels()} ch x {w.getsampwidth()} B")
        pcm = w.readframes(w.getnframes())

    out = sys.stdout.buffer
    chunk = int(args.chunk_ms * 16) * 2 if args.chunk_ms > 0 else len(pcm)
    if chunk <= 0:
        chunk = len(pcm)
    for off in range(0, len(pcm), chunk):
        out.write(pcm[off:off + chunk])
        out.flush()
        if args.realtime:
            time.sleep(chunk / 2 / 16000.0)


if __name__ == "__main__":
    main()
