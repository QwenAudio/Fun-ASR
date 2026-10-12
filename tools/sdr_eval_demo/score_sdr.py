#!/usr/bin/env python3
"""Score speaker-attributed ASR: DER, JER, WER, cpWER, and tcpWER."""


import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_FRAME_RATE = 100
_INF = 1 << 29


def parse_sdr_output(text: str, time_offset: float = 0.0) -> list[dict]:


    # Convert chunk-relative timestamps to source-audio coordinates.
    segments = []
    pattern = re.compile(r"\[([^\]\|]+)\|(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\](.*)")

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            continue
        speaker_id = m.group(1)
        start_time = round(float(m.group(2)) + time_offset, 4)
        end_time = round(float(m.group(3)) + time_offset, 4)
        transcript = m.group(4).strip()
        if start_time >= end_time:
            continue
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "speaker_id": speaker_id,
            "text": transcript,
        })

    return segments


def _build_no_score_zones(ref_segs: list[dict], collar: float) -> list[tuple]:
    if collar <= 0 or not ref_segs:
        return []
    zones = []
    for seg in ref_segs:
        zones.append((seg["start_time"] - collar, seg["start_time"] + collar))
        zones.append((seg["end_time"] - collar, seg["end_time"] + collar))
    zones.sort()
    merged = [zones[0]]
    for zs, ze in zones[1:]:
        if zs <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], ze))
        else:
            merged.append((zs, ze))
    return merged


def _scoreable_duration(seg_start: float, seg_end: float,
                        no_score: list[tuple]) -> float:
    if not no_score:
        return seg_end - seg_start
    scoreable = 0.0
    t = seg_start
    for zs, ze in no_score:
        if ze <= t:
            continue
        if zs >= seg_end:
            break
        if zs > t:
            scoreable += zs - t
        t = max(t, ze)
    if t < seg_end:
        scoreable += seg_end - t
    return scoreable


def compute_der_microseg(ref_segs: list[dict], sys_segs: list[dict],
                         duration: float, collar: float = 0.0) -> dict:
    events = []
    for seg in ref_segs:
        events.append((seg["start_time"], "REF_BEG", seg["speaker_id"]))
        events.append((seg["end_time"], "REF_END", seg["speaker_id"]))
    for seg in sys_segs:
        events.append((seg["start_time"], "SYS_BEG", seg["speaker_id"]))
        events.append((seg["end_time"], "SYS_END", seg["speaker_id"]))

    events.sort(key=lambda e: (e[0], 0 if "END" in e[1] else 1))

    ref_active = {}
    sys_active = {}
    spkr_overlap = {}
    prev_time = 0.0

    def _active_spks(counter):
        return {k for k, v in counter.items() if v > 0}

    for time, etype, spk in events:
        if time > prev_time:
            seg_dur = time - prev_time
            ref_set = _active_spks(ref_active)
            sys_set = _active_spks(sys_active)
            if ref_set and seg_dur > 0:
                for r in ref_set:
                    for s in sys_set:
                        spkr_overlap.setdefault(r, {})
                        spkr_overlap[r][s] = spkr_overlap[r].get(s, 0.0) + seg_dur
            prev_time = time
        if etype == "REF_BEG":
            ref_active[spk] = ref_active.get(spk, 0) + 1
        elif etype == "REF_END":
            ref_active[spk] = ref_active.get(spk, 0) - 1
        elif etype == "SYS_BEG":
            sys_active[spk] = sys_active.get(spk, 0) + 1
        elif etype == "SYS_END":
            sys_active[spk] = sys_active.get(spk, 0) - 1

    ref_spks = sorted(spkr_overlap.keys())
    sys_spks_set = set()
    for r in spkr_overlap:
        sys_spks_set.update(spkr_overlap[r].keys())
    sys_spks = sorted(sys_spks_set)

    spk_map = {}
    if ref_spks and sys_spks:
        overlap_matrix = np.zeros((len(ref_spks), len(sys_spks)))
        for i, r in enumerate(ref_spks):
            for j, s in enumerate(sys_spks):
                overlap_matrix[i, j] = spkr_overlap.get(r, {}).get(s, 0.0)
        ri, si = linear_sum_assignment(-overlap_matrix)
        for r_idx, s_idx in zip(ri, si):
            if overlap_matrix[r_idx, s_idx] > 0:
                spk_map[ref_spks[r_idx]] = sys_spks[s_idx]

    no_score = _build_no_score_zones(ref_segs, collar)

    ref_active = {}
    sys_active = {}
    prev_time = 0.0
    scored = miss = fa = spk_err = 0.0

    for time, etype, spk in events:
        if time > prev_time:
            seg_dur = (_scoreable_duration(prev_time, time, no_score)
                       if no_score else time - prev_time)
            if seg_dur > 0:
                ref_set = _active_spks(ref_active)
                sys_set = _active_spks(sys_active)
                nref = len(ref_set)
                nsys = len(sys_set)
                scored += seg_dur * nref
                miss += seg_dur * max(nref - nsys, 0)
                fa += seg_dur * max(nsys - nref, 0)
                nmap = sum(1 for r in ref_set
                           if r in spk_map and spk_map[r] in sys_set)
                spk_err += seg_dur * (min(nref, nsys) - nmap)
            prev_time = time
        if etype == "REF_BEG":
            ref_active[spk] = ref_active.get(spk, 0) + 1
        elif etype == "REF_END":
            ref_active[spk] = ref_active.get(spk, 0) - 1
        elif etype == "SYS_BEG":
            sys_active[spk] = sys_active.get(spk, 0) + 1
        elif etype == "SYS_END":
            sys_active[spk] = sys_active.get(spk, 0) - 1

    if scored == 0:
        return {"der": 0.0, "miss": 0.0, "fa": 0.0, "spk_err": 0.0,
                "scored_s": 0.0, "miss_s": 0.0, "fa_s": 0.0, "spk_err_s": 0.0}
    return {
        "der": round((miss + fa + spk_err) / scored * 100, 2),
        "miss": round(miss / scored * 100, 2),
        "fa": round(fa / scored * 100, 2),
        "spk_err": round(spk_err / scored * 100, 2),
        "scored_s": scored, "miss_s": miss, "fa_s": fa, "spk_err_s": spk_err,
    }


def segments_to_frames(segments: list[dict], duration: float,
                       frame_rate: int = DEFAULT_FRAME_RATE) -> tuple[np.ndarray, list[str]]:
    step = 1.0 / frame_rate
    n_frames = int(duration / step)
    if not segments:
        return np.zeros((n_frames, 0), dtype=np.int8), []
    speaker_ids = sorted(set(seg["speaker_id"] for seg in segments))
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_ids)}
    matrix = np.zeros((n_frames, len(speaker_ids)), dtype=np.int8)
    times = step * np.arange(n_frames)
    for seg in segments:
        bi = int(np.searchsorted(times, seg["start_time"]))
        ei = int(np.searchsorted(times, seg["end_time"]))
        if bi < ei:
            matrix[bi:ei, spk_to_idx[seg["speaker_id"]]] = 1
    return matrix, speaker_ids


def compute_jer(ref_matrix: np.ndarray, sys_matrix: np.ndarray) -> dict:
    n_ref = ref_matrix.shape[1] if ref_matrix.ndim == 2 else 0
    n_sys = sys_matrix.shape[1] if sys_matrix.ndim == 2 else 0
    if n_ref == 0 and n_sys == 0:
        return {"jer": 0.0}
    if n_ref == 0:
        return {"jer": 100.0}
    if n_sys == 0:
        return {"jer": 100.0}
    ref_durs = ref_matrix.sum(axis=0).astype(np.float64)
    sys_durs = sys_matrix.sum(axis=0).astype(np.float64)
    intersect = ref_matrix.T.astype(np.float64) @ sys_matrix.astype(np.float64)
    union = ref_durs[:, None] + sys_durs[None, :] - intersect
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard_dist = 1.0 - intersect / union
    jaccard_dist = np.nan_to_num(jaccard_dist, nan=1.0)
    ri, si = linear_sum_assignment(jaccard_dist)
    per_spk = np.ones(n_ref) * 100.0
    for r, s in zip(ri, si):
        per_spk[r] = jaccard_dist[r, s] * 100.0
    return {"jer": round(float(per_spk.mean()), 2),
            "per_spk_jer": per_spk.tolist()}


_CJK_PUNCT = "，。！？、；：""''（）《》【】…—·「」『』〈〉“”‘’"


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or
            0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2A6DF or
            0x3040 <= o <= 0x30FF)


def normalize_text(text: str) -> str:
    return text.lower().strip()


def tokenize(text: str, mode: str = "mixed") -> list[str]:


    text = normalize_text(text)
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isalnum() and not _is_cjk(ch):
            if mode == "char":
                tokens.append(ch)
            else:
                buf.append(ch)
        elif _is_cjk(ch):
            if buf:
                tokens.append("".join(buf)); buf = []
            tokens.append(ch)
        else:
            if buf:
                tokens.append("".join(buf)); buf = []
    if buf:
        tokens.append("".join(buf))
    return tokens


def _char_based_intervals(interval: tuple, tokens: list[str]) -> list[tuple]:

    n = len(tokens)
    if n == 0:
        return []
    if n == 1:
        return [interval]
    lens = np.array([max(len(t), 1) for t in tokens], dtype=np.float64)
    ends = np.cumsum(lens)
    total = ends[-1]
    span = (interval[1] - interval[0]) / total
    starts = np.concatenate([[0.0], ends[:-1]])
    return [(interval[0] + span * s, interval[0] + span * e)
            for s, e in zip(starts, ends)]


def extract_words(segments: list[dict], text_key: str, mode: str,
                  as_points: bool = False, collar: float = 0.0) -> list[dict]:


    words = []
    for seg in sorted(segments, key=lambda s: (s["start_time"], s["end_time"])):
        toks = tokenize(seg.get(text_key, ""), mode)
        if not toks:
            continue
        intervals = _char_based_intervals(
            (seg["start_time"], seg["end_time"]), toks)
        for tok, (a, b) in zip(toks, intervals):
            if as_points:
                p = (a + b) / 2.0
                a, b = p - collar, p + collar
            words.append({"spk": seg.get("speaker_id", ""),
                          "token": tok, "start": a, "end": b})
    return words


def _levenshtein_matrix(ref: list, hyp: list,
                        overlap_mask: 'np.ndarray | None' = None) -> np.ndarray:


    n, m = len(ref), len(hyp)
    D = np.empty((n + 1, m + 1), dtype=np.int32)
    D[0] = np.arange(m + 1, dtype=np.int32)
    if m == 0:
        D[:, 0] = np.arange(n + 1, dtype=np.int32)
        return D
    hyp_arr = np.array(hyp, dtype=object)
    j_idx = np.arange(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        prev = D[i - 1].astype(np.int64)
        sub_cost = np.where(hyp_arr == ref[i - 1], 0, 1).astype(np.int64)
        if overlap_mask is not None:
            sub_cost = np.where(overlap_mask[i - 1], sub_cost, _INF)
        diag = prev[:-1] + sub_cost
        up = prev[1:] + 1
        tentative = np.empty(m + 1, dtype=np.int64)
        tentative[0] = i
        tentative[1:] = np.minimum(diag, up)
        a = tentative - j_idx
        D[i] = (np.minimum.accumulate(a) + j_idx).astype(np.int32)
    return D


def _backtrack(D: np.ndarray, ref: list, hyp: list,
               overlap_mask: 'np.ndarray | None' = None) -> dict:

    i, j = len(ref), len(hyp)
    ins = dele = sub = cor = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            allow = True if overlap_mask is None else bool(overlap_mask[i - 1, j - 1])
            if allow:
                c = 0 if hyp[j - 1] == ref[i - 1] else 1
                if int(D[i, j]) == int(D[i - 1, j - 1]) + c:
                    if c == 0:
                        cor += 1
                    else:
                        sub += 1
                    i -= 1; j -= 1
                    continue
        if i > 0 and int(D[i, j]) == int(D[i - 1, j]) + 1:
            dele += 1; i -= 1
        elif j > 0 and int(D[i, j]) == int(D[i, j - 1]) + 1:
            ins += 1; j -= 1
        elif i > 0:
            dele += 1; i -= 1
        else:
            ins += 1; j -= 1
    return {"ins": ins, "del": dele, "sub": sub, "cor": cor}


def levenshtein(ref: list, hyp: list) -> dict:
    D = _levenshtein_matrix(ref, hyp)
    bt = _backtrack(D, ref, hyp)
    return {"errors": int(D[len(ref), len(hyp)]), "length": len(ref), **bt}


def tc_levenshtein(ref: list, hyp: list,
                   ref_int: list, hyp_int: list) -> dict:

    n, m = len(ref), len(hyp)
    if n and m:
        rs = np.array([x[0] for x in ref_int])[:, None]
        re_ = np.array([x[1] for x in ref_int])[:, None]
        hs = np.array([x[0] for x in hyp_int])[None, :]
        he = np.array([x[1] for x in hyp_int])[None, :]
        overlap = (rs < he) & (re_ > hs)
    else:
        overlap = np.zeros((n, m), dtype=bool)
    D = _levenshtein_matrix(ref, hyp, overlap_mask=overlap)
    bt = _backtrack(D, ref, hyp, overlap_mask=overlap)
    return {"errors": int(D[n, m]), "length": n, **bt}


def _min_permutation(ref_by_spk: dict, hyp_by_spk: dict, dist_fn) -> dict:


    r_keys = list(ref_by_spk.keys())
    h_keys = list(hyp_by_spk.keys())
    empty = ([], [])
    size = max(len(r_keys), len(h_keys))
    total = {"errors": 0, "length": 0, "ins": 0, "del": 0, "sub": 0,
             "missed_spk": max(0, len(r_keys) - len(h_keys)),
             "falarm_spk": max(0, len(h_keys) - len(r_keys)),
             "scored_spk": len(r_keys)}
    if size == 0:
        return total
    R = [ref_by_spk[k] for k in r_keys] + [empty] * (size - len(r_keys))
    H = [hyp_by_spk[k] for k in h_keys] + [empty] * (size - len(h_keys))
    dcache = {}
    cost = np.zeros((size, size), dtype=np.int64)
    for a in range(size):
        for b in range(size):
            d = dist_fn(R[a], H[b])
            dcache[(a, b)] = d
            cost[a, b] = d["errors"]
    ri, ci = linear_sum_assignment(cost)
    for a, b in zip(ri, ci):
        d = dcache[(a, b)]
        for k in ("errors", "length", "ins", "del", "sub"):
            total[k] += d[k]
    return total


def score_sample_wer(sample: dict, mode: str, tcp_collar: float) -> dict:

    ref_segs = sample.get("segments_ref", [])
    hyp_segs = sample.get("segments_hyp", [])


    # Interpolate token times; expand hypothesis midpoints for tcpWER.
    ref_words = extract_words(ref_segs, "raw_transcript", mode, as_points=False)
    hyp_words = extract_words(hyp_segs, "text", mode, as_points=False)
    hyp_words_tc = extract_words(hyp_segs, "text", mode,
                                 as_points=True, collar=tcp_collar)


    ref_stream = [w["token"] for w in ref_words]
    hyp_stream = [w["token"] for w in hyp_words]
    wer = levenshtein(ref_stream, hyp_stream)


    def group(words, with_time=False):
        g: dict = {}
        for w in words:
            g.setdefault(w["spk"], ([], []))
            g[w["spk"]][0].append(w["token"])
            if with_time:
                g[w["spk"]][1].append((w["start"], w["end"]))
        return g

    cp = _min_permutation(group(ref_words), group(hyp_words),
                          lambda a, b: levenshtein(a[0], b[0]))


    tcp = _min_permutation(
        group(ref_words, with_time=True), group(hyp_words_tc, with_time=True),
        lambda a, b: tc_levenshtein(a[0], b[0], a[1], b[1]))

    return {"wer": wer, "cpwer": cp, "tcpwer": tcp,
            "ref_words": len(ref_stream), "hyp_words": len(hyp_stream)}


def _rate(errors, length):
    return round(errors / length * 100, 2) if length > 0 else 0.0


def classify_sample(ref_len: int, hyp_len: int, hallu_ratio: float = 2.0,
                    trunc_ratio: float = 0.5) -> str:


    if ref_len <= 0:
        return "no_ref"
    r = hyp_len / ref_len
    if r > hallu_ratio:
        return "hallu"
    if r < trunc_ratio:
        return "trunc"
    return "clean"


def score_wer_all(samples: list[dict], mode: str, tcp_collar: float,
                  hallu_ratio: float = 2.0, trunc_ratio: float = 0.5) -> dict:


    blank = lambda: {k: {"errors": 0, "length": 0, "ins": 0, "del": 0, "sub": 0}
                     for k in ("wer", "cpwer", "tcpwer")}
    # Keep both length-filtered and all-reference corpus scores.
    agg_clean = blank()
    agg_all = blank()
    details = {}
    cnt = {"hallu": 0, "trunc": 0, "clean": 0, "no_ref": 0}

    for sample in samples:
        m = score_sample_wer(sample, mode, tcp_collar)
        status = classify_sample(m["ref_words"], m["hyp_words"],
                                 hallu_ratio, trunc_ratio)
        cnt[status] += 1
        if status != "no_ref":
            for k in ("wer", "cpwer", "tcpwer"):
                for f in ("errors", "length", "ins", "del", "sub"):
                    agg_all[k][f] += m[k][f]
                    if status == "clean":
                        agg_clean[k][f] += m[k][f]
        details[sample.get("id", "")] = {
            "WER": _rate(m["wer"]["errors"], m["wer"]["length"]),
            "cpWER": _rate(m["cpwer"]["errors"], m["cpwer"]["length"]),
            "tcpWER": _rate(m["tcpwer"]["errors"], m["tcpwer"]["length"]),
            "ref_words": m["ref_words"],
            "hyp_words": m["hyp_words"],
            "len_ratio": round(m["hyp_words"] / m["ref_words"], 2) if m["ref_words"] else None,
            "status": status,
        }

    scored = cnt["hallu"] + cnt["trunc"] + cnt["clean"]
    metrics = {"wer_scored_count": scored, "wer_noref_count": cnt["no_ref"]}
    if scored == 0:

        return {"metrics": metrics, "details": details}

    metrics.update({
        "hallu_rate": round(cnt["hallu"] / scored, 4),
        "trunc_rate": round(cnt["trunc"] / scored, 4),
        "hallu_count": cnt["hallu"],
        "trunc_count": cnt["trunc"],
        "clean_count": cnt["clean"],
    })

    for key, prefix in (("wer", "WER"), ("cpwer", "cpWER"), ("tcpwer", "tcpWER")):
        a = agg_clean[key]
        metrics[prefix] = _rate(a["errors"], a["length"])
        metrics[f"{prefix}_ins"] = _rate(a["ins"], a["length"])
        metrics[f"{prefix}_del"] = _rate(a["del"], a["length"])
        metrics[f"{prefix}_sub"] = _rate(a["sub"], a["length"])

    for key, prefix in (("wer", "WER"), ("cpwer", "cpWER"), ("tcpwer", "tcpWER")):
        a = agg_all[key]
        metrics[f"{prefix}_all"] = _rate(a["errors"], a["length"])
    return {"metrics": metrics, "details": details}


def offset_segments(segments: list[dict], offset: float) -> list[dict]:
    result = []
    for seg in segments:
        onset = round(seg["start_time"] - offset, 4)
        dur = round(seg["end_time"] - seg["start_time"], 4)
        result.append({**seg, "start_time": onset, "end_time": onset + dur})
    return result


def score_sample(sample: dict, frame_rate: int = DEFAULT_FRAME_RATE,
                 collar: float = 0.0) -> dict:
    chunk_range = sample.get("chunk_range")
    if chunk_range:
        time_offset = chunk_range[0]
        chunk_dur = chunk_range[1] - chunk_range[0]
    else:
        time_offset = 0.0
        chunk_dur = 0.0

    ref_segs = offset_segments(sample.get("segments_ref", []), time_offset)
    hyp_segs = offset_segments(sample.get("segments_hyp", []), time_offset)

    if chunk_dur > 0:
        clipped = []
        for seg in hyp_segs:
            s = max(seg["start_time"], 0.0)
            e = min(seg["end_time"], chunk_dur)
            if s < e:
                clipped.append({**seg, "start_time": s, "end_time": e})
        hyp_segs = clipped

    all_segs = ref_segs + hyp_segs
    max_end = max((s["end_time"] for s in all_segs), default=0.0)
    duration = max(chunk_dur, max_end)

    ref_matrix, ref_spks = segments_to_frames(ref_segs, duration, frame_rate)
    sys_matrix, sys_spks = segments_to_frames(hyp_segs, duration, frame_rate)

    return {
        "der": compute_der_microseg(ref_segs, hyp_segs, duration, collar=collar),
        "jer": compute_jer(ref_matrix, sys_matrix),
        "n_ref_spk": len(ref_spks),
        "n_hyp_spk": len(sys_spks),
        "duration": duration,
    }


def score_all(samples: list[dict], collar: float, frame_rate: int) -> dict:
    ders = []
    jers = []
    all_spk_jers = []
    total_scored = total_miss = total_fa = total_spk_err = 0.0
    details = []

    for sample in samples:
        m = score_sample(sample, frame_rate, collar)
        ders.append(m["der"]["der"])
        jers.append(m["jer"]["jer"])
        all_spk_jers.extend(m["jer"].get("per_spk_jer", []))

        total_scored += m["der"]["scored_s"]
        total_miss += m["der"]["miss_s"]
        total_fa += m["der"]["fa_s"]
        total_spk_err += m["der"]["spk_err_s"]

        details.append({
            "id": sample.get("id", ""),
            "duration": round(m["duration"], 3),
            "n_ref_spk": m["n_ref_spk"],
            "n_hyp_spk": m["n_hyp_spk"],
            "DER": m["der"]["der"],
            "DER_miss": m["der"]["miss"],
            "DER_fa": m["der"]["fa"],
            "DER_spk_err": m["der"]["spk_err"],
            "JER": m["jer"]["jer"],
        })

    n = len(samples)
    if n == 0:
        return {"metrics": {}, "eval_details": []}

    global_der = ((total_miss + total_fa + total_spk_err) / total_scored * 100
                  if total_scored > 0 else 0.0)
    global_miss = total_miss / total_scored * 100 if total_scored > 0 else 0.0
    global_fa = total_fa / total_scored * 100 if total_scored > 0 else 0.0
    global_spk_err = total_spk_err / total_scored * 100 if total_scored > 0 else 0.0
    global_jer = float(np.mean(all_spk_jers)) if all_spk_jers else 0.0

    count_correct = sum(1 for d in details if d["n_ref_spk"] == d["n_hyp_spk"])
    count_mae = sum(abs(d["n_ref_spk"] - d["n_hyp_spk"]) for d in details) / n

    metrics = {
        "DER": round(global_der, 2),
        "DER_miss": round(global_miss, 2),
        "DER_fa": round(global_fa, 2),
        "DER_spk_err": round(global_spk_err, 2),
        "DER_sample_avg": round(float(np.mean(ders)), 2),
        "DER_sample_median": round(float(np.median(ders)), 2),
        "JER": round(global_jer, 2),
        "JER_sample_avg": round(float(np.mean(jers)), 2),
        "JER_sample_median": round(float(np.median(jers)), 2),
        "spk_count_accuracy": round(count_correct / n, 4),
        "spk_count_mae": round(count_mae, 2),
    }

    return {"metrics": metrics, "eval_details": details}


def convert_v2_payload(payload: dict[str, Any]) -> dict[str, Any]:

    converted = dict(payload)
    raw_results = payload.get("results")
    if not isinstance(raw_results, dict):
        return converted


    if isinstance(raw_results.get("collars"), dict):
        converted["result_schema_version"] = 3
        return converted

    legacy_collars = {
        str(key): value
        for key, value in raw_results.items()
        if str(key).startswith("collar=") and isinstance(value, dict)
    }
    if not legacy_collars:
        return converted

    preferred = legacy_collars.get("collar=0.25")
    if preferred is None:
        preferred = legacy_collars[next(iter(legacy_collars))]

    preferred_metrics = preferred.get("metrics", {})
    if not isinstance(preferred_metrics, dict):
        preferred_metrics = {}
    shared_metrics = {
        key: value
        for key, value in preferred_metrics.items()
        if not str(key).startswith("DER")
    }

    preferred_details = preferred.get("eval_details", [])
    if not isinstance(preferred_details, list):
        preferred_details = []
    shared_details = [
        {
            key: value
            for key, value in detail.items()
            if not str(key).startswith("DER")
        }
        for detail in preferred_details
        if isinstance(detail, dict)
    ]

    preferred_params = preferred.get("eval_params", {})
    if not isinstance(preferred_params, dict):
        preferred_params = {}
    shared_params = {
        key: value
        for key, value in preferred_params.items()
        if key != "collar"
    }

    collars: dict[str, dict[str, Any]] = {}
    for collar_key, result in legacy_collars.items():
        metrics = result.get("metrics", {})
        details = result.get("eval_details", [])
        params = result.get("eval_params", {})
        if not isinstance(metrics, dict):
            metrics = {}
        if not isinstance(details, list):
            details = []
        if not isinstance(params, dict):
            params = {}
        collars[collar_key] = {
            "metrics": {
                key: value
                for key, value in metrics.items()
                if str(key).startswith("DER")
            },
            "eval_details": [
                {
                    key: value
                    for key, value in detail.items()
                    if key == "id" or str(key).startswith("DER")
                }
                for detail in details
                if isinstance(detail, dict)
            ],
            "eval_params": {
                "collar": params.get(
                    "collar", float(collar_key.removeprefix("collar="))
                ),
            },
        }

    converted["result_schema_version"] = 3
    converted["results"] = {
        "metrics": shared_metrics,
        "eval_details": shared_details,
        "eval_params": shared_params,
        "collars": collars,
    }
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="SDR scoring: DER/JER and WER/cpWER/tcpWER")
    parser.add_argument("--input", required=True,
                        help="Inference JSONL containing model_infer_result")
    parser.add_argument("--collar", type=float, nargs="+", default=[0.0],
                        help="DER collars in seconds (default: 0.0)")
    parser.add_argument("--tcp-collar", type=float, default=5.0,
                        help="tcpWER time collar in seconds (default: 5.0)")
    parser.add_argument("--tokenize", choices=["mixed", "char"], default="mixed",
                        help="Tokenization: mixed=CJK characters and other words; char=characters")
    parser.add_argument("--hallu-ratio", type=float, default=2.0,
                        help="Upper hypothesis/reference token ratio (default: 2.0)")
    parser.add_argument("--trunc-ratio", type=float, default=0.5,
                        help="Lower hypothesis/reference token ratio (default: 0.5)")
    parser.add_argument("--resolution", type=float, default=10.0,
                        help="JER frame resolution in milliseconds (default: 10)")
    parser.add_argument("--model-path", default="",
                        help="Model path for output metadata")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = input_path.parent
    base_name = input_path.stem
    resolved_jsonl_path = output_dir / f"{base_name}_resolved.jsonl"
    resolved_json_path = output_dir / f"{base_name}_resolved.json"


    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print(f"[ERROR] Input file is empty: {input_path}", file=sys.stderr)
        sys.exit(1)


    parse_fail_count = 0
    for entry in entries:
        chunk_range = entry.get("chunk_range", [0.0, 0.0])
        time_offset = chunk_range[0] if chunk_range else 0.0
        model_result = entry.get("model_infer_result", "")
        segments_hyp = parse_sdr_output(model_result, time_offset)
        entry["segments_hyp"] = segments_hyp
        if not segments_hyp and model_result.strip():
            parse_fail_count += 1


    with open(resolved_jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[INFO] Resolved JSONL: {resolved_jsonl_path}")
    print(f"  Samples: {len(entries)}")
    hyp_counts = [len(e["segments_hyp"]) for e in entries]
    print(f"  segments_hyp: min={min(hyp_counts)}, max={max(hyp_counts)}, "
          f"avg={sum(hyp_counts)/len(hyp_counts):.1f}")
    if parse_fail_count:
        print(f"  [WARN] {parse_fail_count} non-empty predictions yielded no valid segments")


    print(f"\n[INFO] Computing WER/cpWER/tcpWER (tokenize={args.tokenize}, "
          f"tcp_collar={args.tcp_collar}, hallu_ratio={args.hallu_ratio}, "
          f"trunc_ratio={args.trunc_ratio}) ...")
    wer_all = score_wer_all(entries, args.tokenize, args.tcp_collar,
                            hallu_ratio=args.hallu_ratio,
                            trunc_ratio=args.trunc_ratio)
    wm = wer_all["metrics"]
    if "WER" not in wm:
        print(f"  [N/A] No reference transcripts (no_ref={wm['wer_noref_count']}/"
              f"{len(entries)}); skipping WER/cpWER/tcpWER")
    else:
        print(f"  hallu_rate={wm['hallu_rate']:.4f} ({wm['hallu_count']} overlong) "
              f"trunc_rate={wm['trunc_rate']:.4f} ({wm['trunc_count']} underlong) "
              f"clean={wm['clean_count']}/{wm['wer_scored_count']} "
              f"(no_ref={wm['wer_noref_count']})")
        print(f"  [clean] WER={wm['WER']:.2f}%  cpWER={wm['cpWER']:.2f}%  "
              f"tcpWER={wm['tcpWER']:.2f}%")
        print(f"  [all]   WER={wm['WER_all']:.2f}%  cpWER={wm['cpWER_all']:.2f}%  "
              f"tcpWER={wm['tcpWER_all']:.2f}%")


    frame_rate = int(1000.0 / args.resolution)
    results = {}

    for collar in args.collar:
        collar_key = f"collar={collar}"
        print(f"\n[INFO] Scoring {collar_key} ...")

        result = score_all(entries, collar, frame_rate)

        result["metrics"].update(wm)
        for d in result["eval_details"]:
            wd = wer_all["details"].get(d["id"])
            if wd:
                d.update({"WER": wd["WER"], "cpWER": wd["cpWER"],
                          "tcpWER": wd["tcpWER"], "len_ratio": wd["len_ratio"],
                          "status": wd["status"]})
        result["eval_params"] = {
            "collar": collar,
            "frame_resolution_ms": args.resolution,
            "tcp_collar": args.tcp_collar,
            "tokenize": args.tokenize,
            "hallu_ratio": args.hallu_ratio,
            "trunc_ratio": args.trunc_ratio,
        }
        results[collar_key] = result

        m = result["metrics"]
        print(f"  DER={m['DER']:.2f}% (miss={m['DER_miss']:.2f}%, "
              f"fa={m['DER_fa']:.2f}%, spk_err={m['DER_spk_err']:.2f}%)")
        print(f"  DER_sample_avg={m['DER_sample_avg']:.2f}%")
        print(f"  JER={m['JER']:.2f}%")
        if "WER" in m:
            print(f"  hallu_rate={m['hallu_rate']:.4f} trunc_rate={m['trunc_rate']:.4f}  "
                  f"[clean] WER={m['WER']:.2f}%  cpWER={m['cpWER']:.2f}%  "
                  f"tcpWER={m['tcpWER']:.2f}%")
        else:
            print("  WER/cpWER/tcpWER: N/A (no reference transcripts)")
        print(f"  spk_count_accuracy={m['spk_count_accuracy']:.4f}, "
              f"mae={m['spk_count_mae']:.2f}")


    eval_set_name = base_name
    if eval_set_name.endswith("_result"):
        eval_set_name = eval_set_name[:-len("_result")]

    output = {
        "eval_set": eval_set_name,
        "task_type": "sdr",
        "model_path": args.model_path or "",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_count": len(entries),
        "resolved_jsonl": resolved_jsonl_path.name,
        "results": results,
    }

    # Store shared metrics once and DER separately for each collar.
    output = convert_v2_payload(output)

    with open(resolved_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Scores: {resolved_json_path}")


if __name__ == "__main__":
    main()
