"""Stage 0 signal-vs-noise analysis (owner review, 2026-10-05).

The extractor's keypoint jitter (test 2) is its noise floor as structure
ground truth. This script asks the question that actually decides whether the
extractor is usable: is per-frame subject motion in each motion bin (still /
slow / fast, spec 6.1) larger than that noise floor?

- Noise floor: 2px-jitter keypoint shift, p50/p90, on the deterministic 8-clip
  test sample, measured on the raw path (smooth=False, as in test 2) and on
  the production path (smooth=True, EMA per config).
- Signal: per-frame subject motion in px/frame from the production (smoothed)
  structure, computed on the same valid-point mask as the jitter probe.
- Attribution: frame pairs are binned by the spec's window/bin mechanism
  (build_windows + assign_bin on the production structure).
- Diagnostics for the low-reliability clips (Merantau_clip_0309 / 0327):
  motion, blur, visible-point fraction, face rate, camera translation.

Writes reports/stage0_signal_noise_*.json and prints the decision table.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sectvid.config import load_config, root_path
from sectvid.data import clips as data_clips
from sectvid.data.windows import build_windows
from sectvid.eval import report
from sectvid.structure.extractor import StructureExtractor

CFG = load_config()
V = CFG["structure"]["validation"]
CLIP_IDS = data_clips.discover_clips(CFG)


def pick_clips(n):
    """Same deterministic sample as tests/test_2_extractor_validation.py."""
    props = {cid: data_clips.clip_properties(CFG, cid) for cid in CLIP_IDS}
    pool = [c for c, p in props.items() if p["width"] >= 480 and p["n_frames"] >= V["frames_per_clip"]]
    rng = np.random.default_rng(CFG["seed"])
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[int(i)] for i in idx]


def shift(frame, k):
    out = cv2.copyMakeBorder(frame, 0, 0, 0, k, cv2.BORDER_REPLICATE)
    return out[:, k:]


def jitter_probe(ext, frames, smooth, k):
    """Per-frame-pair mean keypoint shift (px) when the frame is jittered k px."""
    width = frames[0].shape[1]
    shifts = []
    for i in range(0, min(24, len(frames)), 4):
        seq = ext.extract_sequence([frames[i], shift(frames[i], k)], smooth=smooth)
        base, moved = seq[0], seq[1]
        if not base.ok or not moved.ok:
            continue
        valid = (base.body[:, 2] >= 0.5) & (moved.body[:, 2] >= 0.5)
        if valid.sum() < 8:
            continue
        shifts.append(float(np.linalg.norm(moved.body[valid, :2] - base.body[valid, :2], axis=1).mean() * width))
    return shifts


def frame_motion_px(structs, width, min_conf=0.5):
    """Per-frame-pair subject motion (px/frame) on the same valid mask as the
    jitter probe: points with conf >= min_conf in both frames."""
    out = []
    for i in range(len(structs) - 1):
        a, b = structs[i], structs[i + 1]
        if not (a.ok and b.ok):
            out.append(None)
            continue
        valid = (a.body[:, 2] >= min_conf) & (b.body[:, 2] >= min_conf)
        if valid.sum() < 8:
            out.append(None)
            continue
        out.append(float(np.linalg.norm(b.body[valid, :2] - a.body[valid, :2], axis=1).mean() * width))
    return out


def diagnostics(frames, raw, smooth):
    width = frames[0].shape[1]
    motion = frame_motion_px(raw, width)
    motion = [m for m in motion if m is not None]
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    blur = [float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in gray]
    vis = [float((s.body[:, 2] >= 0.5).mean()) for s in raw if s.ok]
    cam = [float(np.linalg.norm(s.camera[:2, 2])) for s in smooth[1:] if s.ok]
    return {
        "n_frames": len(frames),
        "width": width,
        "motion_px_per_frame_p50": float(np.percentile(motion, 50)) if motion else None,
        "motion_px_per_frame_p90": float(np.percentile(motion, 90)) if motion else None,
        "blur_laplacian_var_mean": float(np.mean(blur)),
        "blur_laplacian_var_p50": float(np.percentile(blur, 50)),
        "visible_point_fraction": float(np.mean(vis)) if vis else None,
        "face_rate": float(np.mean([s.head_conf > 0 for s in raw])),
        "camera_translation_px_p50": float(np.percentile(cam, 50)) if cam else None,
    }


def main():
    ext = StructureExtractor(CFG)
    picks = pick_clips(V["clips"])
    noise_raw, noise_smooth, noise_raw_per_clip, noise_smooth_per_clip = [], [], {}, {}
    bins_payload = {"still": [], "slow": [], "fast": []}
    per_clip = {}
    raw_by_clip, smooth_by_clip = {}, {}

    for cid in picks:
        frames = [f for f in data_clips.load_frames(CFG, cid, 0, 48) if f.mean() > 1.0]
        if len(frames) < 8:
            continue
        width = frames[0].shape[1]
        raw = ext.extract_sequence(frames, smooth=False)
        smooth = ext.extract_sequence(frames, smooth=True)
        raw_by_clip[cid], smooth_by_clip[cid] = raw, smooth
        k = int(V["jitter_px"])
        nr = jitter_probe(ext, frames, smooth=False, k=k)
        ns = jitter_probe(ext, frames, smooth=True, k=k)
        noise_raw.extend(nr)
        noise_smooth.extend(ns)
        noise_raw_per_clip[cid] = {"p50_px": float(np.percentile(nr, 50)) if nr else None,
                                   "p90_px": float(np.percentile(nr, 90)) if nr else None, "n": len(nr)}
        noise_smooth_per_clip[cid] = {"p50_px": float(np.percentile(ns, 50)) if ns else None,
                                      "p90_px": float(np.percentile(ns, 90)) if ns else None, "n": len(ns)}

        motion = frame_motion_px(smooth, width)
        windows = build_windows(CFG, cid, smooth)
        frame_bins = [None] * len(frames)
        for w in windows:
            for i in range(w["start"], min(w["end"], len(frames))):
                if frame_bins[i] is None:
                    frame_bins[i] = w["bin"]
        for i, m in enumerate(motion):
            if m is None:
                continue
            b = frame_bins[i]
            if b in bins_payload:
                bins_payload[b].append(m)
        per_clip[cid] = {
            "n_frames": len(frames), "width": width,
            "windows_per_bin": {b: sum(1 for w in windows if w["bin"] == b) for b in ("still", "slow", "fast")},
        }

    def stats(xs):
        if not xs:
            return {"n": 0}
        a = np.array(xs)
        return {"n": len(a), "p50_px": float(np.percentile(a, 50)),
                "p90_px": float(np.percentile(a, 90)), "max_px": float(a.max())}

    noise = {
        "raw_smoothing_off": stats(noise_raw),
        "production_smoothing_on": stats(noise_smooth),
    }
    bins = {b: stats(xs) for b, xs in bins_payload.items()}
    table = {}
    for b in ("still", "slow", "fast"):
        s = bins[b]
        if s.get("n", 0) == 0:
            table[b] = {"motion": None, "ratio_vs_noise_p90": None}
            continue
        table[b] = {
            "motion_p50_px_per_frame": s["p50_px"],
            "noise_p50_px": noise["production_smoothing_on"]["p50_px"],
            "noise_p90_px": noise["production_smoothing_on"]["p90_px"],
            "motion_p50_over_noise_p90": round(s["p50_px"] / noise["production_smoothing_on"]["p90_px"], 2),
        }

    flagged = [c for c in picks if c in ("Merantau_clip_0309", "Merantau_clip_0327")]
    controls = [c for c in picks if c not in ("Merantau_clip_0309", "Merantau_clip_0327")]
    diag = {}
    for cid in flagged + controls[:1]:
        if cid in raw_by_clip:
            frames = data_clips.load_frames(CFG, cid, 0, 48)
            frames = [f for f in frames if f.mean() > 1.0]
            diag[cid] = diagnostics(frames, raw_by_clip[cid], smooth_by_clip[cid])

    payload = {
        "stage": 0,
        "purpose": "signal-vs-noise for extractor-as-ground-truth (owner review 2026-10-05)",
        "resolution_note": "all sampled clips are 512px wide; px values are at 512px",
        "noise_floor": noise,
        "noise_per_clip": {"raw": noise_raw_per_clip, "production": noise_smooth_per_clip},
        "motion_per_bin": bins,
        "decision_table": table,
        "per_clip": per_clip,
        "diagnostics_low_reliability": diag,
    }
    p = report.write_json_report(root_path(CFG, CFG["report"]["dir"]), "stage0_signal_noise",
                                 payload, CFG, CFG["seed"])
    print(json.dumps({"noise": noise, "bins": bins, "table": table,
                      "diagnostics": diag, "report": str(p)}, indent=1))


if __name__ == "__main__":
    main()
