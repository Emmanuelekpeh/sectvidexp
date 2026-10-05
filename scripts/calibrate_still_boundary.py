"""Calibrate the still-bin boundary (owner decision, option B, 2026-10-05).

A still window is one with "no reliably detectable motion": the subject's
per-frame displacement is not reliably above the extractor's jitter noise
floor, and its net window displacement is not reliably above the noise
random-walk bound. This script:

  1. Ranks all cached clips by observed per-frame subject speed (camera
     corrected) to find still candidates.
  2. Runs the 2px jitter probe on the still candidates + known-slow controls
     to measure each clip's noise floor.
  3. Reports, per candidate: per-frame speed p50/p90 (px), 32-frame window
     net displacement p50 (px), jitter noise p50/p90 (px), and the two
     separation ratios used for the boundary.
  4. Writes contact sheets (real frames) so a human can verify which windows
     are genuinely still vs. slow (spec 9: a human must look at the sequences).

Writes reports/still_calibration_*.json + reports/still_sheets_*.png.
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sectvid.config import load_config, root_path
from sectvid.data import clips as data_clips
from sectvid.data import windows as W
from sectvid.eval import report
from sectvid.structure import cache
from sectvid.structure.extractor import StructureExtractor
from sectvid.structure.schema import step_speed

N_CANDIDATES = 10
N_CONTROLS = 4
JITTER_FRAMES = 24
JITTER_K = 2


def centroid_px(s, width, min_conf=0.5):
    valid = s.body[:, 2] >= min_conf
    if valid.sum() < 8:
        return None
    return s.body[valid, :2] * np.array([width, s.silhouette.shape[0]], dtype=np.float64)


def subject_speeds_px(structs, width):
    """Per-frame SUBJECT displacement (px) on observed steps, camera-corrected.
    Requires camera_conf > 0 so fallback identity transform doesn't pollute
    subject-motion estimates (spec 6.2)."""
    from sectvid.structure.schema import camera_corrected_prev_points
    speeds, nets = [], []
    for i in range(len(structs) - 1):
        a, b = structs[i], structs[i + 1]
        if not W.step_observed(a, b, 0.5, require_camera_conf=True):
            continue
        pa, _ = camera_corrected_prev_points(a, b, 0.5)
        valid = (a.body[:, 2] >= 0.5) & (b.body[:, 2] >= 0.5)
        if valid.sum() < 8:
            continue
        d = (b.body[valid, :2] - pa[valid]) * np.array([width, b.silhouette.shape[0]], dtype=np.float64)
        speeds.append(float(np.linalg.norm(d, axis=1).mean()))
    return speeds


def net_displacement_px(structs, width, wstart):
    """Window (32 frames) net SUBJECT displacement (px): chain of
    camera-corrected per-frame displacements — a still subject random-walks
    around its position; a moving subject accumulates.
    Requires camera_conf > 0 so fallback identity transform doesn't pollute
    subject-motion estimates (spec 6.2)."""
    from sectvid.structure.schema import camera_corrected_prev_points
    wlen = 32
    seg = structs[wstart:wstart + wlen]
    if len(seg) < 16:
        return None
    acc = np.zeros(2)
    for i in range(1, len(seg)):
        a, b = seg[i - 1], seg[i]
        if not W.step_observed(a, b, 0.5, require_camera_conf=True):
            continue
        pa, _ = camera_corrected_prev_points(a, b, 0.5)
        valid = (a.body[:, 2] >= 0.5) & (b.body[:, 2] >= 0.5)
        if valid.sum() < 8:
            continue
        acc += (b.body[valid, :2] - pa[valid]).mean(axis=0) * np.array([width, b.silhouette.shape[0]], dtype=np.float64)
    return float(np.linalg.norm(acc))


def jitter_probe(ext, frames, width):
    shifts = []
    for i in range(0, min(JITTER_FRAMES, len(frames)), 4):
        out = cv2.copyMakeBorder(frames[i], 0, 0, 0, JITTER_K, cv2.BORDER_REPLICATE)
        seq = ext.extract_sequence([frames[i], out[:, JITTER_K:]], smooth=False)
        if not seq[0].ok or not seq[1].ok:
            continue
        valid = (seq[0].body[:, 2] >= 0.5) & (seq[1].body[:, 2] >= 0.5)
        if valid.sum() < 8:
            continue
        shifts.append(float(np.linalg.norm(
            seq[1].body[valid, :2] - seq[0].body[valid, :2], axis=1).mean() * width))
    return shifts


def main():
    cfg = load_config()
    ext = StructureExtractor(cfg)
    ids = data_clips.discover_clips(cfg)

    ranked = []
    for cid in ids:
        try:
            structs = cache.load_structure(cfg, cid, include_silhouette=False)
        except FileNotFoundError:
            continue
        if len(structs) < 64:
            continue
        props = data_clips.clip_properties(cfg, cid)
        width = props["width"]
        speeds = subject_speeds_px(structs, width)
        if len(speeds) < 16:
            continue
        obs_frac = len(speeds) / (len(structs) - 1)
        if obs_frac < 0.4:
            continue
        nets = [net_displacement_px(structs, width, s) for s in range(0, len(structs) - 32, 32)]
        nets = [n for n in nets if n is not None]
        ranked.append({
            "clip": cid, "width": width,
            "speed_p50": float(np.percentile(speeds, 50)),
            "speed_p90": float(np.percentile(speeds, 90)),
            "net_p50": float(np.percentile(nets, 50)) if nets else None,
            "obs_frac": obs_frac,
        })
    ranked.sort(key=lambda r: r["speed_p50"])
    candidates = [r["clip"] for r in ranked[:N_CANDIDATES]]
    controls = [r["clip"] for r in ranked[len(ranked) - N_CONTROLS:]]
    print(f"still candidates (lowest observed subject speed): {candidates}")
    print(f"slow controls: {controls}")

    rows, sheets = [], []
    for cid in candidates + controls:
        structs = cache.load_structure(cfg, cid, include_silhouette=False)
        props = data_clips.clip_properties(cfg, cid)
        width = props["width"]
        shifts = jitter_probe(ext, data_clips.load_frames(cfg, cid, 0, 64), width)
        speeds = subject_speeds_px(structs, width)
        nets = [net_displacement_px(structs, width, s) for s in range(0, len(structs) - 32, 32)]
        nets = [n for n in nets if n is not None]
        noise_p50 = float(np.percentile(shifts, 50)) if shifts else None
        noise_p90 = float(np.percentile(shifts, 90)) if shifts else None
        # slowest 32-frame window for the contact sheet - use frame indices, not speeds indices
        def window_mean_speed(frame_start):
            frame_end = min(frame_start + 32, len(structs))
            obs_speeds = []
            for i in range(frame_start, frame_end - 1):
                a, b = structs[i], structs[i + 1]
                if W.step_observed(a, b, 0.5, require_camera_conf=True):
                    dt = b.dt if b.dt > 0 else a.dt
                    dt = dt if dt > 0 else 1.0 / 24.0
                    obs_speeds.append(step_speed(a, b, 0.5) / dt)
            return float(np.mean(obs_speeds)) if obs_speeds else float('inf')
        w_idx = min(range(max(len(structs) - 32, 1)), key=window_mean_speed)
        # load frames at the actual window position for the contact sheet
        frames_at_window = data_clips.load_frames(cfg, cid, w_idx, w_idx + 8)
        strip = frames_at_window
        label = f"{cid} speed_p50={np.percentile(speeds, 50):.1f}px net_p50={np.percentile(nets, 50) if nets else 0:.1f}px noise_p90={noise_p90 if noise_p90 else 0:.1f}px"
        sheets.append((strip, label))
        rows.append({
            "clip": cid,
            "speed_p50_px": float(np.percentile(speeds, 50)),
            "speed_p90_px": float(np.percentile(speeds, 90)),
            "net_p50_px": float(np.percentile(nets, 50)) if nets else None,
            "jitter_p50_px": noise_p50,
            "jitter_p90_px": noise_p90,
            "speed_p50_over_noise_p90": (float(np.percentile(speeds, 50)) / noise_p90) if noise_p90 else None,
            "net_p50_over_sqrtn_noise_p90": (float(np.percentile(nets, 50)) / (np.sqrt(31) * noise_p90)) if (nets and noise_p90) else None,
            "obs_frac": None,
        })

    rows.sort(key=lambda r: (r["speed_p50_px"] or 0))
    payload = {"purpose": "still-bin boundary calibration (owner option B, 2026-10-05)",
               "clips": rows}
    p = report.write_json_report(root_path(cfg, cfg["report"]["dir"]), "still_calibration", payload, cfg, cfg["seed"])
    sheet_path = root_path(cfg, cfg["report"]["dir"]) / f"still_sheets_{cfg['seed']}.png"
    report.render_contact_sheet(sheets, sheet_path, cols=8, scale=1)
    print(json.dumps({"report": str(p), "sheet": str(sheet_path), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
