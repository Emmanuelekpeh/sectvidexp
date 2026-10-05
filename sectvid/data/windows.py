"""Windowing and motion bins (spec 6.1).

Windows are the unit of sampling. Bins classify subject motion (camera removed)
into still / slow / fast. Still windows are kept, not deleted, and sampling uses
fixed quotas so no bin can dominate.

Windows where the pose extractor could not observe enough frames get the
"unobserved" label, NOT a motion bin: extractor failure must not be counted
as zero motion, or missing poses would teach the model to hold still.
"""
import numpy as np

from ..structure.schema import step_speed


def cut_windows(n_frames, length, stride):
    out = []
    for start in range(0, max(n_frames - length + 1, 0), stride):
        out.append((start, start + length))
    if not out and n_frames >= length:
        out = [(n_frames - length, n_frames)]
    return out


def assign_bin(subject_speed, bins_cfg):
    if subject_speed < bins_cfg["still_max_speed"]:
        return "still"
    if subject_speed >= bins_cfg["fast_min_speed"]:
        return "fast"
    return "slow"


def step_observed(a, b, min_conf=0.3, require_camera_conf=False):
    """True when both frames carry a usable pose (both ok, >=4 points with
    conf >= min_conf). Such a step can contribute to the subject-motion
    estimate, whether the subject moved or was genuinely still.
    
    If require_camera_conf=True, also requires b.camera_conf > 0 so that
    camera-corrected subject motion is not computed from a fallback identity
    transform (which would measure raw image motion = camera + subject)."""
    if not (a.ok and b.ok):
        return False
    if require_camera_conf and b.camera_conf <= 0.0:
        return False
    return int(((a.body[:, 2] >= min_conf) & (b.body[:, 2] >= min_conf)).sum()) >= 4


def window_subject_speed(structs, min_conf=0.3):
    """(mean SUBJECT speed in normalized units/second over observed steps,
    observed step fraction). Steps with missing/low-confidence detections are
    excluded from the speed average rather than counted as zero motion; the
    fraction reports how much of the window was actually observed.
    
    Requires camera_conf > 0 for camera-corrected subject motion (spec 6.2:
    bins classify subject motion with camera removed). Steps with zero camera
    confidence are treated as unobserved for subject motion."""
    if len(structs) < 2:
        return 0.0, 0.0
    obs, speeds = [], []
    for i in range(len(structs) - 1):
        a, b = structs[i], structs[i + 1]
        if step_observed(a, b, min_conf, require_camera_conf=True):
            obs.append(1)
            dt = b.dt if b.dt > 0 else a.dt
            dt = dt if dt > 0 else 1.0 / 24.0
            speeds.append(step_speed(a, b, min_conf) / dt)
        else:
            obs.append(0)
    frac = float(np.mean(obs))
    speed = float(np.mean(speeds)) if speeds else 0.0
    return speed, frac


def window_camera_stats(structs):
    """Coarse camera stats from estimated affines: mean translation magnitude."""
    if len(structs) < 2:
        return {"mean_translation": 0.0, "camera_conf_mean": 0.0}
    trans = []
    confs = []
    for s in structs[1:]:
        t = s.camera[:2, 2]
        if s.camera_conf > 0.0:
            confs.append(s.camera_conf)
        trans.append(float(np.linalg.norm(t)))
    return {
        "mean_translation": float(np.mean(trans)) if trans else 0.0,
        "camera_conf_mean": float(np.mean(confs)) if confs else 0.0,
    }


def build_windows(cfg, clip_id, structs):
    n = len(structs)
    wlen = cfg["window"]["length"]
    stride = cfg["window"]["stride"]
    bins_cfg = cfg["motion_bins"]
    out = []
    min_obs = float(bins_cfg.get("min_observed_frac", 0.5))
    for start, end in cut_windows(n, wlen, stride):
        window = structs[start:end]
        subject, obs_frac = window_subject_speed(window)
        bin_name = "unobserved" if obs_frac < min_obs else assign_bin(subject, bins_cfg)
        out.append({
            "clip_id": clip_id,
            "start": int(start),
            "end": int(end),
            "subject_speed": subject,
            "observed_frac": obs_frac,
            "bin": bin_name,
            "camera_stats": window_camera_stats(window),
        })
    return out


def sample_windows(windows, quotas, n, seed):
    """Fixed-quota sampling across bins (spec 6.1). Never lets one bin dominate."""
    rng = np.random.default_rng(seed)
    by_bin = {"still": [], "slow": [], "fast": [], "unobserved": []}
    for w in windows:
        by_bin[w["bin"]].append(w)
    chosen = []
    for bin_name, quota in quotas.items():
        count = int(round(quota * n))
        pool = by_bin[bin_name]
        if len(pool) == 0:
            continue
        idx = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
        chosen.extend(pool[int(i)] for i in idx)
    return chosen
