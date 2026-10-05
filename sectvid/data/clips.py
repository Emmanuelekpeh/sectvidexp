"""Clip discovery, properties, and meta labels (spec 6.1)."""
import json

import cv2
from pathlib import Path

from ..config import root_path

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def clips_dir(cfg):
    return root_path(cfg, cfg["data"]["clips_dir"])


def meta_dir(cfg):
    return root_path(cfg, cfg["data"]["meta_dir"])


def discover_clips(cfg):
    """All video files under the data dir, recursively. Clip id = filename without extension."""
    found = []
    for p in sorted(clips_dir(cfg).rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            found.append(p.stem)
    return found


def clip_path(cfg, clip_id):
    matches = list(clips_dir(cfg).rglob(f"{clip_id}{VIDEO_EXTS[0]}"))
    for ext in VIDEO_EXTS[1:]:
        matches += list(clips_dir(cfg).rglob(f"{clip_id}{ext}"))
    if not matches:
        raise FileNotFoundError(clip_id)
    return matches[0]


def clip_properties(cfg, clip_id):
    cap = cv2.VideoCapture(str(clip_path(cfg, clip_id)))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip {clip_id}")
    props = {
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return props


def default_meta(clip_id, props):
    return {
        "identity_id": clip_id,
        "behavior": "unlabeled",
        "direction": "null",
        "fps": props["fps"],
        "camera_motion_stats": None,
        "bins": None,
        "extractor_reliability": "ok",
        # False until a human assigns/verifies the identity (spec 6.1: splits
        # are held out by identity, so unreviewed defaults must not be used).
        "identity_reviewed": False,
    }


def normalized_meta(cfg, clip_id):
    """load_meta with missing keys filled from default_meta (migration-safe
    for meta files written before a key existed)."""
    meta = load_meta(cfg, clip_id)
    base = default_meta(clip_id, clip_properties(cfg, clip_id))
    for key, val in base.items():
        meta.setdefault(key, val)
    return meta


def set_meta(cfg, clip_id, updates):
    """Update (or create) a clip's meta JSON with the given fields."""
    p = meta_dir(cfg) / f"{clip_id}.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = default_meta(clip_id, clip_properties(cfg, clip_id))
    meta.update(updates)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def load_meta(cfg, clip_id):
    p = meta_dir(cfg) / f"{clip_id}.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return default_meta(clip_id, clip_properties(cfg, clip_id))


def ensure_meta(cfg, clip_id):
    p = meta_dir(cfg) / f"{clip_id}.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(default_meta(clip_id, clip_properties(cfg, clip_id)), f, indent=1)
    return load_meta(cfg, clip_id)


def identity_of(clip_id, cfg):
    meta = load_meta(cfg, clip_id)
    return str(meta.get("identity_id") or clip_id)


def load_frames(cfg, clip_id, start=0, end=None, resize=None):
    """Decode frames [start, end). end=None reads to the clip's end."""
    cap = cv2.VideoCapture(str(clip_path(cfg, clip_id)))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip {clip_id}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None:
        end = total
    frames = []
    if int(start) > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    for i in range(int(start), int(end)):
        ret, frame = cap.read()
        if not ret:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    return frames
