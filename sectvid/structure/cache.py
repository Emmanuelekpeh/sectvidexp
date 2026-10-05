"""Structure cache: data/structure/<clip_id>.npz (spec 6.1)."""
import numpy as np
from pathlib import Path

from ..config import root_path
from .schema import Structure


def _cache_path(cfg, clip_id):
    return root_path(cfg, cfg["data"]["structure_dir"], f"{clip_id}.npz")


def has_structure(cfg, clip_id):
    """Cheap existence check for a clip's structure cache (no array loads)."""
    return _cache_path(cfg, clip_id).exists()


def validate_structure(cfg, clip_id):
    """Check that a cached npz is readable, without loading the large
    silhouette array. Raises on missing or corrupt cache."""
    p = _cache_path(cfg, clip_id)
    if not p.exists():
        raise FileNotFoundError(f"no structure cache for {clip_id}: {p}")
    with np.load(p) as d:
        for key in ("body", "camera", "ok", "frame_index", "dt", "height", "width"):
            d[key]


def save_structure(cfg, clip_id, structs):
    p = _cache_path(cfg, clip_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = len(structs)
    h = structs[0].silhouette.shape[0]
    w = structs[0].silhouette.shape[1]
    np.savez_compressed(
        p,
        body=np.stack([s.body for s in structs]),
        head_pose=np.stack([s.head_pose for s in structs]),
        head_pos=np.stack([s.head_pos for s in structs]),
        head_scale=np.array([s.head_scale for s in structs], dtype=np.float32),
        head_conf=np.array([s.head_conf for s in structs], dtype=np.float32),
        face_params=np.stack([s.face_params for s in structs]),
        face_box=np.stack([s.face_box for s in structs]),
        silhouette=np.stack([s.silhouette for s in structs]),
        camera=np.stack([s.camera for s in structs]),
        camera_conf=np.array([s.camera_conf for s in structs], dtype=np.float32),
        ok=np.array([s.ok for s in structs], dtype=bool),
        frame_index=np.array([s.frame_index for s in structs], dtype=np.int64),
        dt=np.array([s.dt for s in structs], dtype=np.float32),
        height=h,
        width=w,
    )
    return p


def load_structure(cfg, clip_id, include_silhouette=True):
    """Load a clip's structure cache. include_silhouette=False skips the
    (large) per-frame silhouette arrays: all frames share one zero mask,
    saving ~100s of MB per clip for camera/subject/binning work that never
    touches the silhouette."""
    p = _cache_path(cfg, clip_id)
    if not p.exists():
        raise FileNotFoundError(f"no structure cache for {clip_id}: {p}")
    d = np.load(p)
    n = int(d["body"].shape[0])
    h, w = int(d["height"]), int(d["width"])
    zero_sil = np.zeros((h, w), dtype=np.uint8)
    out = []
    for i in range(n):
        out.append(Structure(
            frame_index=int(d["frame_index"][i]),
            dt=float(d["dt"][i]),
            body=d["body"][i].astype(np.float32),
            head_pose=d["head_pose"][i].astype(np.float32),
            head_pos=d["head_pos"][i].astype(np.float32),
            head_scale=float(d["head_scale"][i]),
            head_conf=float(d["head_conf"][i]),
            face_params=d["face_params"][i].astype(np.float32),
            face_box=d["face_box"][i].astype(np.float32),
            silhouette=d["silhouette"][i] if include_silhouette else zero_sil,
            camera=d["camera"][i].astype(np.float32),
            camera_conf=float(d["camera_conf"][i]),
            ok=bool(d["ok"][i]),
        ))
    return out
