"""Structure encoder/decoder for planner (spec 6.3).

Converts Structure objects to/from flat feature vectors for the transformer.
"""
import dataclasses
import numpy as np
from typing import Tuple

from ..structure.schema import Structure, NUM_BODY_POINTS, torso_length


# Feature dimensions
BODY_DIM = NUM_BODY_POINTS * 3  # 17 * 3 = 51 (x, y, conf)
HEAD_POSE_DIM = 3  # yaw, pitch, roll
HEAD_POS_DIM = 2  # x, y normalized
HEAD_SCALE_DIM = 1
HEAD_CONF_DIM = 1
FACE_PARAMS_DIM = 52  # blendshapes
FACE_BOX_DIM = 4  # x, y, w, h
DT_DIM = 1
# Camera is not predicted (pixel-space, estimated from background)
# Silhouette is not predicted (pixel-space, for drawer)

TOTAL_FEATURE_DIM = (
    BODY_DIM + HEAD_POSE_DIM + HEAD_POS_DIM + HEAD_SCALE_DIM + HEAD_CONF_DIM +
    FACE_PARAMS_DIM + FACE_BOX_DIM + DT_DIM
)  # 51 + 3 + 2 + 1 + 1 + 52 + 4 + 1 = 115


def encode_structure(s: Structure) -> np.ndarray:
    """Encode a Structure into a flat feature vector (float32)."""
    feats = np.concatenate([
        s.body.astype(np.float32).flatten(),      # 51
        s.head_pose.astype(np.float32),           # 3
        s.head_pos.astype(np.float32),            # 2
        np.array([s.head_scale], dtype=np.float32),  # 1
        np.array([s.head_conf], dtype=np.float32),   # 1
        s.face_params.astype(np.float32),         # 52
        s.face_box.astype(np.float32),            # 4
        np.array([s.dt], dtype=np.float32),       # 1
    ])
    assert feats.shape == (TOTAL_FEATURE_DIM,), f"Expected {TOTAL_FEATURE_DIM}, got {feats.shape}"
    return feats


def decode_structure(feats: np.ndarray, frame_index: int, height: int, width: int, prev: Structure) -> Structure:
    """Decode a flat feature vector into a Structure.
    
    Camera and silhouette are inherited from prev (not predicted).
    ok flag is set based on body confidence.
    """
    assert feats.shape == (TOTAL_FEATURE_DIM,), f"Expected {TOTAL_FEATURE_DIM}, got {feats.shape}"
    
    idx = 0
    body = feats[idx:idx + BODY_DIM].reshape(NUM_BODY_POINTS, 3)
    idx += BODY_DIM
    head_pose = feats[idx:idx + HEAD_POSE_DIM]
    idx += HEAD_POSE_DIM
    head_pos = feats[idx:idx + HEAD_POS_DIM]
    idx += HEAD_POS_DIM
    head_scale = float(feats[idx])
    idx += HEAD_SCALE_DIM
    head_conf = float(feats[idx])
    idx += HEAD_CONF_DIM
    face_params = feats[idx:idx + FACE_PARAMS_DIM]
    idx += FACE_PARAMS_DIM
    face_box = feats[idx:idx + FACE_BOX_DIM]
    idx += FACE_BOX_DIM
    dt = float(feats[idx])
    
    # Clamp body coordinates to [0, 1]
    body = body.copy()
    body[:, :2] = np.clip(body[:, :2], 0.0, 1.0)
    body[:, 2] = np.clip(body[:, 2], 0.0, 1.0)
    
    # Clamp face box
    face_box = np.clip(face_box, 0.0, 1.0)
    
    # Clamp dt to reasonable range
    dt = np.clip(dt, 1.0 / 60.0, 1.0 / 8.0)
    
    # Determine ok flag from body confidence
    ok = bool((body[:, 2] >= 0.3).sum() >= 4)
    
    return Structure(
        frame_index=frame_index,
        dt=dt,
        body=body.astype(np.float32),
        head_pose=head_pose.astype(np.float32),
        head_pos=head_pos.astype(np.float32),
        head_scale=head_scale,
        head_conf=head_conf,
        face_params=face_params.astype(np.float32),
        face_box=face_box.astype(np.float32),
        silhouette=prev.silhouette,  # Not predicted, inherit
        camera=prev.camera,           # Not predicted, inherit
        camera_conf=prev.camera_conf, # Not predicted, inherit
        ok=ok,
    )


def get_body_confidence_mask(body: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
    """Get boolean mask of valid body keypoints (N,)."""
    return body[:, 2] >= min_conf


def get_feature_weights() -> np.ndarray:
    """Per-feature loss weights: higher weight for position, lower for confidence."""
    weights = np.ones(TOTAL_FEATURE_DIM, dtype=np.float32)
    idx = 0
    # Body: higher weight for position (x,y), lower for confidence
    for _ in range(NUM_BODY_POINTS):
        weights[idx] = 1.0     # x
        weights[idx + 1] = 1.0 # y
        weights[idx + 2] = 0.3 # conf
        idx += 3
    # Head pose: all equal
    weights[idx:idx + HEAD_POSE_DIM] = 1.0
    idx += HEAD_POSE_DIM
    # Head pos: equal
    weights[idx:idx + HEAD_POS_DIM] = 1.0
    idx += HEAD_POS_DIM
    # Head scale: lower weight
    weights[idx] = 0.5
    idx += HEAD_SCALE_DIM
    # Head conf: low weight
    weights[idx] = 0.2
    idx += HEAD_CONF_DIM
    # Face params: blendshapes, moderate weight
    weights[idx:idx + FACE_PARAMS_DIM] = 0.3
    idx += FACE_PARAMS_DIM
    # Face box: moderate weight
    weights[idx:idx + FACE_BOX_DIM] = 0.5
    idx += FACE_BOX_DIM
    # dt: moderate weight
    weights[idx] = 0.5
    return weights


def compute_torso_normalizer(body: np.ndarray) -> float:
    """Get torso length for normalizing keypoint errors."""
    tl = torso_length(body)
    return max(tl, 1e-4)  # Avoid division by zero


if __name__ == "__main__":
    # Quick test
    s = Structure.zeros(0, 1/24.0, 48, 64)
    s.body[:, 2] = 1.0
    s.body[:, 0] = 0.5
    s.body[:, 1] = 0.5
    s.head_pose = np.array([0.1, 0.0, 0.0])
    s.head_pos = np.array([0.5, 0.3])
    s.head_scale = 0.1
    s.head_conf = 1.0
    s.face_params[:] = 0.0
    s.face_box = np.array([0.3, 0.1, 0.2, 0.2])
    s.dt = 1/24.0
    
    feats = encode_structure(s)
    print(f"Encoded shape: {feats.shape}")
    
    s2 = decode_structure(feats, 1, 48, 64, s)
    print(f"Decoded ok: {s2.ok}")
    print(f"Body shape: {s2.body.shape}")
    print(f"dt: {s2.dt}")