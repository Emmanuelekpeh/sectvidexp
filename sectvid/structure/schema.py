"""Per-frame structure schema (spec 6.2).

Coordinates are image-normalized to [0, 1] (x right, y down).
Body keypoints follow the COCO-17 order:
  0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear,
  5 left shoulder, 6 right shoulder, 7 left elbow, 8 right elbow,
  9 left wrist, 10 right wrist, 11 left hip, 12 right hip,
  13 left knee, 14 right knee, 15 left ankle, 16 right ankle
where left/right refer to the subject's left/right.
"""
import dataclasses

import numpy as np

NUM_BODY_POINTS = 17
# subject-left / subject-right index pairs
LEFT_RIGHT_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
# COCO-17 names, subject-side
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

DIRECTIONS = ("left", "right", "toward", "away", "none", "null")


@dataclasses.dataclass
class Structure:
    frame_index: int
    dt: float
    body: np.ndarray          # (17, 3) float32 [x, y, conf]
    head_pose: np.ndarray     # (3,) float32 [yaw, pitch, roll] radians
    head_pos: np.ndarray      # (2,) float32, normalized
    head_scale: float         # inter-ocular distance in normalized units
    head_conf: float
    face_params: np.ndarray   # (52,) float32 blendshapes, zeros when no face
    face_box: np.ndarray      # (4,) float32 [x, y, w, h] normalized
    silhouette: np.ndarray    # (H, W) uint8, 0/255
    camera: np.ndarray        # (2, 3) float32 affine (pixel coords), prev -> this frame
    camera_conf: float
    ok: bool

    @staticmethod
    def zeros(frame_index, dt, height, width):
        ident = np.eye(2, 3, dtype=np.float32)
        return Structure(
            frame_index=int(frame_index),
            dt=float(dt),
            body=np.zeros((NUM_BODY_POINTS, 3), dtype=np.float32),
            head_pose=np.zeros(3, dtype=np.float32),
            head_pos=np.zeros(2, dtype=np.float32),
            head_scale=0.0,
            head_conf=0.0,
            face_params=np.zeros(52, dtype=np.float32),
            face_box=np.zeros(4, dtype=np.float32),
            silhouette=np.zeros((height, width), dtype=np.uint8),
            camera=ident,
            camera_conf=0.0,
            ok=False,
        )


def torso_length(body, min_conf=0.3):
    """Torso length in normalized units from shoulder/hip midpoints."""
    shoulders = body[[5, 6], :2]
    hips = body[[11, 12], :2]
    conf_ok = body[[5, 6, 11, 12], 2] >= min_conf
    if conf_ok.all():
        return float(np.linalg.norm(shoulders.mean(axis=0) - hips.mean(axis=0)))
    return 0.0


def camera_corrected_prev_points(a, b, min_conf=0.3):
    """Previous-frame body points transformed into the current frame by
    b.camera (pixel-space affine, p_cur = A p_prev), removing camera motion.

    Returns (prev_points (N, 2) normalized, valid mask). Falls back to raw
    previous points when the silhouette carries no image size.
    """
    conf = (a.body[:, 2] >= min_conf) & (b.body[:, 2] >= min_conf)
    h, w = b.silhouette.shape[:2]
    if h <= 0 or w <= 0:
        return a.body[:, :2].copy(), conf
    scale = np.array([w, h], dtype=np.float64)
    A = b.camera.astype(np.float64)
    pa = (a.body[:, :2].astype(np.float64) * scale) @ A[:2, :2].T + A[:2, 2]
    return (pa / scale).astype(np.float32), conf


def step_speed(a, b, min_conf=0.3, camera_correct=True):
    """Mean per-point SUBJECT displacement (normalized units) from a to b,
    valid points only. With camera_correct=True the previous frame's points
    are transformed by b.camera first, so background motion is excluded
    (spec 6.2: bins classify subject motion, camera removed)."""
    if not (a.ok and b.ok):
        return 0.0
    conf = (a.body[:, 2] >= min_conf) & (b.body[:, 2] >= min_conf)
    if conf.sum() < 4:
        return 0.0
    if camera_correct:
        pa, _ = camera_corrected_prev_points(a, b, min_conf)
    else:
        pa = a.body[:, :2]
    return float(np.mean(np.linalg.norm(b.body[conf, :2] - pa[conf], axis=1)))


def mean_speeds(structs, min_conf=0.3, camera_correct=True):
    """Per-step subject speed in normalized units / second (camera removed
    by default; see step_speed)."""
    speeds = np.zeros(max(len(structs) - 1, 1), dtype=np.float32)
    for i in range(len(structs) - 1):
        a, b = structs[i], structs[i + 1]
        dt = b.dt if b.dt > 0 else a.dt
        dt = dt if dt > 0 else 1.0 / 24.0
        speeds[i] = step_speed(a, b, min_conf, camera_correct=camera_correct) / dt
    return speeds


def step_displacement(a, b, min_conf=0.3, camera_correct=True):
    """Per-point SUBJECT displacement vectors from a to b over valid points
    ((N_valid, 2), normalized units). With camera_correct=True the previous
    frame's points are transformed by b.camera first, excluding background
    motion (spec 6.2). None when either frame is missing or <4 valid points."""
    if not (a.ok and b.ok):
        return None
    valid = (a.body[:, 2] >= min_conf) & (b.body[:, 2] >= min_conf)
    if valid.sum() < 4:
        return None
    if camera_correct:
        pa, _ = camera_corrected_prev_points(a, b, min_conf)
    else:
        pa = a.body[:, :2]
    return b.body[valid, :2] - pa[valid]


def flip_structure(s):
    """Horizontal flip: mirrors coordinates and swaps left/right keypoints.

    Clip-level direction labels are flipped with flip_direction (spec 6.1).
    """
    body = s.body.copy()
    body[:, 0] = 1.0 - body[:, 0]
    for li, ri in LEFT_RIGHT_PAIRS:
        body[[li, ri]] = body[[ri, li]]

    head_pose = s.head_pose.copy()
    head_pose[0] = -head_pose[0]
    head_pose[2] = -head_pose[2]

    head_pos = s.head_pos.copy()
    head_pos[0] = 1.0 - head_pos[0]

    x, y, w, h = (float(v) for v in s.face_box)
    face_box = np.array([1.0 - w - x, y, w, h], dtype=np.float32)

    # The camera affine is pixel-space (camera.py): the horizontal mirror of
    # x_px is W - x_px, so the flip translation is the image width in pixels.
    F = np.array([[-1.0, 0.0], [0.0, 1.0]])
    t = np.array([float(s.silhouette.shape[1]), 0.0])
    A = s.camera.astype(np.float64)
    A2, c = A[:2, :2], A[:2, 2]
    A2p = F @ A2 @ F
    cp = F @ (A2 @ t + c) + t
    camera = np.zeros((2, 3), dtype=np.float64)
    camera[:2, :2] = A2p
    camera[:2, 2] = cp
    camera = camera.astype(np.float32)

    return dataclasses.replace(
        s,
        body=body,
        head_pose=head_pose.astype(np.float32),
        head_pos=head_pos.astype(np.float32),
        face_box=face_box,
        silhouette=np.ascontiguousarray(s.silhouette[:, ::-1]),
        camera=camera,
    )


def flip_direction(direction):
    """Flip a clip-level direction label under horizontal mirror (spec 6.1)."""
    if direction in ("left", "right"):
        return "right" if direction == "left" else "left"
    return direction
