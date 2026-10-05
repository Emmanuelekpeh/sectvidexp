"""Structure extraction (spec 6.2): 17 keypoints, head pose, face parameters,
person silhouette, per-frame camera affine, and per-point confidence.

Extractor: MediaPipe Tasks (pose landmarker, face landmarker, image segmenter).
Extractor noise is tracked, not hidden: every point carries confidence and all
downstream losses must weight by it (spec 6.2, known risk).
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .camera import estimate_background_affine, background_mask
from .schema import Structure, NUM_BODY_POINTS

# MediaPipe pose landmark index -> COCO-17 index (subject-side, no mirror)
_POSE_TO_COCO = {
    0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
    11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10,
    23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
}

# 3D head template (x right, y down, z into scene; inter-ocular distance = 1.0)
# Face mesh index -> canonical point. Validated at Stage 0 (jitter/round trip);
# absolute pose quality is a Stage 2/3 concern.
_HEAD_TEMPLATE = {
    1: (0.0, 0.0, -0.15),
    152: (0.0, 0.5, 0.25),
    33: (0.5, 0.05, 0.15),
    133: (0.25, 0.08, 0.30),
    362: (-0.25, 0.08, 0.30),
    263: (-0.5, 0.05, 0.15),
    10: (0.0, -0.4, 0.25),
    454: (-0.45, 0.25, 0.30),
    234: (0.45, 0.25, 0.30),
    93: (0.0, 0.18, -0.05),
    178: (0.0, 0.3, 0.0),
}


def ensure_models(cfg):
    model_dir = Path(cfg["_root"]) / cfg["structure"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, url in (("pose", "pose_model_url"), ("face", "face_model_url"), ("seg", "seg_model_url")):
        url = cfg["structure"][url]
        p = model_dir / url.rsplit("/", 1)[-1]
        if not p.exists():
            urllib.request.urlretrieve(url, p)
        paths[key] = str(p)
    return paths


def _head_pose_from_face(face_landmarks_px, width, height):
    """PnP against a canonical face template. Returns (yaw, pitch, roll) rad, confidence."""
    from scipy.spatial.transform import Rotation
    model = np.array([_HEAD_TEMPLATE[i] for i in _HEAD_TEMPLATE], dtype=np.float64)
    idx = np.array(sorted(_HEAD_TEMPLATE), dtype=np.int32)
    pts = np.array([face_landmarks_px[i] for i in idx], dtype=np.float64)
    K = np.array([[float(height), 0, width / 2.0],
                  [0, float(height), height / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    result = cv2.solvePnP(
        model, pts, K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE
    )
    ok, rvec, tvec = result[0], result[1], result[2]
    if not ok:
        return np.zeros(3, dtype=np.float32), 0.0
    R = Rotation.from_rotvec(rvec.reshape(3)).as_matrix()
    e = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    pose = np.array([e[1], e[0], e[2]], dtype=np.float64)  # [yaw, pitch, roll]
    proj = cv2.projectPoints(model, R, tvec, K, np.zeros(4))[0]
    reproj_err = float(np.linalg.norm(proj - pts).mean())
    conf = float(np.clip(1.0 - reproj_err / (0.2 * height), 0.0, 1.0))
    return pose.astype(np.float32), conf


class StructureExtractor:
    def __init__(self, cfg, model_dir=None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        self._mp_tasks = mp_tasks
        self._mp_vision = mp_vision

        if model_dir is None:
            paths = ensure_models(cfg)
        else:
            s = cfg["structure"]
            paths = {"pose": str(Path(model_dir) / s["pose_model_url"].rsplit("/", 1)[-1]),
                     "face": str(Path(model_dir) / s["face_model_url"].rsplit("/", 1)[-1]),
                     "seg": str(Path(model_dir) / s["seg_model_url"].rsplit("/", 1)[-1])}
        self._model_paths = paths
        self.cfg = cfg
        self.pose = mp_vision.PoseLandmarker.create_from_options(
            self._pose_options(mp_vision.RunningMode.IMAGE))
        self.face = mp_vision.FaceLandmarker.create_from_options(
            self._face_options(mp_vision.RunningMode.IMAGE))
        base = mp_tasks.BaseOptions
        self.seg = mp_vision.ImageSegmenter.create_from_options(
            mp_vision.ImageSegmenterOptions(
                base_options=base(model_asset_path=paths["seg"]),
            ))

    def _pose_options(self, running_mode):
        base = self._mp_tasks.BaseOptions
        s = self.cfg["structure"]
        return self._mp_vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=self._model_paths["pose"]),
            running_mode=running_mode,
            num_poses=int(s.get("max_poses", 3)),
            min_pose_detection_confidence=s["min_detection_conf"],
            min_tracking_confidence=s["min_detection_conf"],
        )

    def _face_options(self, running_mode):
        base = self._mp_tasks.BaseOptions
        s = self.cfg["structure"]
        return self._mp_vision.FaceLandmarkerOptions(
            base_options=base(model_asset_path=self._model_paths["face"]),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=s["min_detection_conf"],
            output_face_blendshapes=True,
        )

    def _mp_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

    def _segment(self, bgr, img):
        res = self.seg.segment(img)
        mask = np.asarray(res.confidence_masks[0].numpy_view())
        h, w = bgr.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        thr = int(self.cfg["structure"].get("mask_threshold", 127))
        return ((mask > thr).astype(np.uint8)) * 255

    def _pose_body(self, pose_landmarks, prev_centroid=None):
        if not pose_landmarks:
            return None, None
        lms = self._select_subject(pose_landmarks, prev_centroid)
        body = np.zeros((NUM_BODY_POINTS, 3), dtype=np.float32)
        for mp_idx, coco_idx in _POSE_TO_COCO.items():
            lm = lms[mp_idx]
            body[coco_idx, 0] = lm.x
            body[coco_idx, 1] = lm.y
            body[coco_idx, 2] = float(lm.visibility or 0.0)
        vis = body[:, 2] > 0.3
        if vis.sum() < 4:
            return None, None
        return body, body[vis, :2].mean(axis=0)

    @staticmethod
    def _select_subject(pose_landmarks, prev_centroid=None):
        """Pick the primary subject: largest visible landmark area, with temporal
        continuity when a previous centroid is known (prevents subject jumps
        under fast motion / jitter in multi-person frames).
        """
        def score(lms):
            pts = np.array([[lm.x, lm.y] for lm in lms if (lm.visibility or 0.0) > 0.3])
            if len(pts) < 8:
                return -1.0
            area = float((pts.max(axis=0) - pts.min(axis=0)).prod())
            conf = float(np.mean([(1.0 if (lm.visibility or 0.0) > 0.3 else 0.0) for lm in lms]))
            centroid = pts.mean(axis=0)
            if prev_centroid is not None:
                area *= np.exp(-np.linalg.norm(centroid - prev_centroid) / 0.25)
            return area * conf
        return max(pose_landmarks, key=score)

    def _face(self, res, bgr):
        if not res.face_landmarks:
            return (np.zeros(3, np.float32), np.zeros(2, np.float32), 0.0, 0.0,
                    np.zeros(52, np.float32), np.zeros(4, np.float32))
        lms = res.face_landmarks[0]
        h, w = bgr.shape[:2]
        px = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float64)
        pose, conf = _head_pose_from_face(px, w, h)
        model_idx = sorted(_HEAD_TEMPLATE)
        head_pos = px[model_idx].mean(axis=0) / np.array([w, h])
        interocular_px = float(np.linalg.norm(px[33] - px[263]))
        head_scale = interocular_px / w
        xs_all = np.array([lm.x for lm in lms])
        ys_all = np.array([lm.y for lm in lms])
        face_box = np.array([float(xs_all.min()), float(ys_all.min()),
                             float(xs_all.max() - xs_all.min()),
                             float(ys_all.max() - ys_all.min())], dtype=np.float32)
        blend = np.zeros(self.cfg["structure"]["face_blendshape_count"], dtype=np.float32)
        for i, c in enumerate(res.face_blendshapes[0][:]):
            val = getattr(c, "score", None)
            if val is None:
                val = getattr(c, "weight", 0.0)
            blend[i] = float(val)
        return pose, head_pos.astype(np.float32), float(head_scale), float(conf), blend, face_box

    def extract_frame(self, bgr, frame_index=0, dt=1.0 / 24.0):
        img = self._mp_image(bgr)
        silhouette = self._segment(bgr, img)
        body, _ = self._pose_body(self.pose.detect(img).pose_landmarks)
        pose, head_pos, head_scale, head_conf, blend, face_box = self._face(self.face.detect(img), bgr)
        ok = body is not None
        if body is None:
            body = np.zeros((NUM_BODY_POINTS, 3), dtype=np.float32)
        ident = np.eye(2, 3, dtype=np.float32)
        return Structure(
            frame_index=int(frame_index),
            dt=float(dt),
            body=body,
            head_pose=pose,
            head_pos=head_pos,
            head_scale=float(head_scale),
            head_conf=head_conf,
            face_params=blend,
            face_box=face_box,
            silhouette=silhouette,
            camera=ident,
            camera_conf=0.0,
            ok=bool(ok),
        )

    def extract_sequence(self, bgr_frames, fps=24.0, smooth=True):
        """Per-clip extraction using VIDEO-mode trackers: the pose and face
        landmarkers hold tracking state so the same subject stays locked
        across frames (one-shot detection jumps subjects under fast motion)."""
        dt = 1.0 / float(fps) if fps > 0 else 1.0 / 24.0
        vision = self._mp_vision
        pose_video = vision.PoseLandmarker.create_from_options(
            self._pose_options(vision.RunningMode.VIDEO))
        face_video = vision.FaceLandmarker.create_from_options(
            self._face_options(vision.RunningMode.VIDEO))
        out = []
        prev_gray = None
        prev_bg = None
        prev_centroid = None
        alpha_new = 1.0 - float(self.cfg["structure"]["smoothing"])
        smoothed = None
        try:
            for i, bgr in enumerate(bgr_frames):
                img = self._mp_image(bgr)
                ts_ms = int(round(i * 1000.0 * dt))
                pose_res = pose_video.detect_for_video(img, ts_ms)
                face_res = face_video.detect_for_video(img, ts_ms)
                silhouette = self._segment(bgr, img)
                body, centroid = self._pose_body(pose_res.pose_landmarks, prev_centroid)
                if centroid is not None:
                    prev_centroid = centroid
                pose, head_pos, head_scale, head_conf, blend, face_box = self._face(face_res, bgr)
                ok = body is not None
                if body is None:
                    body = np.zeros((NUM_BODY_POINTS, 3), dtype=np.float32)
                s = Structure(
                    frame_index=int(i), dt=dt, body=body,
                    head_pose=pose, head_pos=head_pos,
                    head_scale=float(head_scale), head_conf=head_conf,
                    face_params=blend, face_box=face_box,
                    silhouette=silhouette,
                    camera=np.eye(2, 3, dtype=np.float32),
                    camera_conf=0.0, ok=bool(ok),
                )
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if i > 0 and prev_gray is not None:
                    bg_now = background_mask(s.silhouette, self.cfg["structure"]["camera"]["person_margin_px"])
                    A, conf = estimate_background_affine(
                        prev_gray, prev_bg, gray, bg_now,
                        max_features=self.cfg["structure"]["camera"]["max_features"],
                        ransac_thresh_px=self.cfg["structure"]["camera"]["ransac_thresh_px"],
                        min_inliers=self.cfg["structure"]["camera"]["min_inliers"],
                    )
                    s = Structure(**{**s.__dict__, "camera": A, "camera_conf": conf})
                if smooth and s.ok:
                    if smoothed is not None:
                        conf_ok = s.body[:, 2] >= self.cfg["structure"]["min_point_conf"]
                        conf_ok &= smoothed[:, 2] > 0
                        cur = s.body.copy()
                        cur[conf_ok, :2] = (alpha_new * s.body[conf_ok, :2]
                                            + (1.0 - alpha_new) * smoothed[conf_ok, :2])
                    else:
                        cur = s.body.copy()
                    smoothed = cur
                    s = Structure(**{**s.__dict__, "body": cur})
                prev_gray, prev_bg = gray, background_mask(
                    s.silhouette, self.cfg["structure"]["camera"]["person_margin_px"]
                )
                out.append(s)
        finally:
            pose_video.close()
            face_video.close()
        return out
