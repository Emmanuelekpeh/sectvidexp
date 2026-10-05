"""Test 2: extractor validation (spec 10.2). Jitter, dropout rate, and
camera/subject separation on clips with known motion.

Camera separation is checked against a synthetic clip with a known camera
affine (the estimable part) plus camera confidence on real data; the
subject-residual part of the check requires real moving subjects and is
recorded, not asserted, until the data supports a threshold.
"""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from sectvid.config import load_config, root_path
from sectvid.data import clips as data_clips
from sectvid.structure import camera
from sectvid.structure.extractor import StructureExtractor

CFG = load_config()
V = CFG["structure"]["validation"]
CLIP_IDS = data_clips.discover_clips(CFG)


def _pick_clips(n):
    props = {cid: data_clips.clip_properties(CFG, cid) for cid in CLIP_IDS}
    pool = [c for c, p in props.items() if p["width"] >= 480 and p["n_frames"] >= V["frames_per_clip"]]
    if not pool:
        pool = [c for c, p in props.items() if p["n_frames"] >= V["frames_per_clip"]]
    rng = np.random.default_rng(CFG["seed"])
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[int(i)] for i in idx]


HAS_DATA = len(CLIP_IDS) > 0


@unittest.skipUnless(HAS_DATA, "no clips in data/")
class TestExtractorValidation(unittest.TestCase):
    results = {}

    @classmethod
    def setUpClass(cls):
        cls.extractor = StructureExtractor(CFG)
        cls.v = V
        rng = np.random.default_rng(CFG["seed"])
        cls.clips = {}
        for cid in _pick_clips(cls.v["clips"]):
            frames = [f for f in data_clips.load_frames(CFG, cid, 0, cls.v["frames_per_clip"]) if f.mean() > 1.0]
            if len(frames) < 4:
                continue
            cls.clips[cid] = frames
            props = data_clips.clip_properties(CFG, cid)
            cls.results[cid] = {"width": props["width"], "fps": props["fps"]}
        if not cls.clips:
            raise RuntimeError("no usable frames from sampled clips")

    @classmethod
    def tearDownClass(cls):
        from sectvid.eval.report import write_json_report
        if HAS_DATA:
            write_json_report(root_path(CFG, CFG["report"]["dir"]), "test2_extractor_validation",
                              cls.results, CFG, CFG["seed"])

    @staticmethod
    def _shift(frame, k):
        import cv2
        out = cv2.copyMakeBorder(frame, 0, 0, 0, k, cv2.BORDER_REPLICATE)
        return out[:, k:]

    def test_jitter(self):
        """Under a pixel jitter, the sequence tracker should follow the same
        subject. Stability is asserted as p90 of the per-frame mean keypoint
        shift (the extractor's noise floor as structure ground truth); per-point
        confidence carries the rest (spec 6.2). Per-clip p90 is recorded so the
        gate report can flag low-reliability clips."""
        limit = self.v["jitter_limit_px"]
        shifts = []
        per_clip = {}
        for cid, frames in self.clips.items():
            width = frames[0].shape[1]
            clip_shifts = []
            for i in range(0, len(frames), 4):
                k = int(self.v["jitter_px"])
                seq = self.extractor.extract_sequence([frames[i], self._shift(frames[i], k)], smooth=False)
                base, moved = seq[0], seq[1]
                if not base.ok or not moved.ok:
                    continue
                valid = (base.body[:, 2] >= 0.5) & (moved.body[:, 2] >= 0.5)
                if valid.sum() < 8:
                    continue
                d_px = np.linalg.norm(moved.body[valid, :2] - base.body[valid, :2], axis=1) * width
                clip_shifts.append(float(d_px.mean()))
            if clip_shifts:
                per_clip[cid] = {"p90_px": float(np.percentile(clip_shifts, 90)), "n": len(clip_shifts)}
                shifts.extend(clip_shifts)
        self.assertGreaterEqual(len(shifts), 8, f"too few measured frames: {len(shifts)}")
        arr = np.array(shifts)
        p90 = float(np.percentile(arr, 90))
        self.results["jitter_max_mean_shift_px"] = float(arr.max())
        self.results["jitter_p90_mean_shift_px"] = p90
        self.results["jitter_n_frames"] = len(shifts)
        self.results["jitter_per_clip"] = per_clip
        self.assertLessEqual(p90, limit, f"p90 jitter shift {p90:.2f}px > {limit}px")

    def test_dropout_rate(self):
        total = 0
        missing = 0
        for cid, frames in self.clips.items():
            for s in self.extractor.extract_sequence(frames, smooth=False):
                total += 1
                if not s.ok:
                    missing += 1
                    self.results.setdefault("dropout_examples", []).append({"clip": cid, "frame": int(s.frame_index)})
        rate = missing / max(total, 1)
        self.results["dropout_rate"] = rate
        self.results["n_frames_checked"] = total
        self.assertLessEqual(rate, self.v["max_dropout"], f"dropout {rate:.2f} > {self.v['max_dropout']}")

    def test_camera_known_motion_synthetic(self):
        """Camera affine estimation on a synthetic clip with a known 4px/frame translation."""
        rng = np.random.default_rng(CFG["seed"])
        W, H, n, step = 640, 480, 10, 4
        small = rng.integers(0, 255, (48, 60), dtype=np.uint8)
        bg = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        bg = (bg.astype(np.float32) + rng.normal(0, 8, bg.shape)).clip(0, 255).astype(np.uint8)
        ext = np.repeat(bg, 3, axis=1)
        bg_mask = np.full((H, W), 255, dtype=np.uint8)
        errs = []
        for t in range(n - 1):
            prev = ext[:, step * t:step * t + W]
            cur = ext[:, step * (t + 1):step * (t + 1) + W]
            cam_cfg = CFG["structure"]["camera"]
            A, conf = camera.estimate_background_affine(
                prev, bg_mask, cur, bg_mask,
                max_features=cam_cfg["max_features"],
                ransac_thresh_px=cam_cfg["ransac_thresh_px"],
                min_inliers=cam_cfg["min_inliers"],
            )
            err = float(np.linalg.norm(A[:2, 2] - np.array([step, 0.0])))
            errs.append(err)
        errs = np.array(errs)
        self.results["camera_translation_err_px"] = float(errs.mean())
        self.results["camera_translation_err_norm"] = float(errs.mean()) / W
        self.assertLess(float(errs.mean()) / W, self.v["max_camera_translation_err"])

    def test_camera_confidence_on_real_data(self):
        """Camera confidence on real data; the subject-residual magnitude is
        recorded, not asserted (no threshold supported by data yet). The
        camera affine is pixel-space (camera.py), so predictions run in px."""
        cid = next(iter(self.clips))
        frames = self.clips[cid][: self.v["frames_per_clip"] // 2]
        h, w = frames[0].shape[:2]
        scale = np.array([w, h], dtype=np.float32)
        seq = self.extractor.extract_sequence(frames)
        confs = [s.camera_conf for s in seq[1:]]
        self.results["camera_conf_mean_real"] = float(np.mean(confs))
        self.assertTrue(all(np.isfinite(s.camera).all() for s in seq))
        resid_px = []
        for s in seq[1:]:
            if not s.ok:
                continue
            if s.frame_index > 0:
                prev = seq[s.frame_index - 1]
                if prev.ok:
                    prev_px = prev.body[:, :2] * scale
                    cur_px = s.body[:, :2] * scale
                    pred = prev_px @ s.camera[:2, :2].T + s.camera[:2, 2]
                    resid_px.append(float(np.linalg.norm(cur_px - pred)))
        if resid_px:
            self.results["camera_subject_residual_px_p50"] = float(np.percentile(resid_px, 50))
        self.assertGreater(float(np.mean(confs)), 0.0, "camera affine never estimated on real data")


if __name__ == "__main__":
    unittest.main()
