"""Test 6: windowing and motion bins (spec 6.1, Stage 1).

Bins must classify SUBJECT motion (camera removed) and must not label
extractor failure as stillness: windows with too few observed steps get the
"unobserved" label and are excluded from motion sampling.
"""
import unittest

import numpy as np

from sectvid.config import load_config
from sectvid.data import windows as W
from sectvid.structure.schema import Structure

CFG = load_config()
WLEN = CFG["window"]["length"]
H, Wpx = 48, 64
FPS = 24.0


def _seq(speed_px=0.0, camera_px=0.0, ok_mask=None):
    """Subject moves `speed_px` px/frame relative to the background; the
    background (camera) drifts `camera_px` px/frame, so image-space position
    is camera + subject: a world-still subject (speed 0) moves WITH the
    background."""
    structs = []
    for i in range(WLEN):
        s = Structure.zeros(i, 1.0 / FPS, H, Wpx)
        s.ok = True
        s.body[:, 2] = 1.0
        s.body[:, 0] = 0.5 + ((camera_px + speed_px) * i) / Wpx
        s.body[:, 1] = 0.5
        cam = np.eye(2, 3, dtype=np.float32)
        cam[:2, 2] = [camera_px, 0.0]
        s.camera = cam
        s.camera_conf = 0.9
        if ok_mask is not None and not ok_mask[i]:
            s.ok = False
            s.body[:, 2] = 0.0
        structs.append(s)
    return structs


def _bin(name, seq):
    ws = W.build_windows(CFG, "synth", seq)
    assert len(ws) == 1, ws
    return ws[0]


def _norm_s(px_per_frame):
    return px_per_frame / Wpx * FPS


class TestWindowBins(unittest.TestCase):
    def test_still(self):
        w = _bin("still", _seq(speed_px=0.0))
        self.assertEqual(w["bin"], "still")
        self.assertAlmostEqual(w["observed_frac"], 1.0)
        self.assertEqual(w["subject_speed"], 0.0)

    def test_slow(self):
        w = _bin("slow", _seq(speed_px=0.15))
        self.assertEqual(w["bin"], "slow")
        self.assertAlmostEqual(w["subject_speed"], _norm_s(0.15), places=5)

    def test_fast(self):
        w = _bin("fast", _seq(speed_px=0.5))
        self.assertEqual(w["bin"], "fast")
        self.assertGreaterEqual(w["subject_speed"], CFG["motion_bins"]["fast_min_speed"])

    def test_camera_motion_is_not_subject_motion(self):
        # static subject, 2px/frame camera drift: raw displacement would be
        # fast-bin speed, camera-corrected it must be still
        w = _bin("camera", _seq(speed_px=0.0, camera_px=2.0))
        self.assertEqual(w["bin"], "still")
        self.assertLess(w["subject_speed"], CFG["motion_bins"]["still_max_speed"])

    def test_missing_poses_are_unobserved_not_still(self):
        ok_mask = [i % 2 == 0 for i in range(WLEN)]  # alternating ok -> no observed steps
        w = _bin("unobserved", _seq(speed_px=0.0, ok_mask=ok_mask))
        self.assertEqual(w["bin"], "unobserved")
        self.assertEqual(w["observed_frac"], 0.0)

    def test_sample_never_picks_unobserved(self):
        ws = []
        for i in range(4):
            ws.append({"bin": "unobserved", "clip_id": f"c{i}"})
        for i in range(4):
            ws.append({"bin": "still", "clip_id": f"s{i}"})
        for i in range(4):
            ws.append({"bin": "slow", "clip_id": f"m{i}"})
        for i in range(4):
            ws.append({"bin": "fast", "clip_id": f"f{i}"})
        rng_quota = CFG["motion_bins"]["sample_quotas"]
        chosen = W.sample_windows(ws, rng_quota, n=6, seed=0)
        self.assertEqual(len(chosen), 6)
        self.assertTrue(all(w["bin"] != "unobserved" for w in chosen))


if __name__ == "__main__":
    unittest.main()
