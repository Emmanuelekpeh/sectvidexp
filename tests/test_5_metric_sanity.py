"""Test 5: metric sanity (spec 10.5). A static video must be flagged frozen; a
cross-fade of two frames must score high on ghosting; a perturbed real video
must score near real on identity."""
import unittest

import numpy as np

from sectvid.eval import metrics
from sectvid.eval.baselines import copy_last_frame, hold_last_structure
from sectvid.structure.schema import Structure


def _frames(n, h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (h, w), dtype=np.uint8) for _ in range(n)]


def _still_pair(n, h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (n, h, w), dtype=np.uint8)
    base[0] = base[0]  # frame 0 is the reference
    return [base[0] for _ in range(n)]


def _moving_square(n=16, h=48, w=64, speed=2):
    frames = []
    for i in range(n):
        f = np.zeros((h, w), dtype=np.uint8)
        x0 = int(4 + i * speed)
        f[10:30, x0:x0 + 20] = 200
        f[35:45, 10:30] = 90
        frames.append(f)
    return frames


class TestMetricSanity(unittest.TestCase):
    def test_static_video_flagged_frozen(self):
        frames = _still_pair(32)
        st = [Structure.zeros(i, 1.0 / 24.0, 48, 64) for i in range(32)]
        for s in st:
            s.ok = True
            s.body = np.array([[0.5, 0.5, 1.0]] * 17, dtype=np.float32)
        self.assertTrue(metrics.is_frozen(st)["frozen"])
        self.assertAlmostEqual(metrics.is_frozen(st)["fraction_still"], 1.0)

    def test_crossfade_scores_high_on_ghosting(self):
        moving = _moving_square(16)
        a, b = moving[2], moving[14]
        gen = [(a.astype(np.float32) * (1 - i / 15.0) + b.astype(np.float32) * (i / 15.0)).astype(np.uint8)
               for i in range(16)]
        score = metrics.ghosting_score(gen, moving, ks=(1, 2, 4), ratio=0.9)
        self.assertGreater(score["flag_rate"], 0.5)
        control = metrics.ghosting_score(moving, moving, ks=(1, 2, 4), ratio=0.9)
        self.assertLess(control["flag_rate"], 0.1)

    def test_perturbed_real_near_real_on_identity(self):
        rng = np.random.default_rng(1)
        real = _frames(4, seed=3)
        perturbed = [np.clip(f.astype(np.int32) + rng.integers(-3, 4, f.shape), 0, 255).astype(np.uint8)
                     for f in real]
        encoder = metrics.IdentityEncoder(kind="probe")
        emb = encoder.embed(perturbed + real)
        for i in range(len(real)):
            sim = float(np.dot(emb[i], emb[len(real) + i]))
            self.assertGreater(sim, 0.99, f"frame {i}: {sim}")

    def test_sharpness_and_flicker_identity(self):
        frames = _frames(16, seed=5)
        self.assertAlmostEqual(metrics.sharpness_ratio(frames, frames), 1.0, places=5)
        self.assertAlmostEqual(metrics.flicker_ratio(frames, frames, window=3), 1.0, places=5)

    def test_angular_error_90_degrees(self):
        st = []
        for i in range(4):
            s = Structure.zeros(i, 1.0 / 24.0, 48, 64)
            s.ok = True
            s.body = np.zeros((17, 3), dtype=np.float32)
            s.body[:, 2] = 1.0
            s.body[0, :2] = [0.5, 0.5]
            for j in range(1, 17):
                s.body[j, :2] = [0.5 + 0.02 * np.cos(j * 0.7), 0.5 + 0.02 * np.sin(j * 0.7)]
            s.body[:, 0] += 0.01 * i
            st.append(s)
        pred = []
        for s in st:
            p = Structure(**{**s.__dict__})
            body = p.body.copy()
            for j in range(1, 17):
                x = body[j, 0] - 0.5
                y = body[j, 1] - 0.5
                body[j, 0] = 0.5 - y
                body[j, 1] = 0.5 + x
            p = Structure(**{**p.__dict__, "body": body})
            pred.append(p)
        res = metrics.angular_error(pred, st)
        self.assertIsNotNone(res["angular_err_deg"])
        self.assertGreater(res["angular_err_deg"], 60.0)

    def test_motion_magnitude_ratio(self):
        true_st = []
        for i in range(4):
            s = Structure.zeros(i, 1.0 / 24.0, 48, 64)
            s.ok = True
            s.body = np.zeros((17, 3), dtype=np.float32)
            s.body[:, 2] = 1.0
            s.body[0, :2] = [0.4 + 0.05 * i, 0.5]
            for j in range(1, 17):
                s.body[j, :2] = [0.4 + 0.05 * i + 0.01 * j % 3, 0.5 + j / 30.0]
            true_st.append(s)
        pred_st = [Structure(**{**s.__dict__, "body": s.body.copy()}) for s in true_st]
        self.assertAlmostEqual(metrics.motion_magnitude_ratio(pred_st, true_st), 1.0, places=5)
        for s in pred_st:
            s.body[:, 0] = s.body[:, 0] * 0.5
        self.assertAlmostEqual(metrics.motion_magnitude_ratio(pred_st, true_st), 0.5, places=3)

    def test_frozen_and_hallucinated_rates_direction(self):
        def make(dx):
            st = []
            for i in range(4):
                s = Structure.zeros(i, 1.0 / 24.0, 48, 64)
                s.ok = True
                s.body = np.zeros((17, 3), dtype=np.float32)
                s.body[:, 2] = 1.0
                s.body[0, :2] = [0.3 + dx * i, 0.5]
                for j in range(1, 17):
                    s.body[j, :2] = [0.3 + dx * i, 0.4 + 0.01 * j]
                st.append(s)
            return st
        moving_true = make(0.02)
        frozen_pred = make(0.0)
        fr = metrics.frozen_rate(frozen_pred, moving_true)
        self.assertIsNotNone(fr)
        self.assertGreater(fr, 0.5)
        still_true = make(0.0)
        wild_pred = make(0.03)
        hm = metrics.hallucinated_motion_rate(wild_pred, still_true)
        self.assertIsNotNone(hm)
        self.assertGreater(hm, 0.5)

    def test_baseline_outputs_shape(self):
        frames = _frames(32)
        c = copy_last_frame(frames, 8)
        self.assertEqual(len(c), 24)
        self.assertTrue(all(np.array_equal(x, frames[7]) for x in c))
        st = [Structure.zeros(i, 1.0 / 24.0, 48, 64) for i in range(32)]
        for s in st:
            s.ok = True
            s.body = np.zeros((17, 3), dtype=np.float32)
            s.body[:, 2] = 1.0
            s.body[0, :2] = [0.3, 0.5]
            s.body[1, :2] = [0.4, 0.5]
        h = hold_last_structure(st, 8)
        self.assertEqual(len(h), 24)


if __name__ == "__main__":
    unittest.main()
