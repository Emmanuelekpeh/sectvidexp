"""Test 3: normalization/flip round trip for structure (spec 10.3)."""
import unittest

import numpy as np

from sectvid.structure.schema import (Structure, flip_structure, flip_direction,
                                      torso_length, NUM_BODY_POINTS, LEFT_RIGHT_PAIRS)


def _synthetic_structure(seed=0):
    rng = np.random.default_rng(seed)
    h, w = 48, 64
    body = np.zeros((NUM_BODY_POINTS, 3), dtype=np.float32)
    body[:, 0] = rng.uniform(0.2, 0.8, NUM_BODY_POINTS).astype(np.float32)
    body[:, 1] = rng.uniform(0.1, 0.9, NUM_BODY_POINTS).astype(np.float32)
    body[:, 2] = 1.0
    silhouette = (rng.random((h, w)) > 0.5).astype(np.uint8) * 255
    # Camera affine is pixel-space (camera.py): translation in px.
    A = np.array([[1.02, 0.05, 2.0], [0.02, 0.98, 3.0]], dtype=np.float32)
    return Structure(
        frame_index=7,
        dt=1.0 / 24.0,
        body=body,
        head_pose=np.array([0.3, -0.1, 0.05], dtype=np.float32),
        head_pos=np.array([0.4, 0.3], dtype=np.float32),
        head_scale=0.08,
        head_conf=0.9,
        face_params=rng.random(52, dtype=np.float32),
        face_box=np.array([0.35, 0.2, 0.1, 0.15], dtype=np.float32),
        silhouette=silhouette,
        camera=A,
        camera_conf=0.7,
        ok=True,
    )


class TestFlipRoundTrip(unittest.TestCase):
    def test_coordinate_mirror(self):
        s = _synthetic_structure()
        f = flip_structure(s)
        # nose (index 0) has no pair: x mirrors directly
        np.testing.assert_allclose(f.body[0, 0], 1.0 - s.body[0, 0], atol=1e-6)
        np.testing.assert_allclose(f.body[0, 1], s.body[0, 1], atol=1e-6)
        # y coordinates are preserved (as a multiset), x are mirrored
        np.testing.assert_allclose(np.sort(f.body[:, 1]), np.sort(s.body[:, 1]), atol=1e-6)
        np.testing.assert_allclose(np.sort(f.body[:, 0]), np.sort(1.0 - s.body[:, 0]), atol=1e-6)
        np.testing.assert_allclose(f.head_pos, [1.0 - s.head_pos[0], s.head_pos[1]], atol=1e-6)
        x, y, w, h = (float(v) for v in s.face_box)
        np.testing.assert_allclose(f.face_box, [1.0 - w - x, y, w, h], atol=1e-6)

    def test_left_right_swap(self):
        s = _synthetic_structure()
        f = flip_structure(s)
        for li, ri in LEFT_RIGHT_PAIRS:
            expected_li = np.array([1.0 - s.body[ri, 0], s.body[ri, 1]], dtype=np.float64)
            expected_ri = np.array([1.0 - s.body[li, 0], s.body[li, 1]], dtype=np.float64)
            np.testing.assert_allclose(f.body[li, :2], expected_li, atol=1e-6)
            np.testing.assert_allclose(f.body[ri, :2], expected_ri, atol=1e-6)

    def test_head_pose_mirror(self):
        s = _synthetic_structure()
        f = flip_structure(s)
        np.testing.assert_allclose(f.head_pose[0], -s.head_pose[0], atol=1e-6)
        np.testing.assert_allclose(f.head_pose[1], s.head_pose[1], atol=1e-6)
        np.testing.assert_allclose(f.head_pose[2], -s.head_pose[2], atol=1e-6)

    def test_silhouette_mirror(self):
        s = _synthetic_structure()
        f = flip_structure(s)
        np.testing.assert_array_equal(f.silhouette, s.silhouette[:, ::-1])

    def test_camera_affine_mirror(self):
        s = _synthetic_structure()
        f = flip_structure(s)
        w = s.silhouette.shape[1]
        F = np.array([[-1.0, 0.0], [0.0, 1.0]])
        t = np.array([float(w), 0.0])
        A2, c = s.camera[:2, :2], s.camera[:2, 2]
        expected = np.zeros((2, 3))
        expected[:2, :2] = F @ A2 @ F
        expected[:2, 2] = F @ (A2 @ t + c) + t
        np.testing.assert_allclose(f.camera, expected, atol=1e-4)

    def test_camera_translation_only(self):
        ident_pos = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 4.0]], dtype=np.float32)
        s = _synthetic_structure()
        s = Structure(**{**s.__dict__, "camera": ident_pos})
        f = flip_structure(s)
        # a rightward camera translation (px) becomes leftward in the mirror
        np.testing.assert_allclose(f.camera[:2, 2], [-5.0, 4.0], atol=1e-5)

    def test_camera_mirrors_background_point(self):
        """A background point mapped prev->cur by the original camera must land
        at the mirrored position under the flipped camera, at a non-square size."""
        s = _synthetic_structure()
        f = flip_structure(s)
        w, h = s.silhouette.shape[1], s.silhouette.shape[0]
        p_prev = np.array([12.0, 20.0])
        p_cur = s.camera @ np.append(p_prev, 1.0)
        p_prev_m = np.array([w - p_prev[0], p_prev[1]])
        p_cur_m = f.camera @ np.append(p_prev_m, 1.0)
        np.testing.assert_allclose(p_cur_m[:2], [w - p_cur[0], p_cur[1]], atol=1e-3)

    def test_double_flip_is_identity(self):
        s = _synthetic_structure()
        g = flip_structure(flip_structure(s))
        np.testing.assert_allclose(g.body, s.body, atol=1e-5)
        np.testing.assert_allclose(g.head_pose, s.head_pose, atol=1e-5)
        np.testing.assert_allclose(g.head_pos, s.head_pos, atol=1e-5)
        np.testing.assert_allclose(g.face_box, s.face_box, atol=1e-6)
        np.testing.assert_allclose(g.camera, s.camera, atol=1e-5)
        np.testing.assert_array_equal(g.silhouette, s.silhouette)

    def test_direction_labels(self):
        self.assertEqual(flip_direction("left"), "right")
        self.assertEqual(flip_direction("right"), "left")
        self.assertEqual(flip_direction("toward"), "toward")
        self.assertEqual(flip_direction("away"), "away")
        self.assertEqual(flip_direction("none"), "none")

    def test_torso_length_sane(self):
        s = _synthetic_structure()
        self.assertGreater(torso_length(s.body), 0.0)
        self.assertLess(torso_length(s.body), 1.0)


if __name__ == "__main__":
    unittest.main()
