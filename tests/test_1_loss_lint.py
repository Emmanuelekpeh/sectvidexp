"""Test 1: loss lint (spec 10.1, 8). Fails the build if an unregistered loss, or
a full-frame pixel-regression loss, touches model output.
"""
import tempfile
import unittest
from pathlib import Path

from sectvid import loss_registry


class TestLossLint(unittest.TestCase):
    def test_registry_matches_spec(self):
        expected = {"structure", "timing", "velocity", "magnitude",
                    "generative", "perceptual", "identity", "structure_obedience"}
        self.assertEqual(set(loss_registry.get_registry()), expected)

    def test_current_tree_is_clean(self):
        violations = loss_registry.lint(Path(__file__).resolve().parent.parent)
        self.assertEqual(violations, [], "\n".join(violations))

    def test_seeded_violation_is_caught(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bad = root / "sectvid" / "bad_module.py"
            bad.parent.mkdir(parents=True)
            bad.write_text("import torch.nn as nn\nloss = nn.MSELoss()\n", encoding="utf-8")
            self.assertTrue(loss_registry.lint(root))

    def test_eval_package_is_allowed_to_use_pixel_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ok = root / "sectvid" / "eval" / "metrics.py"
            ok.parent.mkdir(parents=True)
            ok.write_text("import torch.nn.functional as F\nx = F.mse_loss(a, b)\n", encoding="utf-8")
            self.assertEqual(loss_registry.lint(root), [])

    def test_unregistered_loss_rejected(self):
        with self.assertRaises(ValueError):
            loss_registry.assert_registered("mystery_loss")
        loss_registry.assert_registered("structure")


if __name__ == "__main__":
    unittest.main()
