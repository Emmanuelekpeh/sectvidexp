"""Test 4: reference-leak test (spec 10.4). No reference within N frames of its
target; no identity or clip overlap across splits."""
import unittest

from sectvid.data.references import select_references, reference_leak_violations
from sectvid.data.splits import split_by_identity, leakage_report


class TestReferenceLeak(unittest.TestCase):
    GAP = 8

    def _window(self, start=32, end=64):
        return {"clip_id": "A", "start": start, "end": end}

    def test_no_reference_near_target_in_same_clip(self):
        for seed in range(20):
            refs = select_references(
                identity_clips=["A", "B"],
                target_window=self._window(),
                available_frames={"A": list(range(100)), "B": list(range(100))},
                min_gap_frames=self.GAP,
                count=5,
                seed=seed,
                prefer_other_clip=True,
            )
            self.assertEqual(reference_leak_violations(refs, self._window(), self.GAP), [])
            for clip_id, idx in refs:
                if clip_id == "A":
                    self.assertLess(idx, 32 - self.GAP)
                    self.assertGreaterEqual(idx, 64 + self.GAP)

    def test_prefer_other_clip(self):
        refs = select_references(
            identity_clips=["A", "B"],
            target_window=self._window(),
            available_frames={"A": list(range(100)), "B": list(range(100))},
            min_gap_frames=self.GAP,
            count=4,
            seed=0,
            prefer_other_clip=True,
        )
        self.assertTrue(refs)
        self.assertTrue(all(clip_id == "B" for clip_id, _ in refs))

    def test_same_clip_fallback_when_no_other_clip(self):
        refs = select_references(
            identity_clips=["A"],
            target_window=self._window(),
            available_frames={"A": list(range(200))},
            min_gap_frames=self.GAP,
            count=4,
            seed=0,
        )
        self.assertEqual(len(refs), 4)
        self.assertTrue(all(clip_id == "A" for clip_id, _ in refs))
        self.assertEqual(reference_leak_violations(refs, self._window(), self.GAP), [])

    def test_violations_detected_directly(self):
        w = self._window()
        bad = [("A", 40), ("A", 30), ("B", 10)]
        self.assertEqual(len(reference_leak_violations(bad, w, self.GAP)), 2)


class TestSplitLeakage(unittest.TestCase):
    def test_no_clip_or_identity_overlap(self):
        clip_to_identity = {}
        for i in range(30):
            clip_to_identity[f"clip{i:02d}"] = f"ident{i % 8:02d}"
        for seed in range(10):
            train, val, unseen = split_by_identity(
                clip_to_identity, identity_holdout_frac=0.25, test_frac=0.25, seed=seed
            )
            report = leakage_report(train, val, unseen, clip_to_identity)
            self.assertTrue(report["clean"], report)
            self.assertEqual(len(train) + len(val) + len(unseen), 30)

    def test_held_out_identities_never_in_train(self):
        clip_to_identity = {f"clip{i:02d}": f"ident{i % 5:02d}" for i in range(15)}
        train, val, unseen = split_by_identity(
            clip_to_identity, identity_holdout_frac=0.4, test_frac=0.0, seed=0
        )
        unseen_idents = {clip_to_identity[c] for c in unseen}
        self.assertGreater(len(unseen_idents), 0)
        self.assertEqual(unseen_idents & {clip_to_identity[c] for c in train}, set())


if __name__ == "__main__":
    unittest.main()
