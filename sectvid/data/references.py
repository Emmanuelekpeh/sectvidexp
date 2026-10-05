"""Reference sheet selection (spec 6.4).

References come from the same identity but a different time than the target,
never adjacent frames: no reference within `min_gap_frames` of any target frame
in the same clip. Enforced here and re-checked by the reference-leak test.
"""
import numpy as np


def window_frames(target_window):
    start, end = target_window["start"], target_window["end"]
    return set(range(start, end))


def reference_leak_violations(reference_frames, target_window, min_gap_frames):
    """reference_frames: list of (clip_id, frame_index) from one identity."""
    target_clip = target_window["clip_id"]
    frames = window_frames(target_window)
    lo = target_window["start"] - min_gap_frames
    hi = target_window["end"] + min_gap_frames
    violations = []
    for clip_id, idx in reference_frames:
        if clip_id != target_clip:
            continue
        if frames.__contains__(idx) or lo <= idx < hi:
            violations.append((clip_id, int(idx)))
    return violations


def select_references(identity_clips, target_window, available_frames, min_gap_frames,
                      count, seed, prefer_other_clip=True):
    """Pick `count` reference frames for a target window.

    identity_clips: clip ids of the same identity.
    available_frames: dict clip_id -> list of frame indices with detected person.
    """
    rng = np.random.default_rng(seed)
    target_clip = target_window["clip_id"]
    other_clips = [c for c in sorted(identity_clips) if c != target_clip]
    same_clip = [target_clip]

    lo = target_window["start"] - min_gap_frames
    hi = target_window["end"] + min_gap_frames

    def pool(clip_ids):
        out = []
        for clip_id in clip_ids:
            for f in available_frames.get(clip_id, []):
                if clip_id == target_clip and lo <= f < hi:
                    continue
                out.append((clip_id, f))
        return out

    other = pool(other_clips)
    same = pool(same_clip)
    candidates = other if (prefer_other_clip and other) else (same + other)
    if not candidates:
        return []
    pick = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    return [candidates[int(i)] for i in pick]
