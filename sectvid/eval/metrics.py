"""Evaluation metrics (spec 9).

All metrics are computed per step of rollout, per behavior, and per motion bin
by the caller; this module provides the primitives. Freeze detection is a
detector only, never a pass criterion (spec 9).
"""
import numpy as np

from ..structure.schema import torso_length, mean_speeds, step_displacement


def _as_gray(frames):
    frames = np.asarray(frames)
    if frames.ndim == 3:
        return frames
    return np.mean(frames, axis=-1)


def structure_error(pred, true, min_conf=0.3):
    out = {}
    if not (pred.ok and true.ok):
        out.update({"body_err": 1.0, "head_pose_err_deg": 90.0, "head_pos_err": 1.0,
                    "face_err": 1.0, "silhouette_iou": 0.0, "any_detection": bool(pred.ok or true.ok)})
        return out
    scale = torso_length(true.body, min_conf)
    if scale < 1e-3:
        scale = true.head_scale * 3.0
    if scale < 1e-3:
        scale = 0.1
    pa, pb = pred.body, true.body
    valid = (pa[:, 2] >= min_conf) & (pb[:, 2] >= min_conf)
    if valid.sum() > 0:
        d = np.linalg.norm(pa[valid, :2] - pb[valid, :2], axis=1) / scale
        out["body_err"] = float(d.mean())
        out["n_joints"] = int(valid.sum())
    else:
        out["body_err"] = 1.0
        out["n_joints"] = 0
    out["head_pose_err_deg"] = float(np.mean(np.abs(pred.head_pose - true.head_pose)) * 180.0 / np.pi)
    out["head_pos_err"] = float(np.linalg.norm(pred.head_pos - true.head_pos))
    out["face_err"] = float(np.mean(np.abs(pred.face_params - true.face_params)))
    out["silhouette_iou"] = float(moving_region_iou(pred.silhouette, true.silhouette))
    out["any_detection"] = True
    return out


def angular_error(pred_structs, true_structs, min_conf=0.3):
    """Per-step direction error of subject displacement, magnitude-weighted (spec 9)."""
    angles, weights = [], []
    for i in range(min(len(pred_structs), len(true_structs)) - 1):
        dp = step_displacement(pred_structs[i], pred_structs[i + 1], min_conf)
        dt = step_displacement(true_structs[i], true_structs[i + 1], min_conf)
        if dp is None or dt is None:
            continue
        dp, dt = dp.mean(axis=0), dt.mean(axis=0)
        mag = float(np.linalg.norm(dt))
        if mag < 1e-6:
            continue
        cos = float(np.clip(np.dot(dp, dt) / (np.linalg.norm(dp) * mag + 1e-12), -1.0, 1.0))
        angles.append(np.arccos(cos) * 180.0 / np.pi)
        weights.append(mag)
    if not angles:
        return {"angular_err_deg": None, "n_steps": 0, "magnitude_weighted": True}
    w = np.array(weights)
    return {"angular_err_deg": float(np.average(angles, weights=w)), "n_steps": len(angles),
            "magnitude_weighted": True}


def motion_magnitude_ratio(pred_structs, true_structs):
    """Predicted/real mean subject speed. None when the real window has no motion."""
    sp = mean_speeds(pred_structs)
    st = mean_speeds(true_structs)
    if len(st) == 0 or float(np.mean(st)) < 1e-4:
        return None
    return float(np.mean(sp)) / float(np.mean(st))


def frozen_rate(pred_structs, true_structs, pct=5.0):
    """On bins with real motion: fraction of rollout steps below the real 5th
    percentile of speed (spec 9)."""
    st = mean_speeds(true_structs)
    moving = st[st > 1e-4]
    if len(moving) == 0:
        return None
    thresh = float(np.percentile(moving, pct))
    sp = mean_speeds(pred_structs)
    if len(sp) == 0:
        return None
    return float(np.mean(sp < thresh))


def hallucinated_motion_rate(pred_structs, true_structs, pct=95.0):
    """On still bins: fraction of rollout steps above the real 95th percentile
    of speed (spec 9)."""
    st = mean_speeds(true_structs)
    thresh = float(np.percentile(st, pct))
    sp = mean_speeds(pred_structs)
    if len(sp) == 0:
        return None
    return float(np.mean(sp > thresh))


def is_frozen(pred_structs, eps=1e-3):
    """Freeze detector only (spec 9): a flag, never a pass criterion."""
    sp = mean_speeds(pred_structs)
    if len(sp) == 0:
        return {"frozen": None, "fraction_still": None}
    fraction = float(np.mean(sp < eps))
    return {"frozen": fraction > 0.9, "fraction_still": fraction}


def moving_region_iou(pred_mask, true_mask):
    pred_mask = np.asarray(pred_mask) > 0
    true_mask = np.asarray(true_mask) > 0
    inter = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def _frame_dist(a, b):
    a, b = _as_gray(a), _as_gray(b)
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def flicker_ratio(gen_frames, real_frames, window=5):
    """High-frequency temporal energy of generated frames vs real footage of the
    same window (spec 9)."""
    def energy(frames):
        g = _as_gray(frames).astype(np.float64)
        n = len(g)
        if n < 2:
            return 0.0
        lo = np.stack([g[max(0, i - window):min(n, i + window + 1)].mean(axis=0)
                       for i in range(n)])
        return float(np.mean((g - lo) ** 2))
    real_e = energy(real_frames)
    gen_e = energy(gen_frames)
    if real_e < 1e-12:
        return None
    return gen_e / real_e


def sharpness_ratio(gen_frames, real_frames):
    """Laplacian variance vs real footage, to catch accidental sharpening/blur."""
    import cv2
    def var(frames):
        g = _as_gray(frames)
        vals = [float(cv2.Laplacian(g[i], cv2.CV_64F).var()) for i in range(len(g))]
        return float(np.mean(vals))
    real_v = var(real_frames)
    gen_v = var(gen_frames)
    if real_v < 1e-12:
        return None
    return gen_v / real_v


def ghosting_score(gen_frames, real_frames, ks=(1, 2, 4), ratio=0.9):
    """Ghosting / cross-fade score (spec 9). For generated frame t, compare against
    alpha-blends of frames t-k and t+k. A frame is flagged when a blend explains
    the generated frame meaningfully better than it explains real footage of the
    same window (i.e. the generated frame looks like one image dissolving into
    another). `ratio` is the "as well as" margin: flag when d_gen < ratio * d_real.
    When real footage is itself blend-exact (d_real ~ 0), a blend that explains
    the generated frame but not the real one is flagged."""
    gen = _as_gray(gen_frames).astype(np.float64)
    real = _as_gray(real_frames).astype(np.float64)
    n = min(len(gen), len(real))
    flags = np.zeros(n, dtype=bool)
    ks = [k for k in ks if 2 * k < n]
    for t in range(1, n - 1):
        best_gen, best_real = np.inf, np.inf
        for k in ks:
            if t - k < 0 or t + k >= n:
                continue
            blend_g = 0.5 * (gen[t - k] + gen[t + k])
            blend_r = 0.5 * (real[t - k] + real[t + k])
            best_gen = min(best_gen, _frame_dist(gen[t], blend_g))
            best_real = min(best_real, _frame_dist(real[t], blend_r))
        if best_gen >= np.inf:
            continue
        if best_real > 1e-6:
            flags[t] = best_gen < ratio * best_real
        else:
            flags[t] = best_gen > 1e-6
    interior = flags[1:n - 1]
    if interior.size == 0:
        return {"flags": flags.tolist(), "flag_rate": None, "n_interior": 0}
    return {"flags": flags.tolist(), "flag_rate": float(interior.mean()),
            "n_interior": int(interior.size)}


_LPIPS_NET = None


def lpips_vs_real(gen_frame, real_frame):
    """Perceptual distance vs real where a real future exists (spec 9)."""
    global _LPIPS_NET
    import lpips
    import torch
    import cv2
    if _LPIPS_NET is None:
        _LPIPS_NET = lpips.LPIPS(net="alex")

    def to_t(x):
        g = x if x.ndim == 3 else np.stack([x] * 3, axis=-1)
        rgb = cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return t * 2.0 - 1.0
    with torch.no_grad():
        return float(_LPIPS_NET(to_t(gen_frame), to_t(real_frame)).item())


class IdentityEncoder:
    """Frozen encoder for identity similarity (spec 9). `vit` is an
    ImageNet-pretrained ViT-B/16 (frozen); `probe` is a download-free fallback
    (resized pixels). Model choice for the drawer itself still needs owner
    approval, but this metric encoder does not."""

    def __init__(self, kind="vit", device="cpu"):
        self.kind = kind
        self.device = device
        if kind in ("vit", "dinov2"):
            import torch
            from torchvision import models
            self._net = models.vit_b_16(weights="IMAGENET1K_V1").eval()
            self._net.to(device)
            for p in self._net.parameters():
                p.requires_grad_(False)
            self._torch = torch
        elif kind == "probe":
            pass
        else:
            raise ValueError(f"unknown identity encoder {kind}")

    def embed(self, frames):
        import cv2
        import numpy as np
        out = []
        for f in frames:
            g = f if f.ndim == 3 else np.stack([f] * 3, axis=-1)
            if self.kind in ("vit", "dinov2"):
                torch = self._torch
                rgb = cv2.resize(cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_BGR2RGB), (224, 224))
                t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                t = (t - mean) / std
                with torch.no_grad():
                    out.append(self._net(t.to(self.device)).flatten())
            else:
                small = cv2.resize(g.astype(np.uint8), (32, 32)).astype(np.float32).flatten()
                vec = small / (np.linalg.norm(small) + 1e-9)
                out.append(vec)
        v = np.stack(out)
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
        return v


def identity_cosine(pred_frames, reference_frames, encoder):
    """Cosine similarity to the reference sheet, over steps (spec 9)."""
    ref = encoder.embed(reference_frames)
    ref_mean = ref.mean(axis=0)
    ref_mean = ref_mean / (np.linalg.norm(ref_mean) + 1e-9)
    preds = encoder.embed(pred_frames)
    return [float(float(np.dot(p, ref_mean))) for p in preds]
