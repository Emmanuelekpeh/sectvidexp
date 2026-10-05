"""Baseline generators, always run (spec 9).

Structure baselines: hold-last-structure, constant-velocity extrapolation.
Frame baselines: copy-last-frame, and an evaluation-only flow-warp reference
(spec 4: warp/flow may exist only as an evaluation reference).

Every baseline takes a context prefix and produces the remainder of the window.
"""
import dataclasses

import cv2
import numpy as np

from ..structure.schema import camera_corrected_prev_points


def context_len(cfg):
    return int(cfg["planner"]["context_len"])


def hold_last_structure(structs, n_context):
    last = structs[n_context - 1]
    return [last for _ in range(len(structs) - n_context)]


def constant_velocity(structs, n_context):
    """Structure extrapolation at the last observed SUBJECT velocity (spec 9).
    The velocity is camera-corrected: the previous frame's points are
    transformed by the camera affine, so background motion is excluded."""
    prev, last = structs[n_context - 2], structs[n_context - 1]
    if not (prev.ok and last.ok):
        return [last for _ in range(len(structs) - n_context)]
    valid = (prev.body[:, 2] >= 0.3) & (last.body[:, 2] >= 0.3)
    if valid.sum() < 4:
        return [last for _ in range(len(structs) - n_context)]
    pa, _ = camera_corrected_prev_points(prev, last, 0.3)
    v = np.zeros_like(last.body[:, :2])
    v[valid] = last.body[valid, :2] - pa[valid]
    dt = last.dt if last.dt > 0 else 1.0 / 24.0
    out = []
    for i in range(1, len(structs) - n_context + 1):
        body = last.body.copy()
        body[:, 0] = np.clip(last.body[:, 0] + i * dt * v[:, 0], 0.0, 1.0)
        body[:, 1] = np.clip(last.body[:, 1] + i * dt * v[:, 1], 0.0, 1.0)
        out.append(dataclasses.replace(last, frame_index=last.frame_index + i, body=body))
    return out


def copy_last_frame(frames, n_context):
    return [frames[n_context - 1] for _ in range(len(frames) - n_context)]


def flow_warp_reference(frames, n_context):
    """Evaluation-only reference: warp the last context frame forward using
    optical flow between real consecutive frames. Uses real future frames for
    flow estimation; allowed only as an evaluation reference (spec 4, 9).
    """
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    cur = frames[n_context - 1].copy()
    out = []
    for i in range(n_context, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(
            gray[i - 1], gray[i], None, 0.5, 3, 21, 3, 5, 1.2, flags=0
        )
        cur = cv2.remap(cur, flow, None, interpolation=cv2.INTER_LINEAR)
        out.append(cur)
    return out


def run_stage0(cfg):
    """End-to-end Stage 0 run (spec 12): sample windows, run every baseline,
    compute the spec 9 metric battery per step, write JSON + contact sheets.

    No model code exists yet (planner/drawer are Stage 2/3), so structure
    baselines are scored at structure level and frame baselines at frame level;
    drawer-rendered structure baselines arrive with Stage 3.
    """
    import json

    from ..config import root_path
    from ..data import clips as data_clips
    from ..data.windows import build_windows
    from ..data.references import select_references
    from ..structure.cache import load_structure, save_structure
    from ..structure.extractor import StructureExtractor
    from . import metrics, report

    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    sample_n = int(cfg["eval"]["sample_clips"])
    frames_per = int(cfg["eval"]["frames_per_clip"])
    ctx = context_len(cfg)
    wlen = min(int(cfg["window"]["length"]), frames_per)
    ecfg = cfg["eval"]

    ids = data_clips.discover_clips(cfg)
    wide = [c for c in ids if data_clips.clip_properties(cfg, c)["width"] >= 480]
    pool = wide or ids
    picks = [pool[int(i)] for i in rng.choice(len(pool), size=min(sample_n, len(pool)), replace=False)]

    extractor = StructureExtractor(cfg)
    encoder = metrics.IdentityEncoder(kind=ecfg["identity"]["encoder"], device=ecfg["identity"]["device"])
    gate_dir = root_path(cfg, cfg["report"]["dir"])
    bins_seen = set()
    rows_real = []
    real_frame_seqs = {}
    rows_by_baseline = {}
    per_step = {"structure": {}, "frame": {}}
    summary = {"n_clips": len(picks), "window_length": wlen, "context": ctx,
               "clips": picks}

    for cid in picks:
        frames = data_clips.load_frames(cfg, cid, 0, min(frames_per, data_clips.clip_properties(cfg, cid)["n_frames"]))
        try:
            structs = load_structure(cfg, cid)
        except FileNotFoundError:
            structs = extractor.extract_sequence(frames)
            save_structure(cfg, cid, structs)
        window = structs[:wlen]
        real_frames = frames[:wlen]
        real_frame_seqs[cid] = real_frames
        windows = build_windows(cfg, cid, structs)
        for w in windows:
            bins_seen.add(w["bin"])
        ok_frames_all = [i for i, s in enumerate(structs) if s.ok]
        refs = select_references(
            identity_clips=[cid],
            target_window={"clip_id": cid, "start": 0, "end": wlen},
            available_frames={cid: [f for f in ok_frames_all if f >= wlen] or []},
            min_gap_frames=cfg["reference"]["min_gap_frames"],
            count=cfg["reference"]["count"],
            seed=seed,
        )
        reference_frames = [frames[i] for _, i in refs] if refs else []
        summary.setdefault("reference_used", {})[cid] = len(refs)


        base_ctx = min(ctx, wlen - 1)

        for name, pred in (
            ("hold_last_structure", hold_last_structure(window, base_ctx)),
            ("constant_velocity_structure", constant_velocity(window, base_ctx)),
        ):
            steps = {}
            for k, (p, t) in enumerate(zip(pred, window[base_ctx:])):
                steps[k] = metrics.structure_error(p, t)
            per_step["structure"].setdefault(name, []).append({
                "clip": cid, "steps": steps,
                "angular": metrics.angular_error([window[base_ctx - 1]] + pred, [window[base_ctx - 1]] + list(window[base_ctx:])),
                "motion_magnitude_ratio": metrics.motion_magnitude_ratio(
                    [window[base_ctx - 1]] + pred, [window[base_ctx - 1]] + list(window[base_ctx:])),
                "frozen_rate": metrics.frozen_rate([window[base_ctx - 1]] + pred, [window[base_ctx - 1]] + list(window[base_ctx:])),
                "hallucinated_motion_rate": metrics.hallucinated_motion_rate(
                    [window[base_ctx - 1]] + pred, [window[base_ctx - 1]] + list(window[base_ctx:])),
                "is_frozen": metrics.is_frozen(pred),
            })
            rows_by_baseline.setdefault(name, []).append(
                report.stick_figure(pred[-1], real_frames[0].shape[0], real_frames[0].shape[1])
            )

        for name, pred in (
            ("copy_last_frame", copy_last_frame(real_frames, base_ctx)),
            ("flow_warp_reference", flow_warp_reference(real_frames, base_ctx)),
        ):
            steps = {}
            for k, (g, r) in enumerate(zip(pred, real_frames[base_ctx:])):
                steps[k] = {
                    "ghosting": None,
                    "flicker": None,
                    "sharpness": None,
                }
            ghost = metrics.ghosting_score(pred, real_frames[base_ctx:], ks=tuple(ecfg["ghosting_ks"]),
                                           ratio=float(ecfg.get("ghosting_ratio", 0.9)))
            flick = metrics.flicker_ratio(pred, real_frames[base_ctx:], window=ecfg["flicker_window"])
            sharp = metrics.sharpness_ratio(pred, real_frames[base_ctx:])
            lpips = None
            if ecfg["lpips"]:
                lpips_vals = []
                for g, r in zip(pred, real_frames[base_ctx:]):
                    try:
                        lpips_vals.append(metrics.lpips_vs_real(g, r))
                    except Exception:
                        lpips_vals.append(None)
                lpips = float(np.nanmean([v for v in lpips_vals if v is not None])) if any(v is not None for v in lpips_vals) else None
            ident = None
            if reference_frames:
                ident = metrics.identity_cosine(pred, reference_frames, encoder)
            per_step["frame"].setdefault(name, []).append({
                "clip": cid,
                "ghosting": ghost, "flicker_ratio": flick, "sharpness_ratio": sharp,
                "lpips_vs_real": lpips, "identity_cosine": ident,
            })
            rows_by_baseline.setdefault(name, []).append(np.mean(pred, axis=0).astype(np.uint8))

        rows_real.append(np.mean(real_frames, axis=0).astype(np.uint8))

    summary["bins"] = sorted(bins_seen)
    payload = {
        "stage": 0,
        "baseline_kinds": {
            "hold_last_structure": "predictive",
            "constant_velocity_structure": "predictive",
            "copy_last_frame": "predictive",
            "flow_warp_reference": ("oracle_eval_only: warp uses optical flow estimated "
                                    "from real future frames; an evaluation reference "
                                    "(spec 4), never a predictive baseline (spec 9)"),
        },
        "per_step": per_step,
        "summary": summary,
    }
    p = report.write_json_report(gate_dir, "stage0_baselines", payload, cfg, seed)

    cell = lambda arrs: [np.mean(a, axis=0).astype(np.uint8) for a in arrs]
    sheet_rows = [("real", cell(rows_real))]
    for name, per_clip in rows_by_baseline.items():
        sheet_rows.append((name, cell(per_clip)))
    try:
        sheet = gate_dir / f"stage0_contact_{seed}.png"
        report.render_contact_sheet(sheet_rows, sheet, cols=max(len(picks), 1), scale=1)
    except Exception:
        pass

    for cid in picks:
        try:
            report.write_video(real_frame_seqs[cid],
                               gate_dir / f"stage0_real_{cid}_{seed}.mp4", fps=24.0)
        except Exception:
            pass

    print(f"stage0 report: {p}")
    return payload


if __name__ == "__main__":
    import sys
    from sectvid.config import load_config
    run_stage0(load_config(sys.argv[1] if len(sys.argv) > 1 else None))
