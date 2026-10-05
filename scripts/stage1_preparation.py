"""Stage 1 preparation (spec 12): windowing, motion bins, splits.

Gate: bins populated on the full dataset; no leakage; identity curation
complete (a precondition added after Stage 0 review: splits are
identity-first, so unreviewed identity defaults must not feed splits).

Writes reports/stage1_preparation_*.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sectvid.config import load_config, root_path
from sectvid.data import clips as data_clips
from sectvid.data import splits
from sectvid.data.windows import build_windows
from sectvid.eval import report
from sectvid.structure import cache


def main():
    cfg = load_config()
    ids = data_clips.discover_clips(cfg)
    seed = cfg["seed"]

    payload = {"stage": 1, "n_clips": len(ids), "window": cfg["window"],
               "motion_bins": {k: v for k, v in cfg["motion_bins"].items() if k != "sample_quotas"}}
    payload["motion_bins"]["sample_quotas"] = cfg["motion_bins"]["sample_quotas"]

    bins_cfg = cfg["motion_bins"]
    bin_windows = {"still": 0, "slow": 0, "fast": 0, "unobserved": 0}
    bin_speeds = {"still": [], "slow": [], "fast": [], "unobserved": []}
    clip_stats = {}
    missing = []

    for cid in ids:
        try:
            structs = cache.load_structure(cfg, cid, include_silhouette=False)
        except FileNotFoundError:
            missing.append(cid)
            continue
        windows = build_windows(cfg, cid, structs)
        ok_rate = float(sum(s.ok for s in structs)) / max(len(structs), 1)
        clip_stats[cid] = {"n_frames": len(structs), "ok_rate": ok_rate,
                           "n_windows": len(windows),
                           "bins": {b: sum(1 for w in windows if w["bin"] == b) for b in bin_windows}}
        for w in windows:
            bin_windows[w["bin"]] += 1
            bin_speeds[w["bin"]].append(w["subject_speed"])

    payload["cache_complete"] = not missing
    payload["missing_caches"] = missing
    payload["clip_stats"] = clip_stats
    payload["bins"] = {
        b: {"n_windows": bin_windows[b],
            "subject_speed_norm_per_s_p50": (float(np.percentile(bin_speeds[b], 50)) if bin_speeds[b] else None),
            "subject_speed_norm_per_s_p90": (float(np.percentile(bin_speeds[b], 90)) if bin_speeds[b] else None)}
        for b in ("still", "slow", "fast", "unobserved")
    }
    empty_bins = [b for b in ("still", "slow", "fast") if bin_windows[b] == 0]
    payload["bins_populated"] = not empty_bins
    payload["empty_bins"] = empty_bins

    metas = {cid: data_clips.normalized_meta(cfg, cid) for cid in ids if cid not in missing}
    unreviewed = splits.curation_check(metas)
    payload["identity_curation"] = {"unreviewed": unreviewed, "complete": not unreviewed}

    payload["splits"] = None
    payload["leakage"] = None
    if not unreviewed:
        clip_to_identity = {cid: str(m["identity_id"]) for cid, m in metas.items()}
        train, val, unseen = splits.split_by_identity(
            clip_to_identity,
            cfg["splits"]["identity_holdout_frac"],
            cfg["splits"]["test_frac"],
            cfg["splits"]["seed"],
        )
        payload["splits"] = {
            "train": sorted(train), "val": sorted(val), "unseen": sorted(unseen),
            "counts": {"train": len(train), "val": len(val), "unseen": len(unseen)},
            "identities_per_split": {
                name: sorted({clip_to_identity[c] for c in group})
                for name, group in (("train", train), ("val", val), ("unseen", unseen))
            },
        }
        payload["leakage"] = splits.leakage_report(train, val, unseen, clip_to_identity)

    gate = {
        "cache_complete": payload["cache_complete"],
        "bins_populated": payload["bins_populated"],
        "identity_curation_complete": payload["identity_curation"]["complete"],
        "leakage_clean": bool(payload["leakage"]["clean"]) if payload["leakage"] else False,
    }
    gate["gate"] = "PASS" if all(gate.values()) else ("INCOMPLETE" if payload["splits"] is None and gate["identity_curation_complete"] is False else "FAIL")
    payload["gate"] = gate

    p = report.write_json_report(root_path(cfg, cfg["report"]["dir"]), "stage1_preparation",
                                 payload, cfg, seed)
    print(json.dumps({"report": str(p), "gate": gate,
                      "bins": payload["bins"], "empty_bins": empty_bins,
                      "missing_caches": len(missing),
                      "unreviewed_identities": len(unreviewed)}, indent=1))
    return p


if __name__ == "__main__":
    main()
