"""Run artifacts: JSON reports, contact sheets, and videos (spec 9).

Fixed seeds, saved as JSON plus auto-rendered contact sheets/videos; a human
must look at the sequences (spec 9).
"""
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__ as SECTVID_VERSION


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _versions(cfg, seed):
    return {
        "sectvid_version": SECTVID_VERSION,
        "config_path": cfg.get("_config_path"),
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_json_report(report_dir, name, payload, cfg, seed):
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}_{_stamp()}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"meta": _versions(cfg, seed), **payload}, f, indent=1)
    return p


def render_contact_sheet(rows, out_path, cols=8, scale=2, labels=None):
    """rows: list of (frames, label). One row per rollout; columns are steps."""
    import cv2
    cells = []
    for r, row in enumerate(rows):
        frames, label = row
        strip = []
        for fr in frames:
            g = fr if fr.ndim == 3 else np.stack([fr] * 3, axis=-1)
            g = np.ascontiguousarray(g.astype(np.uint8)[:, :, ::-1])
            if scale != 1:
                g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            if label and r == 0:
                cv2.putText(g, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            strip.append(g)
        cells.append(strip)
    w = max(c.shape[1] for row in cells for c in row)
    h = max(c.shape[0] for row in cells for c in row)
    out = np.zeros((h * len(cells), w * min(cols, max(len(r) for r in cells)), 3), np.uint8)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row[:cols]):
            out[r * h:r * h + cell.shape[0], c * w:c * w + cell.shape[1]] = cell
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    return Path(out_path)


def write_video(frames, out_path, fps=24.0):
    import cv2
    frames = [f if f.ndim == 3 else np.stack([f] * 3, axis=-1) for f in frames]
    h, w = frames[0].shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(np.ascontiguousarray(f.astype(np.uint8)))
    vw.release()
    return out_path


def stage0_gate(payload):
    """Stage 0 gate (spec 12): baselines run end to end; all tests pass.
    Kill condition: extractor too noisy to serve as ground truth."""
    gate = {}
    for key in ("tests_all_pass", "baselines_end_to_end"):
        gate[key] = bool(payload.get(key, False))
    ext = payload.get("extractor_validation", {})
    too_noisy = (
        (ext.get("jitter_p90_px") is not None and ext.get("jitter_p90_px", 0) > ext.get("jitter_limit_px", 99))
        or (ext.get("dropout_rate") is not None and ext.get("dropout_rate", 0) > ext.get("dropout_limit", 1.0))
        or (ext.get("camera_translation_err") is not None
            and ext.get("camera_translation_err", 0) > ext.get("camera_err_limit", 99))
    )
    gate["extractor_too_noisy"] = too_noisy
    gate["gate"] = "FAIL" if too_noisy else ("PASS" if gate["tests_all_pass"] and gate["baselines_end_to_end"] else "INCOMPLETE")
    return gate


STICK_EDGES = [
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def stick_figure(s, h, w, min_conf=0.3, color=(0, 255, 0), thickness=3):
    """Render a structure as a stick figure (spec 12, Stage 2 pencil test)."""
    import cv2
    img = np.zeros((h, w, 3), np.uint8)
    pts = {}
    for j in range(s.body.shape[0]):
        if s.body[j, 2] >= min_conf:
            pts[j] = (int(s.body[j, 0] * w), int(s.body[j, 1] * h))
    for a, b in STICK_EDGES:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], color, thickness)
    if 0 in pts:
        cv2.circle(img, pts[0], max(4, w // 80), (0, 255, 255), -1)
    return img


def _pytest_summary(report_dir):
    """Parse the newest pytest JUnit XML (run pytest with
    --junitxml=<report_dir>/pytest_results.xml). Returns
    {"tests", "failures", "errors", "skipped", "all_pass"} or {} if absent."""
    out = Path(report_dir)
    xmls = sorted(out.glob("pytest_results*.xml"))
    if not xmls:
        return {}
    import xml.etree.ElementTree as ET
    root = ET.parse(str(xmls[-1])).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return {}

    def attr(name, default=0):
        return int(suite.attrib.get(name, default) or default)

    tests, failures, errors = attr("tests"), attr("failures"), attr("errors")
    return {
        "file": xmls[-1].name,
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": attr("skipped"),
        "all_pass": tests > 0 and failures == 0 and errors == 0,
    }


def write_stage0_gate(report_dir, cfg):
    """Merge pytest results, the newest stage0_baselines and
    test2_extractor_validation JSONs into the Stage 0 gate report.
    Call after pytest and the baseline run."""
    out = Path(report_dir)
    baselines = sorted(out.glob("stage0_baselines_*.json"))
    validation = sorted(out.glob("test2_extractor_validation_*.json"))
    payload = {}
    summary = _pytest_summary(out)
    if summary:
        payload["tests_all_pass"] = summary["all_pass"]
        payload["pytest_summary"] = summary
    else:
        payload["tests_all_pass"] = False
    if baselines:
        with open(baselines[-1], "r", encoding="utf-8") as f:
            payload["baselines_end_to_end"] = True
            payload["baselines_report"] = baselines[-1].name
    else:
        payload["baselines_end_to_end"] = False
    if validation:
        with open(validation[-1], "r", encoding="utf-8") as f:
            data = json.load(f)
        v = cfg["structure"]["validation"]
        ext = data.get("extractor_validation", data)
        payload["extractor_validation"] = {
            "jitter_p90_px": ext.get("jitter_p90_mean_shift_px"),
            "jitter_max_px": ext.get("jitter_max_mean_shift_px"),
            "jitter_limit_px": v["jitter_limit_px"],
            "jitter_per_clip": ext.get("jitter_per_clip", {}),
            "dropout_rate": ext.get("dropout_rate"),
            "dropout_limit": v["max_dropout"],
            "camera_translation_err": ext.get("camera_translation_err_norm"),
            "camera_err_limit": v["max_camera_translation_err"],
        }
    else:
        payload["extractor_validation"] = {}
    gate = stage0_gate(payload)
    p = out / f"stage0_gate_{_stamp()}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"gate": gate, "evidence": payload}, f, indent=1)
    return p, gate
