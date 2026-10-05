"""Config loading for sectvid v4. Every guess value lives in configs/*.yaml."""
import os
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_CACHE = None


def _validate(cfg):
    required = ["data", "window", "motion_bins", "splits", "structure",
                "reference", "planner", "drawer", "eval", "report"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"config missing required keys: {missing}")
    if "sample_quotas" not in cfg["motion_bins"]:
        raise KeyError("config motion_bins.sample_quotas is required")
    total = sum(float(v) for v in cfg["motion_bins"]["sample_quotas"].values())
    if abs(total - 1.0) > 1e-6:
        raise KeyError(f"motion_bins.sample_quotas must sum to 1.0, got {total}")


def load_config(path=None, use_cache=True):
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and use_cache:
        return _CONFIG_CACHE
    p = Path(path) if path else Path(
        os.environ.get("SECTVID_CONFIG", str(_PROJECT_ROOT / "configs" / "default.yaml"))
    )
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    cfg["_root"] = str(_PROJECT_ROOT)
    cfg["_config_path"] = str(p)
    if use_cache:
        _CONFIG_CACHE = cfg
    return cfg


def root_path(cfg, *parts):
    return Path(cfg["_root"]) / Path(*parts)
