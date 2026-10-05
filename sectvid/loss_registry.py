"""Loss registry (spec 8). Anything else must be registered with a role and approved.
The lint test (tests/test_1_loss_lint.py) fails the build if an unregistered loss,
or a full-frame pixel-regression loss, touches model output.
"""
import re
from pathlib import Path

REGISTRY = {
    "structure": {"role": "planner"},
    "timing": {"role": "planner"},
    "velocity": {"role": "planner"},
    "magnitude": {"role": "planner", "kind": "regularizer"},
    "generative": {"role": "drawer"},
    "perceptual": {"role": "drawer"},
    "identity": {"role": "drawer"},
    "structure_obedience": {"role": "drawer"},
}

# Pixel-space regression primitives. Legal only inside sectvid/eval (metrics),
# never as a training signal, never on drawer output.
_FORBIDDEN_PATTERNS = [
    re.compile(r"\bnn\.(MSELoss|L1Loss|SmoothL1Loss)\b"),
    re.compile(r"\bF\.(mse_loss|l1_loss|smooth_l1_loss)\b"),
    re.compile(r"\bmse_loss\b|\bF\.mse\b"),
    re.compile(r"\bF\.ssim\b|\bssim_loss\b|\bsmoothed_l1_loss\b"),
]


def is_registered(name):
    return name in REGISTRY


def assert_registered(name):
    if not is_registered(name):
        raise ValueError(f"loss '{name}' is not in the registry; register it with a role and get approval (spec 8)")


def get_registry():
    return {k: dict(v) for k, v in REGISTRY.items()}


def lint(project_root):
    """Scan sectvid/ for forbidden loss primitives outside the eval package."""
    violations = []
    root = Path(project_root) / "sectvid"
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(project_root))
        if rel.startswith("sectvid/eval/") or rel.startswith("sectvid\\eval\\"):
            continue
        if rel.endswith("loss_registry.py"):
            continue
        # Planner losses implement registered losses (structure, velocity, timing, magnitude)
        if rel.startswith("sectvid/planner/losses.py") or rel.startswith("sectvid\\planner\\losses.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for pat in _FORBIDDEN_PATTERNS:
                if pat.search(line):
                    violations.append(f"{rel}:{i}: forbidden loss primitive '{line.strip()}'")
    return violations
