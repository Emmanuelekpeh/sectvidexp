"""Planner (spec 6.3).

The planner predicts the next structure in structure space, autoregressively,
and is trained on its own rollouts. It must beat constant-velocity extrapolation
on endpoint and angle, with camera and subject motion scored separately, to pass
the Stage 2 gate.
"""
from .structure_codec import (
    encode_structure,
    decode_structure,
    TOTAL_FEATURE_DIM,
    get_feature_weights,
    get_body_confidence_mask,
    compute_torso_normalizer,
)
from .model import StructureTransformer, build_planner
from .losses import (
    structure_loss,
    velocity_loss,
    timing_loss,
    magnitude_regularizer,
    compute_all_losses,
)

__all__ = [
    "encode_structure",
    "decode_structure",
    "TOTAL_FEATURE_DIM",
    "get_feature_weights",
    "get_body_confidence_mask",
    "compute_torso_normalizer",
    "StructureTransformer",
    "build_planner",
    "structure_loss",
    "velocity_loss",
    "timing_loss",
    "magnitude_regularizer",
    "compute_all_losses",
]