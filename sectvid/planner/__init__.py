"""Planner (spec 6.3). No model code until Stage 2.

The planner predicts the next structure in structure space, autoregressively,
and is trained on its own rollouts. It must beat constant-velocity extrapolation
on endpoint and angle, with camera and subject motion scored separately, to pass
the Stage 2 gate.
"""
