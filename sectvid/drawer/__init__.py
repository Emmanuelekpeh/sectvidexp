"""Drawer (spec 6.4). No model code until Stage 3.

The drawer is fine-tuned from a pretrained image generator (owner-approved),
conditioned on rendered structure plus reference attention. It renders each
frame in full from structure + reference; pixel errors never feed the planner.
Model choice requires owner approval.
"""
