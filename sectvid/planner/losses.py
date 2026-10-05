"""Planner losses (spec 8, 6.3).

Registered losses:
- structure: coordinate error, confidence-weighted
- timing: dt error
- velocity: per-step speed error
- magnitude: regularizer only
"""
import torch
import torch.nn.functional as F

from .structure_codec import (
    TOTAL_FEATURE_DIM,
    BODY_DIM,
    NUM_BODY_POINTS,
    compute_torso_normalizer,
    get_feature_weights,
)


# Feature indices for slicing
BODY_START = 0
BODY_END = BODY_DIM  # 51
HEAD_POSE_START = BODY_END
HEAD_POSE_END = HEAD_POSE_START + 3  # 54
HEAD_POS_START = HEAD_POSE_END
HEAD_POS_END = HEAD_POS_START + 2  # 56
HEAD_SCALE_IDX = HEAD_POS_END  # 56
HEAD_CONF_IDX = HEAD_SCALE_IDX + 1  # 57
FACE_PARAMS_START = HEAD_CONF_IDX + 1  # 58
FACE_PARAMS_END = FACE_PARAMS_START + 52  # 110
FACE_BOX_START = FACE_PARAMS_END
FACE_BOX_END = FACE_BOX_START + 4  # 114
DT_IDX = FACE_BOX_END  # 114


def structure_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_body_conf: torch.Tensor = None,
    feature_weights: torch.Tensor = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Coordinate error on keypoints/head/face weighted by confidence.
    
    Args:
        pred: (B, D) or (B, T, D) predicted features
        target: (B, D) or (B, T, D) target features
        target_body_conf: (B, 17) or (B, T, 17) body keypoint confidences from target
        feature_weights: (D,) per-feature weights
        reduction: 'mean', 'sum', 'none'
    
    Returns:
        Scalar loss or per-sample loss
    """
    if feature_weights is None:
        feature_weights = get_feature_weights()
        feature_weights = torch.from_numpy(feature_weights).to(pred.device).float()
    
    # Weighted MSE
    diff = pred - target
    weighted_diff = diff * feature_weights.to(pred.device)
    loss = (weighted_diff ** 2).sum(dim=-1)  # (B,) or (B, T)
    
    if target_body_conf is not None:
        # Additional confidence weighting on body keypoints
        # target_body_conf: (B, 17) or (B*T, 17) -> expand to match body feature dim (51)
        body_conf_expanded = target_body_conf.repeat_interleave(3, dim=-1)  # (B, 51) or (B*T, 51)
        
        # Create per-sample feature weights for body portion
        # feature_weights is (D,), we need to expand to (B, D) or (B*T, D)
        batch_size = body_conf_expanded.shape[0]
        f_weights = feature_weights.unsqueeze(0).expand(batch_size, -1).clone().to(pred.device)  # (B, D)
        f_weights[:, BODY_START:BODY_END] = body_conf_expanded.to(pred.device)
        
        # Recompute weighted diff with per-sample weights
        weighted_diff = diff * f_weights
        loss = (weighted_diff ** 2).sum(dim=-1)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def velocity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_prev: torch.Tensor = None,
    pred_prev: torch.Tensor = None,
    dt: torch.Tensor = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Velocity loss: predicted speed vs target speed.
    
    Uses body keypoints to compute per-step displacement magnitude.
    
    Args:
        pred: (B, D) current prediction
        target: (B, D) current target
        target_prev: (B, D) previous target (for target velocity)
        pred_prev: (B, D) previous prediction (for predicted velocity)
        dt: (B,) time step
        reduction: 'mean', 'sum', 'none'
    """
    if pred_prev is None or target_prev is None:
        return torch.tensor(0.0, device=pred.device)
    
    # Extract body keypoints
    pred_body = pred[..., BODY_START:BODY_END].view(-1, NUM_BODY_POINTS, 3)
    target_body = target[..., BODY_START:BODY_END].view(-1, NUM_BODY_POINTS, 3)
    pred_prev_body = pred_prev[..., BODY_START:BODY_END].view(-1, NUM_BODY_POINTS, 3)
    target_prev_body = target_prev[..., BODY_START:BODY_END].view(-1, NUM_BODY_POINTS, 3)
    
    # Valid keypoints (conf >= 0.3)
    pred_conf = pred_body[..., 2] >= 0.3
    target_conf = target_body[..., 2] >= 0.3
    valid = pred_conf & target_conf
    
    if valid.sum() < 4:
        return torch.tensor(0.0, device=pred.device)
    
    # Per-point displacements
    pred_disp = pred_body[..., :2] - pred_prev_body[..., :2]
    target_disp = target_body[..., :2] - target_prev_body[..., :2]
    
    # Mean speed over valid points
    pred_speed = torch.norm(pred_disp, dim=-1)  # (B, 17)
    target_speed = torch.norm(target_disp, dim=-1)
    
    pred_speed = (pred_speed * valid.float()).sum(dim=-1) / valid.float().sum(dim=-1).clamp(min=1)
    target_speed = (target_speed * valid.float()).sum(dim=-1) / valid.float().sum(dim=-1).clamp(min=1)
    
    if dt is not None:
        pred_speed = pred_speed / dt.clamp(min=1e-3)
        target_speed = target_speed / dt.clamp(min=1e-3)
    
    loss = F.mse_loss(pred_speed, target_speed, reduction=reduction)
    return loss


def timing_loss(pred_dt: torch.Tensor, target_dt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Timing loss on dt prediction."""
    return F.mse_loss(pred_dt, target_dt, reduction=reduction)


def magnitude_regularizer(
    pred: torch.Tensor,
    target: torch.Tensor = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Magnitude regularizer: penalize excessive predicted motion magnitude.
    Only a regularizer, not a primary signal (spec 8).
    """
    pred_body = pred[..., BODY_START:BODY_END].view(-1, NUM_BODY_POINTS, 3)
    pred_conf = pred_body[..., 2] >= 0.3
    if pred_conf.sum() < 4:
        return torch.tensor(0.0, device=pred.device)
    
    # Mean displacement magnitude (assuming prev frame available)
    # This is a simple magnitude penalty on the predicted positions themselves
    # to prevent runaway predictions
    valid_positions = pred_body[..., :2][pred_conf]
    if valid_positions.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # Penalize deviation from center (encourage reasonable positions)
    center_penalty = ((valid_positions - 0.5) ** 2).mean()
    return center_penalty * 0.01  # Small weight as regularizer


def compute_all_losses(
    pred_feats: torch.Tensor,
    target_feats: torch.Tensor,
    pred_dt: torch.Tensor,
    target_dt: torch.Tensor,
    pred_prev_feats: torch.Tensor = None,
    target_prev_feats: torch.Tensor = None,
    feature_weights: torch.Tensor = None,
    loss_weights: dict = None,
) -> dict:
    """
    Compute all planner losses with weights.
    
    Args:
        pred_feats: (B, D) or (B, T, D)
        target_feats: (B, D) or (B, T, D)
        pred_dt: (B,) or (B, T)
        target_dt: (B,) or (B, T)
        pred_prev_feats: (B, D) previous prediction (for velocity)
        target_prev_feats: (B, D) previous target (for velocity)
        feature_weights: (D,) per-feature weights
        loss_weights: dict with keys 'structure', 'velocity', 'timing', 'magnitude'
    
    Returns:
        dict of individual losses and total
    """
    if loss_weights is None:
        loss_weights = {
            "structure": 1.0,
            "velocity": 0.5,
            "timing": 0.2,
            "magnitude": 0.01,
        }
    
    # Body confidence from target for structure loss weighting
    target_body_conf = target_feats[..., BODY_START:BODY_END].view(-1, NUM_BODY_POINTS, 3)[..., 2]
    
    L_struct = structure_loss(pred_feats, target_feats, target_body_conf, feature_weights)
    L_vel = velocity_loss(pred_feats, target_feats, target_prev_feats, pred_prev_feats, target_dt)
    L_time = timing_loss(pred_dt, target_dt)
    L_mag = magnitude_regularizer(pred_feats)
    
    total = (
        loss_weights["structure"] * L_struct +
        loss_weights["velocity"] * L_vel +
        loss_weights["timing"] * L_time +
        loss_weights["magnitude"] * L_mag
    )
    
    return {
        "total": total,
        "structure": L_struct,
        "velocity": L_vel,
        "timing": L_time,
        "magnitude": L_mag,
    }


if __name__ == "__main__":
    # Quick test
    B, D = 4, TOTAL_FEATURE_DIM
    pred = torch.randn(B, D)
    target = torch.randn(B, D)
    pred_dt = torch.rand(B) * 0.1 + 1/30.0
    target_dt = torch.rand(B) * 0.1 + 1/30.0
    
    losses = compute_all_losses(pred, target, pred_dt, target_dt)
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")