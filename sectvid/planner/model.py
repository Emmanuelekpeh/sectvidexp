"""Planner model: causal transformer over structure tokens (spec 6.3).

Input: sequence of encoded structures (context_len frames)
Output: next structure prediction (same feature dim), next dt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .structure_codec import TOTAL_FEATURE_DIM, get_feature_weights


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for sequence positions."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :]


class StructureTransformer(nn.Module):
    """Causal transformer for structure autoregression."""
    
    def __init__(
        self,
        feature_dim: int = TOTAL_FEATURE_DIM,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_context: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.max_context = max_context
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(max_context, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_context, max_context, dtype=torch.bool), diagonal=1),
            persistent=False
        )
        
        # Output heads
        self.output_proj = nn.Linear(d_model, feature_dim)
        self.dt_head = nn.Linear(d_model, 1)
        
        # Feature loss weights (registered as buffer for device placement)
        self.register_buffer(
            "feature_weights",
            torch.from_numpy(get_feature_weights()).float(),
            persistent=False
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        clean_context_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) input structure features
            clean_context_mask: (B, T) boolean, True where real structure is fed (excluded from loss)
        
        Returns:
            pred_feats: (B, D) next structure prediction
            pred_dt: (B, 1) next dt prediction
        """
        B, T, D = x.shape
        assert D == self.feature_dim
        
        # Project and add positional encoding
        h = self.input_proj(x)
        h = self.pos_enc(h)
        
        # Apply causal mask
        mask = self.causal_mask[:T, :T]
        h = self.transformer(h, mask=mask)
        
        # Predict next step from last context token
        last_h = h[:, -1, :]  # (B, D)
        
        pred_feats = self.output_proj(last_h)  # (B, D)
        pred_dt = F.softplus(self.dt_head(last_h)) + 1.0 / 60.0  # Positive dt, min 1/60
        
        return pred_feats, pred_dt.squeeze(-1)
    
    def rollout(
        self,
        context: torch.Tensor,
        steps: int,
        clean_context_prob: float = 0.0,
        rng: torch.Generator = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive rollout on own predictions.
        
        Args:
            context: (B, T, D) initial context
            steps: number of steps to roll out
            clean_context_prob: probability of feeding real structure (FAR clean context)
            rng: random generator for clean context sampling
        
        Returns:
            preds: (B, steps, D) predicted features
            dts: (B, steps) predicted dts
        """
        B, T, D = context.shape
        device = context.device
        
        preds = []
        dts = []
        curr_context = context.clone()
        
        for step in range(steps):
            pred_feat, pred_dt = self.forward(curr_context)
            preds.append(pred_feat)
            dts.append(pred_dt)
            
            # Build next context: shift left, append prediction
            next_frame = pred_feat.unsqueeze(1)  # (B, 1, D)
            curr_context = torch.cat([curr_context[:, 1:, :], next_frame], dim=1)
            
            # Clean context: with probability, replace with real (not implemented here,
            # would need ground truth; handled in training loop)
        
        return torch.stack(preds, dim=1), torch.stack(dts, dim=1)


def build_planner(cfg) -> StructureTransformer:
    """Build planner from config."""
    return StructureTransformer(
        feature_dim=TOTAL_FEATURE_DIM,
        d_model=cfg.get("planner", {}).get("d_model", 256),
        n_heads=cfg.get("planner", {}).get("n_heads", 4),
        n_layers=cfg.get("planner", {}).get("n_layers", 4),
        d_ff=cfg.get("planner", {}).get("d_ff", 1024),
        max_context=cfg.get("planner", {}).get("context_len", 12),
        dropout=cfg.get("planner", {}).get("dropout", 0.1),
    )


if __name__ == "__main__":
    # Quick test
    model = StructureTransformer()
    B, T, D = 2, 12, TOTAL_FEATURE_DIM
    x = torch.randn(B, T, D)
    pred_feat, pred_dt = model(x)
    print(f"Pred feat: {pred_feat.shape}, pred dt: {pred_dt.shape}")
    
    # Test rollout
    preds, dts = model.rollout(x, steps=4)
    print(f"Rollout preds: {preds.shape}, dts: {dts.shape}")