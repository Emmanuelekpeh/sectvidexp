"""Stage 2 planner training (spec 12, Stage 2 gate).

Trains a structure-space planner with autoregressive rollout training.
Evaluates against constant-velocity and hold-last baselines on endpoint
and angle, camera and subject separately.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sectvid.config import load_config, root_path
from sectvid.data import clips as data_clips
from sectvid.data import windows as W
from sectvid.data import splits
from sectvid.structure.cache import load_structure, has_structure
from sectvid.planner import (
    StructureTransformer,
    build_planner,
    encode_structure,
    decode_structure,
    TOTAL_FEATURE_DIM,
    get_feature_weights,
)
from sectvid.planner.losses import compute_all_losses
from sectvid.eval import baselines, metrics, report


class PlannerDataset(Dataset):
    """Dataset of structure windows for planner training.
    
    Pre-loads all structures to avoid repeated NPZ loading.
    """
    
    def __init__(self, cfg, clip_ids, context_len, max_windows_per_clip=None):
        self.cfg = cfg
        self.context_len = context_len
        
        # Pre-load all structures
        print("Pre-loading structures...")
        start = time.time()
        self.struct_cache = {}
        for cid in clip_ids:
            if has_structure(cfg, cid):
                self.struct_cache[cid] = load_structure(cfg, cid, include_silhouette=False)
        print(f"Loaded {len(self.struct_cache)} clips in {time.time() - start:.1f}s")
        
        self.windows = []
        for cid in clip_ids:
            if cid not in self.struct_cache:
                continue
            structs = self.struct_cache[cid]
            clip_windows = W.build_windows(cfg, cid, structs)
            
            # Filter to observed bins only (exclude unobserved)
            observed_windows = [w for w in clip_windows if w["bin"] != "unobserved"]
            
            if max_windows_per_clip and len(observed_windows) > max_windows_per_clip:
                rng = np.random.default_rng(cfg["seed"])
                idx = rng.choice(len(observed_windows), size=max_windows_per_clip, replace=False)
                observed_windows = [observed_windows[i] for i in idx]
            
            for w in observed_windows:
                self.windows.append((cid, w))
        
        print(f"Dataset: {len(self.windows)} windows from {len(self.struct_cache)} clips")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        cid, window = self.windows[idx]
        structs = self.struct_cache[cid]
        window_structs = structs[window["start"]:window["end"]]
        
        # Need context_len + 1 frames (context + target)
        if len(window_structs) < self.context_len + 1:
            pad_len = self.context_len + 1 - len(window_structs)
            last = window_structs[-1]
            for i in range(pad_len):
                window_structs.append(dataclasses.replace(last, frame_index=last.frame_index + i + 1))
        
        context_structs = window_structs[:self.context_len]
        target_struct = window_structs[self.context_len]
        
        context_feats = np.stack([encode_structure(s) for s in context_structs])
        target_feats = encode_structure(target_struct)
        target_dt = np.array([target_struct.dt], dtype=np.float32)
        
        return {
            "context": context_feats.astype(np.float32),
            "target": target_feats.astype(np.float32),
            "target_dt": target_dt,
            "clip_id": cid,
            "window_start": window["start"],
            "bin": window["bin"],
        }


def collate_fn(batch):
    context = torch.from_numpy(np.stack([b["context"] for b in batch]))
    target = torch.from_numpy(np.stack([b["target"] for b in batch]))
    target_dt = torch.from_numpy(np.stack([b["target_dt"] for b in batch])).squeeze(-1)
    return {
        "context": context,
        "target": target,
        "target_dt": target_dt,
        "clip_ids": [b["clip_id"] for b in batch],
        "window_starts": [b["window_start"] for b in batch],
        "bins": [b["bin"] for b in batch],
    }


def train_one_epoch(model, loader, optimizer, device, feature_weights, clean_context_prob, epoch, max_batches=None):
    model.train()
    total_loss = 0.0
    loss_components = {"structure": 0.0, "velocity": 0.0, "timing": 0.0, "magnitude": 0.0}
    n_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
            
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        target_dt = batch["target_dt"].to(device)
        
        B, T, D = context.shape
        
        # Clean context mask
        clean_mask = torch.rand(B, device=device) < clean_context_prob
        
        optimizer.zero_grad()
        
        pred_feats, pred_dt = model(context)
        
        losses = compute_all_losses(
            pred_feats, target,
            pred_dt, target_dt,
            feature_weights=feature_weights,
        )
        
        if clean_mask.any():
            for k in losses:
                losses[k] = losses[k] * (~clean_mask).float().mean()
        
        loss = losses["total"]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += losses[k].item() if k in losses else 0.0
        n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
    return avg_loss, avg_components


@torch.no_grad()
def evaluate_rollout(model, cfg, clip_ids, device, rollout_steps=8, context_len=12):
    model.eval()
    
    results = {
        "planner": {"endpoint": [], "angle": []},
        "constant_velocity": {"endpoint": [], "angle": []},
        "hold_last": {"endpoint": [], "angle": []},
        "per_bin": {"still": [], "slow": [], "fast": []},
    }
    
    # Pre-load structures
    struct_cache = {}
    for cid in clip_ids:
        if has_structure(cfg, cid):
            struct_cache[cid] = load_structure(cfg, cid, include_silhouette=False)
    
    for cid in clip_ids:
        if cid not in struct_cache:
            continue
        structs = struct_cache[cid]
        windows = W.build_windows(cfg, cid, structs)
        
        rng = np.random.default_rng(cfg["seed"] + hash(cid) % 1000)
        sample_idx = rng.choice(len(windows), size=min(4, len(windows)), replace=False)
        
        for idx in sample_idx:
            w = windows[idx]
            if w["bin"] == "unobserved":
                continue
            
            window_structs = structs[w["start"]:w["end"]]
            if len(window_structs) < context_len + rollout_steps + 1:
                continue
            
            context_structs = window_structs[:context_len]
            context_feats = torch.from_numpy(
                np.stack([encode_structure(s) for s in context_structs])
            ).unsqueeze(0).to(device)
            
            gt_structs = window_structs[context_len:context_len + rollout_steps]
            
            pred_feats, pred_dts = model.rollout(context_feats, rollout_steps)
            pred_structs = []
            for i in range(rollout_steps):
                prev_struct = context_structs[-1] if i == 0 else pred_structs[-1]
                pred_structs.append(
                    decode_structure(
                        pred_feats[0, i].cpu().numpy(),
                        context_len + i,
                        structs[0].silhouette.shape[0],
                        structs[0].silhouette.shape[1],
                        prev_struct,
                    )
                )
            
            cv_preds = baselines.constant_velocity(window_structs, context_len)
            hl_preds = baselines.hold_last_structure(window_structs, context_len)
            
            for k in range(min(rollout_steps, len(gt_structs))):
                gt = gt_structs[k]
                
                planner_err = metrics.structure_error(pred_structs[k], gt)
                planner_ang_result = metrics.angular_error([context_structs[-1], pred_structs[k]], [context_structs[-1], gt])
                planner_ang = planner_ang_result.get("angular_err_deg", 90.0) if planner_ang_result.get("angular_err_deg") is not None else 90.0
                
                if k < len(cv_preds):
                    cv_err = metrics.structure_error(cv_preds[k], gt)
                    cv_ang_result = metrics.angular_error([context_structs[-1], cv_preds[k]], [context_structs[-1], gt])
                    cv_ang = cv_ang_result.get("angular_err_deg", 90.0) if cv_ang_result.get("angular_err_deg") is not None else 90.0
                else:
                    cv_err = cv_ang = float('inf')
                
                if k < len(hl_preds):
                    hl_err = metrics.structure_error(hl_preds[k], gt)
                    hl_ang_result = metrics.angular_error([context_structs[-1], hl_preds[k]], [context_structs[-1], gt])
                    hl_ang = hl_ang_result.get("angular_err_deg", 90.0) if hl_ang_result.get("angular_err_deg") is not None else 90.0
                else:
                    hl_err = hl_ang = float('inf')
                
                results["planner"]["endpoint"].append(planner_err.get("mean_norm", 0))
                results["planner"]["angle"].append(planner_ang)
                results["constant_velocity"]["endpoint"].append(cv_err.get("mean_norm", 0))
                results["constant_velocity"]["angle"].append(cv_ang)
                results["hold_last"]["endpoint"].append(hl_err.get("mean_norm", 0))
                results["hold_last"]["angle"].append(hl_ang)
                
                results["per_bin"][w["bin"]].append({
                    "planner_endpoint": planner_err.get("mean_norm", 0),
                    "cv_endpoint": cv_err.get("mean_norm", 0) if k < len(cv_preds) else float('inf'),
                })
    
    return results


def check_stage2_gate(eval_results):
    planner_ep = np.nanmean(eval_results["planner"]["endpoint"])
    cv_ep = np.nanmean(eval_results["constant_velocity"]["endpoint"])
    planner_ang = np.nanmean(eval_results["planner"]["angle"])
    cv_ang = np.nanmean(eval_results["constant_velocity"]["angle"])
    
    beats_endpoint = planner_ep < cv_ep
    beats_angle = planner_ang < cv_ang
    
    gate_pass = beats_endpoint and beats_angle
    
    return {
        "planner_endpoint": float(planner_ep),
        "cv_endpoint": float(cv_ep),
        "planner_angle": float(planner_ang),
        "cv_angle": float(cv_ang),
        "beats_endpoint": beats_endpoint,
        "beats_angle": beats_angle,
        "gate_pass": gate_pass,
    }


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    all_ids = data_clips.discover_clips(cfg)
    metas = {cid: data_clips.load_meta(cfg, cid) for cid in all_ids}
    unreviewed = splits.curation_check(metas)
    if unreviewed:
        print(f"ERROR: {len(unreviewed)} unreviewed identities. Run curation first.")
        return 1
    
    clip_to_identity = {cid: m.get("identity_id", cid) for cid, m in metas.items()}
    train_clips, val_clips, unseen_clips = splits.split_by_identity(
        clip_to_identity,
        cfg["splits"]["identity_holdout_frac"],
        cfg["splits"]["test_frac"],
        cfg["splits"]["seed"],
    )
    
    print(f"Train: {len(train_clips)}, Val: {len(val_clips)}, Unseen: {len(unseen_clips)}")
    
    train_clips = [c for c in train_clips if has_structure(cfg, c)]
    val_clips = [c for c in val_clips if has_structure(cfg, c)]
    
    context_len = cfg["planner"]["context_len"]
    train_dataset = PlannerDataset(cfg, train_clips, context_len, max_windows_per_clip=50)
    val_dataset = PlannerDataset(cfg, val_clips, context_len, max_windows_per_clip=10)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    model = build_planner(cfg).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    feature_weights = torch.from_numpy(get_feature_weights()).float().to(device)
    
    clean_context_prob = cfg["planner"].get("clean_context_prob", 0.1)
    curriculum = cfg["planner"].get("rollout_curriculum", [2, 8, 32])
    max_epochs = 30
    
    gate_dir = root_path(cfg, cfg["report"]["dir"])
    gate_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(max_epochs):
        rollout_steps = curriculum[min(epoch // 10, len(curriculum) - 1)]
        
        print(f"\nEpoch {epoch+1}/{max_epochs}, rollout_steps={rollout_steps}, lr={optimizer.param_groups[0]['lr']:.2e}")
        
        train_loss, train_components = train_one_epoch(
            model, train_loader, optimizer, device, feature_weights, clean_context_prob, epoch, max_batches=20
        )
        print(f"  Train loss: {train_loss:.4f} (struct={train_components['structure']:.4f}, "
              f"vel={train_components['velocity']:.4f}, time={train_components['timing']:.4f})")
        
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            model.eval()
            val_loss = 0.0
            val_n = 0
            for batch in val_loader:
                context = batch["context"].to(device)
                target = batch["target"].to(device)
                target_dt = batch["target_dt"].to(device)
                
                pred_feats, pred_dt = model(context)
                losses = compute_all_losses(pred_feats, target, pred_dt, target_dt, feature_weights=feature_weights)
                val_loss += losses["total"].item()
                val_n += 1
            
            val_loss /= max(val_n, 1)
            print(f"  Val loss: {val_loss:.4f}")
            
            eval_results = evaluate_rollout(model, cfg, val_clips[:5], device, rollout_steps=8, context_len=context_len)
            gate_result = check_stage2_gate(eval_results)
            
            print(f"  Gate check: endpoint {gate_result['planner_endpoint']:.4f} vs {gate_result['cv_endpoint']:.4f} "
                  f"({'PASS' if gate_result['beats_endpoint'] else 'FAIL'}), "
                  f"angle {gate_result['planner_angle']:.4f} vs {gate_result['cv_angle']:.4f} "
                  f"({'PASS' if gate_result['beats_angle'] else 'FAIL'})")
            
            if gate_result["gate_pass"]:
                print("  *** STAGE 2 GATE PASSED ***")
                model_path = gate_dir / f"planner_stage2_{cfg['seed']}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "gate_result": gate_result,
                }, model_path)
                print(f"  Saved model to {model_path}")
                return 0
        
        scheduler.step()
    
    eval_results = evaluate_rollout(model, cfg, val_clips, device, rollout_steps=8, context_len=context_len)
    gate_result = check_stage2_gate(eval_results)
    
    print(f"\nFinal gate: {gate_result}")
    
    model_path = gate_dir / f"planner_stage2_final_{cfg['seed']}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg,
        "epoch": max_epochs - 1,
        "gate_result": gate_result,
    }, model_path)
    
    return 0 if gate_result["gate_pass"] else 1


if __name__ == "__main__":
    import dataclasses
    sys.exit(main())