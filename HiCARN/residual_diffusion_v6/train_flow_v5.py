#!/usr/bin/env python3
"""
Rectified Flow v5 - 完整修复版

关键改进（基于 v4 诊断）：
1. raw_res_corr 监控：分清"残差学没学到" vs "更新幅度太小"
2. 两阶段训练：Phase A 先学残差方向，Phase B 再加结构约束
3. 2-step unrolled recon：对齐训练和推理
4. 更大的有效更新幅度：alpha=0.15, g_scale=0.4

v4 问题诊断：
- res_corr ~0.01：模型没学到残差方向
- effective_step = alpha * gate ≈ 0.1 * 0.18 ≈ 0.018：太小
- 训练-推理不一致：单步 vs ODE 积分

修复策略：
- Phase A (epoch 1-10): lambda_res=10, lambda_insulation=0.05 (先学残差)
- Phase B (epoch 11+): 逐步加回结构约束
- 2-step unrolled recon 对齐推理
- raw_res_corr 监控真实残差学习情况
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import math
from scipy import stats


# ================================================================
# TAD/Loop Utilities
# ================================================================

def upper_triangle_mask(H, W, min_diag=2, device="cpu"):
    yy = torch.arange(H, device=device).view(H, 1)
    xx = torch.arange(W, device=device).view(1, W)
    return (xx - yy) >= min_diag


@torch.no_grad()
def topk_peak_mask(x, topk_ratio=0.01, min_diag=2):
    B, C, H, W = x.shape
    tri = upper_triangle_mask(H, W, min_diag=min_diag, device=x.device).view(1, 1, H, W)
    x_use = x.clone()
    x_use[~tri.expand_as(x_use)] = -1e9
    k = max(1, int(H * W * topk_ratio))
    flat = x_use.view(B, -1)
    idx = torch.topk(flat, k, dim=1).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(1, idx, 1.0)
    return mask.view(B, 1, H, W)


def insulation_vector(contact, window=5, eps=1e-6):
    B, C, H, W = contact.shape
    if torch.isnan(contact).any():
        return torch.zeros(B, H, device=contact.device)
    pad = window
    x = F.pad(contact[:, 0], (pad, pad, pad, pad), mode="constant", value=0.0)
    vals = []
    for i in range(H):
        block = x[:, i+pad-window:i+pad, i+pad:i+pad+window]
        vals.append(block.sum(dim=(1, 2)))
    ins = torch.stack(vals, dim=1)
    ins = torch.log1p(ins.clamp(min=0) + eps)
    return ins - ins.mean(dim=1, keepdim=True)


def insulation_loss(pred_gt, gt, window=5):
    return F.smooth_l1_loss(insulation_vector(pred_gt, window), insulation_vector(gt, window))


def anchor_loss(pred_gt, condition, peak_mask):
    diff = (pred_gt - condition).abs()
    return (diff * peak_mask).sum() / peak_mask.sum().clamp(min=1.0)


# ================================================================
# Gate Head
# ================================================================

class GateHead(nn.Module):
    def __init__(self, in_ch=1, hidden=16, g_scale=0.4):
        super().__init__()
        self.g_scale = g_scale
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
    
    def forward(self, condition):
        return torch.sigmoid(self.net(condition)) * self.g_scale


# ================================================================
# Flow Model
# ================================================================

def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class FlowUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, channel_mults=(1, 2, 4)):
        super().__init__()
        time_emb_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch = base_channels
        channels = [ch]
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.encoder.extend([ResBlock(ch, out_ch, time_emb_dim), ResBlock(out_ch, out_ch, time_emb_dim)])
            self.downsample.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            channels.append(out_ch)
            ch = out_ch
        
        self.mid1 = ResBlock(ch, ch, time_emb_dim)
        self.mid2 = ResBlock(ch, ch, time_emb_dim)
        
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            self.decoder.extend([ResBlock(ch + channels.pop(), out_ch, time_emb_dim), ResBlock(out_ch, out_ch, time_emb_dim)])
            ch = out_ch
        
        self.final_conv = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, padding=1))
    
    def forward(self, x, t):
        t_emb = self.time_mlp(get_timestep_embedding(t, self.init_conv.out_channels))
        h = self.init_conv(x)
        skips = [h]
        for i in range(0, len(self.encoder), 2):
            h = self.encoder[i](h, t_emb)
            h = self.encoder[i+1](h, t_emb)
            skips.append(h)
            h = self.downsample[i//2](h)
        h = self.mid2(self.mid1(h, t_emb), t_emb)
        for i in range(0, len(self.decoder), 2):
            h = self.upsample[i//2](h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = self.decoder[i](h, t_emb)
            h = self.decoder[i+1](h, t_emb)
        return self.final_conv(h)


# ================================================================
# Two-Phase Loss Schedule
# ================================================================

def get_phase_weights(epoch, phase_a_epochs=10):
    """
    Phase A (1-10): 先学残差方向
    Phase B (11+): 逐步加回结构约束
    """
    if epoch <= phase_a_epochs:
        # Phase A: 残差对齐
        progress = epoch / phase_a_epochs
        return {
            'lambda_res': 10.0,           # 高！
            'lambda_flow': 0.1,           # 低
            'recon_weight': 0.5,
            'lambda_insulation': 0.05,    # 很低
            'lambda_anchor': 0.8,
            'lambda_gate': 0.1,
            'lambda_gate_bg': 1.0,
        }
    else:
        # Phase B: 逐步加回结构约束
        progress = min(1.0, (epoch - phase_a_epochs) / 20)  # 20 epochs to full
        return {
            'lambda_res': 10.0 - 5.0 * progress,  # 10 -> 5
            'lambda_flow': 0.1 + 0.1 * progress,  # 0.1 -> 0.2
            'recon_weight': 0.5,
            'lambda_insulation': 0.05 + 0.45 * progress,  # 0.05 -> 0.5
            'lambda_anchor': 0.8 + 0.7 * progress,  # 0.8 -> 1.5
            'lambda_gate': 0.1 + 0.1 * progress,  # 0.1 -> 0.2
            'lambda_gate_bg': 1.0,
        }


# ================================================================
# Data Loading
# ================================================================

def ensure_nchw(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[:, np.newaxis, :, :]
    elif arr.ndim == 4:
        if arr.shape[1] == 1:
            return arr
        elif arr.shape[-1] == 1:
            return np.transpose(arr, (0, 3, 1, 2))
    raise ValueError(f"Cannot convert shape {arr.shape}")


def load_data(hicarn_path, gt_path):
    print(f"  Loading HiCARN: {hicarn_path}")
    hicarn = ensure_nchw(np.load(hicarn_path))
    print(f"    Shape: {hicarn.shape}, range [{hicarn.min():.4f}, {hicarn.max():.4f}]")
    print(f"  Loading GT: {gt_path}")
    gt = ensure_nchw(np.load(gt_path))
    print(f"    Shape: {gt.shape}, range [{gt.min():.4f}, {gt.max():.4f}]")
    return hicarn.astype(np.float32), gt.astype(np.float32)


# ================================================================
# 2-Step Unrolled ODE (for training-inference alignment)
# ================================================================

def unrolled_ode_2step(model, gate_head, condition, alpha, detach_intermediate=True):
    """
    2-step unrolled ODE to align training with inference.
    Returns pred after 2 ODE steps (t=0 -> t=0.5 -> t=1)
    """
    B = condition.shape[0]
    device = condition.device
    
    # Step 1: t=0 -> t=0.5
    t0 = torch.zeros(B, device=device)
    v0 = model(condition, t0)
    v0 = torch.clamp(v0, -10, 10)
    gate = gate_head(condition)
    
    x_half = condition + 0.5 * alpha * gate * v0
    if detach_intermediate:
        x_half = x_half.detach()  # Don't backprop through first step
    x_half = torch.clamp(x_half, -5, 5)
    
    # Step 2: t=0.5 -> t=1
    t_half = torch.full((B,), 0.5, device=device)
    v_half = model(x_half, t_half)
    v_half = torch.clamp(v_half, -10, 10)
    
    x_final = x_half + 0.5 * alpha * gate * v_half
    x_final = torch.clamp(x_final, -5, 5)
    
    return x_final


# ================================================================
# Training
# ================================================================

def train_epoch(model, gate_head, optimizer, dataloader, device, epoch, args):
    model.train()
    gate_head.train()
    
    # Get phase-specific weights
    weights = get_phase_weights(epoch, args.phase_a_epochs)
    
    totals = {k: 0.0 for k in ['loss', 'flow', 'res', 'recon', 'recon2', 'ins', 'anc', 'gate', 'gate_bg']}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for condition, gt in pbar:
        condition = condition.to(device)
        gt = gt.to(device)
        B = condition.shape[0]
        
        # Target residual
        target_residual = gt - condition
        
        # ============================================
        # 1. Flow prediction at random t
        # ============================================
        t = torch.rand(B, device=device)
        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * condition + t_expand * gt
        
        pred_residual = model(x_t, t)
        pred_residual = torch.clamp(pred_residual, -10, 10)
        
        # Flow loss (降权)
        flow_loss = F.mse_loss(pred_residual, target_residual)
        
        # ============================================
        # 2. Direct Residual Supervision (关键！)
        # ============================================
        res_loss = F.smooth_l1_loss(pred_residual, target_residual)
        
        # ============================================
        # 3. Gate mechanism
        # ============================================
        peak_mask = topk_peak_mask(condition, topk_ratio=args.gate_topk_ratio, min_diag=args.min_diag)
        gate = gate_head(condition)
        
        # Single-step reconstruction
        pred_gt = condition + args.alpha * gate * pred_residual
        pred_gt = torch.clamp(pred_gt, -5, 5)
        
        recon_loss = F.mse_loss(pred_gt, gt)
        
        # ============================================
        # 4. 2-Step Unrolled Recon (对齐推理)
        # ============================================
        pred_gt_2step = unrolled_ode_2step(model, gate_head, condition, args.alpha)
        recon_loss_2step = F.mse_loss(pred_gt_2step, gt)
        
        # ============================================
        # 5. Structure losses
        # ============================================
        ins_loss = insulation_loss(pred_gt, gt, window=args.insul_window)
        anc_loss = anchor_loss(pred_gt, condition, peak_mask)
        
        # Gate losses
        gate_prior = 1.0 - peak_mask
        gate_loss = F.mse_loss(gate / args.g_scale, gate_prior)
        
        bg_mask = 1.0 - peak_mask
        gate_bg_mean = (gate * bg_mask).sum() / bg_mask.sum().clamp(min=1.0)
        gate_bg_penalty = F.relu(gate_bg_mean - args.gate_bg_target)
        
        # ============================================
        # 6. Total loss with phase weights
        # ============================================
        loss = (
            weights['lambda_flow'] * flow_loss +
            weights['lambda_res'] * res_loss +
            weights['recon_weight'] * recon_loss +
            0.2 * recon_loss_2step +  # 2-step alignment loss
            weights['lambda_insulation'] * ins_loss +
            weights['lambda_anchor'] * anc_loss +
            weights['lambda_gate'] * gate_loss +
            weights['lambda_gate_bg'] * gate_bg_penalty
        )
        
        if torch.isnan(loss):
            continue
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(gate_head.parameters()), 1.0)
        optimizer.step()
        
        totals['loss'] += loss.item()
        totals['flow'] += flow_loss.item()
        totals['res'] += res_loss.item()
        totals['recon'] += recon_loss.item()
        totals['recon2'] += recon_loss_2step.item()
        totals['ins'] += ins_loss.item()
        totals['anc'] += anc_loss.item()
        totals['gate'] += gate_loss.item()
        totals['gate_bg'] += gate_bg_penalty.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'res': f'{res_loss.item():.4f}',
            'g_bg': f'{gate_bg_mean.item():.3f}'
        })
    
    return {k: v / max(num_batches, 1) for k, v in totals.items()}, weights


@torch.no_grad()
def sample_ode(model, gate_head, condition, device, num_steps, alpha):
    """ODE sampling"""
    B = condition.shape[0]
    x = condition.clone()
    gate = gate_head(condition)
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.full((B,), i / num_steps, device=device)
        v = torch.clamp(model(x, t), -10, 10)
        x = torch.clamp(x + dt * alpha * gate * v, -10, 10)
    
    return x


@torch.no_grad()
def validate(model, gate_head, hicarn_val, gt_val, device, num_steps, alpha, args, seed=42):
    model.eval()
    gate_head.eval()
    torch.manual_seed(seed)
    
    n = min(500, len(hicarn_val))
    condition = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    # Sample
    pred = torch.clamp(sample_ode(model, gate_head, condition, device, num_steps, alpha), -5, 5)
    
    if torch.isnan(pred).any():
        return {'mse': float('inf'), 'pcc': 0, 'raw_res_corr': 0, 'applied_res_corr': 0,
                'mse_hicarn': 0.07, 'pcc_hicarn': 0.95, 'ins_corr': 0, 'ins_corr_hicarn': 0,
                'gate_peaks': 0.5, 'gate_bg': 0.5, 'improved': False}
    
    # Basic metrics
    mse = F.mse_loss(pred, gt).item()
    mse_hicarn = F.mse_loss(condition, gt).item()
    
    pred_np = pred.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = condition.cpu().numpy().flatten()
    
    pcc, _ = stats.pearsonr(pred_np, gt_np)
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    # ============================================
    # RAW vs APPLIED res_corr (关键监控！)
    # ============================================
    target_residual = gt - condition
    applied_residual = pred - condition  # What we actually applied
    
    # Raw: model output at t=0 (not multiplied by gate/alpha)
    t0 = torch.zeros(n, device=device)
    raw_pred_residual = model(condition, t0)
    
    raw_res_corr, _ = stats.pearsonr(
        raw_pred_residual.cpu().numpy().flatten(),
        target_residual.cpu().numpy().flatten()
    )
    applied_res_corr, _ = stats.pearsonr(
        applied_residual.cpu().numpy().flatten(),
        target_residual.cpu().numpy().flatten()
    )
    
    # Insulation
    ins_pred = insulation_vector(pred, args.insul_window).cpu().numpy()
    ins_gt = insulation_vector(gt, args.insul_window).cpu().numpy()
    ins_hicarn = insulation_vector(condition, args.insul_window).cpu().numpy()
    ins_corr, _ = stats.pearsonr(ins_pred.flatten(), ins_gt.flatten())
    ins_corr_hicarn, _ = stats.pearsonr(ins_hicarn.flatten(), ins_gt.flatten())
    
    # Gate stats
    gate = gate_head(condition)
    peak_mask = topk_peak_mask(condition, args.gate_topk_ratio)
    gate_peaks = (gate * peak_mask).sum() / peak_mask.sum().clamp(min=1)
    gate_bg = (gate * (1 - peak_mask)).sum() / (1 - peak_mask).sum().clamp(min=1)
    
    return {
        'mse': mse,
        'pcc': float(pcc),
        'mse_hicarn': mse_hicarn,
        'pcc_hicarn': float(pcc_hicarn),
        'raw_res_corr': float(raw_res_corr),      # 关键！模型是否学对残差
        'applied_res_corr': float(applied_res_corr),  # 实际应用的残差相关性
        'ins_corr': float(ins_corr),
        'ins_corr_hicarn': float(ins_corr_hicarn),
        'gate_peaks': float(gate_peaks.item()),
        'gate_bg': float(gate_bg.item()),
        'improved': mse < mse_hicarn
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_hicarn', type=str, required=True)
    parser.add_argument('--train_gt', type=str, required=True)
    parser.add_argument('--val_hicarn', type=str, default=None)
    parser.add_argument('--val_gt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints_flow_v5')
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=20)
    
    # Key parameters (更激进的默认值)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--g_scale', type=float, default=0.4)
    parser.add_argument('--gate_bg_target', type=float, default=0.12)
    
    # Phase control
    parser.add_argument('--phase_a_epochs', type=int, default=10,
                       help='Epochs for Phase A (residual alignment)')
    
    # Gate/Anchor
    parser.add_argument('--gate_topk_ratio', type=float, default=0.015)
    parser.add_argument('--insul_window', type=int, default=5)
    parser.add_argument('--min_diag', type=int, default=2)
    
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    print("\n" + "="*80)
    print("RECTIFIED FLOW v5 - 两阶段训练 + 2-step 对齐")
    print("="*80)
    print(f"Phase A (epoch 1-{args.phase_a_epochs}): 残差对齐")
    print(f"  lambda_res=10, lambda_insulation=0.05")
    print(f"Phase B (epoch {args.phase_a_epochs+1}+): 结构约束逐步加回")
    print(f"关键参数:")
    print(f"  alpha={args.alpha}, g_scale={args.g_scale}")
    print(f"  gate_bg_target={args.gate_bg_target}")
    print(f"  num_steps={args.num_steps}")
    
    # Load data
    print("\nLoading data...")
    hicarn_train, gt_train = load_data(args.train_hicarn, args.train_gt)
    
    if args.val_hicarn and args.val_gt:
        hicarn_val, gt_val = load_data(args.val_hicarn, args.val_gt)
    else:
        split = int(len(hicarn_train) * 0.9)
        hicarn_val, gt_val = hicarn_train[split:], gt_train[split:]
        hicarn_train, gt_train = hicarn_train[:split], gt_train[:split]
        print(f"  Using 10% for validation")
    
    residual = gt_train - hicarn_train
    print(f"\nResidual: mean={residual.mean():.6f}, std={residual.std():.6f}")
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(hicarn_train).float(), torch.from_numpy(gt_train).float()),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    
    # Baseline
    print("\n" + "="*80)
    mse_baseline = np.mean((hicarn_val - gt_val) ** 2)
    pcc_baseline, _ = stats.pearsonr(hicarn_val.flatten(), gt_val.flatten())
    print(f"BASELINE: MSE={mse_baseline:.6f}, PCC={pcc_baseline:.4f}")
    
    # Model
    print("="*80)
    model = FlowUNet(base_channels=args.base_channels).to(device)
    gate_head = GateHead(g_scale=args.g_scale).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    
    optimizer = optim.AdamW(list(model.parameters()) + list(gate_head.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    
    # Resume
    start_epoch = 1
    best_mse, best_pcc = mse_baseline, pcc_baseline
    history = []
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'gate_head_state_dict' in ckpt:
            gate_head.load_state_dict(ckpt['gate_head_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_mse = ckpt.get('best_mse', mse_baseline)
        print(f"Resumed from epoch {start_epoch - 1}")
    
    # Training
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print("关键监控:")
    print("  raw_res_corr: 模型是否学对残差方向 (应该 >0.1)")
    print("  applied_res_corr: 实际应用效果")
    print("  如果 raw > applied: 步子太小")
    print("  如果 raw ≈ 0: 模型没学对")
    
    for epoch in range(start_epoch, args.epochs + 1):
        losses, weights = train_epoch(model, gate_head, optimizer, train_loader, device, epoch, args)
        scheduler.step()
        
        # Determine phase
        phase = "A (残差对齐)" if epoch <= args.phase_a_epochs else "B (结构约束)"
        
        if epoch % 3 == 0 or epoch == 1:
            val = validate(model, gate_head, hicarn_val, gt_val, device, args.num_steps, args.alpha, args)
            
            improved = ""
            if val['mse'] < best_mse:
                best_mse = val['mse']
                improved = " [best]"
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'gate_head_state_dict': gate_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mse': best_mse, 'val': val, 'args': vars(args)
                }, output_dir / 'best_model.pt')
            
            if val['pcc'] > best_pcc:
                best_pcc = val['pcc']
            
            status = "✓" if val['improved'] else "⚠"
            
            print(f"\n  Epoch {epoch} [{phase}]: {status}")
            print(f"    Losses: res={losses['res']:.4f}, recon={losses['recon']:.4f}, recon2={losses['recon2']:.4f}")
            print(f"    λ: res={weights['lambda_res']:.1f}, ins={weights['lambda_insulation']:.2f}")
            print(f"    MSE={val['mse']:.6f} (HiCARN:{val['mse_hicarn']:.6f})")
            print(f"    PCC={val['pcc']:.4f} (HiCARN:{val['pcc_hicarn']:.4f})")
            print(f"    raw_res_corr={val['raw_res_corr']:.4f} ← 模型学对了吗?")
            print(f"    applied_res_corr={val['applied_res_corr']:.4f} ← 实际效果")
            print(f"    ins_corr={val['ins_corr']:.4f} (HiCARN:{val['ins_corr_hicarn']:.4f})")
            print(f"    gate: peaks={val['gate_peaks']:.3f}, bg={val['gate_bg']:.3f}")
            
            # Diagnostics
            if val['raw_res_corr'] > 0.1:
                print(f"    ✓ raw_res_corr > 0.1, 模型在学对残差方向!")
            if val['raw_res_corr'] > val['applied_res_corr'] + 0.05:
                print(f"    ⚠ raw > applied: 可以增大 alpha 或 g_scale")
            if val['raw_res_corr'] < 0.05 and epoch > 5:
                print(f"    ❌ raw_res_corr 很低，模型没学对!")
            
            print(f"    {improved}")
            
            history.append({'epoch': epoch, 'losses': losses, 'val': val, 'phase': phase})
        
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'gate_head_state_dict': gate_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mse': best_mse
            }, output_dir / f'checkpoint_{epoch}.pt')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"HiCARN: MSE={mse_baseline:.6f}, PCC={pcc_baseline:.4f}")
    print(f"Best:   MSE={best_mse:.6f}, PCC={best_pcc:.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()
