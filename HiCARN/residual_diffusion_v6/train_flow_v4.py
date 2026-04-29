#!/usr/bin/env python3
"""
Rectified Flow v4 - 修复版

关键修复（基于诊断）：
1. 添加 alpha 参数控制残差幅度：pred_gt = condition + alpha * gate * pred_residual
2. 添加 Direct Residual Supervision：L_res = || pred_residual - target_residual ||
3. 添加 Gate BG Penalty：惩罚背景区域 gate 太大
4. 添加 g_scale 全局缩放 gate
5. 降低 flow_loss 权重，提高 recon 和 residual supervision

问题诊断：
- res_corr ≈ 0：模型没学到正确的残差方向
- MSE 比 HiCARN 差 27%：残差更新太 aggressive
- gate bg ≈ 0.98：几乎全开，等于没有保护

修复策略：
- alpha 控制残差幅度（默认 0.1，很保守）
- lambda_res 直接监督残差（关键！）
- gate_bg_penalty 压低背景区域的 gate
- g_scale 全局缩放 gate 输出
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
# TAD/Loop Preservation Utilities
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
    assert C == 1
    
    if torch.isnan(contact).any():
        return torch.zeros(B, H, device=contact.device)
    
    pad = window
    x = F.pad(contact[:, 0], (pad, pad, pad, pad), mode="constant", value=0.0)
    
    vals = []
    for i in range(H):
        r0 = i + pad - window
        r1 = i + pad
        c0 = i + pad
        c1 = i + pad + window
        block = x[:, r0:r1, c0:c1]
        s = block.sum(dim=(1, 2))
        vals.append(s)
    
    ins = torch.stack(vals, dim=1)
    ins = torch.log1p(ins.clamp(min=0) + eps)
    ins_mean = ins.mean(dim=1, keepdim=True)
    ins = ins - ins_mean
    return ins


def insulation_loss(pred_gt, gt, window=5):
    ip = insulation_vector(pred_gt, window=window)
    ig = insulation_vector(gt, window=window)
    return F.smooth_l1_loss(ip, ig)


def anchor_loss(pred_gt, condition, peak_mask, reduction="mean"):
    diff = (pred_gt - condition).abs()
    loss = diff * peak_mask
    if reduction == "mean":
        denom = peak_mask.sum().clamp(min=1.0)
        return loss.sum() / denom
    return loss.mean()


# ================================================================
# Gate Head with Global Scale
# ================================================================

class GateHead(nn.Module):
    """
    Gate with global scale to prevent bg gate from being too high.
    """
    def __init__(self, in_ch=1, hidden=16, g_scale=0.3):
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
        # Sigmoid * g_scale to limit max gate value
        return torch.sigmoid(self.net(condition)) * self.g_scale


# ================================================================
# Time Embedding & Model
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
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.shortcut(x)


class FlowUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=256
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        ch = base_channels
        channels = [ch]
        
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.encoder.append(ResBlock(ch, out_ch, time_emb_dim))
            self.encoder.append(ResBlock(out_ch, out_ch, time_emb_dim))
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
            self.decoder.append(ResBlock(ch + channels.pop(), out_ch, time_emb_dim))
            self.decoder.append(ResBlock(out_ch, out_ch, time_emb_dim))
            ch = out_ch
        
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.init_conv.out_channels)
        t_emb = self.time_mlp(t_emb)
        
        h = self.init_conv(x)
        
        skips = [h]
        for i in range(0, len(self.encoder), 2):
            h = self.encoder[i](h, t_emb)
            h = self.encoder[i+1](h, t_emb)
            skips.append(h)
            h = self.downsample[i//2](h)
        
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        
        for i in range(0, len(self.decoder), 2):
            h = self.upsample[i//2](h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = self.decoder[i](h, t_emb)
            h = self.decoder[i+1](h, t_emb)
        
        return self.final_conv(h)


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
    raise ValueError(f"Cannot convert shape {arr.shape} to NCHW")


def load_data(hicarn_path, gt_path):
    print(f"  Loading HiCARN: {hicarn_path}")
    hicarn = np.load(hicarn_path)
    hicarn = ensure_nchw(hicarn)
    print(f"    Shape: {hicarn.shape}, range [{hicarn.min():.4f}, {hicarn.max():.4f}]")
    
    print(f"  Loading GT: {gt_path}")
    gt = np.load(gt_path)
    gt = ensure_nchw(gt)
    print(f"    Shape: {gt.shape}, range [{gt.min():.4f}, {gt.max():.4f}]")
    
    assert hicarn.shape == gt.shape
    return hicarn.astype(np.float32), gt.astype(np.float32)


# ================================================================
# Training v4 with Residual Supervision
# ================================================================

def train_epoch(
    model,
    gate_head,
    optimizer,
    dataloader,
    device,
    epoch,
    args
):
    """
    Training with:
    1. alpha-scaled gated residual
    2. Direct residual supervision (关键！)
    3. Gate BG penalty
    """
    model.train()
    gate_head.train()
    
    total_loss = 0
    total_flow = 0
    total_recon = 0
    total_res = 0
    total_ins = 0
    total_anc = 0
    total_gate = 0
    total_gate_bg = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for condition, gt in pbar:
        condition = condition.to(device)
        gt = gt.to(device)
        
        batch_size = condition.shape[0]
        
        # ============================================
        # 1. Sample time and compute x_t
        # ============================================
        t = torch.rand(batch_size, device=device)
        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * condition + t_expand * gt
        
        # Target velocity = residual
        target_residual = gt - condition
        
        # ============================================
        # 2. Predict velocity/residual
        # ============================================
        pred_residual = model(x_t, t)
        pred_residual = torch.clamp(pred_residual, -10, 10)
        
        # Flow loss (降权)
        flow_loss = F.mse_loss(pred_residual, target_residual)
        
        # ============================================
        # 3. Direct Residual Supervision (关键！)
        # ============================================
        res_loss = F.smooth_l1_loss(pred_residual, target_residual)
        
        # ============================================
        # 4. Gate mechanism with alpha scaling
        # ============================================
        peak_mask = topk_peak_mask(
            condition, 
            topk_ratio=args.gate_topk_ratio,
            min_diag=args.min_diag
        )
        
        gate = gate_head(condition)  # Already scaled by g_scale
        
        # Alpha-scaled gated reconstruction
        # pred_gt = condition + alpha * gate * pred_residual
        pred_gt = condition + args.alpha * gate * pred_residual
        pred_gt = torch.clamp(pred_gt, -5, 5)
        
        # ============================================
        # 5. Reconstruction loss
        # ============================================
        recon_loss = F.mse_loss(pred_gt, gt)
        
        # ============================================
        # 6. TAD/Loop preservation losses
        # ============================================
        ins_loss = insulation_loss(pred_gt, gt, window=args.insul_window)
        anc_loss = anchor_loss(pred_gt, condition, peak_mask)
        
        # Gate loss (peaks should be low)
        gate_prior = 1.0 - peak_mask
        gate_loss = F.mse_loss(gate / args.g_scale, gate_prior)  # Normalized
        
        # ============================================
        # 7. Gate BG Penalty (关键！压低背景 gate)
        # ============================================
        bg_mask = 1.0 - peak_mask
        gate_bg_mean = (gate * bg_mask).sum() / bg_mask.sum().clamp(min=1.0)
        # Penalize if bg gate > target (e.g., 0.15)
        gate_bg_penalty = F.relu(gate_bg_mean - args.gate_bg_target)
        
        # ============================================
        # 8. Total loss
        # ============================================
        loss = (
            args.lambda_flow * flow_loss +
            args.lambda_res * res_loss +          # 关键！
            args.recon_weight * recon_loss +
            args.lambda_insulation * ins_loss +
            args.lambda_anchor * anc_loss +
            args.lambda_gate * gate_loss +
            args.lambda_gate_bg * gate_bg_penalty  # 关键！
        )
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ NaN detected! Skipping batch.")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(gate_head.parameters()), 
            1.0
        )
        
        if torch.isnan(grad_norm):
            print(f"\n⚠️ NaN gradient! Skipping.")
            continue
        
        optimizer.step()
        
        total_loss += loss.item()
        total_flow += flow_loss.item()
        total_recon += recon_loss.item()
        total_res += res_loss.item()
        total_ins += ins_loss.item()
        total_anc += anc_loss.item()
        total_gate += gate_loss.item()
        total_gate_bg += gate_bg_penalty.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'res': f'{res_loss.item():.4f}',
            'g_bg': f'{gate_bg_mean.item():.3f}'
        })
    
    return {
        'total': total_loss / max(num_batches, 1),
        'flow': total_flow / max(num_batches, 1),
        'recon': total_recon / max(num_batches, 1),
        'residual': total_res / max(num_batches, 1),
        'insulation': total_ins / max(num_batches, 1),
        'anchor': total_anc / max(num_batches, 1),
        'gate': total_gate / max(num_batches, 1),
        'gate_bg': total_gate_bg / max(num_batches, 1)
    }


@torch.no_grad()
def sample_ode(model, gate_head, condition, device, num_steps, alpha):
    """ODE sampling with alpha scaling"""
    batch_size = condition.shape[0]
    
    x = condition.clone()
    gate = gate_head(condition)
    gate = torch.clamp(gate, 0, 1)
    
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = i / num_steps
        t_batch = torch.full((batch_size,), t, device=device)
        
        v = model(x, t_batch)
        v = torch.clamp(v, -10, 10)
        
        # Alpha-scaled gated step
        x = x + dt * alpha * gate * v
        x = torch.clamp(x, -10, 10)
    
    return x


@torch.no_grad()
def validate(model, gate_head, hicarn_val, gt_val, device, num_steps, alpha, args, seed=42):
    model.eval()
    gate_head.eval()
    torch.manual_seed(seed)
    
    n = min(500, len(hicarn_val))
    condition = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    pred = sample_ode(model, gate_head, condition, device, num_steps, alpha)
    pred = torch.clamp(pred, -5, 5)
    
    if torch.isnan(pred).any():
        return {
            'mse': float('inf'), 'pcc': 0.0, 'mse_hicarn': 0.07,
            'pcc_hicarn': 0.95, 'res_corr': 0.0, 'ins_corr': 0.0,
            'ins_corr_hicarn': 0.0, 'gate_on_peaks': 0.5,
            'gate_off_peaks': 0.5, 'improved': False
        }
    
    mse = F.mse_loss(pred, gt).item()
    mse_hicarn = F.mse_loss(condition, gt).item()
    
    pred_np = pred.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = condition.cpu().numpy().flatten()
    
    if np.isnan(pred_np).any():
        pcc, pcc_hicarn = 0.0, 0.95
    else:
        pcc, _ = stats.pearsonr(pred_np, gt_np)
        pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    pred_residual = (pred - condition).cpu().numpy()
    ideal_residual = (gt - condition).cpu().numpy()
    
    if np.isnan(pred_residual).any():
        res_corr = 0.0
    else:
        res_corr, _ = stats.pearsonr(pred_residual.flatten(), ideal_residual.flatten())
    
    window = args.insul_window
    ins_pred = insulation_vector(pred, window=window).cpu().numpy()
    ins_gt = insulation_vector(gt, window=window).cpu().numpy()
    ins_hicarn = insulation_vector(condition, window=window).cpu().numpy()
    
    if np.isnan(ins_pred).any():
        ins_corr, ins_corr_hicarn = 0.0, 0.0
    else:
        ins_corr, _ = stats.pearsonr(ins_pred.flatten(), ins_gt.flatten())
        ins_corr_hicarn, _ = stats.pearsonr(ins_hicarn.flatten(), ins_gt.flatten())
    
    gate = gate_head(condition)
    peak_mask = topk_peak_mask(condition, topk_ratio=args.gate_topk_ratio)
    gate_on_peaks = (gate * peak_mask).sum() / peak_mask.sum().clamp(min=1.0)
    gate_off_peaks = (gate * (1 - peak_mask)).sum() / (1 - peak_mask).sum().clamp(min=1.0)
    
    return {
        'mse': mse,
        'pcc': float(pcc),
        'mse_hicarn': mse_hicarn,
        'pcc_hicarn': float(pcc_hicarn),
        'res_corr': float(res_corr),
        'ins_corr': float(ins_corr),
        'ins_corr_hicarn': float(ins_corr_hicarn),
        'gate_on_peaks': float(gate_on_peaks.item()),
        'gate_off_peaks': float(gate_off_peaks.item()),
        'improved': mse < mse_hicarn
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--train_hicarn', type=str, required=True)
    parser.add_argument('--train_gt', type=str, required=True)
    parser.add_argument('--val_hicarn', type=str, default=None)
    parser.add_argument('--val_gt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints_flow_v4')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=10)
    
    # 关键参数
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Residual scaling (保守值 0.05-0.2)')
    parser.add_argument('--g_scale', type=float, default=0.3,
                       help='Global gate scale (0.1-0.5)')
    parser.add_argument('--gate_bg_target', type=float, default=0.15,
                       help='Target for background gate (0.1-0.3)')
    
    # Loss weights
    parser.add_argument('--lambda_flow', type=float, default=0.3,
                       help='Flow loss weight (降低)')
    parser.add_argument('--lambda_res', type=float, default=2.0,
                       help='Residual supervision weight (关键！)')
    parser.add_argument('--recon_weight', type=float, default=0.4,
                       help='Reconstruction loss weight (提高)')
    parser.add_argument('--lambda_insulation', type=float, default=0.2,
                       help='Insulation loss (先降低)')
    parser.add_argument('--lambda_anchor', type=float, default=1.5,
                       help='Anchor loss')
    parser.add_argument('--lambda_gate', type=float, default=0.2,
                       help='Gate loss')
    parser.add_argument('--lambda_gate_bg', type=float, default=0.5,
                       help='Gate BG penalty (关键！)')
    
    # Gate/Anchor
    parser.add_argument('--gate_topk_ratio', type=float, default=0.015)
    parser.add_argument('--insul_window', type=int, default=5)
    parser.add_argument('--min_diag', type=int, default=2)
    
    # Resume
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Print config
    # ================================================================
    print("\n" + "="*80)
    print("RECTIFIED FLOW v4 - 修复版")
    print("="*80)
    print(f"关键参数:")
    print(f"  alpha:          {args.alpha} (残差幅度控制)")
    print(f"  g_scale:        {args.g_scale} (gate 全局缩放)")
    print(f"  gate_bg_target: {args.gate_bg_target} (背景 gate 目标)")
    print(f"Loss weights:")
    print(f"  lambda_flow:    {args.lambda_flow} (降低)")
    print(f"  lambda_res:     {args.lambda_res} (关键！)")
    print(f"  recon_weight:   {args.recon_weight} (提高)")
    print(f"  lambda_gate_bg: {args.lambda_gate_bg} (关键！)")
    
    # ================================================================
    # Load data
    # ================================================================
    print("\nLoading data...")
    hicarn_train, gt_train = load_data(args.train_hicarn, args.train_gt)
    
    if args.val_hicarn and args.val_gt:
        hicarn_val, gt_val = load_data(args.val_hicarn, args.val_gt)
    else:
        split = int(len(hicarn_train) * 0.9)
        hicarn_val = hicarn_train[split:]
        gt_val = gt_train[split:]
        hicarn_train = hicarn_train[:split]
        gt_train = gt_train[:split]
        print(f"  Using 10% for validation")
    
    residual = gt_train - hicarn_train
    print(f"\nResidual (GT - HiCARN):")
    print(f"  Mean: {residual.mean():.6f}")
    print(f"  Std:  {residual.std():.6f}")
    
    train_dataset = TensorDataset(
        torch.from_numpy(hicarn_train).float(),
        torch.from_numpy(gt_train).float()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    # ================================================================
    # Baseline
    # ================================================================
    print("\n" + "="*80)
    print("BASELINE")
    print("="*80)
    
    mse_baseline = np.mean((hicarn_val - gt_val) ** 2)
    pcc_baseline, _ = stats.pearsonr(hicarn_val.flatten(), gt_val.flatten())
    print(f"HiCARN: MSE={mse_baseline:.6f}, PCC={pcc_baseline:.4f}")
    
    # ================================================================
    # Model
    # ================================================================
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = FlowUNet(
        in_channels=1, out_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4)
    ).to(device)
    
    gate_head = GateHead(in_ch=1, hidden=16, g_scale=args.g_scale).to(device)
    
    print(f"Flow model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Gate head:  {sum(p.numel() for p in gate_head.parameters()) / 1e3:.1f}K")
    
    optimizer = optim.AdamW(
        list(model.parameters()) + list(gate_head.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Resume
    start_epoch = 1
    best_mse = mse_baseline
    best_pcc = pcc_baseline
    history = []
    
    if args.resume:
        print(f"\n📂 Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'gate_head_state_dict' in ckpt:
            gate_head.load_state_dict(ckpt['gate_head_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_mse = ckpt.get('best_mse', mse_baseline)
        best_pcc = ckpt.get('best_pcc', pcc_baseline)
        print(f"  Resumed from epoch {start_epoch - 1}")
    
    # ================================================================
    # Training
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print("期望看到:")
    print("  - res_corr 快速上升 (>0.1 in first epochs)")
    print("  - gate_bg < gate_bg_target")
    print("  - MSE 接近或好于 HiCARN")
    
    for epoch in range(start_epoch, args.epochs + 1):
        losses = train_epoch(
            model, gate_head, optimizer, train_loader,
            device, epoch, args
        )
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1:
            val = validate(
                model, gate_head, hicarn_val, gt_val,
                device, args.num_steps, args.alpha, args
            )
            
            improved = ""
            if val['mse'] < best_mse:
                best_mse = val['mse']
                improved += " [best MSE]"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'gate_head_state_dict': gate_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mse': best_mse,
                    'best_pcc': best_pcc,
                    'val_metrics': val,
                    'args': vars(args)
                }, output_dir / 'best_model.pt')
            
            if val['pcc'] > best_pcc:
                best_pcc = val['pcc']
                improved += " [best PCC]"
            
            status = "✓" if val['improved'] else "⚠"
            
            print(f"\n  Epoch {epoch}: {status}")
            print(f"    Losses: flow={losses['flow']:.4f}, res={losses['residual']:.4f}, g_bg={losses['gate_bg']:.4f}")
            print(f"    MSE={val['mse']:.6f} (HiCARN:{val['mse_hicarn']:.6f})")
            print(f"    PCC={val['pcc']:.4f} (HiCARN:{val['pcc_hicarn']:.4f})")
            print(f"    res_corr={val['res_corr']:.4f} ← 关键！应该 >0.1")
            print(f"    ins_corr={val['ins_corr']:.4f} (HiCARN:{val['ins_corr_hicarn']:.4f})")
            print(f"    gate: peaks={val['gate_on_peaks']:.3f}, bg={val['gate_off_peaks']:.3f}")
            
            if val['res_corr'] > 0.1:
                print(f"    ✓ res_corr > 0.1, 模型在学习正确方向！")
            if val['gate_off_peaks'] < args.gate_bg_target + 0.1:
                print(f"    ✓ gate_bg 被控制住了")
            
            if improved:
                print(f"    {improved}")
            
            history.append({'epoch': epoch, 'losses': losses, 'val': val})
        
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'gate_head_state_dict': gate_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mse': best_mse,
                'best_pcc': best_pcc
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'gate_head_state_dict': gate_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mse': best_mse,
                'best_pcc': best_pcc
            }, output_dir / 'latest_checkpoint.pt')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"HiCARN: MSE={mse_baseline:.6f}, PCC={pcc_baseline:.4f}")
    print(f"Best:   MSE={best_mse:.6f}, PCC={best_pcc:.4f}")
    
    if best_mse < mse_baseline:
        print("✓ SUCCESS!")
    else:
        print("⚠ Did not improve")
    
    print(f"\nSaved to: {output_dir}")


if __name__ == '__main__':
    main()
