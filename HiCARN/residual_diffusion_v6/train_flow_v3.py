#!/usr/bin/env python3
"""
Rectified Flow v3 with TAD/Loop Preservation

核心改进（基于评论）：
1. Insulation Score Loss - 推动 TAD 结构对齐
2. Gate/Anchor 机制 - 保护 loop peaks 不被乱改
3. Peak/HF Loss - 高频结构和峰位一致性

数学框架（Rectified Flow: HiCARN -> GT）：
- x_0 = HiCARN (起点)
- x_1 = GT (终点)
- x_t = (1-t) * HiCARN + t * GT
- velocity = GT - HiCARN

关键改动：
- pred_gt = condition + gate * pred_residual (gate 控制哪里改、哪里不改)
- anchor loss 保护 HiCARN 的 top peaks (loop)
- insulation loss 对齐 TAD 边界结构
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
    """Keep only upper triangle with offset from diagonal"""
    yy = torch.arange(H, device=device).view(H, 1)
    xx = torch.arange(W, device=device).view(1, W)
    return (xx - yy) >= min_diag


@torch.no_grad()
def topk_peak_mask(x, topk_ratio=0.01, min_diag=2):
    """
    Find top-k peaks in upper triangle (these are potential loops to anchor).
    
    x: [B, 1, H, W] - use condition (HiCARN) to define anchors
    return: mask [B, 1, H, W] in {0, 1}
    """
    B, C, H, W = x.shape
    tri = upper_triangle_mask(H, W, min_diag=min_diag, device=x.device).view(1, 1, H, W)
    x_use = x.clone()
    x_use[~tri.expand_as(x_use)] = -1e9
    
    k = max(1, int(H * W * topk_ratio))
    flat = x_use.view(B, -1)
    idx = torch.topk(flat, k, dim=1).indices  # [B, k]
    mask = torch.zeros_like(flat)
    mask.scatter_(1, idx, 1.0)
    return mask.view(B, 1, H, W)


def insulation_vector(contact, window=5, eps=1e-6):
    """
    Compute differentiable insulation score (TAD boundary indicator).
    
    For each diagonal bin i, sum a window x window square that crosses the diagonal:
      rows: [i-window, i)
      cols: [i, i+window)
    
    contact: [B, 1, H, W]
    return: [B, H] (log1p normalized)
    """
    B, C, H, W = contact.shape
    assert C == 1
    
    # Check for NaN
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
    
    ins = torch.stack(vals, dim=1)  # [B, H]
    
    # Log + per-sample normalize (stability across chromosomes)
    ins = torch.log1p(ins.clamp(min=0) + eps)
    ins_mean = ins.mean(dim=1, keepdim=True)
    ins = ins - ins_mean
    
    return ins


def insulation_loss(pred_gt, gt, window=5):
    """Push TAD structure alignment via insulation score matching"""
    ip = insulation_vector(pred_gt, window=window)
    ig = insulation_vector(gt, window=window)
    return F.smooth_l1_loss(ip, ig)


def anchor_loss(pred_gt, condition, peak_mask, reduction="mean"):
    """
    Encourage: at anchor peaks, do NOT deviate from condition (protect loops).
    """
    diff = (pred_gt - condition).abs()
    loss = diff * peak_mask
    if reduction == "mean":
        denom = peak_mask.sum().clamp(min=1.0)
        return loss.sum() / denom
    return loss.mean()


def frequency_loss(pred, gt, sigma=2.0):
    """
    Separate low/high frequency and weight them differently.
    Low freq: Gaussian blur
    High freq: Original - Low freq
    """
    # Check for NaN/Inf
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        return torch.tensor(0.0, device=pred.device)
    
    # Create Gaussian kernel
    kernel_size = int(6 * sigma + 1) | 1  # Ensure odd
    x = torch.arange(kernel_size, device=pred.device, dtype=pred.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / (kernel_1d.sum() + 1e-8)
    kernel_2d = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
    
    # Low frequency (blur)
    pad = kernel_size // 2
    pred_low = F.conv2d(F.pad(pred, (pad, pad, pad, pad), mode='reflect'), kernel_2d)
    gt_low = F.conv2d(F.pad(gt, (pad, pad, pad, pad), mode='reflect'), kernel_2d)
    
    # High frequency (detail)
    pred_high = pred - pred_low
    gt_high = gt - gt_low
    
    # Combined loss (weight high freq more)
    loss_low = F.mse_loss(pred_low, gt_low)
    loss_high = F.mse_loss(pred_high, gt_high)
    
    return loss_low + 2.0 * loss_high


# ================================================================
# Gate Head (learnable gate for residual application)
# ================================================================

class GateHead(nn.Module):
    """
    Predict a gate map in [0,1] that decides where residual is applied.
    Input: condition (HiCARN) [B, 1, H, W]
    Output: gate [B, 1, H, W]
    
    Goal: gate ≈ 0 on peaks (protect loops), gate ≈ 1 elsewhere (allow modification)
    """
    def __init__(self, in_ch=1, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
    
    def forward(self, condition):
        return torch.sigmoid(self.net(condition))


# ================================================================
# Time Embedding
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


# ================================================================
# Flow Model
# ================================================================

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
    """
    UNet for Rectified Flow: predicts velocity from x_t
    """
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
# Training with TAD/Loop Preservation
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
    Training with gate/anchor mechanism for TAD/loop preservation.
    
    Flow: HiCARN -> GT
    - x_t = (1-t) * HiCARN + t * GT
    - velocity = GT - HiCARN
    
    Reconstruction:
    - pred_residual = model output (velocity prediction)
    - pred_gt = condition + gate * pred_residual
    
    Losses:
    - flow_loss: MSE(pred_velocity, target_velocity)
    - recon_loss: MSE(pred_gt, gt) + frequency_loss
    - insulation_loss: TAD boundary alignment
    - anchor_loss: protect loop peaks
    - gate_loss: encourage gate=0 on peaks, gate=1 elsewhere
    """
    model.train()
    gate_head.train()
    
    total_loss = 0
    total_flow = 0
    total_recon = 0
    total_ins = 0
    total_anc = 0
    total_gate = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for condition, gt in pbar:
        condition = condition.to(device)  # HiCARN = x_0
        gt = gt.to(device)                # GT = x_1
        
        batch_size = condition.shape[0]
        
        # ============================================
        # 1. Sample time and compute x_t
        # ============================================
        t = torch.rand(batch_size, device=device)
        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * condition + t_expand * gt
        
        # Target velocity
        target_velocity = gt - condition
        
        # ============================================
        # 2. Predict velocity
        # ============================================
        pred_velocity = model(x_t, t)
        
        # Clamp velocity to prevent explosion
        pred_velocity = torch.clamp(pred_velocity, -10, 10)
        
        # Flow loss (main diffusion objective)
        flow_loss = F.mse_loss(pred_velocity, target_velocity)
        
        # ============================================
        # 3. Gate and anchor mechanism
        # ============================================
        # Peak mask from condition (anchor these loop peaks)
        peak_mask = topk_peak_mask(
            condition, 
            topk_ratio=args.gate_topk_ratio,
            min_diag=args.min_diag
        )
        
        # Gate: where to apply residual
        gate = gate_head(condition)
        
        # Gated reconstruction
        # At t=1, x_t = gt, velocity should take us from x_t back
        # For reconstruction, we use the predicted residual at t=0
        pred_residual = pred_velocity  # velocity = gt - condition = residual
        pred_gt = condition + gate * pred_residual
        pred_gt = torch.clamp(pred_gt, -5, 5)  # Prevent extreme values
        
        # ============================================
        # 4. Reconstruction losses
        # ============================================
        # Basic reconstruction
        recon_loss = F.mse_loss(pred_gt, gt)
        
        # Frequency-separated loss (optional)
        if args.use_freq_loss:
            freq_loss = frequency_loss(pred_gt, gt)
            recon_loss = recon_loss + 0.5 * freq_loss
        
        # ============================================
        # 5. TAD/Loop preservation losses
        # ============================================
        # Insulation loss (TAD alignment)
        ins_loss = insulation_loss(pred_gt, gt, window=args.insul_window)
        
        # Anchor loss (protect loop peaks)
        anc_loss = anchor_loss(pred_gt, condition, peak_mask)
        
        # Gate regularization (gate=0 on peaks, gate=1 elsewhere)
        gate_prior = 1.0 - peak_mask
        gate_loss = F.mse_loss(gate, gate_prior)
        
        # ============================================
        # 6. Total loss with NaN protection
        # ============================================
        loss = (
            flow_loss +
            args.recon_weight * recon_loss +
            args.lambda_insulation * ins_loss +
            args.lambda_anchor * anc_loss +
            args.lambda_gate * gate_loss
        )
        
        # Skip if NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ NaN detected! Skipping batch.")
            print(f"  flow={flow_loss.item():.4f}, recon={recon_loss.item():.4f}")
            print(f"  ins={ins_loss.item():.4f}, anc={anc_loss.item():.4f}, gate={gate_loss.item():.4f}")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(gate_head.parameters()), 
            1.0
        )
        
        # Skip if gradient is NaN
        if torch.isnan(grad_norm):
            print(f"\n⚠️ NaN gradient detected! Skipping batch.")
            continue
        
        optimizer.step()
        
        total_loss += loss.item()
        total_flow += flow_loss.item()
        total_recon += recon_loss.item()
        total_ins += ins_loss.item()
        total_anc += anc_loss.item()
        total_gate += gate_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'flow': f'{flow_loss.item():.4f}',
            'anc': f'{anc_loss.item():.4f}'
        })
    
    return {
        'total': total_loss / num_batches,
        'flow': total_flow / num_batches,
        'recon': total_recon / num_batches,
        'insulation': total_ins / num_batches,
        'anchor': total_anc / num_batches,
        'gate': total_gate / num_batches
    }


@torch.no_grad()
def sample_ode(model, gate_head, condition, device, num_steps=10):
    """
    ODE sampling with gate mechanism.
    
    从 condition (HiCARN) 开始，沿 velocity 流向 GT。
    使用 gate 控制 residual 的应用。
    """
    batch_size = condition.shape[0]
    
    # Start from condition
    x = condition.clone()
    
    # Gate (fixed for this condition)
    gate = gate_head(condition)
    gate = torch.clamp(gate, 0, 1)  # Ensure gate is in [0, 1]
    
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = i / num_steps
        t_batch = torch.full((batch_size,), t, device=device)
        
        # Predict velocity
        v = model(x, t_batch)
        
        # Clamp velocity to prevent explosion
        v = torch.clamp(v, -10, 10)
        
        # Gated step
        x = x + dt * gate * v
        
        # Clamp x to prevent NaN
        x = torch.clamp(x, -10, 10)
    
    return x


@torch.no_grad()
def validate(model, gate_head, hicarn_val, gt_val, device, num_steps=10, args=None, seed=42):
    """Validate with proper metrics"""
    model.eval()
    gate_head.eval()
    torch.manual_seed(seed)
    
    n = min(500, len(hicarn_val))
    condition = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    # Sample
    pred = sample_ode(model, gate_head, condition, device, num_steps)
    pred = torch.clamp(pred, -5, 5)
    
    # Handle NaN
    if torch.isnan(pred).any():
        print("⚠️ NaN in predictions, returning baseline metrics")
        return {
            'mse': float('inf'),
            'pcc': 0.0,
            'mse_hicarn': F.mse_loss(condition, gt).item(),
            'pcc_hicarn': 0.95,
            'res_corr': 0.0,
            'ins_corr': 0.0,
            'ins_corr_hicarn': 0.0,
            'gate_on_peaks': 0.5,
            'gate_off_peaks': 0.5,
            'improved': False
        }
    
    # Basic metrics
    mse = F.mse_loss(pred, gt).item()
    mse_hicarn = F.mse_loss(condition, gt).item()
    
    pred_np = pred.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = condition.cpu().numpy().flatten()
    
    # Check for NaN in numpy arrays
    if np.isnan(pred_np).any() or np.isnan(gt_np).any():
        pcc = 0.0
        pcc_hicarn = 0.0
    else:
        pcc, _ = stats.pearsonr(pred_np, gt_np)
        pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    # Residual correlation
    pred_residual = (pred - condition).cpu().numpy()
    ideal_residual = (gt - condition).cpu().numpy()
    
    if np.isnan(pred_residual).any():
        res_corr = 0.0
    else:
        res_corr, _ = stats.pearsonr(pred_residual.flatten(), ideal_residual.flatten())
    
    # Insulation correlation (TAD metric)
    window = args.insul_window if args else 5
    ins_pred = insulation_vector(pred, window=window).cpu().numpy()
    ins_gt = insulation_vector(gt, window=window).cpu().numpy()
    ins_hicarn = insulation_vector(condition, window=window).cpu().numpy()
    
    if np.isnan(ins_pred).any() or np.isnan(ins_gt).any():
        ins_corr = 0.0
        ins_corr_hicarn = 0.0
    else:
        ins_corr, _ = stats.pearsonr(ins_pred.flatten(), ins_gt.flatten())
        ins_corr_hicarn, _ = stats.pearsonr(ins_hicarn.flatten(), ins_gt.flatten())
    
    # Gate statistics
    gate = gate_head(condition)
    topk_ratio = args.gate_topk_ratio if args else 0.01
    peak_mask = topk_peak_mask(condition, topk_ratio=topk_ratio)
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
    parser.add_argument('--output_dir', type=str, default='checkpoints_flow_v3')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=10)
    
    # Loss weights (保守默认值)
    parser.add_argument('--recon_weight', type=float, default=0.1,
                       help='Reconstruction loss weight (0.05-0.3)')
    parser.add_argument('--lambda_insulation', type=float, default=0.5,
                       help='Insulation loss weight for TAD (0.1-2.0)')
    parser.add_argument('--lambda_anchor', type=float, default=1.0,
                       help='Anchor loss weight for loop protection (0.3-3.0)')
    parser.add_argument('--lambda_gate', type=float, default=0.2,
                       help='Gate regularization weight (0.05-0.5)')
    
    # Gate/Anchor parameters
    parser.add_argument('--gate_topk_ratio', type=float, default=0.01,
                       help='Top-k ratio for peak mask (0.005-0.03)')
    parser.add_argument('--insul_window', type=int, default=5,
                       help='Insulation window size (3-7)')
    parser.add_argument('--min_diag', type=int, default=2,
                       help='Min diagonal offset for peak mask (1-4)')
    
    # Optional
    parser.add_argument('--use_freq_loss', action='store_true',
                       help='Use frequency-separated loss')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Load data
    # ================================================================
    print("\n" + "="*80)
    print("RECTIFIED FLOW v3 with TAD/Loop Preservation")
    print("="*80)
    print(f"Loss weights:")
    print(f"  recon_weight:      {args.recon_weight}")
    print(f"  lambda_insulation: {args.lambda_insulation} (TAD)")
    print(f"  lambda_anchor:     {args.lambda_anchor} (loop protection)")
    print(f"  lambda_gate:       {args.lambda_gate}")
    print(f"Gate parameters:")
    print(f"  gate_topk_ratio:   {args.gate_topk_ratio}")
    print(f"  insul_window:      {args.insul_window}")
    print(f"  min_diag:          {args.min_diag}")
    
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
    
    # Residual stats
    residual = gt_train - hicarn_train
    print(f"\nResidual (GT - HiCARN):")
    print(f"  Mean: {residual.mean():.6f}")
    print(f"  Std:  {residual.std():.6f}")
    
    train_dataset = TensorDataset(
        torch.from_numpy(hicarn_train).float(),
        torch.from_numpy(gt_train).float()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
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
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4)
    ).to(device)
    
    gate_head = GateHead(in_ch=1, hidden=16).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    gate_params = sum(p.numel() for p in gate_head.parameters())
    print(f"Flow model: {num_params / 1e6:.2f}M parameters")
    print(f"Gate head:  {gate_params / 1e3:.1f}K parameters")
    
    optimizer = optim.AdamW(
        list(model.parameters()) + list(gate_head.parameters()),
        lr=args.lr,
        weight_decay=1e-5
    )
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Resume from checkpoint
    start_epoch = 1
    best_mse = mse_baseline
    best_pcc = pcc_baseline
    history = []
    
    if args.resume:
        print(f"\n📂 Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'gate_head_state_dict' in checkpoint:
            gate_head.load_state_dict(checkpoint['gate_head_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_mse = checkpoint.get('best_mse', mse_baseline)
        best_pcc = checkpoint.get('best_pcc', pcc_baseline)
        
        # Load history if exists
        history_path = output_dir / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
        
        print(f"  Resumed from epoch {start_epoch - 1}")
        print(f"  Best MSE so far: {best_mse:.6f}")
        print(f"  Best PCC so far: {best_pcc:.4f}")
    
    # ================================================================
    # Training
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print("监控指标:")
    print("  - anchor_loss: 应该下降 (loop 被保护)")
    print("  - ins_corr: 应该上升 (TAD 结构对齐)")
    print("  - gate_on_peaks < gate_off_peaks (peaks 区域 gate 小)")
    
    for epoch in range(start_epoch, args.epochs + 1):
        losses = train_epoch(
            model, gate_head, optimizer, train_loader,
            device, epoch, args
        )
        
        # Update learning rate
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate(
                model, gate_head, hicarn_val, gt_val,
                device, args.num_steps, args
            )
            
            improved = ""
            if val_metrics['mse'] < best_mse:
                best_mse = val_metrics['mse']
                improved += " [best MSE]"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'gate_head_state_dict': gate_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mse': best_mse,
                    'best_pcc': best_pcc,
                    'val_metrics': val_metrics,
                    'args': vars(args),
                    'config': {
                        'base_channels': args.base_channels,
                        'channel_mults': [1, 2, 4],
                        'num_steps': args.num_steps
                    }
                }, output_dir / 'best_model.pt')
            
            if val_metrics['pcc'] > best_pcc:
                best_pcc = val_metrics['pcc']
                improved += " [best PCC]"
            
            status = "✓" if val_metrics['improved'] else "⚠"
            
            print(f"\n  Epoch {epoch}: {status}")
            print(f"    Losses: flow={losses['flow']:.4f}, anchor={losses['anchor']:.4f}, ins={losses['insulation']:.4f}")
            print(f"    MSE={val_metrics['mse']:.6f} (HiCARN:{val_metrics['mse_hicarn']:.6f})")
            print(f"    PCC={val_metrics['pcc']:.4f} (HiCARN:{val_metrics['pcc_hicarn']:.4f})")
            print(f"    res_corr={val_metrics['res_corr']:.4f}")
            print(f"    ins_corr={val_metrics['ins_corr']:.4f} (HiCARN:{val_metrics['ins_corr_hicarn']:.4f}) ← TAD")
            print(f"    gate: peaks={val_metrics['gate_on_peaks']:.3f}, bg={val_metrics['gate_off_peaks']:.3f}")
            
            if val_metrics['gate_on_peaks'] < val_metrics['gate_off_peaks'] - 0.1:
                print(f"    ✓ Gate is working (protecting peaks)")
            
            if improved:
                print(f"    {improved}")
            
            history.append({
                'epoch': epoch,
                'losses': losses,
                'val_metrics': val_metrics
            })
        
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
            
            # Also save latest checkpoint for easy resume
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'gate_head_state_dict': gate_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mse': best_mse,
                'best_pcc': best_pcc
            }, output_dir / 'latest_checkpoint.pt')
    
    # Save
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    print(f"\nHiCARN baseline: MSE={mse_baseline:.6f}, PCC={pcc_baseline:.4f}")
    print(f"Best refined:    MSE={best_mse:.6f}, PCC={best_pcc:.4f}")
    
    if best_mse < mse_baseline:
        print(f"\n✓ SUCCESS: Improved over HiCARN!")
    else:
        print(f"\n⚠ Did not improve over HiCARN in MSE")
    
    print(f"\nResults saved to: {output_dir}")
    
    print("\n" + "="*80)
    print("调参建议:")
    print("="*80)
    print("如果 loop 掉:")
    print("  - lambda_anchor ↑ (1.0 → 2.0)")
    print("  - gate_topk_ratio ↑ (0.01 → 0.02)")
    print("如果 TAD 不够:")
    print("  - lambda_insulation ↑ (0.5 → 1.0)")
    print("  - insul_window ↑ (5 → 7)")
    print("如果像素指标变差:")
    print("  - recon_weight ↑ (0.1 → 0.2)")


if __name__ == '__main__':
    main()
