#!/usr/bin/env python3
"""
Diffusion Refinement v6 - 基于评论的关键改进

关键改动：
1. 显式2通道条件输入 (x_t + condition) - 已有，保持
2. t 采样偏向 0 (Beta分布或混合采样)
3. 新增 cosine 方向 loss，防止模型输出小幅度糊弄
4. 更新超参数范围

目标：让 raw_res_corr 从 ~0.02 提升到 0.10+
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
import sys
from scipy import stats


# ================================================================
# Model (same as before)
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


class SimpleConditionedUNet(nn.Module):
    """
    2通道输入的 UNet：
    - 输入: concat([x_t, condition], dim=1) -> 2 channels
    - 输出: 残差 (1 channel)
    
    这是 Step 1 的关键改动，让模型显式地看到 condition
    """
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=256,
        parameterization='v'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.parameterization = parameterization
        
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
# Scheduler
# ================================================================

class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        parameterization='v'
    ):
        self.num_train_timesteps = num_train_timesteps
        self.parameterization = parameterization
        
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x_start, noise, timesteps):
        sqrt_alpha = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def get_v_target(self, x_start, noise, timesteps):
        sqrt_alpha = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x_start
    
    def predict_x0_from_v(self, x_t, v, timesteps):
        sqrt_alpha = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        while sqrt_alpha.dim() < x_t.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v
    
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self


# ================================================================
# Time Sampling Strategies (Step 2)
# ================================================================

def sample_timesteps_biased(
    batch_size: int,
    num_timesteps: int,
    device: torch.device,
    strategy: str = 'beta',
    p0: float = 0.3,
    beta_a: float = 0.5,
    beta_b: float = 2.0
):
    """
    Step 2: t 采样偏向 0
    
    因为我们最在意 t≈0 时的模型表现（推理从 condition 出发），
    Uniform 采样会让模型在 t≈0 区域看到的数据很少。
    
    Args:
        strategy: 'beta' 或 'mixed'
            - 'beta': t ~ Beta(beta_a, beta_b)，向 0 偏
            - 'mixed': 以 p0 概率取 t=0，其余 Uniform(0,1)
        p0: mixed 策略中取 t=0 的概率
        beta_a, beta_b: Beta 分布参数，默认 (0.5, 2.0) 明显向 0 偏
    """
    if strategy == 'beta':
        # Beta 分布采样，然后映射到 [0, num_timesteps-1]
        # Beta(0.5, 2.0) 明显偏向 0
        t_continuous = torch.distributions.Beta(beta_a, beta_b).sample((batch_size,))
        t = (t_continuous * num_timesteps).long().clamp(0, num_timesteps - 1)
        return t.to(device)
    
    elif strategy == 'mixed':
        # 混合采样：以 p0 概率取 t=0，其余 Uniform
        t = torch.randint(0, num_timesteps, (batch_size,), device=device)
        # 随机选择哪些样本用 t=0
        mask = torch.rand(batch_size, device=device) < p0
        t[mask] = 0
        return t
    
    else:  # 'uniform'
        return torch.randint(0, num_timesteps, (batch_size,), device=device)


# ================================================================
# Direction Loss (Step 3)
# ================================================================

def cosine_direction_loss(pred_res, target_res, eps=1e-8):
    """
    Step 3: 方向性 loss
    
    防止模型用"小幅度输出"糊弄过去。
    L1/L2 loss 下模型容易学成输出很小的 residual（L1不会特别大），
    但方向不对，所以相关性几乎是 0。
    
    这个 loss 直接针对 raw_res_corr 上不去的问题。
    
    L_dir = 1 - cosine(pred_res, target_res)
    
    对每个样本 flatten 再算 cosine similarity
    """
    batch_size = pred_res.shape[0]
    
    # Flatten each sample
    pred_flat = pred_res.view(batch_size, -1)
    target_flat = target_res.view(batch_size, -1)
    
    # Compute cosine similarity for each sample
    pred_norm = pred_flat / (pred_flat.norm(dim=1, keepdim=True) + eps)
    target_norm = target_flat / (target_flat.norm(dim=1, keepdim=True) + eps)
    
    cosine_sim = (pred_norm * target_norm).sum(dim=1)  # [batch_size]
    
    # Loss = 1 - cosine_similarity (want to maximize similarity)
    loss = (1 - cosine_sim).mean()
    
    return loss


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


def compute_residual_stats(hicarn, gt):
    residual = gt - hicarn
    res_mean = float(residual.mean())
    res_std = float(residual.std())
    
    print(f"\nResidual statistics:")
    print(f"  Mean: {res_mean:.6f}")
    print(f"  Std:  {res_std:.6f}")
    print(f"  Range: [{residual.min():.4f}, {residual.max():.4f}]")
    
    return res_mean, res_std


# ================================================================
# Training (v6 with all improvements)
# ================================================================

def train_epoch_v6(
    model, 
    scheduler, 
    optimizer, 
    dataloader, 
    device,
    res_mean,
    res_std,
    epoch,
    # Core parameters
    alpha=0.15,
    # Loss weights (updated defaults from comments)
    lambda_res=10.0,      # L1 residual loss weight (range: 5-20)
    lambda_dir=0.5,       # NEW: cosine direction loss (range: 0.2-1.0)
    lambda_recon=0.5,     # reconstruction loss (range: 0.2-0.8)
    lambda_magnitude=0.0, # magnitude constraint (can keep 0 or small)
    # Time sampling (Step 2)
    t_sampling='beta',    # 'beta', 'mixed', or 'uniform'
    t_p0=0.3,             # for 'mixed': probability of t=0
    t_beta_a=0.5,         # for 'beta': Beta(a, b) parameters
    t_beta_b=2.0,
):
    """
    v6 训练函数，包含所有关键改进：
    
    1. 显式2通道输入 (已有)
    2. t 采样偏向 0 (Step 2)
    3. cosine 方向 loss (Step 3)
    """
    model.train()
    
    total_loss = 0
    total_diff_loss = 0
    total_res_loss = 0
    total_dir_loss = 0
    total_recon_loss = 0
    total_mag_loss = 0
    num_batches = 0
    
    # Track direction correlation during training
    dir_corrs = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for hicarn, gt in pbar:
        hicarn = hicarn.to(device)
        gt = gt.to(device)
        
        batch_size = hicarn.shape[0]
        
        # Compute and normalize residual (target)
        residual = gt - hicarn
        residual_centered = residual - res_mean
        residual_scaled = residual_centered / res_std  # This is the target
        
        # ============================================
        # Step 2: 采样 t 时偏向 0
        # ============================================
        t = sample_timesteps_biased(
            batch_size, scheduler.num_train_timesteps, device,
            strategy=t_sampling, p0=t_p0, beta_a=t_beta_a, beta_b=t_beta_b
        )
        
        # Sample noise
        noise = torch.randn_like(residual_scaled)
        
        # Add noise to residual
        residual_noisy = scheduler.add_noise(residual_scaled, noise, t)
        
        # ============================================
        # Step 1: 显式2通道输入 concat([x_t, condition])
        # ============================================
        x_input = torch.cat([residual_noisy, hicarn], dim=1)  # 2 channels
        pred = model(x_input, t)
        
        # Loss 1: Diffusion loss (v-prediction)
        target_v = scheduler.get_v_target(residual_scaled, noise, t)
        diff_loss = F.mse_loss(pred, target_v)
        
        # Predict x0 from v
        pred_x0 = scheduler.predict_x0_from_v(residual_noisy, pred, t)
        
        # ============================================
        # Step 3: 方向性 loss (cosine similarity)
        # ============================================
        # pred_x0 是预测的 scaled residual，residual_scaled 是目标
        dir_loss = cosine_direction_loss(pred_x0, residual_scaled)
        
        # Loss 2: L1 residual loss
        res_loss = F.l1_loss(pred_x0, residual_scaled)
        
        # Loss 3: Magnitude constraint (keep small or 0)
        mag_loss = torch.mean(pred_x0 ** 2)
        
        # Loss 4: Reconstruction loss
        pred_residual = pred_x0 * res_std + res_mean
        pred_final = hicarn + alpha * pred_residual
        recon_loss = F.mse_loss(pred_final, gt)
        
        # ============================================
        # Total loss
        # ============================================
        loss = (
            diff_loss + 
            lambda_res * res_loss +
            lambda_dir * dir_loss +
            lambda_recon * recon_loss +
            lambda_magnitude * mag_loss
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_diff_loss += diff_loss.item()
        total_res_loss += res_loss.item()
        total_dir_loss += dir_loss.item()
        total_recon_loss += recon_loss.item()
        total_mag_loss += mag_loss.item()
        num_batches += 1
        
        # Compute direction correlation for monitoring
        with torch.no_grad():
            corr = F.cosine_similarity(
                pred_x0.view(batch_size, -1),
                residual_scaled.view(batch_size, -1),
                dim=1
            ).mean().item()
            dir_corrs.append(corr)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dir': f'{dir_loss.item():.3f}',
            'corr': f'{corr:.3f}'
        })
    
    return {
        'total': total_loss / num_batches,
        'diffusion': total_diff_loss / num_batches,
        'residual': total_res_loss / num_batches,
        'direction': total_dir_loss / num_batches,
        'reconstruction': total_recon_loss / num_batches,
        'magnitude': total_mag_loss / num_batches,
        'train_dir_corr': np.mean(dir_corrs)  # 训练时的方向相关性
    }


@torch.no_grad()
def validate_proper_sampling(
    model,
    scheduler,
    hicarn_val,
    gt_val,
    res_mean,
    res_std,
    alpha,
    device,
    num_steps=20,
    seed=42
):
    """
    正确的验证：从纯噪声开始采样，不使用任何 GT 信息
    """
    model.eval()
    
    torch.manual_seed(seed)
    
    n = min(200, len(hicarn_val))
    hicarn = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    # 从纯噪声开始
    x_t = torch.randn_like(hicarn)
    
    # DDIM 采样
    timesteps = torch.linspace(
        scheduler.num_train_timesteps - 1, 0, num_steps
    ).long().to(device)
    
    pred_x0 = None
    for i, t in enumerate(timesteps):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        
        # 显式2通道输入
        x_input = torch.cat([x_t, hicarn], dim=1)
        
        pred = model(x_input, t_batch)
        pred_x0 = scheduler.predict_x0_from_v(x_t, pred, t_batch)
        
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_t = scheduler.alphas_cumprod[t]
            alpha_t_next = scheduler.alphas_cumprod[t_next]
            
            pred_noise = (x_t - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt()
            x_t = alpha_t_next.sqrt() * pred_x0 + (1 - alpha_t_next).sqrt() * pred_noise
        else:
            x_t = pred_x0
    
    # Unscale residual
    pred_residual = pred_x0 * res_std + res_mean
    
    # Final with alpha scaling
    pred_final = hicarn + alpha * pred_residual
    pred_final = torch.clamp(pred_final, -5, 5)
    
    # ============================================
    # Metrics
    # ============================================
    mse = F.mse_loss(pred_final, gt).item()
    mse_hicarn = F.mse_loss(hicarn, gt).item()
    
    oracle_final = hicarn + alpha * (gt - hicarn)
    mse_oracle = F.mse_loss(oracle_final, gt).item()
    
    final_np = pred_final.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = hicarn.cpu().numpy().flatten()
    
    pcc, _ = stats.pearsonr(final_np, gt_np)
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    residual_np = pred_residual.cpu().numpy()
    ideal_residual = (gt - hicarn).cpu().numpy()
    
    # ============================================
    # 关键指标：raw_res_corr (预测残差与理想残差的相关性)
    # 目标：从 ~0.02 提升到 0.10+
    # ============================================
    raw_res_corr, _ = stats.pearsonr(
        residual_np.flatten(), ideal_residual.flatten()
    )
    
    # 方向相关性 (应该和 raw_res_corr 类似)
    dir_corr = F.cosine_similarity(
        torch.from_numpy(residual_np).view(1, -1),
        torch.from_numpy(ideal_residual).view(1, -1),
        dim=1
    ).item()
    
    return {
        'mse': mse,
        'pcc': float(pcc),
        'mse_hicarn': mse_hicarn,
        'pcc_hicarn': float(pcc_hicarn),
        'mse_oracle': mse_oracle,
        'mse_improvement': mse_hicarn - mse,
        'pcc_improvement': float(pcc) - float(pcc_hicarn),
        'residual_std': float(residual_np.std()),
        'ideal_residual_std': float(ideal_residual.std()),
        'raw_res_corr': float(raw_res_corr),  # 关键指标！目标 > 0.10
        'dir_corr': float(dir_corr),
        'improved_over_hicarn': float(pcc) > float(pcc_hicarn)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument('--train_hicarn', type=str, required=True)
    parser.add_argument('--train_gt', type=str, required=True)
    parser.add_argument('--val_hicarn', type=str, default=None)
    parser.add_argument('--val_gt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints_v6')
    
    # Training basics
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    
    # Core parameter
    parser.add_argument('--alpha', type=float, default=0.15,
                       help='Residual scaling factor (range: 0.10-0.25, start with 0.15)')
    
    # Loss weights (updated from comments)
    parser.add_argument('--lambda_res', type=float, default=10.0,
                       help='L1 residual loss weight (range: 5-20)')
    parser.add_argument('--lambda_dir', type=float, default=0.5,
                       help='NEW: Cosine direction loss weight (range: 0.2-1.0)')
    parser.add_argument('--lambda_recon', type=float, default=0.5,
                       help='Reconstruction loss weight (range: 0.2-0.8)')
    parser.add_argument('--lambda_magnitude', type=float, default=0.0,
                       help='Magnitude constraint (can keep 0 or small)')
    
    # Time sampling (Step 2)
    parser.add_argument('--t_sampling', type=str, default='beta',
                       choices=['beta', 'mixed', 'uniform'],
                       help='Time sampling strategy (beta recommended)')
    parser.add_argument('--t_p0', type=float, default=0.3,
                       help='For mixed: probability of sampling t=0')
    parser.add_argument('--t_beta_a', type=float, default=0.5,
                       help='Beta distribution parameter a')
    parser.add_argument('--t_beta_b', type=float, default=2.0,
                       help='Beta distribution parameter b (larger = more bias to 0)')
    
    # Validation
    parser.add_argument('--val_steps', type=int, default=20,
                       help='Number of DDIM steps for validation')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Print configuration
    # ================================================================
    print("\n" + "="*80)
    print("DIFFUSION REFINEMENT v6 (with key improvements)")
    print("="*80)
    print(f"\n关键改进:")
    print(f"  1. 显式2通道输入 (x_t + condition)")
    print(f"  2. t 采样偏向 0 (strategy: {args.t_sampling})")
    print(f"  3. cosine 方向 loss (lambda_dir: {args.lambda_dir})")
    
    print(f"\n超参数:")
    print(f"  alpha: {args.alpha} (range: 0.10-0.25)")
    print(f"  lambda_res: {args.lambda_res} (range: 5-20)")
    print(f"  lambda_dir: {args.lambda_dir} (range: 0.2-1.0)")
    print(f"  lambda_recon: {args.lambda_recon} (range: 0.2-0.8)")
    print(f"  t_sampling: {args.t_sampling}")
    if args.t_sampling == 'beta':
        print(f"    Beta({args.t_beta_a}, {args.t_beta_b})")
    elif args.t_sampling == 'mixed':
        print(f"    p0={args.t_p0}")
    
    # ================================================================
    # Load data
    # ================================================================
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    hicarn_train, gt_train = load_data(args.train_hicarn, args.train_gt)
    
    if args.val_hicarn and args.val_gt:
        hicarn_val, gt_val = load_data(args.val_hicarn, args.val_gt)
    else:
        split = int(len(hicarn_train) * 0.9)
        hicarn_val = hicarn_train[split:]
        gt_val = gt_train[split:]
        hicarn_train = hicarn_train[:split]
        gt_train = gt_train[:split]
        print(f"  Using 10% for validation: train={len(hicarn_train)}, val={len(hicarn_val)}")
    
    res_mean, res_std = compute_residual_stats(hicarn_train, gt_train)
    
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
    # Baseline & Oracle
    # ================================================================
    print("\n" + "="*80)
    print("BASELINE & ORACLE")
    print("="*80)
    
    mse_baseline = np.mean((hicarn_val - gt_val) ** 2)
    pcc_baseline, _ = stats.pearsonr(hicarn_val.flatten(), gt_val.flatten())
    
    mse_oracle = (1 - args.alpha) ** 2 * mse_baseline
    
    print(f"HiCARN baseline:")
    print(f"  MSE: {mse_baseline:.6f}")
    print(f"  PCC: {pcc_baseline:.4f}")
    print(f"\nOracle (if model learns perfect residual):")
    print(f"  MSE: {mse_oracle:.6f} (= (1-{args.alpha})^2 * {mse_baseline:.6f})")
    
    # ================================================================
    # Model
    # ================================================================
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = SimpleConditionedUNet(
        in_channels=2,  # x_t + condition
        out_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4),
        parameterization='v'
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    print(f"Input channels: 2 (x_t + condition)")
    print(f"Output channels: 1 (residual)")
    
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_timesteps,
        parameterization='v'
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # ================================================================
    # Training
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print(f"\n目标: raw_res_corr > 0.10 (当前通常 ~0.01-0.04)")
    print(f"达标后可以调大 alpha ({args.alpha} -> 0.20-0.25)")
    
    best_pcc = pcc_baseline
    best_mse = mse_baseline
    best_raw_res_corr = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch_v6(
            model, scheduler, optimizer, train_loader,
            device, res_mean, res_std, epoch,
            alpha=args.alpha,
            lambda_res=args.lambda_res,
            lambda_dir=args.lambda_dir,
            lambda_recon=args.lambda_recon,
            lambda_magnitude=args.lambda_magnitude,
            t_sampling=args.t_sampling,
            t_p0=args.t_p0,
            t_beta_a=args.t_beta_a,
            t_beta_b=args.t_beta_b
        )
        
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate_proper_sampling(
                model, scheduler, hicarn_val, gt_val,
                res_mean, res_std, args.alpha, device,
                num_steps=args.val_steps
            )
            
            improved = ""
            if val_metrics['mse'] < best_mse:
                best_mse = val_metrics['mse']
                improved += " [best MSE]"
            
            if val_metrics['raw_res_corr'] > best_raw_res_corr:
                best_raw_res_corr = val_metrics['raw_res_corr']
                improved += " [best raw_res_corr]"
            
            if val_metrics['pcc'] > best_pcc:
                best_pcc = val_metrics['pcc']
                improved += " [best PCC]"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_metrics': val_metrics,
                    'res_std': res_std,
                    'res_mean': res_mean,
                    'alpha': args.alpha,
                    'config': {
                        'in_channels': 2,
                        'base_channels': args.base_channels,
                        'channel_mults': [1, 2, 4],
                        'parameterization': 'v',
                        'num_timesteps': args.num_timesteps,
                        'alpha': args.alpha,
                        'lambda_res': args.lambda_res,
                        'lambda_dir': args.lambda_dir,
                        'lambda_recon': args.lambda_recon,
                        't_sampling': args.t_sampling
                    }
                }, output_dir / 'best_model.pt')
            
            # Status indicators
            status_pcc = "✓" if val_metrics['improved_over_hicarn'] else "⚠"
            status_corr = "✓" if val_metrics['raw_res_corr'] > 0.10 else "⚠"
            
            print(f"\n  Epoch {epoch}:")
            print(f"    MSE:          {val_metrics['mse']:.6f} (HiCARN: {val_metrics['mse_hicarn']:.6f}) {status_pcc}")
            print(f"    PCC:          {val_metrics['pcc']:.4f} (HiCARN: {val_metrics['pcc_hicarn']:.4f})")
            print(f"    raw_res_corr: {val_metrics['raw_res_corr']:.4f} (目标 > 0.10) {status_corr}")
            print(f"    dir_corr:     {val_metrics['dir_corr']:.4f}")
            print(f"    train_corr:   {losses['train_dir_corr']:.4f}")
            print(f"    res_std:      {val_metrics['residual_std']:.4f} (ideal: {val_metrics['ideal_residual_std']:.4f})")
            
            if improved:
                print(f"    {improved}")
            
            # Suggest next steps based on raw_res_corr
            if val_metrics['raw_res_corr'] > 0.12:
                print(f"    💡 raw_res_corr 达标! 可以考虑调大 alpha (当前: {args.alpha})")
            
            history.append({
                'epoch': epoch,
                'losses': losses,
                'val_metrics': val_metrics
            })
        
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'res_std': res_std,
                'res_mean': res_mean,
                'alpha': args.alpha,
                'config': {
                    'in_channels': 2,
                    'base_channels': args.base_channels,
                    'alpha': args.alpha,
                    't_sampling': args.t_sampling
                }
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final config
    config = {
        'alpha': args.alpha,
        'lambda_res': args.lambda_res,
        'lambda_dir': args.lambda_dir,
        'lambda_recon': args.lambda_recon,
        't_sampling': args.t_sampling,
        't_beta_a': args.t_beta_a,
        't_beta_b': args.t_beta_b,
        't_p0': args.t_p0,
        'base_channels': args.base_channels,
        'num_timesteps': args.num_timesteps,
        'best_raw_res_corr': best_raw_res_corr,
        'best_pcc': best_pcc,
        'best_mse': best_mse
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    print(f"\nHiCARN baseline:")
    print(f"  MSE: {mse_baseline:.6f}")
    print(f"  PCC: {pcc_baseline:.4f}")
    
    print(f"\nOracle (upper bound):")
    print(f"  MSE: {mse_oracle:.6f}")
    
    print(f"\nBest refined:")
    print(f"  MSE:          {best_mse:.6f} (Δ={mse_baseline - best_mse:+.6f})")
    print(f"  PCC:          {best_pcc:.4f} (Δ={best_pcc - pcc_baseline:+.4f})")
    print(f"  raw_res_corr: {best_raw_res_corr:.4f} (目标 > 0.10)")
    
    if best_pcc > pcc_baseline:
        print(f"\n✓ SUCCESS: Improved over HiCARN!")
    else:
        print(f"\n⚠ Did not improve over HiCARN")
    
    if best_raw_res_corr > 0.10:
        print(f"✓ raw_res_corr 达标! 下一步可以调大 alpha")
    else:
        print(f"⚠ raw_res_corr 未达标，建议:")
        print(f"  - 增大 lambda_dir (当前: {args.lambda_dir})")
        print(f"  - 检查 t_sampling 设置")
        print(f"  - 训练更多 epochs")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"\n建议的下一步超参数搜索范围:")
    print(f"  alpha: 0.10 – 0.25 (当前: {args.alpha})")
    print(f"  lambda_res: 5 – 20 (当前: {args.lambda_res})")
    print(f"  lambda_dir: 0.2 – 1.0 (当前: {args.lambda_dir})")
    print(f"  lambda_recon: 0.2 – 0.8 (当前: {args.lambda_recon})")


if __name__ == '__main__':
    main()
