#!/usr/bin/env python3
"""
Diffusion Refinement v11 - Gate 分离约束 + 极简 Warmup

================================================================================
v10 → v11 关键改进：

1. Gate 分离约束 (gate_peaks < gate_bg)
   - 新增 L_gate_sep = ReLU(g_peaks - g_bg + margin)
   - 确保 peak 区域 gate 明显小于 bg gate
   - 参数: --gate_sep_margin (default 0.03)

2. 极简 Warmup 策略
   - Warmup 阶段只开 res + dir loss
   - 结构保护 loss (peak/ins/gate) 权重极低 (0.001)
   - 让模型先学会残差方向，再加保护

3. 更激进的 lambda_dir
   - 默认从 0.5 → 2.0
   - 可调到 5.0 如果 raw_res_corr 还是上不去

4. 训练时打印 gate 分离度
   - gate_separation = gate_bg - gate_peaks
   - 目标: separation > margin

================================================================================
诊断要点：

如果 raw_res_corr < 0.05 after 5 epochs:
  → 尝试 --peak_protection_mode none
  → 增大 --lambda_dir 5.0

如果 gate_peaks ≈ gate_bg:
  → 增大 --gate_sep_margin 0.05
  → 增大 --lambda_gate_sep 0.5

================================================================================
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
# Model
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


class GatedConditionedUNet(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=256,
        parameterization='v',
        cond_norm_type='learnable',
        output_gate=True,
        g_scale=0.5
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.parameterization = parameterization
        self.cond_norm_type = cond_norm_type
        self.output_gate = output_gate
        self.g_scale = g_scale
        
        if cond_norm_type == 'learnable':
            self.cond_transform = nn.Sequential(
                nn.Conv2d(1, 1, 1, bias=True),
            )
            nn.init.constant_(self.cond_transform[0].weight, 0.4)
            nn.init.constant_(self.cond_transform[0].bias, -0.5)
        else:
            self.cond_transform = None
        
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
        
        self.final_norm = nn.GroupNorm(8, ch)
        self.final_act = nn.SiLU()
        self.residual_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        if output_gate:
            self.gate_conv = nn.Sequential(
                nn.Conv2d(ch, ch // 2, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(ch // 2, 1, 1),
                nn.Sigmoid()
            )
        else:
            self.gate_conv = None
    
    def normalize_condition(self, cond, cond_mean=None, cond_std=None):
        if self.cond_norm_type == 'learnable' and self.cond_transform is not None:
            return self.cond_transform(cond)
        elif self.cond_norm_type == 'fixed' and cond_mean is not None:
            return (cond - cond_mean) / (cond_std + 1e-8)
        else:
            return cond
    
    def forward(self, x_t, cond, t, cond_mean=None, cond_std=None):
        cond_norm = self.normalize_condition(cond, cond_mean, cond_std)
        x = torch.cat([x_t, cond_norm], dim=1)
        
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
        
        h = self.final_norm(h)
        h = self.final_act(h)
        
        residual = self.residual_conv(h)
        
        if self.output_gate and self.gate_conv is not None:
            gate = self.gate_conv(h) * self.g_scale
            return residual, gate
        else:
            return residual, None


# ================================================================
# Scheduler
# ================================================================

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, parameterization='v'):
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
# Time Sampling
# ================================================================

def sample_timesteps_biased(batch_size, num_timesteps, device, strategy='beta',
                            p0=0.3, beta_a=0.5, beta_b=2.0):
    if strategy == 'beta':
        t_continuous = torch.distributions.Beta(beta_a, beta_b).sample((batch_size,))
        t = (t_continuous * num_timesteps).long().clamp(0, num_timesteps - 1)
        return t.to(device)
    elif strategy == 'mixed':
        t = torch.randint(0, num_timesteps, (batch_size,), device=device)
        mask = torch.rand(batch_size, device=device) < p0
        t[mask] = 0
        return t
    else:
        return torch.randint(0, num_timesteps, (batch_size,), device=device)


# ================================================================
# Loss Functions
# ================================================================

def cosine_direction_loss(pred_res, target_res, eps=1e-8):
    batch_size = pred_res.shape[0]
    pred_flat = pred_res.view(batch_size, -1)
    target_flat = target_res.view(batch_size, -1)
    
    pred_norm = pred_flat / (pred_flat.norm(dim=1, keepdim=True) + eps)
    target_norm = target_flat / (target_flat.norm(dim=1, keepdim=True) + eps)
    
    cosine_sim = (pred_norm * target_norm).sum(dim=1)
    return (1 - cosine_sim).mean()


def compute_insulation_score(matrix, window_size=5):
    B, C, H, W = matrix.shape
    assert H == W
    
    insulation = torch.zeros(B, H, device=matrix.device)
    
    for i in range(window_size, H - window_size):
        start = max(0, i - window_size)
        end = min(H, i + window_size)
        local_sum = matrix[:, 0, start:end, start:end].mean(dim=(1, 2))
        insulation[:, i] = local_sum
    
    return insulation


def insulation_loss(pred_matrix, gt_matrix, window_size=5):
    pred_ins = compute_insulation_score(pred_matrix, window_size)
    gt_ins = compute_insulation_score(gt_matrix, window_size)
    return F.mse_loss(pred_ins, gt_ins)


def get_peak_mask(matrix, topk_ratio=0.02):
    B, C, H, W = matrix.shape
    flat = matrix.view(B, -1)
    
    k = max(1, int(flat.shape[1] * topk_ratio))
    topk_vals, _ = torch.topk(flat, k, dim=1)
    threshold = topk_vals[:, -1].view(B, 1, 1, 1)
    
    mask = (matrix >= threshold).float()
    return mask


def get_peak_mask_with_diagnostics(matrix, topk_ratio=0.02):
    B, C, H, W = matrix.shape
    flat = matrix.view(B, -1)
    total_pixels = flat.shape[1]
    
    k = max(1, int(total_pixels * topk_ratio))
    topk_vals, _ = torch.topk(flat, k, dim=1)
    threshold = topk_vals[:, -1].view(B, 1, 1, 1)
    
    mask = (matrix >= threshold).float()
    actual_coverage = mask.sum() / (B * total_pixels)
    
    return mask, {
        'k': k,
        'total_pixels': total_pixels,
        'threshold': threshold.mean().item(),
        'actual_coverage': actual_coverage.item(),
    }


def peak_preservation_loss(pred_matrix, gt_matrix, topk_ratio=0.02, mode='strict'):
    """
    峰值保护 loss
    
    mode:
    - 'strict': 惩罚 pred 与 gt 在 peak 区域的差异
    - 'adaptive': 只惩罚"远离 GT"的更新
    - 'none': 返回 0
    """
    if mode == 'none':
        return torch.tensor(0.0, device=pred_matrix.device)
    
    peak_mask = get_peak_mask(gt_matrix, topk_ratio)
    
    if mode == 'strict':
        diff = (pred_matrix - gt_matrix) ** 2
        peak_loss = (diff * peak_mask).sum() / (peak_mask.sum() + 1e-8)
        return peak_loss
    
    elif mode == 'adaptive':
        # 只惩罚 pred 比 gt 小的情况（峰值被压低了）
        diff = F.relu(gt_matrix - pred_matrix) ** 2
        peak_loss = (diff * peak_mask).sum() / (peak_mask.sum() + 1e-8)
        return peak_loss
    
    else:
        return torch.tensor(0.0, device=pred_matrix.device)


def high_frequency_loss(pred_matrix, gt_matrix):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=pred_matrix.dtype, device=pred_matrix.device)
    sobel_y = sobel_x.t()
    
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    pred_gx = F.conv2d(pred_matrix, sobel_x, padding=1)
    pred_gy = F.conv2d(pred_matrix, sobel_y, padding=1)
    pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8)
    
    gt_gx = F.conv2d(gt_matrix, sobel_x, padding=1)
    gt_gy = F.conv2d(gt_matrix, sobel_y, padding=1)
    gt_grad = torch.sqrt(gt_gx**2 + gt_gy**2 + 1e-8)
    
    return F.mse_loss(pred_grad, gt_grad)


def gate_regularization_loss(gate, cond, gate_bg_target=0.12, topk_ratio=0.02):
    """原始 gate 正则化"""
    peak_mask = get_peak_mask(cond, topk_ratio)
    bg_mask = 1 - peak_mask
    gate_target = gate_bg_target * bg_mask
    return F.mse_loss(gate, gate_target)


def gate_separation_loss(gate, cond, margin=0.03, topk_ratio=0.02):
    """
    Gate 分离约束: 确保 gate_peaks < gate_bg
    
    L_gate_sep = ReLU(g_peaks - g_bg + margin)
    
    目标: gate 在 peak 区域应该明显小于 bg 区域
    """
    peak_mask = get_peak_mask(cond, topk_ratio)
    bg_mask = 1 - peak_mask
    
    # 计算 peak 和 bg 区域的平均 gate 值
    gate_peaks = (gate * peak_mask).sum() / (peak_mask.sum() + 1e-8)
    gate_bg = (gate * bg_mask).sum() / (bg_mask.sum() + 1e-8)
    
    # 惩罚 gate_peaks >= gate_bg - margin
    # 即要求 gate_peaks < gate_bg - margin
    sep_loss = F.relu(gate_peaks - gate_bg + margin)
    
    return sep_loss, gate_peaks.item(), gate_bg.item()


def anchor_loss(pred_matrix, gt_matrix):
    mean_loss = (pred_matrix.mean() - gt_matrix.mean()) ** 2
    std_loss = (pred_matrix.std() - gt_matrix.std()) ** 2
    return mean_loss + std_loss


# ================================================================
# Loss Weight Scheduler (极简 Warmup)
# ================================================================

class LossWeightSchedulerV11:
    """
    v11: 极简 Warmup 策略
    
    Warmup 阶段: 只开 res + dir，结构 loss 极低 (0.001)
    Rampup 阶段: 线性增加结构 loss
    Full 阶段: 达到目标权重
    """
    def __init__(
        self,
        warmup_epochs=20,
        rampup_epochs=10,
        # 目标权重
        target_lambda_insulation=0.3,
        target_lambda_peak=0.5,
        target_lambda_hf=0.2,
        target_lambda_anchor=0.5,
        target_lambda_gate=0.3,
        target_lambda_gate_sep=0.3,
        # 极简 warmup: 几乎为 0
        min_lambda_insulation=0.001,
        min_lambda_peak=0.001,
        min_lambda_hf=0.001,
        min_lambda_anchor=0.01,
        min_lambda_gate=0.001,
        min_lambda_gate_sep=0.01,
    ):
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        
        self.target = {
            'lambda_insulation': target_lambda_insulation,
            'lambda_peak': target_lambda_peak,
            'lambda_hf': target_lambda_hf,
            'lambda_anchor': target_lambda_anchor,
            'lambda_gate': target_lambda_gate,
            'lambda_gate_sep': target_lambda_gate_sep,
        }
        
        self.min = {
            'lambda_insulation': min_lambda_insulation,
            'lambda_peak': min_lambda_peak,
            'lambda_hf': min_lambda_hf,
            'lambda_anchor': min_lambda_anchor,
            'lambda_gate': min_lambda_gate,
            'lambda_gate_sep': min_lambda_gate_sep,
        }
    
    def get_weights(self, epoch):
        weights = {}
        
        for key in self.target:
            if epoch <= self.warmup_epochs:
                weights[key] = self.min[key]
            elif epoch <= self.warmup_epochs + self.rampup_epochs:
                progress = (epoch - self.warmup_epochs) / self.rampup_epochs
                weights[key] = self.min[key] + progress * (self.target[key] - self.min[key])
            else:
                weights[key] = self.target[key]
        
        return weights
    
    def get_stage(self, epoch):
        if epoch <= self.warmup_epochs:
            return 1, "Warmup (res+dir only, structure≈0)"
        elif epoch <= self.warmup_epochs + self.rampup_epochs:
            return 2, "Rampup (gradually add structure)"
        else:
            return 3, "Full (all losses active)"


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
    print(f"\nResidual statistics: mean={res_mean:.6f}, std={res_std:.6f}")
    return res_mean, res_std


def compute_condition_stats(hicarn):
    cond_mean = float(hicarn.mean())
    cond_std = float(hicarn.std())
    print(f"Condition statistics: mean={cond_mean:.6f}, std={cond_std:.6f}")
    return cond_mean, cond_std


# ================================================================
# Training v11
# ================================================================

def train_epoch_v11(
    model, scheduler, optimizer, dataloader, device,
    res_mean, res_std, cond_mean, cond_std, epoch,
    alpha=0.15,
    lambda_res=10.0,
    lambda_dir=2.0,  # 默认更高
    lambda_recon=0.5,
    lambda_insulation=0.1,
    lambda_peak=0.5,
    lambda_hf=0.2,
    lambda_anchor=0.5,
    lambda_gate=0.3,
    lambda_gate_sep=0.3,  # 新增
    gate_bg_target=0.12,
    gate_topk_ratio=0.02,
    gate_sep_margin=0.03,  # 新增
    insul_window=5,
    peak_protection_mode='adaptive',  # 默认改为 adaptive
    t_sampling='beta',
    t_p0=0.3,
    t_beta_a=0.5,
    t_beta_b=2.0,
):
    """v11 训练函数"""
    model.train()
    
    metrics = {
        'total': 0, 'diffusion': 0, 'residual': 0, 'direction': 0,
        'reconstruction': 0, 'insulation': 0, 'peak': 0, 'hf': 0,
        'anchor': 0, 'gate': 0, 'gate_sep': 0
    }
    
    gate_stats = {'peaks': 0, 'bg': 0, 'separation': 0}
    dir_corrs = []
    num_batches = 0
    
    cond_mean_t = torch.tensor(cond_mean, device=device)
    cond_std_t = torch.tensor(cond_std, device=device)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for hicarn, gt in pbar:
        hicarn = hicarn.to(device)
        gt = gt.to(device)
        batch_size = hicarn.shape[0]
        
        residual = gt - hicarn
        residual_centered = residual - res_mean
        residual_scaled = residual_centered / res_std
        
        t = sample_timesteps_biased(
            batch_size, scheduler.num_train_timesteps, device,
            strategy=t_sampling, p0=t_p0, beta_a=t_beta_a, beta_b=t_beta_b
        )
        
        noise = torch.randn_like(residual_scaled)
        residual_noisy = scheduler.add_noise(residual_scaled, noise, t)
        
        pred_v, gate = model(residual_noisy, hicarn, t, cond_mean_t, cond_std_t)
        
        # ============================================
        # 核心 Losses (始终有效)
        # ============================================
        target_v = scheduler.get_v_target(residual_scaled, noise, t)
        diff_loss = F.mse_loss(pred_v, target_v)
        
        pred_x0 = scheduler.predict_x0_from_v(residual_noisy, pred_v, t)
        
        dir_loss = cosine_direction_loss(pred_x0, residual_scaled)
        res_loss = F.l1_loss(pred_x0, residual_scaled)
        
        pred_residual = pred_x0 * res_std + res_mean
        if gate is not None:
            pred_final = hicarn + alpha * gate * pred_residual
        else:
            pred_final = hicarn + alpha * pred_residual
        recon_loss = F.mse_loss(pred_final, gt)
        
        # ============================================
        # 结构保护 Losses (warmup 时权重极低)
        # ============================================
        ins_loss = insulation_loss(pred_final, gt, window_size=insul_window)
        peak_loss = peak_preservation_loss(pred_final, gt, topk_ratio=gate_topk_ratio, mode=peak_protection_mode)
        hf_loss = high_frequency_loss(pred_final, gt)
        anc_loss = anchor_loss(pred_final, gt)
        
        # Gate losses
        if gate is not None:
            gate_loss = gate_regularization_loss(gate, hicarn, gate_bg_target, gate_topk_ratio)
            gate_sep_loss, g_peaks, g_bg = gate_separation_loss(gate, hicarn, gate_sep_margin, gate_topk_ratio)
            gate_stats['peaks'] += g_peaks
            gate_stats['bg'] += g_bg
            gate_stats['separation'] += (g_bg - g_peaks)
        else:
            gate_loss = torch.tensor(0.0, device=device)
            gate_sep_loss = torch.tensor(0.0, device=device)
        
        # ============================================
        # Total loss
        # ============================================
        loss = (
            diff_loss +
            lambda_res * res_loss +
            lambda_dir * dir_loss +
            lambda_recon * recon_loss +
            lambda_insulation * ins_loss +
            lambda_peak * peak_loss +
            lambda_hf * hf_loss +
            lambda_anchor * anc_loss +
            lambda_gate * gate_loss +
            lambda_gate_sep * gate_sep_loss
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        metrics['total'] += loss.item()
        metrics['diffusion'] += diff_loss.item()
        metrics['residual'] += res_loss.item()
        metrics['direction'] += dir_loss.item()
        metrics['reconstruction'] += recon_loss.item()
        metrics['insulation'] += ins_loss.item()
        metrics['peak'] += peak_loss.item()
        metrics['hf'] += hf_loss.item()
        metrics['anchor'] += anc_loss.item()
        metrics['gate'] += gate_loss.item() if gate is not None else 0
        metrics['gate_sep'] += gate_sep_loss.item() if gate is not None else 0
        
        num_batches += 1
        
        with torch.no_grad():
            corr = F.cosine_similarity(
                pred_x0.view(batch_size, -1),
                residual_scaled.view(batch_size, -1),
                dim=1
            ).mean().item()
            dir_corrs.append(corr)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'corr': f'{corr:.3f}',
            'sep': f'{gate_stats["separation"]/max(1,num_batches):.3f}'
        })
    
    for k in metrics:
        metrics[k] /= num_batches
    metrics['train_dir_corr'] = np.mean(dir_corrs)
    
    for k in gate_stats:
        gate_stats[k] /= num_batches
    metrics['gate_stats'] = gate_stats
    
    return metrics


@torch.no_grad()
def validate_v11(
    model, scheduler, hicarn_val, gt_val,
    res_mean, res_std, cond_mean, cond_std, alpha, device,
    gate_topk_ratio=0.02,
    num_steps=20, seed=42
):
    """验证函数"""
    model.eval()
    torch.manual_seed(seed)
    
    n = min(200, len(hicarn_val))
    hicarn = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    cond_mean_t = torch.tensor(cond_mean, device=device)
    cond_std_t = torch.tensor(cond_std, device=device)
    
    x_t = torch.randn(n, 1, hicarn.shape[2], hicarn.shape[3], device=device)
    
    timesteps = torch.linspace(
        scheduler.num_train_timesteps - 1, 0, num_steps
    ).long().to(device)
    
    pred_x0 = None
    gate_final = None
    
    for i, t in enumerate(timesteps):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        
        pred_v, gate = model(x_t, hicarn, t_batch, cond_mean_t, cond_std_t)
        pred_x0 = scheduler.predict_x0_from_v(x_t, pred_v, t_batch)
        
        if i == len(timesteps) - 1:
            gate_final = gate
        
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_t = scheduler.alphas_cumprod[t]
            alpha_t_next = scheduler.alphas_cumprod[t_next]
            
            pred_noise = (x_t - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt()
            x_t = alpha_t_next.sqrt() * pred_x0 + (1 - alpha_t_next).sqrt() * pred_noise
        else:
            x_t = pred_x0
    
    pred_residual = pred_x0 * res_std + res_mean
    
    if gate_final is not None:
        pred_final = hicarn + alpha * gate_final * pred_residual
        gate_mean = gate_final.mean().item()
    else:
        pred_final = hicarn + alpha * pred_residual
        gate_mean = 1.0
    
    pred_final = torch.clamp(pred_final, -5, 5)
    
    # Peak diagnostics
    peak_mask, mask_diag = get_peak_mask_with_diagnostics(gt, gate_topk_ratio)
    bg_mask = 1 - peak_mask
    
    peak_mse_refined_vs_hicarn = ((pred_final - hicarn) ** 2 * peak_mask).sum() / (peak_mask.sum() + 1e-8)
    peak_mse_refined_vs_gt = ((pred_final - gt) ** 2 * peak_mask).sum() / (peak_mask.sum() + 1e-8)
    peak_mse_hicarn_vs_gt = ((hicarn - gt) ** 2 * peak_mask).sum() / (peak_mask.sum() + 1e-8)
    
    if peak_mse_hicarn_vs_gt > 1e-8:
        peak_improvement = (peak_mse_hicarn_vs_gt - peak_mse_refined_vs_gt) / peak_mse_hicarn_vs_gt
    else:
        peak_improvement = 0.0
    
    bg_mse_refined_vs_gt = ((pred_final - gt) ** 2 * bg_mask).sum() / (bg_mask.sum() + 1e-8)
    bg_mse_hicarn_vs_gt = ((hicarn - gt) ** 2 * bg_mask).sum() / (bg_mask.sum() + 1e-8)
    
    if bg_mse_hicarn_vs_gt > 1e-8:
        bg_improvement = (bg_mse_hicarn_vs_gt - bg_mse_refined_vs_gt) / bg_mse_hicarn_vs_gt
    else:
        bg_improvement = 0.0
    
    # Gate separation
    if gate_final is not None:
        gate_peaks = (gate_final * peak_mask).sum() / (peak_mask.sum() + 1e-8)
        gate_bg = (gate_final * bg_mask).sum() / (bg_mask.sum() + 1e-8)
        gate_separation = gate_bg - gate_peaks
    else:
        gate_peaks = gate_bg = gate_separation = torch.tensor(1.0)
    
    # 整体指标
    mse = F.mse_loss(pred_final, gt).item()
    mse_hicarn = F.mse_loss(hicarn, gt).item()
    
    final_np = pred_final.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = hicarn.cpu().numpy().flatten()
    
    pcc, _ = stats.pearsonr(final_np, gt_np)
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    residual_np = pred_residual.cpu().numpy()
    ideal_residual = (gt - hicarn).cpu().numpy()
    
    raw_res_corr, _ = stats.pearsonr(
        residual_np.flatten(), ideal_residual.flatten()
    )
    
    return {
        'mse': mse,
        'pcc': float(pcc),
        'mse_hicarn': mse_hicarn,
        'pcc_hicarn': float(pcc_hicarn),
        'raw_res_corr': float(raw_res_corr),
        'improved_over_hicarn': float(pcc) > float(pcc_hicarn),
        
        'gate_mean': float(gate_mean),
        'gate_peaks': float(gate_peaks.item() if torch.is_tensor(gate_peaks) else gate_peaks),
        'gate_bg': float(gate_bg.item() if torch.is_tensor(gate_bg) else gate_bg),
        'gate_separation': float(gate_separation.item() if torch.is_tensor(gate_separation) else gate_separation),
        
        'peak_coverage': mask_diag['actual_coverage'],
        'peak_mse_refined_vs_hicarn': peak_mse_refined_vs_hicarn.item(),
        'peak_mse_refined_vs_gt': peak_mse_refined_vs_gt.item(),
        'peak_improvement': float(peak_improvement.item() if torch.is_tensor(peak_improvement) else peak_improvement),
        'bg_improvement': float(bg_improvement.item() if torch.is_tensor(bg_improvement) else bg_improvement),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument('--train_hicarn', type=str, required=True)
    parser.add_argument('--train_gt', type=str, required=True)
    parser.add_argument('--val_hicarn', type=str, default=None)
    parser.add_argument('--val_gt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints_v11')
    
    # Training basics
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--cond_norm', type=str, default='learnable',
                       choices=['learnable', 'fixed', 'none'])
    parser.add_argument('--use_gate', action='store_true', default=True)
    parser.add_argument('--g_scale', type=float, default=0.5)
    
    # Core parameter
    parser.add_argument('--alpha', type=float, default=0.15)
    
    # Residual learning losses
    parser.add_argument('--lambda_res', type=float, default=10.0)
    parser.add_argument('--lambda_dir', type=float, default=2.0,
                       help='Direction loss weight (increased from 0.5)')
    parser.add_argument('--lambda_recon', type=float, default=0.5)
    
    # Structure protection losses
    parser.add_argument('--lambda_insulation', type=float, default=0.3)
    parser.add_argument('--lambda_peak', type=float, default=0.5)
    parser.add_argument('--lambda_hf', type=float, default=0.2)
    parser.add_argument('--lambda_anchor', type=float, default=0.5)
    parser.add_argument('--lambda_gate', type=float, default=0.3)
    parser.add_argument('--lambda_gate_sep', type=float, default=0.3,
                       help='Gate separation loss weight')
    
    # Gate parameters
    parser.add_argument('--gate_bg_target', type=float, default=0.12)
    parser.add_argument('--gate_topk_ratio', type=float, default=0.02)
    parser.add_argument('--gate_sep_margin', type=float, default=0.03,
                       help='Margin for gate separation (gate_peaks < gate_bg - margin)')
    
    # Insulation window
    parser.add_argument('--insul_window', type=int, default=5)
    
    # Peak protection mode
    parser.add_argument('--peak_protection_mode', type=str, default='adaptive',
                       choices=['strict', 'adaptive', 'none'],
                       help='Default changed to adaptive')
    
    # Time sampling
    parser.add_argument('--t_sampling', type=str, default='beta',
                       choices=['beta', 'mixed', 'uniform'])
    parser.add_argument('--t_p0', type=float, default=0.3)
    parser.add_argument('--t_beta_a', type=float, default=0.5)
    parser.add_argument('--t_beta_b', type=float, default=2.0)
    
    # Validation
    parser.add_argument('--val_steps', type=int, default=20)
    
    # Schedule parameters
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--rampup_epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    # Gate constraint check
    if args.gate_bg_target > args.g_scale:
        raise ValueError(
            f"gate_bg_target ({args.gate_bg_target}) must be <= g_scale ({args.g_scale})!"
        )
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Print configuration
    # ================================================================
    print("\n" + "="*80)
    print("DIFFUSION REFINEMENT v11 (Gate Separation + Minimal Warmup)")
    print("="*80)
    print(f"\nv11 关键改进:")
    print(f"  1. Gate 分离约束: gate_peaks < gate_bg - {args.gate_sep_margin}")
    print(f"  2. 极简 Warmup: 结构 loss 权重 ≈ 0.001")
    print(f"  3. lambda_dir 默认 {args.lambda_dir} (更高)")
    print(f"  4. peak_protection_mode 默认 '{args.peak_protection_mode}'")
    
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
    
    res_mean, res_std = compute_residual_stats(hicarn_train, gt_train)
    cond_mean, cond_std = compute_condition_stats(hicarn_train)
    
    train_dataset = TensorDataset(
        torch.from_numpy(hicarn_train).float(),
        torch.from_numpy(gt_train).float()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    # Create weight scheduler
    weight_scheduler = LossWeightSchedulerV11(
        warmup_epochs=args.warmup_epochs,
        rampup_epochs=args.rampup_epochs,
        target_lambda_insulation=args.lambda_insulation,
        target_lambda_peak=args.lambda_peak,
        target_lambda_hf=args.lambda_hf,
        target_lambda_anchor=args.lambda_anchor,
        target_lambda_gate=args.lambda_gate,
        target_lambda_gate_sep=args.lambda_gate_sep,
    )
    
    # ================================================================
    # Model
    # ================================================================
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = GatedConditionedUNet(
        in_channels=2,
        out_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4),
        parameterization='v',
        cond_norm_type=args.cond_norm,
        output_gate=args.use_gate,
        g_scale=args.g_scale
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_timesteps,
        parameterization='v'
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Baseline
    mse_baseline = np.mean((hicarn_val - gt_val) ** 2)
    pcc_baseline, _ = stats.pearsonr(hicarn_val.flatten(), gt_val.flatten())
    print(f"\nBaseline: MSE={mse_baseline:.6f}, PCC={pcc_baseline:.4f}")
    
    # ================================================================
    # Training
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_pcc = pcc_baseline
    best_raw_res_corr = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        current_weights = weight_scheduler.get_weights(epoch)
        stage_num, stage_name = weight_scheduler.get_stage(epoch)
        
        if epoch == 1 or epoch == args.warmup_epochs + 1 or epoch == args.warmup_epochs + args.rampup_epochs + 1:
            print(f"\n  >>> Stage {stage_num}: {stage_name}")
            print(f"      Weights: ins={current_weights['lambda_insulation']:.4f}, "
                  f"peak={current_weights['lambda_peak']:.4f}, "
                  f"gate_sep={current_weights['lambda_gate_sep']:.4f}")
        
        losses = train_epoch_v11(
            model, scheduler, optimizer, train_loader,
            device, res_mean, res_std, cond_mean, cond_std, epoch,
            alpha=args.alpha,
            lambda_res=args.lambda_res,
            lambda_dir=args.lambda_dir,
            lambda_recon=args.lambda_recon,
            lambda_insulation=current_weights['lambda_insulation'],
            lambda_peak=current_weights['lambda_peak'],
            lambda_hf=current_weights['lambda_hf'],
            lambda_anchor=current_weights['lambda_anchor'],
            lambda_gate=current_weights['lambda_gate'],
            lambda_gate_sep=current_weights['lambda_gate_sep'],
            gate_bg_target=args.gate_bg_target,
            gate_topk_ratio=args.gate_topk_ratio,
            gate_sep_margin=args.gate_sep_margin,
            insul_window=args.insul_window,
            peak_protection_mode=args.peak_protection_mode,
            t_sampling=args.t_sampling,
            t_p0=args.t_p0,
            t_beta_a=args.t_beta_a,
            t_beta_b=args.t_beta_b
        )
        
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate_v11(
                model, scheduler, hicarn_val, gt_val,
                res_mean, res_std, cond_mean, cond_std, args.alpha, device,
                gate_topk_ratio=args.gate_topk_ratio,
                num_steps=args.val_steps
            )
            
            improved = ""
            if val_metrics['raw_res_corr'] > best_raw_res_corr:
                best_raw_res_corr = val_metrics['raw_res_corr']
                improved += " [best corr]"
            
            if val_metrics['pcc'] > best_pcc:
                best_pcc = val_metrics['pcc']
                improved += " [best PCC]"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_metrics': val_metrics,
                    'res_mean': res_mean, 'res_std': res_std,
                    'cond_mean': cond_mean, 'cond_std': cond_std,
                    'alpha': args.alpha,
                    'config': vars(args)
                }, output_dir / 'best_model.pt')
            
            status_corr = "✓" if val_metrics['raw_res_corr'] > 0.10 else "⚠"
            status_pcc = "✓" if val_metrics['improved_over_hicarn'] else "⚠"
            
            print(f"\n  Epoch {epoch} (Stage {stage_num}):")
            print(f"    MSE: {val_metrics['mse']:.6f} (HiCARN: {val_metrics['mse_hicarn']:.6f}) {status_pcc}")
            print(f"    PCC: {val_metrics['pcc']:.4f} (HiCARN: {val_metrics['pcc_hicarn']:.4f})")
            print(f"    raw_res_corr: {val_metrics['raw_res_corr']:.4f} {status_corr}")
            
            # Gate separation (关键新增!)
            sep_status = "✓" if val_metrics['gate_separation'] > args.gate_sep_margin else "⚠"
            print(f"    gate: peaks={val_metrics['gate_peaks']:.3f}, bg={val_metrics['gate_bg']:.3f}, "
                  f"separation={val_metrics['gate_separation']:.3f} {sep_status}")
            
            print(f"    peak_improvement: {val_metrics['peak_improvement']*100:+.2f}%")
            print(f"    bg_improvement:   {val_metrics['bg_improvement']*100:+.2f}%")
            
            if improved:
                print(f"    {improved}")
            
            # 早期诊断
            if epoch <= 5 and val_metrics['raw_res_corr'] < 0.05:
                print(f"    ⚠️ raw_res_corr 很低! 建议尝试:")
                print(f"       --peak_protection_mode none")
                print(f"       --lambda_dir 5.0")
            
            history.append({
                'epoch': epoch,
                'stage': stage_num,
                'losses': losses,
                'val_metrics': val_metrics
            })
        
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': vars(args)
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=float)
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best PCC: {best_pcc:.4f} (baseline: {pcc_baseline:.4f})")
    print(f"Best raw_res_corr: {best_raw_res_corr:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
