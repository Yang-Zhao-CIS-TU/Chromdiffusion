#!/usr/bin/env python3
"""
Diffusion Refinement v9 - 完整版 + 关键收尾修复

================================================================================
v8 → v9 关键修复：

1. Insulation window 可调 (之前写死为 5)
   - 新增 --insul_window 参数，可设置 5/7/9 等

2. Gate 约束检查 (gate_bg_target <= g_scale)
   - 启动时检查 gate_bg_target 是否可达
   - 如果不可达会报错并给出建议

3. 动态 epoch schedule (之前是静态 clamp)
   - Stage 1 (前 warmup_epochs): 结构 loss 权重很小，专注残差学习
   - Stage 2 (warmup_epochs 后): 线性 warmup 结构 loss 权重
   - 这样才能"先学残差、再逐步加结构约束"

4. 验证时噪声初始化使用 residual 尺度 (更稳健)

================================================================================
训练策略：

  阶段1 (epoch 1 ~ warmup_epochs):
    - 结构 loss 权重极低（只做"防伤"）
    - 专注让 raw_res_corr > 0.10

  阶段2 (epoch warmup_epochs+1 ~ warmup_epochs + rampup_epochs):
    - 结构 loss 权重线性增加
    - 逐步加入 TAD/Loop 保护

  阶段3 (epoch > warmup_epochs + rampup_epochs):
    - 结构 loss 权重达到目标值
    - 确保 TAD/Loop 不掉

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
# Model with Condition Normalization + Gate
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
    """
    带 Gate 机制的 UNet
    """
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
        
        # Condition normalization
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
        
        # Encoder
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
        
        # Middle
        self.mid1 = ResBlock(ch, ch, time_emb_dim)
        self.mid2 = ResBlock(ch, ch, time_emb_dim)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            self.decoder.append(ResBlock(ch + channels.pop(), out_ch, time_emb_dim))
            self.decoder.append(ResBlock(out_ch, out_ch, time_emb_dim))
            ch = out_ch
        
        # Output heads
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
    """计算 insulation score (TAD 边界检测)"""
    B, C, H, W = matrix.shape
    assert H == W, "Matrix must be square"
    
    insulation = torch.zeros(B, H, device=matrix.device)
    
    for i in range(window_size, H - window_size):
        start = max(0, i - window_size)
        end = min(H, i + window_size)
        local_sum = matrix[:, 0, start:end, start:end].mean(dim=(1, 2))
        insulation[:, i] = local_sum
    
    return insulation


def insulation_loss(pred_matrix, gt_matrix, window_size=5):
    """Insulation score loss - 保护 TAD 边界 (window_size 可调)"""
    pred_ins = compute_insulation_score(pred_matrix, window_size)
    gt_ins = compute_insulation_score(gt_matrix, window_size)
    return F.mse_loss(pred_ins, gt_ins)


def get_peak_mask(matrix, topk_ratio=0.02):
    """获取峰值区域的 mask"""
    B, C, H, W = matrix.shape
    flat = matrix.view(B, -1)
    
    k = max(1, int(flat.shape[1] * topk_ratio))
    topk_vals, _ = torch.topk(flat, k, dim=1)
    threshold = topk_vals[:, -1].view(B, 1, 1, 1)
    
    mask = (matrix >= threshold).float()
    return mask


def peak_preservation_loss(pred_matrix, gt_matrix, topk_ratio=0.02):
    """峰值保护 loss - 确保 loop 峰值不被破坏"""
    peak_mask = get_peak_mask(gt_matrix, topk_ratio)
    diff = (pred_matrix - gt_matrix) ** 2
    peak_loss = (diff * peak_mask).sum() / (peak_mask.sum() + 1e-8)
    return peak_loss


def high_frequency_loss(pred_matrix, gt_matrix):
    """高频信息保护 loss"""
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
    """Gate 正则化 loss"""
    peak_mask = get_peak_mask(cond, topk_ratio)
    bg_mask = 1 - peak_mask
    gate_target = gate_bg_target * bg_mask
    return F.mse_loss(gate, gate_target)


def anchor_loss(pred_matrix, gt_matrix):
    """Anchor loss - 整体结构一致性"""
    mean_loss = (pred_matrix.mean() - gt_matrix.mean()) ** 2
    std_loss = (pred_matrix.std() - gt_matrix.std()) ** 2
    return mean_loss + std_loss


# ================================================================
# Loss Weight Scheduler (动态 epoch schedule)
# ================================================================

class LossWeightScheduler:
    """
    动态调整结构 loss 权重
    
    Schedule:
    - epoch 1 ~ warmup_epochs: 使用 min_weight (专注残差学习)
    - epoch warmup_epochs+1 ~ warmup_epochs+rampup_epochs: 线性 warmup 到 target_weight
    - epoch > warmup_epochs+rampup_epochs: 使用 target_weight
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
        # 最小权重 (warmup 期间)
        min_lambda_insulation=0.02,
        min_lambda_peak=0.05,
        min_lambda_hf=0.02,
        min_lambda_anchor=0.1,
        min_lambda_gate=0.05,
    ):
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        
        self.target = {
            'lambda_insulation': target_lambda_insulation,
            'lambda_peak': target_lambda_peak,
            'lambda_hf': target_lambda_hf,
            'lambda_anchor': target_lambda_anchor,
            'lambda_gate': target_lambda_gate,
        }
        
        self.min = {
            'lambda_insulation': min_lambda_insulation,
            'lambda_peak': min_lambda_peak,
            'lambda_hf': min_lambda_hf,
            'lambda_anchor': min_lambda_anchor,
            'lambda_gate': min_lambda_gate,
        }
    
    def get_weights(self, epoch):
        """获取当前 epoch 的 loss 权重"""
        weights = {}
        
        for key in self.target:
            if epoch <= self.warmup_epochs:
                # Warmup 期间：使用最小权重
                weights[key] = self.min[key]
            elif epoch <= self.warmup_epochs + self.rampup_epochs:
                # Rampup 期间：线性增加
                progress = (epoch - self.warmup_epochs) / self.rampup_epochs
                weights[key] = self.min[key] + progress * (self.target[key] - self.min[key])
            else:
                # 之后：使用目标权重
                weights[key] = self.target[key]
        
        return weights
    
    def get_stage(self, epoch):
        """获取当前训练阶段"""
        if epoch <= self.warmup_epochs:
            return 1, "Warmup (focus on residual)"
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
# Training (v9 with dynamic schedule)
# ================================================================

def train_epoch_v9(
    model, scheduler, optimizer, dataloader, device,
    res_mean, res_std, cond_mean, cond_std, epoch,
    # Core parameters
    alpha=0.15,
    # Residual learning losses (fixed)
    lambda_res=10.0,
    lambda_dir=0.5,
    lambda_recon=0.5,
    # Structure protection losses (from scheduler)
    lambda_insulation=0.1,
    lambda_peak=0.5,
    lambda_hf=0.2,
    lambda_anchor=0.5,
    lambda_gate=0.3,
    # Gate parameters
    gate_bg_target=0.12,
    gate_topk_ratio=0.02,
    # Insulation window (可调!)
    insul_window=5,
    # Time sampling
    t_sampling='beta',
    t_p0=0.3,
    t_beta_a=0.5,
    t_beta_b=2.0,
):
    """v9 训练函数 - 支持可调 insulation window"""
    model.train()
    
    metrics = {
        'total': 0, 'diffusion': 0, 'residual': 0, 'direction': 0,
        'reconstruction': 0, 'insulation': 0, 'peak': 0, 'hf': 0,
        'anchor': 0, 'gate': 0
    }
    dir_corrs = []
    num_batches = 0
    
    cond_mean_t = torch.tensor(cond_mean, device=device)
    cond_std_t = torch.tensor(cond_std, device=device)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for hicarn, gt in pbar:
        hicarn = hicarn.to(device)
        gt = gt.to(device)
        batch_size = hicarn.shape[0]
        
        # 计算残差
        residual = gt - hicarn
        residual_centered = residual - res_mean
        residual_scaled = residual_centered / res_std
        
        # 采样 t
        t = sample_timesteps_biased(
            batch_size, scheduler.num_train_timesteps, device,
            strategy=t_sampling, p0=t_p0, beta_a=t_beta_a, beta_b=t_beta_b
        )
        
        # 加噪声
        noise = torch.randn_like(residual_scaled)
        residual_noisy = scheduler.add_noise(residual_scaled, noise, t)
        
        # 模型前向
        pred_v, gate = model(residual_noisy, hicarn, t, cond_mean_t, cond_std_t)
        
        # ============================================
        # 残差学习 Losses
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
        # 结构保护 Losses (使用可调 insul_window)
        # ============================================
        ins_loss = insulation_loss(pred_final, gt, window_size=insul_window)
        peak_loss = peak_preservation_loss(pred_final, gt, topk_ratio=gate_topk_ratio)
        hf_loss = high_frequency_loss(pred_final, gt)
        anc_loss = anchor_loss(pred_final, gt)
        
        if gate is not None:
            gate_loss = gate_regularization_loss(gate, hicarn, gate_bg_target, gate_topk_ratio)
        else:
            gate_loss = torch.tensor(0.0, device=device)
        
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
            lambda_gate * gate_loss
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
        num_batches += 1
        
        with torch.no_grad():
            corr = F.cosine_similarity(
                pred_x0.view(batch_size, -1),
                residual_scaled.view(batch_size, -1),
                dim=1
            ).mean().item()
            dir_corrs.append(corr)
        
        gate_mean = gate.mean().item() if gate is not None else 0
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'corr': f'{corr:.3f}',
            'gate': f'{gate_mean:.3f}'
        })
    
    for k in metrics:
        metrics[k] /= num_batches
    metrics['train_dir_corr'] = np.mean(dir_corrs)
    
    return metrics


@torch.no_grad()
def validate_v9(
    model, scheduler, hicarn_val, gt_val,
    res_mean, res_std, cond_mean, cond_std, alpha, device,
    num_steps=20, seed=42
):
    """验证函数 - 使用 residual 尺度初始化噪声"""
    model.eval()
    torch.manual_seed(seed)
    
    n = min(200, len(hicarn_val))
    hicarn = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    cond_mean_t = torch.tensor(cond_mean, device=device)
    cond_std_t = torch.tensor(cond_std, device=device)
    
    # ============================================
    # 修复：使用 residual 尺度初始化噪声 (更稳健)
    # 之前是 torch.randn_like(hicarn)，形状对但语义不一致
    # ============================================
    # 初始噪声应该是 residual_scaled 的尺度 (mean=0, std=1)
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
    
    # 应用残差
    pred_residual = pred_x0 * res_std + res_mean
    
    if gate_final is not None:
        pred_final = hicarn + alpha * gate_final * pred_residual
        gate_mean = gate_final.mean().item()
        peak_mask = get_peak_mask(hicarn)
        gate_peaks = (gate_final * peak_mask).sum() / (peak_mask.sum() + 1e-8)
        gate_bg = (gate_final * (1 - peak_mask)).sum() / ((1 - peak_mask).sum() + 1e-8)
    else:
        pred_final = hicarn + alpha * pred_residual
        gate_mean = 1.0
        gate_peaks = gate_bg = torch.tensor(1.0)
    
    pred_final = torch.clamp(pred_final, -5, 5)
    
    # Metrics
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
        'gate_mean': float(gate_mean),
        'gate_peaks': float(gate_peaks.item() if torch.is_tensor(gate_peaks) else gate_peaks),
        'gate_bg': float(gate_bg.item() if torch.is_tensor(gate_bg) else gate_bg),
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
    parser.add_argument('--output_dir', type=str, default='checkpoints_v9')
    
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
    parser.add_argument('--g_scale', type=float, default=0.5,
                       help='Gate output scale (must be >= gate_bg_target)')
    
    # Core parameter
    parser.add_argument('--alpha', type=float, default=0.15)
    
    # Residual learning losses
    parser.add_argument('--lambda_res', type=float, default=10.0)
    parser.add_argument('--lambda_dir', type=float, default=0.5)
    parser.add_argument('--lambda_recon', type=float, default=0.5)
    
    # Structure protection losses (目标值，会被 schedule 控制)
    parser.add_argument('--lambda_insulation', type=float, default=0.3,
                       help='Target insulation loss weight')
    parser.add_argument('--lambda_peak', type=float, default=0.5,
                       help='Target peak preservation loss weight')
    parser.add_argument('--lambda_hf', type=float, default=0.2,
                       help='Target high-frequency loss weight')
    parser.add_argument('--lambda_anchor', type=float, default=0.5,
                       help='Target anchor loss weight')
    parser.add_argument('--lambda_gate', type=float, default=0.3,
                       help='Target gate regularization weight')
    
    # Gate parameters
    parser.add_argument('--gate_bg_target', type=float, default=0.12,
                       help='Target gate value for background (must be <= g_scale)')
    parser.add_argument('--gate_topk_ratio', type=float, default=0.02)
    
    # Insulation window (新增可调!)
    parser.add_argument('--insul_window', type=int, default=5,
                       help='Insulation score window size (5/7/9 for different TAD scales)')
    
    # Time sampling
    parser.add_argument('--t_sampling', type=str, default='beta',
                       choices=['beta', 'mixed', 'uniform'])
    parser.add_argument('--t_p0', type=float, default=0.3)
    parser.add_argument('--t_beta_a', type=float, default=0.5)
    parser.add_argument('--t_beta_b', type=float, default=2.0)
    
    # Validation
    parser.add_argument('--val_steps', type=int, default=20)
    
    # Schedule parameters (新增!)
    parser.add_argument('--warmup_epochs', type=int, default=20,
                       help='Epochs to warmup (focus on residual)')
    parser.add_argument('--rampup_epochs', type=int, default=10,
                       help='Epochs to rampup structure loss weights')
    
    args = parser.parse_args()
    
    # ============================================
    # 修复2: 检查 gate_bg_target <= g_scale
    # ============================================
    if args.gate_bg_target > args.g_scale:
        raise ValueError(
            f"gate_bg_target ({args.gate_bg_target}) must be <= g_scale ({args.g_scale})!\n"
            f"Gate output is sigmoid * g_scale, so max gate value is {args.g_scale}.\n"
            f"If gate_bg_target > g_scale, the target is unreachable.\n"
            f"Suggestion: either increase g_scale or decrease gate_bg_target."
        )
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Print configuration
    # ================================================================
    print("\n" + "="*80)
    print("DIFFUSION REFINEMENT v9 (with dynamic schedule)")
    print("="*80)
    print(f"\nv9 关键修复:")
    print(f"  1. Insulation window 可调: {args.insul_window}")
    print(f"  2. Gate 约束检查: gate_bg_target ({args.gate_bg_target}) <= g_scale ({args.g_scale}) ✓")
    print(f"  3. 动态 epoch schedule:")
    print(f"     - Warmup epochs: {args.warmup_epochs} (专注残差)")
    print(f"     - Rampup epochs: {args.rampup_epochs} (逐步加结构)")
    print(f"  4. 验证噪声用 residual 尺度初始化")
    
    print(f"\n目标 loss 权重:")
    print(f"  lambda_insulation: {args.lambda_insulation}")
    print(f"  lambda_peak: {args.lambda_peak}")
    print(f"  lambda_hf: {args.lambda_hf}")
    print(f"  lambda_anchor: {args.lambda_anchor}")
    print(f"  lambda_gate: {args.lambda_gate}")
    
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
    
    # ================================================================
    # Create loss weight scheduler
    # ================================================================
    weight_scheduler = LossWeightScheduler(
        warmup_epochs=args.warmup_epochs,
        rampup_epochs=args.rampup_epochs,
        target_lambda_insulation=args.lambda_insulation,
        target_lambda_peak=args.lambda_peak,
        target_lambda_hf=args.lambda_hf,
        target_lambda_anchor=args.lambda_anchor,
        target_lambda_gate=args.lambda_gate,
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
    print(f"Gate enabled: {args.use_gate}, g_scale: {args.g_scale}")
    
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_timesteps,
        parameterization='v'
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # ================================================================
    # Baseline
    # ================================================================
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
        # ============================================
        # 修复3: 获取当前 epoch 的动态权重
        # ============================================
        current_weights = weight_scheduler.get_weights(epoch)
        stage_num, stage_name = weight_scheduler.get_stage(epoch)
        
        # 打印 stage 变化
        if epoch == 1 or epoch == args.warmup_epochs + 1 or epoch == args.warmup_epochs + args.rampup_epochs + 1:
            print(f"\n  >>> Stage {stage_num}: {stage_name}")
            print(f"      Weights: ins={current_weights['lambda_insulation']:.3f}, "
                  f"peak={current_weights['lambda_peak']:.3f}, "
                  f"gate={current_weights['lambda_gate']:.3f}")
        
        losses = train_epoch_v9(
            model, scheduler, optimizer, train_loader,
            device, res_mean, res_std, cond_mean, cond_std, epoch,
            alpha=args.alpha,
            lambda_res=args.lambda_res,
            lambda_dir=args.lambda_dir,
            lambda_recon=args.lambda_recon,
            # 使用动态权重
            lambda_insulation=current_weights['lambda_insulation'],
            lambda_peak=current_weights['lambda_peak'],
            lambda_hf=current_weights['lambda_hf'],
            lambda_anchor=current_weights['lambda_anchor'],
            lambda_gate=current_weights['lambda_gate'],
            gate_bg_target=args.gate_bg_target,
            gate_topk_ratio=args.gate_topk_ratio,
            insul_window=args.insul_window,  # 可调!
            t_sampling=args.t_sampling,
            t_p0=args.t_p0,
            t_beta_a=args.t_beta_a,
            t_beta_b=args.t_beta_b
        )
        
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate_v9(
                model, scheduler, hicarn_val, gt_val,
                res_mean, res_std, cond_mean, cond_std, args.alpha, device,
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
            print(f"    gate: mean={val_metrics['gate_mean']:.3f}, peaks={val_metrics['gate_peaks']:.3f}, bg={val_metrics['gate_bg']:.3f}")
            print(f"    losses: ins={losses['insulation']:.4f}, peak={losses['peak']:.4f}")
            print(f"    weights: ins={current_weights['lambda_insulation']:.3f}, peak={current_weights['lambda_peak']:.3f}")
            
            if improved:
                print(f"    {improved}")
            
            history.append({
                'epoch': epoch,
                'stage': stage_num,
                'losses': losses,
                'weights': current_weights,
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
    
    if best_raw_res_corr > 0.10:
        print("\n✓ raw_res_corr 达标!")
        print("  下一步:")
        print("  1. 每 5 epoch 跑 benchmark_combined_TK.py 检查 TAD/Loop")
        print("  2. 如果 TAD/Loop 不掉，可以尝试增大 alpha")
    else:
        print("\n⚠ raw_res_corr 未达标")
        print("  建议:")
        print("  1. 增大 warmup_epochs (让模型有更多时间学残差)")
        print("  2. 增大 lambda_dir (方向 loss)")
        print("  3. 检查数据质量")


if __name__ == '__main__':
    main()
