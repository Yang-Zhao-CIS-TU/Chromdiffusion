#!/usr/bin/env python3
"""
Diffusion Refinement v14 - Top-K Overlap Metrics + Structure-Oriented Training

================================================================================
v13 → v14 关键改进：

A. Top-K Overlap 指标 (用于验证和选择 best model)
   - IoU@K: 预测和GT的top-k像素重叠度
   - Precision@K, Recall@K
   - 支持多个 K 值 (1%, 0.5%, 0.1%)
   - 在 upper triangle, |i-j| >= min_diag 区域计算

B. 双 Best Model 保存策略
   - best_model_pcc.pt: 按 PCC 选择 (全局质量)
   - best_model_iou.pt: 按 IoU@K 选择 (loop/TAD 精度)
   - 同时保存，用户可选择合适的

C. 改进的超参数默认值 (结构导向)
   - 更高的 lambda_insulation, lambda_peak
   - 更合理的 gate 正则化参数
   - 混合时间采样策略

D. RAW 空间评估 (可选)
   - 支持加载 preprocessor 进行反归一化
   - 在 raw 空间计算 top-k overlap

================================================================================
用法:
    python train_v14.py \
      --train_hicarn /path/to/predictions_norm.npy \
      --train_gt /path/to/ground_truth.npy \
      --epochs 100 \
      --best_metric iou_0.5  # 用 IoU@0.5% 选择 best model

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
    
    def add_noise(self, x0, noise, timesteps):
        sqrt_alpha = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    def get_velocity(self, x0, noise, timesteps):
        sqrt_alpha = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x0
    
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
# Helper Functions
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


def get_peak_mask(matrix, topk_ratio=0.02):
    """Get mask for top-k% highest values"""
    B, C, H, W = matrix.shape
    flat = matrix.view(B, -1)
    
    k = max(1, int(flat.shape[1] * topk_ratio))
    topk_vals, _ = torch.topk(flat, k, dim=1)
    threshold = topk_vals[:, -1].view(B, 1, 1, 1)
    
    mask = (matrix >= threshold).float()
    return mask


def get_topk_abs_mask(matrix, topk_ratio=0.05):
    """Get mask for top-k% highest absolute values"""
    B, C, H, W = matrix.shape
    flat = matrix.abs().view(B, -1)
    
    k = max(1, int(flat.shape[1] * topk_ratio))
    topk_vals, _ = torch.topk(flat, k, dim=1)
    threshold = topk_vals[:, -1].view(B, 1, 1, 1)
    
    mask = (matrix.abs() >= threshold).float()
    return mask


def get_diag_mask(H, W, min_diag=2, upper=True, device='cpu'):
    """
    Get diagonal mask for Hi-C matrix.
    Excludes pixels within min_diag of diagonal, keeps upper triangle if upper=True.
    """
    i_idx = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
    j_idx = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
    
    # Distance from diagonal
    diag_dist = (j_idx - i_idx).abs()
    
    # Mask: far enough from diagonal
    mask = (diag_dist >= min_diag).float()
    
    # Upper triangle only
    if upper:
        mask = mask * (j_idx >= i_idx).float()
    
    return mask


# ================================================================
# Top-K Overlap Metrics (核心新增)
# ================================================================

def compute_topk_overlap_single(pred, gt, k_perc, min_diag=2, upper=True):
    """
    Compute top-k overlap for a single sample.
    
    Args:
        pred: (H, W) numpy array - predicted matrix
        gt: (H, W) numpy array - ground truth matrix
        k_perc: float - percentage (e.g., 0.5 for 0.5%)
        min_diag: int - minimum distance from diagonal
        upper: bool - whether to use upper triangle only
    
    Returns:
        dict with iou, precision, recall, hit1
    """
    H, W = pred.shape
    
    # Create diagonal mask
    i_idx, j_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    diag_dist = np.abs(j_idx - i_idx)
    
    mask = diag_dist >= min_diag
    if upper:
        mask = mask & (j_idx >= i_idx)
    
    # Get valid pixels
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    n_valid = len(pred_valid)
    
    if n_valid < 10:
        return {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'hit1': 0.0}
    
    # Compute K
    K = max(1, int(n_valid * k_perc / 100.0))
    
    # Get top-K indices
    pred_topk_idx = set(np.argsort(pred_valid)[-K:])
    gt_topk_idx = set(np.argsort(gt_valid)[-K:])
    
    # Compute metrics
    intersection = len(pred_topk_idx & gt_topk_idx)
    union = len(pred_topk_idx | gt_topk_idx)
    
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / K if K > 0 else 0.0
    recall = intersection / K if K > 0 else 0.0  # Same as precision when both have K elements
    
    # Hit@1: is GT's top-1 in pred's top-K?
    gt_top1_idx = np.argmax(gt_valid)
    hit1 = 1.0 if gt_top1_idx in pred_topk_idx else 0.0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'hit1': hit1
    }


def compute_topk_overlap_batch(pred_batch, gt_batch, k_perc_list=[1.0, 0.5, 0.1], min_diag=2, upper=True):
    """
    Compute top-k overlap for a batch of samples.
    
    Args:
        pred_batch: (N, H, W) or (N, 1, H, W) numpy array
        gt_batch: (N, H, W) or (N, 1, H, W) numpy array
        k_perc_list: list of percentages to compute
        min_diag: minimum distance from diagonal
        upper: use upper triangle only
    
    Returns:
        dict with mean/median for each k_perc and metric
    """
    # Squeeze to (N, H, W)
    if pred_batch.ndim == 4:
        pred_batch = pred_batch[:, 0, :, :]
    if gt_batch.ndim == 4:
        gt_batch = gt_batch[:, 0, :, :]
    
    N = len(pred_batch)
    
    results = {f'iou_{k}': [] for k in k_perc_list}
    results.update({f'prec_{k}': [] for k in k_perc_list})
    results.update({f'hit1_{k}': [] for k in k_perc_list})
    
    for i in range(N):
        for k in k_perc_list:
            metrics = compute_topk_overlap_single(
                pred_batch[i], gt_batch[i], k, min_diag, upper
            )
            results[f'iou_{k}'].append(metrics['iou'])
            results[f'prec_{k}'].append(metrics['precision'])
            results[f'hit1_{k}'].append(metrics['hit1'])
    
    # Aggregate
    agg_results = {}
    for key, values in results.items():
        agg_results[f'{key}_mean'] = float(np.mean(values))
        agg_results[f'{key}_median'] = float(np.median(values))
    
    return agg_results


# ================================================================
# Loss Functions
# ================================================================

def insulation_score(matrix, window_size=5):
    """Compute insulation score for TAD boundary detection"""
    kernel = torch.ones(1, 1, window_size, window_size, device=matrix.device)
    kernel = kernel / (window_size * window_size)
    
    insulation = F.conv2d(matrix, kernel, padding=window_size//2)
    return insulation


def insulation_loss(pred_matrix, gt_matrix, window_size=5):
    """Loss to match insulation scores (TAD structure)"""
    pred_insul = insulation_score(pred_matrix, window_size)
    gt_insul = insulation_score(gt_matrix, window_size)
    return F.mse_loss(pred_insul, gt_insul)


def tad_boundary_loss(pred_matrix, gt_matrix, window_size=5):
    """
    Loss specifically for TAD boundary sharpness.
    Computes gradient of insulation score (boundaries are where gradient is high).
    """
    pred_insul = insulation_score(pred_matrix, window_size)
    gt_insul = insulation_score(gt_matrix, window_size)
    
    # Sobel for gradient
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=pred_matrix.dtype, device=pred_matrix.device)
    sobel_y = sobel_x.t()
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    # Gradient of insulation score
    pred_gx = F.conv2d(pred_insul, sobel_x, padding=1)
    pred_gy = F.conv2d(pred_insul, sobel_y, padding=1)
    pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8)
    
    gt_gx = F.conv2d(gt_insul, sobel_x, padding=1)
    gt_gy = F.conv2d(gt_insul, sobel_y, padding=1)
    gt_grad = torch.sqrt(gt_gx**2 + gt_gy**2 + 1e-8)
    
    return F.mse_loss(pred_grad, gt_grad)


def loop_position_loss(pred_matrix, gt_matrix, topk_ratio=0.01, temperature=0.1):
    """
    Loss for loop peak position accuracy.
    Uses soft top-k to encourage peaks at correct locations.
    """
    B, C, H, W = pred_matrix.shape
    
    # Get GT peak mask (strict top-k)
    gt_flat = gt_matrix.view(B, -1)
    k = max(1, int(gt_flat.shape[1] * topk_ratio))
    topk_vals, topk_idx = torch.topk(gt_flat, k, dim=1)
    
    # Create soft target: high at GT peak locations
    gt_peak_mask = torch.zeros_like(gt_flat)
    gt_peak_mask.scatter_(1, topk_idx, 1.0)
    gt_peak_mask = gt_peak_mask.view(B, C, H, W)
    
    # Soft prediction peaks (softmax over spatial)
    pred_flat = pred_matrix.view(B, -1)
    pred_soft = F.softmax(pred_flat / temperature, dim=1)
    pred_soft = pred_soft.view(B, C, H, W)
    
    # Loss: pred should have high probability at GT peak locations
    # Negative log likelihood style
    loss = -torch.mean(torch.log(pred_soft + 1e-8) * gt_peak_mask)
    
    return loss


def off_diagonal_structure_loss(pred_matrix, gt_matrix, min_diag=2):
    """
    Loss specifically for off-diagonal structure (loops are off-diagonal).
    """
    B, C, H, W = pred_matrix.shape
    device = pred_matrix.device
    
    # Create off-diagonal mask
    diag_mask = get_diag_mask(H, W, min_diag=min_diag, upper=True, device=device)
    diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    
    # Only compute loss on off-diagonal region
    pred_off = pred_matrix * diag_mask
    gt_off = gt_matrix * diag_mask
    
    # MSE on off-diagonal
    mse_loss = F.mse_loss(pred_off, gt_off)
    
    # Also match the relative ranking (important for loop detection)
    pred_flat = pred_off.view(B, -1)
    gt_flat = gt_off.view(B, -1)
    
    # Ranking loss via correlation
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    gt_centered = gt_flat - gt_flat.mean(dim=1, keepdim=True)
    
    corr = (pred_centered * gt_centered).sum(dim=1) / (
        pred_centered.norm(dim=1) * gt_centered.norm(dim=1) + 1e-8
    )
    rank_loss = 1.0 - corr.mean()
    
    return mse_loss + 0.5 * rank_loss


def peak_preservation_loss(pred_matrix, gt_matrix, topk_ratio=0.02):
    """Loss to preserve peak values"""
    peak_mask = get_peak_mask(gt_matrix, topk_ratio)
    pred_peaks = pred_matrix * peak_mask
    gt_peaks = gt_matrix * peak_mask
    return F.mse_loss(pred_peaks, gt_peaks)


def high_frequency_loss(pred_matrix, gt_matrix):
    """Loss to preserve high-frequency details (edges)"""
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


def gate_regularization_loss(gate, cond, gate_bg_target=0.10, topk_ratio=0.02):
    """Regularize gate to be low in background"""
    peak_mask = get_peak_mask(cond, topk_ratio)
    bg_mask = 1 - peak_mask
    gate_target = gate_bg_target * bg_mask
    return F.mse_loss(gate, gate_target)


def gate_separation_loss(gate, cond, margin=0.05, topk_ratio=0.02):
    """Force gate to differentiate peaks from background"""
    peak_mask = get_peak_mask(cond, topk_ratio)
    bg_mask = 1 - peak_mask
    
    gate_peaks = (gate * peak_mask).sum() / (peak_mask.sum() + 1e-8)
    gate_bg = (gate * bg_mask).sum() / (bg_mask.sum() + 1e-8)
    
    # We want gate_peaks < gate_bg - margin (gate should be LOW on peaks)
    # Or alternatively: gate_peaks > gate_bg + margin (gate HIGH on peaks)
    # Let's use: gate should be HIGHER on peaks
    sep_loss = F.relu(gate_bg - gate_peaks + margin)
    
    return sep_loss, gate_peaks.item(), gate_bg.item()


def anchor_loss(pred_matrix, gt_matrix):
    """Loss to match global statistics"""
    mean_loss = (pred_matrix.mean() - gt_matrix.mean()) ** 2
    std_loss = (pred_matrix.std() - gt_matrix.std()) ** 2
    return mean_loss + std_loss


def compute_residual_weight_map(residual_scaled, topk_ratio=0.05, peak_weight=3.0, bg_weight=1.0):
    """
    Compute weight map for residual loss.
    Higher weight on structure regions (high |residual|).
    """
    topk_mask = get_topk_abs_mask(residual_scaled, topk_ratio)
    weight_map = bg_weight + (peak_weight - bg_weight) * topk_mask
    return weight_map


def weighted_l1_loss(pred, target, weight_map):
    """Weighted L1 loss"""
    return (torch.abs(pred - target) * weight_map).mean()


# ================================================================
# Loss Weight Scheduler
# ================================================================

class LossWeightSchedulerV14:
    """
    v14: Alpha Warmup + Gate Warmup + Structure-Oriented Defaults
    
    核心修复:
    1. Alpha ramp 放慢 (rampup_epochs 30, alpha 0.12-0.15)
    2. Gate 也做 warmup (不要一上来就全开)
    3. 残差权重更保守 (peak 6.0, bg 1.0)
    """
    def __init__(
        self,
        warmup_epochs=20,
        rampup_epochs=30,          # 修复A: 放慢 ramp (从10→30)
        gate_warmup_epochs=15,     # 修复B: gate 也做 warmup
        alpha_warmup=0.05,
        alpha_full=0.12,           # 修复A: 降低目标 alpha (从0.2→0.12)
        # 结构导向的默认值
        target_lambda_insulation=1.0,
        target_lambda_tad_boundary=0.5,
        target_lambda_peak=1.5,
        target_lambda_loop_pos=0.3,
        target_lambda_offdiag=0.3,
        target_lambda_hf=0.5,
        target_lambda_anchor=0.2,
        target_lambda_gate=0.2,
        target_lambda_gate_sep=0.8,
        min_lambda_insulation=0.001,
        min_lambda_tad_boundary=0.001,
        min_lambda_peak=0.001,
        min_lambda_loop_pos=0.001,
        min_lambda_offdiag=0.001,
        min_lambda_hf=0.001,
        min_lambda_anchor=0.01,
        min_lambda_gate=0.001,
        min_lambda_gate_sep=0.01,
    ):
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.gate_warmup_epochs = gate_warmup_epochs
        self.alpha_warmup = alpha_warmup
        self.alpha_full = alpha_full
        
        self.target = {
            'lambda_insulation': target_lambda_insulation,
            'lambda_tad_boundary': target_lambda_tad_boundary,
            'lambda_peak': target_lambda_peak,
            'lambda_loop_pos': target_lambda_loop_pos,
            'lambda_offdiag': target_lambda_offdiag,
            'lambda_hf': target_lambda_hf,
            'lambda_anchor': target_lambda_anchor,
            'lambda_gate': target_lambda_gate,
            'lambda_gate_sep': target_lambda_gate_sep,
        }
        
        self.min = {
            'lambda_insulation': min_lambda_insulation,
            'lambda_tad_boundary': min_lambda_tad_boundary,
            'lambda_peak': min_lambda_peak,
            'lambda_loop_pos': min_lambda_loop_pos,
            'lambda_offdiag': min_lambda_offdiag,
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
    
    def get_alpha(self, epoch):
        """Alpha warmup: 更慢的 ramp"""
        if epoch <= self.warmup_epochs:
            return self.alpha_warmup
        elif epoch <= self.warmup_epochs + self.rampup_epochs:
            progress = (epoch - self.warmup_epochs) / self.rampup_epochs
            return self.alpha_warmup + progress * (self.alpha_full - self.alpha_warmup)
        else:
            return self.alpha_full
    
    def get_gate_scale(self, epoch):
        """
        修复B: Gate warmup
        Stage1: gate_scale = 0 (完全 bypass)
        Stage2 前半: gate_scale 从 0 → 1 线性上升
        Stage2 后半 + Stage3: gate_scale = 1
        """
        if epoch <= self.warmup_epochs:
            return 0.0  # Stage1: gate bypass
        elif epoch <= self.warmup_epochs + self.gate_warmup_epochs:
            # Gate 慢慢开启
            progress = (epoch - self.warmup_epochs) / self.gate_warmup_epochs
            return progress
        else:
            return 1.0
    
    def use_gate_in_forward(self, epoch):
        return epoch > self.warmup_epochs
    
    def get_stage(self, epoch):
        if epoch <= self.warmup_epochs:
            return 1, f"Warmup (gate BYPASSED, alpha={self.alpha_warmup})"
        elif epoch <= self.warmup_epochs + self.rampup_epochs:
            gate_scale = self.get_gate_scale(epoch)
            return 2, f"Rampup (gate_scale={gate_scale:.2f}, alpha ramping)"
        else:
            return 3, f"Full (alpha={self.alpha_full})"


# ================================================================
# Data Loading
# ================================================================

def load_data(hicarn_path, gt_path):
    print(f"Loading HiCARN from: {hicarn_path}")
    hicarn = np.load(hicarn_path)
    hicarn = ensure_nchw(hicarn).astype(np.float32)
    
    print(f"Loading GT from: {gt_path}")
    gt = np.load(gt_path)
    gt = ensure_nchw(gt).astype(np.float32)
    
    print(f"  HiCARN: {hicarn.shape}, range [{hicarn.min():.4f}, {hicarn.max():.4f}]")
    print(f"  GT:     {gt.shape}, range [{gt.min():.4f}, {gt.max():.4f}]")
    
    return hicarn, gt


def compute_residual_stats(hicarn, gt):
    residual = gt - hicarn
    res_mean = residual.mean()
    res_std = residual.std()
    print(f"  Residual: mean={res_mean:.6f}, std={res_std:.6f}")
    return float(res_mean), float(res_std)


def compute_condition_stats(hicarn):
    cond_mean = hicarn.mean()
    cond_std = hicarn.std()
    print(f"  Condition: mean={cond_mean:.6f}, std={cond_std:.6f}")
    return float(cond_mean), float(cond_std)


# ================================================================
# Training Functions
# ================================================================

def sample_timesteps(batch_size, num_timesteps, device, method='mixed', p0=0.3, beta_a=2.0, beta_b=5.0):
    """
    Sample timesteps for training.
    
    Methods:
    - uniform: Uniform sampling
    - beta: Beta distribution (more samples at low/high noise)
    - mixed: Mix of uniform and importance sampling
    """
    if method == 'uniform':
        return torch.randint(0, num_timesteps, (batch_size,), device=device)
    
    elif method == 'beta':
        beta_samples = torch.distributions.Beta(beta_a, beta_b).sample((batch_size,))
        return (beta_samples * num_timesteps).long().to(device)
    
    elif method == 'mixed':
        # Mix: p0 probability of uniform, (1-p0) of beta
        uniform_mask = torch.rand(batch_size, device=device) < p0
        
        uniform_t = torch.randint(0, num_timesteps, (batch_size,), device=device)
        beta_samples = torch.distributions.Beta(beta_a, beta_b).sample((batch_size,)).to(device)
        beta_t = (beta_samples * num_timesteps).long()
        
        return torch.where(uniform_mask, uniform_t, beta_t)
    
    else:
        return torch.randint(0, num_timesteps, (batch_size,), device=device)


def train_epoch_v14(
    model, scheduler, optimizer, dataloader,
    device, res_mean, res_std, cond_mean, cond_std, epoch,
    alpha=0.12,
    gate_scale=1.0,               # 修复B: gate warmup scale
    lambda_res=10.0,
    lambda_dir=2.0,
    lambda_recon=0.3,
    lambda_insulation=1.0,
    lambda_tad_boundary=0.5,
    lambda_peak=1.5,
    lambda_loop_pos=0.3,
    lambda_offdiag=0.3,
    lambda_hf=0.5,
    lambda_anchor=0.2,
    lambda_gate=0.2,
    lambda_gate_sep=0.8,
    gate_bg_target=0.08,
    gate_topk_ratio=0.015,
    gate_sep_margin=0.08,
    insul_window=5,
    use_gate_in_forward=True,
    res_loss_weighted=True,
    res_weight_topk_ratio=0.05,
    res_weight_peak=6.0,          # 修复C: 更保守
    res_weight_bg=1.0,            # 修复C: 更保守
    t_sampling='mixed',
    t_p0=0.3,
    t_beta_a=2.0,
    t_beta_b=5.0,
    loop_topk_ratio=0.01,
    min_diag=2
):
    model.train()
    
    total_loss = 0
    loss_components = {
        'diff': 0, 'res': 0, 'dir': 0, 'recon': 0,
        'insul': 0, 'tad_bnd': 0, 'peak': 0, 'loop_pos': 0, 
        'offdiag': 0, 'hf': 0, 'anchor': 0,
        'gate': 0, 'gate_sep': 0
    }
    
    cond_mean_t = torch.tensor(cond_mean, device=device)
    cond_std_t = torch.tensor(cond_std, device=device)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (hicarn, gt) in enumerate(pbar):
        hicarn = hicarn.to(device)
        gt = gt.to(device)
        B = hicarn.shape[0]
        
        # Compute target residual
        residual = gt - hicarn
        residual_scaled = (residual - res_mean) / res_std
        
        # Sample timesteps
        t = sample_timesteps(B, scheduler.num_train_timesteps, device, t_sampling, t_p0, t_beta_a, t_beta_b)
        
        # Initialize with residual-scale noise
        noise = torch.randn_like(residual_scaled) * 0.1 + residual_scaled * 0.05
        
        # Add noise
        x_t = scheduler.add_noise(residual_scaled, noise, t)
        
        # Forward
        pred_v, gate = model(x_t, hicarn, t, cond_mean_t, cond_std_t)
        
        # Get target v
        target_v = scheduler.get_velocity(residual_scaled, noise, t)
        
        # Diffusion loss
        diff_loss = F.mse_loss(pred_v, target_v)
        
        # Predict x0 for auxiliary losses
        pred_x0 = scheduler.predict_x0_from_v(x_t, pred_v, t)
        
        # Residual loss (weighted)
        if res_loss_weighted:
            weight_map = compute_residual_weight_map(
                residual_scaled, res_weight_topk_ratio, res_weight_peak, res_weight_bg
            )
            res_loss = weighted_l1_loss(pred_x0, residual_scaled, weight_map)
        else:
            res_loss = F.l1_loss(pred_x0, residual_scaled)
        
        # Direction loss
        pred_res_np = pred_x0.detach().cpu().numpy().flatten()
        target_res_np = residual_scaled.detach().cpu().numpy().flatten()
        if len(pred_res_np) > 10:
            corr, _ = stats.pearsonr(pred_res_np, target_res_np)
            dir_loss = torch.tensor(1.0 - max(0, corr), device=device)
        else:
            dir_loss = torch.tensor(0.0, device=device)
        
        # Reconstruct final prediction
        pred_residual = pred_x0 * res_std + res_mean
        
        # 修复B: gate_scale 控制 "gated vs ungated" 的混合比例
        # 而不是直接乘在 gate 上（那样会让 delta 接近 0）
        if use_gate_in_forward and gate is not None:
            gated_pred = hicarn + alpha * gate * pred_residual
            ungated_pred = hicarn + alpha * pred_residual
            # gate_scale=0: 完全 ungated; gate_scale=1: 完全 gated
            pred_final = gate_scale * gated_pred + (1 - gate_scale) * ungated_pred
        else:
            pred_final = hicarn + alpha * pred_residual
        
        pred_final = torch.clamp(pred_final, -5, 5)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(pred_final, gt)
        
        # Structure losses (核心: TAD + Loop)
        insul_loss = insulation_loss(pred_final, gt, insul_window)
        tad_bnd_loss = tad_boundary_loss(pred_final, gt, insul_window)  # 新增
        peak_loss = peak_preservation_loss(pred_final, gt, gate_topk_ratio)
        loop_pos_loss = loop_position_loss(pred_final, gt, loop_topk_ratio)  # 新增
        offdiag_loss = off_diagonal_structure_loss(pred_final, gt, min_diag)  # 新增
        hf_loss = high_frequency_loss(pred_final, gt)
        anch_loss = anchor_loss(pred_final, gt)
        
        # Gate losses
        if gate is not None:
            gate_reg_loss = gate_regularization_loss(gate, hicarn, gate_bg_target, gate_topk_ratio)
            gate_sep_loss_val, _, _ = gate_separation_loss(gate, hicarn, gate_sep_margin, gate_topk_ratio)
        else:
            gate_reg_loss = torch.tensor(0.0, device=device)
            gate_sep_loss_val = torch.tensor(0.0, device=device)
        
        # Total loss (结构导向权重)
        loss = (
            diff_loss +
            lambda_res * res_loss +
            lambda_dir * dir_loss +
            lambda_recon * recon_loss +
            lambda_insulation * insul_loss +
            lambda_tad_boundary * tad_bnd_loss +      # 新增
            lambda_peak * peak_loss +
            lambda_loop_pos * loop_pos_loss +          # 新增
            lambda_offdiag * offdiag_loss +            # 新增
            lambda_hf * hf_loss +
            lambda_anchor * anch_loss +
            lambda_gate * gate_reg_loss +
            lambda_gate_sep * gate_sep_loss_val
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        loss_components['diff'] += diff_loss.item()
        loss_components['res'] += res_loss.item()
        loss_components['dir'] += dir_loss.item()
        loss_components['recon'] += recon_loss.item()
        loss_components['insul'] += insul_loss.item()
        loss_components['tad_bnd'] += tad_bnd_loss.item()
        loss_components['peak'] += peak_loss.item()
        loss_components['loop_pos'] += loop_pos_loss.item()
        loss_components['offdiag'] += offdiag_loss.item()
        loss_components['hf'] += hf_loss.item()
        loss_components['anchor'] += anch_loss.item()
        loss_components['gate'] += gate_reg_loss.item()
        loss_components['gate_sep'] += gate_sep_loss_val.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'diff': f'{diff_loss.item():.4f}',
            'res': f'{res_loss.item():.4f}'
        })
    
    n_batches = len(dataloader)
    metrics = {k: v / n_batches for k, v in loss_components.items()}
    metrics['total'] = total_loss / n_batches
    
    return metrics


# ================================================================
# Validation Function (with Top-K Overlap)
# ================================================================

@torch.no_grad()
def validate_v14(
    model, scheduler, hicarn_val, gt_val,
    res_mean, res_std, cond_mean, cond_std, alpha, device,
    gate_scale=1.0,                # 新增: gate warmup scale
    gate_topk_ratio=0.02,
    res_corr_topk_ratio=0.05,
    use_gate_in_forward=True,
    num_steps=20, seed=42,
    topk_perc_list=[1.0, 0.5, 0.1],
    topk_min_diag=2,
    topk_upper=True
):
    """
    Validation with Top-K Overlap metrics.
    """
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
    
    # Restore residual
    pred_residual = pred_x0 * res_std + res_mean
    
    # Effective residual (with gate)
    if gate_final is not None:
        effective_residual = gate_final * pred_residual
        gate_mean = gate_final.mean().item()
    else:
        effective_residual = pred_residual
        gate_mean = 1.0
    
    # Apply to HiCARN (修复: 使用 gate_scale 混合)
    if use_gate_in_forward and gate_final is not None:
        gated_pred = hicarn + alpha * gate_final * pred_residual
        ungated_pred = hicarn + alpha * pred_residual
        pred_final = gate_scale * gated_pred + (1 - gate_scale) * ungated_pred
    else:
        pred_final = hicarn + alpha * pred_residual
    
    pred_final = torch.clamp(pred_final, -5, 5)
    
    # ============================================
    # Delta metrics (新增: 诊断模型实际改动量)
    # ============================================
    delta = pred_final - hicarn
    delta_mean = delta.abs().mean().item()
    delta_max = delta.abs().max().item()
    delta_std = delta.std().item()
    
    # Ideal residual
    ideal_residual = gt - hicarn
    
    # ============================================
    # Residual correlation metrics
    # ============================================
    pred_res_np = pred_residual.cpu().numpy().flatten()
    eff_res_np = effective_residual.cpu().numpy().flatten()
    ideal_res_np = ideal_residual.cpu().numpy().flatten()
    
    raw_res_corr, _ = stats.pearsonr(pred_res_np, ideal_res_np)
    eff_res_corr, _ = stats.pearsonr(eff_res_np, ideal_res_np)
    
    # Partitioned correlation
    topk_mask = get_topk_abs_mask(ideal_residual, res_corr_topk_ratio)
    topk_mask_np = topk_mask.cpu().numpy().flatten().astype(bool)
    
    if topk_mask_np.sum() > 10:
        corr_topk_raw, _ = stats.pearsonr(pred_res_np[topk_mask_np], ideal_res_np[topk_mask_np])
        corr_topk_eff, _ = stats.pearsonr(eff_res_np[topk_mask_np], ideal_res_np[topk_mask_np])
    else:
        corr_topk_raw = corr_topk_eff = 0.0
    
    # ============================================
    # Peak/Background metrics
    # ============================================
    peak_mask = get_peak_mask(gt, gate_topk_ratio)
    bg_mask = 1 - peak_mask
    
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
    
    # Gate statistics
    if gate_final is not None:
        gate_peaks = (gate_final * peak_mask).sum() / (peak_mask.sum() + 1e-8)
        gate_bg = (gate_final * bg_mask).sum() / (bg_mask.sum() + 1e-8)
        gate_separation = gate_peaks - gate_bg  # Positive if gate higher on peaks
    else:
        gate_peaks = gate_bg = gate_separation = torch.tensor(0.0)
    
    # ============================================
    # Global metrics
    # ============================================
    mse = F.mse_loss(pred_final, gt).item()
    mse_hicarn = F.mse_loss(hicarn, gt).item()
    
    final_np = pred_final.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = hicarn.cpu().numpy().flatten()
    
    pcc, _ = stats.pearsonr(final_np, gt_np)
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    # ============================================
    # Top-K Overlap (核心新增)
    # ============================================
    pred_final_np = pred_final.cpu().numpy()
    gt_np_4d = gt.cpu().numpy()
    hicarn_np_4d = hicarn.cpu().numpy()
    
    # Compute for refined
    topk_metrics_refined = compute_topk_overlap_batch(
        pred_final_np, gt_np_4d, topk_perc_list, topk_min_diag, topk_upper
    )
    
    # Compute for HiCARN (baseline comparison)
    topk_metrics_hicarn = compute_topk_overlap_batch(
        hicarn_np_4d, gt_np_4d, topk_perc_list, topk_min_diag, topk_upper
    )
    
    # Build metrics dict
    metrics = {
        'mse': mse,
        'pcc': float(pcc),
        'mse_hicarn': mse_hicarn,
        'pcc_hicarn': float(pcc_hicarn),
        'improved_over_hicarn': float(pcc) > float(pcc_hicarn),
        
        # Delta metrics (新增: 诊断模型实际改动量)
        'delta_mean': delta_mean,
        'delta_max': delta_max,
        'delta_std': delta_std,
        
        # Residual correlation
        'raw_res_corr': float(raw_res_corr),
        'eff_res_corr': float(eff_res_corr),
        'corr_topk_raw': float(corr_topk_raw),
        'corr_topk_eff': float(corr_topk_eff),
        
        # Gate statistics
        'gate_mean': float(gate_mean),
        'gate_peaks': float(gate_peaks.item() if torch.is_tensor(gate_peaks) else gate_peaks),
        'gate_bg': float(gate_bg.item() if torch.is_tensor(gate_bg) else gate_bg),
        'gate_separation': float(gate_separation.item() if torch.is_tensor(gate_separation) else gate_separation),
        
        # Gate scale (新增)
        'gate_scale': gate_scale,
        
        # Peak/Background improvement
        'peak_improvement': float(peak_improvement.item() if torch.is_tensor(peak_improvement) else peak_improvement),
        'bg_improvement': float(bg_improvement.item() if torch.is_tensor(bg_improvement) else bg_improvement),
        
        'alpha': alpha,
        'use_gate_in_forward': use_gate_in_forward,
    }
    
    # Add top-k metrics for refined
    for key, value in topk_metrics_refined.items():
        metrics[f'topk_{key}'] = value
    
    # Add top-k metrics for HiCARN baseline
    for key, value in topk_metrics_hicarn.items():
        metrics[f'topk_{key}_hicarn'] = value
    
    # Add convenience keys for best model selection
    # Default: use IoU@0.5% mean as the main top-k metric
    if 'iou_0.5_mean' in topk_metrics_refined:
        metrics['score_topk_iou'] = topk_metrics_refined['iou_0.5_mean']
    elif 'iou_1.0_mean' in topk_metrics_refined:
        metrics['score_topk_iou'] = topk_metrics_refined['iou_1.0_mean']
    else:
        metrics['score_topk_iou'] = 0.0
    
    return metrics


# ================================================================
# Main
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument('--train_hicarn', type=str, required=True)
    parser.add_argument('--train_gt', type=str, required=True)
    parser.add_argument('--val_hicarn', type=str, default=None)
    parser.add_argument('--val_gt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints_v14')
    
    # Training basics
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--cond_norm', type=str, default='learnable')
    parser.add_argument('--use_gate', action='store_true', default=True)
    parser.add_argument('--g_scale', type=float, default=0.5)
    
    # Alpha (修复A: 更保守的 alpha)
    parser.add_argument('--alpha_warmup', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.12)  # 从0.15/0.20降到0.12
    
    # Loss weights (结构导向 - 保持全局质量，提升结构精度)
    # 核心策略: 降低全局重建权重，大幅提升结构 loss
    parser.add_argument('--lambda_res', type=float, default=10.0)
    parser.add_argument('--lambda_dir', type=float, default=2.0)
    parser.add_argument('--lambda_recon', type=float, default=0.3)       # 降低: 不过度约束全局
    parser.add_argument('--lambda_insulation', type=float, default=1.0)  # 大幅提高: TAD 边界
    parser.add_argument('--lambda_peak', type=float, default=1.5)        # 大幅提高: Loop 峰值
    parser.add_argument('--lambda_hf', type=float, default=0.5)          # 提高: 边缘清晰
    parser.add_argument('--lambda_anchor', type=float, default=0.2)      # 降低: 不锁死全局统计
    parser.add_argument('--lambda_gate', type=float, default=0.2)        # 略降
    parser.add_argument('--lambda_gate_sep', type=float, default=0.8)    # 大幅提高: gate 必须区分 peak/bg
    
    # Gate parameters (让 gate 更精准地定位结构区域)
    parser.add_argument('--gate_bg_target', type=float, default=0.08)    # 降低: 背景更少改动
    parser.add_argument('--gate_topk_ratio', type=float, default=0.015)  # 更严格的 peak 定义
    parser.add_argument('--gate_sep_margin', type=float, default=0.08)   # 提高: 更大分离
    
    # Structure loss parameters
    parser.add_argument('--insul_window', type=int, default=5)
    parser.add_argument('--lambda_tad_boundary', type=float, default=0.5,
                       help='Weight for TAD boundary sharpness loss')
    parser.add_argument('--lambda_loop_pos', type=float, default=0.3,
                       help='Weight for loop position accuracy loss')
    parser.add_argument('--lambda_offdiag', type=float, default=0.3,
                       help='Weight for off-diagonal structure loss')
    parser.add_argument('--loop_topk_ratio', type=float, default=0.01,
                       help='Top-k ratio for loop position loss')
    parser.add_argument('--min_diag', type=int, default=2,
                       help='Minimum distance from diagonal for off-diag loss')
    
    # 修复C: 残差权重更保守 (让模型先稳定"不过度牺牲背景")
    parser.add_argument('--res_loss_weighted', type=bool, default=True)
    parser.add_argument('--res_weight_topk_ratio', type=float, default=0.05)  # 保守
    parser.add_argument('--res_weight_peak', type=float, default=6.0)         # 保守 (从8→6)
    parser.add_argument('--res_weight_bg', type=float, default=1.0)           # 保守 (从0.5→1.0)
    
    # Time sampling (混合)
    parser.add_argument('--t_sampling', type=str, default='mixed',
                       choices=['uniform', 'beta', 'mixed'])
    parser.add_argument('--t_p0', type=float, default=0.3)
    parser.add_argument('--t_beta_a', type=float, default=2.0)
    parser.add_argument('--t_beta_b', type=float, default=5.0)
    
    # Validation
    parser.add_argument('--val_steps', type=int, default=20)
    parser.add_argument('--res_corr_topk_ratio', type=float, default=0.05)
    
    # Top-K Overlap parameters
    parser.add_argument('--topk_perc_list', type=float, nargs='+', default=[1.0, 0.5, 0.1])
    parser.add_argument('--topk_min_diag', type=int, default=2)
    parser.add_argument('--topk_upper', type=bool, default=True)
    
    # Best model selection
    parser.add_argument('--best_metric', type=str, default='both',
                       choices=['pcc', 'iou_0.5', 'iou_0.1', 'iou_1.0', 'both'],
                       help='Metric for selecting best model')
    
    # Schedule parameters (修复A&B: 更慢的 ramp)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--rampup_epochs', type=int, default=30)       # 从10→30
    parser.add_argument('--gate_warmup_epochs', type=int, default=15)  # 新增: gate 也做 warmup
    
    # Resume
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Print configuration
    # ================================================================
    print("\n" + "="*80)
    print("DIFFUSION REFINEMENT v14 (Top-K Overlap + Structure-Oriented)")
    print("="*80)
    print(f"\nv14 关键改进:")
    print(f"  A. Top-K Overlap 指标: IoU@{args.topk_perc_list}")
    print(f"  B. 双 Best Model: best_model_pcc.pt + best_model_iou.pt")
    print(f"  C. 结构导向默认值: lambda_peak={args.lambda_peak}, lambda_insulation={args.lambda_insulation}")
    print(f"  D. 混合时间采样: {args.t_sampling}")
    
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
    weight_scheduler = LossWeightSchedulerV14(
        warmup_epochs=args.warmup_epochs,
        rampup_epochs=args.rampup_epochs,
        gate_warmup_epochs=args.gate_warmup_epochs,  # 修复B
        alpha_warmup=args.alpha_warmup,
        alpha_full=args.alpha,
        target_lambda_insulation=args.lambda_insulation,
        target_lambda_tad_boundary=args.lambda_tad_boundary,
        target_lambda_peak=args.lambda_peak,
        target_lambda_loop_pos=args.lambda_loop_pos,
        target_lambda_offdiag=args.lambda_offdiag,
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
    
    # ================================================================
    # Resume (if specified)
    # ================================================================
    start_epoch = 1
    best_pcc = None
    best_iou = 0.0
    best_raw_res_corr = 0.0
    history = []
    
    if args.resume:
        print("\n" + "="*80)
        print("RESUMING FROM CHECKPOINT")
        print("="*80)
        
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_pcc = checkpoint.get('best_pcc')
        best_iou = checkpoint.get('best_iou', 0.0)
        best_raw_res_corr = checkpoint.get('best_raw_res_corr', 0.0)
        
        if 'history' in checkpoint:
            history = checkpoint['history']
        
        print(f"  Resuming from epoch {start_epoch}")
    
    # Baseline
    mse_baseline = np.mean((hicarn_val - gt_val) ** 2)
    pcc_baseline, _ = stats.pearsonr(hicarn_val.flatten(), gt_val.flatten())
    print(f"\nBaseline: MSE={mse_baseline:.6f}, PCC={pcc_baseline:.4f}")
    
    if best_pcc is None:
        best_pcc = pcc_baseline
    
    # ================================================================
    # Training
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    for epoch in range(start_epoch, args.epochs + 1):
        current_weights = weight_scheduler.get_weights(epoch)
        current_alpha = weight_scheduler.get_alpha(epoch)
        current_gate_scale = weight_scheduler.get_gate_scale(epoch)  # 修复B: gate warmup
        use_gate = weight_scheduler.use_gate_in_forward(epoch)
        stage_num, stage_name = weight_scheduler.get_stage(epoch)
        
        if epoch == 1 or epoch == args.warmup_epochs + 1 or epoch == args.warmup_epochs + args.rampup_epochs + 1:
            print(f"\n  >>> Stage {stage_num}: {stage_name}")
            print(f"      alpha={current_alpha:.3f}, gate_scale={current_gate_scale:.2f}, use_gate={use_gate}")
        
        losses = train_epoch_v14(
            model, scheduler, optimizer, train_loader,
            device, res_mean, res_std, cond_mean, cond_std, epoch,
            alpha=current_alpha,
            gate_scale=current_gate_scale,  # 修复B: 传入 gate_scale
            lambda_res=args.lambda_res,
            lambda_dir=args.lambda_dir,
            lambda_recon=args.lambda_recon,
            lambda_insulation=current_weights['lambda_insulation'],
            lambda_tad_boundary=current_weights.get('lambda_tad_boundary', args.lambda_tad_boundary),
            lambda_peak=current_weights['lambda_peak'],
            lambda_loop_pos=current_weights.get('lambda_loop_pos', args.lambda_loop_pos),
            lambda_offdiag=current_weights.get('lambda_offdiag', args.lambda_offdiag),
            lambda_hf=current_weights['lambda_hf'],
            lambda_anchor=current_weights['lambda_anchor'],
            lambda_gate=current_weights['lambda_gate'],
            lambda_gate_sep=current_weights['lambda_gate_sep'],
            gate_bg_target=args.gate_bg_target,
            gate_topk_ratio=args.gate_topk_ratio,
            gate_sep_margin=args.gate_sep_margin,
            insul_window=args.insul_window,
            use_gate_in_forward=use_gate,
            res_loss_weighted=args.res_loss_weighted,
            res_weight_topk_ratio=args.res_weight_topk_ratio,
            res_weight_peak=args.res_weight_peak,
            res_weight_bg=args.res_weight_bg,
            t_sampling=args.t_sampling,
            t_p0=args.t_p0,
            t_beta_a=args.t_beta_a,
            t_beta_b=args.t_beta_b,
            loop_topk_ratio=args.loop_topk_ratio,
            min_diag=args.min_diag
        )
        
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate_v14(
                model, scheduler, hicarn_val, gt_val,
                res_mean, res_std, cond_mean, cond_std, current_alpha, device,
                gate_scale=current_gate_scale,  # 新增: 传入 gate_scale
                gate_topk_ratio=args.gate_topk_ratio,
                res_corr_topk_ratio=args.res_corr_topk_ratio,
                use_gate_in_forward=use_gate,
                num_steps=args.val_steps,
                topk_perc_list=args.topk_perc_list,
                topk_min_diag=args.topk_min_diag,
                topk_upper=args.topk_upper
            )
            
            improved = ""
            
            # Check raw_res_corr
            if val_metrics['raw_res_corr'] > best_raw_res_corr:
                best_raw_res_corr = val_metrics['raw_res_corr']
                improved += " [best raw_corr]"
            
            # ============================================
            # Save best_model_pcc.pt
            # ============================================
            if val_metrics['pcc'] > best_pcc:
                best_pcc = val_metrics['pcc']
                improved += " [best PCC]"
                
                if args.best_metric in ['pcc', 'both']:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': val_metrics,
                        'res_mean': res_mean, 'res_std': res_std,
                        'cond_mean': cond_mean, 'cond_std': cond_std,
                        'alpha': current_alpha,
                        'best_pcc': best_pcc,
                        'best_iou': best_iou,
                        'best_raw_res_corr': best_raw_res_corr,
                        'history': history,
                        'config': vars(args)
                    }, output_dir / 'best_model_pcc.pt')
            
            # ============================================
            # Save best_model_iou.pt
            # ============================================
            current_iou = val_metrics.get('score_topk_iou', 0.0)
            if current_iou > best_iou:
                best_iou = current_iou
                improved += f" [best IoU={best_iou:.4f}]"
                
                if args.best_metric in ['iou_0.5', 'iou_0.1', 'iou_1.0', 'both']:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': val_metrics,
                        'res_mean': res_mean, 'res_std': res_std,
                        'cond_mean': cond_mean, 'cond_std': cond_std,
                        'alpha': current_alpha,
                        'best_pcc': best_pcc,
                        'best_iou': best_iou,
                        'best_raw_res_corr': best_raw_res_corr,
                        'history': history,
                        'config': vars(args)
                    }, output_dir / 'best_model_iou.pt')
            
            # Also save as best_model.pt (for backward compatibility)
            # Use the metric specified by --best_metric
            save_best = False
            if args.best_metric == 'pcc' and val_metrics['pcc'] >= best_pcc:
                save_best = True
            elif args.best_metric in ['iou_0.5', 'iou_0.1', 'iou_1.0'] and current_iou >= best_iou:
                save_best = True
            elif args.best_metric == 'both' and (val_metrics['pcc'] >= best_pcc or current_iou >= best_iou):
                # Save the one that improved
                save_best = True
            
            if save_best:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'res_mean': res_mean, 'res_std': res_std,
                    'cond_mean': cond_mean, 'cond_std': cond_std,
                    'alpha': current_alpha,
                    'best_pcc': best_pcc,
                    'best_iou': best_iou,
                    'best_raw_res_corr': best_raw_res_corr,
                    'history': history,
                    'config': vars(args)
                }, output_dir / 'best_model.pt')
            
            # Print validation results
            status_pcc = "✓" if val_metrics['improved_over_hicarn'] else "⚠"
            status_raw = "✓" if val_metrics['raw_res_corr'] > 0.10 else "⚠"
            status_topk = "✓" if val_metrics['corr_topk_raw'] > 0.15 else "⚠"
            status_delta = "✓" if val_metrics['delta_mean'] > 0.01 else "⚠"  # 检查模型是否真的在改动
            
            print(f"\n  Epoch {epoch} (Stage {stage_num}, α={current_alpha:.3f}, gate_scale={current_gate_scale:.2f}, gate={'ON' if use_gate else 'BYPASS'}):")
            print(f"    MSE: {val_metrics['mse']:.6f} (HiCARN: {val_metrics['mse_hicarn']:.6f}) {status_pcc}")
            print(f"    PCC: {val_metrics['pcc']:.4f} (HiCARN: {val_metrics['pcc_hicarn']:.4f})")
            
            # Delta metrics (新增: 诊断模型实际改动量)
            print(f"    --- Delta (pred - hicarn) ---")
            print(f"    delta_mean: {val_metrics['delta_mean']:.4f}, delta_max: {val_metrics['delta_max']:.4f}, delta_std: {val_metrics['delta_std']:.4f} {status_delta}")
            
            # Residual correlation
            print(f"    --- Residual Correlation ---")
            print(f"    raw_res_corr:  {val_metrics['raw_res_corr']:.4f} {status_raw}")
            print(f"    corr_topk_raw: {val_metrics['corr_topk_raw']:.4f} {status_topk}")
            
            # Gate statistics
            print(f"    --- Gate (scale={current_gate_scale:.2f}) ---")
            print(f"    gate: mean={val_metrics['gate_mean']:.3f}, peaks={val_metrics['gate_peaks']:.3f}, "
                  f"bg={val_metrics['gate_bg']:.3f}, sep={val_metrics['gate_separation']:.3f}")
            
            # Peak/Background improvement
            print(f"    peak_improvement: {val_metrics['peak_improvement']*100:+.2f}%")
            print(f"    bg_improvement:   {val_metrics['bg_improvement']*100:+.2f}%")
            
            # Top-K Overlap (核心新增)
            print(f"    --- Top-K Overlap ---")
            for k in args.topk_perc_list:
                iou_key = f'topk_iou_{k}_mean'
                iou_hicarn_key = f'topk_iou_{k}_mean_hicarn'
                if iou_key in val_metrics:
                    iou_val = val_metrics[iou_key]
                    iou_hicarn = val_metrics.get(iou_hicarn_key, 0.0)
                    imp = (iou_val - iou_hicarn) / max(iou_hicarn, 1e-6) * 100
                    print(f"    IoU@{k}%: {iou_val:.4f} (HiCARN: {iou_hicarn:.4f}, {imp:+.1f}%)")
            
            if improved:
                print(f"    {improved}")
            
            history.append({
                'epoch': epoch,
                'stage': stage_num,
                'alpha': current_alpha,
                'use_gate': use_gate,
                'losses': losses,
                'val_metrics': val_metrics
            })
        
        # Periodic checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'res_mean': res_mean, 'res_std': res_std,
                'cond_mean': cond_mean, 'cond_std': cond_std,
                'best_pcc': best_pcc,
                'best_iou': best_iou,
                'best_raw_res_corr': best_raw_res_corr,
                'history': history,
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
    print(f"Best IoU@0.5%: {best_iou:.4f}")
    print(f"Best raw_res_corr: {best_raw_res_corr:.4f}")
    print(f"\nSaved models:")
    print(f"  - best_model_pcc.pt (best PCC)")
    print(f"  - best_model_iou.pt (best IoU@K)")
    print(f"  - best_model.pt (based on --best_metric)")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
