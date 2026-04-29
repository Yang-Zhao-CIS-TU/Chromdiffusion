#!/usr/bin/env python3
"""
Diffusion Refinement v15 - Correct Tile-Level Structure Proxy Losses

================================================================================
v14 → v15 关键改进：

核心认知改变：
  ❌ 不能在 40×40 tile 上直接做 TAD/Loop loss
  ❌ tile 内的 (i==j) 不等于真实 Hi-C 对角线
  ❌ 很多 tile 根本不包含主对角线
  
  ✅ tile 级 loss 只能是 TAD/Loop 的"代理（proxy）"
  ✅ 必须使用 distance-aware mask
  ✅ 需要 tile 拼接一致性约束

三个正确的结构代理 Loss：
  1. Distance-Aware Gradient Loss (TAD 边界代理)
     - 在距离敏感区域让预测和 GT 梯度一致
     - 学的是"边界感"，不是 TAD 本身
     
  2. Distance-Aware Top-K Loss (Loop 代理)
     - Top-K 只在 distance-aware mask 内计算
     - 学的是"强信号点对齐"
     
  3. Tile Stitch Consistency Loss (关键！)
     - 相邻 tile 拼接时边界一致
     - 唯一直接约束整条染色体重建质量的 loss

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
# Model (与 v14 相同)
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
# Distance-Aware Mask (核心！使用真实的 d_tile)
# ================================================================

def compute_distance_aware_mask(H, W, d_tile, tau=20.0, device='cpu'):
    """
    计算 distance-aware mask。
    
    核心公式:
        d_eff(i, j) = d_tile + (j - i)
        mask = exp(-|d_eff| / tau)
    
    物理含义:
        - d_tile: tile 相对于整条染色体主对角线的位置
        - (j - i): tile 内相对对角线偏移
        - d_eff: 这个像素到整条 Hi-C 主对角线的真实距离
    
    Args:
        H, W: tile 尺寸 (40x40)
        d_tile: tile 的 distance offset (来自你的 distance 数据)
                = 0 表示 tile 覆盖主对角线
                < 0 表示 tile 在主对角线下方
                > 0 表示 tile 在主对角线上方
        tau: 距离衰减参数 (bins)，推荐 20.0
        device: torch device
    
    Returns:
        mask: (H, W) tensor, 值域 (0, 1]
    """
    i_idx = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1)
    j_idx = torch.arange(W, device=device, dtype=torch.float32).view(1, -1)
    
    # 核心公式: d_eff = d_tile + (j - i)
    d_eff = d_tile + (j_idx - i_idx)
    
    # Distance-aware mask: 靠近真实对角线的地方权重高
    mask = torch.exp(-torch.abs(d_eff) / tau)
    
    return mask


def compute_batch_distance_masks(batch_size, H, W, d_tiles, tau=20.0, device='cpu'):
    """
    为一个 batch 的 tiles 计算 distance-aware masks。
    
    Args:
        batch_size: batch 大小
        H, W: tile 尺寸
        d_tiles: (batch_size,) tensor/array 或 scalar
                 每个 tile 的真实 distance offset
        tau: 距离衰减参数
        device: torch device
    
    Returns:
        masks: (batch_size, 1, H, W) tensor
    """
    if isinstance(d_tiles, (int, float)):
        # 如果所有 tile 有相同的 d_tile (fallback)
        mask = compute_distance_aware_mask(H, W, d_tiles, tau, device)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, H, W)
    
    # Per-sample d_tile
    if isinstance(d_tiles, np.ndarray):
        d_tiles = torch.from_numpy(d_tiles).to(device)
    elif isinstance(d_tiles, list):
        d_tiles = torch.tensor(d_tiles, device=device)
    
    masks = []
    for d in d_tiles:
        mask = compute_distance_aware_mask(H, W, float(d), tau, device)
        masks.append(mask)
    
    return torch.stack(masks, dim=0).unsqueeze(1)  # (B, 1, H, W)


# ================================================================
# Structure Proxy Losses (正确的 tile 级结构 loss)
# ================================================================

def distance_aware_gradient_loss(pred, gt, d_mask, eps=1e-8):
    """
    Proxy 1: Distance-Aware Gradient Loss (TAD 边界代理)
    
    在距离敏感区域让预测和 GT 的梯度一致。
    学的是"边界感"，不是 TAD 本身。
    
    Args:
        pred: (B, 1, H, W) predicted matrix
        gt: (B, 1, H, W) ground truth matrix
        d_mask: (B, 1, H, W) distance-aware mask
        eps: numerical stability
    
    Returns:
        loss: scalar
    """
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    
    # Compute gradients
    pred_gx = F.conv2d(pred, sobel_x, padding=1)
    pred_gy = F.conv2d(pred, sobel_y, padding=1)
    pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + eps)
    
    gt_gx = F.conv2d(gt, sobel_x, padding=1)
    gt_gy = F.conv2d(gt, sobel_y, padding=1)
    gt_grad = torch.sqrt(gt_gx**2 + gt_gy**2 + eps)
    
    # Weighted by distance mask
    loss = (d_mask * (pred_grad - gt_grad) ** 2).sum() / (d_mask.sum() + eps)
    
    return loss


def distance_aware_topk_loss(pred, gt, d_mask, topk_ratio=0.02, eps=1e-8):
    """
    Proxy 2: Distance-Aware Top-K Loss (Loop 代理)
    
    Top-K 只在 distance-aware mask 内计算。
    学的是"强信号点对齐"。
    
    Args:
        pred: (B, 1, H, W) predicted matrix
        gt: (B, 1, H, W) ground truth matrix
        d_mask: (B, 1, H, W) distance-aware mask
        topk_ratio: top-k 比例
        eps: numerical stability
    
    Returns:
        loss: scalar
    """
    B = pred.shape[0]
    
    # Flatten and apply mask
    pred_flat = pred.view(B, -1)
    gt_flat = gt.view(B, -1)
    mask_flat = d_mask.view(B, -1)
    
    # Only consider masked positions
    # Use mask as weight for importance
    weighted_gt = gt_flat * mask_flat
    
    # Find top-k in weighted GT
    k = max(1, int(pred_flat.shape[1] * topk_ratio))
    _, topk_idx = torch.topk(weighted_gt, k, dim=1)
    
    # Gather values at top-k positions
    pred_topk = torch.gather(pred_flat, 1, topk_idx)
    gt_topk = torch.gather(gt_flat, 1, topk_idx)
    
    # MSE on top-k positions
    loss = F.mse_loss(pred_topk, gt_topk)
    
    return loss


def distance_aware_peak_preservation_loss(pred, gt, d_mask, topk_ratio=0.02, eps=1e-8):
    """
    另一种 Peak Loss: 在 distance-aware mask 内保持峰值。
    
    Args:
        pred: (B, 1, H, W) predicted matrix
        gt: (B, 1, H, W) ground truth matrix
        d_mask: (B, 1, H, W) distance-aware mask
        topk_ratio: top-k 比例
    
    Returns:
        loss: scalar
    """
    B, C, H, W = gt.shape
    
    # Get peak mask from GT (top-k values)
    gt_flat = gt.view(B, -1)
    k = max(1, int(gt_flat.shape[1] * topk_ratio))
    topk_vals, _ = torch.topk(gt_flat, k, dim=1)
    threshold = topk_vals[:, -1].view(B, 1, 1, 1)
    
    peak_mask = (gt >= threshold).float()
    
    # Combine with distance mask
    combined_mask = peak_mask * d_mask
    
    # Loss on combined masked region
    pred_peaks = pred * combined_mask
    gt_peaks = gt * combined_mask
    
    loss = ((pred_peaks - gt_peaks) ** 2).sum() / (combined_mask.sum() + eps)
    
    return loss


# ================================================================
# Tile Stitch Consistency Loss (关键！)
# ================================================================

def tile_stitch_consistency_loss(pred_batch, overlap_size=4):
    """
    Proxy 3: Tile Stitch Consistency Loss (关键！)
    
    相邻 tile 拼接时边界一致：
        L_stitch = |pred_A_overlap - pred_B_overlap|
    
    这是唯一直接约束"整条染色体重建质量"的 loss。
    
    由于我们没有真正的相邻 tile 信息，这里用 tile 内部的一致性作为代理：
    - 上下边界的平滑性
    - 左右边界的平滑性
    
    真正的实现需要在数据加载时提供相邻 tile 对。
    
    Args:
        pred_batch: (B, 1, H, W) predicted matrices
        overlap_size: 边界检查的像素数
    
    Returns:
        loss: scalar
    """
    B, C, H, W = pred_batch.shape
    
    # 边界平滑性损失 (作为 stitch 的代理)
    # 如果边界不平滑，拼接时会出现明显的接缝
    
    # 水平方向：相邻列的差异
    h_diff = pred_batch[:, :, :, 1:] - pred_batch[:, :, :, :-1]
    h_loss = (h_diff ** 2).mean()
    
    # 垂直方向：相邻行的差异
    v_diff = pred_batch[:, :, 1:, :] - pred_batch[:, :, :-1, :]
    v_loss = (v_diff ** 2).mean()
    
    # 边界区域应该更平滑 (对于拼接)
    # 检查边界 overlap_size 个像素
    left_boundary = pred_batch[:, :, :, :overlap_size]
    right_boundary = pred_batch[:, :, :, -overlap_size:]
    top_boundary = pred_batch[:, :, :overlap_size, :]
    bottom_boundary = pred_batch[:, :, -overlap_size:, :]
    
    # 边界的二阶梯度应该小 (平滑)
    left_smooth = ((left_boundary[:, :, :, 1:] - left_boundary[:, :, :, :-1]) ** 2).mean()
    right_smooth = ((right_boundary[:, :, :, 1:] - right_boundary[:, :, :, :-1]) ** 2).mean()
    top_smooth = ((top_boundary[:, :, 1:, :] - top_boundary[:, :, :-1, :]) ** 2).mean()
    bottom_smooth = ((bottom_boundary[:, :, 1:, :] - bottom_boundary[:, :, :-1, :]) ** 2).mean()
    
    boundary_loss = (left_smooth + right_smooth + top_smooth + bottom_smooth) / 4
    
    # 总损失：全局平滑 + 边界平滑
    total_loss = 0.5 * (h_loss + v_loss) + 0.5 * boundary_loss
    
    return total_loss


def symmetric_consistency_loss(pred):
    """
    对称性一致性损失。
    
    Hi-C 矩阵应该是对称的，但 tile 可能只包含上三角。
    这个 loss 鼓励 tile 内部的对称性。
    
    Args:
        pred: (B, 1, H, W) predicted matrices
    
    Returns:
        loss: scalar
    """
    # pred 和它的转置应该相似
    pred_T = pred.transpose(2, 3)
    loss = F.mse_loss(pred, pred_T)
    return loss


# ================================================================
# Traditional Losses (保留，但不依赖对角线)
# ================================================================

def anchor_loss(pred, gt):
    """Loss to match global statistics"""
    mean_loss = (pred.mean() - gt.mean()) ** 2
    std_loss = (pred.std() - gt.std()) ** 2
    return mean_loss + std_loss


def gate_regularization_loss(gate, d_mask, bg_target=0.05, eps=1e-8):
    """
    Gate 正则化：在远离对角线的区域，gate 应该更小。
    
    Args:
        gate: (B, 1, H, W) gate values
        d_mask: (B, 1, H, W) distance-aware mask
        bg_target: 背景区域的目标 gate 值
    """
    # 在 d_mask 小的地方（远离对角线），gate 应该接近 bg_target
    # 在 d_mask 大的地方（靠近对角线），gate 可以更自由
    
    # 背景 mask: 1 - d_mask (远离对角线)
    bg_mask = 1 - d_mask
    
    # 背景区域的 gate 应该接近 bg_target
    bg_loss = ((gate - bg_target) ** 2 * bg_mask).sum() / (bg_mask.sum() + eps)
    
    return bg_loss


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


class HiCDataset(torch.utils.data.Dataset):
    """
    Hi-C Dataset with distance offset information.
    
    每个 sample 返回: (hicarn, gt, d_tile)
    """
    def __init__(self, hicarn, gt, d_offsets=None):
        """
        Args:
            hicarn: (N, 1, H, W) numpy array
            gt: (N, 1, H, W) numpy array
            d_offsets: (N,) numpy array of distance offsets, or None
        """
        self.hicarn = torch.from_numpy(hicarn).float()
        self.gt = torch.from_numpy(gt).float()
        
        if d_offsets is not None:
            self.d_offsets = torch.from_numpy(d_offsets.astype(np.float32))
        else:
            # 默认所有 tile 在对角线上
            self.d_offsets = torch.zeros(len(hicarn), dtype=torch.float32)
    
    def __len__(self):
        return len(self.hicarn)
    
    def __getitem__(self, idx):
        return self.hicarn[idx], self.gt[idx], self.d_offsets[idx]


def sample_timesteps(batch_size, num_timesteps, device, method='mixed', p0=0.3, beta_a=2.0, beta_b=5.0):
    if method == 'uniform':
        return torch.randint(0, num_timesteps, (batch_size,), device=device)
    elif method == 'beta':
        beta_samples = torch.distributions.Beta(beta_a, beta_b).sample((batch_size,))
        return (beta_samples * num_timesteps).long().to(device)
    elif method == 'mixed':
        uniform_mask = torch.rand(batch_size, device=device) < p0
        uniform_t = torch.randint(0, num_timesteps, (batch_size,), device=device)
        beta_samples = torch.distributions.Beta(beta_a, beta_b).sample((batch_size,)).to(device)
        beta_t = (beta_samples * num_timesteps).long()
        return torch.where(uniform_mask, uniform_t, beta_t)
    else:
        return torch.randint(0, num_timesteps, (batch_size,), device=device)


# ================================================================
# Top-K Overlap Metrics (用于验证)
# ================================================================

def compute_topk_overlap_single(pred, gt, k_perc, min_diag=2, upper=True):
    H, W = pred.shape
    i_idx, j_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    diag_dist = np.abs(j_idx - i_idx)
    
    mask = diag_dist >= min_diag
    if upper:
        mask = mask & (j_idx >= i_idx)
    
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    n_valid = len(pred_valid)
    
    if n_valid < 10:
        return {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'hit1': 0.0}
    
    K = max(1, int(n_valid * k_perc / 100.0))
    
    pred_topk_idx = set(np.argsort(pred_valid)[-K:])
    gt_topk_idx = set(np.argsort(gt_valid)[-K:])
    
    intersection = len(pred_topk_idx & gt_topk_idx)
    union = len(pred_topk_idx | gt_topk_idx)
    
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / K if K > 0 else 0.0
    recall = intersection / K if K > 0 else 0.0
    
    gt_top1_idx = np.argmax(gt_valid)
    hit1 = 1.0 if gt_top1_idx in pred_topk_idx else 0.0
    
    return {'iou': iou, 'precision': precision, 'recall': recall, 'hit1': hit1}


def compute_topk_overlap_batch(pred_batch, gt_batch, k_perc_list=[1.0, 0.5, 0.1], min_diag=2, upper=True):
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
            metrics = compute_topk_overlap_single(pred_batch[i], gt_batch[i], k, min_diag, upper)
            results[f'iou_{k}'].append(metrics['iou'])
            results[f'prec_{k}'].append(metrics['precision'])
            results[f'hit1_{k}'].append(metrics['hit1'])
    
    agg_results = {}
    for key, values in results.items():
        agg_results[f'{key}_mean'] = float(np.mean(values))
        agg_results[f'{key}_median'] = float(np.median(values))
    
    return agg_results


# ================================================================
# Loss Weight Scheduler
# ================================================================

class LossWeightSchedulerV15:
    def __init__(
        self,
        warmup_epochs=20,
        rampup_epochs=30,
        gate_warmup_epochs=15,
        alpha_warmup=0.04,
        alpha_full=0.10,
    ):
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.gate_warmup_epochs = gate_warmup_epochs
        self.alpha_warmup = alpha_warmup
        self.alpha_full = alpha_full
    
    def get_alpha(self, epoch):
        if epoch <= self.warmup_epochs:
            return self.alpha_warmup
        elif epoch <= self.warmup_epochs + self.rampup_epochs:
            progress = (epoch - self.warmup_epochs) / self.rampup_epochs
            return self.alpha_warmup + progress * (self.alpha_full - self.alpha_warmup)
        else:
            return self.alpha_full
    
    def get_gate_scale(self, epoch):
        if epoch <= self.warmup_epochs:
            return 0.0
        elif epoch <= self.warmup_epochs + self.gate_warmup_epochs:
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
# Training Function
# ================================================================

def train_epoch_v15(
    model, scheduler, optimizer, dataloader,
    device, res_mean, res_std, cond_mean, cond_std, epoch,
    alpha=0.10,
    gate_scale=1.0,
    tau=20.0,  # 推荐 20.0，不是 50.0
    # Loss weights
    lambda_res=10.0,
    lambda_dir=2.0,
    lambda_recon=0.7,
    lambda_grad=0.5,        # Distance-aware gradient loss (TAD proxy)
    lambda_topk=0.5,        # Distance-aware top-k loss (Loop proxy)
    lambda_stitch=0.3,      # Tile stitch consistency loss
    lambda_symmetric=0.2,   # Symmetric consistency loss
    lambda_anchor=0.3,
    lambda_gate=0.3,
    # Parameters
    topk_ratio=0.02,
    gate_bg_target=0.05,
    use_gate_in_forward=True,
    res_loss_weighted=True,
    res_weight_topk_ratio=0.05,
    res_weight_peak=6.0,
    res_weight_bg=1.0,
    t_sampling='mixed',
    t_p0=0.3,
    t_beta_a=2.0,
    t_beta_b=5.0
):
    model.train()
    
    total_loss = 0
    loss_components = {
        'diff': 0, 'res': 0, 'dir': 0, 'recon': 0,
        'grad': 0, 'topk': 0, 'stitch': 0, 'symmetric': 0,
        'anchor': 0, 'gate': 0
    }
    
    cond_mean_t = torch.tensor(cond_mean, device=device)
    cond_std_t = torch.tensor(cond_std, device=device)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch_data in enumerate(pbar):
        # 解包数据：支持有/无 distance 的情况
        if len(batch_data) == 3:
            hicarn, gt, d_tiles = batch_data
            d_tiles = d_tiles.to(device)
        else:
            hicarn, gt = batch_data
            d_tiles = torch.zeros(hicarn.shape[0], device=device)
        
        hicarn = hicarn.to(device)
        gt = gt.to(device)
        B, C, H, W = hicarn.shape
        
        # 使用真实的 per-sample d_tile 计算 distance-aware mask
        d_mask = compute_batch_distance_masks(B, H, W, d_tiles, tau, device)
        
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
            topk_mask = (torch.abs(residual_scaled) >= torch.quantile(
                torch.abs(residual_scaled).view(B, -1), 1 - res_weight_topk_ratio, dim=1
            ).view(B, 1, 1, 1)).float()
            weight_map = res_weight_bg + (res_weight_peak - res_weight_bg) * topk_mask
            res_loss = (torch.abs(pred_x0 - residual_scaled) * weight_map).mean()
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
        
        if use_gate_in_forward and gate is not None:
            gated_pred = hicarn + alpha * gate * pred_residual
            ungated_pred = hicarn + alpha * pred_residual
            pred_final = gate_scale * gated_pred + (1 - gate_scale) * ungated_pred
        else:
            pred_final = hicarn + alpha * pred_residual
        
        pred_final = torch.clamp(pred_final, -5, 5)
        
        # ============================================
        # 正确的结构代理 Losses (使用真实 d_mask)
        # ============================================
        
        # Reconstruction loss (保护全局质量)
        recon_loss = F.mse_loss(pred_final, gt)
        
        # Proxy 1: Distance-Aware Gradient Loss (TAD 边界代理)
        grad_loss = distance_aware_gradient_loss(pred_final, gt, d_mask)
        
        # Proxy 2: Distance-Aware Top-K Loss (Loop 代理)
        topk_loss = distance_aware_topk_loss(pred_final, gt, d_mask, topk_ratio)
        
        # Proxy 3: Tile Stitch Consistency Loss
        stitch_loss = tile_stitch_consistency_loss(pred_final)
        
        # Symmetric consistency loss
        symmetric_loss = symmetric_consistency_loss(pred_final)
        
        # Anchor loss (全局统计)
        anch_loss = anchor_loss(pred_final, gt)
        
        # Gate regularization (使用 distance-aware mask)
        if gate is not None:
            gate_loss = gate_regularization_loss(gate, d_mask, gate_bg_target)
        else:
            gate_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        loss = (
            diff_loss +
            lambda_res * res_loss +
            lambda_dir * dir_loss +
            lambda_recon * recon_loss +
            lambda_grad * grad_loss +
            lambda_topk * topk_loss +
            lambda_stitch * stitch_loss +
            lambda_symmetric * symmetric_loss +
            lambda_anchor * anch_loss +
            lambda_gate * gate_loss
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
        loss_components['grad'] += grad_loss.item()
        loss_components['topk'] += topk_loss.item()
        loss_components['stitch'] += stitch_loss.item()
        loss_components['symmetric'] += symmetric_loss.item()
        loss_components['anchor'] += anch_loss.item()
        loss_components['gate'] += gate_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'grad': f'{grad_loss.item():.4f}',
            'topk': f'{topk_loss.item():.4f}'
        })
    
    n_batches = len(dataloader)
    metrics = {k: v / n_batches for k, v in loss_components.items()}
    metrics['total'] = total_loss / n_batches
    
    return metrics


# ================================================================
# Validation Function
# ================================================================

@torch.no_grad()
def validate_v15(
    model, scheduler, hicarn_val, gt_val,
    res_mean, res_std, cond_mean, cond_std, alpha, device,
    gate_scale=1.0,
    d_offsets_val=None,  # Per-sample distance offsets
    tau=20.0,            # 推荐 20.0
    topk_ratio=0.02,
    use_gate_in_forward=True,
    num_steps=20, seed=42,
    topk_perc_list=[1.0, 0.5, 0.1],
    topk_min_diag=2,
    topk_upper=True
):
    model.eval()
    torch.manual_seed(seed)
    
    n = min(200, len(hicarn_val))
    hicarn = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    B, C, H, W = hicarn.shape
    
    cond_mean_t = torch.tensor(cond_mean, device=device)
    cond_std_t = torch.tensor(cond_std, device=device)
    
    # Compute distance-aware mask (使用真实 d_offsets 或默认 0)
    if d_offsets_val is not None:
        d_tiles = d_offsets_val[:n]
    else:
        d_tiles = 0  # 默认所有 tile 在对角线上
    d_mask = compute_batch_distance_masks(B, H, W, d_tiles, tau, device)
    
    x_t = torch.randn(n, 1, H, W, device=device)
    
    timesteps = torch.linspace(scheduler.num_train_timesteps - 1, 0, num_steps).long().to(device)
    
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
    
    # Apply to HiCARN
    if use_gate_in_forward and gate_final is not None:
        gated_pred = hicarn + alpha * gate_final * pred_residual
        ungated_pred = hicarn + alpha * pred_residual
        pred_final = gate_scale * gated_pred + (1 - gate_scale) * ungated_pred
        gate_mean = gate_final.mean().item()
    else:
        pred_final = hicarn + alpha * pred_residual
        gate_mean = 1.0
    
    pred_final = torch.clamp(pred_final, -5, 5)
    
    # Delta metrics
    delta = pred_final - hicarn
    delta_mean = delta.abs().mean().item()
    delta_max = delta.abs().max().item()
    delta_std = delta.std().item()
    
    # Global metrics
    mse = F.mse_loss(pred_final, gt).item()
    mse_hicarn = F.mse_loss(hicarn, gt).item()
    
    final_np = pred_final.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = hicarn.cpu().numpy().flatten()
    
    pcc, _ = stats.pearsonr(final_np, gt_np)
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    # Structure proxy metrics
    grad_loss = distance_aware_gradient_loss(pred_final, gt, d_mask).item()
    topk_loss = distance_aware_topk_loss(pred_final, gt, d_mask, topk_ratio).item()
    stitch_loss = tile_stitch_consistency_loss(pred_final).item()
    
    # Top-K Overlap
    pred_final_np = pred_final.cpu().numpy()
    gt_np_4d = gt.cpu().numpy()
    hicarn_np_4d = hicarn.cpu().numpy()
    
    topk_metrics_refined = compute_topk_overlap_batch(pred_final_np, gt_np_4d, topk_perc_list, topk_min_diag, topk_upper)
    topk_metrics_hicarn = compute_topk_overlap_batch(hicarn_np_4d, gt_np_4d, topk_perc_list, topk_min_diag, topk_upper)
    
    # Build metrics dict
    metrics = {
        'mse': mse,
        'pcc': float(pcc),
        'mse_hicarn': mse_hicarn,
        'pcc_hicarn': float(pcc_hicarn),
        'improved_over_hicarn': float(pcc) > float(pcc_hicarn),
        
        'delta_mean': delta_mean,
        'delta_max': delta_max,
        'delta_std': delta_std,
        
        'gate_mean': float(gate_mean),
        'gate_scale': gate_scale,
        
        # Structure proxy losses
        'grad_loss': grad_loss,
        'topk_loss': topk_loss,
        'stitch_loss': stitch_loss,
        
        'alpha': alpha,
        'use_gate_in_forward': use_gate_in_forward,
    }
    
    # Add top-k metrics
    for key, value in topk_metrics_refined.items():
        metrics[f'topk_{key}'] = value
    for key, value in topk_metrics_hicarn.items():
        metrics[f'topk_{key}_hicarn'] = value
    
    if 'iou_0.5_mean' in topk_metrics_refined:
        metrics['score_topk_iou'] = topk_metrics_refined['iou_0.5_mean']
    elif 'iou_1.0_mean' in topk_metrics_refined:
        metrics['score_topk_iou'] = topk_metrics_refined['iou_1.0_mean']
    else:
        metrics['score_topk_iou'] = 0.0
    
    return metrics


# ================================================================
# Data Loading (distance 从 npz 获取，数据从原来的 npy 获取)
# ================================================================

def load_data(hicarn_path, gt_path):
    """
    加载 HiCARN 和 GT 数据（不包含 distance）。
    """
    print(f"Loading HiCARN from: {hicarn_path}")
    hicarn = np.load(hicarn_path)
    hicarn = ensure_nchw(hicarn).astype(np.float32)
    
    print(f"Loading GT from: {gt_path}")
    gt = np.load(gt_path)
    gt = ensure_nchw(gt).astype(np.float32)
    
    print(f"  HiCARN: {hicarn.shape}, range [{hicarn.min():.4f}, {hicarn.max():.4f}]")
    print(f"  GT:     {gt.shape}, range [{gt.min():.4f}, {gt.max():.4f}]")
    
    return hicarn, gt


def load_distance_info(npz_path):
    """
    从 NPZ 文件加载 distance 和 location 信息。
    
    NPZ 格式:
        distance[k] = ['-40', 'chr1']  # offset, chromosome
        location[k] = [row_start, col_start]  # tile 在染色体矩阵中的位置
    
    Returns:
        d_offsets: (N,) numpy array of int32
        locations: (N, 2) numpy array (可选，用于 stitch loss)
    """
    print(f"Loading distance info from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"  NPZ keys: {data.files}")
    
    d_offsets = None
    locations = None
    
    if 'distance' in data.files:
        distances = data['distance']
        d_offsets = parse_distance_offsets(distances)
        print(f"  Distance offsets: shape={d_offsets.shape}, range=[{d_offsets.min()}, {d_offsets.max()}]")
    
    if 'location' in data.files:
        locations = data['location']
        print(f"  Locations: shape={locations.shape}")
    
    return d_offsets, locations


def parse_distance_offsets(distances):
    """
    解析 distance 数组，提取 d_tile offset。
    
    Input format: 
        distances[k] = ['-40', 'chr1']
    
    Output:
        d_offsets[k] = -40 (int)
    """
    N = len(distances)
    d_offsets = np.zeros(N, dtype=np.int32)
    
    for i in range(N):
        try:
            # distance[i][0] 是 offset 字符串，如 '-40', '0', '80'
            d_offsets[i] = int(distances[i, 0])
        except (ValueError, IndexError):
            d_offsets[i] = 0
    
    return d_offsets


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
# Main
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument('--train_hicarn', type=str, required=True,
                       help='HiCARN predictions .npy file')
    parser.add_argument('--train_gt', type=str, required=True,
                       help='Ground truth .npy file')
    parser.add_argument('--distance_npz', type=str, default=None,
                       help='NPZ file containing distance and location info')
    parser.add_argument('--val_hicarn', type=str, default=None)
    parser.add_argument('--val_gt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints_v15')
    
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
    
    # Alpha
    parser.add_argument('--alpha_warmup', type=float, default=0.04)
    parser.add_argument('--alpha', type=float, default=0.10)
    
    # Distance-aware parameters (关键！)
    parser.add_argument('--tau', type=float, default=20.0,
                       help='Distance decay parameter (bins), 推荐 20.0')
    
    # Loss weights (结构代理)
    parser.add_argument('--lambda_res', type=float, default=10.0)
    parser.add_argument('--lambda_dir', type=float, default=2.0)
    parser.add_argument('--lambda_recon', type=float, default=0.7)
    parser.add_argument('--lambda_grad', type=float, default=0.5,
                       help='Distance-aware gradient loss (TAD proxy)')
    parser.add_argument('--lambda_topk', type=float, default=0.5,
                       help='Distance-aware top-k loss (Loop proxy)')
    parser.add_argument('--lambda_stitch', type=float, default=0.3,
                       help='Tile stitch consistency loss')
    parser.add_argument('--lambda_symmetric', type=float, default=0.2,
                       help='Symmetric consistency loss')
    parser.add_argument('--lambda_anchor', type=float, default=0.3)
    parser.add_argument('--lambda_gate', type=float, default=0.3)
    
    # Other parameters
    parser.add_argument('--topk_ratio', type=float, default=0.02)
    parser.add_argument('--gate_bg_target', type=float, default=0.05)
    parser.add_argument('--res_loss_weighted', type=bool, default=True)
    parser.add_argument('--res_weight_topk_ratio', type=float, default=0.05)
    parser.add_argument('--res_weight_peak', type=float, default=6.0)
    parser.add_argument('--res_weight_bg', type=float, default=1.0)
    
    # Time sampling
    parser.add_argument('--t_sampling', type=str, default='mixed')
    parser.add_argument('--t_p0', type=float, default=0.3)
    parser.add_argument('--t_beta_a', type=float, default=2.0)
    parser.add_argument('--t_beta_b', type=float, default=5.0)
    
    # Validation
    parser.add_argument('--val_steps', type=int, default=20)
    parser.add_argument('--topk_perc_list', type=float, nargs='+', default=[1.0, 0.5, 0.1])
    parser.add_argument('--topk_min_diag', type=int, default=2)
    parser.add_argument('--topk_upper', type=bool, default=True)
    
    # Schedule parameters
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--rampup_epochs', type=int, default=30)
    parser.add_argument('--gate_warmup_epochs', type=int, default=15)
    
    # Best model selection
    parser.add_argument('--best_metric', type=str, default='both',
                       choices=['pcc', 'iou_0.5', 'iou_0.1', 'iou_1.0', 'both'])
    
    # Early stopping
    parser.add_argument('--early_stop', action='store_true', default=False)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--early_stop_metric', type=str, default='iou',
                       choices=['pcc', 'iou', 'mse'])
    
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
    print("DIFFUSION REFINEMENT v15 (Correct Structure Proxy Losses)")
    print("="*80)
    print(f"\nv15 关键改进:")
    print(f"  A. Distance-Aware Gradient Loss (TAD 边界代理)")
    print(f"  B. Distance-Aware Top-K Loss (Loop 代理)")
    print(f"  C. Tile Stitch Consistency Loss (拼接一致性)")
    print(f"  D. 不再使用物理意义错误的 diagonal/TAD/insulation loss")
    
    # ================================================================
    # Load data
    # ================================================================
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # 加载 HiCARN 和 GT 数据
    hicarn_train, gt_train = load_data(args.train_hicarn, args.train_gt)
    
    # 加载 distance 信息（可选）
    d_offsets_train = None
    locations_train = None
    if args.distance_npz:
        d_offsets_train, locations_train = load_distance_info(args.distance_npz)
        
        # 验证数量匹配
        if d_offsets_train is not None and len(d_offsets_train) != len(hicarn_train):
            print(f"  WARNING: distance count ({len(d_offsets_train)}) != data count ({len(hicarn_train)})")
            print(f"  Will use d_tile=0 for all samples")
            d_offsets_train = None
    
    # 验证/分割数据
    if args.val_hicarn and args.val_gt:
        hicarn_val, gt_val = load_data(args.val_hicarn, args.val_gt)
        d_offsets_val = None  # 验证集暂不使用 distance
    else:
        split = int(len(hicarn_train) * 0.9)
        hicarn_val = hicarn_train[split:]
        gt_val = gt_train[split:]
        if d_offsets_train is not None:
            d_offsets_val = d_offsets_train[split:]
            d_offsets_train = d_offsets_train[:split]
        else:
            d_offsets_val = None
        hicarn_train = hicarn_train[:split]
        gt_train = gt_train[:split]
    
    res_mean, res_std = compute_residual_stats(hicarn_train, gt_train)
    cond_mean, cond_std = compute_condition_stats(hicarn_train)
    
    # 使用自定义 Dataset (包含 distance 信息)
    train_dataset = HiCDataset(hicarn_train, gt_train, d_offsets_train)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(hicarn_val)}")
    if d_offsets_train is not None:
        print(f"  ✓ Using per-sample distance offsets")
        print(f"    Distance range: [{d_offsets_train.min()}, {d_offsets_train.max()}]")
    else:
        print(f"  ⚠ No distance info, using d_tile=0 for all samples")
    
    # Create weight scheduler
    weight_scheduler = LossWeightSchedulerV15(
        warmup_epochs=args.warmup_epochs,
        rampup_epochs=args.rampup_epochs,
        gate_warmup_epochs=args.gate_warmup_epochs,
        alpha_warmup=args.alpha_warmup,
        alpha_full=args.alpha,
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
    # Resume / Initialize
    # ================================================================
    start_epoch = 1
    best_pcc = None
    best_iou = 0.0
    best_raw_res_corr = 0.0
    history = []
    early_stop_counter = 0
    early_stop_best = None
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_pcc = checkpoint.get('best_pcc')
        best_iou = checkpoint.get('best_iou', 0.0)
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
        current_alpha = weight_scheduler.get_alpha(epoch)
        current_gate_scale = weight_scheduler.get_gate_scale(epoch)
        use_gate = weight_scheduler.use_gate_in_forward(epoch)
        stage_num, stage_name = weight_scheduler.get_stage(epoch)
        
        if epoch == 1 or epoch == args.warmup_epochs + 1 or epoch == args.warmup_epochs + args.rampup_epochs + 1:
            print(f"\n  >>> Stage {stage_num}: {stage_name}")
            print(f"      alpha={current_alpha:.3f}, gate_scale={current_gate_scale:.2f}, use_gate={use_gate}")
        
        losses = train_epoch_v15(
            model, scheduler, optimizer, train_loader,
            device, res_mean, res_std, cond_mean, cond_std, epoch,
            alpha=current_alpha,
            gate_scale=current_gate_scale,
            tau=args.tau,
            lambda_res=args.lambda_res,
            lambda_dir=args.lambda_dir,
            lambda_recon=args.lambda_recon,
            lambda_grad=args.lambda_grad,
            lambda_topk=args.lambda_topk,
            lambda_stitch=args.lambda_stitch,
            lambda_symmetric=args.lambda_symmetric,
            lambda_anchor=args.lambda_anchor,
            lambda_gate=args.lambda_gate,
            topk_ratio=args.topk_ratio,
            gate_bg_target=args.gate_bg_target,
            use_gate_in_forward=use_gate,
            res_loss_weighted=args.res_loss_weighted,
            res_weight_topk_ratio=args.res_weight_topk_ratio,
            res_weight_peak=args.res_weight_peak,
            res_weight_bg=args.res_weight_bg,
            t_sampling=args.t_sampling,
            t_p0=args.t_p0,
            t_beta_a=args.t_beta_a,
            t_beta_b=args.t_beta_b
        )
        
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate_v15(
                model, scheduler, hicarn_val, gt_val,
                res_mean, res_std, cond_mean, cond_std, current_alpha, device,
                gate_scale=current_gate_scale,
                d_offsets_val=d_offsets_val,  # 使用真实的 distance offsets
                tau=args.tau,
                topk_ratio=args.topk_ratio,
                use_gate_in_forward=use_gate,
                num_steps=args.val_steps,
                topk_perc_list=args.topk_perc_list,
                topk_min_diag=args.topk_min_diag,
                topk_upper=args.topk_upper
            )
            
            improved = ""
            
            # Save best models
            current_iou = val_metrics.get('score_topk_iou', 0.0)
            
            if val_metrics['pcc'] > best_pcc:
                best_pcc = val_metrics['pcc']
                improved += " [best PCC]"
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
                    'config': vars(args)
                }, output_dir / 'best_model_pcc.pt')
            
            if current_iou > best_iou:
                best_iou = current_iou
                improved += f" [best IoU={best_iou:.4f}]"
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
                    'config': vars(args)
                }, output_dir / 'best_model_iou.pt')
            
            # Print results
            status_pcc = "✓" if val_metrics['improved_over_hicarn'] else "⚠"
            status_delta = "✓" if val_metrics['delta_mean'] > 0.01 else "⚠"
            
            print(f"\n  Epoch {epoch} (Stage {stage_num}, α={current_alpha:.3f}, gate_scale={current_gate_scale:.2f}):")
            print(f"    MSE: {val_metrics['mse']:.6f} (HiCARN: {val_metrics['mse_hicarn']:.6f}) {status_pcc}")
            print(f"    PCC: {val_metrics['pcc']:.4f} (HiCARN: {val_metrics['pcc_hicarn']:.4f})")
            print(f"    --- Delta ---")
            print(f"    delta_mean: {val_metrics['delta_mean']:.4f}, delta_max: {val_metrics['delta_max']:.4f} {status_delta}")
            print(f"    --- Structure Proxy Losses ---")
            print(f"    grad_loss: {val_metrics['grad_loss']:.4f} (TAD proxy)")
            print(f"    topk_loss: {val_metrics['topk_loss']:.4f} (Loop proxy)")
            print(f"    stitch_loss: {val_metrics['stitch_loss']:.4f} (拼接一致性)")
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
            
            # Early stopping
            if args.early_stop:
                if args.early_stop_metric == 'iou':
                    current_metric = current_iou
                    improved_early = current_iou > (early_stop_best or 0)
                elif args.early_stop_metric == 'pcc':
                    current_metric = val_metrics['pcc']
                    improved_early = val_metrics['pcc'] > (early_stop_best or 0)
                else:
                    current_metric = -val_metrics['mse']
                    improved_early = current_metric > (early_stop_best or float('-inf'))
                
                if improved_early:
                    early_stop_best = current_metric
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= args.early_stop_patience:
                        print(f"\n  *** EARLY STOPPING at epoch {epoch} ***")
                        break
        
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
                'history': history,
                'config': vars(args)
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=float)
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best PCC: {best_pcc:.4f} (baseline: {pcc_baseline:.4f})")
    print(f"Best IoU@0.5%: {best_iou:.4f}")
    print(f"\nSaved models:")
    print(f"  - best_model_pcc.pt")
    print(f"  - best_model_iou.pt")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
