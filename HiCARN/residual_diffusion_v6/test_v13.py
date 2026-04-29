#!/usr/bin/env python3
"""
Test script for Diffusion Refinement v13

用法:
python test_v13.py \
  --checkpoint checkpoints_v13/best_model.pt \
  --output_dir test_results_v13

默认测试 chr18-22，输入输出路径:
- HiCARN: /home/yangz/data/hic_data/HiCARN/hicarn_predictions/${chr}/predictions_norm.npy
- GT: /data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40/hr_test_${chr}.npy
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import math
from scipy import stats
import argparse


# ================================================================
# HiCPreprocessor / RobustHiCPreprocessor (for loading preprocessor.pt)
# ================================================================

def ensure_nchw(arr):
    """Ensure array is (N, C, H, W)."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[:, np.newaxis, :, :]
    elif arr.ndim == 4:
        if arr.shape[1] in [1, 3]:
            return arr
        elif arr.shape[-1] in [1, 3]:
            return np.transpose(arr, (0, 3, 1, 2))
        elif arr.shape[1] == 1 and arr.shape[-1] == 1:
            return arr
        else:
            raise ValueError(f"Cannot infer channel axis for shape {arr.shape}")
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape={arr.shape}")


class RobustHiCPreprocessor:
    """
    Robust Hi-C preprocessing using median and IQR instead of mean/std.
    
    Key features:
    - Log1p transformation for sparse data
    - Median/IQR normalization (robust to outliers)
    - Clipping to [-5, 5] range for training stability
    """
    
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None  # Actually stores median
        self.X_std = None   # Actually stores IQR
        self.Y_mean = None  # Actually stores median
        self.Y_std = None   # Actually stores IQR
        self._is_fitted = False

    def fit(self, X_low, Y_high, verbose=True):
        X_low = ensure_nchw(X_low)
        Y_high = ensure_nchw(Y_high)
        
        X_log = np.log1p(X_low)
        Y_log = np.log1p(Y_high)
        
        self.X_mean = np.median(X_log)
        self.X_std = (np.percentile(X_log, 75) - np.percentile(X_log, 25)) + 1e-8
        self.Y_mean = np.median(Y_log)
        self.Y_std = (np.percentile(Y_log, 75) - np.percentile(Y_log, 25)) + 1e-8
        self._is_fitted = True
        return self

    def preprocess(self, X_low, Y_high=None):
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before preprocessing!")
        
        X_low = ensure_nchw(X_low)
        X_log = np.log1p(X_low)
        Xn = (X_log - self.X_mean) / self.X_std
        Xn = np.clip(Xn, -5, 5).astype(np.float32)
        
        if Y_high is None:
            return Xn, None
        
        Y_high = ensure_nchw(Y_high)
        Y_log = np.log1p(Y_high)
        Yn = (Y_log - self.Y_mean) / self.Y_std
        Yn = np.clip(Yn, -5, 5).astype(np.float32)
        
        return Xn, Yn

    def preprocess_hr(self, Y_high):
        """
        Preprocess HR (ground truth) data only:
        1. Log1p transform
        2. Standardize using Y_mean and Y_std
        3. Clip to [-5, 5]
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before preprocessing!")
        
        Y_high = ensure_nchw(Y_high)
        Y_log = np.log1p(Y_high)
        Yn = (Y_log - self.Y_mean) / self.Y_std
        Yn = np.clip(Yn, -5, 5).astype(np.float32)
        return Yn

    def postprocess(self, Y_norm):
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before postprocessing!")
        
        Y_norm = np.clip(Y_norm, -5, 5)
        Y_log = Y_norm * self.Y_std + self.Y_mean
        Y_counts = np.expm1(Y_log)
        return np.maximum(Y_counts, 0.0)

    def get_stats(self):
        return {
            'X_median': float(self.X_mean),
            'X_iqr': float(self.X_std),
            'Y_median': float(self.Y_mean),
            'Y_iqr': float(self.Y_std),
        }


# Alias for backward compatibility
HiCPreprocessor = RobustHiCPreprocessor


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

def get_peak_mask(matrix, topk_ratio=0.02):
    B, C, H, W = matrix.shape
    flat = matrix.view(B, -1)
    
    k = max(1, int(flat.shape[1] * topk_ratio))
    topk_vals, _ = torch.topk(flat, k, dim=1)
    threshold = topk_vals[:, -1].view(B, 1, 1, 1)
    
    mask = (matrix >= threshold).float()
    return mask


def get_topk_abs_mask(matrix, topk_ratio=0.05):
    B, C, H, W = matrix.shape
    flat = matrix.abs().view(B, -1)
    
    k = max(1, int(flat.shape[1] * topk_ratio))
    topk_vals, _ = torch.topk(flat, k, dim=1)
    threshold = topk_vals[:, -1].view(B, 1, 1, 1)
    
    mask = (matrix.abs() >= threshold).float()
    return mask


# ================================================================
# Inference
# ================================================================

@torch.no_grad()
def run_inference(
    model, scheduler, hicarn, 
    res_mean, res_std, cond_mean, cond_std, 
    alpha, device,
    use_gate=True,
    num_steps=20,
    batch_size=64,
    seed=42
):
    """
    对输入数据进行推理
    
    Args:
        model: 训练好的模型
        scheduler: DDPM scheduler
        hicarn: HiCARN 预测结果 (N, 1, H, W) numpy array
        res_mean, res_std: 残差统计量
        cond_mean, cond_std: condition 统计量
        alpha: 残差缩放因子
        device: 设备
        use_gate: 是否使用 gate
        num_steps: DDIM 采样步数
        batch_size: 推理 batch size
        seed: 随机种子
    
    Returns:
        refined: 精炼后的结果 (N, 1, H, W) numpy array
        gate_values: gate 值 (N, 1, H, W) numpy array (如果 use_gate=True)
        pred_residuals: 预测的残差 (N, 1, H, W) numpy array
    """
    model.eval()
    torch.manual_seed(seed)
    
    N = len(hicarn)
    H, W = hicarn.shape[2], hicarn.shape[3]
    
    cond_mean_t = torch.tensor(cond_mean, device=device)
    cond_std_t = torch.tensor(cond_std, device=device)
    
    timesteps = torch.linspace(
        scheduler.num_train_timesteps - 1, 0, num_steps
    ).long().to(device)
    
    all_refined = []
    all_gates = []
    all_residuals = []
    
    for start_idx in tqdm(range(0, N, batch_size), desc='Inference'):
        end_idx = min(start_idx + batch_size, N)
        batch_hicarn = torch.from_numpy(hicarn[start_idx:end_idx]).float().to(device)
        n = batch_hicarn.shape[0]
        
        # 初始化噪声
        x_t = torch.randn(n, 1, H, W, device=device)
        
        # DDIM 采样
        for i, t in enumerate(timesteps):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            
            pred_v, gate = model(x_t, batch_hicarn, t_batch, cond_mean_t, cond_std_t)
            pred_x0 = scheduler.predict_x0_from_v(x_t, pred_v, t_batch)
            
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t = scheduler.alphas_cumprod[t]
                alpha_t_next = scheduler.alphas_cumprod[t_next]
                
                pred_noise = (x_t - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt()
                x_t = alpha_t_next.sqrt() * pred_x0 + (1 - alpha_t_next).sqrt() * pred_noise
            else:
                x_t = pred_x0
        
        # 还原残差
        pred_residual = pred_x0 * res_std + res_mean
        
        # 应用残差
        if use_gate and gate is not None:
            refined = batch_hicarn + alpha * gate * pred_residual
        else:
            refined = batch_hicarn + alpha * pred_residual
        
        refined = torch.clamp(refined, -5, 5)
        
        all_refined.append(refined.cpu().numpy())
        all_residuals.append(pred_residual.cpu().numpy())
        if gate is not None:
            all_gates.append(gate.cpu().numpy())
    
    refined = np.concatenate(all_refined, axis=0)
    pred_residuals = np.concatenate(all_residuals, axis=0)
    gate_values = np.concatenate(all_gates, axis=0) if all_gates else None
    
    return refined, gate_values, pred_residuals


# ================================================================
# Metrics
# ================================================================

def compute_metrics(refined, hicarn, gt, topk_ratio=0.02, res_corr_topk_ratio=0.05):
    """
    计算评估指标
    
    Returns:
        dict: 包含各种指标的字典
    """
    # 转换为 tensor
    refined_t = torch.from_numpy(refined).float()
    hicarn_t = torch.from_numpy(hicarn).float()
    gt_t = torch.from_numpy(gt).float()
    
    # 基本指标
    mse_refined = F.mse_loss(refined_t, gt_t).item()
    mse_hicarn = F.mse_loss(hicarn_t, gt_t).item()
    
    refined_np = refined.flatten()
    hicarn_np = hicarn.flatten()
    gt_np = gt.flatten()
    
    pcc_refined, _ = stats.pearsonr(refined_np, gt_np)
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    ssim_refined = compute_ssim(refined, gt)
    ssim_hicarn = compute_ssim(hicarn, gt)
    
    # 残差相关性
    ideal_residual = gt - hicarn
    actual_residual = refined - hicarn
    
    res_corr, _ = stats.pearsonr(
        actual_residual.flatten(),
        ideal_residual.flatten()
    )
    
    # 分区残差相关性
    topk_mask = get_topk_abs_mask(
        torch.from_numpy(ideal_residual).float(), 
        res_corr_topk_ratio
    ).numpy().flatten().astype(bool)
    
    if topk_mask.sum() > 10:
        res_corr_topk, _ = stats.pearsonr(
            actual_residual.flatten()[topk_mask],
            ideal_residual.flatten()[topk_mask]
        )
    else:
        res_corr_topk = 0.0
    
    # Peak/Background 分析
    peak_mask = get_peak_mask(gt_t, topk_ratio)
    bg_mask = 1 - peak_mask
    
    peak_mse_refined = ((refined_t - gt_t) ** 2 * peak_mask).sum() / (peak_mask.sum() + 1e-8)
    peak_mse_hicarn = ((hicarn_t - gt_t) ** 2 * peak_mask).sum() / (peak_mask.sum() + 1e-8)
    
    bg_mse_refined = ((refined_t - gt_t) ** 2 * bg_mask).sum() / (bg_mask.sum() + 1e-8)
    bg_mse_hicarn = ((hicarn_t - gt_t) ** 2 * bg_mask).sum() / (bg_mask.sum() + 1e-8)
    
    if peak_mse_hicarn > 1e-8:
        peak_improvement = (peak_mse_hicarn - peak_mse_refined) / peak_mse_hicarn
    else:
        peak_improvement = 0.0
    
    if bg_mse_hicarn > 1e-8:
        bg_improvement = (bg_mse_hicarn - bg_mse_refined) / bg_mse_hicarn
    else:
        bg_improvement = 0.0
    
    return {
        'mse_refined': mse_refined,
        'mse_hicarn': mse_hicarn,
        'mse_improvement': (mse_hicarn - mse_refined) / mse_hicarn * 100,
        
        'pcc_refined': float(pcc_refined),
        'pcc_hicarn': float(pcc_hicarn),
        'pcc_improvement': (pcc_refined - pcc_hicarn) / pcc_hicarn * 100,
        
        'ssim_refined': ssim_refined,
        'ssim_hicarn': ssim_hicarn,
        'ssim_improvement': (ssim_refined - ssim_hicarn) / ssim_hicarn * 100,
        
        'res_corr': float(res_corr),
        'res_corr_topk': float(res_corr_topk),
        
        'peak_mse_refined': peak_mse_refined.item(),
        'peak_mse_hicarn': peak_mse_hicarn.item(),
        'peak_improvement': float(peak_improvement.item() if torch.is_tensor(peak_improvement) else peak_improvement) * 100,
        
        'bg_mse_refined': bg_mse_refined.item(),
        'bg_mse_hicarn': bg_mse_hicarn.item(),
        'bg_improvement': float(bg_improvement.item() if torch.is_tensor(bg_improvement) else bg_improvement) * 100,
    }


def compute_ssim(pred, target, window_size=11):
    """
    计算 SSIM (简化版本)
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    
    # 计算均值和方差
    mu_pred = pred.mean()
    mu_target = target.mean()
    
    sigma_pred = pred.std()
    sigma_target = target.std()
    
    sigma_pred_target = ((pred - mu_pred) * (target - mu_target)).mean()
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred ** 2 + sigma_target ** 2 + C2))
    
    return float(ssim)


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Diffusion Refinement v13')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='test_results_v13',
                       help='Output directory')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                       help='Save refined predictions as .npy files')
    
    # Data paths (with defaults for your setup)
    parser.add_argument('--chromosomes', type=str, nargs='+',
                       default=['chr18', 'chr19', 'chr20', 'chr21', 'chr22'],
                       help='Chromosomes to test')
    parser.add_argument('--hicarn_dir', type=str,
                       default='/home/yangz/data/hic_data/HiCARN/hicarn_predictions',
                       help='Directory containing HiCARN predictions')
    parser.add_argument('--gt_dir', type=str,
                       default='/data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40',
                       help='Directory containing ground truth')
    parser.add_argument('--hicarn_pattern', type=str,
                       default='{chr}/predictions_norm.npy',
                       help='Pattern for HiCARN files (use {chr} as placeholder)')
    parser.add_argument('--gt_pattern', type=str,
                       default='hr_test_{chr}.npy',
                       help='Pattern for GT files (use {chr} as placeholder)')
    
    # GT normalization using preprocessing file
    parser.add_argument('--preprocess_file', type=str, default=None,
                       help='Path to data_preprocessing.pt containing normalization stats')
    parser.add_argument('--gt_already_normalized', action='store_true', default=False,
                       help='GT is already in normalized space (skip normalization)')
    
    # Inference settings
    parser.add_argument('--num_steps', type=int, default=20,
                       help='Number of DDIM sampling steps')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Inference batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Override alpha (optional)
    parser.add_argument('--alpha', type=float, default=None,
                       help='Override alpha from checkpoint')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Load checkpoint
    # ================================================================
    print("\n" + "="*80)
    print("LOADING CHECKPOINT")
    print("="*80)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Get parameters from checkpoint
    res_mean = checkpoint['res_mean']
    res_std = checkpoint['res_std']
    cond_mean = checkpoint['cond_mean']
    cond_std = checkpoint['cond_std']
    alpha = args.alpha if args.alpha is not None else checkpoint.get('alpha', config.get('alpha', 0.15))
    
    print(f"res_mean={res_mean:.6f}, res_std={res_std:.6f}")
    print(f"cond_mean={cond_mean:.6f}, cond_std={cond_std:.6f}")
    print(f"alpha={alpha}")
    
    # ================================================================
    # Load preprocessing stats (for GT normalization)
    # ================================================================
    gt_norm_stats = None
    if args.preprocess_file and not args.gt_already_normalized:
        print("\n" + "="*80)
        print("LOADING PREPROCESSING STATS")
        print("="*80)
        
        if not os.path.exists(args.preprocess_file):
            raise FileNotFoundError(f"Preprocessing file not found: {args.preprocess_file}")
        
        # Load preprocessor - it may be a RobustHiCPreprocessor object or a dict
        preprocess_data = torch.load(args.preprocess_file, map_location='cpu')
        
        # Check if it's a RobustHiCPreprocessor or HiCPreprocessor object directly
        if isinstance(preprocess_data, (RobustHiCPreprocessor, HiCPreprocessor)):
            preprocessor = preprocess_data
            print(f"  Loaded {type(preprocessor).__name__} object directly")
            print(f"  Y_mean (log-median): {preprocessor.Y_mean:.6f}")
            print(f"  Y_std (log-IQR): {preprocessor.Y_std:.6f}")
        
        # Check if preprocessor is stored in a dict (e.g., from checkpoint)
        elif isinstance(preprocess_data, dict):
            if 'preprocessor' in preprocess_data:
                preprocessor = preprocess_data['preprocessor']
                print(f"  Loaded {type(preprocessor).__name__} from dict['preprocessor']")
                print(f"  Y_mean (log-median): {preprocessor.Y_mean:.6f}")
                print(f"  Y_std (log-IQR): {preprocessor.Y_std:.6f}")
            else:
                # Try to extract stats manually
                preprocessor = None
                gt_norm_stats = {}
                
                # Try different key names
                if 'Y_mean' in preprocess_data:
                    gt_norm_stats['Y_mean'] = float(preprocess_data['Y_mean'])
                    gt_norm_stats['Y_std'] = float(preprocess_data['Y_std'])
                
                if gt_norm_stats:
                    print(f"  Loaded normalization stats from dict")
                    print(f"  Y_mean: {gt_norm_stats['Y_mean']:.6f}")
                    print(f"  Y_std: {gt_norm_stats['Y_std']:.6f}")
                    
                    # Create a preprocessor with these stats
                    preprocessor = HiCPreprocessor()
                    preprocessor.Y_mean = gt_norm_stats['Y_mean']
                    preprocessor.Y_std = gt_norm_stats['Y_std']
                else:
                    print(f"  ⚠️ Could not find preprocessor in file")
                    print(f"  Available keys: {list(preprocess_data.keys())}")
                    preprocessor = None
        else:
            print(f"  ⚠️ Unknown preprocessing file format: {type(preprocess_data)}")
            preprocessor = None
    else:
        preprocessor = None
    
    # ================================================================
    # Create model
    # ================================================================
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = GatedConditionedUNet(
        in_channels=2,
        out_channels=1,
        base_channels=config.get('base_channels', 64),
        channel_mults=(1, 2, 4),
        parameterization='v',
        cond_norm_type=config.get('cond_norm', 'learnable'),
        output_gate=config.get('use_gate', True),
        g_scale=config.get('g_scale', 0.5)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=config.get('num_timesteps', 1000),
        parameterization='v'
    ).to(device)
    
    # ================================================================
    # Test each chromosome
    # ================================================================
    print("\n" + "="*80)
    print("TESTING")
    print("="*80)
    
    all_results = {}
    
    for chrom in args.chromosomes:
        print(f"\n--- {chrom} ---")
        
        # Build file paths
        hicarn_path = os.path.join(args.hicarn_dir, args.hicarn_pattern.format(chr=chrom))
        gt_path = os.path.join(args.gt_dir, args.gt_pattern.format(chr=chrom))
        
        print(f"  HiCARN: {hicarn_path}")
        print(f"  GT:     {gt_path}")
        
        # Check files exist
        if not os.path.exists(hicarn_path):
            print(f"  ⚠️ HiCARN file not found, skipping")
            continue
        if not os.path.exists(gt_path):
            print(f"  ⚠️ GT file not found, skipping")
            continue
        
        # Load data
        hicarn = np.load(hicarn_path)
        hicarn = ensure_nchw(hicarn).astype(np.float32)
        
        gt_raw = np.load(gt_path)
        gt_raw = ensure_nchw(gt_raw).astype(np.float32)
        
        print(f"  HiCARN shape: {hicarn.shape}, range [{hicarn.min():.4f}, {hicarn.max():.4f}]")
        print(f"  GT (raw) shape: {gt_raw.shape}, range [{gt_raw.min():.4f}, {gt_raw.max():.4f}]")
        
        # ============================================
        # GT Normalization
        # ============================================
        if args.gt_already_normalized:
            # GT 已经在归一化空间
            gt = gt_raw
            print(f"  GT already normalized, using as-is")
        elif preprocessor is not None:
            # 使用 HiCPreprocessor 归一化 GT
            # 流程: log1p -> (x - Y_mean) / Y_std -> clip(-5, 5)
            gt = preprocessor.preprocess_hr(gt_raw)
            print(f"  GT normalized using HiCPreprocessor:")
            print(f"    log1p -> (x - {preprocessor.Y_mean:.4f}) / {preprocessor.Y_std:.4f} -> clip")
            print(f"  GT after norm: range [{gt.min():.4f}, {gt.max():.4f}]")
        else:
            # 没有归一化信息，发出警告
            gt = gt_raw
            hicarn_range = hicarn.max() - hicarn.min()
            gt_range = gt_raw.max() - gt_raw.min()
            if abs(hicarn_range - gt_range) / max(hicarn_range, gt_range) > 0.5:
                print(f"  ⚠️ WARNING: HiCARN and GT appear to be in different scales!")
                print(f"     HiCARN range: {hicarn_range:.4f}, GT range: {gt_range:.4f}")
                print(f"     Use --preprocess_file to provide normalization stats")
                print(f"     Or use --gt_already_normalized if GT is already normalized")
        
        # Check shape match
        if hicarn.shape != gt.shape:
            print(f"  ⚠️ Shape mismatch, skipping")
            continue
        
        # Run inference
        refined, gate_values, pred_residuals = run_inference(
            model, scheduler, hicarn,
            res_mean, res_std, cond_mean, cond_std,
            alpha, device,
            use_gate=config.get('use_gate', True),
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            seed=args.seed
        )
        
        # Compute metrics
        metrics = compute_metrics(refined, hicarn, gt)
        all_results[chrom] = metrics
        
        # Print results
        print(f"\n  Results for {chrom}:")
        print(f"    MSE:  {metrics['mse_refined']:.6f} (HiCARN: {metrics['mse_hicarn']:.6f}, "
              f"improvement: {metrics['mse_improvement']:+.2f}%)")
        print(f"    PCC:  {metrics['pcc_refined']:.4f} (HiCARN: {metrics['pcc_hicarn']:.4f}, "
              f"improvement: {metrics['pcc_improvement']:+.2f}%)")
        print(f"    SSIM: {metrics['ssim_refined']:.4f} (HiCARN: {metrics['ssim_hicarn']:.4f}, "
              f"improvement: {metrics['ssim_improvement']:+.2f}%)")
        print(f"    res_corr: {metrics['res_corr']:.4f}, res_corr_topk: {metrics['res_corr_topk']:.4f}")
        print(f"    peak_improvement: {metrics['peak_improvement']:+.2f}%")
        print(f"    bg_improvement:   {metrics['bg_improvement']:+.2f}%")
        
        # Save predictions
        if args.save_predictions:
            save_dir = output_dir / chrom
            save_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(save_dir / 'refined.npy', refined)
            np.save(save_dir / 'pred_residuals.npy', pred_residuals)
            if gate_values is not None:
                np.save(save_dir / 'gate_values.npy', gate_values)
            
            print(f"  Saved predictions to {save_dir}")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_results:
        # Aggregate metrics
        avg_metrics = {}
        for key in all_results[list(all_results.keys())[0]].keys():
            values = [all_results[chrom][key] for chrom in all_results]
            avg_metrics[key] = np.mean(values)
        
        print("\nAverage across all chromosomes:")
        print(f"  MSE:  {avg_metrics['mse_refined']:.6f} (HiCARN: {avg_metrics['mse_hicarn']:.6f}, "
              f"improvement: {avg_metrics['mse_improvement']:+.2f}%)")
        print(f"  PCC:  {avg_metrics['pcc_refined']:.4f} (HiCARN: {avg_metrics['pcc_hicarn']:.4f}, "
              f"improvement: {avg_metrics['pcc_improvement']:+.2f}%)")
        print(f"  SSIM: {avg_metrics['ssim_refined']:.4f} (HiCARN: {avg_metrics['ssim_hicarn']:.4f}, "
              f"improvement: {avg_metrics['ssim_improvement']:+.2f}%)")
        print(f"  res_corr: {avg_metrics['res_corr']:.4f}, res_corr_topk: {avg_metrics['res_corr_topk']:.4f}")
        print(f"  peak_improvement: {avg_metrics['peak_improvement']:+.2f}%")
        print(f"  bg_improvement:   {avg_metrics['bg_improvement']:+.2f}%")
        
        # Save results
        results_file = output_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'per_chromosome': all_results,
                'average': avg_metrics,
                'config': {
                    'checkpoint': args.checkpoint,
                    'alpha': alpha,
                    'num_steps': args.num_steps,
                    'chromosomes': args.chromosomes
                }
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
    else:
        print("No results to summarize (all chromosomes skipped)")


if __name__ == '__main__':
    main()
