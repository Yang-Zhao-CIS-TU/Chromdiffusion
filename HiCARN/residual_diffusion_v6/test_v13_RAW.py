#!/usr/bin/env python3
"""
Test script for Diffusion Refinement v13 - Evaluation in RAW Space

This script:
  1. Loads HiCARN predictions (normalized space)
  2. Runs diffusion refinement (normalized space)
  3. Denormalizes BOTH to raw contact counts
  4. Evaluates against RAW ground truth
  5. Computes metrics: PSNR, SNR, SSIM, PCC, SPC, MSE

Usage:
    python test_v13_raw.py \
      --checkpoint checkpoints_v13/best_model.pt \
      --preprocess_file /path/to/preprocessor.pt \
      --chromosomes chr18 chr19 chr20 chr21 chr22 \
      --output_dir test_results_v13_raw
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
from scipy.stats import pearsonr, spearmanr
import argparse


# ================================================================
# RobustHiCPreprocessor (for loading preprocessor.pt)
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
    """
    
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
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
        """Preprocess HR data only"""
        Y_high = ensure_nchw(Y_high)
        Y_log = np.log1p(Y_high)
        Yn = (Y_log - self.Y_mean) / self.Y_std
        Yn = np.clip(Yn, -5, 5).astype(np.float32)
        return Yn

    def postprocess(self, Y_norm):
        """Denormalize to raw contact counts"""
        Y_norm = np.clip(Y_norm, -5, 5)
        Y_log = Y_norm * self.Y_std + self.Y_mean
        Y_counts = np.expm1(Y_log)
        return np.maximum(Y_counts, 0.0)


# Alias
HiCPreprocessor = RobustHiCPreprocessor


# ================================================================
# SSIM Implementation
# ================================================================

class SSIM(nn.Module):
    """SSIM implementation"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, width, sigma):
        gauss = torch.Tensor([math.exp(-(x - width // 2) ** 2 / float(2 * sigma ** 2)) for x in range(width)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel, sigma=3):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.nanmean()
        else:
            return ssim_map.nanmean(1).nanmean(1).nanmean(1)

    def forward(self, img1, img2):
        """Compute SSIM between two 4D tensors"""
        img1 = img1.float()
        img2 = img2.float()
        
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
            
        _, channel, _, _ = img1.size()
        window = self.create_window(self.window_size, channel)
        window = window.type_as(img1)
        
        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


# ================================================================
# Vision Metrics
# ================================================================

class VisionMetrics:
    """Vision metrics calculator for Hi-C data"""
    def __init__(self):
        self.ssim_module = SSIM()
        self.metric_logs = {
            "psnr": [],
            "snr": [],
            "spc": [],
            "pcc": [],
            "ssim": [],
            "mse": [],
        }

    def compute_pcc(self, pred, target):
        """Pearson correlation"""
        pred_np = pred.flatten()
        target_np = target.flatten()
        return pearsonr(pred_np, target_np)[0]

    def compute_spc(self, pred, target):
        """Spearman correlation"""
        pred_np = pred.flatten()
        target_np = target.flatten()
        return spearmanr(pred_np, target_np)[0]

    def compute_mse(self, pred, target):
        """Mean Squared Error"""
        return np.mean((pred - target) ** 2)

    def compute_psnr(self, pred, target):
        """Peak Signal-to-Noise Ratio"""
        mse = self.compute_mse(pred, target)
        if mse < 1e-10:
            return 100.0
        max_val = np.max(target)
        if max_val < 1e-10:
            return 0.0
        return 20 * np.log10(max_val) - 10 * np.log10(mse)

    def compute_snr(self, pred, target):
        """Signal-to-Noise Ratio"""
        signal = np.sum(target)
        noise = np.sqrt(np.sum((target - pred) ** 2))
        if noise < 1e-10:
            return 100.0
        return signal / noise

    def compute_ssim(self, pred, target):
        """Structural Similarity Index"""
        pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        target_t = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
        return self.ssim_module(pred_t, target_t).item()

    def evaluate_sample(self, pred, target):
        """Evaluate a single sample"""
        self.metric_logs['pcc'].append(self.compute_pcc(pred, target))
        self.metric_logs['spc'].append(self.compute_spc(pred, target))
        self.metric_logs['mse'].append(self.compute_mse(pred, target))
        self.metric_logs['psnr'].append(self.compute_psnr(pred, target))
        self.metric_logs['snr'].append(self.compute_snr(pred, target))
        self.metric_logs['ssim'].append(self.compute_ssim(pred, target))

    def evaluate_batch(self, pred_batch, target_batch):
        """Evaluate a batch of samples"""
        for i in range(len(pred_batch)):
            self.evaluate_sample(pred_batch[i], target_batch[i])

    def get_results(self):
        """Get aggregated results"""
        results = {}
        for name, values in self.metric_logs.items():
            if len(values) > 0:
                results[name] = {
                    'mean': float(np.nanmean(values)),
                    'std': float(np.nanstd(values))
                }
        return results

    def reset(self):
        """Reset metric logs"""
        for key in self.metric_logs:
            self.metric_logs[key] = []


# ================================================================
# Model Definition
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
    """Run diffusion inference"""
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
    
    for start_idx in tqdm(range(0, N, batch_size), desc='Inference'):
        end_idx = min(start_idx + batch_size, N)
        batch_hicarn = torch.from_numpy(hicarn[start_idx:end_idx]).float().to(device)
        n = batch_hicarn.shape[0]
        
        x_t = torch.randn(n, 1, H, W, device=device)
        
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
        
        # Denormalize residual
        pred_residual = pred_x0 * res_std + res_mean
        
        # Apply residual
        if use_gate and gate is not None:
            refined = batch_hicarn + alpha * gate * pred_residual
        else:
            refined = batch_hicarn + alpha * pred_residual
        
        refined = torch.clamp(refined, -5, 5)
        all_refined.append(refined.cpu().numpy())
    
    return np.concatenate(all_refined, axis=0)


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Diffusion Refinement v13 in RAW Space')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--preprocess_file', type=str, required=True,
                       help='Path to preprocessor.pt')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='test_results_v13_raw',
                       help='Output directory')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                       help='Save predictions as .npy files')
    
    # Data paths
    parser.add_argument('--chromosomes', type=str, nargs='+',
                       default=['chr18', 'chr19', 'chr20', 'chr21', 'chr22'],
                       help='Chromosomes to test')
    parser.add_argument('--hicarn_dir', type=str,
                       default='/home/yangz/data/hic_data/HiCARN/hicarn_predictions',
                       help='Directory containing HiCARN predictions')
    parser.add_argument('--gt_dir', type=str,
                       default='/data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40',
                       help='Directory containing ground truth (RAW)')
    parser.add_argument('--hicarn_pattern', type=str,
                       default='{chr}/predictions_norm.npy',
                       help='Pattern for HiCARN files')
    parser.add_argument('--gt_pattern', type=str,
                       default='hr_test_{chr}.npy',
                       help='Pattern for GT files (RAW)')
    
    # Inference settings
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=None)
    
    args = parser.parse_args()
    
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
    
    res_mean = checkpoint['res_mean']
    res_std = checkpoint['res_std']
    cond_mean = checkpoint['cond_mean']
    cond_std = checkpoint['cond_std']
    alpha = args.alpha if args.alpha is not None else checkpoint.get('alpha', config.get('alpha', 0.15))
    
    print(f"res_mean={res_mean:.6f}, res_std={res_std:.6f}")
    print(f"alpha={alpha}")
    
    # ================================================================
    # Load preprocessor
    # ================================================================
    print("\n" + "="*80)
    print("LOADING PREPROCESSOR")
    print("="*80)
    
    preprocess_data = torch.load(args.preprocess_file, map_location='cpu')
    
    if isinstance(preprocess_data, (RobustHiCPreprocessor, HiCPreprocessor)):
        preprocessor = preprocess_data
    elif isinstance(preprocess_data, dict) and 'preprocessor' in preprocess_data:
        preprocessor = preprocess_data['preprocessor']
    else:
        raise ValueError(f"Cannot load preprocessor from {args.preprocess_file}")
    
    print(f"  Y_mean (log-median): {preprocessor.Y_mean:.6f}")
    print(f"  Y_std (log-IQR): {preprocessor.Y_std:.6f}")
    
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    scheduler = DDPMScheduler(
        num_train_timesteps=config.get('num_timesteps', 1000),
        parameterization='v'
    ).to(device)
    
    # ================================================================
    # Test each chromosome
    # ================================================================
    print("\n" + "="*80)
    print("TESTING (EVALUATION IN RAW SPACE)")
    print("="*80)
    
    all_results = {}
    
    for chrom in args.chromosomes:
        print(f"\n{'='*60}")
        print(f"  {chrom}")
        print(f"{'='*60}")
        
        hicarn_path = os.path.join(args.hicarn_dir, args.hicarn_pattern.format(chr=chrom))
        gt_path = os.path.join(args.gt_dir, args.gt_pattern.format(chr=chrom))
        
        if not os.path.exists(hicarn_path):
            print(f"  ⚠️ HiCARN file not found: {hicarn_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"  ⚠️ GT file not found: {gt_path}")
            continue
        
        # Load data
        hicarn_norm = np.load(hicarn_path)
        hicarn_norm = ensure_nchw(hicarn_norm).astype(np.float32)
        
        gt_raw = np.load(gt_path)
        gt_raw = ensure_nchw(gt_raw).astype(np.float32)
        
        print(f"  HiCARN (norm): {hicarn_norm.shape}, range [{hicarn_norm.min():.4f}, {hicarn_norm.max():.4f}]")
        print(f"  GT (raw):      {gt_raw.shape}, range [{gt_raw.min():.2f}, {gt_raw.max():.2f}]")
        
        if hicarn_norm.shape != gt_raw.shape:
            print(f"  ⚠️ Shape mismatch, skipping")
            continue
        
        # Run diffusion refinement
        print(f"  Running diffusion refinement...")
        refined_norm = run_inference(
            model, scheduler, hicarn_norm,
            res_mean, res_std, cond_mean, cond_std,
            alpha, device,
            use_gate=config.get('use_gate', True),
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            seed=args.seed
        )
        
        # ============================================
        # Denormalize to RAW space
        # ============================================
        print(f"  Denormalizing to RAW space...")
        
        # Denormalize HiCARN
        hicarn_raw = preprocessor.postprocess(hicarn_norm)
        print(f"    HiCARN (raw): range [{hicarn_raw.min():.2f}, {hicarn_raw.max():.2f}]")
        
        # Denormalize Refined
        refined_raw = preprocessor.postprocess(refined_norm)
        print(f"    Refined (raw): range [{refined_raw.min():.2f}, {refined_raw.max():.2f}]")
        
        # ============================================
        # Evaluate in RAW space
        # ============================================
        print(f"  Evaluating in RAW space...")
        
        # Squeeze to (N, H, W) for metrics
        hicarn_raw_3d = hicarn_raw[:, 0, :, :] if hicarn_raw.ndim == 4 else hicarn_raw
        refined_raw_3d = refined_raw[:, 0, :, :] if refined_raw.ndim == 4 else refined_raw
        gt_raw_3d = gt_raw[:, 0, :, :] if gt_raw.ndim == 4 else gt_raw
        
        # Evaluate HiCARN
        metrics_hicarn = VisionMetrics()
        metrics_hicarn.evaluate_batch(hicarn_raw_3d, gt_raw_3d)
        hicarn_results = metrics_hicarn.get_results()
        
        # Evaluate Refined
        metrics_refined = VisionMetrics()
        metrics_refined.evaluate_batch(refined_raw_3d, gt_raw_3d)
        refined_results = metrics_refined.get_results()
        
        # Store results
        all_results[chrom] = {
            'hicarn': hicarn_results,
            'refined': refined_results,
            'num_samples': len(gt_raw_3d)
        }
        
        # Print comparison
        print(f"\n  Results for {chrom} (RAW SPACE):")
        print(f"  {'Metric':<8} {'HiCARN':>12} {'Refined':>12} {'Improvement':>12}")
        print(f"  {'-'*48}")
        
        for metric in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
            h_val = hicarn_results[metric]['mean']
            r_val = refined_results[metric]['mean']
            
            if metric == 'mse':
                # Lower is better
                imp = (h_val - r_val) / h_val * 100 if h_val > 0 else 0
                imp_str = f"{imp:+.2f}%" if imp > 0 else f"{imp:.2f}%"
            else:
                # Higher is better
                imp = (r_val - h_val) / abs(h_val) * 100 if h_val != 0 else 0
                imp_str = f"{imp:+.2f}%" if imp > 0 else f"{imp:.2f}%"
            
            print(f"  {metric.upper():<8} {h_val:>12.4f} {r_val:>12.4f} {imp_str:>12}")
        
        # Save predictions
        if args.save_predictions:
            save_dir = output_dir / chrom
            save_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(save_dir / 'hicarn_raw.npy', hicarn_raw_3d)
            np.save(save_dir / 'refined_raw.npy', refined_raw_3d)
            np.save(save_dir / 'gt_raw.npy', gt_raw_3d)
            
            print(f"  Saved RAW predictions to {save_dir}")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*80)
    print("SUMMARY (ALL CHROMOSOMES, RAW SPACE)")
    print("="*80)
    
    if all_results:
        # Aggregate
        agg_hicarn = {m: [] for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']}
        agg_refined = {m: [] for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']}
        
        for chrom_results in all_results.values():
            for m in agg_hicarn.keys():
                agg_hicarn[m].append(chrom_results['hicarn'][m]['mean'])
                agg_refined[m].append(chrom_results['refined'][m]['mean'])
        
        print(f"\n{'Metric':<8} {'HiCARN':>12} {'Refined':>12} {'Improvement':>12}")
        print(f"{'-'*48}")
        
        summary = {'hicarn': {}, 'refined': {}}
        
        for metric in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
            h_mean = np.mean(agg_hicarn[metric])
            r_mean = np.mean(agg_refined[metric])
            
            summary['hicarn'][metric] = h_mean
            summary['refined'][metric] = r_mean
            
            if metric == 'mse':
                imp = (h_mean - r_mean) / h_mean * 100 if h_mean > 0 else 0
            else:
                imp = (r_mean - h_mean) / abs(h_mean) * 100 if h_mean != 0 else 0
            
            imp_str = f"{imp:+.2f}%"
            print(f"{metric.upper():<8} {h_mean:>12.4f} {r_mean:>12.4f} {imp_str:>12}")
        
        # Save results
        results_file = output_dir / 'test_results_raw.json'
        with open(results_file, 'w') as f:
            json.dump({
                'evaluation_space': 'raw_contact_counts',
                'per_chromosome': all_results,
                'summary': summary,
                'config': {
                    'checkpoint': args.checkpoint,
                    'alpha': alpha,
                    'num_steps': args.num_steps,
                }
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
    
    print("="*80)


if __name__ == '__main__':
    main()
