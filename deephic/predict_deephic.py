#!/usr/bin/env python3
"""
DeepHiC Prediction/Sampling Script

This script loads a trained DeepHiC model and generates predictions
for specified chromosomes. Outputs results in both normalized and raw space.

Usage:
    python predict_deephic.py \
        --checkpoint checkpoints_deephic/best_model.pt \
        --chromosomes chr18 chr19 chr20 chr21 chr22 \
        --gt_dir /data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40Locations \
        --output_dir predictions_deephic \
        --device cuda:2

Outputs:
    - predictions_norm.npy: Predictions in normalized space
    - predictions_raw.npy: Predictions in raw count space
    - ground_truth_norm.npy: Ground truth in normalized space
    - ground_truth_raw.npy: Ground truth in raw count space
    - input_lr_norm.npy: Low-resolution input in normalized space
    - input_lr_raw.npy: Low-resolution input in raw count space
    - metrics.json: Evaluation metrics (PCC, SSIM, MSE, PSNR)
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ================================================================
# HiCARN-style Preprocessor
# ================================================================

class HiCPreprocessor:
    """
    HiCARN-style preprocessor using log1p + z-score normalization.
    """
    def __init__(self):
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self.fitted = False
    
    def fit(self, X, Y):
        """Fit preprocessor on training data."""
        X_log = np.log1p(X.astype(np.float64))
        Y_log = np.log1p(Y.astype(np.float64))
        
        # Robust statistics using median + IQR
        X_flat = X_log.flatten()
        X_nonzero = X_flat[X_flat > 0]
        if len(X_nonzero) > 0:
            self.X_mean = float(np.median(X_nonzero))
            q75, q25 = np.percentile(X_nonzero, [75, 25])
            self.X_std = float((q75 - q25) / 1.35)
        else:
            self.X_mean, self.X_std = 0.0, 1.0
        
        Y_flat = Y_log.flatten()
        Y_nonzero = Y_flat[Y_flat > 0]
        if len(Y_nonzero) > 0:
            self.Y_mean = float(np.median(Y_nonzero))
            q75, q25 = np.percentile(Y_nonzero, [75, 25])
            self.Y_std = float((q75 - q25) / 1.35)
        else:
            self.Y_mean, self.Y_std = 0.0, 1.0
        
        self.X_std = max(self.X_std, 0.1)
        self.Y_std = max(self.Y_std, 0.1)
        self.fitted = True
    
    def transform_X(self, X):
        X_log = np.log1p(X.astype(np.float64))
        X_norm = (X_log - self.X_mean) / self.X_std
        return np.clip(X_norm, -5, 5).astype(np.float32)
    
    def transform_Y(self, Y):
        Y_log = np.log1p(Y.astype(np.float64))
        Y_norm = (Y_log - self.Y_mean) / self.Y_std
        return np.clip(Y_norm, -5, 5).astype(np.float32)
    
    def inverse_transform_X(self, X_norm):
        """Inverse transform LR predictions to raw counts"""
        X_norm = np.clip(X_norm, -5, 5)
        X_log = X_norm * self.X_std + self.X_mean
        X_raw = np.expm1(X_log)
        return np.maximum(X_raw, 0.0)
    
    def inverse_transform_Y(self, Y_norm):
        """Inverse transform predictions to raw counts"""
        Y_norm = np.clip(Y_norm, -5, 5)
        Y_log = Y_norm * self.Y_std + self.Y_mean
        Y_raw = np.expm1(Y_log)
        return np.maximum(Y_raw, 0.0)
    
    def save(self, path):
        torch.save({
            'X_mean': self.X_mean, 'X_std': self.X_std,
            'Y_mean': self.Y_mean, 'Y_std': self.Y_std,
            'fitted': self.fitted
        }, path)
    
    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu')
        preprocessor = cls()
        preprocessor.X_mean = data['X_mean']
        preprocessor.X_std = data['X_std']
        preprocessor.Y_mean = data['Y_mean']
        preprocessor.Y_std = data['Y_std']
        preprocessor.fitted = data.get('fitted', True)
        return preprocessor
    
    @classmethod
    def from_checkpoint(cls, ckpt):
        """Load preprocessor from checkpoint dict"""
        preprocessor = cls()
        if 'preprocessor' in ckpt:
            p = ckpt['preprocessor']
            preprocessor.X_mean = p['X_mean']
            preprocessor.X_std = p['X_std']
            preprocessor.Y_mean = p['Y_mean']
            preprocessor.Y_std = p['Y_std']
            preprocessor.fitted = True
        return preprocessor


# ================================================================
# DeepHiC Generator (must match training architecture)
# ================================================================

class ResidualBlock(nn.Module):
    """Residual block with two conv layers"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class Generator(nn.Module):
    """
    DeepHiC Generator Network
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_res_blocks=15):
        super().__init__()
        
        # Initial convolution
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 9, padding=4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        res_blocks = [ResidualBlock(num_features) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Post-residual convolution
        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features)
        )
        
        # Final output convolution
        self.conv_last = nn.Conv2d(num_features, out_channels, 9, padding=4)
    
    def forward(self, x):
        first = self.conv_first(x)
        res = self.res_blocks(first)
        res = self.conv_mid(res)
        out = first + res  # Skip connection
        out = self.conv_last(out)
        return out


# ================================================================
# SSIM Module (PyTorch-based)
# ================================================================

class SSIM(nn.Module):
    """PyTorch-based SSIM calculation"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.window = self._create_window(window_size)
    
    def _create_window(self, size, sigma=3):
        gauss = torch.Tensor([math.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2)) for x in range(size)])
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1).mm(gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        return window
    
    def forward(self, img1, img2):
        window = self.window.to(img1.device).type_as(img1)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2)
        
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2) - mu1_mu2
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


# ================================================================
# Vision Metrics (from test_v15.py)
# ================================================================

class VisionMetrics:
    """
    Comprehensive metrics calculator for Hi-C data evaluation.
    
    Metrics computed:
    - PCC: Pearson Correlation Coefficient
    - SPC: Spearman Correlation Coefficient
    - SSIM: Structural Similarity Index
    - PSNR: Peak Signal-to-Noise Ratio
    - SNR: Signal-to-Noise Ratio
    - MSE: Mean Squared Error
    """
    def __init__(self):
        self.ssim_module = SSIM()
        self.reset()
    
    def reset(self):
        self.logs = {k: [] for k in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']}
    
    def _psnr(self, pred, gt):
        mse = np.mean((pred - gt) ** 2)
        if mse == 0: return float('inf')
        max_val = max(gt.max(), pred.max())
        return 10 * np.log10((max_val ** 2) / mse) if max_val > 0 else 0
    
    def _snr(self, pred, gt):
        signal = np.mean(gt ** 2)
        noise = np.mean((pred - gt) ** 2)
        if noise == 0: return float('inf')
        return 10 * np.log10(signal / noise) if signal > 0 else 0
    
    def _pcc(self, pred, gt):
        if np.std(pred) == 0 or np.std(gt) == 0: return 0
        return pearsonr(pred.flatten(), gt.flatten())[0]
    
    def _spc(self, pred, gt):
        return spearmanr(pred.flatten(), gt.flatten())[0]
    
    def _ssim(self, pred, gt):
        p = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        g = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
        return self.ssim_module(p, g).item()
    
    def _mse(self, pred, gt):
        return np.mean((pred - gt) ** 2)
    
    def add_batch(self, pred, gt):
        """Add a batch of predictions and ground truth for evaluation."""
        if pred.ndim == 4: pred = pred[:, 0]
        if gt.ndim == 4: gt = gt[:, 0]
        
        for i in range(len(pred)):
            p, g = pred[i], gt[i]
            self.logs['pcc'].append(self._pcc(p, g))
            self.logs['spc'].append(self._spc(p, g))
            self.logs['ssim'].append(self._ssim(p, g))
            self.logs['psnr'].append(self._psnr(p, g))
            self.logs['snr'].append(self._snr(p, g))
            self.logs['mse'].append(self._mse(p, g))
    
    def get_summary(self):
        """Get summary statistics for all metrics."""
        return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)} 
                for k, v in self.logs.items() if v}
    
    def get_per_sample(self):
        """Get per-sample metrics."""
        n = len(self.logs['pcc'])
        return [
            {k: float(self.logs[k][i]) for k in self.logs.keys()}
            for i in range(n)
        ]


def compute_topk_iou(pred, gt, k_perc=0.5, min_diag=2):
    """
    Compute IoU for top-K percent of values (loop detection metric).
    
    Args:
        pred: Predictions (N, H, W) or (N, 1, H, W)
        gt: Ground truth (same shape)
        k_perc: Percentage of top values to consider
        min_diag: Minimum distance from diagonal
    
    Returns:
        dict with mean and std of IoU
    """
    if pred.ndim == 4: pred = pred[:, 0]
    if gt.ndim == 4: gt = gt[:, 0]
    
    N, H, W = pred.shape
    i_idx, j_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mask = (np.abs(j_idx - i_idx) >= min_diag) & (j_idx >= i_idx)
    
    ious = []
    for i in range(N):
        pv, gv = pred[i][mask], gt[i][mask]
        if len(pv) < 10: continue
        K = max(1, int(len(pv) * k_perc / 100))
        p_topk = set(np.argsort(pv)[-K:])
        g_topk = set(np.argsort(gv)[-K:])
        inter = len(p_topk & g_topk)
        union = len(p_topk | g_topk)
        ious.append(inter / union if union else 0)
    
    return {'mean': float(np.mean(ious)) if ious else 0, 'std': float(np.std(ious)) if ious else 0}


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# ================================================================
# Data Loading
# ================================================================

def load_test_data(gt_dir, chromosomes, ratio=16):
    """
    Load test data for specified chromosomes.
    
    Args:
        gt_dir: Directory containing ground truth data
        chromosomes: List of chromosome names (e.g., ['chr18', 'chr19'])
        ratio: Downsampling ratio (default: 16)
    
    Returns:
        lr_data: Low-resolution data (raw)
        hr_data: High-resolution data (raw)
        chr_indices: Dictionary mapping chromosome to indices
    """
    gt_dir = Path(gt_dir)
    
    all_lr = []
    all_hr = []
    chr_indices = {}
    current_idx = 0
    
    for chrom in chromosomes:
        # Try different file naming patterns for separate .npy files
        # Pattern: lr has ratio, hr does not
        lr_hr_pattern_pairs = [
            # Pattern 1: lr_test_chr18_ratio16.npy + hr_test_chr18.npy
            (f"lr_test_{chrom}_ratio{ratio}.npy", f"hr_test_{chrom}.npy"),
            # Pattern 2: lr_test_chr18_ratio16.npy + hr_test_chr18_ratio16.npy
            (f"lr_test_{chrom}_ratio{ratio}.npy", f"hr_test_{chrom}_ratio{ratio}.npy"),
            # Pattern 3: test_lr_chr18_ratio16.npy + test_hr_chr18.npy
            (f"test_lr_{chrom}_ratio{ratio}.npy", f"test_hr_{chrom}.npy"),
            # Pattern 4: chr18_lr_ratio16.npy + chr18_hr.npy
            (f"{chrom}_lr_ratio{ratio}.npy", f"{chrom}_hr.npy"),
            # Pattern 5: lr_chr18_ratio16.npy + hr_chr18.npy
            (f"lr_{chrom}_ratio{ratio}.npy", f"hr_{chrom}.npy"),
            # Pattern 6: Both with ratio
            (f"lr_{chrom}_ratio{ratio}.npy", f"hr_{chrom}_ratio{ratio}.npy"),
        ]
        
        # Also try .npz patterns
        npz_patterns = [
            f"test_{chrom}_ratio{ratio}.npz",
            f"{chrom}_test_ratio{ratio}.npz",
            f"{chrom}_ratio{ratio}.npz",
            f"test_data_{chrom}.npz",
            f"{chrom}.npz",
        ]
        
        found = False
        
        # First try separate .npy files
        for lr_pattern, hr_pattern in lr_hr_pattern_pairs:
            lr_path = gt_dir / lr_pattern
            hr_path = gt_dir / hr_pattern
            
            if lr_path.exists() and hr_path.exists():
                print(f"  Loading {lr_pattern} and {hr_pattern}")
                lr = np.load(lr_path)
                hr = np.load(hr_path)
                
                n_samples = len(lr)
                chr_indices[chrom] = (current_idx, current_idx + n_samples)
                current_idx += n_samples
                
                all_lr.append(lr)
                all_hr.append(hr)
                
                print(f"    {chrom}: {n_samples} samples")
                found = True
                break
        
        # If not found, try .npz files
        if not found:
            for pattern in npz_patterns:
                filepath = gt_dir / pattern
                if filepath.exists():
                    print(f"  Loading {filepath}")
                    data = np.load(filepath)
                    
                    # Try different key names
                    if 'lr' in data and 'hr' in data:
                        lr = data['lr']
                        hr = data['hr']
                    elif 'X' in data and 'Y' in data:
                        lr = data['X']
                        hr = data['Y']
                    elif 'low' in data and 'high' in data:
                        lr = data['low']
                        hr = data['high']
                    else:
                        keys = list(data.keys())
                        print(f"    Available keys: {keys}")
                        if len(keys) >= 2:
                            lr = data[keys[0]]
                            hr = data[keys[1]]
                        else:
                            print(f"    Warning: Cannot find LR/HR data in {filepath}")
                            continue
                    
                    n_samples = len(lr)
                    chr_indices[chrom] = (current_idx, current_idx + n_samples)
                    current_idx += n_samples
                    
                    all_lr.append(lr)
                    all_hr.append(hr)
                    
                    print(f"    {chrom}: {n_samples} samples")
                    found = True
                    break
        
        if not found:
            print(f"  Warning: No data found for {chrom}")
    
    if len(all_lr) == 0:
        raise ValueError(f"No data found in {gt_dir} for chromosomes {chromosomes}")
    
    lr_data = np.concatenate(all_lr, axis=0)
    hr_data = np.concatenate(all_hr, axis=0)
    
    return lr_data, hr_data, chr_indices


# ================================================================
# Visualization
# ================================================================

def visualize_samples(lr, hr, pred, output_dir, num_samples=4, prefix=''):
    """Visualize sample predictions."""
    num_samples = min(num_samples, len(lr))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(num_samples):
        # Squeeze if needed
        lr_img = lr[i].squeeze()
        hr_img = hr[i].squeeze()
        pred_img = pred[i].squeeze()
        
        vmax = max(hr_img.max(), pred_img.max())
        
        # LR Input
        ax = axes[i, 0]
        im = ax.imshow(lr_img, cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title(f'Sample {i+1}: LR Input')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # HR Ground Truth
        ax = axes[i, 1]
        im = ax.imshow(hr_img, cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title('HR Ground Truth')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Prediction
        ax = axes[i, 2]
        im = ax.imshow(pred_img, cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title('DeepHiC Prediction')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Difference
        ax = axes[i, 3]
        diff = pred_img - hr_img
        vmax_diff = max(abs(diff.min()), abs(diff.max()))
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
        ax.set_title('Difference (Pred - GT)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}sample_predictions.png', dpi=150)
    plt.close()
    print(f"  Saved visualization to {output_dir}/{prefix}sample_predictions.png")


# ================================================================
# Main Prediction Function
# ================================================================

def predict(args):
    """Main prediction function."""
    
    # Create output directory structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'norm').mkdir(exist_ok=True)
    (output_dir / 'raw').mkdir(exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("DeepHiC Prediction")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Chromosomes: {args.chromosomes}")
    print(f"Output directory: {output_dir}")
    
    # ================================================================
    # Load checkpoint
    # ================================================================
    print(f"\n{'='*70}")
    print("LOADING MODEL")
    print(f"{'='*70}")
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Get model config from checkpoint
    config = ckpt.get('config', {})
    num_features = config.get('num_features', 64)
    num_res_blocks = config.get('num_res_blocks', 15)
    
    print(f"  Model config: num_features={num_features}, num_res_blocks={num_res_blocks}")
    
    # Create model
    model = Generator(
        in_channels=1,
        out_channels=1,
        num_features=num_features,
        num_res_blocks=num_res_blocks
    ).to(device)
    
    # Load weights
    model.load_state_dict(ckpt['generator_state_dict'])
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e6:.2f}M")
    
    # Load preprocessor
    preprocessor = HiCPreprocessor.from_checkpoint(ckpt)
    if preprocessor.fitted:
        print(f"  Preprocessor loaded from checkpoint:")
        print(f"    X: mean={preprocessor.X_mean:.4f}, std={preprocessor.X_std:.4f}")
        print(f"    Y: mean={preprocessor.Y_mean:.4f}, std={preprocessor.Y_std:.4f}")
    else:
        # Try to load from file
        preprocessor_path = Path(args.checkpoint).parent / 'preprocessor.pt'
        if preprocessor_path.exists():
            preprocessor = HiCPreprocessor.load(preprocessor_path)
            print(f"  Preprocessor loaded from {preprocessor_path}")
        else:
            raise ValueError("No preprocessor found in checkpoint or directory!")
    
    # ================================================================
    # Process each chromosome
    # ================================================================
    print(f"\n{'='*70}")
    print("PROCESSING CHROMOSOMES")
    print(f"{'='*70}")
    
    gt_dir = Path(args.gt_dir)
    
    # Overall metrics accumulators
    all_metrics_norm = VisionMetrics()
    all_metrics_raw = VisionMetrics()
    
    # Per-chromosome results
    chr_results = {}
    
    for chrom in args.chromosomes:
        print(f"\n>>> Processing {chrom}")
        
        # Try to find data files
        lr_hr_pattern_pairs = [
            (f"lr_test_{chrom}_ratio{args.ratio}.npy", f"hr_test_{chrom}.npy"),
            (f"lr_test_{chrom}_ratio{args.ratio}.npy", f"hr_test_{chrom}_ratio{args.ratio}.npy"),
            (f"lr_{chrom}_ratio{args.ratio}.npy", f"hr_{chrom}.npy"),
        ]
        
        lr_path, hr_path = None, None
        for lr_pattern, hr_pattern in lr_hr_pattern_pairs:
            if (gt_dir / lr_pattern).exists() and (gt_dir / hr_pattern).exists():
                lr_path = gt_dir / lr_pattern
                hr_path = gt_dir / hr_pattern
                break
        
        if lr_path is None or hr_path is None:
            print(f"  Skipping {chrom}: Data files not found")
            continue
        
        print(f"  Loading {lr_path.name} and {hr_path.name}")
        
        # Load data
        lr_raw = np.load(lr_path)
        hr_raw = np.load(hr_path)
        
        # Ensure NCHW format
        if lr_raw.ndim == 3:
            lr_raw = lr_raw[:, np.newaxis, :, :]
        elif lr_raw.ndim == 4 and lr_raw.shape[-1] == 1:
            lr_raw = np.transpose(lr_raw, (0, 3, 1, 2))
        
        if hr_raw.ndim == 3:
            hr_raw = hr_raw[:, np.newaxis, :, :]
        elif hr_raw.ndim == 4 and hr_raw.shape[-1] == 1:
            hr_raw = np.transpose(hr_raw, (0, 3, 1, 2))
        
        print(f"  Samples: {len(lr_raw)}, Shape: {lr_raw.shape}")
        
        # Normalize
        lr_norm = preprocessor.transform_X(lr_raw)
        hr_norm = preprocessor.transform_Y(hr_raw)
        
        # Run predictions
        predictions_norm = []
        with torch.no_grad():
            for i in range(0, len(lr_norm), args.batch_size):
                batch = lr_norm[i:i+args.batch_size]
                batch_tensor = torch.from_numpy(batch).float().to(device)
                pred = model(batch_tensor)
                predictions_norm.append(pred.cpu().numpy())
        
        predictions_norm = np.concatenate(predictions_norm, axis=0)
        
        # Convert to raw space
        predictions_raw = preprocessor.inverse_transform_Y(predictions_norm)
        lr_raw_denorm = preprocessor.inverse_transform_X(lr_norm)
        hr_raw_denorm = preprocessor.inverse_transform_Y(hr_norm)
        
        # ================================================================
        # Compute metrics using VisionMetrics
        # ================================================================
        
        # Normalized space metrics
        chr_metrics_norm = VisionMetrics()
        chr_metrics_norm.add_batch(predictions_norm, hr_norm)
        norm_summary = chr_metrics_norm.get_summary()
        norm_iou = compute_topk_iou(predictions_norm, hr_norm, k_perc=0.5)
        
        # Raw space metrics
        chr_metrics_raw = VisionMetrics()
        chr_metrics_raw.add_batch(predictions_raw, hr_raw_denorm)
        raw_summary = chr_metrics_raw.get_summary()
        raw_iou = compute_topk_iou(predictions_raw, hr_raw_denorm, k_perc=0.5)
        
        # Add to overall accumulators
        all_metrics_norm.add_batch(predictions_norm, hr_norm)
        all_metrics_raw.add_batch(predictions_raw, hr_raw_denorm)
        
        # Print chromosome results
        print(f"\n  [NORMALIZED SPACE]")
        print(f"  {'Metric':<8} {'Value':>10}")
        print(f"  {'-'*20}")
        for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
            print(f"  {m.upper():<8} {norm_summary[m]['mean']:>10.4f}")
        print(f"  {'IoU@0.5':<8} {norm_iou['mean']:>10.4f}")
        
        print(f"\n  [RAW SPACE]")
        print(f"  {'Metric':<8} {'Value':>10}")
        print(f"  {'-'*20}")
        for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
            print(f"  {m.upper():<8} {raw_summary[m]['mean']:>10.4f}")
        print(f"  {'IoU@0.5':<8} {raw_iou['mean']:>10.4f}")
        
        # Store results
        chr_results[chrom] = {
            'n_samples': len(predictions_norm),
            'norm': {**norm_summary, 'iou': norm_iou},
            'raw': {**raw_summary, 'iou': raw_iou}
        }
        
        # ================================================================
        # Save per-chromosome files
        # ================================================================
        
        # Normalized space
        np.save(output_dir / 'norm' / f'predictions_{chrom}.npy', predictions_norm)
        np.save(output_dir / 'norm' / f'ground_truth_{chrom}.npy', hr_norm)
        np.save(output_dir / 'norm' / f'input_lr_{chrom}.npy', lr_norm)
        
        # Raw space
        np.save(output_dir / 'raw' / f'predictions_{chrom}.npy', predictions_raw)
        np.save(output_dir / 'raw' / f'ground_truth_{chrom}.npy', hr_raw_denorm)
        np.save(output_dir / 'raw' / f'input_lr_{chrom}.npy', lr_raw_denorm)
        
        # Per-chromosome metrics JSON
        chr_metrics_data = convert_to_serializable(chr_results[chrom])
        with open(output_dir / f'metrics_{chrom}.json', 'w') as f:
            json.dump(chr_metrics_data, f, indent=2)
        
        print(f"  Saved: predictions_{chrom}.npy, ground_truth_{chrom}.npy, metrics_{chrom}.json")
    
    # ================================================================
    # Overall Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    
    overall_norm = all_metrics_norm.get_summary()
    overall_raw = all_metrics_raw.get_summary()
    
    if overall_norm:
        n_total = overall_norm.get('pcc', {}).get('n', 0)
        
        print(f"\n[NORMALIZED SPACE] - {n_total} total samples")
        print(f"{'Metric':<8} {'Mean':>10} {'Std':>10}")
        print(f"{'-'*30}")
        for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
            mean_val = overall_norm.get(m, {}).get('mean', 0)
            std_val = overall_norm.get(m, {}).get('std', 0)
            print(f"{m.upper():<8} {mean_val:>10.4f} {std_val:>10.4f}")
        
        print(f"\n[RAW SPACE] - {n_total} total samples")
        print(f"{'Metric':<8} {'Mean':>10} {'Std':>10}")
        print(f"{'-'*30}")
        for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
            mean_val = overall_raw.get(m, {}).get('mean', 0)
            std_val = overall_raw.get(m, {}).get('std', 0)
            print(f"{m.upper():<8} {mean_val:>10.4f} {std_val:>10.4f}")
    
    # ================================================================
    # Save overall results
    # ================================================================
    print(f"\n{'='*70}")
    print("SAVING OVERALL RESULTS")
    print(f"{'='*70}")
    
    all_results = {
        'overall': {
            'norm': convert_to_serializable(overall_norm),
            'raw': convert_to_serializable(overall_raw)
        },
        'per_chromosome': convert_to_serializable(chr_results),
        'config': {
            'checkpoint': str(args.checkpoint),
            'chromosomes': args.chromosomes,
            'gt_dir': str(args.gt_dir),
            'device': str(device)
        }
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved evaluation_results.json")
    
    # ================================================================
    # Generate visualizations
    # ================================================================
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    # Visualize first chromosome that has data
    for chrom in args.chromosomes:
        norm_pred_path = output_dir / 'norm' / f'predictions_{chrom}.npy'
        if norm_pred_path.exists():
            pred_norm = np.load(norm_pred_path)
            gt_norm = np.load(output_dir / 'norm' / f'ground_truth_{chrom}.npy')
            lr_norm = np.load(output_dir / 'norm' / f'input_lr_{chrom}.npy')
            
            pred_raw = np.load(output_dir / 'raw' / f'predictions_{chrom}.npy')
            gt_raw = np.load(output_dir / 'raw' / f'ground_truth_{chrom}.npy')
            lr_raw = np.load(output_dir / 'raw' / f'input_lr_{chrom}.npy')
            
            # Visualize normalized
            visualize_samples(lr_norm, gt_norm, pred_norm, output_dir, 
                            num_samples=args.num_vis_samples, prefix=f'norm_{chrom}_')
            
            # Visualize raw
            visualize_samples(lr_raw, gt_raw, pred_raw, output_dir,
                            num_samples=args.num_vis_samples, prefix=f'raw_{chrom}_')
            break
    
    # ================================================================
    # Final Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("PREDICTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Processed chromosomes: {list(chr_results.keys())}")
    print(f"  Total samples: {sum(r['n_samples'] for r in chr_results.values())}")
    print(f"\n  Output structure:")
    print(f"    {output_dir}/")
    print(f"    ├── norm/           # Normalized space outputs")
    print(f"    │   ├── predictions_chrX.npy")
    print(f"    │   ├── ground_truth_chrX.npy")
    print(f"    │   └── input_lr_chrX.npy")
    print(f"    ├── raw/            # Raw space outputs")
    print(f"    │   ├── predictions_chrX.npy")
    print(f"    │   ├── ground_truth_chrX.npy")
    print(f"    │   └── input_lr_chrX.npy")
    print(f"    ├── metrics_chrX.json   # Per-chromosome metrics")
    print(f"    └── evaluation_results.json  # Overall results")
    print(f"{'='*70}\n")


# ================================================================
# Main Entry Point
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DeepHiC Prediction Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (best_model.pt)')
    parser.add_argument('--chromosomes', type=str, nargs='+', required=True,
                        help='Chromosomes to predict (e.g., chr18 chr19 chr20)')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth test data')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='predictions_deephic',
                        help='Output directory for predictions')
    
    # Optional
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda, cuda:0, cpu)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for prediction')
    parser.add_argument('--ratio', type=int, default=16,
                        help='Downsampling ratio')
    parser.add_argument('--num_vis_samples', type=int, default=4,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    predict(args)


if __name__ == '__main__':
    main()
