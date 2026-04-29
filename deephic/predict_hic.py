#!/usr/bin/env python3
"""
Unified Prediction Script for Hi-C Super-Resolution Models

Supports: SRHiC, DeepHiC, HiCNN

This script loads a trained model and generates predictions from NPZ input files.
Outputs results in both normalized and raw space.

Input NPZ format:
    - 'train_lr' or 'lr' or 'X': Low-resolution data
    - 'train_hr' or 'hr' or 'Y': High-resolution ground truth
    - 'distances' (optional): Distance information
    - 'locations' (optional): Location information

Usage:
    python predict_hic.py \
        --model srhic \
        --checkpoint checkpoints_srhic/best_pcc.pt \
        --input_npz /path/to/data.npz \
        --output_dir predictions_srhic \
        --device cuda:2

    # Or with separate test files per chromosome:
    python predict_hic.py \
        --model deephic \
        --checkpoint checkpoints_deephic/best_model.pt \
        --data_dir /path/to/data_dir \
        --chromosomes chr18 chr19 chr20 chr21 chr22 \
        --output_dir predictions_deephic \
        --device cuda:2
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
# Preprocessor
# ================================================================

def ensure_nchw(x):
    """Ensure data is in NCHW format."""
    x = np.asarray(x)
    if x.ndim == 3:
        return x[:, None, :, :]
    elif x.ndim == 4 and x.shape[1] in [1, 3]:
        return x
    elif x.ndim == 4 and x.shape[-1] in [1, 3]:
        return np.transpose(x, (0, 3, 1, 2))
    else:
        raise ValueError(f"Cannot convert to NCHW: shape={x.shape}")


class HiCPreprocessor:
    """
    Preprocessor for Hi-C Data using log1p + robust normalization.
    Compatible with both HiCARN-style and RobustHiC-style preprocessing.
    """
    
    def __init__(self):
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self._is_fitted = False

    def fit(self, X_low, Y_high):
        """Fit normalization statistics on training data."""
        X_low = ensure_nchw(X_low)
        Y_high = ensure_nchw(Y_high)
        
        X_log = np.log1p(X_low)
        Y_log = np.log1p(Y_high)
        
        # Robust statistics: median and IQR
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
        self._is_fitted = True
    
    def transform_X(self, X):
        """Transform LR data to normalized space."""
        X = ensure_nchw(X)
        X_log = np.log1p(X.astype(np.float64))
        X_norm = (X_log - self.X_mean) / self.X_std
        return np.clip(X_norm, -5, 5).astype(np.float32)
    
    def transform_Y(self, Y):
        """Transform HR data to normalized space."""
        Y = ensure_nchw(Y)
        Y_log = np.log1p(Y.astype(np.float64))
        Y_norm = (Y_log - self.Y_mean) / self.Y_std
        return np.clip(Y_norm, -5, 5).astype(np.float32)

    def inverse_transform_X(self, X_norm):
        """Inverse transform LR data to raw space."""
        X_norm = np.clip(X_norm, -5, 5)
        X_log = X_norm * self.X_std + self.X_mean
        X_raw = np.expm1(X_log)
        return np.maximum(X_raw, 0.0)
    
    def inverse_transform_Y(self, Y_norm):
        """Inverse transform HR/predictions to raw space."""
        Y_norm = np.clip(Y_norm, -5, 5)
        Y_log = Y_norm * self.Y_std + self.Y_mean
        Y_raw = np.expm1(Y_log)
        return np.maximum(Y_raw, 0.0)
    
    @classmethod
    def from_checkpoint(cls, ckpt):
        """Load preprocessor from checkpoint dict."""
        preprocessor = cls()
        if 'preprocessor' in ckpt:
            p = ckpt['preprocessor']
            preprocessor.X_mean = p.get('X_mean', p.get('x_mean'))
            preprocessor.X_std = p.get('X_std', p.get('x_std'))
            preprocessor.Y_mean = p.get('Y_mean', p.get('y_mean'))
            preprocessor.Y_std = p.get('Y_std', p.get('y_std'))
            preprocessor._is_fitted = True
        return preprocessor
    
    @classmethod
    def load(cls, path):
        """Load preprocessor from file."""
        state = torch.load(path, map_location='cpu')
        preprocessor = cls()
        preprocessor.X_mean = state.get('X_mean', state.get('x_mean'))
        preprocessor.X_std = state.get('X_std', state.get('x_std'))
        preprocessor.Y_mean = state.get('Y_mean', state.get('y_mean'))
        preprocessor.Y_std = state.get('Y_std', state.get('y_std'))
        preprocessor._is_fitted = True
        return preprocessor


# ================================================================
# Model Architectures
# ================================================================

# ----- SRHiC -----
class SRHiCResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class SRHiC(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_residual_blocks=16, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size, padding=padding),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(
            *[SRHiCResidualBlock(num_features, kernel_size) for _ in range(num_residual_blocks)]
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size, padding=padding),
            nn.BatchNorm2d(num_features)
        )
        self.conv_out = nn.Conv2d(num_features, out_channels, kernel_size, padding=padding)
        self.global_residual = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        identity = x
        out = self.conv_in(x)
        residual = out
        out = self.residual_blocks(out)
        out = self.conv_mid(out)
        out = out + residual
        out = self.conv_out(out)
        out = identity + self.global_residual * out
        return out


class SRHiCLarge(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=128, num_residual_blocks=24, kernel_size=3):
        super().__init__()
        self.conv_in_3x3 = nn.Conv2d(in_channels, num_features // 2, 3, padding=1)
        self.conv_in_5x5 = nn.Conv2d(in_channels, num_features // 2, 5, padding=2)
        self.bn_in = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = nn.Sequential(
            *[SRHiCResidualBlock(num_features, kernel_size) for _ in range(num_residual_blocks)]
        )
        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, out_channels, 3, padding=1)
        )
        self.global_residual = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        identity = x
        feat_3x3 = self.conv_in_3x3(x)
        feat_5x5 = self.conv_in_5x5(x)
        out = torch.cat([feat_3x3, feat_5x5], dim=1)
        out = self.relu(self.bn_in(out))
        residual = out
        out = self.residual_blocks(out)
        out = out + residual
        out = self.refine(out) + out
        out = self.conv_out(out)
        out = identity + self.global_residual * out
        return out


# ----- DeepHiC -----
class DeepHiCResidualBlock(nn.Module):
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


class DeepHiCGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_res_blocks=15):
        super().__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 9, padding=4),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[DeepHiCResidualBlock(num_features) for _ in range(num_res_blocks)])
        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features)
        )
        self.conv_last = nn.Conv2d(num_features, out_channels, 9, padding=4)
    
    def forward(self, x):
        first = self.conv_first(x)
        res = self.res_blocks(first)
        res = self.conv_mid(res)
        out = first + res
        out = self.conv_last(out)
        return out


# ----- HiCNN -----
class HiCNN(nn.Module):
    def __init__(self, num_res_blocks=25, num_features=128, output_full_size=True):
        super().__init__()
        self.output_full_size = output_full_size
        
        if output_full_size:
            self.conv1 = nn.Conv2d(1, 8, 9, padding=4)
            self.conv2 = nn.Conv2d(8, 1, 1)
        else:
            self.conv1 = nn.Conv2d(1, 8, 13)
            self.conv2 = nn.Conv2d(8, 1, 1)
        
        self.conv3 = nn.Conv2d(1, num_features, 3, padding=1, bias=False)
        self.conv4R = nn.Conv2d(num_features, num_features, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(num_features, 1, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.num_res_blocks = num_res_blocks
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        residual = x
        x2 = self.conv3(x)
        out = x2
        for _ in range(self.num_res_blocks):
            out = self.conv4R(self.relu(self.conv4R(self.relu(out))))
            out = torch.add(out, x2)
        out = self.conv5(self.relu(out))
        out = torch.add(out, residual)
        return out


# ================================================================
# SSIM Module
# ================================================================

class SSIM(nn.Module):
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
# Metrics
# ================================================================

class VisionMetrics:
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
        return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)} 
                for k, v in self.logs.items() if v}


def compute_topk_iou(pred, gt, k_perc=0.5, min_diag=2):
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
# Visualization
# ================================================================

def visualize_samples(lr, hr, pred, output_dir, num_samples=4, prefix='', model_name='Model'):
    num_samples = min(num_samples, len(lr))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(num_samples):
        lr_img = lr[i].squeeze()
        hr_img = hr[i].squeeze()
        pred_img = pred[i].squeeze()
        vmax = max(hr_img.max(), pred_img.max())
        
        ax = axes[i, 0]
        im = ax.imshow(lr_img, cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title(f'Sample {i+1}: LR Input')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[i, 1]
        im = ax.imshow(hr_img, cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title('HR Ground Truth')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[i, 2]
        im = ax.imshow(pred_img, cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title(f'{model_name} Prediction')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[i, 3]
        diff = pred_img - hr_img
        vmax_diff = max(abs(diff.min()), abs(diff.max()), 1e-6)
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
        ax.set_title('Difference (Pred - GT)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}sample_predictions.png', dpi=150)
    plt.close()
    print(f"  Saved: {prefix}sample_predictions.png")


# ================================================================
# Data Loading
# ================================================================

def load_npz_data(npz_path):
    """
    Load data from NPZ file with various key formats.
    
    Supported keys:
        LR: 'train_lr', 'lr', 'X', 'low', 'input'
        HR: 'train_hr', 'hr', 'Y', 'high', 'target'
        distances: 'distances', 'distance'
        locations: 'locations', 'location'
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    print(f"  NPZ keys: {keys}")
    
    # Find LR data
    lr_keys = ['train_lr', 'lr', 'X', 'low', 'input']
    lr_data = None
    for k in lr_keys:
        if k in data:
            lr_data = data[k]
            print(f"  LR data from key '{k}': shape={lr_data.shape}")
            break
    
    # Find HR data
    hr_keys = ['train_hr', 'hr', 'Y', 'high', 'target']
    hr_data = None
    for k in hr_keys:
        if k in data:
            hr_data = data[k]
            print(f"  HR data from key '{k}': shape={hr_data.shape}")
            break
    
    # Find optional data
    distances = None
    for k in ['distances', 'distance']:
        if k in data:
            distances = data[k]
            print(f"  Distances from key '{k}': shape={distances.shape}")
            break
    
    locations = None
    for k in ['locations', 'location']:
        if k in data:
            locations = data[k]
            print(f"  Locations from key '{k}': shape={locations.shape}")
            break
    
    if lr_data is None:
        raise ValueError(f"Cannot find LR data. Available keys: {keys}")
    if hr_data is None:
        raise ValueError(f"Cannot find HR data. Available keys: {keys}")
    
    return lr_data, hr_data, distances, locations


# ================================================================
# Model Creation
# ================================================================

def create_model(model_type, config, device):
    """Create model based on type and config."""
    
    if model_type == 'srhic':
        model_size = config.get('model_size', 'base')
        num_features = config.get('num_features', 64)
        num_blocks = config.get('num_blocks', config.get('num_residual_blocks', 16))
        
        if model_size == 'large':
            model = SRHiCLarge(
                in_channels=1, out_channels=1,
                num_features=num_features,
                num_residual_blocks=num_blocks
            )
        else:
            model = SRHiC(
                in_channels=1, out_channels=1,
                num_features=num_features,
                num_residual_blocks=num_blocks
            )
        print(f"  SRHiC ({model_size}): features={num_features}, blocks={num_blocks}")
    
    elif model_type == 'deephic':
        num_features = config.get('num_features', 64)
        num_res_blocks = config.get('num_res_blocks', 15)
        
        model = DeepHiCGenerator(
            in_channels=1, out_channels=1,
            num_features=num_features,
            num_res_blocks=num_res_blocks
        )
        print(f"  DeepHiC: features={num_features}, blocks={num_res_blocks}")
    
    elif model_type == 'hicnn':
        num_res_blocks = config.get('num_res_blocks', 25)
        num_features = config.get('num_features', 128)
        original_arch = config.get('original_arch', False)
        
        model = HiCNN(
            num_res_blocks=num_res_blocks,
            num_features=num_features,
            output_full_size=not original_arch
        )
        print(f"  HiCNN: features={num_features}, blocks={num_res_blocks}")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def load_model_weights(model, ckpt, model_type):
    """Load model weights from checkpoint."""
    if model_type == 'deephic':
        # DeepHiC uses 'generator_state_dict'
        if 'generator_state_dict' in ckpt:
            model.load_state_dict(ckpt['generator_state_dict'])
        else:
            model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt['model_state_dict'])
    return model


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
    model_name = args.model.upper()
    
    print(f"\n{'='*70}")
    print(f"{model_name} Prediction")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input_npz}")
    print(f"Output: {output_dir}")
    
    # ================================================================
    # Load checkpoint
    # ================================================================
    print(f"\n{'='*70}")
    print("LOADING MODEL")
    print(f"{'='*70}")
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get('config', {})
    
    # Create and load model
    model = create_model(args.model, config, device)
    model = load_model_weights(model, ckpt, args.model)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e6:.2f}M")
    
    # Load preprocessor
    preprocessor = HiCPreprocessor.from_checkpoint(ckpt)
    if preprocessor._is_fitted:
        print(f"  Preprocessor from checkpoint:")
        print(f"    X: mean={preprocessor.X_mean:.4f}, std={preprocessor.X_std:.4f}")
        print(f"    Y: mean={preprocessor.Y_mean:.4f}, std={preprocessor.Y_std:.4f}")
    else:
        preprocessor_path = Path(args.checkpoint).parent / 'preprocessor.pt'
        if preprocessor_path.exists():
            preprocessor = HiCPreprocessor.load(preprocessor_path)
            print(f"  Preprocessor from {preprocessor_path}")
        else:
            raise ValueError("No preprocessor found!")
    
    # ================================================================
    # Load data
    # ================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    
    lr_raw, hr_raw, distances, locations = load_npz_data(args.input_npz)
    
    # Ensure NCHW format
    lr_raw = ensure_nchw(lr_raw)
    hr_raw = ensure_nchw(hr_raw)
    
    print(f"  Total samples: {len(lr_raw)}")
    print(f"  LR shape: {lr_raw.shape}, range: [{lr_raw.min():.2f}, {lr_raw.max():.2f}]")
    print(f"  HR shape: {hr_raw.shape}, range: [{hr_raw.min():.2f}, {hr_raw.max():.2f}]")
    
    # ================================================================
    # Normalize data
    # ================================================================
    print(f"\n{'='*70}")
    print("PREPROCESSING")
    print(f"{'='*70}")
    
    lr_norm = preprocessor.transform_X(lr_raw)
    hr_norm = preprocessor.transform_Y(hr_raw)
    
    print(f"  LR norm range: [{lr_norm.min():.4f}, {lr_norm.max():.4f}]")
    print(f"  HR norm range: [{hr_norm.min():.4f}, {hr_norm.max():.4f}]")
    
    # ================================================================
    # Run predictions
    # ================================================================
    print(f"\n{'='*70}")
    print("RUNNING PREDICTIONS")
    print(f"{'='*70}")
    
    predictions_norm = []
    with torch.no_grad():
        for i in tqdm(range(0, len(lr_norm), args.batch_size), desc="Predicting"):
            batch = lr_norm[i:i+args.batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            pred = model(batch_tensor)
            predictions_norm.append(pred.cpu().numpy())
    
    predictions_norm = np.concatenate(predictions_norm, axis=0)
    print(f"  Predictions shape: {predictions_norm.shape}")
    print(f"  Predictions norm range: [{predictions_norm.min():.4f}, {predictions_norm.max():.4f}]")
    
    # Convert to raw space
    predictions_raw = preprocessor.inverse_transform_Y(predictions_norm)
    lr_raw_recon = preprocessor.inverse_transform_X(lr_norm)
    hr_raw_recon = preprocessor.inverse_transform_Y(hr_norm)
    
    print(f"  Predictions raw range: [{predictions_raw.min():.2f}, {predictions_raw.max():.2f}]")
    
    # ================================================================
    # Compute metrics
    # ================================================================
    print(f"\n{'='*70}")
    print("COMPUTING METRICS")
    print(f"{'='*70}")
    
    # Normalized space metrics
    metrics_norm = VisionMetrics()
    metrics_norm.add_batch(predictions_norm, hr_norm)
    norm_summary = metrics_norm.get_summary()
    norm_iou = compute_topk_iou(predictions_norm, hr_norm, k_perc=0.5)
    
    print(f"\n[NORMALIZED SPACE] - {norm_summary['pcc']['n']} samples")
    print(f"{'Metric':<8} {'Mean':>10} {'Std':>10}")
    print(f"{'-'*30}")
    for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
        print(f"{m.upper():<8} {norm_summary[m]['mean']:>10.4f} {norm_summary[m]['std']:>10.4f}")
    print(f"{'IoU@0.5':<8} {norm_iou['mean']:>10.4f} {norm_iou['std']:>10.4f}")
    
    # Raw space metrics
    metrics_raw = VisionMetrics()
    metrics_raw.add_batch(predictions_raw, hr_raw_recon)
    raw_summary = metrics_raw.get_summary()
    raw_iou = compute_topk_iou(predictions_raw, hr_raw_recon, k_perc=0.5)
    
    print(f"\n[RAW SPACE] - {raw_summary['pcc']['n']} samples")
    print(f"{'Metric':<8} {'Mean':>10} {'Std':>10}")
    print(f"{'-'*30}")
    for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
        print(f"{m.upper():<8} {raw_summary[m]['mean']:>10.4f} {raw_summary[m]['std']:>10.4f}")
    print(f"{'IoU@0.5':<8} {raw_iou['mean']:>10.4f} {raw_iou['std']:>10.4f}")
    
    # ================================================================
    # Save results
    # ================================================================
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save normalized space
    np.save(output_dir / 'norm' / 'predictions.npy', predictions_norm)
    np.save(output_dir / 'norm' / 'ground_truth.npy', hr_norm)
    np.save(output_dir / 'norm' / 'input_lr.npy', lr_norm)
    print(f"  Saved norm/predictions.npy: {predictions_norm.shape}")
    print(f"  Saved norm/ground_truth.npy: {hr_norm.shape}")
    print(f"  Saved norm/input_lr.npy: {lr_norm.shape}")
    
    # Save raw space
    np.save(output_dir / 'raw' / 'predictions.npy', predictions_raw)
    np.save(output_dir / 'raw' / 'ground_truth.npy', hr_raw_recon)
    np.save(output_dir / 'raw' / 'input_lr.npy', lr_raw_recon)
    print(f"  Saved raw/predictions.npy: {predictions_raw.shape}")
    print(f"  Saved raw/ground_truth.npy: {hr_raw_recon.shape}")
    print(f"  Saved raw/input_lr.npy: {lr_raw_recon.shape}")
    
    # Save optional data
    if distances is not None:
        np.save(output_dir / 'distances.npy', distances)
        print(f"  Saved distances.npy: {distances.shape}")
    if locations is not None:
        np.save(output_dir / 'locations.npy', locations)
        print(f"  Saved locations.npy: {locations.shape}")
    
    # Save metrics
    all_results = {
        'model': args.model,
        'norm': {**convert_to_serializable(norm_summary), 'iou': convert_to_serializable(norm_iou)},
        'raw': {**convert_to_serializable(raw_summary), 'iou': convert_to_serializable(raw_iou)},
        'config': {
            'checkpoint': str(args.checkpoint),
            'input_npz': str(args.input_npz),
            'n_samples': len(predictions_norm),
            'device': str(device)
        }
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved evaluation_results.json")
    
    # ================================================================
    # Visualize
    # ================================================================
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    visualize_samples(lr_norm, hr_norm, predictions_norm, output_dir,
                     num_samples=args.num_vis_samples, prefix='norm_', model_name=model_name)
    visualize_samples(lr_raw_recon, hr_raw_recon, predictions_raw, output_dir,
                     num_samples=args.num_vis_samples, prefix='raw_', model_name=model_name)
    
    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("PREDICTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Model: {model_name}")
    print(f"  Samples: {len(predictions_norm)}")
    print(f"\n  Results saved to: {output_dir}/")
    print(f"    ├── norm/")
    print(f"    │   ├── predictions.npy")
    print(f"    │   ├── ground_truth.npy")
    print(f"    │   └── input_lr.npy")
    print(f"    ├── raw/")
    print(f"    │   ├── predictions.npy")
    print(f"    │   ├── ground_truth.npy")
    print(f"    │   └── input_lr.npy")
    print(f"    └── evaluation_results.json")
    print(f"{'='*70}\n")


# ================================================================
# Main Entry Point
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Hi-C Super-Resolution Prediction Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['srhic', 'deephic', 'hicnn'],
                        help='Model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input_npz', type=str, required=True,
                        help='Path to input NPZ file with train_lr/train_hr keys')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda, cuda:0, cpu)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for prediction')
    parser.add_argument('--num_vis_samples', type=int, default=4,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    predict(args)


if __name__ == '__main__':
    main()
