#!/usr/bin/env python
"""
SRHiC Training Script with Custom Data Preprocessing

This script combines:
1. YOUR data preprocessing method (log1p + robust normalization with median/IQR)
2. SRHiC deep learning model architecture for Hi-C super-resolution

SRHiC Model Architecture:
- Based on the method from: https://github.com/hzlzldr/SRHiC
- Uses deep residual learning with skip connections
- Designed to enhance low-resolution Hi-C data to high-resolution

Your Data Format:
- Input (LR): 40x40 patches from Hi-C contact matrices
- Output (HR): 40x40 high-resolution patches (same size, enhanced resolution)
- Data stored in NPZ format with shape (N, 40, 40, 1)

Preprocessing Pipeline (YOUR method):
1. Log1p transformation: log(1 + X)
2. Robust normalization using median and IQR (interquartile range)
3. Clipping to [-5, 5] range for training stability

Usage:
    python srhic_training_with_custom_preprocessing.py \
        --data_path /path/to/your_data.npz \
        --save_dir ./srhic_model \
        --epochs 100 \
        --batch_size 32 \
        --lr 1e-4
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# For metrics
from scipy.stats import pearsonr
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# SECTION 1: DATA PREPROCESSING (YOUR METHOD)
# ============================================================================

def ensure_nchw(x):
    """
    Ensure data is in NCHW format (batch, channel, height, width).
    
    Handles various input formats:
    - (N, H, W) -> (N, 1, H, W)
    - (N, C, H, W) -> unchanged if C is 1 or 3
    - (N, H, W, C) -> transpose to (N, C, H, W)
    """
    x = np.asarray(x)
    if x.ndim == 3:
        return x[:, None, :, :]
    elif x.ndim == 4 and x.shape[1] in [1, 3]:
        return x  # Already NCHW
    elif x.ndim == 4 and x.shape[-1] in [1, 3]:
        return np.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    else:
        raise ValueError(f"Cannot convert to NCHW: shape={x.shape}")


class RobustHiCPreprocessor:
    """
    YOUR Preprocessing Method for Hi-C Data
    
    Key features:
    - Log1p transformation for sparse Hi-C data
    - Robust normalization using median and IQR (not mean/std)
    - Clipping to [-5, 5] range for training stability
    
    This matches the preprocessing used in your diffusion model training.
    """
    
    def __init__(self, size=40):
        """
        Initialize preprocessor.
        
        Args:
            size: Expected matrix dimension (default: 40x40)
        """
        self.size = size
        self.X_mean = None  # Stores log-median for LR
        self.X_std = None   # Stores log-IQR for LR
        self.Y_mean = None  # Stores log-median for HR
        self.Y_std = None   # Stores log-IQR for HR
        self._is_fitted = False

    def fit(self, X_low, Y_high, verbose=True):
        """
        Fit normalization statistics on training data.
        
        Computes:
        - Log1p transform
        - Median (robust center)
        - IQR (robust scale) with small epsilon for numerical stability
        
        Args:
            X_low: Low-resolution training data (N, H, W) or (N, H, W, 1) or (N, 1, H, W)
            Y_high: High-resolution training data (same formats)
            verbose: Print statistics if True
        """
        X_low = ensure_nchw(X_low)
        Y_high = ensure_nchw(Y_high)
        
        # Log1p transformation
        X_log = np.log1p(X_low)
        Y_log = np.log1p(Y_high)
        
        # Robust statistics: median and IQR
        self.X_mean = np.median(X_log)
        self.X_std = np.percentile(X_log, 75) - np.percentile(X_log, 25) + 1e-8
        
        self.Y_mean = np.median(Y_log)
        self.Y_std = np.percentile(Y_log, 75) - np.percentile(Y_log, 25) + 1e-8
        
        self._is_fitted = True
        
        if verbose:
            print("\n" + "="*80)
            print("PREPROCESSING STATISTICS (Your Method: Log1p + Robust Normalization)")
            print("="*80)
            print(f"Low Resolution (LR) Data:")
            print(f"  Log-median: {self.X_mean:.6f}")
            print(f"  Log-IQR:    {self.X_std:.6f}")
            print(f"\nHigh Resolution (HR) Data:")
            print(f"  Log-median: {self.Y_mean:.6f}")
            print(f"  Log-IQR:    {self.Y_std:.6f}")
            print("="*80 + "\n")
    
    def preprocess(self, X, Y=None):
        """
        Apply preprocessing to data.
        
        Pipeline:
        1. Log1p transform: log(1 + X)
        2. Z-score with robust statistics: (X_log - median) / IQR
        3. Clip to [-5, 5]
        
        Args:
            X: Low-resolution data
            Y: High-resolution data (optional, for training)
        
        Returns:
            Xn: Normalized low-resolution data
            Yn: Normalized high-resolution data (or None if Y is None)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before preprocessing!")
        
        X = ensure_nchw(X)
        
        # Log1p transform
        X_log = np.log1p(X)
        
        # Z-score normalization with robust statistics
        Xn = (X_log - self.X_mean) / self.X_std
        
        # Clip outliers
        Xn = np.clip(Xn, -5, 5).astype(np.float32)
        
        if Y is None:
            return Xn, None
        
        Y = ensure_nchw(Y)
        
        # Same preprocessing for Y
        Y_log = np.log1p(Y)
        Yn = (Y_log - self.Y_mean) / self.Y_std
        Yn = np.clip(Yn, -5, 5).astype(np.float32)
        
        return Xn, Yn
    
    def preprocess_lr(self, X):
        """Preprocess low-resolution data only."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before preprocessing!")
        
        X = ensure_nchw(X)
        X_log = np.log1p(X)
        Xn = (X_log - self.X_mean) / self.X_std
        Xn = np.clip(Xn, -5, 5).astype(np.float32)
        return Xn

    def preprocess_hr(self, Y):
        """Preprocess high-resolution data only."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before preprocessing!")
        
        Y = ensure_nchw(Y)
        Y_log = np.log1p(Y)
        Yn = (Y_log - self.Y_mean) / self.Y_std
        Yn = np.clip(Yn, -5, 5).astype(np.float32)
        return Yn

    def postprocess(self, Y_norm):
        """
        Inverse preprocessing: convert normalized data back to original scale.
        
        Pipeline:
        1. Clip (safety)
        2. Reverse Z-score: Y_log = Y_norm * IQR + median
        3. Reverse log1p: Y = exp(Y_log) - 1
        4. Ensure non-negative
        
        Args:
            Y_norm: Normalized data from model output
        
        Returns:
            Y_counts: Data in original count space
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before postprocessing!")
        
        # Clip for safety
        Y_norm = np.clip(Y_norm, -5, 5)
        
        # Reverse normalization
        Y_log = Y_norm * self.Y_std + self.Y_mean
        
        # Reverse log1p
        Y_counts = np.expm1(Y_log)
        
        # Ensure non-negative (Hi-C counts are always non-negative)
        return np.maximum(Y_counts, 0.0)
    
    def postprocess_tensor(self, Y_norm):
        """
        Inverse preprocessing for PyTorch tensors.
        
        Args:
            Y_norm: Normalized tensor from model output
        
        Returns:
            Y_counts: Tensor in original count space
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before postprocessing!")
        
        # Clip for safety
        Y_norm = torch.clamp(Y_norm, -5, 5)
        
        # Reverse normalization
        Y_log = Y_norm * self.Y_std + self.Y_mean
        
        # Reverse log1p
        Y_counts = torch.expm1(Y_log)
        
        # Ensure non-negative
        return torch.clamp(Y_counts, min=0.0)

    def get_stats(self):
        """Return preprocessing statistics as dictionary."""
        return {
            'X_median': float(self.X_mean),
            'X_iqr': float(self.X_std),
            'Y_median': float(self.Y_mean),
            'Y_iqr': float(self.Y_std),
        }
    
    def save(self, path):
        """Save preprocessor to file."""
        state = {
            'size': self.size,
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'Y_mean': self.Y_mean,
            'Y_std': self.Y_std,
            '_is_fitted': self._is_fitted,
        }
        torch.save(state, path)
        print(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load preprocessor from file."""
        state = torch.load(path, map_location='cpu')
        preprocessor = cls(size=state.get('size', 40))
        preprocessor.X_mean = state['X_mean']
        preprocessor.X_std = state['X_std']
        preprocessor.Y_mean = state['Y_mean']
        preprocessor.Y_std = state['Y_std']
        preprocessor._is_fitted = state.get('_is_fitted', True)
        print(f"Preprocessor loaded from {path}")
        return preprocessor


# Alias for backward compatibility
HiCPreprocessor = RobustHiCPreprocessor


# ============================================================================
# SECTION 2: SRHiC MODEL ARCHITECTURE
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual Block for SRHiC.
    
    Architecture:
    - Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
    - Skip connection from input to output
    """
    
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual  # Skip connection
        out = self.relu(out)
        return out


class SRHiC(nn.Module):
    """
    SRHiC: Super-Resolution Hi-C Network
    
    Based on the architecture from https://github.com/hzlzldr/SRHiC
    
    This model uses deep residual learning to enhance low-resolution Hi-C 
    contact matrices to high-resolution.
    
    Architecture:
    1. Initial feature extraction
    2. Stack of residual blocks
    3. Global residual learning (LR + learned residual = HR)
    4. Output projection
    
    Key features:
    - Deep residual network with skip connections
    - BatchNorm for training stability
    - Global residual: output = input + network(input)
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_features=64,
        num_residual_blocks=16,
        kernel_size=3
    ):
        """
        Initialize SRHiC model.
        
        Args:
            in_channels: Number of input channels (1 for Hi-C)
            out_channels: Number of output channels (1 for Hi-C)
            num_features: Number of feature channels in residual blocks
            num_residual_blocks: Number of residual blocks (depth)
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        padding = kernel_size // 2
        
        # Initial feature extraction
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size, padding=padding),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        
        # Stack of residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features, kernel_size) 
              for _ in range(num_residual_blocks)]
        )
        
        # After residual blocks
        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size, padding=padding),
            nn.BatchNorm2d(num_features)
        )
        
        # Output projection
        self.conv_out = nn.Conv2d(num_features, out_channels, kernel_size, padding=padding)
        
        # Global residual connection weight (learnable)
        self.global_residual = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Low-resolution input (B, 1, H, W)
        
        Returns:
            High-resolution output (B, 1, H, W)
        """
        # Save input for global residual
        identity = x
        
        # Initial feature extraction
        out = self.conv_in(x)
        
        # Residual blocks
        residual = out
        out = self.residual_blocks(out)
        out = self.conv_mid(out)
        out = out + residual  # Local residual
        
        # Output projection
        out = self.conv_out(out)
        
        # Global residual: output = input + learned_residual
        out = identity + self.global_residual * out
        
        return out


class SRHiCLarge(nn.Module):
    """
    Larger variant of SRHiC with more capacity.
    
    Differences from base SRHiC:
    - More feature channels (128 vs 64)
    - More residual blocks (24 vs 16)
    - Additional feature refinement layers
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_features=128,
        num_residual_blocks=24,
        kernel_size=3
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        # Multi-scale initial feature extraction
        self.conv_in_3x3 = nn.Conv2d(in_channels, num_features // 2, 3, padding=1)
        self.conv_in_5x5 = nn.Conv2d(in_channels, num_features // 2, 5, padding=2)
        
        self.bn_in = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features, kernel_size) 
              for _ in range(num_residual_blocks)]
        )
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features)
        )
        
        # Output
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, out_channels, 3, padding=1)
        )
        
        self.global_residual = nn.Parameter(torch.ones(1))
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        identity = x
        
        # Multi-scale features
        feat_3x3 = self.conv_in_3x3(x)
        feat_5x5 = self.conv_in_5x5(x)
        out = torch.cat([feat_3x3, feat_5x5], dim=1)
        out = self.relu(self.bn_in(out))
        
        # Residual blocks
        residual = out
        out = self.residual_blocks(out)
        out = out + residual
        
        # Refinement
        out = self.refine(out) + out
        
        # Output
        out = self.conv_out(out)
        out = identity + self.global_residual * out
        
        return out


# ============================================================================
# SECTION 3: DATASET
# ============================================================================

class HiCDataset(Dataset):
    """
    PyTorch Dataset for Hi-C super-resolution.
    
    Expects preprocessed data (already normalized).
    """
    
    def __init__(self, X, Y, augment=False):
        """
        Args:
            X: Preprocessed low-resolution data (N, 1, H, W)
            Y: Preprocessed high-resolution data (N, 1, H, W)
            augment: Enable data augmentation
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [-1])
                y = torch.flip(y, [-1])
            
            # Random vertical flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [-2])
                y = torch.flip(y, [-2])
            
            # Random 90-degree rotation
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x = torch.rot90(x, k, [-2, -1])
                y = torch.rot90(y, k, [-2, -1])
        
        return x, y


# ============================================================================
# SECTION 4: LOSS FUNCTIONS
# ============================================================================

class HiCLoss(nn.Module):
    """
    Combined loss function for Hi-C super-resolution.
    
    Components:
    1. MSE Loss: Pixel-wise reconstruction
    2. L1 Loss: Sparse-aware reconstruction
    3. SSIM Loss: Structural similarity (optional)
    """
    
    def __init__(self, mse_weight=1.0, l1_weight=0.1, ssim_weight=0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        loss = 0.0
        
        if self.mse_weight > 0:
            loss += self.mse_weight * self.mse(pred, target)
        
        if self.l1_weight > 0:
            loss += self.l1_weight * self.l1(pred, target)
        
        if self.ssim_weight > 0:
            # Simple differentiable SSIM approximation
            ssim_loss = 1.0 - self._ssim_approx(pred, target)
            loss += self.ssim_weight * ssim_loss
        
        return loss
    
    def _ssim_approx(self, pred, target, window_size=11):
        """Differentiable SSIM approximation."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return ssim_map.mean()


# ============================================================================
# SECTION 5: TRAINING UTILITIES
# ============================================================================

def compute_metrics(pred, target):
    """
    Compute evaluation metrics.
    
    Args:
        pred: Predicted HR (numpy array)
        target: Ground truth HR (numpy array)
    
    Returns:
        dict with MSE, PSNR, PCC, SSIM
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # MSE
    mse = np.mean((pred - target) ** 2)
    
    # PSNR
    if mse > 0:
        max_val = max(target.max(), 1e-8)
        psnr = 10 * np.log10(max_val ** 2 / mse)
    else:
        psnr = float('inf')
    
    # Pearson Correlation
    if np.std(pred) > 1e-8 and np.std(target) > 1e-8:
        pcc, _ = pearsonr(pred, target)
    else:
        pcc = 0.0
    
    # SSIM (computed on 2D matrices)
    if pred.ndim == 1:
        size = int(np.sqrt(len(pred)))
        pred_2d = pred.reshape(size, size)
        target_2d = target.reshape(size, size)
    else:
        pred_2d = pred
        target_2d = target
    
    try:
        ssim_val = ssim(target_2d, pred_2d, data_range=target_2d.max() - target_2d.min())
    except:
        ssim_val = 0.0
    
    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'pcc': float(pcc),
        'ssim': float(ssim_val)
    }


def validate(model, val_loader, criterion, device, preprocessor=None):
    """
    Validate model on validation set.
    
    Args:
        model: SRHiC model
        val_loader: Validation DataLoader
        criterion: Loss function
        device: torch device
        preprocessor: For converting back to original space (optional)
    
    Returns:
        dict with val_loss, pcc, ssim, mse
    """
    model.eval()
    total_loss = 0.0
    all_pcc = []
    all_ssim = []
    all_mse = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            
            # Compute metrics in normalized space
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            
            for i in range(len(pred_np)):
                metrics = compute_metrics(pred_np[i, 0], y_np[i, 0])
                all_pcc.append(metrics['pcc'])
                all_ssim.append(metrics['ssim'])
                all_mse.append(metrics['mse'])
    
    avg_loss = total_loss / len(val_loader)
    avg_pcc = np.mean(all_pcc)
    avg_ssim = np.mean(all_ssim)
    avg_mse = np.mean(all_mse)
    
    return {
        'val_loss': avg_loss,
        'pcc': avg_pcc,
        'ssim': avg_ssim,
        'mse': avg_mse
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path, preprocessor=None):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    if preprocessor:
        checkpoint['preprocessor_stats'] = preprocessor.get_stats()
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('metrics', {})


# ============================================================================
# SECTION 6: VISUALIZATION
# ============================================================================

def plot_training_curves(history, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True)
    
    # PCC
    ax = axes[0, 1]
    ax.plot(history['val_pcc'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PCC')
    ax.set_title('Validation Pearson Correlation')
    ax.grid(True)
    
    # SSIM
    ax = axes[1, 0]
    ax.plot(history['val_ssim'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM')
    ax.set_title('Validation SSIM')
    ax.grid(True)
    
    # MSE
    ax = axes[1, 1]
    ax.plot(history['val_mse'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Validation MSE')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def visualize_samples(model, val_loader, preprocessor, device, save_dir, num_samples=4):
    """Visualize model predictions vs ground truth."""
    model.eval()
    
    # Get a batch
    x_batch, y_batch = next(iter(val_loader))
    x_batch = x_batch[:num_samples].to(device)
    y_batch = y_batch[:num_samples].to(device)
    
    with torch.no_grad():
        pred_batch = model(x_batch)
    
    # Convert to numpy
    x_np = x_batch.cpu().numpy()
    y_np = y_batch.cpu().numpy()
    pred_np = pred_batch.cpu().numpy()
    
    # Post-process to original space
    if preprocessor:
        x_orig = preprocessor.postprocess(x_np)
        y_orig = preprocessor.postprocess(y_np)
        pred_orig = preprocessor.postprocess(pred_np)
    else:
        x_orig, y_orig, pred_orig = x_np, y_np, pred_np
    
    # Plot
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    for i in range(num_samples):
        vmax = max(y_orig[i, 0].max(), pred_orig[i, 0].max())
        
        # Low-res input
        ax = axes[i, 0] if num_samples > 1 else axes[0]
        im = ax.imshow(x_orig[i, 0], cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title(f'Sample {i+1}: LR Input')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # High-res ground truth
        ax = axes[i, 1] if num_samples > 1 else axes[1]
        im = ax.imshow(y_orig[i, 0], cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title('HR Ground Truth')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Prediction
        ax = axes[i, 2] if num_samples > 1 else axes[2]
        im = ax.imshow(pred_orig[i, 0], cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title('SRHiC Prediction')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Difference
        ax = axes[i, 3] if num_samples > 1 else axes[3]
        diff = pred_orig[i, 0] - y_orig[i, 0]
        vmax_diff = max(abs(diff.min()), abs(diff.max()))
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
        ax.set_title('Difference (Pred - GT)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=150)
    plt.close()
    print(f"Sample predictions saved to {save_dir}/sample_predictions.png")


# ============================================================================
# SECTION 7: MAIN TRAINING FUNCTION
# ============================================================================

def train(args):
    """Main training function."""
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device - use specified GPU
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Set CUDA device for operations
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    print(f"\n{'='*80}")
    print(f"SRHiC Training with Custom Preprocessing")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
    print(f"Save directory: {args.save_dir}")
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print(f"\nLoading data from {args.data_path}...")
    data = np.load(args.data_path)
    
    # Support different key names
    if 'X' in data and 'Y' in data:
        X_raw = data['X']
        Y_raw = data['Y']
    elif 'lr' in data and 'hr' in data:
        X_raw = data['lr']
        Y_raw = data['hr']
    elif 'low' in data and 'high' in data:
        X_raw = data['low']
        Y_raw = data['high']
    else:
        # Try to find arrays
        keys = list(data.keys())
        print(f"Available keys: {keys}")
        if len(keys) >= 2:
            X_raw = data[keys[0]]
            Y_raw = data[keys[1]]
        else:
            raise ValueError(f"Cannot find LR and HR data. Available keys: {keys}")
    
    print(f"Raw data shapes: X={X_raw.shape}, Y={Y_raw.shape}")
    
    # -------------------------------------------------------------------------
    # Preprocessing (YOUR METHOD)
    # -------------------------------------------------------------------------
    print("\nApplying YOUR preprocessing method...")
    preprocessor = RobustHiCPreprocessor(size=40)
    preprocessor.fit(X_raw, Y_raw, verbose=True)
    
    X_norm, Y_norm = preprocessor.preprocess(X_raw, Y_raw)
    print(f"Normalized data shapes: X={X_norm.shape}, Y={Y_norm.shape}")
    print(f"X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print(f"Y range: [{Y_norm.min():.3f}, {Y_norm.max():.3f}]")
    
    # Save preprocessor
    preprocessor.save(os.path.join(args.save_dir, 'preprocessor.pt'))
    
    # -------------------------------------------------------------------------
    # Train/Val Split
    # -------------------------------------------------------------------------
    n_samples = len(X_norm)
    n_val = int(n_samples * args.val_ratio)
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, Y_train = X_norm[train_idx], Y_norm[train_idx]
    X_val, Y_val = X_norm[val_idx], Y_norm[val_idx]
    
    print(f"\nTrain samples: {n_train}")
    print(f"Validation samples: {n_val}")
    
    # Create datasets
    train_dataset = HiCDataset(X_train, Y_train, augment=args.augment)
    val_dataset = HiCDataset(X_val, Y_val, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    print(f"\nCreating SRHiC model...")
    if args.model_size == 'large':
        model = SRHiCLarge(
            in_channels=1,
            out_channels=1,
            num_features=args.num_features,
            num_residual_blocks=args.num_blocks
        )
    else:
        model = SRHiC(
            in_channels=1,
            out_channels=1,
            num_features=args.num_features,
            num_residual_blocks=args.num_blocks
        )
    
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # -------------------------------------------------------------------------
    # Optimizer, Scheduler, Loss
    # -------------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None
    
    criterion = HiCLoss(mse_weight=1.0, l1_weight=args.l1_weight, ssim_weight=args.ssim_weight)
    
    # -------------------------------------------------------------------------
    # Resume from checkpoint
    # -------------------------------------------------------------------------
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}...")
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch}")
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_pcc': [],
        'val_ssim': [],
        'val_mse': [],
    }
    
    best_pcc = -float('inf')
    best_loss = float('inf')
    
    print(f"\n{'='*80}")
    print(f"Starting training...")
    print(f"{'='*80}\n")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_metrics = validate(model, val_loader, criterion, device, preprocessor)
        
        # Update scheduler
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()
        
        # Log
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_pcc'].append(val_metrics['pcc'])
        history['val_ssim'].append(val_metrics['ssim'])
        history['val_mse'].append(val_metrics['mse'])
        
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={val_metrics['val_loss']:.4f}, "
              f"PCC={val_metrics['pcc']:.4f}, "
              f"SSIM={val_metrics['ssim']:.4f}")
        
        # Save best model (by PCC)
        if val_metrics['pcc'] > best_pcc:
            best_pcc = val_metrics['pcc']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                os.path.join(args.save_dir, 'best_pcc.pt'),
                preprocessor
            )
            print(f"  -> New best PCC: {best_pcc:.4f}")
        
        # Save best model (by loss)
        if val_metrics['val_loss'] < best_loss:
            best_loss = val_metrics['val_loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                os.path.join(args.save_dir, 'best_loss.pt'),
                preprocessor
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt'),
                preprocessor
            )
    
    # -------------------------------------------------------------------------
    # Final Saving
    # -------------------------------------------------------------------------
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, val_metrics,
        os.path.join(args.save_dir, 'final.pt'),
        preprocessor
    )
    
    # Save training history
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, args.save_dir)
    
    # Visualize samples
    visualize_samples(model, val_loader, preprocessor, device, args.save_dir)
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Best PCC: {best_pcc:.4f}")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Models saved to: {args.save_dir}")
    print(f"{'='*80}\n")


# ============================================================================
# SECTION 8: MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SRHiC Training with Custom Data Preprocessing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to NPZ file with LR and HR data')
    parser.add_argument('--save_dir', type=str, default='./srhic_model',
                        help='Directory to save model and outputs')
    parser.add_argument('--device', type=str, default='cuda:3',
                        help='Device to use (e.g., cuda:0, cuda:3, cpu)')
    
    # Model
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['base', 'large'],
                        help='Model size variant')
    parser.add_argument('--num_features', type=int, default=64,
                        help='Number of feature channels')
    parser.add_argument('--num_blocks', type=int, default=16,
                        help='Number of residual blocks')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for AdamW')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping (0 to disable)')
    
    # Loss
    parser.add_argument('--l1_weight', type=float, default=0.1,
                        help='Weight for L1 loss')
    parser.add_argument('--ssim_weight', type=float, default=0.0,
                        help='Weight for SSIM loss')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler')
    
    # Data
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*80)
    print("Configuration:")
    print("="*80)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*80 + "\n")
    
    train(args)


if __name__ == '__main__':
    main()
