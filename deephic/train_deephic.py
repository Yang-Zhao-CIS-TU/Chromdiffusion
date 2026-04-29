#!/usr/bin/env python3
"""
DeepHiC-PyTorch: GAN-based Hi-C Enhancement

This script trains DeepHiC model with HiCARN-style data preprocessing
(log1p + z-score normalization).

DeepHiC Architecture:
  - Generator: Residual blocks with skip connections (similar to SRGAN)
  - Discriminator: VGG-style CNN with batch normalization
  - Losses: MSE (pixel) + Adversarial + Perceptual (TV)

Key features:
  - Uses HiCARN-style normalization (log1p + z-score)
  - 40x40 input/output tiles
  - No upscaling (same resolution enhancement)

Usage:
    python train_deephic.py \
        --train_npz /path/to/train_data_raw_ratio16.npz \
        --output_dir checkpoints_deephic \
        --epochs 300 \
        --batch_size 64 \
        --device cuda

Reference:
    DeepHiC: A generative adversarial network for enhancing Hi-C data resolution
    Hong et al., PLOS Computational Biology, 2020
    https://github.com/omegahh/DeepHiC
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
import math


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
        
        print(f"  Preprocessor fitted:")
        print(f"    X (LR): mean={self.X_mean:.4f}, std={self.X_std:.4f}")
        print(f"    Y (HR): mean={self.Y_mean:.4f}, std={self.Y_std:.4f}")
    
    def transform_X(self, X):
        X_log = np.log1p(X.astype(np.float64))
        X_norm = (X_log - self.X_mean) / self.X_std
        return np.clip(X_norm, -5, 5).astype(np.float32)
    
    def transform_Y(self, Y):
        Y_log = np.log1p(Y.astype(np.float64))
        Y_norm = (Y_log - self.Y_mean) / self.Y_std
        return np.clip(Y_norm, -5, 5).astype(np.float32)
    
    def inverse_transform_Y(self, Y_norm):
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
        print(f"  Preprocessor saved to: {path}")
    
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


# ================================================================
# DeepHiC Generator
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
    
    Architecture:
      - Initial conv: 1 -> 64 channels
      - N residual blocks (default 15)
      - Post-residual conv
      - Final conv: 64 -> 1 channel
    
    No upscaling since input/output are same resolution (40x40).
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
# DeepHiC Discriminator
# ================================================================

class Discriminator(nn.Module):
    """
    DeepHiC Discriminator Network (VGG-style)
    
    Architecture:
      - 6 conv blocks with increasing channels (64 -> 256)
      - Batch normalization and LeakyReLU
      - Fully connected layers for classification
    """
    def __init__(self, in_channels=1, input_size=40, num_features=64):
        super().__init__()
        
        def conv_block(in_ch, out_ch, stride=1, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.features = nn.Sequential(
            # Block 1: 40x40 -> 40x40
            conv_block(in_channels, num_features, stride=1, bn=False),
            # Block 2: 40x40 -> 20x20
            conv_block(num_features, num_features, stride=2, bn=True),
            # Block 3: 20x20 -> 20x20
            conv_block(num_features, num_features * 2, stride=1, bn=True),
            # Block 4: 20x20 -> 10x10
            conv_block(num_features * 2, num_features * 2, stride=2, bn=True),
            # Block 5: 10x10 -> 10x10
            conv_block(num_features * 2, num_features * 4, stride=1, bn=True),
            # Block 6: 10x10 -> 5x5
            conv_block(num_features * 4, num_features * 4, stride=2, bn=True),
        )
        
        # Calculate feature map size: 40 -> 20 -> 10 -> 5
        feat_size = input_size // 8  # 40 // 8 = 5
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features * 4 * feat_size * feat_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )
    
    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out


# ================================================================
# Losses
# ================================================================

class TVLoss(nn.Module):
    """Total Variation Loss for smoothness"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, x):
        batch_size = x.size(0)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return self.weight * (h_tv + w_tv) / batch_size


class GANLoss(nn.Module):
    """GAN loss (vanilla or LSGAN)"""
    def __init__(self, loss_type='vanilla'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.loss = nn.MSELoss()
    
    def forward(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


# ================================================================
# SSIM Loss
# ================================================================

class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.window = self._create_window(window_size, sigma)
    
    def _create_window(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)
    
    def forward(self, pred, target):
        window = self.window.to(pred.device)
        
        mu1 = F.conv2d(pred, window, padding=self.window_size//2)
        mu2 = F.conv2d(target, window, padding=self.window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim.mean()


# ================================================================
# Dataset
# ================================================================

class HiCDataset(Dataset):
    def __init__(self, lr_data, hr_data):
        self.lr = torch.from_numpy(lr_data).float()
        self.hr = torch.from_numpy(hr_data).float()
    
    def __len__(self):
        return len(self.lr)
    
    def __getitem__(self, idx):
        return self.lr[idx], self.hr[idx]


# ================================================================
# Training Functions
# ================================================================

def train_epoch(
    generator, discriminator,
    g_optimizer, d_optimizer,
    dataloader, device,
    gan_loss_fn, tv_loss_fn, ssim_loss_fn,
    lambda_mse=1.0,
    lambda_adv=0.001,
    lambda_tv=1e-6,
    lambda_ssim=0.1,
    epoch=0
):
    """Train one epoch"""
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    total_mse = 0
    total_adv = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (lr, hr) in enumerate(pbar):
        lr = lr.to(device)
        hr = hr.to(device)
        
        # ============================================
        # Train Discriminator
        # ============================================
        d_optimizer.zero_grad()
        
        fake_hr = generator(lr)
        
        real_pred = discriminator(hr)
        fake_pred = discriminator(fake_hr.detach())
        
        d_loss_real = gan_loss_fn(real_pred, True)
        d_loss_fake = gan_loss_fn(fake_pred, False)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        d_loss.backward()
        d_optimizer.step()
        
        # ============================================
        # Train Generator
        # ============================================
        g_optimizer.zero_grad()
        
        fake_hr = generator(lr)
        
        # MSE loss
        mse_loss = F.mse_loss(fake_hr, hr)
        
        # Adversarial loss
        fake_pred = discriminator(fake_hr)
        adv_loss = gan_loss_fn(fake_pred, True)
        
        # TV loss
        tv_loss = tv_loss_fn(fake_hr)
        
        # SSIM loss
        ssim_loss = ssim_loss_fn(fake_hr, hr)
        
        # Total generator loss
        g_loss = lambda_mse * mse_loss + lambda_adv * adv_loss + lambda_tv * tv_loss + lambda_ssim * ssim_loss
        
        g_loss.backward()
        g_optimizer.step()
        
        # Accumulate
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        total_mse += mse_loss.item()
        total_adv += adv_loss.item()
        
        pbar.set_postfix({
            'G': f'{g_loss.item():.4f}',
            'D': f'{d_loss.item():.4f}',
            'MSE': f'{mse_loss.item():.4f}'
        })
    
    n = len(dataloader)
    return {
        'g_loss': total_g_loss / n,
        'd_loss': total_d_loss / n,
        'mse_loss': total_mse / n,
        'adv_loss': total_adv / n,
    }


@torch.no_grad()
def validate(generator, lr_val, hr_val, device, n_samples=500):
    """Validate generator"""
    generator.eval()
    
    n = min(n_samples, len(lr_val))
    lr = torch.from_numpy(lr_val[:n]).float().to(device)
    hr = hr_val[:n]
    
    # Generate in batches
    batch_size = 64
    outputs = []
    for i in range(0, n, batch_size):
        batch = lr[i:i+batch_size]
        out = generator(batch)
        outputs.append(out.cpu().numpy())
    
    output = np.concatenate(outputs, axis=0)
    
    # MSE
    mse = np.mean((output - hr) ** 2)
    
    # PCC
    pcc_list = []
    for i in range(len(output)):
        pred_flat = output[i].flatten()
        gt_flat = hr[i].flatten()
        if np.std(pred_flat) > 0 and np.std(gt_flat) > 0:
            pcc, _ = pearsonr(pred_flat, gt_flat)
            pcc_list.append(pcc)
    pcc = np.mean(pcc_list) if pcc_list else 0.0
    
    # PSNR
    if mse > 0:
        max_val = max(hr.max(), output.max())
        psnr = 10 * np.log10((max_val ** 2) / mse) if max_val > 0 else 0
    else:
        psnr = float('inf')
    
    return {'mse': float(mse), 'pcc': float(pcc), 'psnr': float(psnr)}


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


def load_data(args):
    """Load training data and apply normalization"""
    preprocessor = None
    
    if args.train_npz:
        print(f"Loading NPZ from: {args.train_npz}")
        data = np.load(args.train_npz, allow_pickle=True)
        print(f"  Keys: {data.files}")
        
        if 'train_lr' in data.files and 'train_hr' in data.files:
            lr_raw = data['train_lr']
            hr_raw = data['train_hr']
        else:
            raise ValueError(f"NPZ must contain 'train_lr' and 'train_hr'. Found: {data.files}")
        
        lr_raw = ensure_nchw(lr_raw).astype(np.float32)
        hr_raw = ensure_nchw(hr_raw).astype(np.float32)
        
        print(f"  LR (raw): {lr_raw.shape}, range [{lr_raw.min():.2f}, {lr_raw.max():.2f}]")
        print(f"  HR (raw): {hr_raw.shape}, range [{hr_raw.min():.2f}, {hr_raw.max():.2f}]")
        
        if lr_raw.max() > 10 or hr_raw.max() > 10:
            print("\n  Detected RAW data, applying normalization...")
            preprocessor = HiCPreprocessor()
            preprocessor.fit(lr_raw, hr_raw)
            lr = preprocessor.transform_X(lr_raw)
            hr = preprocessor.transform_Y(hr_raw)
            print(f"  LR (norm): {lr.shape}, range [{lr.min():.4f}, {lr.max():.4f}]")
            print(f"  HR (norm): {hr.shape}, range [{hr.min():.4f}, {hr.max():.4f}]")
        else:
            print("  Data appears already normalized")
            lr = lr_raw
            hr = hr_raw
    
    elif args.train_lr and args.train_hr:
        print(f"Loading LR from: {args.train_lr}")
        lr = np.load(args.train_lr)
        print(f"Loading HR from: {args.train_hr}")
        hr = np.load(args.train_hr)
        
        lr = ensure_nchw(lr).astype(np.float32)
        hr = ensure_nchw(hr).astype(np.float32)
        
        if lr.max() > 10 or hr.max() > 10:
            print("\n  Detected RAW data, applying normalization...")
            preprocessor = HiCPreprocessor()
            preprocessor.fit(lr, hr)
            lr = preprocessor.transform_X(lr)
            hr = preprocessor.transform_Y(hr)
    else:
        raise ValueError("Must provide either --train_npz or both --train_lr and --train_hr")
    
    return lr, hr, preprocessor


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Train DeepHiC with HiCARN-style preprocessing')
    
    # Data
    parser.add_argument('--train_npz', type=str, default=None)
    parser.add_argument('--train_lr', type=str, default=None)
    parser.add_argument('--train_hr', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints_deephic')
    
    # Model
    parser.add_argument('--num_res_blocks', type=int, default=15,
                       help='Number of residual blocks in generator (default: 15)')
    parser.add_argument('--num_features', type=int, default=64,
                       help='Number of features (default: 64)')
    parser.add_argument('--input_size', type=int, default=40)
    
    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Loss weights
    parser.add_argument('--lambda_mse', type=float, default=1.0)
    parser.add_argument('--lambda_adv', type=float, default=0.001)
    parser.add_argument('--lambda_tv', type=float, default=1e-6)
    parser.add_argument('--lambda_ssim', type=float, default=0.1)
    parser.add_argument('--gan_type', type=str, default='vanilla',
                       choices=['vanilla', 'lsgan'])
    
    # Validation
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # Resume
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.train_npz is None and (args.train_lr is None or args.train_hr is None):
        raise ValueError("Must provide either --train_npz or both --train_lr and --train_hr")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Load data
    # ================================================================
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    lr_data, hr_data, preprocessor = load_data(args)
    
    # Split
    n_samples = len(lr_data)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    lr_train, hr_train = lr_data[train_idx], hr_data[train_idx]
    lr_val, hr_val = lr_data[val_idx], hr_data[val_idx]
    
    print(f"  Train: {len(lr_train)}, Val: {len(lr_val)}")
    
    if preprocessor is not None:
        preprocessor.save(output_dir / 'preprocessor.pt')
    
    train_dataset = HiCDataset(lr_train, hr_train)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    # ================================================================
    # Create models
    # ================================================================
    print("\n" + "="*70)
    print("CREATING MODELS")
    print("="*70)
    
    generator = Generator(
        in_channels=1, out_channels=1,
        num_features=args.num_features,
        num_res_blocks=args.num_res_blocks
    ).to(device)
    
    discriminator = Discriminator(
        in_channels=1,
        input_size=args.input_size,
        num_features=args.num_features
    ).to(device)
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  Generator: {g_params / 1e6:.2f}M parameters")
    print(f"  Discriminator: {d_params / 1e6:.2f}M parameters")
    
    # ================================================================
    # Optimizers and losses
    # ================================================================
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.999))
    
    gan_loss_fn = GANLoss(args.gan_type).to(device)
    tv_loss_fn = TVLoss()
    ssim_loss_fn = SSIMLoss()
    
    # ================================================================
    # Resume
    # ================================================================
    start_epoch = 1
    best_pcc = 0.0
    history = []
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ckpt['generator_state_dict'])
        discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        if 'g_optimizer_state_dict' in ckpt:
            g_optimizer.load_state_dict(ckpt['g_optimizer_state_dict'])
        if 'd_optimizer_state_dict' in ckpt:
            d_optimizer.load_state_dict(ckpt['d_optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_pcc = ckpt.get('best_pcc', 0.0)
    
    # ================================================================
    # Training
    # ================================================================
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    print(f"\nConfig:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Generator LR: {args.lr_g}")
    print(f"  Discriminator LR: {args.lr_d}")
    print(f"  Lambda MSE: {args.lambda_mse}")
    print(f"  Lambda Adv: {args.lambda_adv}")
    print(f"  Lambda TV: {args.lambda_tv}")
    print(f"  Lambda SSIM: {args.lambda_ssim}")
    print(f"  Residual blocks: {args.num_res_blocks}")
    
    # Baseline
    baseline = validate(generator, lr_val, hr_val, device)
    print(f"\nBaseline (untrained): MSE={baseline['mse']:.6f}, PCC={baseline['pcc']:.4f}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        losses = train_epoch(
            generator, discriminator,
            g_optimizer, d_optimizer,
            train_loader, device,
            gan_loss_fn, tv_loss_fn, ssim_loss_fn,
            lambda_mse=args.lambda_mse,
            lambda_adv=args.lambda_adv,
            lambda_tv=args.lambda_tv,
            lambda_ssim=args.lambda_ssim,
            epoch=epoch
        )
        
        # Validate
        val_metrics = validate(generator, lr_val, hr_val, device)
        
        # Log
        print(f"  Train: G={losses['g_loss']:.4f}, D={losses['d_loss']:.4f}, MSE={losses['mse_loss']:.4f}")
        print(f"  Val:   MSE={val_metrics['mse']:.6f}, PCC={val_metrics['pcc']:.4f}, PSNR={val_metrics['psnr']:.2f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            **losses,
            **val_metrics
        })
        
        # Save best model
        if val_metrics['pcc'] > best_pcc:
            best_pcc = val_metrics['pcc']
            print(f"  >>> New best PCC: {best_pcc:.4f}")
            save_dict = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'best_pcc': best_pcc,
                'val_metrics': val_metrics,
                'config': vars(args)
            }
            if preprocessor is not None:
                save_dict['preprocessor'] = {
                    'X_mean': preprocessor.X_mean, 'X_std': preprocessor.X_std,
                    'Y_mean': preprocessor.Y_mean, 'Y_std': preprocessor.Y_std
                }
            torch.save(save_dict, output_dir / 'best_model.pt')
        
        # Periodic checkpoint
        if epoch % 20 == 0:
            save_dict = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'best_pcc': best_pcc,
                'config': vars(args)
            }
            if preprocessor is not None:
                save_dict['preprocessor'] = {
                    'X_mean': preprocessor.X_mean, 'X_std': preprocessor.X_std,
                    'Y_mean': preprocessor.Y_mean, 'Y_std': preprocessor.Y_std
                }
            torch.save(save_dict, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save final
    save_dict = {
        'epoch': args.epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'best_pcc': best_pcc,
        'config': vars(args)
    }
    if preprocessor is not None:
        save_dict['preprocessor'] = {
            'X_mean': preprocessor.X_mean, 'X_std': preprocessor.X_std,
            'Y_mean': preprocessor.Y_mean, 'Y_std': preprocessor.Y_std
        }
    torch.save(save_dict, output_dir / 'final_model.pt')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  Best PCC: {best_pcc:.4f}")
    print(f"  Models saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
