#!/usr/bin/env python3
"""
DFHiC-PyTorch: Hi-C Enhancement using SRGAN Architecture

This is a PyTorch reimplementation of DFHiC (originally TensorFlow 1.x + TensorLayer)
with HiCARN-style data preprocessing (log1p + z-score normalization).

DFHiC Architecture:
  - Generator: SRGAN-style with residual blocks
  - Discriminator: VGG-style deep CNN
  - Losses: L1 + Adversarial + (optional) Perceptual

Data preprocessing (same as HiCARN):
  - log1p transformation
  - z-score normalization using median + IQR

Usage:
    python train_dfhic.py \
        --train_npz /path/to/train_data_raw_ratio16.npz \
        --output_dir checkpoints_dfhic \
        --epochs 200 \
        --batch_size 64 \
        --device cuda

Reference:
    DFHiC: https://github.com/BinWangCSU/DFHiC
    SRGAN: Photo-Realistic Single Image Super-Resolution Using a GAN (CVPR 2017)
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
    
    Preprocessing steps:
        1. log1p: X_log = log(1 + X)
        2. z-score: X_norm = (X_log - mean) / std
    
    Uses median + IQR for robust statistics.
    """
    def __init__(self):
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self.fitted = False
    
    def fit(self, X, Y):
        """
        Fit preprocessor on training data.
        
        Args:
            X: Low-resolution data (raw counts)
            Y: High-resolution data (raw counts)
        """
        # Log transform
        X_log = np.log1p(X.astype(np.float64))
        Y_log = np.log1p(Y.astype(np.float64))
        
        # Compute robust statistics using median + IQR
        # For X (LR)
        X_flat = X_log.flatten()
        X_nonzero = X_flat[X_flat > 0]
        if len(X_nonzero) > 0:
            self.X_mean = float(np.median(X_nonzero))
            q75, q25 = np.percentile(X_nonzero, [75, 25])
            self.X_std = float((q75 - q25) / 1.35)  # IQR to std approximation
        else:
            self.X_mean = 0.0
            self.X_std = 1.0
        
        # For Y (HR)
        Y_flat = Y_log.flatten()
        Y_nonzero = Y_flat[Y_flat > 0]
        if len(Y_nonzero) > 0:
            self.Y_mean = float(np.median(Y_nonzero))
            q75, q25 = np.percentile(Y_nonzero, [75, 25])
            self.Y_std = float((q75 - q25) / 1.35)
        else:
            self.Y_mean = 0.0
            self.Y_std = 1.0
        
        # Ensure std is not too small
        self.X_std = max(self.X_std, 0.1)
        self.Y_std = max(self.Y_std, 0.1)
        
        self.fitted = True
        
        print(f"  Preprocessor fitted:")
        print(f"    X (LR): mean={self.X_mean:.4f}, std={self.X_std:.4f}")
        print(f"    Y (HR): mean={self.Y_mean:.4f}, std={self.Y_std:.4f}")
    
    def transform_X(self, X):
        """Transform low-resolution data"""
        X_log = np.log1p(X.astype(np.float64))
        X_norm = (X_log - self.X_mean) / self.X_std
        return np.clip(X_norm, -5, 5).astype(np.float32)
    
    def transform_Y(self, Y):
        """Transform high-resolution data"""
        Y_log = np.log1p(Y.astype(np.float64))
        Y_norm = (Y_log - self.Y_mean) / self.Y_std
        return np.clip(Y_norm, -5, 5).astype(np.float32)
    
    def inverse_transform_Y(self, Y_norm):
        """Inverse transform predictions to raw counts"""
        Y_norm = np.clip(Y_norm, -5, 5)
        Y_log = Y_norm * self.Y_std + self.Y_mean
        Y_raw = np.expm1(Y_log)
        return np.maximum(Y_raw, 0.0)
    
    def save(self, path):
        """Save preprocessor to file"""
        torch.save({
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'Y_mean': self.Y_mean,
            'Y_std': self.Y_std,
            'fitted': self.fitted
        }, path)
        print(f"  Preprocessor saved to: {path}")
    
    @classmethod
    def load(cls, path):
        """Load preprocessor from file"""
        data = torch.load(path, map_location='cpu')
        preprocessor = cls()
        preprocessor.X_mean = data['X_mean']
        preprocessor.X_std = data['X_std']
        preprocessor.Y_mean = data['Y_mean']
        preprocessor.Y_std = data['Y_std']
        preprocessor.fitted = data.get('fitted', True)
        return preprocessor


# ================================================================
# Generator Network (SRGAN-style)
# ================================================================

class ResidualBlock(nn.Module):
    """Residual block with two conv layers + batch norm + PReLU"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class Generator(nn.Module):
    """
    SRGAN-style Generator for Hi-C enhancement.
    
    Architecture:
      - Initial conv + PReLU
      - N residual blocks (default 16)
      - Post-residual conv + BN
      - Skip connection from initial conv
      - Final conv to output channel
    
    Note: Unlike image SR, we don't do upsampling since input/output
    are same resolution (40x40 -> 40x40).
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_res_blocks=16):
        super().__init__()
        
        # Initial convolution
        self.conv_first = nn.Conv2d(in_channels, num_features, 9, padding=4)
        self.prelu_first = nn.PReLU()
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(num_features))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Post-residual convolution
        self.conv_mid = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_features)
        
        # Final output convolution
        self.conv_last = nn.Conv2d(num_features, out_channels, 9, padding=4)
    
    def forward(self, x):
        # Initial feature extraction
        first = self.prelu_first(self.conv_first(x))
        
        # Residual learning
        res = self.res_blocks(first)
        res = self.bn_mid(self.conv_mid(res))
        
        # Skip connection
        out = first + res
        
        # Final output
        out = self.conv_last(out)
        
        return out


# ================================================================
# Discriminator Network (VGG-style)
# ================================================================

class Discriminator(nn.Module):
    """
    VGG-style Discriminator for SRGAN.
    
    Architecture:
      - 8 conv layers with increasing channels (64 -> 512)
      - LeakyReLU activation
      - Batch normalization after first conv
      - Dense layers for classification
    """
    def __init__(self, in_channels=1, input_size=40):
        super().__init__()
        
        def conv_block(in_ch, out_ch, stride=1, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.features = nn.Sequential(
            # 40x40 -> 40x40
            conv_block(in_channels, 64, stride=1, bn=False),
            # 40x40 -> 20x20
            conv_block(64, 64, stride=2, bn=True),
            # 20x20 -> 20x20
            conv_block(64, 128, stride=1, bn=True),
            # 20x20 -> 10x10
            conv_block(128, 128, stride=2, bn=True),
            # 10x10 -> 10x10
            conv_block(128, 256, stride=1, bn=True),
            # 10x10 -> 5x5
            conv_block(256, 256, stride=2, bn=True),
            # 5x5 -> 5x5
            conv_block(256, 512, stride=1, bn=True),
            # 5x5 -> 3x3 (for 40x40 input: 40->20->10->5->3)
            conv_block(512, 512, stride=2, bn=True),
        )
        
        # Calculate feature map size after convolutions
        # 40 -> 20 -> 10 -> 5 -> 3 (with stride=2 and padding=1)
        # Formula: floor((size + 2*padding - kernel) / stride) + 1
        # For stride=2, padding=1, kernel=3: floor((n + 2 - 3) / 2) + 1 = floor((n-1)/2) + 1
        # 40 -> 20 -> 10 -> 5 -> 3
        feat_size = input_size
        for _ in range(4):  # 4 stride-2 convolutions
            feat_size = (feat_size + 1) // 2  # equivalent to ceil(feat_size / 2) for odd sizes
        
        print(f"  Discriminator feature size: {feat_size}x{feat_size} (512 channels)")
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * feat_size * feat_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )
    
    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out


# ================================================================
# VGG Feature Extractor (for Perceptual Loss)
# ================================================================

class VGGFeatureExtractor(nn.Module):
    """
    VGG19 feature extractor for perceptual loss.
    Extracts features from conv5_4 (layer 35) by default.
    """
    def __init__(self, feature_layer=35, use_bn=False):
        super().__init__()
        try:
            from torchvision.models import vgg19, vgg19_bn
            if use_bn:
                model = vgg19_bn(pretrained=True)
            else:
                model = vgg19(pretrained=True)
            self.features = nn.Sequential(*list(model.features.children())[:feature_layer])
            
            # Freeze VGG
            for param in self.features.parameters():
                param.requires_grad = False
            
            self.available = True
        except Exception as e:
            print(f"Warning: Could not load VGG19 for perceptual loss: {e}")
            self.features = None
            self.available = False
    
    def forward(self, x):
        if self.features is None:
            return None
        
        # VGG expects 3-channel input, replicate single channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        return self.features(x)


# ================================================================
# Dataset
# ================================================================

class HiCDataset(Dataset):
    """Dataset for Hi-C super-resolution"""
    def __init__(self, lr_data, hr_data):
        self.lr = torch.from_numpy(lr_data).float()
        self.hr = torch.from_numpy(hr_data).float()
    
    def __len__(self):
        return len(self.lr)
    
    def __getitem__(self, idx):
        return self.lr[idx], self.hr[idx]


# ================================================================
# Losses
# ================================================================

class GANLoss(nn.Module):
    """GAN loss (can be vanilla or LSGAN)"""
    def __init__(self, loss_type='vanilla'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.loss = nn.MSELoss()
    
    def forward(self, pred, target_is_real):
        if target_is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        return self.loss(pred, target)


# ================================================================
# Training Functions
# ================================================================

def train_epoch(
    generator, discriminator, 
    g_optimizer, d_optimizer,
    dataloader, device,
    gan_loss_fn, vgg_extractor=None,
    lambda_l1=1.0,
    lambda_adv=0.001,
    lambda_vgg=0.006,
    epoch=0
):
    """Train one epoch"""
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    total_l1_loss = 0
    total_adv_loss = 0
    total_vgg_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (lr, hr) in enumerate(pbar):
        lr = lr.to(device)
        hr = hr.to(device)
        batch_size = lr.size(0)
        
        # ============================================
        # Train Discriminator
        # ============================================
        d_optimizer.zero_grad()
        
        # Generate fake images
        fake_hr = generator(lr)
        
        # Real loss
        real_pred = discriminator(hr)
        d_loss_real = gan_loss_fn(real_pred, True)
        
        # Fake loss
        fake_pred = discriminator(fake_hr.detach())
        d_loss_fake = gan_loss_fn(fake_pred, False)
        
        # Total discriminator loss
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        d_loss.backward()
        d_optimizer.step()
        
        # ============================================
        # Train Generator
        # ============================================
        g_optimizer.zero_grad()
        
        # Generate fake images
        fake_hr = generator(lr)
        
        # L1 / Content loss
        l1_loss = F.l1_loss(fake_hr, hr)
        
        # Adversarial loss
        fake_pred = discriminator(fake_hr)
        adv_loss = gan_loss_fn(fake_pred, True)
        
        # VGG / Perceptual loss
        vgg_loss = torch.tensor(0.0, device=device)
        if vgg_extractor is not None and vgg_extractor.available:
            real_features = vgg_extractor(hr)
            fake_features = vgg_extractor(fake_hr)
            if real_features is not None and fake_features is not None:
                vgg_loss = F.mse_loss(fake_features, real_features)
        
        # Total generator loss
        g_loss = lambda_l1 * l1_loss + lambda_adv * adv_loss + lambda_vgg * vgg_loss
        
        g_loss.backward()
        g_optimizer.step()
        
        # Accumulate losses
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        total_l1_loss += l1_loss.item()
        total_adv_loss += adv_loss.item()
        total_vgg_loss += vgg_loss.item()
        
        pbar.set_postfix({
            'G': f'{g_loss.item():.4f}',
            'D': f'{d_loss.item():.4f}',
            'L1': f'{l1_loss.item():.4f}'
        })
    
    n_batches = len(dataloader)
    return {
        'g_loss': total_g_loss / n_batches,
        'd_loss': total_d_loss / n_batches,
        'l1_loss': total_l1_loss / n_batches,
        'adv_loss': total_adv_loss / n_batches,
        'vgg_loss': total_vgg_loss / n_batches,
    }


@torch.no_grad()
def validate(generator, lr_val, hr_val, device, n_samples=200):
    """Validate generator"""
    generator.eval()
    
    n = min(n_samples, len(lr_val))
    lr = torch.from_numpy(lr_val[:n]).float().to(device)
    hr = torch.from_numpy(hr_val[:n]).float().to(device)
    
    # Generate
    fake_hr = generator(lr)
    
    # Compute metrics
    mse = F.mse_loss(fake_hr, hr).item()
    
    # PCC
    fake_np = fake_hr.cpu().numpy().flatten()
    hr_np = hr.cpu().numpy().flatten()
    pcc, _ = pearsonr(fake_np, hr_np)
    
    # PSNR
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    return {
        'mse': mse,
        'pcc': pcc,
        'psnr': psnr,
    }


# ================================================================
# Data Loading
# ================================================================

def ensure_nchw(arr):
    """Ensure array is in NCHW format"""
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
    """
    Load training data and apply normalization.
    
    Supports:
        1. NPZ with raw data (train_lr, train_hr) - will normalize
        2. NPZ with normalized data (already processed)
        3. Separate npy files
    """
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
        
        # Check if data needs normalization (raw data has large values)
        if lr_raw.max() > 10 or hr_raw.max() > 10:
            print("\n  Detected RAW data, applying normalization...")
            
            # Create and fit preprocessor
            preprocessor = HiCPreprocessor()
            preprocessor.fit(lr_raw, hr_raw)
            
            # Transform data
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
        
        print(f"  LR: {lr.shape}, range [{lr.min():.4f}, {lr.max():.4f}]")
        print(f"  HR: {hr.shape}, range [{hr.min():.4f}, {hr.max():.4f}]")
        
        # Check if normalization needed
        if lr.max() > 10 or hr.max() > 10:
            print("\n  Detected RAW data, applying normalization...")
            preprocessor = HiCPreprocessor()
            preprocessor.fit(lr, hr)
            lr = preprocessor.transform_X(lr)
            hr = preprocessor.transform_Y(hr)
            print(f"  LR (norm): range [{lr.min():.4f}, {lr.max():.4f}]")
            print(f"  HR (norm): range [{hr.min():.4f}, {hr.max():.4f}]")
    else:
        raise ValueError("Must provide either --train_npz or both --train_lr and --train_hr")
    
    return lr, hr, preprocessor


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Train DFHiC (PyTorch)')
    
    # Data
    parser.add_argument('--train_npz', type=str, default=None,
                       help='NPZ file with train_lr and train_hr')
    parser.add_argument('--train_lr', type=str, default=None,
                       help='Low-resolution training data (normalized)')
    parser.add_argument('--train_hr', type=str, default=None,
                       help='High-resolution training data (normalized)')
    parser.add_argument('--output_dir', type=str, default='checkpoints_dfhic')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=1e-4,
                       help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=1e-4,
                       help='Discriminator learning rate')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Model
    parser.add_argument('--num_features', type=int, default=64,
                       help='Number of features in generator')
    parser.add_argument('--num_res_blocks', type=int, default=16,
                       help='Number of residual blocks in generator')
    parser.add_argument('--input_size', type=int, default=40,
                       help='Input tile size')
    
    # Loss weights
    parser.add_argument('--lambda_l1', type=float, default=1.0,
                       help='Weight for L1 loss')
    parser.add_argument('--lambda_adv', type=float, default=0.001,
                       help='Weight for adversarial loss')
    parser.add_argument('--lambda_vgg', type=float, default=0.0,
                       help='Weight for VGG perceptual loss (0 to disable)')
    parser.add_argument('--gan_type', type=str, default='vanilla',
                       choices=['vanilla', 'lsgan'],
                       help='GAN loss type')
    
    # Training schedule
    parser.add_argument('--pretrain_epochs', type=int, default=0,
                       help='Epochs to pretrain generator with L1 only')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                       help='Learning rate decay factor')
    parser.add_argument('--decay_every', type=int, default=100,
                       help='Decay learning rate every N epochs')
    
    # Validation
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Validate arguments
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
    
    # Split into train/val
    n_samples = len(lr_data)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    lr_train, hr_train = lr_data[train_idx], hr_data[train_idx]
    lr_val, hr_val = lr_data[val_idx], hr_data[val_idx]
    
    print(f"  Train: {len(lr_train)}, Val: {len(lr_val)}")
    
    # Save preprocessor if created
    if preprocessor is not None:
        preprocessor.save(output_dir / 'preprocessor.pt')
    
    # Create dataloader
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
        in_channels=1,
        out_channels=1,
        num_features=args.num_features,
        num_res_blocks=args.num_res_blocks
    ).to(device)
    
    discriminator = Discriminator(
        in_channels=1,
        input_size=args.input_size
    ).to(device)
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  Generator: {g_params / 1e6:.2f}M parameters")
    print(f"  Discriminator: {d_params / 1e6:.2f}M parameters")
    
    # VGG for perceptual loss
    vgg_extractor = None
    if args.lambda_vgg > 0:
        print("  Loading VGG19 for perceptual loss...")
        vgg_extractor = VGGFeatureExtractor().to(device)
    
    # ================================================================
    # Optimizers and loss
    # ================================================================
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.999))
    
    gan_loss_fn = GANLoss(args.gan_type).to(device)
    
    # Learning rate scheduler
    g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=args.decay_every, gamma=args.lr_decay)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=args.decay_every, gamma=args.lr_decay)
    
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
    print(f"  Lambda L1: {args.lambda_l1}")
    print(f"  Lambda Adv: {args.lambda_adv}")
    print(f"  Lambda VGG: {args.lambda_vgg}")
    print(f"  GAN type: {args.gan_type}")
    
    # Compute baseline
    baseline = validate(generator, lr_val, hr_val, device)
    print(f"\nBaseline (untrained): MSE={baseline['mse']:.6f}, PCC={baseline['pcc']:.4f}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Pretrain with L1 only
        if epoch <= args.pretrain_epochs:
            lambda_adv = 0.0
            print(f"\n>>> Epoch {epoch}/{args.epochs} [Pretrain: L1 only]")
        else:
            lambda_adv = args.lambda_adv
            if epoch == args.pretrain_epochs + 1:
                print(f"\n>>> Starting GAN training")
        
        # Train
        losses = train_epoch(
            generator, discriminator,
            g_optimizer, d_optimizer,
            train_loader, device,
            gan_loss_fn, vgg_extractor,
            lambda_l1=args.lambda_l1,
            lambda_adv=lambda_adv,
            lambda_vgg=args.lambda_vgg,
            epoch=epoch
        )
        
        # Update learning rates
        g_scheduler.step()
        d_scheduler.step()
        
        # Validate
        val_metrics = validate(generator, lr_val, hr_val, device)
        
        # Log
        print(f"  Train: G={losses['g_loss']:.4f}, D={losses['d_loss']:.4f}, L1={losses['l1_loss']:.4f}")
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
            # Add preprocessor info if available
            if preprocessor is not None:
                save_dict['preprocessor'] = {
                    'X_mean': preprocessor.X_mean,
                    'X_std': preprocessor.X_std,
                    'Y_mean': preprocessor.Y_mean,
                    'Y_std': preprocessor.Y_std
                }
            torch.save(save_dict, output_dir / 'best_model.pt')
        
        # Save periodic checkpoint
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
                    'X_mean': preprocessor.X_mean,
                    'X_std': preprocessor.X_std,
                    'Y_mean': preprocessor.Y_mean,
                    'Y_std': preprocessor.Y_std
                }
            torch.save(save_dict, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save final model
    save_dict = {
        'epoch': args.epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'best_pcc': best_pcc,
        'config': vars(args)
    }
    if preprocessor is not None:
        save_dict['preprocessor'] = {
            'X_mean': preprocessor.X_mean,
            'X_std': preprocessor.X_std,
            'Y_mean': preprocessor.Y_mean,
            'Y_std': preprocessor.Y_std
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
