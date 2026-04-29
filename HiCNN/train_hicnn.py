#!/usr/bin/env python3
"""
HiCNN-PyTorch: Hi-C Enhancement using Deep Residual CNN

This script trains HiCNN model with HiCARN-style data preprocessing
(log1p + z-score normalization).

HiCNN Architecture:
  - Initial conv layers: 40x40 -> 28x28
  - 25 residual blocks with 128 channels
  - Final conv to output

Key differences from original HiCNN:
  - Uses HiCARN-style normalization (log1p + z-score) instead of clipping to 100
  - Output is same size as input (40x40) for compatibility
  - Added validation metrics (PCC, SSIM)

Usage:
    python train_hicnn.py \
        --train_npz /path/to/train_data_raw_ratio16.npz \
        --output_dir checkpoints_hicnn \
        --epochs 200 \
        --batch_size 64 \
        --device cuda

Reference:
    HiCNN: A very deep convolutional neural network to better enhance the 
    resolution of Hi-C data (Bioinformatics, 2019)
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
import datetime


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
            self.X_std = float((q75 - q25) / 1.35)
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
# HiCNN Model (Modified for 40x40 -> 40x40)
# ================================================================

class HiCNN(nn.Module):
    """
    HiCNN: Very Deep Residual CNN for Hi-C Enhancement
    
    Original architecture produces 28x28 output from 40x40 input.
    This version is modified to produce 40x40 output.
    
    Architecture:
        - conv1: 1 -> 8 channels, kernel=13, reduces size
        - conv2: 8 -> 1 channel, kernel=1
        - conv3: 1 -> 128 channels, kernel=3, padding=1
        - 25 residual blocks (conv4R)
        - conv5: 128 -> 1 channel, kernel=3, padding=1
    """
    def __init__(self, num_res_blocks=25, num_features=128, output_full_size=True):
        super(HiCNN, self).__init__()
        
        self.output_full_size = output_full_size
        
        if output_full_size:
            # Modified for 40x40 -> 40x40
            self.conv1 = nn.Conv2d(1, 8, 9, padding=4)  # Same size
            self.conv2 = nn.Conv2d(8, 1, 1)
        else:
            # Original: 40x40 -> 28x28
            self.conv1 = nn.Conv2d(1, 8, 13)  # 40-13+1 = 28
            self.conv2 = nn.Conv2d(8, 1, 1)
        
        self.conv3 = nn.Conv2d(1, num_features, 3, padding=1, bias=False)
        self.conv4R = nn.Conv2d(num_features, num_features, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(num_features, 1, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        self.num_res_blocks = num_res_blocks
        
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # Initial convolutions
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        residual = x
        
        # Deep residual blocks
        x2 = self.conv3(x)
        out = x2
        for _ in range(self.num_res_blocks):
            out = self.conv4R(self.relu(self.conv4R(self.relu(out))))
            out = torch.add(out, x2)
        
        # Final convolution
        out = self.conv5(self.relu(out))
        out = torch.add(out, residual)
        
        return out


class HiCNNOriginal(nn.Module):
    """
    Original HiCNN architecture (40x40 -> 28x28)
    For compatibility with original paper.
    """
    def __init__(self, num_res_blocks=25, num_features=128):
        super(HiCNNOriginal, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, 13)  # 40 -> 28
        self.conv2 = nn.Conv2d(8, 1, 1)
        self.conv3 = nn.Conv2d(1, num_features, 3, padding=1, bias=False)
        self.conv4R = nn.Conv2d(num_features, num_features, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(num_features, 1, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        self.num_res_blocks = num_res_blocks
        
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
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
# Dataset
# ================================================================

class HiCDataset(Dataset):
    """Dataset for Hi-C super-resolution"""
    def __init__(self, lr_data, hr_data, crop_center=False, crop_size=28):
        self.lr = torch.from_numpy(lr_data).float()
        self.hr = torch.from_numpy(hr_data).float()
        self.crop_center = crop_center
        self.crop_size = crop_size
        
        # Pre-crop HR if using original HiCNN (28x28 output)
        if crop_center:
            offset = (hr_data.shape[-1] - crop_size) // 2
            self.hr = self.hr[:, :, offset:offset+crop_size, offset:offset+crop_size]
    
    def __len__(self):
        return len(self.lr)
    
    def __getitem__(self, idx):
        return self.lr[idx], self.hr[idx]


# ================================================================
# Training Functions
# ================================================================

def train_epoch(model, dataloader, optimizer, device, clip=0.01):
    """Train one epoch"""
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Handle size mismatch for original HiCNN
        if output.shape != target.shape:
            # Center crop target to match output
            diff = target.shape[-1] - output.shape[-1]
            offset = diff // 2
            target = target[:, :, offset:offset+output.shape[-1], offset:offset+output.shape[-1]]
        
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        lr_current = optimizer.param_groups[0]['lr']
        clip_value = clip / lr_current if lr_current > 0 else clip
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, lr_val, hr_val, device, n_samples=500, crop_center=False, crop_size=28):
    """Validate model"""
    model.eval()
    
    n = min(n_samples, len(lr_val))
    lr = torch.from_numpy(lr_val[:n]).float().to(device)
    hr = hr_val[:n]
    
    # Generate predictions in batches
    batch_size = 64
    outputs = []
    for i in range(0, n, batch_size):
        batch = lr[i:i+batch_size]
        out = model(batch)
        outputs.append(out.cpu().numpy())
    
    output = np.concatenate(outputs, axis=0)
    
    # Handle size mismatch
    if crop_center or output.shape[-1] != hr.shape[-1]:
        diff = hr.shape[-1] - output.shape[-1]
        if diff > 0:
            offset = diff // 2
            hr = hr[:, :, offset:offset+output.shape[-1], offset:offset+output.shape[-1]]
    
    # Compute metrics
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
    
    return {
        'mse': float(mse),
        'pcc': float(pcc),
        'psnr': float(psnr),
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
        
        # Check if data needs normalization
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
        
        print(f"  LR: {lr.shape}, range [{lr.min():.4f}, {lr.max():.4f}]")
        print(f"  HR: {hr.shape}, range [{hr.min():.4f}, {hr.max():.4f}]")
        
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
    parser = argparse.ArgumentParser(description='Train HiCNN with HiCARN-style preprocessing')
    
    # Data
    parser.add_argument('--train_npz', type=str, default=None,
                       help='NPZ file with train_lr and train_hr')
    parser.add_argument('--train_lr', type=str, default=None,
                       help='Low-resolution training data')
    parser.add_argument('--train_hr', type=str, default=None,
                       help='High-resolution training data')
    parser.add_argument('--output_dir', type=str, default='checkpoints_hicnn')
    
    # Model
    parser.add_argument('--num_res_blocks', type=int, default=25,
                       help='Number of residual blocks (default: 25)')
    parser.add_argument('--num_features', type=int, default=128,
                       help='Number of features in residual blocks (default: 128)')
    parser.add_argument('--original_arch', action='store_true',
                       help='Use original HiCNN architecture (40->28)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=0.01,
                       help='Gradient clipping (default: 0.01)')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Validation
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # Resume
    parser.add_argument('--resume', type=str, default=None)
    
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
    
    # Save preprocessor
    if preprocessor is not None:
        preprocessor.save(output_dir / 'preprocessor.pt')
    
    # Create dataloader
    use_original = args.original_arch
    train_dataset = HiCDataset(lr_train, hr_train, crop_center=use_original, crop_size=28)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    # ================================================================
    # Create model
    # ================================================================
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    if use_original:
        print("  Using ORIGINAL HiCNN architecture (40x40 -> 28x28)")
        model = HiCNNOriginal(
            num_res_blocks=args.num_res_blocks,
            num_features=args.num_features
        ).to(device)
    else:
        print("  Using MODIFIED HiCNN architecture (40x40 -> 40x40)")
        model = HiCNN(
            num_res_blocks=args.num_res_blocks,
            num_features=args.num_features,
            output_full_size=True
        ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e6:.2f}M")
    
    # ================================================================
    # Optimizer and scheduler
    # ================================================================
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # ================================================================
    # Resume
    # ================================================================
    start_epoch = 1
    best_pcc = 0.0
    history = []
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
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
    print(f"  Learning rate: {args.lr}")
    print(f"  Residual blocks: {args.num_res_blocks}")
    print(f"  Features: {args.num_features}")
    
    # Baseline
    baseline = validate(model, lr_val, hr_val, device, crop_center=use_original)
    print(f"\nBaseline (untrained): MSE={baseline['mse']:.6f}, PCC={baseline['pcc']:.4f}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n>>> Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.clip)
        
        # Validate
        val_metrics = validate(model, lr_val, hr_val, device, crop_center=use_original)
        
        # Update scheduler
        scheduler.step(val_metrics['mse'])
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val: MSE={val_metrics['mse']:.6f}, PCC={val_metrics['pcc']:.4f}, PSNR={val_metrics['psnr']:.2f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': current_lr,
            **val_metrics
        })
        
        # Save best model
        if val_metrics['pcc'] > best_pcc:
            best_pcc = val_metrics['pcc']
            print(f"  >>> New best PCC: {best_pcc:.4f}")
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_pcc': best_pcc,
                'val_metrics': val_metrics,
                'config': vars(args)
            }
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
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
        'model_state_dict': model.state_dict(),
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
