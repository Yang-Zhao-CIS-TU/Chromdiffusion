#!/usr/bin/env python3
"""
Supervised Residual Refinement - 不用 Diffusion！

核心问题：
- Diffusion 训练时输入包含 GT 信息 (residual_noisy)
- 但采样时只有纯噪声
- 导致 train/test mismatch

解决方案：
- 直接用 supervised learning
- 输入：hicarn
- 输出：residual
- Loss：MSE(hicarn + alpha * residual, gt)

这是最简单、最直接的方法，如果这个不 work，说明任务本身很难。
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
import sys
from scipy import stats


# ================================================================
# Simple UNet for Residual Prediction (No Diffusion!)
# ================================================================

class ResidualUNet(nn.Module):
    """
    简单的 UNet 直接预测 residual
    
    输入: hicarn [B, 1, H, W]
    输出: residual [B, 1, H, W]
    """
    def __init__(self, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_block(1, base_channels)
        self.enc2 = self._make_block(base_channels, base_channels * 2)
        self.enc3 = self._make_block(base_channels * 2, base_channels * 4)
        
        self.pool = nn.MaxPool2d(2)
        
        # Middle
        self.mid = self._make_block(base_channels * 4, base_channels * 4)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 2, stride=2)
        self.dec3 = self._make_block(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._make_block(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._make_block(base_channels * 2, base_channels)
        
        # Output (initialize to small values for conservative start)
        self.out = nn.Conv2d(base_channels, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    
    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Middle
        m = self.mid(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.up3(m)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)


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


def load_data(hicarn_path, gt_path):
    print(f"  Loading HiCARN: {hicarn_path}")
    hicarn = np.load(hicarn_path)
    hicarn = ensure_nchw(hicarn)
    print(f"    Shape: {hicarn.shape}, range [{hicarn.min():.4f}, {hicarn.max():.4f}]")
    
    print(f"  Loading GT: {gt_path}")
    gt = np.load(gt_path)
    gt = ensure_nchw(gt)
    print(f"    Shape: {gt.shape}, range [{gt.min():.4f}, {gt.max():.4f}]")
    
    assert hicarn.shape == gt.shape
    return hicarn.astype(np.float32), gt.astype(np.float32)


# ================================================================
# Training
# ================================================================

def train_epoch(model, optimizer, dataloader, device, alpha):
    model.train()
    total_loss = 0
    total_res_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    
    for hicarn, gt in pbar:
        hicarn = hicarn.to(device)
        gt = gt.to(device)
        
        # 直接预测 residual
        residual = model(hicarn)
        
        # Final = hicarn + alpha * residual
        final = hicarn + alpha * residual
        
        # Target residual
        target_residual = gt - hicarn
        
        # Loss 1: Reconstruction (main)
        recon_loss = F.mse_loss(final, gt)
        
        # Loss 2: Direct residual supervision
        res_loss = F.mse_loss(residual, target_residual)
        
        # Loss 3: L2 regularization to keep residual small
        mag_loss = 0.01 * torch.mean(residual ** 2)
        
        loss = recon_loss + 0.5 * res_loss + mag_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_res_loss += res_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'res': f'{res_loss.item():.4f}'
        })
    
    return total_loss / num_batches, total_res_loss / num_batches


@torch.no_grad()
def validate(model, hicarn_val, gt_val, device, alpha):
    model.eval()
    
    hicarn = torch.from_numpy(hicarn_val).float().to(device)
    gt = torch.from_numpy(gt_val).float().to(device)
    
    # 预测
    residual = model(hicarn)
    final = hicarn + alpha * residual
    
    # MSE
    mse = F.mse_loss(final, gt).item()
    mse_hicarn = F.mse_loss(hicarn, gt).item()
    
    # PCC
    final_np = final.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = hicarn.cpu().numpy().flatten()
    
    pcc, _ = stats.pearsonr(final_np, gt_np)
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    # Residual correlation
    residual_np = residual.cpu().numpy()
    ideal_residual = (gt - hicarn).cpu().numpy()
    
    res_corr, _ = stats.pearsonr(
        residual_np.flatten(), ideal_residual.flatten()
    )
    
    return {
        'mse': mse,
        'pcc': float(pcc),
        'mse_hicarn': mse_hicarn,
        'pcc_hicarn': float(pcc_hicarn),
        'res_corr': float(res_corr),
        'residual_std': float(residual_np.std()),
        'ideal_residual_std': float(ideal_residual.std()),
        'improved': mse < mse_hicarn
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_hicarn', type=str, required=True)
    parser.add_argument('--train_gt', type=str, required=True)
    parser.add_argument('--val_hicarn', type=str, default=None)
    parser.add_argument('--val_gt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='supervised_refinement')
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Residual scaling: final = hicarn + alpha * residual')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--base_channels', type=int, default=64)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Load data
    # ================================================================
    print("\n" + "="*80)
    print("SUPERVISED RESIDUAL REFINEMENT (No Diffusion!)")
    print("="*80)
    print(f"Alpha: {args.alpha}")
    
    print("\nLoading data...")
    hicarn_train, gt_train = load_data(args.train_hicarn, args.train_gt)
    
    if args.val_hicarn and args.val_gt:
        hicarn_val, gt_val = load_data(args.val_hicarn, args.val_gt)
    else:
        split = int(len(hicarn_train) * 0.9)
        hicarn_val = hicarn_train[split:]
        gt_val = gt_train[split:]
        hicarn_train = hicarn_train[:split]
        gt_train = gt_train[:split]
        print(f"  Using 10% for validation")
    
    # Residual stats
    residual = gt_train - hicarn_train
    print(f"\nResidual statistics:")
    print(f"  Mean: {residual.mean():.6f}")
    print(f"  Std:  {residual.std():.6f}")
    
    train_dataset = TensorDataset(
        torch.from_numpy(hicarn_train).float(),
        torch.from_numpy(gt_train).float()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # ================================================================
    # Baseline
    # ================================================================
    print("\n" + "="*80)
    print("BASELINE")
    print("="*80)
    
    mse_baseline = np.mean((hicarn_val - gt_val) ** 2)
    pcc_baseline, _ = stats.pearsonr(hicarn_val.flatten(), gt_val.flatten())
    print(f"HiCARN: MSE={mse_baseline:.6f}, PCC={pcc_baseline:.4f}")
    
    # ================================================================
    # Model
    # ================================================================
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = ResidualUNet(base_channels=args.base_channels).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # ================================================================
    # Training
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print("关键指标：res_corr 应该快速上升（supervised 应该能到 > 0.5）")
    
    best_mse = mse_baseline
    best_pcc = pcc_baseline
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_res = train_epoch(
            model, optimizer, train_loader, device, args.alpha
        )
        scheduler.step()
        
        # Validate
        val_metrics = validate(model, hicarn_val, gt_val, device, args.alpha)
        
        improved = ""
        if val_metrics['mse'] < best_mse:
            best_mse = val_metrics['mse']
            improved += " [best MSE]"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'alpha': args.alpha
            }, output_dir / 'best_model.pt')
        
        if val_metrics['pcc'] > best_pcc:
            best_pcc = val_metrics['pcc']
            improved += " [best PCC]"
        
        status = "✓" if val_metrics['improved'] else "⚠"
        
        print(f"Epoch {epoch:3d}: {status} "
              f"MSE={val_metrics['mse']:.6f} (HiCARN:{val_metrics['mse_hicarn']:.6f}) "
              f"PCC={val_metrics['pcc']:.4f} "
              f"res_corr={val_metrics['res_corr']:.4f}"
              f"{improved}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_metrics': val_metrics
        })
    
    # Save
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    print(f"\nHiCARN baseline:")
    print(f"  MSE: {mse_baseline:.6f}")
    print(f"  PCC: {pcc_baseline:.4f}")
    
    print(f"\nBest refined:")
    print(f"  MSE: {best_mse:.6f} (Δ={mse_baseline - best_mse:+.6f})")
    print(f"  PCC: {best_pcc:.4f} (Δ={best_pcc - pcc_baseline:+.4f})")
    
    if best_mse < mse_baseline:
        print(f"\n✓ SUCCESS: Supervised refinement improved over HiCARN!")
        print(f"  这证明 refinement 是可行的")
    else:
        print(f"\n⚠ Supervised refinement did NOT improve over HiCARN")
        print(f"  这说明：")
        print(f"  1. HiCARN 已经接近最优")
        print(f"  2. 或者 GT 与 HiCARN 的差异太小/太随机")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
