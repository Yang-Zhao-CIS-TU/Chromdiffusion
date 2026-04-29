#!/usr/bin/env python3
"""
修复版 Residual Diffusion 训练脚本 - 适配合并数据文件

关键修复：
1. UNet 输入改为 2 通道: [residual_noisy, hicarn] (channel concatenation)
2. Residual 做 scaling (除以 std_res)
3. 确保 conditioning 真正生效

数据格式：
- --train_hicarn: 合并的 HiCARN predictions (chr1-chr17)
- --train_gt: 合并的 GT (chr1-chr17)
- --val_hicarn: 验证集 HiCARN (chr18)
- --val_gt: 验证集 GT (chr18)
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
import sys
from datetime import datetime
from scipy import stats

# ================================================================
# 简化版 UNet (2通道输入，支持 channel concat conditioning)
# ================================================================

def get_timestep_embedding(timesteps, embedding_dim):
    """Sinusoidal timestep embeddings"""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """Residual block with time embedding"""
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
        
        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.shortcut(x)


class SimpleConditionedUNet(nn.Module):
    """
    简化版 UNet，使用 Channel Concatenation 进行 Conditioning
    
    输入: [residual_noisy, hicarn] concatenated -> [B, 2, H, W]
    输出: predicted v (or eps)
    
    这样确保 conditioning 一定会被使用！
    """
    def __init__(
        self,
        in_channels=2,      # 关键：2通道 (residual + condition)
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=256,
        parameterization='v'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.parameterization = parameterization
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
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
        
        # Middle
        self.mid1 = ResBlock(ch, ch, time_emb_dim)
        self.mid2 = ResBlock(ch, ch, time_emb_dim)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            self.decoder.append(ResBlock(ch + channels.pop(), out_ch, time_emb_dim))
            self.decoder.append(ResBlock(out_ch, out_ch, time_emb_dim))
            ch = out_ch
        
        # Output
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, t):
        """
        Forward pass
        
        Args:
            x: [B, 2, H, W] - concatenated [residual_noisy, hicarn]
            t: [B] - timesteps
        
        注意：这里不再有单独的 condition 参数！
        condition 已经在 x 的第二个通道里了。
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.init_conv.out_channels)
        t_emb = self.time_mlp(t_emb)
        
        # Initial conv
        h = self.init_conv(x)
        
        # Encoder
        skips = [h]
        for i in range(0, len(self.encoder), 2):
            h = self.encoder[i](h, t_emb)
            h = self.encoder[i+1](h, t_emb)
            skips.append(h)
            h = self.downsample[i//2](h)
        
        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        
        # Decoder
        for i in range(0, len(self.decoder), 2):
            h = self.upsample[i//2](h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = self.decoder[i](h, t_emb)
            h = self.decoder[i+1](h, t_emb)
        
        return self.final_conv(h)


# ================================================================
# Noise Scheduler
# ================================================================

class DDPMScheduler:
    """DDPM Scheduler with v-parameterization support"""
    
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        parameterization='v'
    ):
        self.num_train_timesteps = num_train_timesteps
        self.parameterization = parameterization
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For sampling
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
    
    def add_noise(self, x_start, noise, timesteps):
        """Add noise to x_start"""
        sqrt_alpha = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        # Reshape for broadcasting
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def get_v_target(self, x_start, noise, timesteps):
        """Compute v-prediction target: v = sqrt(alpha) * eps - sqrt(1-alpha) * x0"""
        sqrt_alpha = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x_start
    
    def predict_x0_from_v(self, x_t, v, timesteps):
        """Recover x0 from v-prediction: x0 = sqrt(alpha) * x_t - sqrt(1-alpha) * v"""
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
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        return self


# ================================================================
# Preprocessor (for loading)
# ================================================================

class RobustHiCPreprocessor:
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self._is_fitted = False

sys.modules['__main__'].RobustHiCPreprocessor = RobustHiCPreprocessor


# ================================================================
# Data Loading
# ================================================================

def ensure_nchw(arr):
    """Ensure array is [N, C, H, W]"""
    arr = np.asarray(arr)
    
    if arr.ndim == 3:
        # [N, H, W] -> [N, 1, H, W]
        return arr[:, np.newaxis, :, :]
    elif arr.ndim == 4:
        if arr.shape[1] == 1:
            # Already NCHW
            return arr
        elif arr.shape[-1] == 1:
            # NHWC -> NCHW
            return np.transpose(arr, (0, 3, 1, 2))
    
    raise ValueError(f"Cannot convert shape {arr.shape} to NCHW")


def load_data(hicarn_path, gt_path):
    """Load HiCARN predictions and GT"""
    print(f"  Loading HiCARN: {hicarn_path}")
    hicarn = np.load(hicarn_path)
    hicarn = ensure_nchw(hicarn)
    print(f"    Shape: {hicarn.shape}, range [{hicarn.min():.4f}, {hicarn.max():.4f}]")
    
    print(f"  Loading GT: {gt_path}")
    gt = np.load(gt_path)
    gt = ensure_nchw(gt)
    print(f"    Shape: {gt.shape}, range [{gt.min():.4f}, {gt.max():.4f}]")
    
    assert hicarn.shape == gt.shape, f"Shape mismatch: {hicarn.shape} vs {gt.shape}"
    
    return hicarn.astype(np.float32), gt.astype(np.float32)


def compute_residual_stats(hicarn, gt):
    """Compute residual statistics for scaling"""
    residual = gt - hicarn
    
    res_mean = float(residual.mean())
    res_std = float(residual.std())
    
    print(f"\nResidual statistics:")
    print(f"  Mean: {res_mean:.6f}")
    print(f"  Std:  {res_std:.6f}")
    print(f"  Range: [{residual.min():.4f}, {residual.max():.4f}]")
    
    return res_mean, res_std


# ================================================================
# Training Functions
# ================================================================

def train_epoch(
    model, 
    scheduler, 
    optimizer, 
    dataloader, 
    device,
    res_std,
    epoch
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for hicarn, gt in pbar:
        hicarn = hicarn.to(device)
        gt = gt.to(device)
        
        batch_size = hicarn.shape[0]
        
        # Compute and scale residual
        residual = gt - hicarn
        residual_scaled = residual / res_std  # Scale to ~unit variance
        
        # Sample timesteps
        t = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(residual_scaled)
        
        # Add noise to residual
        residual_noisy = scheduler.add_noise(residual_scaled, noise, t)
        
        # ============================================
        # 关键修复：Channel Concatenation
        # ============================================
        # 把 hicarn 作为第二个通道拼进去
        x_input = torch.cat([residual_noisy, hicarn], dim=1)  # [B, 2, H, W]
        
        # Forward
        pred = model(x_input, t)  # 注意：不再单独传 condition
        
        # Compute target
        if scheduler.parameterization == 'v':
            target = scheduler.get_v_target(residual_scaled, noise, t)
        else:
            target = noise
        
        # Loss
        loss = F.mse_loss(pred, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model,
    scheduler,
    hicarn_val,
    gt_val,
    res_std,
    device,
    num_steps=50
):
    """Quick validation by sampling"""
    model.eval()
    
    # Take a subset
    n = min(100, len(hicarn_val))
    hicarn = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    # Sample residuals
    x_t = torch.randn_like(hicarn)  # Start from noise
    
    # Simple uniform timesteps for validation
    timesteps = torch.linspace(scheduler.num_train_timesteps - 1, 0, num_steps).long().to(device)
    
    for i, t in enumerate(timesteps):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        
        # Channel concat
        x_input = torch.cat([x_t, hicarn], dim=1)
        
        # Predict
        pred = model(x_input, t_batch)
        
        # Get pred_x0
        pred_x0 = scheduler.predict_x0_from_v(x_t, pred, t_batch)
        
        # DDIM step (simplified)
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_t = scheduler.alphas_cumprod[t]
            alpha_t_next = scheduler.alphas_cumprod[t_next]
            
            pred_noise = (x_t - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt()
            x_t = alpha_t_next.sqrt() * pred_x0 + (1 - alpha_t_next).sqrt() * pred_noise
        else:
            x_t = pred_x0
    
    # Unscale residual
    residual_pred = x_t * res_std
    
    # Final prediction
    final = hicarn + residual_pred
    
    # Metrics
    mse = F.mse_loss(final, gt).item()
    
    # PCC
    final_np = final.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    pcc, _ = stats.pearsonr(final_np, gt_np)
    
    # Also check HiCARN baseline for comparison
    mse_hicarn = F.mse_loss(hicarn, gt).item()
    hicarn_np = hicarn.cpu().numpy().flatten()
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    return {
        'mse': mse,
        'pcc': float(pcc),
        'mse_hicarn': mse_hicarn,
        'pcc_hicarn': float(pcc_hicarn)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Data paths (single merged files)
    parser.add_argument('--train_hicarn', type=str, required=True,
                       help='Path to training HiCARN predictions (normalized)')
    parser.add_argument('--train_gt', type=str, required=True,
                       help='Path to training GT (normalized)')
    parser.add_argument('--val_hicarn', type=str, default=None,
                       help='Path to validation HiCARN predictions')
    parser.add_argument('--val_gt', type=str, default=None,
                       help='Path to validation GT')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints_conditioned')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Model params
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Load data
    # ================================================================
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    print("\nTraining data:")
    hicarn_train, gt_train = load_data(args.train_hicarn, args.train_gt)
    print(f"Total training: {len(hicarn_train)} samples")
    
    # Validation data
    if args.val_hicarn and args.val_gt:
        print("\nValidation data:")
        hicarn_val, gt_val = load_data(args.val_hicarn, args.val_gt)
        print(f"Total validation: {len(hicarn_val)} samples")
    else:
        # Use last 10% of training data for validation
        split = int(len(hicarn_train) * 0.9)
        hicarn_val = hicarn_train[split:]
        gt_val = gt_train[split:]
        hicarn_train = hicarn_train[:split]
        gt_train = gt_train[:split]
        print(f"\nUsing 10% of training data for validation:")
        print(f"  Training: {len(hicarn_train)} samples")
        print(f"  Validation: {len(hicarn_val)} samples")
    
    # Compute residual statistics
    res_mean, res_std = compute_residual_stats(hicarn_train, gt_train)
    
    # Create dataloader
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
    # Create model
    # ================================================================
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = SimpleConditionedUNet(
        in_channels=2,  # 关键：2通道输入！
        out_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4),
        parameterization='v'
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    print(f"Input channels: {model.in_channels} (residual + hicarn)")
    print(f"Parameterization: {model.parameterization}")
    
    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_timesteps,
        parameterization='v'
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # ================================================================
    # Check HiCARN baseline
    # ================================================================
    print("\n" + "="*80)
    print("HICARN BASELINE")
    print("="*80)
    
    # Compute HiCARN metrics on validation set
    mse_baseline = np.mean((hicarn_val - gt_val) ** 2)
    pcc_baseline, _ = stats.pearsonr(hicarn_val.flatten(), gt_val.flatten())
    print(f"HiCARN MSE: {mse_baseline:.6f}")
    print(f"HiCARN PCC: {pcc_baseline:.4f}")
    print("(Diffusion should improve on these metrics)")
    
    # ================================================================
    # Training loop
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_pcc = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, scheduler, optimizer, train_loader,
            device, res_std, epoch
        )
        
        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate(
                model, scheduler, hicarn_val, gt_val,
                res_std, device
            )
            
            print(f"  Validation:")
            print(f"    Refined:  MSE={val_metrics['mse']:.6f}, PCC={val_metrics['pcc']:.4f}")
            print(f"    HiCARN:   MSE={val_metrics['mse_hicarn']:.6f}, PCC={val_metrics['pcc_hicarn']:.4f}")
            
            # Check if we're improving over HiCARN
            if val_metrics['pcc'] > val_metrics['pcc_hicarn']:
                print(f"    ✓ Improving over HiCARN!")
            else:
                print(f"    ⚠ Not yet better than HiCARN")
            
            # Save best
            if val_metrics['pcc'] > best_pcc:
                best_pcc = val_metrics['pcc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'res_std': res_std,
                    'res_mean': res_mean,
                    'config': {
                        'in_channels': 2,
                        'base_channels': args.base_channels,
                        'channel_mults': [1, 2, 4],
                        'parameterization': 'v',
                        'num_timesteps': args.num_timesteps
                    }
                }, output_dir / 'best_model.pt')
                print(f"    ✓ Saved best model (PCC={val_metrics['pcc']:.4f})")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_mse': val_metrics['mse'],
                'val_pcc': val_metrics['pcc']
            })
        else:
            history.append({
                'epoch': epoch,
                'train_loss': train_loss
            })
        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'res_std': res_std,
                'res_mean': res_mean,
                'config': {
                    'in_channels': 2,
                    'base_channels': args.base_channels,
                    'channel_mults': [1, 2, 4],
                    'parameterization': 'v',
                    'num_timesteps': args.num_timesteps
                }
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'res_std': res_std,
        'res_mean': res_mean,
        'config': {
            'in_channels': 2,
            'base_channels': args.base_channels,
            'channel_mults': [1, 2, 4],
            'parameterization': 'v',
            'num_timesteps': args.num_timesteps
        }
    }, output_dir / 'final_model.pt')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best PCC: {best_pcc:.4f}")
    print(f"HiCARN baseline PCC: {pcc_baseline:.4f}")
    
    if best_pcc > pcc_baseline:
        print(f"✓ Improved over HiCARN by {(best_pcc - pcc_baseline):.4f}")
    else:
        print(f"⚠ Did not improve over HiCARN")
    
    print(f"\nCheckpoints saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Run sampling:")
    print(f"   python sample_conditioned.py \\")
    print(f"       --checkpoint {output_dir}/best_model.pt \\")
    print(f"       --hicarn_path <your_hicarn_test.npy>")
    print(f"\n2. Run conditioning check:")
    print(f"   python check_conditioning.py ...")


if __name__ == '__main__':
    main()
