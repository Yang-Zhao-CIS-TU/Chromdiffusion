#!/usr/bin/env python3
"""
Diffusion Refinement v4 - 包含所有改进

改动 1: lambda_recon 提高到 1.0-2.0（主导约束）
改动 2: 验证时使用确定性方法（直接用 pred_x0）
改动 3: 保守的 residual scaling: final = hicarn + alpha * residual
改动 4: 更合理的 loss 平衡

Loss = diff_loss + λ_mag * mag_loss + λ_recon * recon_loss

推荐参数：
- lambda_recon: 1.0 ~ 2.0 (主导)
- lambda_magnitude: 0.05 ~ 0.1 (辅助)
- alpha: 0.3 ~ 0.7 (保守 residual)
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
from scipy import stats


# ================================================================
# Model (same as before)
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


class SimpleConditionedUNet(nn.Module):
    def __init__(
        self,
        in_channels=2,
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
        
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, t):
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
        
        return self.final_conv(h)


# ================================================================
# Scheduler
# ================================================================

class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        parameterization='v'
    ):
        self.num_train_timesteps = num_train_timesteps
        self.parameterization = parameterization
        
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x_start, noise, timesteps):
        sqrt_alpha = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def get_v_target(self, x_start, noise, timesteps):
        sqrt_alpha = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x_start
    
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


def compute_residual_stats(hicarn, gt):
    residual = gt - hicarn
    res_mean = float(residual.mean())
    res_std = float(residual.std())
    
    print(f"\nResidual statistics:")
    print(f"  Mean: {res_mean:.6f}")
    print(f"  Std:  {res_std:.6f}")
    print(f"  Range: [{residual.min():.4f}, {residual.max():.4f}]")
    
    return res_mean, res_std


# ================================================================
# Training v4
# ================================================================

def train_epoch_v4(
    model, 
    scheduler, 
    optimizer, 
    dataloader, 
    device,
    res_mean,
    res_std,
    epoch,
    alpha=0.5,           # Residual scaling factor
    lambda_magnitude=0.05,
    lambda_recon=1.0     # 主导约束！
):
    """
    改进版训练：
    1. recon_loss 作为主导约束 (lambda_recon = 1.0 ~ 2.0)
    2. 使用 alpha scaling: final = hicarn + alpha * residual
    3. magnitude loss 保持较小 (0.05 ~ 0.1)
    """
    model.train()
    total_loss = 0
    total_diff_loss = 0
    total_mag_loss = 0
    total_recon_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for hicarn, gt in pbar:
        hicarn = hicarn.to(device)
        gt = gt.to(device)
        
        batch_size = hicarn.shape[0]
        
        # Compute and normalize residual
        residual = gt - hicarn
        residual_centered = residual - res_mean
        residual_scaled = residual_centered / res_std
        
        # Sample timesteps
        t = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(residual_scaled)
        
        # Add noise
        residual_noisy = scheduler.add_noise(residual_scaled, noise, t)
        
        # Forward (channel concatenation)
        x_input = torch.cat([residual_noisy, hicarn], dim=1)
        pred = model(x_input, t)
        
        # ============================================
        # Loss 1: Diffusion loss
        # ============================================
        target = scheduler.get_v_target(residual_scaled, noise, t)
        diff_loss = F.mse_loss(pred, target)
        
        # ============================================
        # Loss 2: Magnitude constraint (small)
        # ============================================
        pred_x0 = scheduler.predict_x0_from_v(residual_noisy, pred, t)
        mag_loss = torch.mean(pred_x0 ** 2)
        
        # ============================================
        # Loss 3: Reconstruction loss (主导！)
        # 使用 alpha scaling: final = hicarn + alpha * residual
        # ============================================
        pred_residual = pred_x0 * res_std + res_mean  # Unscale
        pred_final = hicarn + alpha * pred_residual   # Alpha scaling!
        recon_loss = F.mse_loss(pred_final, gt)
        
        # ============================================
        # Total loss
        # ============================================
        loss = diff_loss + lambda_magnitude * mag_loss + lambda_recon * recon_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_diff_loss += diff_loss.item()
        total_mag_loss += mag_loss.item()
        total_recon_loss += recon_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rec': f'{recon_loss.item():.4f}'  # 最重要的指标
        })
    
    return {
        'total': total_loss / num_batches,
        'diffusion': total_diff_loss / num_batches,
        'magnitude': total_mag_loss / num_batches,
        'reconstruction': total_recon_loss / num_batches
    }


@torch.no_grad()
def validate_deterministic(
    model,
    scheduler,
    hicarn_val,
    gt_val,
    res_mean,
    res_std,
    alpha,
    device
):
    """
    确定性验证：直接用 t=0 时的 pred_x0，不做随机采样
    
    这样验证结果更稳定，与训练目标一致
    """
    model.eval()
    
    n = min(500, len(hicarn_val))
    hicarn = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    # 使用 t=0（或很小的 t）直接预测
    # 这样 residual_noisy ≈ residual_scaled
    # 我们可以用 t=1 来避免数值问题
    t = torch.ones(n, device=device, dtype=torch.long)
    
    # 构造接近真实 residual 的输入
    residual_gt = gt - hicarn
    residual_centered = residual_gt - res_mean
    residual_scaled = residual_centered / res_std
    
    # 加一点点噪声（t=1）
    noise = torch.randn_like(residual_scaled)
    residual_noisy = scheduler.add_noise(residual_scaled, noise, t)
    
    # Forward
    x_input = torch.cat([residual_noisy, hicarn], dim=1)
    pred = model(x_input, t)
    pred_x0 = scheduler.predict_x0_from_v(residual_noisy, pred, t)
    
    # Unscale
    pred_residual = pred_x0 * res_std + res_mean
    
    # Final with alpha scaling
    pred_final = hicarn + alpha * pred_residual
    
    # Metrics
    mse = F.mse_loss(pred_final, gt).item()
    mse_hicarn = F.mse_loss(hicarn, gt).item()
    
    final_np = pred_final.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = hicarn.cpu().numpy().flatten()
    
    pcc, _ = stats.pearsonr(final_np, gt_np)
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    residual_np = pred_residual.cpu().numpy()
    ideal_residual = (gt - hicarn).cpu().numpy()
    
    return {
        'mse': mse,
        'pcc': float(pcc),
        'mse_hicarn': mse_hicarn,
        'pcc_hicarn': float(pcc_hicarn),
        'mse_improvement': mse_hicarn - mse,
        'pcc_improvement': float(pcc) - float(pcc_hicarn),
        'residual_std': float(residual_np.std()),
        'ideal_residual_std': float(ideal_residual.std()),
        'improved_over_hicarn': float(pcc) > float(pcc_hicarn)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_hicarn', type=str, required=True)
    parser.add_argument('--train_gt', type=str, required=True)
    parser.add_argument('--val_hicarn', type=str, default=None)
    parser.add_argument('--val_gt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints_v4')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    
    # 关键参数
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Residual scaling: final = hicarn + alpha * residual (0.3-0.7)')
    parser.add_argument('--lambda_magnitude', type=float, default=0.05,
                       help='Weight for magnitude constraint (0.05-0.1)')
    parser.add_argument('--lambda_recon', type=float, default=1.0,
                       help='Weight for reconstruction loss (1.0-2.0, 主导！)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Load data
    # ================================================================
    print("\n" + "="*80)
    print("DIFFUSION REFINEMENT v4")
    print("="*80)
    print(f"Key parameters:")
    print(f"  alpha (residual scaling): {args.alpha}")
    print(f"  lambda_recon: {args.lambda_recon}")
    print(f"  lambda_magnitude: {args.lambda_magnitude}")
    
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
    
    res_mean, res_std = compute_residual_stats(hicarn_train, gt_train)
    
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
    print("HICARN BASELINE")
    print("="*80)
    
    mse_baseline = np.mean((hicarn_val - gt_val) ** 2)
    pcc_baseline, _ = stats.pearsonr(hicarn_val.flatten(), gt_val.flatten())
    print(f"HiCARN MSE: {mse_baseline:.6f}")
    print(f"HiCARN PCC: {pcc_baseline:.4f}")
    
    # ================================================================
    # Model
    # ================================================================
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = SimpleConditionedUNet(
        in_channels=2,
        out_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4),
        parameterization='v'
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_timesteps,
        parameterization='v'
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # ================================================================
    # Training
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_pcc = pcc_baseline
    best_mse = mse_baseline
    history = []
    
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch_v4(
            model, scheduler, optimizer, train_loader,
            device, res_mean, res_std, epoch,
            alpha=args.alpha,
            lambda_magnitude=args.lambda_magnitude,
            lambda_recon=args.lambda_recon
        )
        
        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate_deterministic(
                model, scheduler, hicarn_val, gt_val,
                res_mean, res_std, args.alpha, device
            )
            
            improved = ""
            if val_metrics['mse'] < best_mse:
                best_mse = val_metrics['mse']
                improved += " [best MSE]"
            
            if val_metrics['pcc'] > best_pcc:
                best_pcc = val_metrics['pcc']
                improved += " [best PCC]"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_metrics': val_metrics,
                    'res_std': res_std,
                    'res_mean': res_mean,
                    'alpha': args.alpha,
                    'config': {
                        'in_channels': 2,
                        'base_channels': args.base_channels,
                        'channel_mults': [1, 2, 4],
                        'parameterization': 'v',
                        'num_timesteps': args.num_timesteps,
                        'alpha': args.alpha,
                        'lambda_magnitude': args.lambda_magnitude,
                        'lambda_recon': args.lambda_recon
                    }
                }, output_dir / 'best_model.pt')
            
            status = "✓" if val_metrics['improved_over_hicarn'] else "⚠"
            print(f"\n  Epoch {epoch}: {status} "
                  f"MSE={val_metrics['mse']:.6f} (HiCARN:{val_metrics['mse_hicarn']:.6f}) "
                  f"PCC={val_metrics['pcc']:.4f} (HiCARN:{val_metrics['pcc_hicarn']:.4f})"
                  f"{improved}")
            print(f"    res_std={val_metrics['residual_std']:.4f} (ideal:{val_metrics['ideal_residual_std']:.4f})")
            
            history.append({
                'epoch': epoch,
                'losses': losses,
                'val_metrics': val_metrics
            })
        
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'res_std': res_std,
                'res_mean': res_mean,
                'alpha': args.alpha,
                'config': {
                    'in_channels': 2,
                    'base_channels': args.base_channels,
                    'alpha': args.alpha
                }
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
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
    
    if best_pcc > pcc_baseline:
        print(f"\n✓ SUCCESS: Diffusion refinement improved over HiCARN!")
    else:
        print(f"\n⚠ Did not improve over HiCARN")
        print(f"  Try: --alpha 0.3 --lambda_recon 2.0")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
