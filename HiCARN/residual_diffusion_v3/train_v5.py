#!/usr/bin/env python3
"""
Diffusion Refinement v5 - 修复验证泄漏问题

关键修复：
- validate 时从纯噪声开始采样，不使用 GT 信息
- 使用完整的 DDIM 采样流程

v4 的 Bug：validate_deterministic 用了 GT residual 作为输入，导致指标虚假地好
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
# Training
# ================================================================

def train_epoch_v5(
    model, 
    scheduler, 
    optimizer, 
    dataloader, 
    device,
    res_mean,
    res_std,
    epoch,
    alpha=0.5,
    lambda_magnitude=0.05,
    lambda_recon=1.0
):
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
        
        # Loss 1: Diffusion loss
        target = scheduler.get_v_target(residual_scaled, noise, t)
        diff_loss = F.mse_loss(pred, target)
        
        # Loss 2: Magnitude constraint
        pred_x0 = scheduler.predict_x0_from_v(residual_noisy, pred, t)
        mag_loss = torch.mean(pred_x0 ** 2)
        
        # Loss 3: Reconstruction loss
        pred_residual = pred_x0 * res_std + res_mean
        pred_final = hicarn + alpha * pred_residual
        recon_loss = F.mse_loss(pred_final, gt)
        
        # Total loss
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
            'rec': f'{recon_loss.item():.4f}'
        })
    
    return {
        'total': total_loss / num_batches,
        'diffusion': total_diff_loss / num_batches,
        'magnitude': total_mag_loss / num_batches,
        'reconstruction': total_recon_loss / num_batches
    }


@torch.no_grad()
def validate_proper_sampling(
    model,
    scheduler,
    hicarn_val,
    gt_val,
    res_mean,
    res_std,
    alpha,
    device,
    num_steps=20,  # 验证时用较少步数加速
    seed=42        # 固定 seed 让验证可重复
):
    """
    正确的验证：从纯噪声开始采样，不使用任何 GT 信息
    
    关键：x_t 从 randn 开始，不从 GT residual 开始！
    """
    model.eval()
    
    # 固定随机种子
    torch.manual_seed(seed)
    
    n = min(200, len(hicarn_val))
    hicarn = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    # ============================================
    # 关键：从纯噪声开始！不使用 GT！
    # ============================================
    x_t = torch.randn_like(hicarn)  # 纯随机噪声，没有 GT 信息
    
    # DDIM 采样
    timesteps = torch.linspace(
        scheduler.num_train_timesteps - 1, 0, num_steps
    ).long().to(device)
    
    pred_x0 = None
    for i, t in enumerate(timesteps):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        
        # Channel concat with hicarn (condition)
        x_input = torch.cat([x_t, hicarn], dim=1)
        
        # Model prediction
        pred = model(x_input, t_batch)
        pred_x0 = scheduler.predict_x0_from_v(x_t, pred, t_batch)
        
        # DDIM step
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_t = scheduler.alphas_cumprod[t]
            alpha_t_next = scheduler.alphas_cumprod[t_next]
            
            pred_noise = (x_t - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt()
            x_t = alpha_t_next.sqrt() * pred_x0 + (1 - alpha_t_next).sqrt() * pred_noise
        else:
            x_t = pred_x0
    
    # Unscale residual
    pred_residual = pred_x0 * res_std + res_mean
    
    # Final with alpha scaling
    pred_final = hicarn + alpha * pred_residual
    pred_final = torch.clamp(pred_final, -5, 5)
    
    # ============================================
    # Metrics
    # ============================================
    mse = F.mse_loss(pred_final, gt).item()
    mse_hicarn = F.mse_loss(hicarn, gt).item()
    
    # Oracle (upper bound): what if we use GT residual directly
    oracle_final = hicarn + alpha * (gt - hicarn)  # = (1-alpha)*hicarn + alpha*gt
    mse_oracle = F.mse_loss(oracle_final, gt).item()
    
    final_np = pred_final.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = hicarn.cpu().numpy().flatten()
    
    pcc, _ = stats.pearsonr(final_np, gt_np)
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    residual_np = pred_residual.cpu().numpy()
    ideal_residual = (gt - hicarn).cpu().numpy()
    
    # Correlation between predicted and ideal residual
    corr_residual, _ = stats.pearsonr(
        residual_np.flatten(), ideal_residual.flatten()
    )
    
    return {
        'mse': mse,
        'pcc': float(pcc),
        'mse_hicarn': mse_hicarn,
        'pcc_hicarn': float(pcc_hicarn),
        'mse_oracle': mse_oracle,  # 这个是 (1-alpha)^2 * mse_hicarn
        'mse_improvement': mse_hicarn - mse,
        'pcc_improvement': float(pcc) - float(pcc_hicarn),
        'residual_std': float(residual_np.std()),
        'ideal_residual_std': float(ideal_residual.std()),
        'residual_corr': float(corr_residual),  # 预测残差与理想残差的相关性
        'improved_over_hicarn': float(pcc) > float(pcc_hicarn)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_hicarn', type=str, required=True)
    parser.add_argument('--train_gt', type=str, required=True)
    parser.add_argument('--val_hicarn', type=str, default=None)
    parser.add_argument('--val_gt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints_v5')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lambda_magnitude', type=float, default=0.05)
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    
    parser.add_argument('--val_steps', type=int, default=20,
                       help='Number of DDIM steps for validation')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Load data
    # ================================================================
    print("\n" + "="*80)
    print("DIFFUSION REFINEMENT v5 (with proper validation)")
    print("="*80)
    print(f"Key parameters:")
    print(f"  alpha: {args.alpha}")
    print(f"  lambda_recon: {args.lambda_recon}")
    print(f"  lambda_magnitude: {args.lambda_magnitude}")
    print(f"  val_steps: {args.val_steps}")
    
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
        print(f"  Using 10% for validation: train={len(hicarn_train)}, val={len(hicarn_val)}")
    
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
    # Baseline & Oracle
    # ================================================================
    print("\n" + "="*80)
    print("BASELINE & ORACLE")
    print("="*80)
    
    mse_baseline = np.mean((hicarn_val - gt_val) ** 2)
    pcc_baseline, _ = stats.pearsonr(hicarn_val.flatten(), gt_val.flatten())
    
    # Oracle: hicarn + alpha * (gt - hicarn) = (1-alpha)*hicarn + alpha*gt
    # MSE_oracle = (1-alpha)^2 * MSE_baseline
    mse_oracle = (1 - args.alpha) ** 2 * mse_baseline
    
    print(f"HiCARN baseline:")
    print(f"  MSE: {mse_baseline:.6f}")
    print(f"  PCC: {pcc_baseline:.4f}")
    print(f"\nOracle (if model learns perfect residual):")
    print(f"  MSE: {mse_oracle:.6f} (= (1-{args.alpha})^2 * {mse_baseline:.6f})")
    print(f"\n⚠️  如果 val MSE ≈ {mse_oracle:.4f}，说明验证有 GT 泄漏！")
    
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
        losses = train_epoch_v5(
            model, scheduler, optimizer, train_loader,
            device, res_mean, res_std, epoch,
            alpha=args.alpha,
            lambda_magnitude=args.lambda_magnitude,
            lambda_recon=args.lambda_recon
        )
        
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate_proper_sampling(
                model, scheduler, hicarn_val, gt_val,
                res_mean, res_std, args.alpha, device,
                num_steps=args.val_steps
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
            
            print(f"\n  Epoch {epoch}: {status}")
            print(f"    MSE:     {val_metrics['mse']:.6f} (HiCARN: {val_metrics['mse_hicarn']:.6f}, Oracle: {val_metrics['mse_oracle']:.6f})")
            print(f"    PCC:     {val_metrics['pcc']:.4f} (HiCARN: {val_metrics['pcc_hicarn']:.4f})")
            print(f"    res_std: {val_metrics['residual_std']:.4f} (ideal: {val_metrics['ideal_residual_std']:.4f})")
            print(f"    res_corr: {val_metrics['residual_corr']:.4f} (预测残差与理想残差的相关性)")
            
            if improved:
                print(f"    {improved}")
            
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
    
    print(f"\nOracle (upper bound):")
    print(f"  MSE: {mse_oracle:.6f}")
    
    print(f"\nBest refined:")
    print(f"  MSE: {best_mse:.6f} (Δ={mse_baseline - best_mse:+.6f})")
    print(f"  PCC: {best_pcc:.4f} (Δ={best_pcc - pcc_baseline:+.4f})")
    
    if best_pcc > pcc_baseline:
        print(f"\n✓ SUCCESS: Improved over HiCARN!")
    else:
        print(f"\n⚠ Did not improve over HiCARN")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
