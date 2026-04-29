#!/usr/bin/env python3
"""
修复版采样脚本 - 与 train_conditioned.py 配套使用

关键：
1. 使用 Channel Concatenation: [residual_noisy, hicarn]
2. 使用保存的 res_std 进行反缩放
3. 正确使用 v-parameterization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import math
import sys
from scipy import stats


# ================================================================
# Model (must match training)
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
    
    def predict_x0_from_v(self, x_t, v, t):
        """x0 = sqrt(alpha) * x_t - sqrt(1-alpha) * v"""
        if isinstance(t, int):
            t = torch.tensor([t], device=x_t.device)
        
        sqrt_alpha = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t]).sqrt()
        
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
# For loading preprocessor
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
# Sampling
# ================================================================

@torch.no_grad()
def sample_ddim(
    model,
    scheduler,
    hicarn,
    res_std,
    num_steps=50,
    device='cuda'
):
    """
    DDIM sampling with channel concatenation conditioning
    
    Args:
        model: Trained model (in_channels=2)
        scheduler: DDPMScheduler
        hicarn: HiCARN predictions [B, 1, H, W]
        res_std: Residual scaling factor
        num_steps: Number of denoising steps
        device: Device
    
    Returns:
        final: hicarn + residual [B, 1, H, W]
        residual: Predicted residual [B, 1, H, W]
    """
    batch_size = hicarn.shape[0]
    
    # Start from noise
    x_t = torch.randn_like(hicarn)
    
    # Timesteps
    timesteps = torch.linspace(
        scheduler.num_train_timesteps - 1, 0, num_steps
    ).long().to(device)
    
    pred_x0 = None
    
    for i, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # ============================================
        # 关键：Channel Concatenation
        # ============================================
        x_input = torch.cat([x_t, hicarn], dim=1)  # [B, 2, H, W]
        
        # Predict
        pred = model(x_input, t_batch)
        
        # Get pred_x0
        pred_x0 = scheduler.predict_x0_from_v(x_t, pred, t_batch)
        
        # DDIM step
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            
            alpha_t = scheduler.alphas_cumprod[t]
            alpha_t_next = scheduler.alphas_cumprod[t_next]
            
            # Predicted noise
            pred_noise = (x_t - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt()
            
            # Update x_t
            x_t = alpha_t_next.sqrt() * pred_x0 + (1 - alpha_t_next).sqrt() * pred_noise
        else:
            x_t = pred_x0
    
    # Unscale residual
    residual_scaled = pred_x0
    residual = residual_scaled * res_std
    
    # Final prediction
    final = hicarn + residual
    
    # Clip to reasonable range
    final = torch.clamp(final, -5, 5)
    
    return final, residual


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', {})
    in_channels = config.get('in_channels', 2)
    base_channels = config.get('base_channels', 64)
    parameterization = config.get('parameterization', 'v')
    
    print(f"  in_channels: {in_channels}")
    print(f"  base_channels: {base_channels}")
    print(f"  parameterization: {parameterization}")
    
    # Get residual scaling
    res_std = checkpoint.get('res_std', 1.0)
    res_mean = checkpoint.get('res_mean', 0.0)
    print(f"  res_std: {res_std:.4f}")
    
    # Create model
    model = SimpleConditionedUNet(
        in_channels=in_channels,
        out_channels=1,
        base_channels=base_channels,
        parameterization=parameterization
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        parameterization=parameterization
    ).to(device)
    
    return model, scheduler, res_std


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--hicarn_path', type=str, required=True,
                       help='Path to HiCARN predictions (normalized)')
    parser.add_argument('--output_dir', type=str, default='refined_conditioned')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    model, scheduler, res_std = load_checkpoint(args.checkpoint, device)
    
    # Load HiCARN predictions
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    hicarn = np.load(args.hicarn_path)
    if hicarn.ndim == 3:
        hicarn = hicarn[:, np.newaxis, :, :]
    
    num_samples = len(hicarn)
    print(f"HiCARN shape: {hicarn.shape}")
    print(f"HiCARN range: [{hicarn.min():.4f}, {hicarn.max():.4f}]")
    
    # Sample
    print("\n" + "="*80)
    print("SAMPLING")
    print("="*80)
    print(f"Steps: {args.num_steps}")
    print(f"Batch size: {args.batch_size}")
    
    final_list = []
    residual_list = []
    
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing"):
        start = batch_idx * args.batch_size
        end = min((batch_idx + 1) * args.batch_size, num_samples)
        
        batch_hicarn = torch.from_numpy(hicarn[start:end]).float().to(device)
        
        final_batch, residual_batch = sample_ddim(
            model, scheduler, batch_hicarn,
            res_std, args.num_steps, device
        )
        
        final_list.append(final_batch.cpu().numpy())
        residual_list.append(residual_batch.cpu().numpy())
    
    final = np.concatenate(final_list, axis=0)
    residuals = np.concatenate(residual_list, axis=0)
    
    print(f"\n✓ Sampling complete!")
    print(f"  Final range: [{final.min():.4f}, {final.max():.4f}]")
    print(f"  Residual range: [{residuals.min():.4f}, {residuals.max():.4f}]")
    print(f"  Residual stats: mean={residuals.mean():.4f}, std={residuals.std():.4f}")
    
    # Save
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    np.save(output_dir / 'refined_norm.npy', final)
    np.save(output_dir / 'residuals.npy', residuals)
    np.save(output_dir / 'hicarn_norm.npy', hicarn)
    
    print(f"✓ Saved to {output_dir}")
    
    # Quick sanity check
    print("\n" + "="*80)
    print("SANITY CHECK")
    print("="*80)
    
    # Check correlation between residual and hicarn
    corr_res_hicarn, _ = stats.pearsonr(
        residuals.flatten(), hicarn.flatten()
    )
    print(f"Corr(residual, hicarn): {corr_res_hicarn:.4f}")
    
    if abs(corr_res_hicarn) < 0.01:
        print("⚠️  Warning: Residual has very low correlation with HiCARN")
        print("   This suggests conditioning might not be working properly")
    else:
        print("✓ Residual shows correlation with HiCARN (conditioning working)")
    
    # Check final vs hicarn
    diff = final - hicarn
    print(f"\nFinal - HiCARN:")
    print(f"  Range: [{diff.min():.4f}, {diff.max():.4f}]")
    print(f"  Mean: {diff.mean():.4f}, Std: {diff.std():.4f}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
