#!/usr/bin/env python3
"""
Conditional Rectified Flow for HiC Refinement

核心思路：
- 学习速度场 v_θ(x_t, t, c)，从噪声流向 residual
- x_0 = residual (GT - HiCARN)
- x_1 = noise ~ N(0, I)
- c = hicarn (条件)

训练：
- 线性插值: x_t = (1-t) * x_0 + t * x_1
- 目标速度: v* = x_1 - x_0
- Loss: MSE(v_θ(x_t, t, c), v*)

推理：
- 从 x_1 ~ N(0, I) 出发
- ODE 积分回 x_0
- final = hicarn + x_0

关键改进：
1. Classifier-Free Guidance (CFG) - 训练时随机 drop condition
2. FiLM conditioning - 比 concat 更强的条件注入
3. 从纯噪声出发，强制依赖条件
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
# FiLM Conditioning Layer
# ================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation
    比简单 concat 更强的条件注入方式
    
    output = gamma * input + beta
    其中 gamma, beta 由条件生成
    """
    def __init__(self, in_channels, cond_channels):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_channels, in_channels)
        self.beta_fc = nn.Linear(cond_channels, in_channels)
        
        # Initialize to identity transform
        nn.init.ones_(self.gamma_fc.weight.data[:, 0] if self.gamma_fc.weight.shape[1] > 0 else self.gamma_fc.weight.data)
        nn.init.zeros_(self.gamma_fc.weight.data[:, 1:] if self.gamma_fc.weight.shape[1] > 1 else torch.tensor([]))
        nn.init.zeros_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.weight.data)
        nn.init.zeros_(self.beta_fc.bias.data)
    
    def forward(self, x, cond):
        """
        x: [B, C, H, W]
        cond: [B, cond_channels]
        """
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        return gamma * x + beta


# ================================================================
# Condition Encoder
# ================================================================

class ConditionEncoder(nn.Module):
    """
    将 hicarn 编码为条件向量
    """
    def __init__(self, in_channels=1, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 40 -> 20
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 20 -> 10
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 10 -> 5
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),  # 5 -> 1
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


# ================================================================
# Time Embedding
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


# ================================================================
# ResBlock with FiLM
# ================================================================

class ResBlockFiLM(nn.Module):
    """ResBlock with FiLM conditioning"""
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        # Time embedding
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        # FiLM for condition
        self.film = FiLMLayer(out_ch, cond_dim)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb, cond_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        # Apply FiLM conditioning
        h = self.film(h, cond_emb)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.shortcut(x)


# ================================================================
# Flow Model with FiLM Conditioning
# ================================================================

class FlowUNet(nn.Module):
    """
    UNet for Rectified Flow with FiLM conditioning
    
    输入: x_t [B, 1, H, W] (noisy residual)
    条件: hicarn [B, 1, H, W]
    时间: t [B]
    输出: velocity [B, 1, H, W]
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        cond_dim=256,
        time_emb_dim=256
    ):
        super().__init__()
        
        # Condition encoder
        self.cond_encoder = ConditionEncoder(in_channels=1, out_dim=cond_dim)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial conv (只处理 x_t，不 concat condition)
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        ch = base_channels
        channels = [ch]
        
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.encoder.append(ResBlockFiLM(ch, out_ch, time_emb_dim, cond_dim))
            self.encoder.append(ResBlockFiLM(out_ch, out_ch, time_emb_dim, cond_dim))
            self.downsample.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            channels.append(out_ch)
            ch = out_ch
        
        # Middle
        self.mid1 = ResBlockFiLM(ch, ch, time_emb_dim, cond_dim)
        self.mid2 = ResBlockFiLM(ch, ch, time_emb_dim, cond_dim)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            self.decoder.append(ResBlockFiLM(ch + channels.pop(), out_ch, time_emb_dim, cond_dim))
            self.decoder.append(ResBlockFiLM(out_ch, out_ch, time_emb_dim, cond_dim))
            ch = out_ch
        
        # Output
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, t, cond, cond_drop_mask=None):
        """
        x: [B, 1, H, W] - noisy sample x_t
        t: [B] - time (0 to 1)
        cond: [B, 1, H, W] - hicarn condition
        cond_drop_mask: [B] - bool mask, True = drop condition (for CFG)
        """
        # Encode condition
        cond_emb = self.cond_encoder(cond)  # [B, cond_dim]
        
        # Apply condition dropout (CFG)
        if cond_drop_mask is not None:
            # Zero out condition embedding where mask is True
            cond_emb = cond_emb * (~cond_drop_mask).float().unsqueeze(-1)
        
        # Time embedding
        t_emb = get_timestep_embedding(t, self.init_conv.out_channels)
        t_emb = self.time_mlp(t_emb)
        
        # Initial conv
        h = self.init_conv(x)
        
        # Encoder
        skips = [h]
        for i in range(0, len(self.encoder), 2):
            h = self.encoder[i](h, t_emb, cond_emb)
            h = self.encoder[i+1](h, t_emb, cond_emb)
            skips.append(h)
            h = self.downsample[i//2](h)
        
        # Middle
        h = self.mid1(h, t_emb, cond_emb)
        h = self.mid2(h, t_emb, cond_emb)
        
        # Decoder
        for i in range(0, len(self.decoder), 2):
            h = self.upsample[i//2](h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = self.decoder[i](h, t_emb, cond_emb)
            h = self.decoder[i+1](h, t_emb, cond_emb)
        
        return self.final_conv(h)


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
    
    return res_mean, res_std


# ================================================================
# Training
# ================================================================

def train_epoch(
    model,
    optimizer,
    dataloader,
    device,
    res_mean,
    res_std,
    epoch,
    cond_drop_prob=0.1  # CFG: probability of dropping condition
):
    """
    Rectified Flow training
    
    x_0 = residual (scaled)
    x_1 = noise
    x_t = (1-t) * x_0 + t * x_1
    target = x_1 - x_0 (velocity)
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for hicarn, gt in pbar:
        hicarn = hicarn.to(device)
        gt = gt.to(device)
        
        batch_size = hicarn.shape[0]
        
        # Compute residual (x_0)
        residual = gt - hicarn
        residual_scaled = (residual - res_mean) / res_std  # x_0
        
        # Sample noise (x_1)
        noise = torch.randn_like(residual_scaled)  # x_1
        
        # Sample time t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)
        
        # Linear interpolation: x_t = (1-t) * x_0 + t * x_1
        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * residual_scaled + t_expand * noise
        
        # Target velocity: v* = x_1 - x_0
        target_velocity = noise - residual_scaled
        
        # CFG: randomly drop condition
        cond_drop_mask = torch.rand(batch_size, device=device) < cond_drop_prob
        
        # Forward
        pred_velocity = model(x_t, t, hicarn, cond_drop_mask)
        
        # Loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()
def sample_ode(
    model,
    hicarn,
    res_mean,
    res_std,
    device,
    num_steps=50,
    cfg_scale=1.0  # Classifier-Free Guidance scale
):
    """
    ODE sampling from x_1 (noise) to x_0 (residual)
    
    dx/dt = -v(x, t, c)
    从 t=1 积分到 t=0
    """
    model.eval()
    batch_size = hicarn.shape[0]
    
    # Start from noise (x_1)
    x = torch.randn_like(hicarn)
    
    # Time steps from 1 to 0
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    
    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr  # negative
        
        t_batch = torch.full((batch_size,), t_curr, device=device)
        
        # Predict velocity with CFG
        if cfg_scale != 1.0:
            # Unconditional prediction
            v_uncond = model(x, t_batch, hicarn, 
                           cond_drop_mask=torch.ones(batch_size, dtype=torch.bool, device=device))
            # Conditional prediction
            v_cond = model(x, t_batch, hicarn,
                         cond_drop_mask=torch.zeros(batch_size, dtype=torch.bool, device=device))
            # CFG interpolation
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t_batch, hicarn,
                     cond_drop_mask=torch.zeros(batch_size, dtype=torch.bool, device=device))
        
        # Euler step: x_next = x_curr + dt * v
        # Note: we're going from t=1 to t=0, so dt is negative
        # velocity points from x_0 to x_1, so we subtract it
        x = x + dt * v  # dt is negative, v points x_0 -> x_1
    
    # x is now approximately x_0 (residual_scaled)
    residual = x * res_std + res_mean
    
    return residual


@torch.no_grad()
def validate(
    model,
    hicarn_val,
    gt_val,
    res_mean,
    res_std,
    device,
    num_steps=20,
    cfg_scale=1.0,
    seed=42
):
    """Validate with proper ODE sampling"""
    model.eval()
    torch.manual_seed(seed)
    
    n = min(200, len(hicarn_val))
    hicarn = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    # Sample residual
    residual = sample_ode(model, hicarn, res_mean, res_std, device, num_steps, cfg_scale)
    
    # Final prediction
    final = hicarn + residual
    final = torch.clamp(final, -5, 5)
    
    # Metrics
    mse = F.mse_loss(final, gt).item()
    mse_hicarn = F.mse_loss(hicarn, gt).item()
    
    final_np = final.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = hicarn.cpu().numpy().flatten()
    
    pcc, _ = stats.pearsonr(final_np, gt_np)
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    # Residual correlation
    residual_np = residual.cpu().numpy()
    ideal_residual = (gt - hicarn).cpu().numpy()
    
    res_corr, _ = stats.pearsonr(residual_np.flatten(), ideal_residual.flatten())
    
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
    parser.add_argument('--output_dir', type=str, default='checkpoints_flow')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--cond_dim', type=int, default=256)
    
    # CFG
    parser.add_argument('--cond_drop_prob', type=float, default=0.1,
                       help='Probability of dropping condition during training (for CFG)')
    parser.add_argument('--cfg_scale', type=float, default=2.0,
                       help='Classifier-Free Guidance scale for sampling')
    
    parser.add_argument('--val_steps', type=int, default=20)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Load data
    # ================================================================
    print("\n" + "="*80)
    print("CONDITIONAL RECTIFIED FLOW")
    print("="*80)
    print(f"CFG: cond_drop_prob={args.cond_drop_prob}, cfg_scale={args.cfg_scale}")
    
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
    
    model = FlowUNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4),
        cond_dim=args.cond_dim
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # ================================================================
    # Training
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print("关键指标：res_corr 应该上升，CFG 应该帮助模型使用条件")
    
    best_mse = mse_baseline
    best_res_corr = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, optimizer, train_loader, device,
            res_mean, res_std, epoch,
            cond_drop_prob=args.cond_drop_prob
        )
        
        if epoch % 5 == 0 or epoch == 1:
            # Validate with different CFG scales
            val_metrics = validate(
                model, hicarn_val, gt_val, res_mean, res_std,
                device, args.val_steps, args.cfg_scale
            )
            
            # Also try without CFG for comparison
            val_no_cfg = validate(
                model, hicarn_val, gt_val, res_mean, res_std,
                device, args.val_steps, cfg_scale=1.0
            )
            
            improved = ""
            if val_metrics['mse'] < best_mse:
                best_mse = val_metrics['mse']
                improved += " [best MSE]"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_metrics': val_metrics,
                    'res_std': res_std,
                    'res_mean': res_mean,
                    'cfg_scale': args.cfg_scale,
                    'config': {
                        'base_channels': args.base_channels,
                        'cond_dim': args.cond_dim,
                        'channel_mults': [1, 2, 4]
                    }
                }, output_dir / 'best_model.pt')
            
            if val_metrics['res_corr'] > best_res_corr:
                best_res_corr = val_metrics['res_corr']
                improved += " [best res_corr]"
            
            status = "✓" if val_metrics['improved'] else "⚠"
            
            print(f"\n  Epoch {epoch}: {status}")
            print(f"    With CFG={args.cfg_scale}:")
            print(f"      MSE={val_metrics['mse']:.6f} (HiCARN:{val_metrics['mse_hicarn']:.6f})")
            print(f"      PCC={val_metrics['pcc']:.4f} (HiCARN:{val_metrics['pcc_hicarn']:.4f})")
            print(f"      res_corr={val_metrics['res_corr']:.4f}")
            print(f"    Without CFG:")
            print(f"      MSE={val_no_cfg['mse']:.6f}, res_corr={val_no_cfg['res_corr']:.4f}")
            
            # Check if CFG is helping
            if val_metrics['res_corr'] > val_no_cfg['res_corr'] + 0.05:
                print(f"    ✓ CFG is helping! (res_corr +{val_metrics['res_corr'] - val_no_cfg['res_corr']:.4f})")
            
            if improved:
                print(f"    {improved}")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'val_no_cfg': val_no_cfg
            })
        
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'res_std': res_std,
                'res_mean': res_mean
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    print(f"\nHiCARN baseline: MSE={mse_baseline:.6f}, PCC={pcc_baseline:.4f}")
    print(f"Best refined:    MSE={best_mse:.6f}")
    print(f"Best res_corr:   {best_res_corr:.4f}")
    
    if best_mse < mse_baseline:
        print(f"\n✓ SUCCESS: Flow model improved over HiCARN!")
    else:
        print(f"\n⚠ Did not improve over HiCARN")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
