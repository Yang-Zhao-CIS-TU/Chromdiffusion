#!/usr/bin/env python3
"""
Conditional Rectified Flow v2 - 从 HiCARN 流向 GT

核心修正：
- 之前错误：从 noise 流向 residual（采样时从纯噪声开始，没有信息）
- 现在正确：从 HiCARN 流向 GT（采样时从 HiCARN 开始，有条件信息）

数学：
- x_0 = HiCARN (起点，已知)
- x_1 = GT (终点，训练时已知)
- x_t = (1-t) * x_0 + t * x_1 = (1-t) * HiCARN + t * GT
- velocity v* = x_1 - x_0 = GT - HiCARN = residual

训练：
- 学习 v_θ(x_t, t) 预测 velocity = GT - HiCARN
- Loss: MSE(v_θ(x_t, t), GT - HiCARN)

推理：
- 从 x_0 = HiCARN 开始（不是从噪声！）
- ODE 积分：dx/dt = v_θ(x, t)
- 最终得到 x_1 ≈ GT

关键区别：
- 采样从 HiCARN 开始，不是从噪声开始
- 模型学的是"如何从 HiCARN 变到 GT"，不是"如何从噪声生成 residual"
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
from scipy import stats


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
# Simple UNet for Flow (输入 x_t，输出 velocity)
# ================================================================

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


class FlowUNet(nn.Module):
    """
    简单 UNet 预测 velocity
    
    输入: x_t [B, 1, H, W] - 当前状态（HiCARN 和 GT 之间的插值）
    输出: velocity [B, 1, H, W] - 预测的速度场
    
    注意：这里不需要额外的 condition 输入，因为 x_t 本身就包含了 HiCARN 的信息！
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=256
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
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
        
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, t):
        """
        x: [B, 1, H, W] - x_t (interpolation between HiCARN and GT)
        t: [B] - time (0 = HiCARN, 1 = GT)
        """
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

def train_epoch(model, optimizer, dataloader, device, epoch):
    """
    Rectified Flow training: HiCARN -> GT
    
    x_0 = HiCARN
    x_1 = GT
    x_t = (1-t) * HiCARN + t * GT
    target velocity = GT - HiCARN
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for hicarn, gt in pbar:
        hicarn = hicarn.to(device)  # x_0
        gt = gt.to(device)          # x_1
        
        batch_size = hicarn.shape[0]
        
        # Sample time t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)
        
        # Linear interpolation: x_t = (1-t) * x_0 + t * x_1
        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * hicarn + t_expand * gt
        
        # Target velocity: v* = x_1 - x_0 = GT - HiCARN
        target_velocity = gt - hicarn
        
        # Forward: predict velocity from x_t
        pred_velocity = model(x_t, t)
        
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
def sample_ode(model, hicarn, device, num_steps=10):
    """
    ODE sampling: 从 HiCARN 流向 GT
    
    关键：从 x_0 = HiCARN 开始！不是从噪声开始！
    
    dx/dt = v_θ(x, t)
    从 t=0 积分到 t=1
    """
    model.eval()
    batch_size = hicarn.shape[0]
    
    # 从 HiCARN 开始！
    x = hicarn.clone()
    
    # Time steps from 0 to 1
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = i / num_steps
        t_batch = torch.full((batch_size,), t, device=device)
        
        # Predict velocity
        v = model(x, t_batch)
        
        # Euler step: x_next = x + dt * v
        x = x + dt * v
    
    return x


@torch.no_grad()
def validate(model, hicarn_val, gt_val, device, num_steps=10, seed=42):
    """Validate"""
    model.eval()
    torch.manual_seed(seed)
    
    n = min(500, len(hicarn_val))
    hicarn = torch.from_numpy(hicarn_val[:n]).float().to(device)
    gt = torch.from_numpy(gt_val[:n]).float().to(device)
    
    # Sample: flow from HiCARN to predicted GT
    pred = sample_ode(model, hicarn, device, num_steps)
    pred = torch.clamp(pred, -5, 5)
    
    # Metrics
    mse = F.mse_loss(pred, gt).item()
    mse_hicarn = F.mse_loss(hicarn, gt).item()
    
    pred_np = pred.cpu().numpy().flatten()
    gt_np = gt.cpu().numpy().flatten()
    hicarn_np = hicarn.cpu().numpy().flatten()
    
    pcc, _ = stats.pearsonr(pred_np, gt_np)
    pcc_hicarn, _ = stats.pearsonr(hicarn_np, gt_np)
    
    # Residual correlation
    pred_residual = (pred - hicarn).cpu().numpy()
    ideal_residual = (gt - hicarn).cpu().numpy()
    
    res_corr, _ = stats.pearsonr(pred_residual.flatten(), ideal_residual.flatten())
    
    return {
        'mse': mse,
        'pcc': float(pcc),
        'mse_hicarn': mse_hicarn,
        'pcc_hicarn': float(pcc_hicarn),
        'res_corr': float(res_corr),
        'pred_residual_std': float(pred_residual.std()),
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
    parser.add_argument('--output_dir', type=str, default='checkpoints_flow_v2')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=10,
                       help='Number of ODE steps for sampling')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Load data
    # ================================================================
    print("\n" + "="*80)
    print("RECTIFIED FLOW v2: HiCARN -> GT")
    print("="*80)
    print("关键改进：从 HiCARN 开始流向 GT，不是从噪声开始！")
    
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
    print(f"\nResidual (GT - HiCARN) statistics:")
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
    
    model = FlowUNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4)
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
    print("模型学习：从 x_t 预测 velocity = GT - HiCARN")
    print("采样时：从 HiCARN 开始，沿 velocity 积分到 GT")
    
    best_mse = mse_baseline
    best_res_corr = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, optimizer, train_loader, device, epoch)
        
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate(
                model, hicarn_val, gt_val, device, args.num_steps
            )
            
            improved = ""
            if val_metrics['mse'] < best_mse:
                best_mse = val_metrics['mse']
                improved += " [best MSE]"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_metrics': val_metrics,
                    'config': {
                        'base_channels': args.base_channels,
                        'channel_mults': [1, 2, 4],
                        'num_steps': args.num_steps
                    }
                }, output_dir / 'best_model.pt')
            
            if val_metrics['res_corr'] > best_res_corr:
                best_res_corr = val_metrics['res_corr']
                improved += " [best res_corr]"
            
            status = "✓" if val_metrics['improved'] else "⚠"
            
            print(f"\n  Epoch {epoch}: {status}")
            print(f"    MSE={val_metrics['mse']:.6f} (HiCARN:{val_metrics['mse_hicarn']:.6f})")
            print(f"    PCC={val_metrics['pcc']:.4f} (HiCARN:{val_metrics['pcc_hicarn']:.4f})")
            print(f"    res_corr={val_metrics['res_corr']:.4f} ← 应该接近 1.0！")
            print(f"    pred_res_std={val_metrics['pred_residual_std']:.4f} (ideal:{val_metrics['ideal_residual_std']:.4f})")
            
            if improved:
                print(f"    {improved}")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_metrics': val_metrics
            })
        
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
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
    
    if best_res_corr > 0.5:
        print(f"✓ res_corr > 0.5: Model learned meaningful residual pattern")
    else:
        print(f"⚠ res_corr low: Model may not be learning the right pattern")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
