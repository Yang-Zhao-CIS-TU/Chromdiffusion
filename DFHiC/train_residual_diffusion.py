#!/usr/bin/env python3
"""
Vanilla Residual Diffusion Training Script

Self-contained training script for Hi-C refinement using residual diffusion.
Works with predictions from DeepHiC, SRHiC, HiCNN, or any other base model.

The diffusion model learns to predict the residual (GT - prediction) and
refines the base model's output.

Usage:
    # Train on SRHiC predictions
    python train_residual_diffusion.py \
        --pred_path predictions_srhic/norm/predictions.npy \
        --gt_path predictions_srhic/norm/ground_truth.npy \
        --output_dir checkpoints_diffusion_srhic \
        --device cuda:0

    # Train on DeepHiC predictions
    python train_residual_diffusion.py \
        --pred_path predictions_deephic/norm/predictions.npy \
        --gt_path predictions_deephic/norm/ground_truth.npy \
        --output_dir checkpoints_diffusion_deephic \
        --device cuda:0

    # Train on HiCNN predictions
    python train_residual_diffusion.py \
        --pred_path predictions_hicnn/norm/predictions.npy \
        --gt_path predictions_hicnn/norm/ground_truth.npy \
        --output_dir checkpoints_diffusion_hicnn \
        --device cuda:0
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import json
import math
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# Model: Residual Diffusion U-Net
# ============================================================================

def get_timestep_embedding(timesteps, embedding_dim):
    """Sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
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
        h = self.dropout(h)
        
        return h + self.shortcut(x)


class ResidualDiffusionUNet(nn.Module):
    """
    U-Net for residual diffusion.
    
    Takes:
        - x_t: Noisy residual at timestep t
        - t: Timestep
        - condition: Base model prediction (for conditioning)
    
    Predicts:
        - noise (epsilon)
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        time_emb_dim=256,
        dropout=0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input: noisy residual + condition
        self.init_conv = nn.Conv2d(in_channels * 2, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        channels = [base_channels]
        ch = base_channels
        
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResidualBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
            channels.append(ch)
            self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        
        # Middle
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for mult in reversed(channel_multipliers):
            out_ch = base_channels * mult
            self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            for _ in range(num_res_blocks):
                self.decoder_blocks.append(ResidualBlock(ch + channels.pop(), out_ch, time_emb_dim, dropout))
                ch = out_ch
        
        # Output
        self.final_norm = nn.GroupNorm(8, ch)
        self.final_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_t, timesteps, condition):
        """
        Args:
            x_t: Noisy residual [B, 1, H, W]
            timesteps: Timestep indices [B]
            condition: Base model prediction [B, 1, H, W]
        
        Returns:
            Predicted noise [B, 1, H, W]
        """
        # Time embedding
        t_emb = get_timestep_embedding(timesteps, self.base_channels)
        t_emb = self.time_mlp(t_emb)
        
        # Concatenate input and condition
        x = torch.cat([x_t, condition], dim=1)
        h = self.init_conv(x)
        
        # Encoder
        skips = [h]
        block_idx = 0
        for i, downsample in enumerate(self.downsample):
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, t_emb)
                block_idx += 1
            skips.append(h)
            h = downsample(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        block_idx = 0
        for i, upsample in enumerate(self.upsample):
            h = upsample(h)
            h = torch.cat([h, skips.pop()], dim=1)
            for _ in range(self.num_res_blocks):
                h = self.decoder_blocks[block_idx](h, t_emb)
                block_idx += 1
        
        # Output
        h = self.final_norm(h)
        h = F.silu(h)
        return self.final_conv(h)


# ============================================================================
# Diffusion Scheduler
# ============================================================================

class DDPMScheduler:
    """DDPM noise scheduler."""
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='linear'
    ):
        self.num_train_timesteps = num_train_timesteps
        
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == 'scaled_linear':
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        elif beta_schedule == 'cosine':
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x0, noise, timesteps):
        """Add noise to x0 at given timesteps."""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].to(x0.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].to(x0.device)
        
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self


# ============================================================================
# Data Loading
# ============================================================================

def ensure_nchw(x):
    """Ensure data is in NCHW format."""
    x = np.asarray(x)
    if x.ndim == 3:
        return x[:, None, :, :]
    elif x.ndim == 4 and x.shape[1] in [1, 3]:
        return x
    elif x.ndim == 4 and x.shape[-1] in [1, 3]:
        return np.transpose(x, (0, 3, 1, 2))
    else:
        raise ValueError(f"Cannot convert to NCHW: shape={x.shape}")


class ResidualNormalizer:
    """Normalizer for residuals (GT - prediction)."""
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
    
    def fit(self, residuals):
        self.mean = float(np.mean(residuals))
        self.std = float(np.std(residuals))
        self.std = max(self.std, 1e-6)
        print(f"  Residual normalizer: mean={self.mean:.6f}, std={self.std:.6f}")
    
    def transform(self, x):
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x):
        return x * self.std + self.mean
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'mean': self.mean, 'std': self.std}, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        normalizer = cls()
        normalizer.mean = data['mean']
        normalizer.std = data['std']
        return normalizer


class HiCRefinementDataset(Dataset):
    """Dataset for diffusion refinement training."""
    def __init__(self, predictions, ground_truth, normalizer, augment=False):
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.normalizer = normalizer
        self.augment = augment
        
        self.residuals = ground_truth - predictions
        self.residuals_norm = normalizer.transform(self.residuals)
    
    def __len__(self):
        return len(self.predictions)
    
    def __getitem__(self, idx):
        condition = torch.from_numpy(self.predictions[idx]).float()
        residual_norm = torch.from_numpy(self.residuals_norm[idx]).float()
        residual_raw = torch.from_numpy(self.residuals[idx]).float()
        gt = torch.from_numpy(self.ground_truth[idx]).float()
        
        if self.augment:
            if torch.rand(1) > 0.5:
                condition = torch.flip(condition, [-1])
                residual_norm = torch.flip(residual_norm, [-1])
                residual_raw = torch.flip(residual_raw, [-1])
                gt = torch.flip(gt, [-1])
            if torch.rand(1) > 0.5:
                condition = torch.flip(condition, [-2])
                residual_norm = torch.flip(residual_norm, [-2])
                residual_raw = torch.flip(residual_raw, [-2])
                gt = torch.flip(gt, [-2])
            if torch.rand(1) > 0.5:
                condition = condition.transpose(-1, -2)
                residual_norm = residual_norm.transpose(-1, -2)
                residual_raw = residual_raw.transpose(-1, -2)
                gt = gt.transpose(-1, -2)
        
        return {
            'condition': condition,
            'residual_norm': residual_norm,
            'residual_raw': residual_raw,
            'gt': gt
        }


def load_data(pred_path, gt_path):
    """Load predictions and ground truth."""
    print(f"  Loading predictions: {pred_path}")
    predictions = np.load(pred_path)
    predictions = ensure_nchw(predictions).astype(np.float32)
    print(f"    Shape: {predictions.shape}, Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    print(f"  Loading ground truth: {gt_path}")
    ground_truth = np.load(gt_path)
    ground_truth = ensure_nchw(ground_truth).astype(np.float32)
    print(f"    Shape: {ground_truth.shape}, Range: [{ground_truth.min():.4f}, {ground_truth.max():.4f}]")
    
    return predictions, ground_truth


def create_dataloaders(predictions, ground_truth, batch_size, train_split, num_workers, seed, augment=False):
    """Create train and validation dataloaders."""
    residuals = ground_truth - predictions
    normalizer = ResidualNormalizer()
    normalizer.fit(residuals)
    
    dataset = HiCRefinementDataset(predictions, ground_truth, normalizer, augment=augment)
    
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    print(f"  Train: {train_size}, Val: {val_size}")
    
    return train_loader, val_loader, normalizer


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        condition = batch['condition'].to(device)
        residual_norm = batch['residual_norm'].to(device)
        
        batch_size = condition.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(residual_norm)
        
        # Forward diffusion
        noisy_residual = scheduler.add_noise(residual_norm, noise, timesteps)
        
        # Predict noise
        noise_pred = model(noisy_residual, timesteps, condition)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, scheduler, normalizer, device, epoch, num_steps=50):
    """Validate with full denoising."""
    model.eval()
    
    total_loss = 0
    all_pcc_base = []
    all_pcc_refined = []
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Validating'):
        condition = batch['condition'].to(device)
        residual_norm = batch['residual_norm'].to(device)
        gt = batch['gt'].to(device)
        
        batch_size = condition.shape[0]
        
        # Compute loss
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(residual_norm)
        noisy_residual = scheduler.add_noise(residual_norm, noise, timesteps)
        noise_pred = model(noisy_residual, timesteps, condition)
        loss = F.mse_loss(noise_pred, noise)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Full denoising for metrics (first few batches only)
        if num_batches <= 5:
            x_t = torch.randn_like(residual_norm)
            
            step_size = scheduler.num_train_timesteps // num_steps
            timesteps_denoise = list(range(scheduler.num_train_timesteps - 1, -1, -step_size))
            
            for t in timesteps_denoise:
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                noise_pred = model(x_t, t_batch, condition)
                
                alpha = scheduler.alphas_cumprod[t]
                alpha_prev = scheduler.alphas_cumprod[max(t - step_size, 0)]
                
                sqrt_alpha = alpha.sqrt()
                sqrt_one_minus_alpha = (1 - alpha).sqrt()
                x0_pred = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
                
                if t > 0:
                    x_t = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred
                else:
                    x_t = x0_pred
            
            residual_pred = x_t.cpu().numpy() * normalizer.std + normalizer.mean
            refined = condition.cpu().numpy() + residual_pred
            
            for i in range(batch_size):
                cond_i = condition[i, 0].cpu().numpy().flatten()
                gt_i = gt[i, 0].cpu().numpy().flatten()
                refined_i = refined[i, 0].flatten()
                
                if np.std(cond_i) > 0 and np.std(gt_i) > 0:
                    all_pcc_base.append(pearsonr(cond_i, gt_i)[0])
                if np.std(refined_i) > 0 and np.std(gt_i) > 0:
                    all_pcc_refined.append(pearsonr(refined_i, gt_i)[0])
    
    metrics = {
        'loss': total_loss / num_batches,
        'pcc_base': np.mean(all_pcc_base) if all_pcc_base else 0,
        'pcc_refined': np.mean(all_pcc_refined) if all_pcc_refined else 0
    }
    
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, normalizer, config, filepath):
    """Save checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'normalizer_mean': normalizer.mean,
        'normalizer_std': normalizer.std,
        'config': config
    }
    torch.save(checkpoint, filepath)
    print(f"  Saved: {filepath}")


def plot_curves(history, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    
    axes[0].plot(epochs, train_loss, label='Train')
    axes[0].plot(epochs, val_loss, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    
    pcc_base = [h.get('pcc_base', 0) for h in history]
    pcc_refined = [h.get('pcc_refined', 0) for h in history]
    
    axes[1].plot(epochs, pcc_base, label='Base Model')
    axes[1].plot(epochs, pcc_refined, label='Refined')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PCC')
    axes[1].set_title('Validation PCC')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Vanilla Residual Diffusion')
    
    # Data
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to base model predictions (.npy)')
    parser.add_argument('--gt_path', type=str, required=True,
                       help='Path to ground truth (.npy)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_diffusion',
                       help='Output directory')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    
    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear',
                       choices=['linear', 'scaled_linear', 'cosine'])
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_split', type=float, default=0.9)
    parser.add_argument('--augment', action='store_true')
    
    # System
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load Data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    predictions, ground_truth = load_data(args.pred_path, args.gt_path)
    
    train_loader, val_loader, normalizer = create_dataloaders(
        predictions, ground_truth,
        batch_size=args.batch_size,
        train_split=args.train_split,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=args.augment
    )
    
    normalizer.save(output_dir / 'residual_normalizer.json')
    
    # Create Model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    model = ResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        num_res_blocks=args.num_res_blocks
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e6:.2f}M")
    
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Resume
    start_epoch = 1
    best_pcc = -float('inf')
    history = []
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
    
    # Training
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_metrics = validate(model, val_loader, scheduler, normalizer, device, epoch)
        
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'pcc_base': val_metrics['pcc_base'],
            'pcc_refined': val_metrics['pcc_refined'],
            'lr': current_lr
        })
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_metrics['loss']:.6f}")
        print(f"  PCC: Base={val_metrics['pcc_base']:.4f}, Refined={val_metrics['pcc_refined']:.4f}")
        
        # Save best
        if val_metrics['pcc_refined'] > best_pcc:
            best_pcc = val_metrics['pcc_refined']
            save_checkpoint(model, optimizer, epoch, val_metrics, normalizer, vars(args),
                          output_dir / 'best_model.pt')
            print(f"  ✓ New best PCC: {best_pcc:.4f}")
        
        # Periodic save
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics, normalizer, vars(args),
                          output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Final
    save_checkpoint(model, optimizer, args.epochs, val_metrics, normalizer, vars(args),
                  output_dir / 'final_model.pt')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    plot_curves(history, output_dir)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"  Best PCC: {best_pcc:.4f}")
    print(f"  Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
