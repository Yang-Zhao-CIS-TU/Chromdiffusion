#!/usr/bin/env python3
"""
Test/Inference Script for Vanilla Residual Diffusion

Evaluates in both NORMALIZED and RAW spaces.
Uses same denormalization method as test_v15.py

Usage:
    python test_residual_diffusion.py \
        --checkpoint checkpoints_diffusion_deephic/best_model.pt \
        --pred_dir predictions_deephic/norm \
        --gt_dir /data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40Locations \
        --preprocess_file checkpoints_deephic/preprocessor.pt \
        --chromosomes chr18 chr19 chr20 chr21 chr22 \
        --output_dir test_results_diffusion_deephic \
        --device cuda:0
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# Model
# ============================================================================

def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.dropout(F.silu(self.norm2(self.conv2(h))))
        return h + self.shortcut(x)


class ResidualDiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 channel_multipliers=(1, 2, 4), num_res_blocks=2, time_emb_dim=256, dropout=0.1):
        super().__init__()
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.init_conv = nn.Conv2d(in_channels * 2, base_channels, 3, padding=1)
        
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch = base_channels
        encoder_channels = [ch]
        
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResidualBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                encoder_channels.append(ch)
            self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for mult in reversed(channel_multipliers):
            out_ch = base_channels * mult
            self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            for _ in range(num_res_blocks):
                skip_ch = encoder_channels.pop()
                self.decoder_blocks.append(ResidualBlock(ch + skip_ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
        
        self.final_norm = nn.GroupNorm(8, ch)
        self.final_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(self, x_t, timesteps, condition):
        t_emb = self.time_mlp(get_timestep_embedding(timesteps, self.base_channels))
        h = self.init_conv(torch.cat([x_t, condition], dim=1))
        
        skips = [h]
        block_idx = 0
        for downsample in self.downsample:
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, t_emb)
                skips.append(h)
                block_idx += 1
            h = downsample(h)
        
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)
        
        block_idx = 0
        for upsample in self.upsample:
            h = upsample(h)
            for _ in range(self.num_res_blocks):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.decoder_blocks[block_idx](h, t_emb)
                block_idx += 1
        
        return self.final_conv(F.silu(self.final_norm(h)))


# ============================================================================
# Scheduler
# ============================================================================

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear'):
        self.num_train_timesteps = num_train_timesteps
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == 'cosine':
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self


# ============================================================================
# Preprocessor (same as test_v15.py)
# ============================================================================

class RobustHiCPreprocessor:
    """Preprocessor matching test_v15.py exactly."""
    def __init__(self):
        self.Y_mean = None
        self.Y_std = None
    
    def postprocess(self, Y_norm):
        """Convert normalized to raw space (same as test_v15.py)."""
        Y_norm = np.clip(Y_norm, -5, 5)
        Y_log = Y_norm * self.Y_std + self.Y_mean
        Y_counts = np.expm1(Y_log)
        return np.maximum(Y_counts, 0.0)


def load_preprocessor(path):
    """Load preprocessor (same as test_v15.py)."""
    data = torch.load(path, map_location='cpu')
    
    preproc = RobustHiCPreprocessor()
    
    if isinstance(data, dict):
        preproc.Y_mean = data.get('Y_mean', data.get('y_mean', 0.0))
        preproc.Y_std = data.get('Y_std', data.get('y_std', 1.0))
    elif hasattr(data, 'Y_mean'):
        preproc.Y_mean = data.Y_mean
        preproc.Y_std = data.Y_std
    else:
        preproc.Y_mean = getattr(data, 'y_mean', 0.0)
        preproc.Y_std = getattr(data, 'y_std', 1.0)
    
    print(f"  Preprocessor: Y_mean={preproc.Y_mean:.4f}, Y_std={preproc.Y_std:.4f}")
    return preproc


class ResidualNormalizer:
    """Normalizer for diffusion residuals."""
    def __init__(self, mean=0.0, std=1.0):
        self.mean, self.std = mean, std
    
    def inverse_transform(self, x):
        return x * self.std + self.mean


# ============================================================================
# Metrics
# ============================================================================

class SSIM(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / 18) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        self.window = gauss.unsqueeze(1).mm(gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    def forward(self, img1, img2):
        window = self.window.to(img1.device).type_as(img1)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2)
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2) - mu1 ** 2
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2) - mu2 ** 2
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2) - mu1 * mu2
        C1, C2 = 0.0001, 0.0009
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim.mean()


def compute_metrics(pred, gt, ssim_module=None, device=None):
    """Compute evaluation metrics."""
    pred_flat, gt_flat = pred.flatten(), gt.flatten()
    metrics = {'mse': float(np.mean((pred - gt) ** 2))}
    
    if np.std(pred_flat) > 0 and np.std(gt_flat) > 0:
        metrics['pcc'] = float(pearsonr(pred_flat, gt_flat)[0])
        metrics['spc'] = float(spearmanr(pred_flat, gt_flat)[0])
    else:
        metrics['pcc'], metrics['spc'] = 0.0, 0.0
    
    if metrics['mse'] > 0:
        max_val = max(gt.max(), pred.max(), 1e-6)
        metrics['psnr'] = 20 * np.log10(max_val / np.sqrt(metrics['mse']))
        signal_power = np.mean(gt ** 2)
        metrics['snr'] = 10 * np.log10(signal_power / metrics['mse'])
    else:
        metrics['psnr'], metrics['snr'] = float('inf'), float('inf')
    
    if ssim_module is not None and device is not None:
        pred_t = torch.from_numpy(pred).float().to(device)
        gt_t = torch.from_numpy(gt).float().to(device)
        if pred_t.dim() == 3:
            pred_t, gt_t = pred_t.unsqueeze(0), gt_t.unsqueeze(0)
        metrics['ssim'] = float(ssim_module(pred_t, gt_t).cpu())
    
    return metrics


# ============================================================================
# Utilities
# ============================================================================

def ensure_nchw(x):
    """Ensure data is in NCHW format."""
    x = np.asarray(x)
    if x.ndim == 3:
        return x[:, None, :, :]
    elif x.ndim == 4:
        return x if x.shape[1] in [1, 3] else np.transpose(x, (0, 3, 1, 2))
    raise ValueError(f"Cannot convert to NCHW: shape={x.shape}")


@torch.no_grad()
def denoise(model, scheduler, condition, device, num_steps=50, batch_size=32):
    """Run full denoising."""
    model.eval()
    N = condition.shape[0]
    all_residuals = []
    
    for start_idx in tqdm(range(0, N, batch_size), desc='Denoising'):
        end_idx = min(start_idx + batch_size, N)
        cond_batch = torch.from_numpy(condition[start_idx:end_idx]).float().to(device)
        B = cond_batch.shape[0]
        x_t = torch.randn_like(cond_batch)
        
        step_size = scheduler.num_train_timesteps // num_steps
        for t in range(scheduler.num_train_timesteps - 1, -1, -step_size):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = model(x_t, t_batch, cond_batch)
            
            alpha = scheduler.alphas_cumprod[t]
            alpha_prev = scheduler.alphas_cumprod[max(t - step_size, 0)]
            x0_pred = (x_t - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
            x_t = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred if t > 0 else x0_pred
        
        all_residuals.append(x_t.cpu().numpy())
    
    return np.concatenate(all_residuals, axis=0)


def visualize_samples(base, gt, refined, output_path, title_prefix='', num_samples=4):
    """Visualize samples."""
    num_samples = min(num_samples, len(base))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(num_samples):
        vmax = max(gt[i].max(), refined[i].max(), base[i].max())
        for j, (img, title) in enumerate([(base[i], 'Base'), (gt[i], 'GT'), (refined[i], 'Refined')]):
            axes[i, j].imshow(img.squeeze(), cmap='Reds', vmin=0, vmax=vmax)
            axes[i, j].set_title(f'{title_prefix}{title}')
            axes[i, j].axis('off')
        
        diff = refined[i].squeeze() - gt[i].squeeze()
        vmax_diff = max(abs(diff.min()), abs(diff.max()), 1e-6)
        axes[i, 3].imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
        axes[i, 3].set_title(f'{title_prefix}Diff')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Residual Diffusion (Norm + Raw Space)')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--preprocess_file', type=str, required=True,
                       help='Path to preprocessor.pt (same as test_v15.py)')
    
    # Normalized predictions
    parser.add_argument('--pred_path', type=str, default=None)
    parser.add_argument('--pred_dir', type=str, default=None)
    
    # Ground truth
    parser.add_argument('--gt_dir', type=str, default=None,
                       help='Directory with raw GT files (hr_test_{chrom}.npy)')
    
    parser.add_argument('--chromosomes', type=str, nargs='+', 
                       default=['chr18', 'chr19', 'chr20', 'chr21', 'chr22'])
    parser.add_argument('--ratio', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='./test_results_diffusion')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'norm').mkdir(exist_ok=True)
    (out / 'raw').mkdir(exist_ok=True)
    
    # ========================================================================
    # Load Model
    # ========================================================================
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    res_normalizer = ResidualNormalizer(
        checkpoint.get('normalizer_mean', 0), 
        checkpoint.get('normalizer_std', 1)
    )
    print(f"  Residual Normalizer: mean={res_normalizer.mean:.6f}, std={res_normalizer.std:.6f}")
    
    model = ResidualDiffusionUNet(
        base_channels=config.get('base_channels', 64),
        channel_multipliers=tuple(config.get('channel_multipliers', [1, 2, 4])),
        num_res_blocks=config.get('num_res_blocks', 2)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    scheduler = DDPMScheduler(
        num_train_timesteps=config.get('num_timesteps', 1000),
        beta_schedule=config.get('beta_schedule', 'linear')
    ).to(device)
    
    # ========================================================================
    # Load Preprocessor (same as test_v15.py)
    # ========================================================================
    print("\n" + "="*70)
    print("LOADING PREPROCESSOR")
    print("="*70)
    
    preproc = load_preprocessor(args.preprocess_file)
    
    # ========================================================================
    # Process Chromosomes
    # ========================================================================
    print("\n" + "="*70)
    print("PROCESSING")
    print("="*70)
    
    all_results = {'norm': {}, 'raw': {}}
    ssim_module = SSIM().to(device)
    
    for chrom in args.chromosomes:
        print(f"\n>>> {chrom}")
        
        # Find prediction file (normalized)
        pred_path = None
        for pattern in [
            f'predictions_{chrom}.npy',
            f'predictions_norm_{chrom}.npy',
            f'{chrom}/predictions_norm.npy',
            f'{chrom}/predictions.npy',
        ]:
            if args.pred_dir and (Path(args.pred_dir) / pattern).exists():
                pred_path = Path(args.pred_dir) / pattern
                break
        
        if pred_path is None:
            print(f"  Skip: Predictions not found")
            continue
        
        # Find GT file - PRIORITIZE raw GT from gt_dir
        gt_path = None
        gt_is_raw = False
        
        # Check for raw GT in gt_dir FIRST (has chromosome-specific ranges)
        if args.gt_dir:
            for pattern in [
                f'hr_test_{chrom}.npy',
                f'hr_test_{chrom}_ratio{args.ratio}.npy',
            ]:
                if (Path(args.gt_dir) / pattern).exists():
                    gt_path = Path(args.gt_dir) / pattern
                    gt_is_raw = True
                    break
        
        # Fallback to normalized GT in pred_dir (only if raw GT not found)
        if gt_path is None and args.pred_dir:
            for pattern in [
                f'ground_truth_{chrom}.npy',
                f'{chrom}/ground_truth.npy',
            ]:
                if (Path(args.pred_dir) / pattern).exists():
                    gt_path = Path(args.pred_dir) / pattern
                    gt_is_raw = False
                    break
        
        if gt_path is None:
            print(f"  Skip: GT not found")
            continue
        
        # Load data
        base_norm = ensure_nchw(np.load(pred_path)).astype(np.float32)
        gt_data = ensure_nchw(np.load(gt_path)).astype(np.float32)
        
        print(f"  Base: {base_norm.shape} from {pred_path}")
        print(f"  GT: {gt_data.shape} from {gt_path}")
        
        # Determine GT space and load appropriately
        if gt_is_raw or gt_data.max() > 50:
            gt_raw = gt_data
            print(f"    -> RAW space (max={gt_data.max():.2f})")
        else:
            # GT is already normalized - convert to raw using global preprocessor
            gt_norm_original = gt_data
            gt_raw = preproc.postprocess(gt_norm_original)
            print(f"    -> NORM space (max={gt_data.max():.4f})")
        
        # Run diffusion refinement (in normalized space)
        residuals_norm = denoise(model, scheduler, base_norm, device, args.num_steps, args.batch_size)
        refined_norm = base_norm + res_normalizer.inverse_transform(residuals_norm)
        
        print(f"  Refined (norm): [{refined_norm.min():.4f}, {refined_norm.max():.4f}]")
        
        # Compute per-chromosome stats from raw GT
        gt_raw_log = np.log1p(gt_raw)
        chrom_mean = float(np.mean(gt_raw_log))
        chrom_std = float(np.std(gt_raw_log))
        chrom_std = max(chrom_std, 1e-6)
        
        # Normalize GT using per-chromosome stats for fair NORM space comparison
        # This puts GT in the same normalized space as predictions
        gt_norm = (gt_raw_log - chrom_mean) / chrom_std
        gt_norm = gt_norm.astype(np.float32)
        
        print(f"  Per-chrom stats: mean={chrom_mean:.4f}, std={chrom_std:.4f}")
        print(f"  GT (norm): [{gt_norm.min():.4f}, {gt_norm.max():.4f}], mean={gt_norm.mean():.4f}")
        print(f"  Base (norm): mean={base_norm.mean():.4f}")
        print(f"  Refined (norm): mean={refined_norm.mean():.4f}")
        
        # For RAW space: use GLOBAL preprocessor (as trained)
        base_raw = preproc.postprocess(base_norm)
        refined_raw = preproc.postprocess(refined_norm)
        
        print(f"  Base (raw): [{base_raw.min():.2f}, {base_raw.max():.2f}], mean={base_raw.mean():.2f}")
        print(f"  Refined (raw): [{refined_raw.min():.2f}, {refined_raw.max():.2f}], mean={refined_raw.mean():.2f}")
        print(f"  GT (raw): [{gt_raw.min():.2f}, {gt_raw.max():.2f}], mean={gt_raw.mean():.2f}")
        
        # Save outputs
        np.save(out / 'norm' / f"base_{chrom}.npy", base_norm)
        np.save(out / 'norm' / f"refined_{chrom}.npy", refined_norm)
        np.save(out / 'norm' / f"gt_{chrom}.npy", gt_norm)
        np.save(out / 'raw' / f"base_{chrom}.npy", base_raw)
        np.save(out / 'raw' / f"refined_{chrom}.npy", refined_raw)
        np.save(out / 'raw' / f"gt_{chrom}.npy", gt_raw)
        
        # Compute metrics
        for space, (base, refined, gt) in [
            ('NORM', (base_norm, refined_norm, gt_norm)),
            ('RAW', (base_raw, refined_raw, gt_raw))
        ]:
            if gt is None:
                continue
            
            metrics_base = compute_metrics(base, gt, ssim_module, device)
            metrics_refined = compute_metrics(refined, gt, ssim_module, device)
            
            all_results[space.lower()][chrom] = {
                'base': metrics_base,
                'refined': metrics_refined,
                'n_samples': len(base)
            }
            
            print(f"\n  [{space}]")
            print(f"  {'Metric':<8} {'Base':>10} {'Refined':>10} {'Δ':>10}")
            print(f"  {'-'*40}")
            for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
                bv, rv = metrics_base.get(m, 0), metrics_refined.get(m, 0)
                d = (bv - rv) if m == 'mse' else (rv - bv)
                print(f"  {m.upper():<8} {bv:>10.4f} {rv:>10.4f} {d:>+10.4f}")
        
        # Visualize
        visualize_samples(base_norm, gt_norm, refined_norm, 
                        out / 'norm' / f'samples_{chrom}.png', 'NORM: ')
        visualize_samples(base_raw, gt_raw, refined_raw, 
                         out / 'raw' / f'samples_{chrom}.png', 'RAW: ')
    
    # ========================================================================
    # Overall Summary
    # ========================================================================
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    for space in ['norm', 'raw']:
        if not all_results[space]:
            continue
        
        total = sum(r['n_samples'] for r in all_results[space].values())
        if total == 0:
            continue
        
        metrics_list = ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']
        avg_base = {m: sum(r['base'].get(m, 0) * r['n_samples'] for r in all_results[space].values()) / total 
                    for m in metrics_list}
        avg_ref = {m: sum(r['refined'].get(m, 0) * r['n_samples'] for r in all_results[space].values()) / total 
                   for m in metrics_list}
        
        print(f"\n[{space.upper()}] - {total} samples")
        print(f"{'Metric':<8} {'Base':>15} {'Refined':>15} {'Improv%':>10}")
        print(f"{'-'*50}")
        
        for m in metrics_list:
            bv, rv = avg_base[m], avg_ref[m]
            if m == 'mse':
                imp = ((bv - rv) / bv * 100) if bv != 0 else 0
            else:
                imp = ((rv - bv) / abs(bv) * 100) if bv != 0 else 0
            print(f"{m.upper():<8} {bv:>15.4f} {rv:>15.4f} {imp:>+9.2f}%")
        
        all_results[f'summary_{space}'] = {
            'total_samples': total,
            'base': avg_base,
            'refined': avg_ref
        }
    
    # Save results
    with open(out / 'evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results: {out / 'evaluation_results.json'}")
    print(f"Outputs: {out}/norm/, {out}/raw/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
