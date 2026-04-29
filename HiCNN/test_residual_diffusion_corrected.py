#!/usr/bin/env python3
"""
Test/Inference Script for Vanilla Residual Diffusion - CORRECTED VERSION

Key changes:
- Predictions: Convert to raw using GLOBAL preprocessor (as trained)
- GT: Keep in original raw space (no conversion)
- NORM space: Fair comparison with per-chromosome normalized GT

Usage:
    python test_residual_diffusion.py \
        --checkpoint checkpoints_diffusion_hicnn/best_model.pt \
        --pred_dir predictions_hicnn/norm \
        --gt_dir /data/.../40x40Locations \
        --preprocess_file checkpoints_hicnn/preprocessor.pt \
        --chromosomes chr18 chr19 chr20 chr21 chr22 \
        --output_dir test_results_diffusion_hicnn \
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
# Model Architecture
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


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, 
                 channel_multipliers=(1, 2, 4), num_res_blocks=2, time_emb_dim=256):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch = base_channels
        
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResBlock(ch, out_ch, time_emb_dim))
                ch = out_ch
            self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        
        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for mult in reversed(channel_multipliers):
            out_ch = base_channels * mult
            self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            for _ in range(num_res_blocks + 1):
                self.decoder_blocks.append(ResBlock(ch, out_ch, time_emb_dim))
                ch = out_ch
        
        self.final_norm = nn.GroupNorm(8, ch)
        self.final_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.init_conv.out_channels)
        t_emb = self.time_mlp(t_emb)
        
        h = self.init_conv(x)
        
        # Encoder
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, t_emb)
            if (i + 1) % 2 == 0 and (i + 1) // 2 <= len(self.downsample):
                h = self.downsample[(i + 1) // 2 - 1](h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        for i, block in enumerate(self.decoder_blocks):
            if i % 3 == 0 and i // 3 < len(self.upsample):
                h = self.upsample[i // 3](h)
            h = block(h, t_emb)
        
        h = F.silu(self.final_norm(h))
        return self.final_conv(h)


# ============================================================================
# Scheduler
# ============================================================================

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear'):
        self.num_train_timesteps = num_train_timesteps
        
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
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
# Preprocessor
# ============================================================================

class RobustHiCPreprocessor:
    def __init__(self):
        self.Y_mean = None
        self.Y_std = None
    
    def postprocess(self, Y_norm):
        """Convert normalized to raw using GLOBAL stats"""
        Y_norm = np.clip(Y_norm, -5, 5)
        Y_log = Y_norm * self.Y_std + self.Y_mean
        Y_counts = np.expm1(Y_log)
        return np.maximum(Y_counts, 0.0)


def load_preprocessor(path):
    """Load preprocessor - handles both dict and object formats"""
    preprocessor_data = torch.load(path, map_location='cpu')
    
    if isinstance(preprocessor_data, dict):
        Y_mean = preprocessor_data.get('Y_mean', preprocessor_data.get('y_mean', 0.0))
        Y_std = preprocessor_data.get('Y_std', preprocessor_data.get('y_std', 1.0))
    else:
        Y_mean = getattr(preprocessor_data, 'Y_mean', 0.0)
        Y_std = getattr(preprocessor_data, 'Y_std', 1.0)
    
    preprocessor = RobustHiCPreprocessor()
    preprocessor.Y_mean = Y_mean
    preprocessor.Y_std = Y_std
    
    print(f"  Preprocessor: Y_mean={preprocessor.Y_mean:.4f}, Y_std={preprocessor.Y_std:.4f}")
    return preprocessor


# ============================================================================
# Metrics
# ============================================================================

from math import exp

class SSIM(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / 18) for x in range(window_size)])
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


def compute_metrics(pred, gt, ssim_module, device):
    """Compute all metrics"""
    metrics = {}
    N = pred.shape[0]
    
    # Flatten for correlation metrics
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    
    pcc_list, spc_list = [], []
    for i in range(N):
        p, g = pred_flat[i], gt_flat[i]
        if np.std(p) > 0 and np.std(g) > 0:
            pcc_list.append(pearsonr(p, g)[0])
            spc_list.append(spearmanr(p, g)[0])
    
    metrics['pcc'] = float(np.mean(pcc_list)) if pcc_list else 0.0
    metrics['spc'] = float(np.mean(spc_list)) if spc_list else 0.0
    
    # MSE
    mse = np.mean((pred - gt) ** 2)
    metrics['mse'] = float(mse)
    
    # PSNR
    if mse > 0:
        max_val = max(gt.max(), pred.max(), 1e-6)
        metrics['psnr'] = float(20 * np.log10(max_val / np.sqrt(mse)))
    else:
        metrics['psnr'] = 100.0
    
    # SNR
    if mse > 0:
        signal_power = np.mean(gt ** 2)
        metrics['snr'] = float(10 * np.log10(signal_power / mse))
    else:
        metrics['snr'] = 100.0
    
    # SSIM
    pred_t = torch.from_numpy(pred).float().to(device)
    gt_t = torch.from_numpy(gt).float().to(device)
    
    if pred_t.dim() == 3:
        pred_t = pred_t.unsqueeze(1)
        gt_t = gt_t.unsqueeze(1)
    
    with torch.no_grad():
        metrics['ssim'] = float(ssim_module(pred_t, gt_t).cpu())
    
    return metrics


# ============================================================================
# Denoising
# ============================================================================

def ensure_nchw(x):
    x = np.asarray(x)
    if x.ndim == 3:
        return x[:, None, :, :]
    elif x.ndim == 4:
        return x if x.shape[1] in [1, 3] else np.transpose(x, (0, 3, 1, 2))
    raise ValueError(f"Cannot convert to NCHW: shape={x.shape}")


@torch.no_grad()
def denoise(model, scheduler, base_norm, device, num_steps=50, batch_size=64):
    """Run DDPM denoising"""
    model.eval()
    N = base_norm.shape[0]
    all_residuals = []
    
    step_size = scheduler.num_train_timesteps // num_steps
    
    for start_idx in tqdm(range(0, N, batch_size), desc='Denoising'):
        end_idx = min(start_idx + batch_size, N)
        B = end_idx - start_idx
        
        # Start from noise
        x_t = torch.randn(B, 1, 40, 40, device=device)
        
        # Denoise
        for t in range(scheduler.num_train_timesteps - 1, -1, -step_size):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = model(x_t, t_batch)
            
            alpha_t = scheduler.alphas_cumprod[t]
            alpha_prev = scheduler.alphas_cumprod[max(t - step_size, 0)] if t > 0 else torch.tensor(1.0, device=device)
            
            # Predict x0
            x0 = (x_t - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            
            if t > 0:
                # DDIM step
                x_t = alpha_prev.sqrt() * x0 + (1 - alpha_prev).sqrt() * noise_pred
            else:
                x_t = x0
        
        all_residuals.append(x_t.cpu().numpy())
    
    return np.concatenate(all_residuals, axis=0)


# ============================================================================
# Visualization
# ============================================================================

def visualize_samples(base, gt, refined, output_path, title_prefix=''):
    """Visualize sample predictions"""
    try:
        n_samples = min(4, base.shape[0])
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            vmin = min(base[i, 0].min(), gt[i, 0].min(), refined[i, 0].min())
            vmax = max(base[i, 0].max(), gt[i, 0].max(), refined[i, 0].max())
            
            if vmax > vmin:
                axes[i, 0].imshow(base[i, 0], cmap='Reds', vmin=vmin, vmax=vmax)
                axes[i, 0].set_title(f'{title_prefix}Base')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(gt[i, 0], cmap='Reds', vmin=vmin, vmax=vmax)
                axes[i, 1].set_title(f'{title_prefix}GT')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(refined[i, 0], cmap='Reds', vmin=vmin, vmax=vmax)
                axes[i, 2].set_title(f'{title_prefix}Refined')
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  Warning: Visualization failed: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--preprocess_file', type=str, required=True)
    parser.add_argument('--chromosomes', nargs='+', default=['chr18', 'chr19', 'chr20', 'chr21', 'chr22'])
    parser.add_argument('--output_dir', type=str, default='test_results')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'norm').mkdir(exist_ok=True)
    (out / 'raw').mkdir(exist_ok=True)
    
    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
    res_normalizer_mean = checkpoint.get('normalizer_mean', 0.0)
    res_normalizer_std = checkpoint.get('normalizer_std', 1.0)
    
    print(f"  Residual Normalizer: mean={res_normalizer_mean:.6f}, std={res_normalizer_std:.6f}")
    
    model = SimpleUNet(
        base_channels=config.get('base_channels', 64),
        channel_multipliers=config.get('channel_multipliers', [1, 2, 4]),
        num_res_blocks=config.get('num_res_blocks', 2)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    scheduler = DDPMScheduler(
        num_train_timesteps=config.get('num_timesteps', 1000),
        beta_schedule=config.get('beta_schedule', 'linear')
    ).to(device)
    
    # Load preprocessor
    print("\n" + "="*70)
    print("LOADING PREPROCESSOR")
    print("="*70)
    preproc = load_preprocessor(args.preprocess_file)
    
    # Process chromosomes
    print("\n" + "="*70)
    print("PROCESSING")
    print("="*70)
    
    all_results = {'norm': {}, 'raw': {}}
    ssim_module = SSIM().to(device)
    
    for chrom in args.chromosomes:
        print(f"\n>>> {chrom}")
        
        # Find prediction file
        pred_path = None
        for pattern in [
            f'predictions_{chrom}.npy',
            f'predictions_norm_{chrom}.npy',
            f'{chrom}/predictions_norm.npy',
            f'{chrom}/predictions.npy',
        ]:
            if (Path(args.pred_dir) / pattern).exists():
                pred_path = Path(args.pred_dir) / pattern
                break
        
        if pred_path is None:
            print(f"  Skip: predictions not found")
            continue
        
        # Load GT - PRIORITIZE raw GT from gt_dir
        gt_path = None
        gt_is_raw = False
        
        if args.gt_dir:
            for pattern in [f'hr_test_{chrom}.npy', f'{chrom}.npy']:
                test_path = Path(args.gt_dir) / pattern
                if test_path.exists():
                    gt_path = test_path
                    gt_is_raw = True
                    break
        
        # Fallback to normalized GT
        if gt_path is None:
            for pattern in [f'ground_truth_{chrom}.npy', f'{chrom}/ground_truth.npy']:
                test_path = Path(args.pred_dir) / pattern
                if test_path.exists():
                    gt_path = test_path
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
        
        # Determine if GT is raw or normalized
        if gt_is_raw or gt_data.max() > 50:
            gt_raw = gt_data
            print(f"    -> RAW space (max={gt_data.max():.2f})")
        else:
            gt_raw = preproc.postprocess(gt_data)
            print(f"    -> NORMALIZED space (max={gt_data.max():.4f})")
        
        # Run diffusion refinement
        residuals_norm = denoise(model, scheduler, base_norm, device, args.num_steps, args.batch_size)
        residuals_denorm = residuals_norm * res_normalizer_std + res_normalizer_mean
        refined_norm = base_norm + residuals_denorm
        
        print(f"  Refined (norm): [{refined_norm.min():.4f}, {refined_norm.max():.4f}]")
        
        # Normalize GT using per-chromosome stats for FAIR NORM SPACE COMPARISON
        gt_raw_log = np.log1p(gt_raw)
        chrom_mean = float(np.mean(gt_raw_log))
        chrom_std = float(np.std(gt_raw_log))
        chrom_std = max(chrom_std, 1e-6)
        
        gt_norm = ((gt_raw_log - chrom_mean) / chrom_std).astype(np.float32)
        
        print(f"  Per-chrom stats: mean={chrom_mean:.4f}, std={chrom_std:.4f}")
        print(f"  GT (norm): [{gt_norm.min():.4f}, {gt_norm.max():.4f}], mean={gt_norm.mean():.4f}")
        print(f"  Base (norm): mean={base_norm.mean():.4f}")
        print(f"  Refined (norm): mean={refined_norm.mean():.4f}")
        
        # Convert predictions to RAW using GLOBAL preprocessor (as trained)
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
        try:
            visualize_samples(base_norm, gt_norm, refined_norm, 
                            out / 'norm' / f'samples_{chrom}.png', 'NORM: ')
            visualize_samples(base_raw, gt_raw, refined_raw, 
                             out / 'raw' / f'samples_{chrom}.png', 'RAW: ')
        except Exception as e:
            print(f"  Warning: Visualization failed: {e}")
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    for space in ['norm', 'raw']:
        if not all_results[space]:
            continue
        
        total = sum(r['n_samples'] for r in all_results[space].values())
        print(f"\n[{space.upper()}] - {total} samples")
        
        # Weighted averages
        base_metrics = {}
        refined_metrics = {}
        
        for metric in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
            base_vals = []
            refined_vals = []
            weights = []
            
            for chrom_results in all_results[space].values():
                n = chrom_results['n_samples']
                base_vals.append(chrom_results['base'].get(metric, 0) * n)
                refined_vals.append(chrom_results['refined'].get(metric, 0) * n)
                weights.append(n)
            
            if sum(weights) > 0:
                base_metrics[metric] = sum(base_vals) / sum(weights)
                refined_metrics[metric] = sum(refined_vals) / sum(weights)
        
        print(f"{'Metric':<8} {'Base':>10} {'Refined':>10} {'Improv':>10}")
        print(f"{'-'*40}")
        for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
            bv = base_metrics.get(m, 0)
            rv = refined_metrics.get(m, 0)
            if m == 'mse':
                imp = f"{(bv - rv) / bv * 100:+.2f}%" if bv != 0 else "N/A"
            else:
                imp = f"{(rv - bv) / abs(bv) * 100:+.2f}%" if bv != 0 else "N/A"
            print(f"{m.upper():<8} {bv:>10.4f} {rv:>10.4f} {imp:>10}")
    
    # Save results
    results_path = out / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results: {results_path}")
    print(f"Outputs: {out}/norm/, {out}/raw/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
