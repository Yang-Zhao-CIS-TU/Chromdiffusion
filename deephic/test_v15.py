#!/usr/bin/env python3
"""
Test/Inference Script for Diffusion Refinement v15

Evaluates in both NORMALIZED and RAW spaces.
Metrics: PCC, SPC, SSIM, PSNR, SNR, MSE, IoU@K

Usage:
    python test_v15.py \
        --checkpoint checkpoints_v15/best_model_iou.pt \
        --preprocess_file /home/yangz/data/hic_data/HiCARN/hicarn_predictions/preprocessor.pt \
        --chromosomes chr18 chr19 chr20 chr21 chr22 \
        --hicarn_dir /home/yangz/data/hic_data/HiCARN/hicarn_predictions \
        --gt_dir /data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40 \
        --output_dir test_results_v15 \
        --num_steps 20 \
        --device cuda
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import math
from math import exp


# ================================================================
# Model Definition
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
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class GatedConditionedUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=64,
                 channel_mults=(1, 2, 4), time_emb_dim=256,
                 cond_norm_type='learnable', output_gate=True, g_scale=0.5):
        super().__init__()
        self.cond_norm_type = cond_norm_type
        self.output_gate = output_gate
        self.g_scale = g_scale
        
        if cond_norm_type == 'learnable':
            self.cond_transform = nn.Sequential(
                nn.Conv2d(1, 1, 1, bias=True),
            )
            nn.init.constant_(self.cond_transform[0].weight, 0.4)
            nn.init.constant_(self.cond_transform[0].bias, -0.5)
        else:
            self.cond_transform = None
        
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
        
        self.final_norm = nn.GroupNorm(8, ch)
        self.residual_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        if output_gate:
            self.gate_conv = nn.Sequential(
                nn.Conv2d(ch, ch // 2, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(ch // 2, 1, 1),
                nn.Sigmoid()
            )
        else:
            self.gate_conv = None
    
    def forward(self, x_t, cond, t, cond_mean=None, cond_std=None):
        if self.cond_transform is not None:
            cond_norm = self.cond_transform(cond)
        else:
            cond_norm = cond
        
        x = torch.cat([x_t, cond_norm], dim=1)
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
        
        h = F.silu(self.final_norm(h))
        residual = self.residual_conv(h)
        
        gate = self.gate_conv(h) * self.g_scale if self.gate_conv else None
        return residual, gate


# ================================================================
# Scheduler
# ================================================================

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
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
# Preprocessor
# ================================================================

class RobustHiCPreprocessor:
    def __init__(self):
        self.Y_mean = None
        self.Y_std = None
    
    def postprocess(self, Y_norm):
        Y_norm = np.clip(Y_norm, -5, 5)
        Y_log = Y_norm * self.Y_std + self.Y_mean
        Y_counts = np.expm1(Y_log)
        return np.maximum(Y_counts, 0.0)


def load_preprocessor(path):
    preprocessor = torch.load(path, map_location='cpu')
    print(f"  Preprocessor: Y_mean={preprocessor.Y_mean:.4f}, Y_std={preprocessor.Y_std:.4f}")
    return preprocessor


# ================================================================
# SSIM
# ================================================================

class SSIM(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.window = self._create_window(window_size)
    
    def _create_window(self, size, sigma=3):
        gauss = torch.Tensor([exp(-(x - size // 2) ** 2 / (2 * sigma ** 2)) for x in range(size)])
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1).mm(gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        return window
    
    def forward(self, img1, img2):
        window = self.window.to(img1.device).type_as(img1)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2)
        
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2) - mu1_mu2
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


# ================================================================
# Metrics
# ================================================================

class VisionMetrics:
    def __init__(self):
        self.ssim_module = SSIM()
        self.reset()
    
    def reset(self):
        self.logs = {k: [] for k in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']}
    
    def _psnr(self, pred, gt):
        mse = np.mean((pred - gt) ** 2)
        if mse == 0: return float('inf')
        max_val = max(gt.max(), pred.max())
        return 10 * np.log10((max_val ** 2) / mse) if max_val > 0 else 0
    
    def _snr(self, pred, gt):
        signal = np.mean(gt ** 2)
        noise = np.mean((pred - gt) ** 2)
        if noise == 0: return float('inf')
        return 10 * np.log10(signal / noise) if signal > 0 else 0
    
    def _pcc(self, pred, gt):
        if np.std(pred) == 0 or np.std(gt) == 0: return 0
        return pearsonr(pred.flatten(), gt.flatten())[0]
    
    def _spc(self, pred, gt):
        return spearmanr(pred.flatten(), gt.flatten())[0]
    
    def _ssim(self, pred, gt):
        p = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        g = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
        return self.ssim_module(p, g).item()
    
    def _mse(self, pred, gt):
        return np.mean((pred - gt) ** 2)
    
    def add_batch(self, pred, gt):
        if pred.ndim == 4: pred = pred[:, 0]
        if gt.ndim == 4: gt = gt[:, 0]
        
        for i in range(len(pred)):
            p, g = pred[i], gt[i]
            self.logs['pcc'].append(self._pcc(p, g))
            self.logs['spc'].append(self._spc(p, g))
            self.logs['ssim'].append(self._ssim(p, g))
            self.logs['psnr'].append(self._psnr(p, g))
            self.logs['snr'].append(self._snr(p, g))
            self.logs['mse'].append(self._mse(p, g))
    
    def get_summary(self):
        return {k: {'mean': np.mean(v), 'std': np.std(v), 'n': len(v)} 
                for k, v in self.logs.items() if v}


def compute_topk_iou(pred, gt, k_perc=0.5, min_diag=2):
    if pred.ndim == 4: pred = pred[:, 0]
    if gt.ndim == 4: gt = gt[:, 0]
    
    N, H, W = pred.shape
    i_idx, j_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mask = (np.abs(j_idx - i_idx) >= min_diag) & (j_idx >= i_idx)
    
    ious = []
    for i in range(N):
        pv, gv = pred[i][mask], gt[i][mask]
        if len(pv) < 10: continue
        K = max(1, int(len(pv) * k_perc / 100))
        p_topk = set(np.argsort(pv)[-K:])
        g_topk = set(np.argsort(gv)[-K:])
        inter = len(p_topk & g_topk)
        union = len(p_topk | g_topk)
        ious.append(inter / union if union else 0)
    
    return {'mean': np.mean(ious) if ious else 0, 'std': np.std(ious) if ious else 0}


# ================================================================
# Utilities
# ================================================================

def ensure_nchw(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3: return arr[:, np.newaxis, :, :]
    if arr.ndim == 4:
        if arr.shape[1] == 1: return arr
        if arr.shape[-1] == 1: return np.transpose(arr, (0, 3, 1, 2))
    raise ValueError(f"Cannot convert {arr.shape} to NCHW")


# ================================================================
# Inference
# ================================================================

@torch.no_grad()
def refine_predictions(model, scheduler, hicarn, res_mean, res_std, 
                       cond_mean, cond_std, alpha, device, num_steps=20, batch_size=64):
    model.eval()
    N, _, H, W = hicarn.shape
    
    cond_mean_t = torch.tensor(cond_mean, device=device)
    cond_std_t = torch.tensor(cond_std, device=device)
    
    timesteps = torch.linspace(scheduler.num_train_timesteps - 1, 0, num_steps).long().to(device)
    
    all_refined, all_gates = [], []
    
    for start in tqdm(range(0, N, batch_size), desc='Refining'):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(hicarn[start:end]).float().to(device)
        B = len(batch)
        
        x_t = torch.randn(B, 1, H, W, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            pred_v, gate = model(x_t, batch, t_batch, cond_mean_t, cond_std_t)
            pred_x0 = scheduler.predict_x0_from_v(x_t, pred_v, t_batch)
            
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                a_t = scheduler.alphas_cumprod[t]
                a_next = scheduler.alphas_cumprod[t_next]
                noise = (x_t - a_t.sqrt() * pred_x0) / (1 - a_t).sqrt()
                x_t = a_next.sqrt() * pred_x0 + (1 - a_next).sqrt() * noise
            else:
                x_t = pred_x0
        
        pred_res = x_t * res_std + res_mean
        refined = batch + alpha * (gate * pred_res if gate is not None else pred_res)
        refined = torch.clamp(refined, -5, 5)
        
        all_refined.append(refined.cpu().numpy())
        if gate is not None:
            all_gates.append(gate.cpu().numpy())
    
    return np.concatenate(all_refined), np.concatenate(all_gates) if all_gates else None


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--preprocess_file', required=True)
    parser.add_argument('--chromosomes', nargs='+', required=True)
    parser.add_argument('--hicarn_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--ratio', type=int, default=16)
    parser.add_argument('--save_gates', action='store_true')
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
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get('config', {})
    res_mean = ckpt.get('res_mean', 0.0)
    res_std = ckpt.get('res_std', 1.0)
    cond_mean = ckpt.get('cond_mean', 0.0)
    cond_std = ckpt.get('cond_std', 1.0)
    alpha = args.alpha or ckpt.get('alpha', 0.10)
    
    print(f"  res_mean={res_mean:.6f}, res_std={res_std:.6f}, alpha={alpha:.4f}")
    
    model = GatedConditionedUNet(
        base_channels=cfg.get('base_channels', 64),
        cond_norm_type=cfg.get('cond_norm', 'learnable'),
        output_gate=cfg.get('use_gate', True),
        g_scale=cfg.get('g_scale', 0.5)
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    scheduler = DDPMScheduler(cfg.get('num_timesteps', 1000)).to(device)
    
    # Load preprocessor
    print("\n" + "="*70)
    print("LOADING PREPROCESSOR")
    print("="*70)
    preproc = load_preprocessor(args.preprocess_file)
    
    # Process chromosomes
    print("\n" + "="*70)
    print("PROCESSING")
    print("="*70)
    
    results = {'norm': {}, 'raw': {}}
    h_norm_all, r_norm_all = VisionMetrics(), VisionMetrics()
    h_raw_all, r_raw_all = VisionMetrics(), VisionMetrics()
    
    for chrom in args.chromosomes:
        print(f"\n>>> {chrom}")
        
        # Load data - support multiple path patterns
        # Pattern 1: {hicarn_dir}/{chrom}/predictions_norm.npy
        # Pattern 2: {hicarn_dir}/preds_lr_test_{chrom}_ratio{ratio}.npy
        hicarn_path = None
        for pattern in [
            os.path.join(args.hicarn_dir, chrom, "predictions_norm.npy"),
            os.path.join(args.hicarn_dir, f"preds_lr_test_{chrom}_ratio{args.ratio}.npy"),
            os.path.join(args.hicarn_dir, f"predictions_norm_{chrom}.npy"),
        ]:
            if os.path.exists(pattern):
                hicarn_path = pattern
                break
        
        if hicarn_path is None:
            print(f"  Skip: HiCARN predictions not found for {chrom}")
            continue
        
        hicarn_norm = ensure_nchw(np.load(hicarn_path)).astype(np.float32)
        print(f"  HiCARN: {hicarn_norm.shape} from {hicarn_path}")
        
        # Load GT - support multiple path patterns
        # Pattern 1: {hicarn_dir}/{chrom}/ground_truth.npy (normalized, same dir as HiCARN)
        # Pattern 2: {gt_dir}/hr_test_{chrom}_ratio{ratio}.npy (raw)
        # Pattern 3: {gt_dir}/{chrom}/ground_truth.npy
        gt_path = None
        gt_is_raw = False
        
        for pattern, is_raw in [
            (os.path.join(args.hicarn_dir, chrom, "ground_truth.npy"), False),  # Normalized GT in HiCARN dir
            (os.path.join(args.gt_dir, f"hr_test_{chrom}_ratio{args.ratio}.npy"), True),  # Raw GT
            (os.path.join(args.gt_dir, chrom, "ground_truth.npy"), False),
        ]:
            if os.path.exists(pattern):
                gt_path = pattern
                gt_is_raw = is_raw
                break
        
        gt_norm, gt_raw = None, None
        
        if gt_path is not None:
            gt_data = ensure_nchw(np.load(gt_path)).astype(np.float32)
            print(f"  GT: {gt_data.shape} from {gt_path}")
            
            # Determine if GT is raw or normalized
            # Heuristic: raw data typically has max > 50
            if gt_is_raw or gt_data.max() > 50:
                gt_raw = gt_data
                print(f"    -> RAW space (max={gt_data.max():.2f})")
            else:
                gt_norm = gt_data
                gt_raw = preproc.postprocess(gt_norm)
                print(f"    -> NORMALIZED space (max={gt_data.max():.4f})")
        else:
            print(f"  GT: Not found")
        
        # Refine
        refined_norm, gates = refine_predictions(
            model, scheduler, hicarn_norm,
            res_mean, res_std, cond_mean, cond_std, alpha, device,
            args.num_steps, args.batch_size
        )
        
        # Denormalize
        hicarn_raw = preproc.postprocess(hicarn_norm)
        refined_raw = preproc.postprocess(refined_norm)
        
        # Save
        np.save(out / 'norm' / f"hicarn_{chrom}.npy", hicarn_norm)
        np.save(out / 'norm' / f"refined_{chrom}.npy", refined_norm)
        np.save(out / 'raw' / f"hicarn_{chrom}.npy", hicarn_raw)
        np.save(out / 'raw' / f"refined_{chrom}.npy", refined_raw)
        
        if gt_norm is not None:
            np.save(out / 'norm' / f"gt_{chrom}.npy", gt_norm)
        if gt_raw is not None:
            np.save(out / 'raw' / f"gt_{chrom}.npy", gt_raw)
        
        # Metrics
        for space, (h, r, g) in [
            ('NORM', (hicarn_norm, refined_norm, gt_norm)),
            ('RAW', (hicarn_raw, refined_raw, gt_raw))
        ]:
            if g is None:
                continue
            
            hm, rm = VisionMetrics(), VisionMetrics()
            hm.add_batch(h, g)
            rm.add_batch(r, g)
            hs, rs = hm.get_summary(), rm.get_summary()
            
            h_iou = compute_topk_iou(h, g, 0.5)
            r_iou = compute_topk_iou(r, g, 0.5)
            
            print(f"\n  [{space}]")
            print(f"  {'Metric':<8} {'HiCARN':>10} {'Refined':>10} {'Δ':>10}")
            print(f"  {'-'*40}")
            for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
                hv, rv = hs[m]['mean'], rs[m]['mean']
                d = (hv - rv) if m == 'mse' else (rv - hv)
                print(f"  {m.upper():<8} {hv:>10.4f} {rv:>10.4f} {d:>+10.4f}")
            print(f"  {'IoU@0.5':<8} {h_iou['mean']:>10.4f} {r_iou['mean']:>10.4f} {r_iou['mean']-h_iou['mean']:>+10.4f}")
            
            if space == 'NORM':
                h_norm_all.add_batch(h, g)
                r_norm_all.add_batch(r, g)
            else:
                h_raw_all.add_batch(h, g)
                r_raw_all.add_batch(r, g)
    
    # Summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    for name, hm, rm in [('NORMALIZED', h_norm_all, r_norm_all), ('RAW', h_raw_all, r_raw_all)]:
        hs, rs = hm.get_summary(), rm.get_summary()
        if not hs:
            continue
        
        n = hs.get('pcc', {}).get('n', 0)
        print(f"\n[{name}] - {n} samples")
        print(f"{'Metric':<8} {'HiCARN':>15} {'Refined':>15} {'Improv%':>10}")
        print(f"{'-'*50}")
        
        for m in ['pcc', 'spc', 'ssim', 'psnr', 'snr', 'mse']:
            hv = hs.get(m, {}).get('mean', 0)
            hs_std = hs.get(m, {}).get('std', 0)
            rv = rs.get(m, {}).get('mean', 0)
            rs_std = rs.get(m, {}).get('std', 0)
            imp = ((hv - rv) / hv * 100) if m == 'mse' and hv != 0 else \
                  ((rv - hv) / abs(hv) * 100 if hv != 0 else 0)
            print(f"{m.upper():<8} {hv:>7.4f}±{hs_std:<5.4f} {rv:>7.4f}±{rs_std:<5.4f} {imp:>+9.2f}%")
    
    # Save results
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    results_path = out / 'evaluation_results.json'
    results_data = {
        'norm': convert_to_serializable(h_norm_all.get_summary()),
        'raw': convert_to_serializable(h_raw_all.get_summary()),
        'refined_norm': convert_to_serializable(r_norm_all.get_summary()),
        'refined_raw': convert_to_serializable(r_raw_all.get_summary()),
        'config': {'alpha': float(alpha), 'num_steps': args.num_steps}
    }
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results: {results_path}")
    print(f"Outputs: {out}/norm/, {out}/raw/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
