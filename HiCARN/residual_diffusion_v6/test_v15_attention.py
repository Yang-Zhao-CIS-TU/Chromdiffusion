#!/usr/bin/env python3
"""
Test/Inference Script for Diffusion Refinement v15_ATT (v15 + bottleneck attention)

Architecture is imported from train_v15_attention.py so train/test stay identical.
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
# Model / scheduler: identical to train_v15_attention.py
# ================================================================

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v15_attention import GatedConditionedUNet, DDPMScheduler


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
                       cond_mean, cond_std, alpha, device, num_steps=20, batch_size=64,
                       seed=None):
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
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
    parser.add_argument('--seed', type=int, default=None,
                       help='Optional sampling seed (defaults to checkpoint config seed).')
    # Optional overrides; if omitted, architecture is read from checkpoint config
    parser.add_argument('--use_self_attention', action='store_true', default=False)
    parser.add_argument('--no_self_attention', action='store_true', default=False)
    parser.add_argument('--use_cross_attention', action='store_true', default=False)
    parser.add_argument('--no_cross_attention', action='store_true', default=False)
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
    print(f"  model_version={cfg.get('model_version', 'unknown')}")

    if args.no_self_attention:
        use_self = False
    elif args.use_self_attention:
        use_self = True
    else:
        use_self = cfg.get('use_self_attention', True)
    if args.no_cross_attention:
        use_cross = False
    elif args.use_cross_attention:
        use_cross = True
    else:
        use_cross = cfg.get('use_cross_attention', True)
    self_heads = cfg.get('self_attn_heads', 4)
    cross_heads = cfg.get('cross_attn_heads', 4)
    self_scale = cfg.get('self_attn_init_scale', 0.1)
    cross_scale = cfg.get('cross_attn_init_scale', 0.1)
    channel_mults = tuple(cfg.get('channel_mults', (1, 2, 4)))

    print(f"  use_self_attention={use_self} (heads={self_heads})")
    print(f"  use_cross_attention={use_cross} (heads={cross_heads})")
    
    model = GatedConditionedUNet(
        base_channels=cfg.get('base_channels', 64),
        channel_mults=channel_mults,
        cond_norm_type=cfg.get('cond_norm', 'learnable'),
        output_gate=cfg.get('use_gate', True),
        g_scale=cfg.get('g_scale', 0.5),
        use_self_attention=use_self,
        use_cross_attention=use_cross,
        self_attn_heads=self_heads,
        cross_attn_heads=cross_heads,
        self_attn_init_scale=self_scale,
        cross_attn_init_scale=cross_scale,
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing or unexpected:
        print("  Checkpoint load (strict=False):")
        print(f"    Missing keys: {list(missing)}")
        print(f"    Unexpected keys: {list(unexpected)}")
    model.eval()
    
    scheduler = DDPMScheduler(num_train_timesteps=cfg.get('num_timesteps', 1000)).to(device)

    infer_seed = args.seed if args.seed is not None else cfg.get('seed', None)
    if infer_seed is not None:
        torch.manual_seed(int(infer_seed))
        print(f"  sampling seed={infer_seed}")
    
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
            args.num_steps, args.batch_size,
            seed=None,
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
        'config': {
            'alpha': float(alpha),
            'num_steps': args.num_steps,
            'model_version': cfg.get('model_version', 'v16_attention'),
            'use_self_attention': bool(use_self),
            'use_cross_attention': bool(use_cross),
            'self_attn_heads': int(self_heads),
            'cross_attn_heads': int(cross_heads),
        }
    }
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results: {results_path}")
    print(f"Outputs: {out}/norm/, {out}/raw/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
