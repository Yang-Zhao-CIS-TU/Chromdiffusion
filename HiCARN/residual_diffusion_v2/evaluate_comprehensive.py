#!/usr/bin/env python3
"""
Comprehensive Hi-C Evaluation with All Metrics

Metrics:
- MSE, MAE, RMSE
- PSNR, SNR
- PCC (Pearson), SCC/SPC (Spearman)
- SSIM
- GDS (Genome Distance Stratified correlation)
- Top-K Overlap, Hit@K, IoU
- Peak Distance

Also compares with HiCARN baseline to see if diffusion helps.
"""

import numpy as np
import torch
import sys
from pathlib import Path
import argparse
from scipy import stats
from math import log10, sqrt
import json
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# Setup for torch.load
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


def ensure_nhw(arr):
    """Ensure array is (N, H, W) format."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    elif arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[1] == 1:
            return arr.squeeze(1)
        elif arr.shape[-1] == 1:
            return arr.squeeze(-1)
    raise ValueError(f"Cannot convert shape {arr.shape} to (N, H, W)")


def normalize_data(raw_data, Y_median, Y_iqr):
    """Normalize raw data to match training space"""
    log_data = np.log1p(raw_data)
    norm_data = (log_data - Y_median) / Y_iqr
    norm_data = np.clip(norm_data, -5, 5)
    return norm_data.astype(np.float32)


def load_preprocessor_stats(preprocessor_path):
    """Load Y_median and Y_iqr from preprocessor"""
    try:
        checkpoint = torch.load(preprocessor_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(preprocessor_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'preprocessor' in checkpoint:
        prep = checkpoint['preprocessor']
    else:
        prep = checkpoint
    
    return prep.Y_mean, prep.Y_std


# ================================================================
# All Metrics
# ================================================================

def compute_mse(pred, gt):
    """Mean Squared Error"""
    return float(np.mean((pred - gt) ** 2))

def compute_mae(pred, gt):
    """Mean Absolute Error"""
    return float(np.mean(np.abs(pred - gt)))

def compute_rmse(pred, gt):
    """Root Mean Squared Error"""
    return float(sqrt(np.mean((pred - gt) ** 2)))

def compute_psnr(pred, gt):
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 100.0
    max_val = max(np.abs(pred).max(), np.abs(gt).max())
    if max_val == 0:
        return 0.0
    return float(10 * log10((max_val ** 2) / mse))

def compute_snr(pred, gt):
    """Signal-to-Noise Ratio"""
    signal_power = np.mean(gt ** 2)
    noise_power = np.mean((pred - gt) ** 2)
    if noise_power == 0:
        return 100.0
    return float(10 * log10(signal_power / noise_power))

def compute_pcc(pred, gt):
    """Pearson Correlation Coefficient"""
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    valid = ~(np.isnan(pred_flat) | np.isnan(gt_flat) | np.isinf(pred_flat) | np.isinf(gt_flat))
    if valid.sum() == 0:
        return 0.0
    corr, _ = stats.pearsonr(pred_flat[valid], gt_flat[valid])
    return float(corr) if not np.isnan(corr) else 0.0

def compute_scc(pred, gt):
    """Spearman Correlation Coefficient (SCC/SPC)"""
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    valid = ~(np.isnan(pred_flat) | np.isnan(gt_flat))
    if valid.sum() == 0:
        return 0.0
    corr, _ = stats.spearmanr(pred_flat[valid], gt_flat[valid])
    return float(corr) if not np.isnan(corr) else 0.0

def compute_ssim(pred, gt):
    """Structural Similarity Index (per sample average)"""
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    ssim_scores = []
    for i in range(len(pred)):
        p, g = pred[i].astype(np.float64), gt[i].astype(np.float64)
        
        # Normalize to [0, 1]
        p_min, p_max = p.min(), p.max()
        g_min, g_max = g.min(), g.max()
        
        if p_max - p_min > 0:
            p = (p - p_min) / (p_max - p_min)
        if g_max - g_min > 0:
            g = (g - g_min) / (g_max - g_min)
        
        mu1, mu2 = np.mean(p), np.mean(g)
        sigma1_sq, sigma2_sq = np.var(p), np.var(g)
        sigma12 = np.mean((p - mu1) * (g - mu2))
        
        C1, C2 = 0.01**2, 0.03**2
        ssim_val = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_scores.append(ssim_val)
    
    return float(np.mean(ssim_scores))

def compute_gds(pred, gt, num_diagonals=20):
    """
    Genome Distance Stratified correlation (GDS)
    
    Computes correlation at each diagonal distance and averages.
    Important for Hi-C data as different genomic distances have different properties.
    """
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    correlations = []
    
    for diag in range(1, num_diagonals + 1):
        pred_diag_vals = []
        gt_diag_vals = []
        
        for i in range(len(pred)):
            # Extract diagonal
            p_diag = np.diagonal(pred[i], offset=diag)
            g_diag = np.diagonal(gt[i], offset=diag)
            
            pred_diag_vals.extend(p_diag)
            gt_diag_vals.extend(g_diag)
        
        pred_diag_vals = np.array(pred_diag_vals)
        gt_diag_vals = np.array(gt_diag_vals)
        
        # Filter valid values
        valid = ~(np.isnan(pred_diag_vals) | np.isnan(gt_diag_vals))
        if valid.sum() > 10:  # Need enough points
            corr, _ = stats.pearsonr(pred_diag_vals[valid], gt_diag_vals[valid])
            if not np.isnan(corr):
                correlations.append(corr)
    
    return float(np.mean(correlations)) if correlations else 0.0

def get_top_k_positions(matrix, k=20):
    """Get positions of top-k values"""
    flat = matrix.flatten()
    top_k_indices = np.argsort(flat)[-k:][::-1]
    h, w = matrix.shape[-2], matrix.shape[-1]
    return set((idx // w, idx % w) for idx in top_k_indices)

def compute_top_k_overlap(pred, gt, k=20):
    """Top-K position overlap"""
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    overlaps = []
    for i in range(len(pred)):
        pred_top_k = get_top_k_positions(pred[i], k)
        gt_top_k = get_top_k_positions(gt[i], k)
        overlaps.append(len(pred_top_k & gt_top_k) / k)
    
    return float(np.mean(overlaps))

def compute_hit_at_k(pred, gt, k=1):
    """Hit@K: Is top-1 pred in top-K gt?"""
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    hits = []
    for i in range(len(pred)):
        pred_top_1 = get_top_k_positions(pred[i], 1)
        gt_top_k = get_top_k_positions(gt[i], k)
        hits.append(1.0 if len(pred_top_1 & gt_top_k) > 0 else 0.0)
    
    return float(np.mean(hits))

def compute_peak_iou(pred, gt, percentile=95):
    """IoU of peak regions"""
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    ious = []
    for i in range(len(pred)):
        pred_thresh = np.percentile(pred[i], percentile)
        gt_thresh = np.percentile(gt[i], percentile)
        
        pred_peaks = pred[i] >= pred_thresh
        gt_peaks = gt[i] >= gt_thresh
        
        intersection = np.sum(pred_peaks & gt_peaks)
        union = np.sum(pred_peaks | gt_peaks)
        ious.append(intersection / union if union > 0 else 1.0)
    
    return float(np.mean(ious))

def compute_peak_distance(pred, gt):
    """Average distance between predicted and GT peaks"""
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    distances = []
    for i in range(len(pred)):
        pred_peak = np.unravel_index(np.argmax(pred[i]), pred[i].shape)
        gt_peak = np.unravel_index(np.argmax(gt[i]), gt[i].shape)
        dist = sqrt((pred_peak[0] - gt_peak[0])**2 + (pred_peak[1] - gt_peak[1])**2)
        distances.append(dist)
    
    return float(np.mean(distances))

def compute_random_baseline_metrics(gt, matrix_size=40):
    """Compute expected metrics for random predictions"""
    gt = ensure_nhw(gt)
    n_samples = len(gt)
    
    # Random peak distance (expected for uniform random on square)
    # For 40x40 matrix, average distance between two random points ≈ 18.8
    expected_dist = matrix_size * sqrt(2) / 3  # Approximate
    
    # Random Top-K overlap
    # Probability of overlap = k / (H*W)
    total_pixels = matrix_size * matrix_size
    expected_top20 = 20 / total_pixels * 20 / 20  # ≈ 1.25%
    
    return {
        'random_peak_distance': expected_dist,
        'random_top20_overlap': expected_top20,
    }


def evaluate_all(pred, gt, name="", verbose=True):
    """Compute all metrics"""
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    results = {}
    
    # Basic
    results['mse'] = compute_mse(pred, gt)
    results['mae'] = compute_mae(pred, gt)
    results['rmse'] = compute_rmse(pred, gt)
    
    # Signal quality
    results['psnr'] = compute_psnr(pred, gt)
    results['snr'] = compute_snr(pred, gt)
    
    # Correlation
    results['pcc'] = compute_pcc(pred, gt)
    results['scc'] = compute_scc(pred, gt)  # Spearman
    
    # Structural
    results['ssim'] = compute_ssim(pred, gt)
    results['gds'] = compute_gds(pred, gt)
    
    # Peak localization
    results['top_5'] = compute_top_k_overlap(pred, gt, k=5)
    results['top_10'] = compute_top_k_overlap(pred, gt, k=10)
    results['top_20'] = compute_top_k_overlap(pred, gt, k=20)
    results['top_50'] = compute_top_k_overlap(pred, gt, k=50)
    
    # Hit@K
    results['hit_1'] = compute_hit_at_k(pred, gt, k=1)
    results['hit_5'] = compute_hit_at_k(pred, gt, k=5)
    results['hit_10'] = compute_hit_at_k(pred, gt, k=10)
    
    # IoU
    results['iou_95'] = compute_peak_iou(pred, gt, percentile=95)
    
    # Peak distance
    results['peak_dist'] = compute_peak_distance(pred, gt)
    
    # Stats
    results['n_samples'] = int(pred.shape[0])
    results['pred_range'] = [float(pred.min()), float(pred.max())]
    results['gt_range'] = [float(gt.min()), float(gt.max())]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"METRICS: {name}")
        print(f"{'='*60}")
        print(f"\n[Reconstruction]")
        print(f"  MSE:  {results['mse']:.6f}")
        print(f"  MAE:  {results['mae']:.6f}")
        print(f"  RMSE: {results['rmse']:.6f}")
        
        print(f"\n[Signal Quality]")
        print(f"  PSNR: {results['psnr']:.2f} dB")
        print(f"  SNR:  {results['snr']:.2f} dB")
        
        print(f"\n[Correlation]")
        print(f"  PCC (Pearson):  {results['pcc']:.4f}")
        print(f"  SCC (Spearman): {results['scc']:.4f}")
        
        print(f"\n[Structural]")
        print(f"  SSIM: {results['ssim']:.4f}")
        print(f"  GDS:  {results['gds']:.4f}")
        
        print(f"\n[Peak Localization - Top-K Overlap]")
        print(f"  Top-5:  {results['top_5']*100:5.1f}%")
        print(f"  Top-10: {results['top_10']*100:5.1f}%")
        print(f"  Top-20: {results['top_20']*100:5.1f}%")
        print(f"  Top-50: {results['top_50']*100:5.1f}%")
        
        print(f"\n[Hit@K]")
        print(f"  Hit@1:  {results['hit_1']*100:5.1f}%")
        print(f"  Hit@5:  {results['hit_5']*100:5.1f}%")
        print(f"  Hit@10: {results['hit_10']*100:5.1f}%")
        
        print(f"\n[Peak IoU & Distance]")
        print(f"  IoU@95%:      {results['iou_95']*100:5.1f}%")
        print(f"  Peak Dist:    {results['peak_dist']:.2f} pixels")
        
        print(f"\n[Data]")
        print(f"  Samples: {results['n_samples']}")
        print(f"  Pred: [{results['pred_range'][0]:.3f}, {results['pred_range'][1]:.3f}]")
        print(f"  GT:   [{results['gt_range'][0]:.3f}, {results['gt_range'][1]:.3f}]")
    
    return results


def compare_results(results_dict):
    """Compare multiple methods"""
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    methods = list(results_dict.keys())
    
    metrics = [
        ('MSE', 'mse', '{:.4f}', False),
        ('PSNR', 'psnr', '{:.2f}', True),
        ('SNR', 'snr', '{:.2f}', True),
        ('PCC', 'pcc', '{:.4f}', True),
        ('SCC', 'scc', '{:.4f}', True),
        ('SSIM', 'ssim', '{:.4f}', True),
        ('GDS', 'gds', '{:.4f}', True),
        ('Top-20', 'top_20', '{:.1%}', True),
        ('Hit@1', 'hit_1', '{:.1%}', True),
        ('IoU@95', 'iou_95', '{:.1%}', True),
        ('Peak Dist', 'peak_dist', '{:.2f}', False),
    ]
    
    # Header
    header = f"{'Metric':<12}"
    for m in methods:
        header += f" {m:<15}"
    header += " Better"
    print(header)
    print("-" * len(header))
    
    # Rows
    for name, key, fmt, higher_better in metrics:
        row = f"{name:<12}"
        values = []
        for m in methods:
            val = results_dict[m].get(key, 0)
            values.append(val)
            row += f" {fmt.format(val):<15}"
        
        # Determine which is better
        if len(values) == 2:
            if higher_better:
                better = methods[0] if values[0] > values[1] else methods[1]
            else:
                better = methods[0] if values[0] < values[1] else methods[1]
            diff = abs(values[0] - values[1])
            if diff < 0.001:
                better = "~Same"
            row += f" {better}"
        
        print(row)
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Hi-C evaluation')
    
    parser.add_argument('--refined_norm_path', type=str, required=True,
                       help='Path to refined predictions (normalized)')
    parser.add_argument('--hicarn_norm_path', type=str, default=None,
                       help='Path to HiCARN predictions (normalized) for comparison')
    parser.add_argument('--gt_path', type=str, required=True,
                       help='Path to ground truth (raw)')
    parser.add_argument('--preprocessor_path', type=str, required=True,
                       help='Path to preprocessor')
    parser.add_argument('--output_json', type=str, default=None,
                       help='Output JSON path')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPREHENSIVE HI-C EVALUATION")
    print("="*80)
    
    # Load preprocessor
    print(f"\nLoading preprocessor: {args.preprocessor_path}")
    Y_median, Y_iqr = load_preprocessor_stats(args.preprocessor_path)
    print(f"  Y_median: {Y_median:.4f}")
    print(f"  Y_iqr:    {Y_iqr:.4f}")
    
    # Load GT and normalize
    print(f"\nLoading GT: {args.gt_path}")
    gt_raw = np.load(args.gt_path)
    gt_norm = normalize_data(gt_raw, Y_median, Y_iqr)
    print(f"  Raw shape: {gt_raw.shape}, range: [{gt_raw.min():.0f}, {gt_raw.max():.0f}]")
    print(f"  Norm shape: {gt_norm.shape}, range: [{gt_norm.min():.3f}, {gt_norm.max():.3f}]")
    
    # Load refined predictions
    print(f"\nLoading refined: {args.refined_norm_path}")
    refined_norm = np.load(args.refined_norm_path)
    print(f"  Shape: {refined_norm.shape}, range: [{refined_norm.min():.3f}, {refined_norm.max():.3f}]")
    
    # Evaluate refined
    refined_results = evaluate_all(refined_norm, gt_norm, name="Refined (Diffusion)", verbose=True)
    
    # Load and evaluate HiCARN baseline if provided
    if args.hicarn_norm_path:
        print(f"\nLoading HiCARN: {args.hicarn_norm_path}")
        hicarn_norm = np.load(args.hicarn_norm_path)
        print(f"  Shape: {hicarn_norm.shape}, range: [{hicarn_norm.min():.3f}, {hicarn_norm.max():.3f}]")
        
        hicarn_results = evaluate_all(hicarn_norm, gt_norm, name="HiCARN (Baseline)", verbose=True)
        
        # Compare
        compare_results({
            'Refined': refined_results,
            'HiCARN': hicarn_results
        })
        
        # Is diffusion helping?
        print("\n" + "="*80)
        print("ANALYSIS: Is diffusion refinement helping?")
        print("="*80)
        
        improvements = {
            'PCC': refined_results['pcc'] - hicarn_results['pcc'],
            'SSIM': refined_results['ssim'] - hicarn_results['ssim'],
            'Top-20': refined_results['top_20'] - hicarn_results['top_20'],
            'Peak Dist': hicarn_results['peak_dist'] - refined_results['peak_dist'],
        }
        
        helping = sum(1 for v in improvements.values() if v > 0)
        
        for metric, diff in improvements.items():
            status = "✓ Better" if diff > 0 else "✗ Worse" if diff < 0 else "= Same"
            print(f"  {metric:<12}: {diff:+.4f} {status}")
        
        if helping >= 3:
            print(f"\n✓ Diffusion refinement is HELPING ({helping}/4 metrics improved)")
        else:
            print(f"\n✗ Diffusion refinement is HURTING ({helping}/4 metrics improved)")
            print("  Consider checking training or using HiCARN directly.")
    
    # Random baseline
    random_baseline = compute_random_baseline_metrics(gt_norm)
    print(f"\n[Random Baseline Reference]")
    print(f"  Expected random peak distance: {random_baseline['random_peak_distance']:.2f} pixels")
    print(f"  Expected random Top-20 overlap: {random_baseline['random_top20_overlap']*100:.2f}%")
    
    if refined_results['peak_dist'] > random_baseline['random_peak_distance'] * 0.9:
        print(f"\n⚠️  WARNING: Peak distance ({refined_results['peak_dist']:.2f}) is near random level!")
        print(f"   This suggests the model is not learning meaningful peak positions.")
    
    # Save
    if args.output_json:
        output = {
            'refined': refined_results,
            'random_baseline': random_baseline
        }
        if args.hicarn_norm_path:
            output['hicarn'] = hicarn_results
        
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Results saved to: {args.output_json}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
