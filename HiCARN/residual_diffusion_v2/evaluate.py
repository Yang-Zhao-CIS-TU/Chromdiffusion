#!/usr/bin/env python3
"""
Hi-C Prediction Evaluation Script

Computes comprehensive metrics between predicted Hi-C matrices and ground truth
in RAW space (contact counts).

Metrics computed:
- MSE, MAE, RMSE (reconstruction quality)
- PCC, SSIM, PSNR (similarity)
- Top-K overlap (peak localization)
- Hit@K, IoU (peak detection accuracy)

Usage:
    python evaluate.py \
        --pred_path refined_chr19/refined_raw.npy \
        --gt_path /path/to/ground_truth_raw.npy \
        --output_json chr19_results.json
"""

import numpy as np
import argparse
import json
from pathlib import Path
from scipy import stats
from math import log10
import warnings
warnings.filterwarnings('ignore')


def ensure_nhw(arr):
    """
    Ensure array is (N, H, W) format.
    Handles both NCHW (N, 1, H, W) and NHWC (N, H, W, 1) formats.
    """
    arr = np.asarray(arr)
    
    if arr.ndim == 2:
        # (H, W) -> (1, H, W)
        return arr[np.newaxis, ...]
    
    elif arr.ndim == 3:
        # Already (N, H, W)
        return arr
    
    elif arr.ndim == 4:
        # Could be NCHW (N, 1, H, W) or NHWC (N, H, W, 1)
        if arr.shape[1] == 1:
            # NCHW format: (N, 1, H, W) -> (N, H, W)
            return arr.squeeze(1)
        elif arr.shape[-1] == 1:
            # NHWC format: (N, H, W, 1) -> (N, H, W)
            return arr.squeeze(-1)
        else:
            raise ValueError(f"Cannot convert shape {arr.shape} to (N, H, W)")
    
    else:
        raise ValueError(f"Unexpected array dimension: {arr.ndim}")


# ================================================================
# SSIM Implementation (no skimage dependency)
# ================================================================

def compute_ssim_manual(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Compute SSIM between two images.
    Simple implementation without skimage dependency.
    """
    # Ensure float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Mean
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Variance and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_val = numerator / denominator
    return float(ssim_val)


# ================================================================
# Basic Metrics
# ================================================================

def compute_mse(pred, gt):
    """Mean Squared Error"""
    return float(np.mean((pred - gt) ** 2))

def compute_mae(pred, gt):
    """Mean Absolute Error"""
    return float(np.mean(np.abs(pred - gt)))

def compute_rmse(pred, gt):
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

def compute_pcc(pred, gt):
    """Pearson Correlation Coefficient"""
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Remove NaN and Inf
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(gt_flat) | 
                   np.isinf(pred_flat) | np.isinf(gt_flat))
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]
    
    if len(pred_flat) == 0:
        return 0.0
    
    corr, _ = stats.pearsonr(pred_flat, gt_flat)
    return float(corr) if not np.isnan(corr) else 0.0

def compute_spearman(pred, gt):
    """Spearman Correlation Coefficient"""
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(gt_flat) | 
                   np.isinf(pred_flat) | np.isinf(gt_flat))
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]
    
    if len(pred_flat) == 0:
        return 0.0
    
    corr, _ = stats.spearmanr(pred_flat, gt_flat)
    return float(corr) if not np.isnan(corr) else 0.0

def compute_ssim_score(pred, gt):
    """Structural Similarity Index (per sample, then averaged)"""
    # Standardize to (N, H, W)
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    ssim_scores = []
    for i in range(len(pred)):
        p = pred[i]
        g = gt[i]
        
        # Normalize to [0, 1] for SSIM
        max_val = max(p.max(), g.max())
        if max_val > 0:
            p_norm = p / max_val
            g_norm = g / max_val
        else:
            p_norm = p
            g_norm = g
        
        try:
            # Try skimage first
            from skimage.metrics import structural_similarity as ssim_sk
            score = ssim_sk(p_norm, g_norm, data_range=1.0)
        except ImportError:
            # Fallback to manual implementation
            score = compute_ssim_manual(p_norm, g_norm)
        except Exception:
            score = compute_ssim_manual(p_norm, g_norm)
        
        ssim_scores.append(score)
    
    return float(np.mean(ssim_scores))

def compute_psnr(pred, gt):
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 100.0
    
    max_val = max(pred.max(), gt.max())
    if max_val == 0:
        return 0.0
    
    psnr = 10 * log10((max_val ** 2) / mse)
    return float(psnr)


# ================================================================
# Peak/Localization Metrics
# ================================================================

def get_top_k_positions(matrix, k=20):
    """Get positions of top-k values in a matrix"""
    if matrix.ndim > 2:
        matrix = matrix.squeeze()
    
    flat = matrix.flatten()
    top_k_indices = np.argsort(flat)[-k:][::-1]  # Descending order
    
    positions = set()
    h, w = matrix.shape[-2], matrix.shape[-1]
    for idx in top_k_indices:
        row = idx // w
        col = idx % w
        positions.add((row, col))
    
    return positions

def compute_top_k_overlap(pred, gt, k=20):
    """
    Compute overlap between top-k positions in pred and gt.
    
    Returns:
        float: Fraction of top-k positions that match (0 to 1)
    """
    # Standardize to (N, H, W)
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    overlaps = []
    for i in range(len(pred)):
        pred_top_k = get_top_k_positions(pred[i], k)
        gt_top_k = get_top_k_positions(gt[i], k)
        
        overlap = len(pred_top_k & gt_top_k) / k
        overlaps.append(overlap)
    
    return float(np.mean(overlaps))

def compute_hit_at_k(pred, gt, k=1):
    """
    Hit@K: Check if the top-1 (or top-k) position in pred matches any of top-k in gt.
    
    Args:
        pred: Predicted matrices
        gt: Ground truth matrices
        k: Number of top positions to consider
    
    Returns:
        float: Fraction of samples where top-1 pred is in top-k gt
    """
    # Standardize to (N, H, W)
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    hits = []
    for i in range(len(pred)):
        pred_top_1 = get_top_k_positions(pred[i], 1)
        gt_top_k = get_top_k_positions(gt[i], k)
        
        hit = 1.0 if len(pred_top_1 & gt_top_k) > 0 else 0.0
        hits.append(hit)
    
    return float(np.mean(hits))

def compute_peak_iou(pred, gt, threshold_percentile=95):
    """
    IoU of peak regions (values above threshold percentile).
    
    Args:
        pred: Predicted matrices
        gt: Ground truth matrices
        threshold_percentile: Percentile threshold to define "peak" region
    
    Returns:
        float: Average IoU of peak regions
    """
    # Standardize to (N, H, W)
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    ious = []
    for i in range(len(pred)):
        # Define peak regions
        pred_thresh = np.percentile(pred[i], threshold_percentile)
        gt_thresh = np.percentile(gt[i], threshold_percentile)
        
        pred_peaks = pred[i] >= pred_thresh
        gt_peaks = gt[i] >= gt_thresh
        
        # Compute IoU
        intersection = np.sum(pred_peaks & gt_peaks)
        union = np.sum(pred_peaks | gt_peaks)
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
        
        ious.append(iou)
    
    return float(np.mean(ious))

def compute_peak_distance(pred, gt):
    """
    Average distance between predicted peak and ground truth peak.
    
    Returns:
        float: Average Euclidean distance (lower is better)
    """
    # Standardize to (N, H, W)
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    distances = []
    for i in range(len(pred)):
        # Find peak positions
        pred_peak_idx = np.unravel_index(np.argmax(pred[i]), pred[i].shape)
        gt_peak_idx = np.unravel_index(np.argmax(gt[i]), gt[i].shape)
        
        # Euclidean distance
        dist = np.sqrt((pred_peak_idx[0] - gt_peak_idx[0])**2 + 
                       (pred_peak_idx[1] - gt_peak_idx[1])**2)
        distances.append(dist)
    
    return float(np.mean(distances))


# ================================================================
# Main Evaluation Function
# ================================================================

def evaluate(pred, gt, verbose=True):
    """
    Compute all metrics between predictions and ground truth.
    
    Args:
        pred: Predicted matrices (N, H, W), (N, 1, H, W), or (N, H, W, 1)
        gt: Ground truth matrices (N, H, W), (N, 1, H, W), or (N, H, W, 1)
        verbose: Print results
    
    Returns:
        dict: All computed metrics
    """
    # Standardize to (N, H, W) format
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    if verbose:
        print(f"\n  Standardized shapes: pred={pred.shape}, gt={gt.shape}")
    
    # Ensure same shape
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch after standardization: pred={pred.shape}, gt={gt.shape}")
    
    results = {}
    
    # ============================================================
    # Basic Reconstruction Metrics
    # ============================================================
    if verbose:
        print("\n" + "="*60)
        print("RECONSTRUCTION METRICS")
        print("="*60)
    
    results['mse'] = compute_mse(pred, gt)
    results['mae'] = compute_mae(pred, gt)
    results['rmse'] = compute_rmse(pred, gt)
    
    if verbose:
        print(f"  MSE:  {results['mse']:.6f}")
        print(f"  MAE:  {results['mae']:.6f}")
        print(f"  RMSE: {results['rmse']:.6f}")
    
    # ============================================================
    # Similarity Metrics
    # ============================================================
    if verbose:
        print("\n" + "="*60)
        print("SIMILARITY METRICS")
        print("="*60)
    
    results['pcc'] = compute_pcc(pred, gt)
    results['spearman'] = compute_spearman(pred, gt)
    results['ssim'] = compute_ssim_score(pred, gt)
    results['psnr'] = compute_psnr(pred, gt)
    
    if verbose:
        print(f"  PCC (Pearson):  {results['pcc']:.4f}")
        print(f"  Spearman:       {results['spearman']:.4f}")
        print(f"  SSIM:           {results['ssim']:.4f}")
        print(f"  PSNR:           {results['psnr']:.2f} dB")
    
    # ============================================================
    # Peak Localization Metrics
    # ============================================================
    if verbose:
        print("\n" + "="*60)
        print("PEAK LOCALIZATION METRICS")
        print("="*60)
    
    results['top_5_overlap'] = compute_top_k_overlap(pred, gt, k=5)
    results['top_10_overlap'] = compute_top_k_overlap(pred, gt, k=10)
    results['top_20_overlap'] = compute_top_k_overlap(pred, gt, k=20)
    results['top_50_overlap'] = compute_top_k_overlap(pred, gt, k=50)
    
    if verbose:
        print(f"  Top-5 Overlap:  {results['top_5_overlap']*100:.1f}%")
        print(f"  Top-10 Overlap: {results['top_10_overlap']*100:.1f}%")
        print(f"  Top-20 Overlap: {results['top_20_overlap']*100:.1f}%")
        print(f"  Top-50 Overlap: {results['top_50_overlap']*100:.1f}%")
    
    # ============================================================
    # Hit@K Metrics
    # ============================================================
    if verbose:
        print("\n" + "="*60)
        print("HIT@K METRICS (Top-1 pred in Top-K gt)")
        print("="*60)
    
    results['hit_at_1'] = compute_hit_at_k(pred, gt, k=1)
    results['hit_at_3'] = compute_hit_at_k(pred, gt, k=3)
    results['hit_at_5'] = compute_hit_at_k(pred, gt, k=5)
    results['hit_at_10'] = compute_hit_at_k(pred, gt, k=10)
    
    if verbose:
        print(f"  Hit@1:  {results['hit_at_1']*100:.1f}%")
        print(f"  Hit@3:  {results['hit_at_3']*100:.1f}%")
        print(f"  Hit@5:  {results['hit_at_5']*100:.1f}%")
        print(f"  Hit@10: {results['hit_at_10']*100:.1f}%")
    
    # ============================================================
    # IoU Metrics
    # ============================================================
    if verbose:
        print("\n" + "="*60)
        print("PEAK IoU METRICS")
        print("="*60)
    
    results['iou_90'] = compute_peak_iou(pred, gt, threshold_percentile=90)
    results['iou_95'] = compute_peak_iou(pred, gt, threshold_percentile=95)
    results['iou_99'] = compute_peak_iou(pred, gt, threshold_percentile=99)
    
    if verbose:
        print(f"  IoU (90th %ile): {results['iou_90']*100:.1f}%")
        print(f"  IoU (95th %ile): {results['iou_95']*100:.1f}%")
        print(f"  IoU (99th %ile): {results['iou_99']*100:.1f}%")
    
    # ============================================================
    # Peak Distance
    # ============================================================
    if verbose:
        print("\n" + "="*60)
        print("PEAK DISTANCE")
        print("="*60)
    
    results['peak_distance'] = compute_peak_distance(pred, gt)
    
    if verbose:
        print(f"  Avg Peak Distance: {results['peak_distance']:.2f} pixels")
    
    # ============================================================
    # Data Statistics
    # ============================================================
    results['pred_min'] = float(pred.min())
    results['pred_max'] = float(pred.max())
    results['pred_mean'] = float(pred.mean())
    results['gt_min'] = float(gt.min())
    results['gt_max'] = float(gt.max())
    results['gt_mean'] = float(gt.mean())
    results['n_samples'] = int(pred.shape[0])
    
    if verbose:
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        print(f"  Samples: {results['n_samples']}")
        print(f"  Pred range: [{results['pred_min']:.3f}, {results['pred_max']:.3f}], mean={results['pred_mean']:.3f}")
        print(f"  GT range:   [{results['gt_min']:.3f}, {results['gt_max']:.3f}], mean={results['gt_mean']:.3f}")
    
    return results


def compare_methods(results_dict, output_path=None):
    """
    Compare multiple methods side by side.
    
    Args:
        results_dict: {method_name: results} dictionary
        output_path: Optional path to save comparison table
    """
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    methods = list(results_dict.keys())
    
    # Key metrics to compare
    key_metrics = [
        ('PCC', 'pcc', '{:.4f}'),
        ('SSIM', 'ssim', '{:.4f}'),
        ('MSE', 'mse', '{:.4f}'),
        ('Top-20 Overlap', 'top_20_overlap', '{:.1%}'),
        ('Hit@1', 'hit_at_1', '{:.1%}'),
        ('Hit@5', 'hit_at_5', '{:.1%}'),
        ('IoU (95%)', 'iou_95', '{:.1%}'),
        ('Peak Dist', 'peak_distance', '{:.2f}'),
    ]
    
    # Print header
    header = f"{'Metric':<20}"
    for method in methods:
        header += f" {method:<15}"
    print(header)
    print("-" * len(header))
    
    # Print each metric
    comparison_data = {}
    for name, key, fmt in key_metrics:
        row = f"{name:<20}"
        comparison_data[name] = {}
        for method in methods:
            val = results_dict[method].get(key, 0)
            row += f" {fmt.format(val):<15}"
            comparison_data[name][method] = val
        print(row)
    
    print("="*80)
    
    # Save comparison
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\nComparison saved to: {output_path}")
    
    return comparison_data


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Hi-C predictions against ground truth'
    )
    
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to predictions (.npy file)')
    parser.add_argument('--gt_path', type=str, required=True,
                       help='Path to ground truth (.npy file)')
    parser.add_argument('--output_json', type=str, default=None,
                       help='Path to save results as JSON')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    # Optional: Compare with baseline
    parser.add_argument('--baseline_path', type=str, default=None,
                       help='Path to baseline predictions for comparison')
    parser.add_argument('--baseline_name', type=str, default='Baseline',
                       help='Name for baseline method in comparison')
    parser.add_argument('--pred_name', type=str, default='Refined',
                       help='Name for prediction method in comparison')
    
    args = parser.parse_args()
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    pred = np.load(args.pred_path)
    gt = np.load(args.gt_path)
    
    print(f"Predictions: {args.pred_path}")
    print(f"  Shape: {pred.shape}")
    print(f"  Range: [{pred.min():.3f}, {pred.max():.3f}]")
    
    print(f"\nGround Truth: {args.gt_path}")
    print(f"  Shape: {gt.shape}")
    print(f"  Range: [{gt.min():.3f}, {gt.max():.3f}]")
    
    # Check for negative values (indicates normalized space)
    if pred.min() < -0.01 or gt.min() < -0.01:
        print("\n⚠️  WARNING: Data contains negative values!")
        print("   This may indicate data is in normalized space, not raw space.")
        print("   Metrics may not be meaningful for normalized data.")
    
    # ========================================================================
    # Evaluate
    # ========================================================================
    
    print("\n" + "="*80)
    print(f"EVALUATING: {args.pred_name}")
    print("="*80)
    
    results = evaluate(pred, gt, verbose=not args.quiet)
    
    # ========================================================================
    # Compare with Baseline (if provided)
    # ========================================================================
    
    if args.baseline_path:
        print("\n" + "="*80)
        print(f"EVALUATING BASELINE: {args.baseline_name}")
        print("="*80)
        
        baseline = np.load(args.baseline_path)
        print(f"Baseline: {args.baseline_path}")
        print(f"  Shape: {baseline.shape}")
        print(f"  Range: [{baseline.min():.3f}, {baseline.max():.3f}]")
        
        baseline_results = evaluate(baseline, gt, verbose=not args.quiet)
        
        # Compare
        results_dict = {
            args.pred_name: results,
            args.baseline_name: baseline_results
        }
        
        comparison_path = None
        if args.output_json:
            comparison_path = args.output_json.replace('.json', '_comparison.json')
        
        compare_methods(results_dict, comparison_path)
        
        # Compute improvement
        print("\n" + "="*80)
        print("IMPROVEMENT SUMMARY")
        print("="*80)
        
        improvements = {
            'PCC': (results['pcc'] - baseline_results['pcc']) / max(abs(baseline_results['pcc']), 1e-6) * 100,
            'SSIM': (results['ssim'] - baseline_results['ssim']) / max(abs(baseline_results['ssim']), 1e-6) * 100,
            'Top-20': (results['top_20_overlap'] - baseline_results['top_20_overlap']) * 100,  # Already in [0,1]
            'Hit@1': (results['hit_at_1'] - baseline_results['hit_at_1']) * 100,
            'IoU': (results['iou_95'] - baseline_results['iou_95']) * 100,
            'Peak Dist': baseline_results['peak_distance'] - results['peak_distance'],  # Lower is better
        }
        
        for metric, imp in improvements.items():
            direction = "↑" if imp > 0 else "↓"
            color_imp = imp if metric != 'Peak Dist' else -imp  # Invert for peak dist
            status = "✓ Better" if color_imp > 0 else "✗ Worse" if color_imp < 0 else "= Same"
            print(f"  {metric:<12}: {direction} {abs(imp):>6.2f}{'%' if metric != 'Peak Dist' else ' px':<3}  {status}")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output_json}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return results


if __name__ == '__main__':
    main()
