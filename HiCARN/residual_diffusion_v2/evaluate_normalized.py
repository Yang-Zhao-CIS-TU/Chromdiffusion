#!/usr/bin/env python3
"""
Evaluate in Normalized Space

Avoids all denormalization scale issues by evaluating directly in normalized space.
Both predictions and GT must be in the same normalized space.
"""

import numpy as np
import torch
import sys
from pathlib import Path
import argparse
from scipy import stats
from math import log10
import json
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# RobustHiCPreprocessor for normalizing GT
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


def normalize_to_match(raw_data, Y_median, Y_iqr):
    """Normalize raw data using preprocessor stats"""
    # log1p transform
    log_data = np.log1p(raw_data)
    # Normalize
    norm_data = (log_data - Y_median) / Y_iqr
    # Clip
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
    
    if hasattr(prep, 'Y_mean') and hasattr(prep, 'Y_std'):
        return prep.Y_mean, prep.Y_std
    else:
        raise ValueError("Could not extract Y_mean/Y_std from preprocessor")


# ================================================================
# Metrics (same as evaluate.py)
# ================================================================

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
        else:
            raise ValueError(f"Cannot convert shape {arr.shape} to (N, H, W)")
    else:
        raise ValueError(f"Unexpected array dimension: {arr.ndim}")


def compute_mse(pred, gt):
    return float(np.mean((pred - gt) ** 2))

def compute_mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)))

def compute_pcc(pred, gt):
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(gt_flat) | 
                   np.isinf(pred_flat) | np.isinf(gt_flat))
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]
    if len(pred_flat) == 0:
        return 0.0
    corr, _ = stats.pearsonr(pred_flat, gt_flat)
    return float(corr) if not np.isnan(corr) else 0.0

def compute_spearman(pred, gt):
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(gt_flat))
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]
    if len(pred_flat) == 0:
        return 0.0
    corr, _ = stats.spearmanr(pred_flat, gt_flat)
    return float(corr) if not np.isnan(corr) else 0.0

def compute_ssim(pred, gt):
    """Simple SSIM implementation"""
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    ssim_scores = []
    for i in range(len(pred)):
        p, g = pred[i], gt[i]
        
        # Normalize to similar range
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-8)
        g_norm = (g - g.min()) / (g.max() - g.min() + 1e-8)
        
        mu1, mu2 = np.mean(p_norm), np.mean(g_norm)
        sigma1_sq, sigma2_sq = np.var(p_norm), np.var(g_norm)
        sigma12 = np.mean((p_norm - mu1) * (g_norm - mu2))
        
        C1, C2 = 0.01**2, 0.03**2
        ssim_val = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_scores.append(ssim_val)
    
    return float(np.mean(ssim_scores))

def get_top_k_positions(matrix, k=20):
    flat = matrix.flatten()
    top_k_indices = np.argsort(flat)[-k:][::-1]
    positions = set()
    h, w = matrix.shape[-2], matrix.shape[-1]
    for idx in top_k_indices:
        positions.add((idx // w, idx % w))
    return positions

def compute_top_k_overlap(pred, gt, k=20):
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
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    ious = []
    for i in range(len(pred)):
        pred_thresh = np.percentile(pred[i], threshold_percentile)
        gt_thresh = np.percentile(gt[i], threshold_percentile)
        
        pred_peaks = pred[i] >= pred_thresh
        gt_peaks = gt[i] >= gt_thresh
        
        intersection = np.sum(pred_peaks & gt_peaks)
        union = np.sum(pred_peaks | gt_peaks)
        
        iou = intersection / union if union > 0 else 1.0
        ious.append(iou)
    
    return float(np.mean(ious))

def compute_peak_distance(pred, gt):
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    distances = []
    for i in range(len(pred)):
        pred_peak_idx = np.unravel_index(np.argmax(pred[i]), pred[i].shape)
        gt_peak_idx = np.unravel_index(np.argmax(gt[i]), gt[i].shape)
        
        dist = np.sqrt((pred_peak_idx[0] - gt_peak_idx[0])**2 + 
                       (pred_peak_idx[1] - gt_peak_idx[1])**2)
        distances.append(dist)
    
    return float(np.mean(distances))


def evaluate_normalized(pred, gt, verbose=True):
    """Evaluate in normalized space"""
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    results = {}
    
    if verbose:
        print(f"\n  Shapes: pred={pred.shape}, gt={gt.shape}")
    
    # Reconstruction
    results['mse'] = compute_mse(pred, gt)
    results['mae'] = compute_mae(pred, gt)
    
    # Similarity
    results['pcc'] = compute_pcc(pred, gt)
    results['spearman'] = compute_spearman(pred, gt)
    results['ssim'] = compute_ssim(pred, gt)
    
    # Peak localization
    results['top_5_overlap'] = compute_top_k_overlap(pred, gt, k=5)
    results['top_10_overlap'] = compute_top_k_overlap(pred, gt, k=10)
    results['top_20_overlap'] = compute_top_k_overlap(pred, gt, k=20)
    results['top_50_overlap'] = compute_top_k_overlap(pred, gt, k=50)
    
    # Hit@K
    results['hit_at_1'] = compute_hit_at_k(pred, gt, k=1)
    results['hit_at_3'] = compute_hit_at_k(pred, gt, k=3)
    results['hit_at_5'] = compute_hit_at_k(pred, gt, k=5)
    results['hit_at_10'] = compute_hit_at_k(pred, gt, k=10)
    
    # IoU
    results['iou_90'] = compute_peak_iou(pred, gt, threshold_percentile=90)
    results['iou_95'] = compute_peak_iou(pred, gt, threshold_percentile=95)
    results['iou_99'] = compute_peak_iou(pred, gt, threshold_percentile=99)
    
    # Peak distance
    results['peak_distance'] = compute_peak_distance(pred, gt)
    
    # Stats
    results['pred_range'] = [float(pred.min()), float(pred.max())]
    results['gt_range'] = [float(gt.min()), float(gt.max())]
    results['n_samples'] = int(pred.shape[0])
    
    if verbose:
        print("\n" + "="*60)
        print("RESULTS (NORMALIZED SPACE)")
        print("="*60)
        print(f"\nReconstruction:")
        print(f"  MSE:  {results['mse']:.6f}")
        print(f"  MAE:  {results['mae']:.6f}")
        
        print(f"\nSimilarity:")
        print(f"  PCC:      {results['pcc']:.4f}")
        print(f"  Spearman: {results['spearman']:.4f}")
        print(f"  SSIM:     {results['ssim']:.4f}")
        
        print(f"\nPeak Localization (Top-K Overlap):")
        print(f"  Top-5:  {results['top_5_overlap']*100:.1f}%")
        print(f"  Top-10: {results['top_10_overlap']*100:.1f}%")
        print(f"  Top-20: {results['top_20_overlap']*100:.1f}%")
        print(f"  Top-50: {results['top_50_overlap']*100:.1f}%")
        
        print(f"\nHit@K:")
        print(f"  Hit@1:  {results['hit_at_1']*100:.1f}%")
        print(f"  Hit@3:  {results['hit_at_3']*100:.1f}%")
        print(f"  Hit@5:  {results['hit_at_5']*100:.1f}%")
        print(f"  Hit@10: {results['hit_at_10']*100:.1f}%")
        
        print(f"\nPeak IoU:")
        print(f"  IoU@90%: {results['iou_90']*100:.1f}%")
        print(f"  IoU@95%: {results['iou_95']*100:.1f}%")
        print(f"  IoU@99%: {results['iou_99']*100:.1f}%")
        
        print(f"\nPeak Distance:")
        print(f"  Avg: {results['peak_distance']:.2f} pixels")
        
        print(f"\nData Statistics:")
        print(f"  Samples: {results['n_samples']}")
        print(f"  Pred range: [{results['pred_range'][0]:.3f}, {results['pred_range'][1]:.3f}]")
        print(f"  GT range:   [{results['gt_range'][0]:.3f}, {results['gt_range'][1]:.3f}]")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate in normalized space')
    
    parser.add_argument('--pred_norm_path', type=str, required=True,
                       help='Path to predictions in normalized space')
    parser.add_argument('--gt_path', type=str, required=True,
                       help='Path to ground truth (raw or normalized)')
    parser.add_argument('--preprocessor_path', type=str, default=None,
                       help='Path to preprocessor (if GT needs normalization)')
    parser.add_argument('--gt_is_normalized', action='store_true',
                       help='Set if GT is already in normalized space')
    parser.add_argument('--output_json', type=str, default=None,
                       help='Path to save results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATION IN NORMALIZED SPACE")
    print("="*80)
    
    # Load predictions (normalized)
    print(f"\nLoading predictions: {args.pred_norm_path}")
    pred_norm = np.load(args.pred_norm_path)
    print(f"  Shape: {pred_norm.shape}")
    print(f"  Range: [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
    
    # Load GT
    print(f"\nLoading ground truth: {args.gt_path}")
    gt = np.load(args.gt_path)
    print(f"  Shape: {gt.shape}")
    print(f"  Range: [{gt.min():.4f}, {gt.max():.4f}]")
    
    # Check if GT needs normalization
    if args.gt_is_normalized:
        print("\n  GT is already normalized")
        gt_norm = gt
    elif gt.min() >= -0.1 and gt.max() > 100:
        # GT is in raw space, need to normalize
        print("\n  GT appears to be in raw space, normalizing...")
        
        if args.preprocessor_path:
            Y_median, Y_iqr = load_preprocessor_stats(args.preprocessor_path)
            print(f"  Using preprocessor stats: median={Y_median:.4f}, IQR={Y_iqr:.4f}")
        else:
            # Estimate from GT itself
            gt_squeezed = ensure_nhw(gt)
            gt_log = np.log1p(gt_squeezed)
            Y_median = np.median(gt_log)
            Y_iqr = np.percentile(gt_log, 75) - np.percentile(gt_log, 25) + 1e-8
            print(f"  Estimated from GT: median={Y_median:.4f}, IQR={Y_iqr:.4f}")
        
        gt_norm = normalize_to_match(gt, Y_median, Y_iqr)
        print(f"  Normalized GT range: [{gt_norm.min():.4f}, {gt_norm.max():.4f}]")
    else:
        # GT might already be normalized (has negative values or small range)
        print("\n  GT appears to already be normalized")
        gt_norm = gt
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATING")
    print("="*80)
    
    results = evaluate_normalized(pred_norm, gt_norm, verbose=True)
    
    # Save
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output_json}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    
    return results


if __name__ == '__main__':
    main()
