#!/usr/bin/env python3
"""
Quick Data Validation Script

Run this BEFORE training to check if data is clean
"""

import numpy as np
import sys

print("="*80)
print("DATA VALIDATION CHECK")
print("="*80)

# Load data
print("\n[1/3] Loading data...")
try:
    pred = np.load('hicarn_predictions/predictions_norm.npy')
    gt = np.load('hicarn_predictions/ground_truth.npy')
    print(f"  ✓ Loaded successfully")
    print(f"    Predictions: {pred.shape}")
    print(f"    Ground truth: {gt.shape}")
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    sys.exit(1)

# Check for NaN/Inf
print("\n[2/3] Checking for NaN/Inf...")
pred_has_nan = np.isnan(pred).any()
pred_has_inf = np.isinf(pred).any()
gt_has_nan = np.isnan(gt).any()
gt_has_inf = np.isinf(gt).any()

if pred_has_nan or pred_has_inf:
    print(f"  ✗ Predictions contain NaN: {pred_has_nan}, Inf: {pred_has_inf}")
    print(f"    NaN count: {np.isnan(pred).sum()}")
    print(f"    Inf count: {np.isinf(pred).sum()}")
    print("\n  ACTION NEEDED: Clean predictions before training!")
else:
    print(f"  ✓ Predictions are clean (no NaN/Inf)")

if gt_has_nan or gt_has_inf:
    print(f"  ✗ Ground truth contains NaN: {gt_has_nan}, Inf: {gt_has_inf}")
    print(f"    NaN count: {np.isnan(gt).sum()}")
    print(f"    Inf count: {np.isinf(gt).sum()}")
    print("\n  ACTION NEEDED: Clean ground truth before training!")
else:
    print(f"  ✓ Ground truth is clean (no NaN/Inf)")

# Check value ranges
print("\n[3/3] Checking value ranges...")
print(f"  Predictions:")
print(f"    Min: {pred.min():.6f}")
print(f"    Max: {pred.max():.6f}")
print(f"    Mean: {pred.mean():.6f}")
print(f"    Std: {pred.std():.6f}")

print(f"  Ground truth:")
print(f"    Min: {gt.min():.6f}")
print(f"    Max: {gt.max():.6f}")
print(f"    Mean: {gt.mean():.6f}")
print(f"    Std: {gt.std():.6f}")

# Check residuals
print("\n[4/3] Checking residuals...")
residual = gt - pred
print(f"  Residuals:")
print(f"    Min: {residual.min():.6f}")
print(f"    Max: {residual.max():.6f}")
print(f"    Mean: {residual.mean():.6f}")
print(f"    Std: {residual.std():.6f}")
print(f"    Variance: {residual.var():.6f}")

if residual.var() < 1e-6:
    print(f"\n  ⚠ WARNING: Residual variance is VERY low!")
    print(f"    This means pred ≈ gt (almost identical)")
    print(f"    Diffusion may have nothing to learn!")

# Final verdict
print("\n" + "="*80)
if pred_has_nan or pred_has_inf or gt_has_nan or gt_has_inf:
    print("✗ DATA VALIDATION FAILED")
    print("\nACTION: Clean your data before training!")
    print("\nTo clean:")
    print("  pred_clean = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)")
    print("  gt_clean = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)")
    print("  np.save('hicarn_predictions/predictions_norm_clean.npy', pred_clean)")
    print("  np.save('hicarn_predictions/ground_truth_clean.npy', gt_clean)")
    sys.exit(1)
else:
    print("✓ DATA VALIDATION PASSED")
    print("\nYour data is clean and ready for training!")
    print("="*80)
    sys.exit(0)
