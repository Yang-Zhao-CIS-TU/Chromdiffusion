"""
Re-denormalize predictions using correct ground truth statistics
"""

import numpy as np

print("="*80)
print("RE-DENORMALIZATION WITH CORRECT STATISTICS")
print("="*80)

# 1. Load ground truth to get correct statistics
print("\n1. Computing correct statistics from ground truth...")
gt = np.load('/data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40/hr_test_chr19.npy')
gt_log = np.log1p(gt)

Y_mean_correct = gt_log.mean()
Y_std_correct = gt_log.std()

print(f"   Correct Y_mean: {Y_mean_correct:.4f}")
print(f"   Correct Y_std: {Y_std_correct:.4f}")

# 2. Load normalized predictions
print("\n2. Loading normalized predictions...")
refined_norm = np.load('refined_predictions_tad/chr19/refined_norm.npy')
print(f"   Shape: {refined_norm.shape}")
print(f"   Range: [{refined_norm.min():.2f}, {refined_norm.max():.2f}]")

# 3. Re-denormalize with correct statistics
print("\n3. Re-denormalizing with CORRECT statistics...")

# Clip
refined_norm_clipped = np.clip(refined_norm, -5, 5)

# Denormalize: Y_log = Y_norm * std + mean
refined_log = refined_norm_clipped * Y_std_correct + Y_mean_correct

# Inverse log: Y = exp(Y_log) - 1
refined_raw_corrected = np.expm1(refined_log)

# Ensure non-negative
refined_raw_corrected = np.maximum(refined_raw_corrected, 0.0)

print(f"   New range: [{refined_raw_corrected.min():.2f}, {refined_raw_corrected.max():.2f}]")

# 4. Compare with original
print("\n4. Comparison:")
refined_raw_old = np.load('refined_predictions_tad/chr19/refined_raw.npy')

print(f"   Old refined_raw: [{refined_raw_old.min():.2f}, {refined_raw_old.max():.2f}]")
print(f"   New refined_raw: [{refined_raw_corrected.min():.2f}, {refined_raw_corrected.max():.2f}]")
print(f"   Ground truth:    [{gt.min():.2f}, {gt.max():.2f}]")

scale_factor = refined_raw_corrected.max() / refined_raw_old.max()
print(f"\n   Scale factor: {scale_factor:.2f}x")

# 5. Save corrected version
output_path = 'refined_predictions_tad/chr19/refined_raw_corrected.npy'
np.save(output_path, refined_raw_corrected)
print(f"\n5. ✓ Saved corrected version to: {output_path}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\nNow run evaluation with the corrected file:")
print("\npython evaluate_raw_space_simple.py \\")
print("    --pred_path refined_predictions_tad/chr19/refined_raw_corrected.npy \\")
print("    --gt_path /data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40/hr_test_chr19.npy \\")
print("    --output_json refined_predictions_tad/chr19/metrics_corrected.json")
