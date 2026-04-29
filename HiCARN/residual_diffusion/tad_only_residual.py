"""
TAD-Only Residual Constructor

CORE PHILOSOPHY:
  Diffusion should ONLY modify "loop-insensitive" structure
  = Low-frequency + Loop-masked residual
  
WHY:
  - Loop peaks are handled perfectly by HiCARN
  - Diffusion modifying loops → destroys loop recall
  - We want: TAD↑, Loop≈, PSNR/SSIM≈
  
STRATEGY:
  1. Extract low-frequency residual (preserves TAD blocks, not loop peaks)
  2. Mask out loop regions (loop = HiCARN only)
  3. Diffusion learns TAD-only residual
"""

import torch
import torch.nn.functional as F
import numpy as np


def construct_tad_only_residual(pred, gt, kernel_size=5, loop_percentile=90):
    """
    Construct TAD-only residual that excludes loop peaks
    
    Args:
        pred: HiCARN predictions [N, C, H, W]
        gt: Ground truth [N, C, H, W]
        kernel_size: For low-pass filter (5-9 recommended for 40x40)
        loop_percentile: Percentile for loop detection (90-95)
    
    Returns:
        residual_tad: TAD-only residual [N, C, H, W]
        loop_mask: Binary mask of loop regions [N, C, H, W]
    
    Pipeline:
        residual_raw = gt - pred
        residual_lf = low_pass(residual_raw)
        loop_mask = pred > percentile(pred, 90)
        residual_tad = residual_lf * (1 - loop_mask)
    """
    # 1. Compute raw residual
    residual_raw = gt - pred
    
    # 2. Low-frequency extraction (preserves TAD blocks)
    # Use average pooling to smooth out loop peaks
    residual_lf = F.avg_pool2d(
        residual_raw,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )
    
    # 3. Detect loop regions (high-intensity peaks in pred)
    # These are regions HiCARN already handles well
    # We compute percentile per sample to be adaptive
    loop_masks = []
    for i in range(pred.shape[0]):
        pred_sample = pred[i]
        threshold = torch.quantile(pred_sample.flatten(), loop_percentile / 100.0)
        loop_mask_sample = (pred_sample > threshold).float()
        loop_masks.append(loop_mask_sample)
    
    loop_mask = torch.stack(loop_masks, dim=0)
    
    # 4. Mask out loop regions from residual
    # loop regions: residual = 0 (no modification)
    # TAD regions: residual = low-freq correction
    residual_tad = residual_lf * (1.0 - loop_mask)
    
    return residual_tad, loop_mask


def construct_tad_only_residual_numpy(pred, gt, kernel_size=5, loop_percentile=90):
    """
    NumPy version for dataset construction
    
    Args:
        pred: HiCARN predictions [H, W] or [C, H, W]
        gt: Ground truth [H, W] or [C, H, W]
        kernel_size: For low-pass filter
        loop_percentile: Percentile for loop detection
    
    Returns:
        residual_tad: TAD-only residual
        loop_mask: Binary mask of loop regions
    """
    # Convert to torch
    if len(pred.shape) == 2:
        pred = pred[np.newaxis, np.newaxis, :, :]  # [1, 1, H, W]
        gt = gt[np.newaxis, np.newaxis, :, :]
        squeeze_output = True
    elif len(pred.shape) == 3:
        pred = pred[np.newaxis, :, :, :]  # [1, C, H, W]
        gt = gt[np.newaxis, :, :, :]
        squeeze_output = False
    else:
        squeeze_output = False
    
    # Convert to torch tensors
    pred_torch = torch.from_numpy(pred).float()
    gt_torch = torch.from_numpy(gt).float()
    
    # Construct TAD-only residual
    residual_tad, loop_mask = construct_tad_only_residual(
        pred_torch, gt_torch, kernel_size, loop_percentile
    )
    
    # Convert back to numpy
    residual_tad = residual_tad.numpy()
    loop_mask = loop_mask.numpy()
    
    # Squeeze if needed
    if squeeze_output:
        residual_tad = residual_tad[0, 0]  # [H, W]
        loop_mask = loop_mask[0, 0]
    else:
        residual_tad = residual_tad[0]  # [C, H, W]
        loop_mask = loop_mask[0]
    
    return residual_tad, loop_mask


# Statistics helper
def analyze_tad_residual(residual_raw, residual_tad, loop_mask):
    """
    Analyze the effect of TAD-only residual construction
    
    Returns:
        stats: Dictionary with statistics
    """
    loop_fraction = loop_mask.mean().item()
    
    # Residual statistics
    raw_std = torch.std(residual_raw).item()
    tad_std = torch.std(residual_tad).item()
    
    # Energy in loop vs TAD regions
    loop_energy = (residual_raw * loop_mask).abs().mean().item()
    tad_energy = (residual_raw * (1 - loop_mask)).abs().mean().item()
    
    stats = {
        'loop_fraction': loop_fraction,
        'raw_residual_std': raw_std,
        'tad_residual_std': tad_std,
        'std_reduction': (raw_std - tad_std) / raw_std,
        'loop_energy': loop_energy,
        'tad_energy': tad_energy,
        'energy_ratio': tad_energy / (loop_energy + 1e-8)
    }
    
    return stats


if __name__ == '__main__':
    # Test
    print("Testing TAD-only residual construction...")
    
    # Create dummy data
    pred = torch.randn(2, 1, 40, 40).abs()
    gt = pred + torch.randn(2, 1, 40, 40) * 0.1
    
    # Add some synthetic loop peaks to pred
    pred[:, :, 10:15, 10:15] += 2.0
    pred[:, :, 25:30, 25:30] += 1.5
    
    # Construct TAD-only residual
    residual_tad, loop_mask = construct_tad_only_residual(pred, gt)
    
    print(f"\nInput shapes:")
    print(f"  pred: {pred.shape}")
    print(f"  gt: {gt.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  residual_tad: {residual_tad.shape}")
    print(f"  loop_mask: {loop_mask.shape}")
    
    # Analyze
    residual_raw = gt - pred
    stats = analyze_tad_residual(residual_raw, residual_tad, loop_mask)
    
    print(f"\nStatistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:.4f}")
    
    print("\n✅ TAD-only residual construction working!")
    print(f"   Loop regions masked: {stats['loop_fraction']*100:.1f}%")
    print(f"   Residual std reduced by: {stats['std_reduction']*100:.1f}%")
