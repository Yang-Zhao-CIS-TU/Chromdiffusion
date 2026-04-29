"""
Sanity Check Visualization for Structure-Preserved Diffusion Training

This script should be called during training to visualize:
1. GT (Ground Truth)
2. HiCARN prediction
3. HiCARN + Diffusion refined prediction
4. Insulation curves for all three
5. TAD boundary positions

Usage:
    During training, call this after each epoch:
    
    visualize_sanity_check(
        gt=gt_sample,
        hicarn=hicarn_sample,
        refined=refined_sample,
        epoch=epoch,
        save_dir='visualizations'
    )

This helps verify:
- ✓ Insulation curve is sharper in refined version
- ✓ Loop features are preserved
- ✓ TAD boundaries are better defined
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def compute_insulation_profile(mat, window=5):
    """
    Compute insulation score for visualization
    
    Args:
        mat: Hi-C matrix (H, W) numpy array
        window: Window size
    
    Returns:
        positions: Genomic positions
        insulation: Insulation scores
    """
    H = mat.shape[0]
    w = min(window, (H - 1) // 2)
    
    if w < 2:
        return None, None
    
    scores = []
    positions = []
    
    for i in range(w, H - w):
        block = mat[i-w:i, i:i+w]
        score = block.mean()
        scores.append(score)
        positions.append(i)
    
    scores = np.array(scores)
    positions = np.array(positions)
    
    # Log transform and clamp
    insulation = np.log(scores + 1e-4)
    insulation = np.clip(insulation, -5, 5)
    
    return positions, insulation


def find_tad_boundaries(insulation, positions, threshold_percentile=10):
    """
    Find TAD boundaries as local minima in insulation
    
    Args:
        insulation: Insulation scores
        positions: Genomic positions
        threshold_percentile: Percentile threshold for calling boundaries
    
    Returns:
        boundary_positions: Positions of TAD boundaries
    """
    if insulation is None or len(insulation) < 3:
        return []
    
    # Find local minima
    boundaries = []
    threshold = np.percentile(insulation, threshold_percentile)
    
    for i in range(1, len(insulation) - 1):
        # Local minimum
        if insulation[i] < insulation[i-1] and insulation[i] < insulation[i+1]:
            # Below threshold
            if insulation[i] < threshold:
                boundaries.append(positions[i])
    
    return boundaries


def visualize_sanity_check(
    gt,
    hicarn,
    refined,
    epoch,
    save_dir='visualizations',
    window=5,
    sample_idx=0
):
    """
    Create sanity check visualization
    
    Args:
        gt: Ground truth Hi-C (1, H, W) or (H, W)
        hicarn: HiCARN prediction (1, H, W) or (H, W)
        refined: HiCARN + Diffusion (1, H, W) or (H, W)
        epoch: Current epoch number
        save_dir: Directory to save visualizations
        window: Insulation window size
        sample_idx: Sample index (for batch processing)
    """
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy and squeeze
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()
    if torch.is_tensor(hicarn):
        hicarn = hicarn.cpu().numpy()
    if torch.is_tensor(refined):
        refined = refined.cpu().numpy()
    
    # Squeeze channel dimension if present
    if gt.ndim == 3:
        gt = gt[0]
    if hicarn.ndim == 3:
        hicarn = hicarn[0]
    if refined.ndim == 3:
        refined = refined[0]
    
    # Compute insulation profiles
    pos_gt, ins_gt = compute_insulation_profile(gt, window)
    pos_hicarn, ins_hicarn = compute_insulation_profile(hicarn, window)
    pos_refined, ins_refined = compute_insulation_profile(refined, window)
    
    # Find TAD boundaries
    boundaries_gt = find_tad_boundaries(ins_gt, pos_gt) if ins_gt is not None else []
    boundaries_hicarn = find_tad_boundaries(ins_hicarn, pos_hicarn) if ins_hicarn is not None else []
    boundaries_refined = find_tad_boundaries(ins_refined, pos_refined) if ins_refined is not None else []
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # Define grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Hi-C matrices
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Row 2: Insulation curves
    ax4 = fig.add_subplot(gs[1, :])
    
    # Row 3: Insulation gradient (boundary strength)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Plot Hi-C matrices
    vmax = np.percentile(gt, 99)
    
    im1 = ax1.imshow(gt, cmap='Reds', vmin=0, vmax=vmax)
    ax1.set_title(f'Ground Truth\nTAD Boundaries: {len(boundaries_gt)}')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    im2 = ax2.imshow(hicarn, cmap='Reds', vmin=0, vmax=vmax)
    ax2.set_title(f'HiCARN\nTAD Boundaries: {len(boundaries_hicarn)}')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    im3 = ax3.imshow(refined, cmap='Reds', vmin=0, vmax=vmax)
    ax3.set_title(f'HiCARN + Diffusion\nTAD Boundaries: {len(boundaries_refined)}')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Plot insulation curves
    if ins_gt is not None:
        ax4.plot(pos_gt, ins_gt, 'b-', label='GT', linewidth=2, alpha=0.7)
    if ins_hicarn is not None:
        ax4.plot(pos_hicarn, ins_hicarn, 'g-', label='HiCARN', linewidth=2, alpha=0.7)
    if ins_refined is not None:
        ax4.plot(pos_refined, ins_refined, 'r-', label='HiCARN+Diff', linewidth=2, alpha=0.7)
    
    # Mark TAD boundaries
    for b in boundaries_gt:
        ax4.axvline(b, color='blue', linestyle='--', alpha=0.3, linewidth=1)
    for b in boundaries_refined:
        ax4.axvline(b, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    ax4.set_xlabel('Genomic Position (bins)')
    ax4.set_ylabel('Insulation Score (log scale)')
    ax4.set_title('Insulation Curves - KEY METRIC: Refined should be sharper!')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Plot insulation gradients (boundary strength)
    if ins_gt is not None and len(ins_gt) > 1:
        grad_gt = np.abs(np.diff(ins_gt))
        ax5.plot(pos_gt[:-1], grad_gt, 'b-', label='GT Gradient', linewidth=2, alpha=0.7)
    
    if ins_hicarn is not None and len(ins_hicarn) > 1:
        grad_hicarn = np.abs(np.diff(ins_hicarn))
        ax5.plot(pos_hicarn[:-1], grad_hicarn, 'g-', label='HiCARN Gradient', linewidth=2, alpha=0.7)
    
    if ins_refined is not None and len(ins_refined) > 1:
        grad_refined = np.abs(np.diff(ins_refined))
        ax5.plot(pos_refined[:-1], grad_refined, 'r-', label='HiCARN+Diff Gradient', linewidth=2, alpha=0.7)
    
    ax5.set_xlabel('Genomic Position (bins)')
    ax5.set_ylabel('|dInsulation/dPosition|')
    ax5.set_title('Boundary Strength - KEY METRIC: Higher peaks = sharper boundaries!')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(f'Epoch {epoch} - Sanity Check (Sample {sample_idx})', 
                 fontsize=16, fontweight='bold')
    
    # Save figure
    save_path = save_dir / f'sanity_check_epoch_{epoch:03d}_sample_{sample_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Saved sanity check: {save_path}")
    
    # Compute and return metrics
    metrics = {
        'n_boundaries_gt': len(boundaries_gt),
        'n_boundaries_hicarn': len(boundaries_hicarn),
        'n_boundaries_refined': len(boundaries_refined)
    }
    
    if ins_refined is not None and ins_hicarn is not None and len(ins_refined) > 0:
        # Boundary sharpness improvement
        grad_hicarn_mean = np.mean(np.abs(np.diff(ins_hicarn))) if len(ins_hicarn) > 1 else 0
        grad_refined_mean = np.mean(np.abs(np.diff(ins_refined))) if len(ins_refined) > 1 else 0
        
        if grad_hicarn_mean > 0:
            metrics['boundary_sharpness_improvement'] = (grad_refined_mean / grad_hicarn_mean - 1) * 100
        else:
            metrics['boundary_sharpness_improvement'] = 0
    
    return metrics


def visualize_batch_sanity_check(
    gt_batch,
    hicarn_batch,
    refined_batch,
    epoch,
    save_dir='visualizations',
    n_samples=3
):
    """
    Visualize multiple samples from batch
    
    Args:
        gt_batch: Ground truth batch (B, 1, H, W)
        hicarn_batch: HiCARN batch (B, 1, H, W)
        refined_batch: Refined batch (B, 1, H, W)
        epoch: Current epoch
        save_dir: Save directory
        n_samples: Number of samples to visualize
    """
    batch_size = gt_batch.shape[0]
    n_samples = min(n_samples, batch_size)
    
    all_metrics = []
    
    for i in range(n_samples):
        metrics = visualize_sanity_check(
            gt=gt_batch[i],
            hicarn=hicarn_batch[i],
            refined=refined_batch[i],
            epoch=epoch,
            save_dir=save_dir,
            sample_idx=i
        )
        all_metrics.append(metrics)
    
    # Print summary
    print(f"\n  📊 Sanity Check Summary (Epoch {epoch}):")
    print(f"     Average TAD boundaries - GT: {np.mean([m['n_boundaries_gt'] for m in all_metrics]):.1f}")
    print(f"     Average TAD boundaries - HiCARN: {np.mean([m['n_boundaries_hicarn'] for m in all_metrics]):.1f}")
    print(f"     Average TAD boundaries - Refined: {np.mean([m['n_boundaries_refined'] for m in all_metrics]):.1f}")
    
    if 'boundary_sharpness_improvement' in all_metrics[0]:
        avg_improvement = np.mean([m['boundary_sharpness_improvement'] for m in all_metrics])
        print(f"     Boundary sharpness improvement: {avg_improvement:+.1f}%")
        
        if avg_improvement > 0:
            print(f"     ✓ Diffusion is IMPROVING boundary sharpness!")
        else:
            print(f"     ⚠ Diffusion is not improving boundaries (yet)")


if __name__ == "__main__":
    # Test visualization
    print("Testing Sanity Check Visualization")
    print("="*80)
    
    # Create dummy data
    H = 40
    
    # GT: sharp TAD boundaries
    gt = np.random.rand(H, H) * 50
    for i in range(0, H, 10):
        gt[i:i+2, :] *= 0.5  # Create boundaries
        gt[:, i:i+2] *= 0.5
    
    # HiCARN: slightly blurred
    from scipy.ndimage import gaussian_filter
    hicarn = gaussian_filter(gt, sigma=0.5)
    
    # Refined: sharper boundaries
    refined = gt + (gt - hicarn) * 0.5
    
    # Visualize
    metrics = visualize_sanity_check(
        gt=gt,
        hicarn=hicarn,
        refined=refined,
        epoch=0,
        save_dir='test_visualizations'
    )
    
    print("\n" + "="*80)
    print("✓ Test visualization saved to test_visualizations/")
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS TO LOOK FOR:")
    print("  1. Insulation curve for refined is SHARPER than HiCARN")
    print("  2. Boundary gradient peaks are HIGHER for refined")
    print("  3. Number of TAD boundaries is similar (not more, not less)")
    print("  4. Loop features are PRESERVED in Hi-C matrices")
    print("\nIf you see these → You're on the RIGHT PATH! 🎯")
