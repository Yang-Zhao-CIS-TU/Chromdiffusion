"""
Boundary-Focused Diffusion Sampling

CRITICAL OUTPUT STRATEGY:
  - TWO separate outputs for different tasks
  - HiCARN baseline → Loop calling (loops are fragile!)
  - Refined (boundary-focused) → TAD calling (TADs improved!)

This is NOT a limitation - it's the CORRECT usage!

Usage:
    python sample_boundary_focused_diffusion.py \
        --checkpoint checkpoints_boundary_focused/best_boundary_focused.pt \
        --pred_path hicarn_predictions/chr18/predictions_norm.npy \
        --gt_path hicarn_predictions/chr18/ground_truth.npy \
        --output_dir refined_boundary/chr18 \
        --use_ddim \
        --num_inference_steps 50
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Add paths
sys.path.insert(0, 'residual_diffusion')

# Import components
from scheduler import DDPMScheduler, DDIMScheduler
from model import ResidualDiffusionUNet
from data_loader import ResidualNormalizer
from structure_losses_boundary_focused import (
    BoundaryFocusedLossCalculator,
    ResidualClipper
)


def compute_insulation_sharpness(hic_matrix, window_size=5):
    """Compute insulation sharpness for evaluation"""
    H = hic_matrix.shape[-1]
    w = min(window_size, (H - 1) // 2)
    
    if w < 2:
        return 0.0
    
    scores = []
    for i in range(w, H - w):
        block = hic_matrix[..., i-w:i, i:i+w]
        score = block.mean()
        scores.append(score)
    
    if len(scores) < 2:
        return 0.0
    
    insulation = np.array(scores)
    insulation = np.log(insulation + 1e-4)
    gradient = np.abs(np.diff(insulation))
    sharpness = np.var(gradient)
    
    return sharpness


def sample_residuals(
    model,
    hicarn_pred,
    scheduler,
    normalizer,
    residual_clipper,
    loss_calculator,
    device,
    num_inference_steps=50,
    use_ddim=True
):
    """
    Sample residuals with boundary masking
    
    Returns:
        residuals_masked: Only updates near boundaries
    """
    model.eval()
    batch_size = hicarn_pred.shape[0]
    
    # Start from noise
    residual = torch.randn(
        batch_size, 1, hicarn_pred.shape[2], hicarn_pred.shape[3],
        device=device
    )
    
    # Get timesteps
    if use_ddim:
        timesteps = scheduler.timesteps
    else:
        timesteps = list(reversed(range(scheduler.num_train_timesteps)))
    
    # Denoising loop
    for t in tqdm(timesteps, desc="Denoising", leave=False):
        with torch.no_grad():
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = model(residual, t_batch, hicarn_pred)
        
        residual, _ = scheduler.step(noise_pred, t, residual)
    
    # Denormalize
    residual_denorm = normalizer.inverse_transform(residual.cpu().numpy())
    residual_denorm = torch.from_numpy(residual_denorm).to(device)
    
    # Clip
    residual_clipped = residual_clipper.clip_residual(residual_denorm, hicarn_pred)
    
    # CRITICAL: Apply boundary mask
    boundary_mask = loss_calculator.get_boundary_mask(hicarn_pred)
    residual_masked = residual_clipped * boundary_mask
    
    return residual_masked, boundary_mask


def main():
    parser = argparse.ArgumentParser(description="Boundary-Focused Diffusion Sampling")
    
    # Paths
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--gt_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='refined_boundary')
    
    # Sampling
    parser.add_argument('--use_ddim', action='store_true')
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Boundary parameters
    parser.add_argument('--insulation_window', type=int, default=5)
    parser.add_argument('--dilation_radius', type=int, default=2)
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("BOUNDARY-FOCUSED DIFFUSION SAMPLING")
    print("="*80)
    
    # Load checkpoint
    print(f"\n[1/6] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    print("\n[2/6] Creating model...")
    config = checkpoint.get('config', {
        'base_channels': 64,
        'channel_multipliers': [1, 2, 4, 8],
        'num_res_blocks': 2
    })
    
    model = ResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=config['base_channels'],
        channel_multipliers=tuple(config['channel_multipliers']),
        num_res_blocks=config['num_res_blocks']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # Create scheduler
    print(f"\n[3/6] Creating {'DDIM' if args.use_ddim else 'DDPM'} scheduler...")
    if args.use_ddim:
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            num_inference_steps=args.num_inference_steps,
            beta_schedule='linear'
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear'
        )
    
    # Create normalizer
    normalizer = ResidualNormalizer()
    if 'normalizer_mean' in checkpoint:
        normalizer.mean = checkpoint['normalizer_mean']
        normalizer.std = checkpoint['normalizer_std']
        normalizer.fitted = True
        print(f"  Loaded normalizer: mean={normalizer.mean:.6f}, std={normalizer.std:.6f}")
    
    # Create loss calculator (for boundary mask)
    loss_calculator = BoundaryFocusedLossCalculator(
        insulation_window=args.insulation_window,
        use_boundary_mask=True,
        dilation_radius=args.dilation_radius
    ).to(device)
    
    residual_clipper = ResidualClipper(clip_factor=0.1)
    
    # Load HiCARN predictions
    print(f"\n[4/6] Loading HiCARN predictions: {args.pred_path}")
    hicarn_pred = np.load(args.pred_path)
    print(f"  Shape: {hicarn_pred.shape}")
    
    # Load ground truth if available
    if args.gt_path:
        gt_data = np.load(args.gt_path)
        print(f"  Loaded ground truth: {gt_data.shape}")
    
    # Sample residuals
    print(f"\n[5/6] Sampling residuals (boundary-focused)...")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Boundary mask dilation: {args.dilation_radius} bins")
    
    all_residuals = []
    all_masks = []
    sharpness_hicarn_list = []
    sharpness_refined_list = []
    
    num_batches = (len(hicarn_pred) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(0, len(hicarn_pred), args.batch_size), desc="Batches"):
        batch = hicarn_pred[i:i + args.batch_size]
        batch_tensor = torch.from_numpy(batch[:, None, :, :]).float().to(device)
        
        residuals_masked, boundary_mask = sample_residuals(
            model=model,
            hicarn_pred=batch_tensor,
            scheduler=scheduler,
            normalizer=normalizer,
            residual_clipper=residual_clipper,
            loss_calculator=loss_calculator,
            device=device,
            num_inference_steps=args.num_inference_steps,
            use_ddim=args.use_ddim
        )
        
        all_residuals.append(residuals_masked.cpu().numpy()[:, 0, :, :])
        all_masks.append(boundary_mask.cpu().numpy()[:, 0, :, :])
        
        # Compute sharpness
        refined_batch = (batch_tensor + residuals_masked).cpu().numpy()
        for j in range(batch_tensor.shape[0]):
            sharp_hicarn = compute_insulation_sharpness(batch[j])
            sharp_refined = compute_insulation_sharpness(refined_batch[j, 0])
            sharpness_hicarn_list.append(sharp_hicarn)
            sharpness_refined_list.append(sharp_refined)
    
    all_residuals = np.concatenate(all_residuals, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Compute refined predictions
    refined_pred = hicarn_pred + all_residuals
    
    print("\n" + "="*80)
    print("SAVING OUTPUTS (DUAL STRATEGY)")
    print("="*80)
    
    # Save normalized space outputs
    norm_output_path = output_dir / 'predictions_norm.npy'
    np.save(norm_output_path, refined_pred)
    print(f"\n✓ Saved refined predictions (normalized): {norm_output_path}")
    
    # Denormalize if GT available
    if args.gt_path:
        gt_max = gt_data.max()
        refined_raw = refined_pred * gt_max
        refined_raw = np.clip(refined_raw, 0, None)
        
        raw_output_path = output_dir / 'refined_raw.npy'
        np.save(raw_output_path, refined_raw)
        print(f"✓ Saved refined predictions (raw): {raw_output_path}")
    
    # CRITICAL: Save TWO task-specific outputs
    print("\n" + "="*80)
    print("TASK-SPECIFIC OUTPUTS (READ THIS!)")
    print("="*80)
    
    # 1. HiCARN for loops
    hicarn_loop_path = output_dir / 'hicarn_for_loops.npy'
    np.save(hicarn_loop_path, hicarn_pred)
    print(f"\n✓ Saved HiCARN baseline: {hicarn_loop_path}")
    print(f"  → USE THIS for LOOP calling (HiCCUPS, chromosight)")
    print(f"  → Why: Loops are 10kb point features - diffusion smooths them!")
    
    # 2. Refined for TADs
    refined_tad_path = output_dir / 'refined_for_tads.npy'
    np.save(refined_tad_path, refined_pred)
    print(f"\n✓ Saved refined predictions: {refined_tad_path}")
    print(f"  → USE THIS for TAD calling (Arrowhead, TADbit)")
    print(f"  → Why: Diffusion improves domain-scale boundaries!")
    
    # Save residuals and masks
    residual_path = output_dir / 'boundary_residuals.npy'
    np.save(residual_path, all_residuals)
    print(f"\n✓ Saved boundary residuals: {residual_path}")
    
    mask_path = output_dir / 'boundary_masks.npy'
    np.save(mask_path, all_masks)
    print(f"✓ Saved boundary masks: {mask_path}")
    print(f"  Average mask coverage: {all_masks.mean()*100:.1f}% (expect 20-40%)")
    
    # Compute and save statistics
    sharpness_hicarn = np.mean(sharpness_hicarn_list)
    sharpness_refined = np.mean(sharpness_refined_list)
    sharpness_improvement = (sharpness_refined - sharpness_hicarn) / sharpness_hicarn * 100
    
    metadata = {
        'checkpoint': str(args.checkpoint),
        'num_samples': len(hicarn_pred),
        'inference_steps': args.num_inference_steps,
        'use_ddim': args.use_ddim,
        'boundary_config': {
            'insulation_window': args.insulation_window,
            'dilation_radius': args.dilation_radius,
            'avg_mask_coverage': float(all_masks.mean())
        },
        'structure_metrics': {
            'sharpness_hicarn': float(sharpness_hicarn),
            'sharpness_refined': float(sharpness_refined),
            'sharpness_improvement_pct': float(sharpness_improvement)
        },
        'usage_guide': {
            'loop_calling': 'Use hicarn_for_loops.npy with HiCCUPS/chromosight',
            'tad_calling': 'Use refined_for_tads.npy with Arrowhead/TADbit',
            'analysis': 'Use boundary_residuals.npy to see what changed',
            'philosophy': 'Diffusion = TAD refiner, not loop enhancer'
        }
    }
    
    metadata_path = output_dir / 'sampling_stats.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved metadata: {metadata_path}")
    
    print("\n" + "="*80)
    print("STRUCTURE IMPROVEMENT SUMMARY")
    print("="*80)
    print(f"\nInsulation Sharpness:")
    print(f"  HiCARN baseline:  {sharpness_hicarn:.6f}")
    print(f"  Refined (boundary): {sharpness_refined:.6f}")
    print(f"  Improvement:      {sharpness_improvement:+.2f}%")
    
    if sharpness_improvement > 0:
        print(f"\n  ✓ POSITIVE improvement! Boundaries are sharper!")
    else:
        print(f"\n  ⚠ Negative improvement - check training")
    
    print(f"\nBoundary Mask Coverage: {all_masks.mean()*100:.1f}%")
    print(f"  (Expect 20-40% for boundary-focused updates)")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. For TAD calling:")
    print(f"   Use: {refined_tad_path}")
    print(f"   Tool: Arrowhead, TADbit, OnTAD")
    print(f"   Expected: TAD F1 +1.5-3.5%, Jaccard +2-6%")
    
    print("\n2. For Loop calling:")
    print(f"   Use: {hicarn_loop_path}")
    print(f"   Tool: HiCCUPS, chromosight")
    print(f"   Expected: Loop F1 similar to HiCARN (±1%)")
    
    print("\n3. Compare metrics:")
    print(f"   TAD: HiCARN vs Refined (expect improvement)")
    print(f"   Loop: HiCARN vs Refined (expect similar)")
    
    print("\n" + "="*80)
    print("SAMPLING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
