"""
Sample from Structure-Preserved Diffusion Model (STABLE VERSION)

This script generates refined Hi-C predictions using the trained diffusion model
with STABLE structure-oriented losses.

CRITICAL: Outputs TWO separate files for different tasks:
  - hicarn_for_loop_calling.npy    → Use for loop calling (HiCCUPS)
  - refined_for_tad_calling.npy    → Use for TAD calling (Arrowhead)
  - diffusion_residuals.npy         → Diffusion-predicted residuals
  - sampling_stats.json             → Metadata and usage guide

Compatible with models trained using:
  - structure_losses_stable.py
  - train_structure_preserved_diffusion_multigpu.py
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Add residual_diffusion to path
sys.path.insert(0, 'residual_diffusion')

# Import local scheduler and model (same as training script)
from scheduler import DDPMScheduler, DDIMScheduler
from model import ResidualDiffusionUNet
from data_loader import ResidualNormalizer

# Import stable losses (for ResidualClipper)
from structure_losses_stable import ResidualClipper


def sample_residuals(
    model,
    hicarn_pred,
    scheduler,
    normalizer,
    residual_clipper,
    device,
    num_inference_steps=50,
    use_ddim=True
):
    """
    Sample residuals using the trained diffusion model
    
    Args:
        model: Trained ResidualDiffusionUNet
        hicarn_pred: HiCARN predictions (B, 1, H, W)
        scheduler: Diffusion scheduler (DDPM or DDIM)
        normalizer: Residual normalizer
        residual_clipper: Residual clipper
        device: Device to run on
        num_inference_steps: Number of denoising steps
        use_ddim: Whether to use DDIM (faster)
    
    Returns:
        residuals: Predicted residuals in original space (B, 1, H, W)
    """
    model.eval()
    batch_size = hicarn_pred.shape[0]
    
    # Start from random noise (in normalized space)
    residual = torch.randn(
        batch_size, 1, hicarn_pred.shape[2], hicarn_pred.shape[3],
        device=device
    )
    
    # Get timesteps
    if use_ddim:
        # DDIM has pre-defined timesteps
        timesteps = scheduler.timesteps
    else:
        # DDPM uses all timesteps
        timesteps = list(reversed(range(scheduler.num_train_timesteps)))
    
    # Denoising loop
    for t in tqdm(timesteps, desc="Denoising", leave=False):
        # Predict noise
        with torch.no_grad():
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = model(residual, t_batch, hicarn_pred)
        
        # Denoise step - local scheduler returns (prev_sample, pred_original_sample)
        residual, _ = scheduler.step(noise_pred, t, residual)
    
    # Denormalize residuals
    residual_denorm = normalizer.inverse_transform(residual.cpu().numpy())
    residual_denorm = torch.from_numpy(residual_denorm).to(device)
    
    # CRITICAL: Clip residuals to prevent large changes
    residual_clipped = residual_clipper.clip_residual(residual_denorm, hicarn_pred)
    
    return residual_clipped


def main():
    parser = argparse.ArgumentParser(description="Sample from Structure-Preserved Diffusion (STABLE)")
    
    # Paths
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to HiCARN predictions (.npy)')
    parser.add_argument('--gt_path', type=str, default=None,
                       help='Path to ground truth (for denormalization to raw space)')
    parser.add_argument('--output_dir', type=str, default='refined_predictions_stable',
                       help='Output directory for refined predictions')
    
    # Sampling
    parser.add_argument('--use_ddim', action='store_true',
                       help='Use DDIM (faster) instead of DDPM')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of denoising steps (default: 50)')
    
    # Device
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("STRUCTURE-PRESERVED DIFFUSION SAMPLING (STABLE VERSION)")
    print("="*80)
    
    # Load checkpoint
    print(f"\n[1/5] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model config from checkpoint (if available)
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"  Using model config from checkpoint")
    else:
        # Default config (should match training)
        config = {
            'base_channels': 64,
            'channel_multipliers': [1, 2, 4, 8],
            'num_res_blocks': 2
        }
        print(f"  Using default model config")
    
    # Create model
    print("\n[2/5] Creating model...")
    model = ResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=config.get('base_channels', 64),
        channel_multipliers=tuple(config.get('channel_multipliers', [1, 2, 4, 8])),
        num_res_blocks=config.get('num_res_blocks', 2)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # Create scheduler
    print(f"\n[3/5] Creating {'DDIM' if args.use_ddim else 'DDPM'} scheduler...")
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
    if 'normalizer_mean' in checkpoint and 'normalizer_std' in checkpoint:
        normalizer.mean = checkpoint['normalizer_mean']
        normalizer.std = checkpoint['normalizer_std']
        normalizer.fitted = True
        print(f"  Loaded normalizer: mean={normalizer.mean:.6f}, std={normalizer.std:.6f}")
    else:
        print("  ⚠ Warning: No normalizer stats in checkpoint")
    
    # Create residual clipper
    residual_clipper = ResidualClipper(clip_factor=0.1)
    
    # Load HiCARN predictions
    print(f"\n[4/5] Loading HiCARN predictions: {args.pred_path}")
    hicarn_pred = np.load(args.pred_path)
    print(f"  Shape: {hicarn_pred.shape}")
    print(f"  Range: [{hicarn_pred.min():.4f}, {hicarn_pred.max():.4f}]")
    
    # Sample residuals
    print(f"\n[5/5] Sampling residuals...")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Batch size: {args.batch_size}")
    
    all_residuals = []
    num_batches = (len(hicarn_pred) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(0, len(hicarn_pred), args.batch_size), desc="Batches"):
        batch = hicarn_pred[i:i + args.batch_size]
        batch_tensor = torch.from_numpy(batch[:, None, :, :]).float().to(device)
        
        residuals = sample_residuals(
            model=model,
            hicarn_pred=batch_tensor,
            scheduler=scheduler,
            normalizer=normalizer,
            residual_clipper=residual_clipper,
            device=device,
            num_inference_steps=args.num_inference_steps,
            use_ddim=args.use_ddim
        )
        
        all_residuals.append(residuals.cpu().numpy()[:, 0, :, :])
    
    all_residuals = np.concatenate(all_residuals, axis=0)
    
    # Compute refined predictions
    refined_pred = hicarn_pred + all_residuals
    
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Save refined predictions in NORMALIZED space
    norm_output_path = output_dir / 'predictions_norm.npy'
    np.save(norm_output_path, refined_pred)
    print(f"\n✓ Saved refined predictions (normalized): {norm_output_path}")
    print(f"  Shape: {refined_pred.shape}")
    print(f"  Range: [{refined_pred.min():.4f}, {refined_pred.max():.4f}]")
    
    # Denormalize to RAW space if ground truth provided
    if args.gt_path:
        print(f"\n[Denormalization] Loading ground truth: {args.gt_path}")
        gt_data = np.load(args.gt_path)
        print(f"  Ground truth shape: {gt_data.shape}")
        
        # Compute normalization parameters from ground truth
        gt_mean = gt_data.mean()
        gt_std = gt_data.std()
        gt_max = gt_data.max()
        
        print(f"  Ground truth stats:")
        print(f"    Mean: {gt_mean:.4f}")
        print(f"    Std:  {gt_std:.4f}")
        print(f"    Max:  {gt_max:.4f}")
        
        # Denormalize: reverse the normalization
        # Assuming normalization was: (x - mean) / std or x / max
        # Try both methods and check which makes sense
        
        # Method 1: Standardization (x - mean) / std
        refined_raw_method1 = refined_pred * gt_std + gt_mean
        
        # Method 2: Max normalization x / max
        refined_raw_method2 = refined_pred * gt_max
        
        # Use method 2 (max normalization) as it's more common for Hi-C
        refined_raw = refined_raw_method2
        
        # Clip to non-negative (Hi-C contact counts can't be negative)
        refined_raw = np.clip(refined_raw, 0, None)
        
        raw_output_path = output_dir / 'refined_raw.npy'
        np.save(raw_output_path, refined_raw)
        print(f"\n✓ Saved refined predictions (raw space): {raw_output_path}")
        print(f"  Shape: {refined_raw.shape}")
        print(f"  Range: [{refined_raw.min():.4f}, {refined_raw.max():.4f}]")
        print(f"  Denormalization method: Max normalization (x_raw = x_norm * {gt_max:.4f})")
    else:
        print(f"\n⚠ No ground truth provided - skipping raw space conversion")
        print(f"  To generate refined_raw.npy, provide --gt_path argument")
    
    # CRITICAL: Save dual outputs for different tasks
    # 1. HiCARN baseline (for loop calling)
    hicarn_output_path = output_dir / 'hicarn_for_loop_calling.npy'
    np.save(hicarn_output_path, hicarn_pred)
    print(f"\n✓ Saved HiCARN baseline: {hicarn_output_path}")
    print(f"  → USE THIS for LOOP CALLING (HiCCUPS)")
    print(f"  → Loops are easily destroyed by diffusion!")
    
    # 2. Refined predictions (for TAD calling) - same as predictions_norm.npy
    refined_output_path = output_dir / 'refined_for_tad_calling.npy'
    np.save(refined_output_path, refined_pred)
    print(f"\n✓ Saved refined predictions: {refined_output_path}")
    print(f"  → USE THIS for TAD CALLING (Arrowhead)")
    print(f"  → Same as predictions_norm.npy")
    
    # 3. Save residuals (for analysis)
    residual_output_path = output_dir / 'diffusion_residuals.npy'
    np.save(residual_output_path, all_residuals)
    print(f"\n✓ Saved diffusion residuals: {residual_output_path}")
    print(f"  → For analysis and visualization")
    
    # Save metadata
    metadata = {
        'checkpoint': str(args.checkpoint),
        'num_samples': len(hicarn_pred),
        'inference_steps': args.num_inference_steps,
        'use_ddim': args.use_ddim,
        'residual_stats': {
            'mean': float(all_residuals.mean()),
            'std': float(all_residuals.std()),
            'min': float(all_residuals.min()),
            'max': float(all_residuals.max())
        },
        'refined_norm_stats': {
            'mean': float(refined_pred.mean()),
            'std': float(refined_pred.std()),
            'min': float(refined_pred.min()),
            'max': float(refined_pred.max())
        },
        'usage_guide': {
            'normalized_space': 'predictions_norm.npy = refined predictions in normalized space',
            'loop_calling': 'hicarn_for_loop_calling.npy = use with HiCCUPS',
            'tad_calling': 'refined_for_tad_calling.npy = use with Arrowhead (same as predictions_norm.npy)',
            'analysis': 'diffusion_residuals.npy = for SCC analysis'
        }
    }
    
    # Add raw space info if denormalized
    if args.gt_path:
        metadata['refined_raw_stats'] = {
            'mean': float(refined_raw.mean()),
            'std': float(refined_raw.std()),
            'min': float(refined_raw.min()),
            'max': float(refined_raw.max())
        }
        metadata['denormalization'] = {
            'method': 'max_normalization',
            'gt_max': float(gt_max),
            'formula': 'x_raw = x_norm * gt_max'
        }
        metadata['usage_guide']['raw_space'] = 'refined_raw.npy = refined predictions in raw contact count space'
    
    metadata_path = output_dir / 'sampling_stats.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved metadata: {metadata_path}")
    
    print("\n" + "="*80)
    print("SAMPLING COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("\nKEY OUTPUT FILES:")
    print("  • predictions_norm.npy         - Refined predictions (normalized space)")
    if args.gt_path:
        print("  • refined_raw.npy              - Refined predictions (raw contact counts)")
    print("\nTASK-SPECIFIC FILES:")
    print("  • hicarn_for_loop_calling.npy  - For LOOP calling (HiCCUPS)")
    print("  • refined_for_tad_calling.npy  - For TAD calling (Arrowhead)")
    print("\nANALYSIS FILES:")
    print("  • diffusion_residuals.npy      - Diffusion residuals")
    print("  • sampling_stats.json          - Metadata")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
