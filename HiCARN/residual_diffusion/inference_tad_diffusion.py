"""
Inference Script for TAD-Only Residual Diffusion
Modified to output predictions_norm.npy and refine_raw.npy

Usage:
    python inference_tad_diffusion.py \
        --checkpoint checkpoints_tad_only/checkpoint_epoch_50.pt \
        --pred_path hicarn_predictions/chr19/predictions_norm.npy \
        --gt_path hicarn_predictions/chr19/ground_truth.npy \
        --output_dir refined_predictions_structure/chr19 \
        --num_steps 50 \
        --loop_percentile 90 \
        --gpu 0

Output:
    refined_predictions_structure/chr19/predictions_norm.npy  (refined, normalized)
    refined_predictions_structure/chr19/refine_raw.npy  (refined, denormalized)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

# Import model and scheduler
import sys
sys.path.insert(0, 'residual_diffusion')
from model import ResidualDiffusionUNet
from scheduler import DDPMScheduler, DDIMScheduler

# Import TAD-only residual constructor
from tad_only_residual import construct_tad_only_residual


def load_checkpoint(checkpoint_path, device):
    """Load trained diffusion model"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint.get('config', {})
    
    # Initialize model
    model = ResidualDiffusionUNet(
        in_channels=config.get('in_channels', 1),
        out_channels=config.get('out_channels', 1),
        base_channels=config.get('base_channels', 64),
        time_emb_dim=config.get('time_emb_dim', 256)
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config


def sample_tad_residual(
    model,
    scheduler,
    hicarn_pred,
    device,
    num_inference_steps=50,
    guidance_scale=0.0
):
    """
    Sample TAD-only residual using DDIM
    
    Args:
        model: Trained diffusion model
        scheduler: DDIM scheduler
        hicarn_pred: HiCARN prediction [1, C, H, W]
        device: torch device
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance (0.0 = no guidance)
    
    Returns:
        residual_tad: Sampled TAD-only residual [1, C, H, W]
    """
    # Start from pure noise
    residual_tad = torch.randn_like(hicarn_pred).to(device)
    
    # Use scheduler's timesteps (DDIMScheduler has this attribute)
    timesteps = scheduler.timesteps
    
    # Denoising loop
    for t in timesteps:
        # Prepare timestep
        if isinstance(t, torch.Tensor):
            t_tensor = t.unsqueeze(0).to(device)
        else:
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        
        # Model prediction
        with torch.no_grad():
            # Predict noise
            noise_pred = model(residual_tad, t_tensor, hicarn_pred)
            
            # Optionally apply classifier-free guidance
            if guidance_scale > 0.0:
                # Unconditional prediction (no condition)
                noise_pred_uncond = model(
                    residual_tad,
                    t_tensor,
                    torch.zeros_like(hicarn_pred)
                )
                
                # Guided prediction
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred - noise_pred_uncond
                )
        
        # Denoise step (DDIM returns prev_sample, pred_original_sample)
        residual_tad, _ = scheduler.step(noise_pred, t, residual_tad, eta=0.0)
    
    return residual_tad


def apply_loop_masking(residual_tad, hicarn_pred, loop_percentile=90):
    """
    Apply loop masking to ensure loops are NOT modified
    
    CRITICAL for preserving loop detection performance
    
    Args:
        residual_tad: Sampled residual [1, C, H, W]
        hicarn_pred: HiCARN prediction [1, C, H, W]
        loop_percentile: Percentile for loop detection
    
    Returns:
        residual_masked: Loop-masked residual
        loop_mask: Binary mask of loop regions
    """
    # Detect loop regions (high-intensity peaks)
    threshold = torch.quantile(hicarn_pred.flatten(), loop_percentile / 100.0)
    loop_mask = (hicarn_pred > threshold).float()
    
    # Mask out loop regions
    residual_masked = residual_tad * (1.0 - loop_mask)
    
    # Statistics
    loop_fraction = loop_mask.mean().item()
    
    return residual_masked, loop_mask, loop_fraction


def refine_hic_batch(
    hicarn_pred_batch,
    model,
    scheduler,
    device,
    num_inference_steps=50,
    guidance_scale=0.0,
    loop_percentile=90,
    apply_masking=True,
    show_progress=True
):
    """
    Refine a batch of Hi-C matrices
    
    Args:
        hicarn_pred_batch: HiCARN predictions [N, C, H, W]
        model: Trained diffusion model
        scheduler: DDPM scheduler
        device: torch device
        num_inference_steps: Sampling steps
        guidance_scale: CFG scale
        loop_percentile: Loop masking threshold
        apply_masking: Whether to apply loop masking
        show_progress: Show progress bar
    
    Returns:
        refined_batch: Refined Hi-C matrices [N, C, H, W]
        residuals_batch: TAD-only residuals [N, C, H, W]
        loop_masks_batch: Loop masks [N, C, H, W] (if apply_masking=True)
    """
    refined_list = []
    residuals_list = []
    loop_masks_list = []
    
    iterator = tqdm(range(len(hicarn_pred_batch)), desc="Refining") if show_progress else range(len(hicarn_pred_batch))
    
    for i in iterator:
        hicarn_pred = hicarn_pred_batch[i:i+1]  # Keep batch dim [1, C, H, W]
        
        # Convert to tensor if needed
        if not isinstance(hicarn_pred, torch.Tensor):
            hicarn_pred = torch.from_numpy(hicarn_pred).float()
        
        hicarn_pred = hicarn_pred.to(device)
        
        # Sample TAD-only residual
        residual_tad = sample_tad_residual(
            model,
            scheduler,
            hicarn_pred,
            device,
            num_inference_steps,
            guidance_scale
        )
        
        # Apply loop masking (CRITICAL)
        if apply_masking:
            residual_tad, loop_mask, _ = apply_loop_masking(
                residual_tad,
                hicarn_pred,
                loop_percentile
            )
        else:
            loop_mask = None
        
        # Add residual to HiCARN prediction
        refined_hic = hicarn_pred + residual_tad
        
        # Ensure non-negative
        refined_hic = torch.clamp(refined_hic, min=0.0)
        
        # Store
        refined_list.append(refined_hic.cpu())
        residuals_list.append(residual_tad.cpu())
        if loop_mask is not None:
            loop_masks_list.append(loop_mask.cpu())
    
    # Stack
    refined_batch = torch.cat(refined_list, dim=0)
    residuals_batch = torch.cat(residuals_list, dim=0)
    if loop_masks_list:
        loop_masks_batch = torch.cat(loop_masks_list, dim=0)
    else:
        loop_masks_batch = None
    
    return refined_batch, residuals_batch, loop_masks_batch


def denormalize_hic(normalized_data, original_data):
    """
    Denormalize refined Hi-C data back to raw scale
    
    Args:
        normalized_data: Normalized refined data [N, C, H, W]
        original_data: Original raw data for scale reference [N, C, H, W]
    
    Returns:
        denormalized_data: Data in original scale
    """
    # Compute original scale statistics
    original_min = original_data.min()
    original_max = original_data.max()
    
    # Assuming normalization was: (data - min) / (max - min)
    # Denormalize: data * (max - min) + min
    denormalized = normalized_data * (original_max - original_min) + original_min
    
    return denormalized


def main(args):
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"TAD-ONLY RESIDUAL DIFFUSION INFERENCE")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Load checkpoint
    model, config = load_checkpoint(args.checkpoint, device)
    
    # Initialize DDIM scheduler for fast sampling
    scheduler = DDIMScheduler(
        num_train_timesteps=config.get('num_train_timesteps', 1000),
        num_inference_steps=args.num_steps,
        beta_schedule=config.get('beta_schedule', 'linear')
    )
    
    # Load HiCARN predictions (normalized)
    print(f"\n{'='*60}")
    print(f"Loading data...")
    print(f"{'='*60}")
    print(f"HiCARN predictions: {args.pred_path}")
    pred_norm = np.load(args.pred_path)
    print(f"  Shape: {pred_norm.shape}")
    print(f"  Range: [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
    
    # Load raw data if available (for denormalization)
    if args.raw_path:
        print(f"\nRaw HiCARN predictions: {args.raw_path}")
        pred_raw = np.load(args.raw_path)
        print(f"  Shape: {pred_raw.shape}")
        print(f"  Range: [{pred_raw.min():.4f}, {pred_raw.max():.4f}]")
    else:
        pred_raw = None
        print(f"\n⚠️  No raw data provided, will estimate denormalization")
    
    # Optionally load ground truth for quick stats
    if args.gt_path:
        print(f"\nGround truth: {args.gt_path}")
        gt_data = np.load(args.gt_path)
        print(f"  Shape: {gt_data.shape}")
    else:
        gt_data = None
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all samples
    print(f"\n{'='*60}")
    print(f"Processing {len(pred_norm)} samples...")
    print(f"Parameters:")
    print(f"  Denoising steps: {args.num_steps}")
    print(f"  Loop percentile: {args.loop_percentile}")
    print(f"  Loop masking: {'Enabled' if args.apply_masking else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # Ensure 4D [N, C, H, W]
    if pred_norm.ndim == 3:
        pred_norm = pred_norm[:, np.newaxis, :, :]
    
    # Refine
    refined_norm, residuals, loop_masks = refine_hic_batch(
        pred_norm,
        model,
        scheduler,
        device,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        loop_percentile=args.loop_percentile,
        apply_masking=args.apply_masking,
        show_progress=True
    )
    
    # Convert to numpy
    refined_norm = refined_norm.numpy()
    residuals = residuals.numpy()
    
    # Denormalize to raw scale
    if pred_raw is not None:
        # Use actual raw data for denormalization
        if pred_raw.ndim == 3:
            pred_raw = pred_raw[:, np.newaxis, :, :]
        refined_raw = denormalize_hic(refined_norm, pred_raw)
    else:
        # Estimate denormalization (assume normalized to [0, 1])
        # Scale back using pred_norm's range
        scale_factor = 1.0 / (pred_norm.max() + 1e-8)
        refined_raw = refined_norm / scale_factor
    
    # Remove channel dimension if it was added
    original_shape = np.load(args.pred_path).shape
    if len(original_shape) == 3:
        refined_norm = refined_norm.squeeze(1)
        refined_raw = refined_raw.squeeze(1)
        residuals = residuals.squeeze(1)
    
    # Save results
    print(f"\n{'='*60}")
    print(f"Saving results...")
    print(f"{'='*60}")
    
    # Save predictions_norm.npy (refined, normalized)
    norm_path = output_dir / 'predictions_norm.npy'
    np.save(norm_path, refined_norm)
    print(f"✅ Normalized refined: {norm_path}")
    print(f"   Shape: {refined_norm.shape}")
    print(f"   Range: [{refined_norm.min():.4f}, {refined_norm.max():.4f}]")
    
    # Save refine_raw.npy (refined, denormalized)
    raw_path = output_dir / 'refine_raw.npy'
    np.save(raw_path, refined_raw)
    print(f"✅ Raw refined: {raw_path}")
    print(f"   Shape: {refined_raw.shape}")
    print(f"   Range: [{refined_raw.min():.4f}, {refined_raw.max():.4f}]")
    
    # Optionally save residuals
    if args.save_residuals:
        residual_path = output_dir / 'residuals_tad.npy'
        np.save(residual_path, residuals)
        print(f"✅ TAD residuals: {residual_path}")
        print(f"   Shape: {residuals.shape}")
        print(f"   Std: {residuals.std():.6f}")
    
    # Optionally save loop masks
    if args.save_masks and loop_masks is not None:
        loop_masks = loop_masks.numpy()
        if loop_masks.ndim == 4 and loop_masks.shape[1] == 1:
            loop_masks = loop_masks.squeeze(1)
        
        mask_path = output_dir / 'loop_masks.npy'
        np.save(mask_path, loop_masks)
        print(f"✅ Loop masks: {mask_path}")
        print(f"   Avg loop fraction: {loop_masks.mean()*100:.1f}%")
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"Statistics:")
    print(f"{'='*60}")
    
    # Residual statistics
    residual_norm_value = np.linalg.norm(residuals.reshape(len(residuals), -1), axis=1).mean()
    signal_norm_value = np.linalg.norm(pred_norm.reshape(len(pred_norm), -1), axis=1).mean()
    rsr = (residual_norm_value / signal_norm_value) * 100
    
    print(f"Residual-to-Signal Ratio (RSR): {rsr:.2f}%")
    print(f"Residual std: {residuals.std():.6f}")
    
    # Change from HiCARN
    diff_norm = refined_norm - np.load(args.pred_path)
    print(f"\nChange from HiCARN (normalized):")
    print(f"  Mean absolute change: {np.abs(diff_norm).mean():.6f}")
    print(f"  Max change: {np.abs(diff_norm).max():.6f}")
    
    # If ground truth available
    if gt_data is not None:
        if gt_data.ndim == 4 and gt_data.shape[1] == 1:
            gt_data = gt_data.squeeze(1)
        
        print(f"\n{'='*60}")
        print(f"Comparison with ground truth:")
        print(f"{'='*60}")
        
        # MSE
        pred_orig = np.load(args.pred_path)
        mse_hicarn = np.mean((pred_orig - gt_data) ** 2)
        mse_refined = np.mean((refined_norm - gt_data) ** 2)
        
        print(f"MSE:")
        print(f"  HiCARN:  {mse_hicarn:.6f}")
        print(f"  Refined: {mse_refined:.6f}")
        print(f"  Change:  {(mse_refined - mse_hicarn)/mse_hicarn*100:+.2f}%")
        
        # PSNR
        def compute_psnr(pred, gt):
            mse = np.mean((pred - gt) ** 2)
            if mse == 0:
                return float('inf')
            max_val = gt.max()
            psnr = 20 * np.log10(max_val / np.sqrt(mse))
            return psnr
        
        psnr_hicarn = compute_psnr(pred_orig, gt_data)
        psnr_refined = compute_psnr(refined_norm, gt_data)
        
        print(f"\nPSNR:")
        print(f"  HiCARN:  {psnr_hicarn:.2f} dB")
        print(f"  Refined: {psnr_refined:.2f} dB")
        print(f"  Change:  {psnr_refined - psnr_hicarn:+.2f} dB")
    
    print(f"\n{'='*60}")
    print(f"✅ INFERENCE COMPLETE!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  {norm_path}")
    print(f"  {raw_path}")
    
    print(f"\nNext steps:")
    print(f"  1. Use predictions_norm.npy for metric evaluation")
    print(f"  2. Use refine_raw.npy for visualization")
    print(f"  3. Run evaluate_metrics.py for comprehensive analysis")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TAD-Only Diffusion Inference')
    
    # Input paths
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to HiCARN predictions_norm.npy')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='Path to ground truth (optional)')
    parser.add_argument('--raw_path', type=str, default=None,
                        help='Path to raw HiCARN predictions for denormalization (optional)')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--save_residuals', action='store_true',
                        help='Save TAD-only residuals')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save loop masks')
    
    # Sampling parameters
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of denoising steps (default: 50)')
    parser.add_argument('--guidance_scale', type=float, default=0.0,
                        help='Classifier-free guidance scale (default: 0.0)')
    
    # Loop masking
    parser.add_argument('--loop_percentile', type=int, default=90,
                        help='Percentile for loop masking (default: 90)')
    parser.add_argument('--apply_masking', action='store_true', default=True,
                        help='Apply loop masking (default: True)')
    parser.add_argument('--no_masking', dest='apply_masking', action='store_false',
                        help='Disable loop masking')
    
    # Device
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (default: 0)')
    
    args = parser.parse_args()
    
    main(args)
