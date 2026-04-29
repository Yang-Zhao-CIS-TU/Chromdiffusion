"""
Sampling Script for Residual Diffusion

Inference pipeline:
1. Load trained diffusion model
2. Load HiCARN predictions
3. Sample residuals: p(Δ|x_HiCARN)
4. Denormalize residuals
5. Refine predictions: x̂ = x_pred + Δ̂
6. Save refined predictions
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json

from model import ResidualDiffusionUNet
from scheduler import DDPMScheduler, DDIMScheduler
from data_loader import ResidualNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description='Sample from Residual Diffusion')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to HiCARN predictions to refine')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save refined predictions')
    
    # Sampling
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of sampling steps (use DDIM for < 1000)')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='DDIM eta parameter (0=deterministic, 1=DDPM)')
    parser.add_argument('--use_ddim', action='store_true',
                       help='Use DDIM sampler (faster)')
    
    # System
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    
    return parser.parse_args()


@torch.no_grad()
def sample_residuals(
    model,
    predictions,
    scheduler,
    normalizer,
    device,
    batch_size=16,
    use_ddim=False,
    eta=0.0
):
    """
    Sample residuals from diffusion model
    
    Args:
        model: trained ResidualDiffusionUNet
        predictions: HiCARN predictions (N, C, H, W)
        scheduler: diffusion scheduler
        normalizer: residual normalizer
        device: device
        batch_size: batch size
        use_ddim: use DDIM sampling
        eta: DDIM eta parameter
    
    Returns:
        refined_predictions: x̂ = x_pred + Δ̂ (N, C, H, W)
        sampled_residuals: Δ̂ (N, C, H, W)
    """
    model.eval()
    
    n_samples = len(predictions)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    refined_predictions = []
    sampled_residuals = []
    
    # Get timesteps
    if use_ddim and isinstance(scheduler, DDIMScheduler):
        timesteps = scheduler.timesteps
    else:
        timesteps = list(range(scheduler.num_train_timesteps - 1, -1, -1))
    
    print(f"Sampling with {len(timesteps)} steps...")
    
    for batch_idx in tqdm(range(n_batches), desc='Sampling'):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        # Get batch
        batch_pred = predictions[start_idx:end_idx]
        
        # Ensure 4D: (B, C, H, W)
        if batch_pred.ndim == 3:
            batch_pred = batch_pred[:, None, :, :]
        
        batch_pred = torch.from_numpy(batch_pred).float().to(device)
        current_batch_size = batch_pred.shape[0]
        
        # Start from random noise
        residual_t = torch.randn_like(batch_pred)
        
        # Iterative denoising
        for t in timesteps:
            # Prepare timestep tensor
            t_tensor = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(residual_t, t_tensor, batch_pred)
            
            # Denoise one step
            if use_ddim and isinstance(scheduler, DDIMScheduler):
                residual_t, _ = scheduler.step(noise_pred, t, residual_t, eta=eta)
            else:
                residual_t, _ = scheduler.step(noise_pred, t, residual_t)
        
        # Final prediction is the denoised residual (in normalized space)
        residual_norm = residual_t
        
        # Denormalize
        residual_raw = normalizer.inverse_transform(
            residual_norm.cpu().numpy()
        )
        residual_raw = torch.from_numpy(residual_raw).to(device)
        
        # Refine predictions: x̂ = x_pred + Δ̂
        refined_batch = batch_pred + residual_raw
        
        # Store results
        refined_predictions.append(refined_batch.cpu().numpy())
        sampled_residuals.append(residual_raw.cpu().numpy())
    
    # Concatenate batches
    refined_predictions = np.concatenate(refined_predictions, axis=0)
    sampled_residuals = np.concatenate(sampled_residuals, axis=0)
    
    # Remove channel dimension if single channel
    if refined_predictions.shape[1] == 1:
        refined_predictions = refined_predictions[:, 0, :, :]
        sampled_residuals = sampled_residuals[:, 0, :, :]
    
    return refined_predictions, sampled_residuals


def main():
    args = parse_args()
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("RESIDUAL DIFFUSION SAMPLING")
    print("="*80)
    
    # ====================================================================
    # LOAD MODEL
    # ====================================================================
    
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load config
    config = checkpoint.get('config', {})
    
    # Create model
    model = ResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=config.get('base_channels', 64),
        channel_multipliers=tuple(config.get('channel_multipliers', [1, 2, 4, 8])),
        num_res_blocks=config.get('num_res_blocks', 2)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Model had validation loss: {checkpoint['loss']:.6f}")
    
    # ====================================================================
    # LOAD NORMALIZER
    # ====================================================================
    
    print("\nLoading residual normalizer...")
    normalizer = ResidualNormalizer()
    normalizer.mean = checkpoint['normalizer_mean']
    normalizer.std = checkpoint['normalizer_std']
    normalizer.fitted = True
    
    print(f"Normalizer stats:")
    print(f"  Mean: {normalizer.mean:.6f}")
    print(f"  Std:  {normalizer.std:.6f}")
    
    # ====================================================================
    # CREATE SCHEDULER
    # ====================================================================
    
    print("\nCreating scheduler...")
    
    num_train_timesteps = config.get('num_timesteps', 1000)
    beta_schedule = config.get('beta_schedule', 'linear')
    
    if args.use_ddim:
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=args.num_inference_steps,
            beta_schedule=beta_schedule
        )
        print(f"Using DDIM sampler with {args.num_inference_steps} steps")
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule
        )
        print(f"Using DDPM sampler with {num_train_timesteps} steps")
    
    # ====================================================================
    # LOAD PREDICTIONS
    # ====================================================================
    
    print("\nLoading HiCARN predictions...")
    predictions = np.load(args.pred_path)
    
    print(f"Loaded predictions:")
    print(f"  Shape: {predictions.shape}")
    print(f"  Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Validate
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        raise ValueError("Predictions contain NaN or Inf!")
    
    # ====================================================================
    # SAMPLE RESIDUALS
    # ====================================================================
    
    print("\n" + "="*80)
    print("SAMPLING RESIDUALS")
    print("="*80)
    
    refined_predictions, sampled_residuals = sample_residuals(
        model=model,
        predictions=predictions,
        scheduler=scheduler,
        normalizer=normalizer,
        device=device,
        batch_size=args.batch_size,
        use_ddim=args.use_ddim,
        eta=args.eta
    )
    
    # ====================================================================
    # SAVE RESULTS
    # ====================================================================
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create output directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save refined predictions
    np.save(args.output_path, refined_predictions)
    print(f"✓ Saved refined predictions: {args.output_path}")
    
    # Save residuals
    residual_path = args.output_path.replace('.npy', '_residuals.npy')
    np.save(residual_path, sampled_residuals)
    print(f"✓ Saved sampled residuals: {residual_path}")
    
    # Save statistics
    stats = {
        'num_samples': int(len(predictions)),
        'original_range': [float(predictions.min()), float(predictions.max())],
        'refined_range': [float(refined_predictions.min()), float(refined_predictions.max())],
        'residual_range': [float(sampled_residuals.min()), float(sampled_residuals.max())],
        'residual_mean': float(sampled_residuals.mean()),
        'residual_std': float(sampled_residuals.std()),
        'sampling_method': 'DDIM' if args.use_ddim else 'DDPM',
        'num_steps': args.num_inference_steps if args.use_ddim else num_train_timesteps
    }
    
    stats_path = args.output_path.replace('.npy', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics: {stats_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SAMPLING COMPLETE")
    print("="*80)
    print(f"Samples processed: {len(predictions)}")
    print(f"Original prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Refined prediction range: [{refined_predictions.min():.4f}, {refined_predictions.max():.4f}]")
    print(f"Sampled residual range: [{sampled_residuals.min():.4f}, {sampled_residuals.max():.4f}]")
    print(f"Residual mean: {sampled_residuals.mean():.4f}")
    print(f"Residual std: {sampled_residuals.std():.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
