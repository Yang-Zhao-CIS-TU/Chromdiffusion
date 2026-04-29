#!/usr/bin/env python3
"""
Sampling Script for Residual Diffusion Model

Inference pipeline:
1. Load trained diffusion model (ImprovedResidualDiffusionUNet)
2. Load HiCARN predictions (normalized space)
3. Sample normalized residuals: p(δ_norm | x_HiCARN_norm)
4. Denormalize residuals using ResidualNormalizer
5. Refine predictions: x̂_norm = x_pred_norm + δ̂
6. Optionally denormalize to raw space using RobustHiCPreprocessor
7. Save outputs

Outputs:
- refined_norm.npy: Refined predictions in normalized space
- refined_raw.npy: Refined predictions in raw space (if preprocessor available)
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from model import ImprovedResidualDiffusionUNet
from scheduler import ImprovedDDPMScheduler


# ================================================================
# RobustHiCPreprocessor - for denormalizing to raw space
# ================================================================

def ensure_nchw(arr):
    """Ensure array is (N, C, H, W)."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[:, np.newaxis, :, :]
    elif arr.ndim == 4:
        if arr.shape[1] in [1, 3]:
            return arr
        elif arr.shape[-1] in [1, 3]:
            return np.transpose(arr, (0, 3, 1, 2))
        else:
            return arr
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape={arr.shape}")


class RobustHiCPreprocessor:
    """
    Class definition to allow torch.load to deserialize the preprocessor.
    Matches the one used in train_hicarn_robust.py
    """
    
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None  # Actually stores median
        self.X_std = None   # Actually stores IQR
        self.Y_mean = None  # Actually stores median
        self.Y_std = None   # Actually stores IQR
        self._is_fitted = False

    def fit(self, X_low, Y_high, verbose=True):
        X_low = ensure_nchw(X_low)
        Y_high = ensure_nchw(Y_high)
        X_log = np.log1p(X_low)
        Y_log = np.log1p(Y_high)
        self.X_mean = np.median(X_log)
        self.X_std = (np.percentile(X_log, 75) - np.percentile(X_log, 25)) + 1e-8
        self.Y_mean = np.median(Y_log)
        self.Y_std = (np.percentile(Y_log, 75) - np.percentile(Y_log, 25)) + 1e-8
        self._is_fitted = True
        return self

    def preprocess(self, X_low, Y_high=None):
        X_low = ensure_nchw(X_low)
        X_log = np.log1p(X_low)
        Xn = (X_log - self.X_mean) / self.X_std
        Xn = np.clip(Xn, -5, 5).astype(np.float32)
        if Y_high is None:
            return Xn, None
        Y_high = ensure_nchw(Y_high)
        Y_log = np.log1p(Y_high)
        Yn = (Y_log - self.Y_mean) / self.Y_std
        Yn = np.clip(Yn, -5, 5).astype(np.float32)
        return Xn, Yn

    def postprocess(self, Y_norm):
        """Convert normalized predictions back to raw space"""
        Y_norm = np.clip(Y_norm, -5, 5)
        Y_log = Y_norm * self.Y_std + self.Y_mean
        Y_counts = np.expm1(Y_log)
        return np.maximum(Y_counts, 0.0)

    def get_stats(self):
        return {
            'X_median': float(self.X_mean),
            'X_iqr': float(self.X_std),
            'Y_median': float(self.Y_mean),
            'Y_iqr': float(self.Y_std),
        }


# Register class for torch.load
sys.modules['__main__'].RobustHiCPreprocessor = RobustHiCPreprocessor
sys.modules['__main__'].ensure_nchw = ensure_nchw


# ================================================================
# ResidualNormalizer - for denormalizing residuals
# ================================================================

class ResidualNormalizer:
    """
    Normalizer for residuals (same as used in training)
    Residual normalization: residual_norm = (residual - mean) / std
    """
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.fitted = False
    
    def transform(self, residuals):
        """Normalize residuals"""
        return (residuals - self.mean) / self.std
    
    def inverse_transform(self, residuals_norm):
        """Denormalize residuals"""
        return residuals_norm * self.std + self.mean


# ================================================================
# Sampling Function
# ================================================================

@torch.no_grad()
def sample_residuals(
    model,
    scheduler,
    predictions,
    normalizer,
    device,
    batch_size=64,
    num_steps=50,
    use_ddim=True,
    ddim_eta=0.0
):
    """
    Sample residuals from diffusion model and refine predictions
    
    Args:
        model: Trained ImprovedResidualDiffusionUNet
        scheduler: ImprovedDDPMScheduler
        predictions: HiCARN predictions in normalized space [N, 1, H, W]
        normalizer: ResidualNormalizer for denormalizing residuals
        device: torch device
        batch_size: Batch size for inference
        num_steps: Number of denoising steps
        use_ddim: Use DDIM (faster) or DDPM
        ddim_eta: DDIM eta parameter (0.0 = deterministic)
    
    Returns:
        refined_predictions: x̂_norm = x_pred_norm + δ̂ [N, 1, H, W]
        sampled_residuals: δ̂ (denormalized) [N, 1, H, W]
    """
    model.eval()
    
    n_samples = len(predictions)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    refined_list = []
    residuals_list = []
    
    # Set timesteps for sampling
    scheduler.set_timesteps(num_steps, device=device, method='uniform')
    
    print(f"Sampling with {num_steps} steps ({'DDIM' if use_ddim else 'DDPM'})...")
    
    for batch_idx in tqdm(range(n_batches), desc='Sampling batches'):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        # Get batch of predictions
        batch_pred = predictions[start_idx:end_idx]
        
        # Ensure 4D: (B, 1, H, W)
        if batch_pred.ndim == 3:
            batch_pred = batch_pred[:, np.newaxis, :, :]
        
        batch_pred_tensor = torch.from_numpy(batch_pred).float().to(device)
        current_batch_size = batch_pred_tensor.shape[0]
        
        # Start from random noise (this will be our residual_t)
        residual_t = torch.randn_like(batch_pred_tensor)
        
        # Iterative denoising
        for t in scheduler.timesteps:
            # Batch timesteps
            t_batch = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise/velocity (condition on HiCARN predictions)
            model_output = model(residual_t, t_batch, batch_pred_tensor)
            
            # Denoise one step
            if use_ddim:
                residual_t, _ = scheduler.step(
                    model_output, t, residual_t,
                    eta=ddim_eta, use_ddim=True
                )
            else:
                residual_t, _ = scheduler.step(
                    model_output, t, residual_t,
                    use_ddim=False
                )
        
        # Final denoised residual (in normalized residual space)
        residual_norm = residual_t.cpu().numpy()
        
        # Denormalize residuals using ResidualNormalizer
        residual_denorm = normalizer.inverse_transform(residual_norm)
        
        # Refine predictions: x̂_norm = x_pred_norm + δ̂
        refined_batch = batch_pred + residual_denorm
        
        # Store results
        refined_list.append(refined_batch)
        residuals_list.append(residual_denorm)
    
    # Concatenate all batches
    refined_predictions = np.concatenate(refined_list, axis=0)
    sampled_residuals = np.concatenate(residuals_list, axis=0)
    
    return refined_predictions, sampled_residuals


def load_checkpoint(checkpoint_path, device='cuda'):
    """
    Load model, scheduler, and normalizer from checkpoint
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', {})
    scheduler_config = checkpoint.get('scheduler_config', {})
    
    # Scheduler parameters
    num_train_timesteps = scheduler_config.get('num_train_timesteps', 
                                               config.get('num_timesteps', 1000))
    parameterization = scheduler_config.get('parameterization', 
                                            config.get('parameterization', 'v'))
    
    # Create scheduler
    scheduler = ImprovedDDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        parameterization=parameterization
    )
    
    # Model parameters
    base_channels = config.get('base_channels', 64)
    channel_mults = tuple(config.get('channel_multipliers', 
                                     config.get('channel_mults', [1, 2, 4, 8])))
    num_res_blocks = config.get('num_res_blocks', 2)
    
    # Create model
    model = ImprovedResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        cond_channels=1,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        attn_levels=(2, 3),
        parameterization=parameterization
    ).to(device)
    
    # Load weights (prefer EMA weights if available)
    if 'ema_shadow' in checkpoint and checkpoint['ema_shadow']:
        print("  Using EMA weights")
        model.load_state_dict(checkpoint['ema_shadow'])
    elif 'ema_state_dict' in checkpoint:
        print("  Using EMA state dict")
        model.load_state_dict(checkpoint['ema_state_dict'])
    else:
        print("  Using regular weights")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Load residual normalizer
    normalizer = ResidualNormalizer()
    if 'normalizer_mean' in checkpoint and 'normalizer_std' in checkpoint:
        normalizer.mean = checkpoint['normalizer_mean']
        normalizer.std = checkpoint['normalizer_std']
        normalizer.fitted = True
        print(f"  Loaded ResidualNormalizer: mean={normalizer.mean:.6f}, std={normalizer.std:.6f}")
    else:
        print("  ⚠️  No normalizer stats in checkpoint, using identity (mean=0, std=1)")
    
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', checkpoint.get('val_loss', 'unknown'))
    print(f"  Loaded from epoch: {epoch}, loss: {loss}")
    
    return model, scheduler, normalizer


def load_hic_preprocessor(preprocessor_path):
    """
    Load HiC preprocessor for denormalization to raw space
    """
    print(f"Loading HiC preprocessor: {preprocessor_path}")
    
    try:
        # Try torch.load with weights_only=False
        try:
            checkpoint = torch.load(preprocessor_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(preprocessor_path, map_location='cpu')
        
        # Extract preprocessor
        if isinstance(checkpoint, dict) and 'preprocessor' in checkpoint:
            preprocessor = checkpoint['preprocessor']
        else:
            preprocessor = checkpoint
        
        # Get stats
        if isinstance(preprocessor, RobustHiCPreprocessor):
            print(f"  Y_median: {preprocessor.Y_mean:.6f}")
            print(f"  Y_iqr: {preprocessor.Y_std:.6f}")
        elif hasattr(preprocessor, 'Y_mean') and hasattr(preprocessor, 'Y_std'):
            print(f"  Y_median: {preprocessor.Y_mean:.6f}")
            print(f"  Y_iqr: {preprocessor.Y_std:.6f}")
        
        return preprocessor
        
    except Exception as e:
        print(f"  ⚠️  Failed to load preprocessor: {e}")
        return None


def denormalize_to_raw(normalized_data, preprocessor):
    """
    Convert normalized predictions to raw space using preprocessor
    
    Uses RobustHiCPreprocessor.postprocess():
        1. Y_log = Y_norm * IQR + median
        2. Y_raw = expm1(Y_log)
        3. Y_raw = max(Y_raw, 0)
    """
    if preprocessor is None:
        return None
    
    try:
        if hasattr(preprocessor, 'postprocess'):
            raw_data = preprocessor.postprocess(normalized_data)
        elif hasattr(preprocessor, 'Y_mean') and hasattr(preprocessor, 'Y_std'):
            # Manual postprocess
            Y_norm = np.clip(normalized_data, -5, 5)
            Y_log = Y_norm * preprocessor.Y_std + preprocessor.Y_mean
            raw_data = np.expm1(Y_log)
            raw_data = np.maximum(raw_data, 0.0)
        else:
            print("  ⚠️  Preprocessor has no postprocess method")
            return None
        
        return raw_data.astype(np.float32)
        
    except Exception as e:
        print(f"  ⚠️  Denormalization failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Sample from trained residual diffusion model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to HiCARN predictions (normalized) .npy file')
    
    # Optional arguments
    parser.add_argument('--preprocessor_path', type=str, default=None,
                       help='Path to HiC preprocessor for denormalization to raw space')
    parser.add_argument('--output_dir', type=str, default='refined_predictions',
                       help='Output directory')
    
    # Sampling parameters
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of denoising steps (default: 50)')
    parser.add_argument('--use_ddim', action='store_true', default=True,
                       help='Use DDIM sampling (faster, default: True)')
    parser.add_argument('--no_ddim', action='store_true',
                       help='Use DDPM sampling (slower, higher quality)')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                       help='DDIM eta (0.0=deterministic, 1.0=stochastic)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    
    args = parser.parse_args()
    
    # Handle DDIM flag
    use_ddim = args.use_ddim and not args.no_ddim
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # LOAD MODEL
    # ================================================================
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    model, scheduler, normalizer = load_checkpoint(args.checkpoint, device)
    
    # ================================================================
    # LOAD HICARN PREDICTIONS
    # ================================================================
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    print(f"Loading HiCARN predictions: {args.pred_path}")
    predictions = np.load(args.pred_path)
    
    # Ensure shape is [N, 1, H, W]
    if predictions.ndim == 3:
        predictions = predictions[:, np.newaxis, :, :]
    
    print(f"  Shape: {predictions.shape}")
    print(f"  Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Mean: {predictions.mean():.4f}")
    
    # Check for issues
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        raise ValueError("Predictions contain NaN or Inf!")
    
    # ================================================================
    # LOAD HIC PREPROCESSOR (OPTIONAL)
    # ================================================================
    if args.preprocessor_path:
        print("\n" + "="*80)
        print("LOADING HIC PREPROCESSOR")
        print("="*80)
        hic_preprocessor = load_hic_preprocessor(args.preprocessor_path)
    else:
        print("\n⚠️  No preprocessor provided, will only output normalized space results")
        hic_preprocessor = None
    
    # ================================================================
    # SAMPLE RESIDUALS AND REFINE
    # ================================================================
    print("\n" + "="*80)
    print("SAMPLING (REFINING PREDICTIONS)")
    print("="*80)
    print(f"Sampling parameters:")
    print(f"  Method: {'DDIM' if use_ddim else 'DDPM'}")
    print(f"  Steps: {args.num_steps}")
    if use_ddim:
        print(f"  DDIM eta: {args.ddim_eta}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Residual normalizer: mean={normalizer.mean:.6f}, std={normalizer.std:.6f}")
    print()
    
    refined_norm, sampled_residuals = sample_residuals(
        model=model,
        scheduler=scheduler,
        predictions=predictions,
        normalizer=normalizer,
        device=device,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        use_ddim=use_ddim,
        ddim_eta=args.ddim_eta
    )
    
    print(f"\n✓ Sampling complete!")
    print(f"  Refined (norm) shape: {refined_norm.shape}")
    print(f"  Refined (norm) range: [{refined_norm.min():.4f}, {refined_norm.max():.4f}]")
    print(f"  Sampled residuals range: [{sampled_residuals.min():.4f}, {sampled_residuals.max():.4f}]")
    print(f"  Residual mean: {sampled_residuals.mean():.4f}")
    print(f"  Residual std: {sampled_residuals.std():.4f}")
    
    # ================================================================
    # SAVE NORMALIZED SPACE RESULTS
    # ================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save refined predictions (normalized space)
    norm_path = output_dir / 'refined_norm.npy'
    np.save(norm_path, refined_norm)
    print(f"✓ Saved normalized predictions: {norm_path}")
    
    # Save sampled residuals
    residual_path = output_dir / 'sampled_residuals.npy'
    np.save(residual_path, sampled_residuals)
    print(f"✓ Saved sampled residuals: {residual_path}")
    
    # ================================================================
    # DENORMALIZE TO RAW SPACE
    # ================================================================
    if hic_preprocessor is not None:
        print("\n" + "="*80)
        print("DENORMALIZING TO RAW SPACE")
        print("="*80)
        
        refined_raw = denormalize_to_raw(refined_norm, hic_preprocessor)
        
        if refined_raw is not None:
            print(f"  Refined (raw) shape: {refined_raw.shape}")
            print(f"  Refined (raw) range: [{refined_raw.min():.4f}, {refined_raw.max():.4f}]")
            print(f"  Refined (raw) mean: {refined_raw.mean():.4f}")
            
            # Save raw space results
            raw_path = output_dir / 'refined_raw.npy'
            np.save(raw_path, refined_raw)
            print(f"✓ Saved raw predictions: {raw_path}")
        else:
            print("⚠️  Denormalization failed, no raw space output")
            raw_path = None
    else:
        print("\n⚠️  Skipping denormalization (no preprocessor)")
        raw_path = None
    
    # ================================================================
    # SAVE METADATA
    # ================================================================
    metadata = {
        'num_samples': int(predictions.shape[0]),
        'input_shape': list(predictions.shape),
        'output_shape': list(refined_norm.shape),
        'input_range': [float(predictions.min()), float(predictions.max())],
        'refined_norm_range': [float(refined_norm.min()), float(refined_norm.max())],
        'residual_range': [float(sampled_residuals.min()), float(sampled_residuals.max())],
        'residual_mean': float(sampled_residuals.mean()),
        'residual_std': float(sampled_residuals.std()),
        'sampling_method': 'DDIM' if use_ddim else 'DDPM',
        'num_steps': args.num_steps,
        'ddim_eta': args.ddim_eta if use_ddim else None,
        'normalizer_mean': float(normalizer.mean),
        'normalizer_std': float(normalizer.std),
        'checkpoint': str(args.checkpoint),
        'pred_path': str(args.pred_path)
    }
    
    if raw_path is not None:
        metadata['refined_raw_range'] = [float(refined_raw.min()), float(refined_raw.max())]
    
    metadata_path = output_dir / 'sampling_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Processed {predictions.shape[0]} samples")
    print(f"\nOutput files in {output_dir}/:")
    print(f"  1. refined_norm.npy     - Refined predictions (normalized space)")
    print(f"  2. sampled_residuals.npy - Sampled residuals")
    if raw_path is not None:
        print(f"  3. refined_raw.npy      - Refined predictions (raw space)")
    print(f"  4. sampling_metadata.json - Sampling metadata")
    
    print(f"\nStatistics:")
    print(f"  Input (HiCARN norm):  [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Output (refined norm): [{refined_norm.min():.4f}, {refined_norm.max():.4f}]")
    print(f"  Residuals: mean={sampled_residuals.mean():.4f}, std={sampled_residuals.std():.4f}")
    
    if raw_path is not None:
        print(f"  Output (raw):  [{refined_raw.min():.4f}, {refined_raw.max():.4f}]")
        print(f"\nNext steps:")
        print(f"  python evaluate.py --pred_path {raw_path} --gt_path <gt_raw.npy>")
    else:
        print(f"\n⚠️  No raw space output - use denormalize_fixed.py to convert refined_norm.npy")
    
    print("="*80)


if __name__ == '__main__':
    main()
