"""
Sampling Script for Residual Diffusion Model

Generate refined Hi-C contact maps from HiCARN predictions using trained diffusion model

Outputs:
- refined_norm.npy: Refined predictions in normalized space
- refined_raw.npy: Refined predictions in raw space (using preprocessor)
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle

from model import ImprovedResidualDiffusionUNet
from scheduler import ImprovedDDPMScheduler


class HiCPreprocessor:
    """Load and use HiCARN preprocessor for denormalization"""
    def __init__(self, preprocessor_path):
        self.preprocessor_path = preprocessor_path
        self.preprocessor = self._load_preprocessor()
    
    def _load_preprocessor(self):
        """Load preprocessor from file"""
        try:
            # Try torch.load first (most common for PyTorch models)
            preprocessor = torch.load(self.preprocessor_path, map_location='cpu')
            print(f"Loaded preprocessor from {self.preprocessor_path} (torch.load)")
        except Exception as e:
            print(f"torch.load failed: {e}")
            # Fallback to pickle
            try:
                with open(self.preprocessor_path, 'rb') as f:
                    preprocessor = pickle.load(f)
                print(f"Loaded preprocessor from {self.preprocessor_path} (pickle.load)")
            except Exception as e2:
                print(f"pickle.load also failed: {e2}")
                raise ValueError(f"Could not load preprocessor from {self.preprocessor_path}")
        
        return preprocessor
    
    def denormalize(self, normalized_data):
        """
        Convert from normalized space back to raw space
        
        Args:
            normalized_data: numpy array [N, 1, H, W] in normalized space
        
        Returns:
            raw_data: numpy array [N, 1, H, W] in raw space
        """
        # Handle case where preprocessor is a dict
        if isinstance(self.preprocessor, dict):
            if 'mean' in self.preprocessor and 'std' in self.preprocessor:
                mean = self.preprocessor['mean']
                std = self.preprocessor['std']
                raw_data = normalized_data * std + mean
                print(f"Denormalized using dict (mean={mean:.3f}, std={std:.3f})")
                return raw_data
            elif 'scaler' in self.preprocessor:
                # Nested scaler in dict
                scaler = self.preprocessor['scaler']
                return self._denormalize_with_scaler(scaler, normalized_data)
            else:
                print("Warning: Dict preprocessor without 'mean'/'std', returning normalized data")
                return normalized_data
        
        # Handle case where preprocessor is a scaler object
        return self._denormalize_with_scaler(self.preprocessor, normalized_data)
    
    def _denormalize_with_scaler(self, scaler, normalized_data):
        """Helper to denormalize using a scaler object"""
        # Try various denormalization methods
        if hasattr(scaler, 'inverse_transform'):
            raw_data = scaler.inverse_transform(normalized_data)
            print("Denormalized using inverse_transform()")
        elif hasattr(scaler, 'denormalize'):
            raw_data = scaler.denormalize(normalized_data)
            print("Denormalized using denormalize()")
        elif hasattr(scaler, 'mean_') and hasattr(scaler, 'std_'):
            # StandardScaler-like
            raw_data = normalized_data * scaler.std_ + scaler.mean_
            print(f"Denormalized using mean_/std_ (mean={scaler.mean_:.3f}, std={scaler.std_:.3f})")
        elif hasattr(scaler, 'scale_') and hasattr(scaler, 'min_'):
            # MinMaxScaler-like
            raw_data = (normalized_data - scaler.min_) / scaler.scale_
            print(f"Denormalized using MinMaxScaler (min={scaler.min_:.3f}, scale={scaler.scale_:.3f})")
        else:
            print("Warning: No denormalization method found, returning normalized data")
            raw_data = normalized_data
        
        return raw_data


@torch.no_grad()
def sample_residual_diffusion(
    model,
    scheduler,
    condition,
    num_steps=50,
    use_ddim=True,
    ddim_eta=0.0,
    device='cuda'
):
    """
    Sample refined residuals using diffusion model
    
    Args:
        model: Trained diffusion model
        scheduler: Noise scheduler
        condition: Condition (HiCARN predictions) [B, 1, H, W]
        num_steps: Number of denoising steps
        use_ddim: Use DDIM (faster) or DDPM (slower, better quality)
        ddim_eta: DDIM eta parameter (0.0 = deterministic)
        device: Device
    
    Returns:
        refined: Refined predictions [B, 1, H, W]
    """
    batch_size = condition.shape[0]
    
    # Start from pure noise
    residual = torch.randn_like(condition)
    
    # Set sampling timesteps
    scheduler.set_timesteps(num_steps, device=device, method='uniform')
    
    # Denoising loop
    for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
        # Batch timesteps
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise/velocity
        model_output = model(residual, t_batch, condition)
        
        # Denoise one step
        if use_ddim:
            residual, pred_original_sample = scheduler.step(
                model_output,
                t,
                residual,
                eta=ddim_eta,
                use_ddim=True
            )
        else:
            residual, pred_original_sample = scheduler.step(
                model_output,
                t,
                residual,
                use_ddim=False
            )
    
    # Final refined prediction
    refined = condition + residual
    
    return refined


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model and scheduler from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get scheduler config
    scheduler_config = checkpoint.get('scheduler_config', {})
    num_train_timesteps = scheduler_config.get('num_train_timesteps', 1000)
    parameterization = scheduler_config.get('parameterization', 'v')
    
    # Create scheduler
    scheduler = ImprovedDDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        parameterization=parameterization
    )
    
    # Create model
    model = ImprovedResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        cond_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_levels=(2, 3),
        parameterization=parameterization
    ).to(device)
    
    # Load weights (try EMA first, then regular)
    if 'ema_shadow' in checkpoint and checkpoint['ema_shadow']:
        print("  Using EMA weights")
        ema_state = checkpoint['ema_shadow']
        # Convert EMA shadow to state dict format
        model_state = {k: v for k, v in ema_state.items()}
        model.load_state_dict(model_state)
    else:
        print("  Using regular weights")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Loaded from epoch: {epoch}")
    
    return model, scheduler


def main():
    parser = argparse.ArgumentParser(description='Sample from trained residual diffusion model')
    
    # Input/Output
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to HiCARN predictions (normalized) .npy file')
    parser.add_argument('--preprocessor_path', type=str, default=None,
                       help='Path to HiCARN preprocessor .pt file (optional, for denormalization)')
    parser.add_argument('--output_dir', type=str, default='refined_predictions',
                       help='Output directory')
    
    # Sampling parameters
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of denoising steps (50 is good balance)')
    parser.add_argument('--use_ddim', action='store_true', default=True,
                       help='Use DDIM sampling (faster, default)')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                       help='DDIM eta (0.0=deterministic, 1.0=stochastic)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model, scheduler = load_checkpoint(args.checkpoint, device)
    
    # Load HiCARN predictions (normalized space)
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    print(f"Loading HiCARN predictions: {args.pred_path}")
    hicarn_pred_norm = np.load(args.pred_path)
    
    # Ensure shape is [N, 1, H, W]
    if hicarn_pred_norm.ndim == 3:
        hicarn_pred_norm = hicarn_pred_norm[:, np.newaxis, :, :]
    
    num_samples = hicarn_pred_norm.shape[0]
    print(f"  Shape: {hicarn_pred_norm.shape}")
    print(f"  Number of samples: {num_samples}")
    print(f"  Range: [{hicarn_pred_norm.min():.3f}, {hicarn_pred_norm.max():.3f}]")
    
    # Load preprocessor (optional)
    if args.preprocessor_path:
        print(f"\nLoading preprocessor: {args.preprocessor_path}")
        try:
            preprocessor = HiCPreprocessor(args.preprocessor_path)
        except Exception as e:
            print(f"⚠️  Failed to load preprocessor: {e}")
            print("⚠️  Will only save normalized space results")
            preprocessor = None
    else:
        print("\n⚠️  No preprocessor provided, will only save normalized space results")
        preprocessor = None
    
    # Sampling
    print("\n" + "="*80)
    print("SAMPLING (REFINING PREDICTIONS)")
    print("="*80)
    print(f"Sampling parameters:")
    print(f"  Method: {'DDIM' if args.use_ddim else 'DDPM'}")
    print(f"  Steps: {args.num_steps}")
    if args.use_ddim:
        print(f"  DDIM eta: {args.ddim_eta}")
    print(f"  Batch size: {args.batch_size}")
    print()
    
    refined_norm_list = []
    
    # Process in batches
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, num_samples)
        
        # Get batch
        batch_hicarn = hicarn_pred_norm[start_idx:end_idx]
        batch_hicarn_tensor = torch.from_numpy(batch_hicarn).float().to(device)
        
        # Sample refined predictions
        refined_batch = sample_residual_diffusion(
            model,
            scheduler,
            batch_hicarn_tensor,
            num_steps=args.num_steps,
            use_ddim=args.use_ddim,
            ddim_eta=args.ddim_eta,
            device=device
        )
        
        # Convert back to numpy
        refined_batch_np = refined_batch.cpu().numpy()
        refined_norm_list.append(refined_batch_np)
    
    # Concatenate all batches
    refined_norm = np.concatenate(refined_norm_list, axis=0)
    
    print(f"\n✓ Sampling complete!")
    print(f"  Refined (norm) shape: {refined_norm.shape}")
    print(f"  Refined (norm) range: [{refined_norm.min():.3f}, {refined_norm.max():.3f}]")
    
    # Save normalized space results
    norm_output_path = output_dir / 'refined_norm.npy'
    np.save(norm_output_path, refined_norm)
    print(f"\n✓ Saved normalized space results: {norm_output_path}")
    
    # Denormalize to raw space (if preprocessor available)
    if preprocessor is not None:
        print("\n" + "="*80)
        print("DENORMALIZING TO RAW SPACE")
        print("="*80)
        
        try:
            refined_raw = preprocessor.denormalize(refined_norm)
            
            print(f"  Refined (raw) shape: {refined_raw.shape}")
            print(f"  Refined (raw) range: [{refined_raw.min():.3f}, {refined_raw.max():.3f}]")
            
            # Save raw space results
            raw_output_path = output_dir / 'refined_raw.npy'
            np.save(raw_output_path, refined_raw)
            print(f"\n✓ Saved raw space results: {raw_output_path}")
        except Exception as e:
            print(f"\n⚠️  Denormalization failed: {e}")
            print("⚠️  Only normalized space results are available")
            raw_output_path = None
    else:
        print("\n⚠️  Skipping denormalization (no preprocessor)")
        raw_output_path = None
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Processed {num_samples} samples")
    print(f"\nOutput files:")
    print(f"  1. {norm_output_path}")
    print(f"     └─ Refined predictions in normalized space")
    
    if raw_output_path:
        print(f"  2. {raw_output_path}")
        print(f"     └─ Refined predictions in raw space")
        print(f"\nNext steps:")
        print(f"  - Use refined_raw.npy for evaluation")
        print(f"  - Compare with ground truth using evaluate.py")
    else:
        print(f"\n⚠️  No raw space output (preprocessor failed or not provided)")
        print(f"\nNext steps:")
        print(f"  - Use refined_norm.npy for analysis")
        print(f"  - Or provide a working preprocessor to get raw space output")
    
    print("="*80)


if __name__ == '__main__':
    main()
