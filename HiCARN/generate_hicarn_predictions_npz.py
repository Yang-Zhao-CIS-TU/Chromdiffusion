"""
Generate HiCARN Predictions from .npz Files

This script generates predictions from your trained HiCARN model for use in residual diffusion.

Input:
  - .npz file with 'train_lr' and 'train_hr' keys
  - Example: /data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40/train_data_raw_ratio16.npz
  - Trained HiCARN checkpoint (.pytorch file)

Output:
  - predictions_norm.npy: HiCARN predictions in normalized space
  - ground_truth.npy: Ground truth HR in normalized space
  - Both ready for residual diffusion training

Usage:
  python generate_hicarn_predictions_npz.py \
      --checkpoint checkpoints_robust/BEST_MODEL.pytorch \
      --data_file /data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40/train_data_raw_ratio16.npz \
      --output_dir hicarn_predictions \
      --gpu 2
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import json

# Import HiCARN model
try:
    from Models.HiCARN_1 import Generator
except ImportError:
    print("Warning: Could not import HiCARN model. Using placeholder.")
    class Generator(torch.nn.Module):
        def __init__(self, num_channels=64):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
        def forward(self, x):
            return self.conv(x)


class RobustHiCPreprocessor:
    """
    Robust HiC Preprocessor using Median + IQR
    
    This should match the preprocessor used during HiCARN training.
    """
    def __init__(self):
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self.fitted = False
    
    def fit(self, X_low, Y_high):
        """Fit normalization statistics"""
        X_log = np.log1p(X_low)
        Y_log = np.log1p(Y_high)
        
        # Use median and IQR for robust statistics
        self.X_mean = np.median(X_log)
        self.X_std = np.percentile(X_log, 75) - np.percentile(X_log, 25)
        self.Y_mean = np.median(Y_log)
        self.Y_std = np.percentile(Y_log, 75) - np.percentile(Y_log, 25)
        
        # Avoid division by zero
        if self.X_std < 1e-8:
            self.X_std = 1.0
        if self.Y_std < 1e-8:
            self.Y_std = 1.0
        
        self.fitted = True
        return self
    
    def preprocess(self, X_low, Y_high=None):
        """Normalize data"""
        # Check if preprocessor has necessary attributes
        if not hasattr(self, 'X_mean') or self.X_mean is None:
            raise ValueError("Preprocessor not fitted! Missing normalization statistics.")
        
        X_log = np.log1p(X_low)
        X_norm = (X_log - self.X_mean) / self.X_std
        X_norm = np.clip(X_norm, -5, 5).astype(np.float32)
        
        if Y_high is not None:
            Y_log = np.log1p(Y_high)
            Y_norm = (Y_log - self.Y_mean) / self.Y_std
            Y_norm = np.clip(Y_norm, -5, 5).astype(np.float32)
            return X_norm, Y_norm
        
        return X_norm
    
    def postprocess(self, Y_norm):
        """Denormalize predictions"""
        # Check if preprocessor has necessary attributes
        if not hasattr(self, 'Y_mean') or self.Y_mean is None:
            raise ValueError("Preprocessor not fitted! Missing normalization statistics.")
        
        # Clip first
        Y_norm = np.clip(Y_norm, -5, 5)
        
        # Denormalize
        Y_log = Y_norm * self.Y_std + self.Y_mean
        
        # Inverse log
        Y_counts = np.expm1(Y_log)
        
        # Ensure non-negative
        Y_counts = np.maximum(Y_counts, 0.0)
        
        return Y_counts
    
    def get_stats(self):
        """Get normalization statistics"""
        # Handle both new and old preprocessor formats
        if hasattr(self, 'X_mean') and self.X_mean is not None:
            return {
                'X_median': float(self.X_mean),
                'X_iqr': float(self.X_std),
                'Y_median': float(self.Y_mean),
                'Y_iqr': float(self.Y_std)
            }
        else:
            return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate HiCARN predictions for diffusion from .npz file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Generate predictions from your .npz file
  python generate_hicarn_predictions_npz.py \\
      --checkpoint checkpoints_robust/01_08_01_52_bestg_robust_HiCARN_1.pytorch \\
      --data_file /data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40/train_data_raw_ratio16.npz \\
      --output_dir hicarn_predictions \\
      --batch_size 32 \\
      --gpu 2
  
  # Then train diffusion
  python residual_diffusion/train.py \\
      --pred_path hicarn_predictions/predictions_norm.npy \\
      --gt_path hicarn_predictions/ground_truth.npy \\
      --output_dir checkpoints_diffusion \\
      --epochs 100 --batch_size 16 --gpu 2
        """
    )
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained HiCARN checkpoint (.pytorch file)')
    
    # Input data (.npz file with train_lr and train_hr keys)
    parser.add_argument('--data_file', type=str,
                       default='/data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40/train_data_raw_ratio16.npz',
                       help='Path to .npz file with train_lr and train_hr keys (default: GM12878 ratio16)')
    
    # Optional: Use subset of data
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Use only first N samples (for quick testing)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./hicarn_predictions',
                       help='Directory to save predictions (default: ./hicarn_predictions)')
    
    # Options
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--save_raw', action='store_true',
                       help='Also save raw (denormalized) predictions for visualization')
    
    return parser.parse_args()


def load_data_from_npz(npz_path, max_samples=None):
    """
    Load LR and HR data from .npz file with 'train_lr' and 'train_hr' keys
    
    Args:
        npz_path: path to .npz file
        max_samples: if not None, use only first N samples (for testing)
        
    Returns:
        lr_data: low-resolution data (N, C, H, W)
        hr_data: high-resolution data (N, C, H, W)
    """
    print("="*80)
    print("LOADING DATA FROM NPZ")
    print("="*80)
    
    print(f"\nLoading from: {npz_path}")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Data file not found: {npz_path}")
    
    # Load .npz file
    data = np.load(npz_path)
    
    # Check available keys
    available_keys = list(data.keys())
    print(f"Available keys in .npz: {available_keys}")
    
    # Load LR and HR with specific keys
    if 'train_lr' not in data or 'train_hr' not in data:
        raise ValueError(f"Expected keys 'train_lr' and 'train_hr' in .npz file, "
                        f"but found: {available_keys}")
    
    lr_data = data['train_lr']
    hr_data = data['train_hr']
    print(f"✓ Successfully loaded 'train_lr' and 'train_hr'")
    
    # Use subset if requested
    if max_samples is not None:
        print(f"\n⚠ Using only first {max_samples} samples for quick testing")
        lr_data = lr_data[:max_samples]
        hr_data = hr_data[:max_samples]
    
    print(f"\nOriginal data shapes:")
    print(f"  LR: {lr_data.shape}")
    print(f"  HR: {hr_data.shape}")
    
    # Validate
    assert lr_data.shape[0] == hr_data.shape[0], \
        f"LR and HR must have same number of samples, got {lr_data.shape[0]} vs {hr_data.shape[0]}"
    
    # Convert to NCHW format: (N, C, H, W)
    if lr_data.ndim == 3:
        # (N, H, W) → (N, 1, H, W)
        print("\nConverting (N, H, W) → (N, 1, H, W)")
        lr_data = lr_data[:, None, :, :]
        hr_data = hr_data[:, None, :, :]
    elif lr_data.ndim == 4:
        if lr_data.shape[-1] == 1:
            # (N, H, W, 1) → (N, 1, H, W)
            print("\nConverting (N, H, W, 1) → (N, 1, H, W)")
            lr_data = lr_data.transpose(0, 3, 1, 2)
            hr_data = hr_data.transpose(0, 3, 1, 2)
        elif lr_data.shape[1] == 1:
            # Already (N, 1, H, W)
            print("\nData already in (N, 1, H, W) format")
        else:
            print(f"\n⚠ Warning: Unexpected shape {lr_data.shape}, assuming (N, C, H, W)")
    
    print(f"\nFinal NCHW format:")
    print(f"  LR: {lr_data.shape}")
    print(f"  HR: {hr_data.shape}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(lr_data)) or np.any(np.isinf(lr_data)):
        raise ValueError("❌ LR data contains NaN or Inf!")
    if np.any(np.isnan(hr_data)) or np.any(np.isinf(hr_data)):
        raise ValueError("❌ HR data contains NaN or Inf!")
    print("✓ No NaN/Inf detected")
    
    print(f"\nData ranges (raw contact counts):")
    print(f"  LR: [{lr_data.min():.2f}, {lr_data.max():.2f}]")
    print(f"  HR: [{hr_data.min():.2f}, {hr_data.max():.2f}]")
    
    print(f"\nData statistics:")
    print(f"  LR mean: {lr_data.mean():.2f}, std: {lr_data.std():.2f}")
    print(f"  HR mean: {hr_data.mean():.2f}, std: {hr_data.std():.2f}")
    
    print("="*80)
    
    return lr_data, hr_data


def load_checkpoint(checkpoint_path, device):
    """
    Load HiCARN checkpoint
    
    Returns:
        model: loaded Generator model
        preprocessor: loaded preprocessor
    """
    print("\n" + "="*80)
    print("LOADING HICARN CHECKPOINT")
    print("="*80)
    
    print(f"\nLoading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model
    model = Generator(num_channels=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    if 'ssim' in checkpoint:
        print(f"  Model SSIM: {checkpoint['ssim']:.6f}")
    
    # Load preprocessor
    if 'preprocessor' in checkpoint:
        preprocessor = checkpoint['preprocessor']
        print(f"✓ Loaded preprocessor from checkpoint")
        
        # Verify preprocessor has required attributes
        required_attrs = ['X_mean', 'X_std', 'Y_mean', 'Y_std']
        missing_attrs = [attr for attr in required_attrs if not hasattr(preprocessor, attr)]
        
        if missing_attrs:
            raise ValueError(f"Preprocessor is missing required attributes: {missing_attrs}")
        
        # Print preprocessing stats if available
        try:
            if hasattr(preprocessor, 'get_stats'):
                stats = preprocessor.get_stats()
                if stats:
                    print(f"\nPreprocessing statistics:")
                    print(f"  LR - Log-Median: {stats.get('X_median', 'N/A'):.6f}, "
                          f"Log-IQR: {stats.get('X_iqr', 'N/A'):.6f}")
                    print(f"  HR - Log-Median: {stats.get('Y_median', 'N/A'):.6f}, "
                          f"Log-IQR: {stats.get('Y_iqr', 'N/A'):.6f}")
            else:
                # Manually print stats if no get_stats method
                print(f"\nPreprocessing statistics:")
                print(f"  LR - Log-Median: {preprocessor.X_mean:.6f}, Log-IQR: {preprocessor.X_std:.6f}")
                print(f"  HR - Log-Median: {preprocessor.Y_mean:.6f}, Log-IQR: {preprocessor.Y_std:.6f}")
        except Exception as e:
            print(f"\n⚠ Warning: Could not display preprocessing stats: {e}")
            print(f"  But preprocessor should still work correctly")
    else:
        raise ValueError("No preprocessor found in checkpoint! "
                        "Please use a checkpoint that contains the preprocessor.")
    
    print("="*80)
    
    return model, preprocessor


@torch.no_grad()
def generate_predictions(
    model,
    preprocessor,
    lr_data,
    hr_data,
    device,
    batch_size=32
):
    """
    Generate HiCARN predictions
    
    Returns:
        predictions_norm: predictions in normalized space (N, H, W)
        ground_truth_norm: ground truth in normalized space (N, H, W)
        predictions_raw: predictions in raw space (N, H, W)
    """
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    model.eval()
    
    # Preprocess data
    print("\nPreprocessing data with HiCARN preprocessor...")
    lr_norm, hr_norm = preprocessor.preprocess(lr_data, hr_data)
    
    print(f"Normalized data ranges:")
    print(f"  LR: [{lr_norm.min():.2f}, {lr_norm.max():.2f}]")
    print(f"  HR: [{hr_norm.min():.2f}, {hr_norm.max():.2f}]")
    
    # Create dataloader
    dataset = TensorDataset(
        torch.from_numpy(lr_norm).float(),
        torch.from_numpy(hr_norm).float()
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference
    predictions_norm_list = []
    
    print(f"\nRunning HiCARN inference on {len(dataset)} samples...")
    pbar = tqdm(dataloader, desc='Generating predictions')
    
    for lr_batch, _ in pbar:
        lr_batch = lr_batch.to(device)
        
        # Forward pass through HiCARN
        pred_batch = model(lr_batch)
        
        # Store normalized predictions
        predictions_norm_list.append(pred_batch.cpu().numpy())
    
    # Concatenate
    predictions_norm = np.concatenate(predictions_norm_list, axis=0)
    
    print(f"\n✓ Generated predictions")
    print(f"  Shape: {predictions_norm.shape}")
    print(f"  Range (normalized): [{predictions_norm.min():.2f}, {predictions_norm.max():.2f}]")
    
    # Postprocess to get raw predictions
    print("\nPostprocessing to raw contact counts...")
    predictions_raw = preprocessor.postprocess(predictions_norm)
    
    print(f"✓ Denormalized predictions")
    print(f"  Shape: {predictions_raw.shape}")
    print(f"  Range (raw counts): [{predictions_raw.min():.2f}, {predictions_raw.max():.2f}]")
    
    # Ground truth in normalized space
    ground_truth_norm = hr_norm
    
    print("="*80)
    
    return predictions_norm, ground_truth_norm, predictions_raw


def save_outputs(
    predictions_norm,
    ground_truth_norm,
    predictions_raw,
    preprocessor,
    output_dir,
    save_raw=False
):
    """Save predictions and metadata for residual diffusion"""
    print("\n" + "="*80)
    print("SAVING OUTPUTS FOR RESIDUAL DIFFUSION")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove channel dimension if single channel
    if predictions_norm.shape[1] == 1:
        predictions_norm = predictions_norm[:, 0, :, :]
        ground_truth_norm = ground_truth_norm[:, 0, :, :]
        predictions_raw = predictions_raw[:, 0, :, :]
    
    # Save normalized predictions (FOR DIFFUSION TRAINING)
    pred_norm_path = os.path.join(output_dir, 'predictions_norm.npy')
    np.save(pred_norm_path, predictions_norm)
    print(f"\n✓ Saved normalized predictions: {pred_norm_path}")
    print(f"  Shape: {predictions_norm.shape}")
    print(f"  Range: [{predictions_norm.min():.4f}, {predictions_norm.max():.4f}]")
    print(f"  → Use this for --pred_path in diffusion training")
    
    # Save normalized ground truth (FOR DIFFUSION TRAINING)
    gt_norm_path = os.path.join(output_dir, 'ground_truth.npy')
    np.save(gt_norm_path, ground_truth_norm)
    print(f"\n✓ Saved normalized ground truth: {gt_norm_path}")
    print(f"  Shape: {ground_truth_norm.shape}")
    print(f"  Range: [{ground_truth_norm.min():.4f}, {ground_truth_norm.max():.4f}]")
    print(f"  → Use this for --gt_path in diffusion training")
    
    # Save raw predictions (optional, for visualization)
    if save_raw:
        pred_raw_path = os.path.join(output_dir, 'predictions_raw.npy')
        np.save(pred_raw_path, predictions_raw)
        print(f"\n✓ Saved raw predictions: {pred_raw_path}")
        print(f"  Shape: {predictions_raw.shape}")
        print(f"  Range: [{predictions_raw.min():.4f}, {predictions_raw.max():.4f}]")
        print(f"  → For visualization/analysis only")
    
    # Compute residual statistics (for reference)
    residuals = ground_truth_norm - predictions_norm
    print(f"\nResidual statistics (GT - Pred):")
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std:  {residuals.std():.6f}")
    print(f"  Range: [{residuals.min():.4f}, {residuals.max():.4f}]")
    
    # Save metadata
    metadata = {
        'num_samples': int(len(predictions_norm)),
        'shape': list(predictions_norm.shape),
        'normalized_range_pred': [float(predictions_norm.min()), float(predictions_norm.max())],
        'normalized_range_gt': [float(ground_truth_norm.min()), float(ground_truth_norm.max())],
        'residual_mean': float(residuals.mean()),
        'residual_std': float(residuals.std()),
        'preprocessing_method': 'robust_median_iqr',
        'note': 'Both predictions and ground truth are in normalized space (log1p + median/IQR)'
    }
    
    # Add preprocessing stats if available
    try:
        if hasattr(preprocessor, 'get_stats'):
            stats = preprocessor.get_stats()
            if stats:
                metadata['preprocessing_stats'] = stats
        elif hasattr(preprocessor, 'X_mean'):
            metadata['preprocessing_stats'] = {
                'X_median': float(preprocessor.X_mean),
                'X_iqr': float(preprocessor.X_std),
                'Y_median': float(preprocessor.Y_mean),
                'Y_iqr': float(preprocessor.Y_std)
            }
    except Exception as e:
        print(f"⚠ Warning: Could not save preprocessing stats: {e}")
    
    if save_raw:
        metadata['raw_range_pred'] = [float(predictions_raw.min()), float(predictions_raw.max())]
    
    metadata_path = os.path.join(output_dir, 'predictions_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata: {metadata_path}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(output_dir, 'hicarn_preprocessor.pt')
    try:
        torch.save(preprocessor, preprocessor_path)
        print(f"✓ Saved preprocessor: {preprocessor_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not save preprocessor: {e}")
    
    print("\n" + "="*80)
    print("✓ FILES READY FOR RESIDUAL DIFFUSION TRAINING")
    print("="*80)
    print(f"\nTo train residual diffusion, use:")
    print(f"\npython residual_diffusion/train.py \\")
    print(f"    --pred_path {pred_norm_path} \\")
    print(f"    --gt_path {gt_norm_path} \\")
    print(f"    --output_dir checkpoints_diffusion \\")
    print(f"    --epochs 100 \\")
    print(f"    --batch_size 16 \\")
    print(f"    --lambda_recon 0.1 \\")
    print(f"    --gpu 0")
    print("="*80)
    
    return pred_norm_path, gt_norm_path


def main():
    args = parse_args()
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"HICARN PREDICTION GENERATION FOR RESIDUAL DIFFUSION")
    print(f"{'='*80}")
    print(f"Using device: {device}")
    
    # Load data from .npz
    lr_data, hr_data = load_data_from_npz(
        args.data_file,
        max_samples=args.max_samples
    )
    
    # Load HiCARN checkpoint
    model, preprocessor = load_checkpoint(args.checkpoint, device)
    
    # Generate predictions
    predictions_norm, ground_truth_norm, predictions_raw = generate_predictions(
        model=model,
        preprocessor=preprocessor,
        lr_data=lr_data,
        hr_data=hr_data,
        device=device,
        batch_size=args.batch_size
    )
    
    # Save outputs
    pred_path, gt_path = save_outputs(
        predictions_norm=predictions_norm,
        ground_truth_norm=ground_truth_norm,
        predictions_raw=predictions_raw,
        preprocessor=preprocessor,
        output_dir=args.output_dir,
        save_raw=args.save_raw
    )
    
    print("\n" + "="*80)
    print("✓ PREDICTION GENERATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated {len(predictions_norm)} predictions from HiCARN")
    print(f"Data shape: {predictions_norm.shape}")
    print(f"\nWhat was generated:")
    print(f"  1. HiCARN predictions (normalized) → for diffusion input")
    print(f"  2. Ground truth (normalized) → for diffusion target")
    print(f"  3. Metadata and statistics → for reference")
    if args.save_raw:
        print(f"  4. Raw predictions (contact counts) → for visualization")
    print(f"\nNext step: Train residual diffusion")
    print(f"The diffusion model will learn: Δ = Ground_Truth - HiCARN_Prediction")
    print(f"Then refine predictions: Refined = HiCARN_Prediction + Δ_sampled")
    print("="*80)


if __name__ == "__main__":
    main()
