"""
Generate HiCARN Predictions for Residual Diffusion

This script:
1. Loads trained HiCARN model
2. Runs inference on test data
3. Saves predictions in NORMALIZED space (predictions_norm.npy)
4. Saves ground truth in NORMALIZED space (ground_truth.npy)

These files are ready for residual diffusion training.
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
    
    def save(self, filepath):
        """Save preprocessor"""
        torch.save(self, filepath)
    
    @staticmethod
    def load(filepath):
        """Load preprocessor"""
        return torch.load(filepath)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate HiCARN predictions for diffusion')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained HiCARN checkpoint (.pytorch file)')
    
    # Input data
    parser.add_argument('--lr_data', type=str, required=True,
                       help='Path to low-resolution test data (.npy or .npz)')
    parser.add_argument('--hr_data', type=str, required=True,
                       help='Path to high-resolution ground truth (.npy or .npz)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./hicarn_predictions',
                       help='Directory to save predictions')
    parser.add_argument('--output_name', type=str, default='predictions',
                       help='Base name for output files')
    
    # Options
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--save_raw', action='store_true',
                       help='Also save raw (denormalized) predictions')
    
    return parser.parse_args()


def load_data(lr_path, hr_path):
    """
    Load LR and HR data
    
    Supports .npy and .npz formats
    """
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load LR data
    print(f"\nLoading LR data from: {lr_path}")
    if lr_path.endswith('.npz'):
        lr_data = np.load(lr_path)['data']
    else:
        lr_data = np.load(lr_path)
    
    # Load HR data
    print(f"Loading HR data from: {hr_path}")
    if hr_path.endswith('.npz'):
        hr_data = np.load(hr_path)['data']
    else:
        hr_data = np.load(hr_path)
    
    print(f"\nData shapes:")
    print(f"  LR: {lr_data.shape}")
    print(f"  HR: {hr_data.shape}")
    
    # Validate
    assert lr_data.shape[0] == hr_data.shape[0], "LR and HR must have same number of samples"
    
    # Convert to NCHW if needed
    if lr_data.ndim == 3:
        lr_data = lr_data[:, None, :, :]
    elif lr_data.ndim == 4 and lr_data.shape[-1] == 1:
        lr_data = lr_data.transpose(0, 3, 1, 2)
    
    if hr_data.ndim == 3:
        hr_data = hr_data[:, None, :, :]
    elif hr_data.ndim == 4 and hr_data.shape[-1] == 1:
        hr_data = hr_data.transpose(0, 3, 1, 2)
    
    print(f"\nAfter NCHW conversion:")
    print(f"  LR: {lr_data.shape}")
    print(f"  HR: {hr_data.shape}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(lr_data)) or np.any(np.isinf(lr_data)):
        raise ValueError("LR data contains NaN or Inf!")
    if np.any(np.isnan(hr_data)) or np.any(np.isinf(hr_data)):
        raise ValueError("HR data contains NaN or Inf!")
    
    print(f"\nData ranges:")
    print(f"  LR: [{lr_data.min():.2f}, {lr_data.max():.2f}]")
    print(f"  HR: [{hr_data.min():.2f}, {hr_data.max():.2f}]")
    
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
    print("LOADING CHECKPOINT")
    print("="*80)
    
    print(f"\nLoading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model
    model = Generator(num_channels=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  SSIM: {checkpoint.get('ssim', 'N/A')}")
    
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
                    print(f"  LR - Median: {stats.get('X_median', 'N/A'):.6f}, "
                          f"IQR: {stats.get('X_iqr', 'N/A'):.6f}")
                    print(f"  HR - Median: {stats.get('Y_median', 'N/A'):.6f}, "
                          f"IQR: {stats.get('Y_iqr', 'N/A'):.6f}")
            else:
                # Manually print stats if no get_stats method
                print(f"\nPreprocessing statistics:")
                print(f"  LR - Median: {preprocessor.X_mean:.6f}, IQR: {preprocessor.X_std:.6f}")
                print(f"  HR - Median: {preprocessor.Y_mean:.6f}, IQR: {preprocessor.Y_std:.6f}")
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
    print("\nPreprocessing data...")
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
    
    print(f"\nRunning inference on {len(dataset)} samples...")
    pbar = tqdm(dataloader, desc='Generating predictions')
    
    for lr_batch, _ in pbar:
        lr_batch = lr_batch.to(device)
        
        # Forward pass
        pred_batch = model(lr_batch)
        
        # Store normalized predictions
        predictions_norm_list.append(pred_batch.cpu().numpy())
    
    # Concatenate
    predictions_norm = np.concatenate(predictions_norm_list, axis=0)
    
    print(f"\n✓ Generated predictions")
    print(f"  Shape: {predictions_norm.shape}")
    print(f"  Range: [{predictions_norm.min():.2f}, {predictions_norm.max():.2f}]")
    
    # Postprocess to get raw predictions
    print("\nPostprocessing to raw space...")
    predictions_raw = preprocessor.postprocess(predictions_norm)
    
    print(f"✓ Denormalized predictions")
    print(f"  Shape: {predictions_raw.shape}")
    print(f"  Range: [{predictions_raw.min():.2f}, {predictions_raw.max():.2f}]")
    
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
    output_name,
    save_raw=False
):
    """Save predictions and metadata"""
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove channel dimension if single channel
    if predictions_norm.shape[1] == 1:
        predictions_norm = predictions_norm[:, 0, :, :]
        ground_truth_norm = ground_truth_norm[:, 0, :, :]
        predictions_raw = predictions_raw[:, 0, :, :]
    
    # Save normalized predictions (for diffusion)
    pred_norm_path = os.path.join(output_dir, f'{output_name}_norm.npy')
    np.save(pred_norm_path, predictions_norm)
    print(f"\n✓ Saved normalized predictions: {pred_norm_path}")
    print(f"  Shape: {predictions_norm.shape}")
    print(f"  Range: [{predictions_norm.min():.4f}, {predictions_norm.max():.4f}]")
    
    # Save normalized ground truth (for diffusion)
    gt_norm_path = os.path.join(output_dir, 'ground_truth.npy')
    np.save(gt_norm_path, ground_truth_norm)
    print(f"\n✓ Saved normalized ground truth: {gt_norm_path}")
    print(f"  Shape: {ground_truth_norm.shape}")
    print(f"  Range: [{ground_truth_norm.min():.4f}, {ground_truth_norm.max():.4f}]")
    
    # Save raw predictions (optional)
    if save_raw:
        pred_raw_path = os.path.join(output_dir, f'{output_name}_raw.npy')
        np.save(pred_raw_path, predictions_raw)
        print(f"\n✓ Saved raw predictions: {pred_raw_path}")
        print(f"  Shape: {predictions_raw.shape}")
        print(f"  Range: [{predictions_raw.min():.4f}, {predictions_raw.max():.4f}]")
    
    # Save metadata
    metadata = {
        'num_samples': int(len(predictions_norm)),
        'shape': list(predictions_norm.shape),
        'normalized_range': [float(predictions_norm.min()), float(predictions_norm.max())],
        'preprocessing_method': 'robust_median_iqr'
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
        metadata['raw_range'] = [float(predictions_raw.min()), float(predictions_raw.max())]
    
    metadata_path = os.path.join(output_dir, f'{output_name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata: {metadata_path}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pt')
    try:
        torch.save(preprocessor, preprocessor_path)
        print(f"✓ Saved preprocessor: {preprocessor_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not save preprocessor: {e}")
    
    print("\n" + "="*80)
    print("FILES READY FOR RESIDUAL DIFFUSION")
    print("="*80)
    print(f"\nFor training diffusion, use:")
    print(f"  --pred_path {pred_norm_path}")
    print(f"  --gt_path {gt_norm_path}")
    print("="*80)
    
    return pred_norm_path, gt_norm_path


def main():
    args = parse_args()
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    lr_data, hr_data = load_data(args.lr_data, args.hr_data)
    
    # Load checkpoint
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
        output_name=args.output_name,
        save_raw=args.save_raw
    )
    
    print("\n" + "="*80)
    print("✓ PREDICTION GENERATION COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Train residual diffusion:")
    print(f"   python residual_diffusion/train.py \\")
    print(f"       --pred_path {pred_path} \\")
    print(f"       --gt_path {gt_path} \\")
    print(f"       --output_dir checkpoints_diffusion \\")
    print(f"       --epochs 100 --batch_size 16")
    print(f"\n2. Check the metadata file for preprocessing details")
    print(f"\n3. Verify predictions look reasonable (visualize some samples)")
    print("="*80)


if __name__ == "__main__":
    main()
