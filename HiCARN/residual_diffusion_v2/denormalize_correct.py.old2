"""
Denormalization Script for RobustHiCPreprocessor

This script correctly implements the reverse transformation for HiC data
preprocessed with RobustHiCPreprocessor (median/IQR + log1p transform).

Preprocessing (forward):
    1. log1p(data)
    2. (log_data - median) / IQR
    3. clip to [-5, 5]

Denormalization (reverse):
    1. log_data = norm_data * IQR + median
    2. data = expm1(log_data)  # exp(x) - 1
    3. max(data, 0)
"""

import numpy as np
import torch
from pathlib import Path
import argparse
import json
import sys


# ================================================================
# Define RobustHiCPreprocessor class so torch.load can work
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
        elif arr.shape[1] == 1 and arr.shape[-1] == 1:
            return arr
        else:
            raise ValueError(f"Cannot infer channel axis for shape {arr.shape}")
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape={arr.shape}")


class RobustHiCPreprocessor:
    """
    Dummy class to allow torch.load to deserialize the preprocessor.
    The actual implementation matches the one in train_hicarn_robust.py
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


# Make RobustHiCPreprocessor available to torch.load
# This is needed because the class was saved from a different module
sys.modules['__main__'].RobustHiCPreprocessor = RobustHiCPreprocessor
sys.modules['__main__'].ensure_nchw = ensure_nchw


# ================================================================
# Main denormalization functions
# ================================================================


def load_preprocessor_stats(preprocessor_path):
    """
    Extract normalization statistics from preprocessor file.
    
    The preprocessor is a RobustHiCPreprocessor object with:
    - Y_mean: actually the median of log1p(Y)
    - Y_std: actually the IQR of log1p(Y)
    
    Returns:
        dict with 'Y_median' and 'Y_iqr'
    """
    print("="*80)
    print("LOADING PREPROCESSOR STATISTICS")
    print("="*80)
    
    try:
        # Load with weights_only=False to allow custom classes
        try:
            checkpoint = torch.load(preprocessor_path, map_location='cpu', weights_only=False)
            print(f"✓ Loaded file successfully")
        except TypeError:
            # Older PyTorch versions don't have weights_only
            checkpoint = torch.load(preprocessor_path, map_location='cpu')
            print(f"✓ Loaded file successfully (older PyTorch)")
        
        # Check what we got
        print(f"  Type: {type(checkpoint)}")
        
        # Check if it's a checkpoint dict with preprocessor
        if isinstance(checkpoint, dict):
            print(f"  Dict keys: {list(checkpoint.keys())}")
            
            if 'preprocessor' in checkpoint:
                preprocessor = checkpoint['preprocessor']
                print(f"✓ Found preprocessor in checkpoint dict")
            else:
                # Maybe the whole dict contains stats directly
                if 'Y_median' in checkpoint or 'Y_mean' in checkpoint:
                    Y_median = checkpoint.get('Y_median', checkpoint.get('Y_mean'))
                    Y_iqr = checkpoint.get('Y_iqr', checkpoint.get('Y_std'))
                    print(f"✓ Found stats directly in dict")
                    print(f"\nPreprocessor statistics:")
                    print(f"  Y_median (log space): {Y_median:.6f}")
                    print(f"  Y_iqr (log space):    {Y_iqr:.6f}")
                    print("="*80)
                    return {'Y_median': float(Y_median), 'Y_iqr': float(Y_iqr)}
                else:
                    preprocessor = checkpoint
        else:
            preprocessor = checkpoint
            print(f"✓ Loaded preprocessor object directly")
        
        # Now extract from preprocessor object
        if isinstance(preprocessor, RobustHiCPreprocessor):
            print(f"✓ Preprocessor is RobustHiCPreprocessor")
            Y_median = float(preprocessor.Y_mean)  # Actually median
            Y_iqr = float(preprocessor.Y_std)      # Actually IQR
            
            print(f"\nPreprocessor statistics:")
            print(f"  Y_median (log space): {Y_median:.6f}")
            print(f"  Y_iqr (log space):    {Y_iqr:.6f}")
            print(f"  Method: Robust (median + IQR)")
            print("="*80)
            
            return {'Y_median': Y_median, 'Y_iqr': Y_iqr}
        
        elif hasattr(preprocessor, 'Y_mean') and hasattr(preprocessor, 'Y_std'):
            Y_median = float(preprocessor.Y_mean)
            Y_iqr = float(preprocessor.Y_std)
            
            print(f"\nPreprocessor statistics:")
            print(f"  Y_median (log space): {Y_median:.6f}")
            print(f"  Y_iqr (log space):    {Y_iqr:.6f}")
            print("="*80)
            
            return {'Y_median': Y_median, 'Y_iqr': Y_iqr}
        
        elif hasattr(preprocessor, 'get_stats'):
            stats = preprocessor.get_stats()
            print(f"✓ Got stats from get_stats() method")
            return {
                'Y_median': stats.get('Y_median', stats.get('Y_mean')),
                'Y_iqr': stats.get('Y_iqr', stats.get('Y_std'))
            }
        
        else:
            print(f"✗ Cannot extract Y_mean/Y_std from preprocessor")
            print(f"Preprocessor type: {type(preprocessor)}")
            if hasattr(preprocessor, '__dict__'):
                print(f"Attributes: {list(preprocessor.__dict__.keys())}")
            return None
    
    except Exception as e:
        print(f"✗ Failed to load preprocessor: {e}")
        import traceback
        traceback.print_exc()
        return None


def estimate_stats_from_norm_and_gt(hicarn_norm_path, gt_path):
    """
    Estimate Y_median and Y_iqr using HiCARN norm predictions and ground truth.
    
    Given:
        - hicarn_norm: normalized predictions
        - gt: ground truth in raw space
    
    Assuming HiCARN predictions and GT are similar, we can estimate the parameters.
    
    NOTE: This only works if ground_truth is in RAW space (non-negative counts).
    If ground_truth is also in normalized space (has negative values), this will fail.
    """
    print("\n" + "="*80)
    print("ESTIMATING STATS FROM NORM PREDICTIONS AND GROUND TRUTH")
    print("="*80)
    
    # Load data
    hicarn_norm = np.load(hicarn_norm_path)
    gt = np.load(gt_path)
    
    print(f"HiCARN norm: {hicarn_norm.shape}, range [{hicarn_norm.min():.3f}, {hicarn_norm.max():.3f}]")
    print(f"Ground truth: {gt.shape}, range [{gt.min():.3f}, {gt.max():.3f}]")
    
    # Check if ground_truth is in raw space (non-negative) or normalized space (has negative values)
    if gt.min() < -0.01:  # Small tolerance for floating point
        print(f"\n⚠️  WARNING: Ground truth has negative values ({gt.min():.3f})")
        print(f"   This suggests ground_truth.npy is in NORMALIZED space, not raw space!")
        print(f"   Cannot estimate parameters from normalized ground truth.")
        print(f"\n   Please provide the preprocessor.pt file to load stats directly.")
        print("="*80)
        return None
    
    # Transform GT to log space
    log_gt = np.log1p(gt)
    
    print(f"Log(GT): range [{log_gt.min():.3f}, {log_gt.max():.3f}]")
    
    # Check for NaN
    if np.isnan(log_gt).any():
        print(f"\n⚠️  WARNING: log1p(ground_truth) contains NaN values!")
        print(f"   This likely means ground_truth has invalid values.")
        print("="*80)
        return None
    
    # Estimate relationship: hicarn_norm ≈ (log_gt - median) / IQR
    # Rearranging: log_gt = hicarn_norm * IQR + median
    # Use linear regression
    
    norm_flat = hicarn_norm.flatten()
    log_flat = log_gt.flatten()
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(norm_flat) | np.isnan(log_flat) | np.isinf(norm_flat) | np.isinf(log_flat))
    norm_flat = norm_flat[valid_mask]
    log_flat = log_flat[valid_mask]
    
    if len(norm_flat) == 0:
        print(f"\n⚠️  No valid data points for estimation!")
        return None
    
    # Sample subset for speed
    subset_size = min(100000, len(norm_flat))
    indices = np.random.choice(len(norm_flat), subset_size, replace=False)
    
    norm_subset = norm_flat[indices]
    log_subset = log_flat[indices]
    
    # Linear regression
    try:
        A = np.vstack([norm_subset, np.ones(len(norm_subset))]).T
        Y_iqr_est, Y_median_est = np.linalg.lstsq(A, log_subset, rcond=None)[0]
    except Exception as e:
        print(f"\n⚠️  Linear regression failed: {e}")
        return None
    
    print(f"\nEstimated statistics:")
    print(f"  Y_median: {Y_median_est:.6f}")
    print(f"  Y_iqr:    {Y_iqr_est:.6f}")
    
    # Verify by reconstructing
    log_reconstructed = norm_subset * Y_iqr_est + Y_median_est
    mse = np.mean((log_reconstructed - log_subset)**2)
    print(f"  Log space MSE: {mse:.6e}")
    
    # Test denormalization
    raw_reconstructed = np.expm1(log_reconstructed)
    gt_subset_actual = gt.flatten()[valid_mask][indices]
    raw_mse = np.mean((raw_reconstructed - gt_subset_actual)**2)
    print(f"  Raw space MSE: {raw_mse:.6e}")
    
    print("\nNote: Using ground truth as reference since predictions_raw.npy not available")
    print("="*80)
    
    return {'Y_median': Y_median_est, 'Y_iqr': Y_iqr_est}


def load_stats_from_json(json_path):
    """
    Load preprocessing statistics from JSON file.
    
    This is an alternative if the preprocessor object can't be loaded.
    """
    print("\n" + "="*80)
    print("LOADING STATS FROM JSON")
    print("="*80)
    
    with open(json_path, 'r') as f:
        stats = json.load(f)
    
    print(f"✓ Loaded from {json_path}")
    print(f"  Y_median: {stats['Y_median']:.6f}")
    print(f"  Y_iqr:    {stats['Y_iqr']:.6f}")
    print("="*80)
    
    return stats


def robust_denormalize(normalized_data, Y_median, Y_iqr):
    """
    Denormalize using RobustHiCPreprocessor method.
    
    This implements the exact reverse of the preprocessing:
    
    Forward (preprocess):
        log_data = log1p(raw_data)
        norm_data = (log_data - median) / IQR
        norm_data = clip(norm_data, -5, 5)
    
    Reverse (denormalize):
        norm_data = clip(norm_data, -5, 5)  # safety
        log_data = norm_data * IQR + median
        raw_data = expm1(log_data)
        raw_data = max(raw_data, 0)
    
    Args:
        normalized_data: Data in normalized space (clipped to [-5, 5])
        Y_median: Median of log1p(Y) from training data
        Y_iqr: IQR of log1p(Y) from training data
    
    Returns:
        Data in original contact count space
    """
    # Step 1: Clip to safe range (in case of out-of-range predictions)
    # Training clipped to [-5, 5], so we do the same
    norm_clipped = np.clip(normalized_data, -5, 5)
    
    # Step 2: Reverse normalization (go back to log space)
    log_data = norm_clipped * Y_iqr + Y_median
    
    # Step 3: Reverse log transform
    # log_data = log1p(raw_data) = log(raw_data + 1)
    # So: raw_data = exp(log_data) - 1 = expm1(log_data)
    raw_data = np.expm1(log_data)
    
    # Step 4: Ensure non-negative (Hi-C data represents counts)
    raw_data = np.maximum(raw_data, 0.0)
    
    return raw_data.astype(np.float32)


def estimate_stats_from_hicarn(hicarn_norm_path, hicarn_raw_path):
    """
    Estimate Y_median and Y_iqr by reverse-engineering from HiCARN data.
    
    Given:
        - hicarn_norm: normalized predictions
        - hicarn_raw: raw predictions
    
    We know:
        hicarn_norm = (log1p(hicarn_raw) - median) / IQR
    
    So:
        log1p(hicarn_raw) = hicarn_norm * IQR + median
    
    We can solve for median and IQR using the relationship.
    """
    print("\n" + "="*80)
    print("ESTIMATING STATS FROM HICARN DATA")
    print("="*80)
    
    # Load data
    hicarn_norm = np.load(hicarn_norm_path)
    hicarn_raw = np.load(hicarn_raw_path)
    
    print(f"HiCARN norm: {hicarn_norm.shape}, range [{hicarn_norm.min():.3f}, {hicarn_norm.max():.3f}]")
    print(f"HiCARN raw:  {hicarn_raw.shape}, range [{hicarn_raw.min():.3f}, {hicarn_raw.max():.3f}]")
    
    # Transform raw to log space
    log_data = np.log1p(hicarn_raw)
    
    # Now we have:
    # hicarn_norm ≈ (log_data - median) / IQR
    # 
    # Rearranging:
    # log_data = hicarn_norm * IQR + median
    #
    # Use linear regression: log_data = a * hicarn_norm + b
    # where a = IQR, b = median
    
    norm_flat = hicarn_norm.flatten()
    log_flat = log_data.flatten()
    
    # Sample subset for speed
    subset_size = min(100000, len(norm_flat))
    indices = np.random.choice(len(norm_flat), subset_size, replace=False)
    
    norm_subset = norm_flat[indices]
    log_subset = log_flat[indices]
    
    # Linear regression
    A = np.vstack([norm_subset, np.ones(len(norm_subset))]).T
    Y_iqr_est, Y_median_est = np.linalg.lstsq(A, log_subset, rcond=None)[0]
    
    print(f"\nEstimated statistics:")
    print(f"  Y_median: {Y_median_est:.6f}")
    print(f"  Y_iqr:    {Y_iqr_est:.6f}")
    
    # Verify by reconstructing
    log_reconstructed = norm_subset * Y_iqr_est + Y_median_est
    mse = np.mean((log_reconstructed - log_subset)**2)
    print(f"  Reconstruction MSE: {mse:.6e}")
    
    # Test denormalization
    raw_reconstructed = np.expm1(log_reconstructed)
    raw_subset_actual = hicarn_raw.flatten()[indices]
    raw_mse = np.mean((raw_reconstructed - raw_subset_actual)**2)
    print(f"  Raw space MSE:      {raw_mse:.6e}")
    
    print("="*80)
    
    return {'Y_median': Y_median_est, 'Y_iqr': Y_iqr_est}


def denormalize_directory(input_dir, Y_median, Y_iqr, output_key='refined_raw.npy'):
    """
    Denormalize refined_norm.npy in a directory using RobustHiCPreprocessor method.
    
    Args:
        input_dir: Directory containing refined_norm.npy
        Y_median: Median of log1p(Y) from training
        Y_iqr: IQR of log1p(Y) from training
        output_key: Output filename
    """
    input_path = Path(input_dir)
    norm_file = input_path / 'refined_norm.npy'
    
    if not norm_file.exists():
        print(f"⚠️  {norm_file} not found, skipping")
        return
    
    # Load normalized data
    norm_data = np.load(norm_file)
    
    # Denormalize using robust method
    raw_data = robust_denormalize(norm_data, Y_median, Y_iqr)
    
    # Save
    output_file = input_path / output_key
    np.save(output_file, raw_data)
    
    print(f"✓ {input_dir}:")
    print(f"    Norm range:  [{norm_data.min():.3f}, {norm_data.max():.3f}]")
    print(f"    Raw range:   [{raw_data.min():.3f}, {raw_data.max():.3f}]")
    print(f"    Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Denormalize refined predictions using RobustHiCPreprocessor method'
    )
    
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['load', 'estimate', 'manual'],
                       help='Mode: load (from checkpoint), estimate (from HiCARN data), manual (provide values)')
    
    # Directories to process
    parser.add_argument('--refined_dirs', type=str, nargs='+',
                       default=['refined_chr18', 'refined_chr19', 'refined_chr20', 
                               'refined_chr21', 'refined_chr22'],
                       help='Directories containing refined_norm.npy files')
    
    # For load mode
    parser.add_argument('--checkpoint_path', type=str,
                       default=None,
                       help='Path to checkpoint containing preprocessor (for load mode)')
    
    parser.add_argument('--stats_json', type=str,
                       default=None,
                       help='Path to preprocessing_stats.json (alternative to checkpoint)')
    
    # For estimate mode
    parser.add_argument('--hicarn_norm', type=str,
                       default='/home/yangz/data/hic_data/HiCARN/hicarn_predictions/chr18/predictions_norm.npy',
                       help='Path to HiCARN normalized predictions')
    parser.add_argument('--hicarn_raw', type=str,
                       default=None,
                       help='Path to HiCARN raw predictions (optional, can use ground_truth instead)')
    parser.add_argument('--ground_truth', type=str,
                       default=None,
                       help='Path to ground truth (alternative to hicarn_raw)')
    parser.add_argument('--preprocessor', type=str,
                       default='/home/yangz/data/hic_data/HiCARN/hicarn_predictions/chr18/preprocessor.pt',
                       help='Path to preprocessor.pt (for estimate mode)')
    
    # For manual mode
    parser.add_argument('--Y_median', type=float, default=None,
                       help='Y_median (median of log1p(Y)) for manual mode')
    parser.add_argument('--Y_iqr', type=float, default=None,
                       help='Y_iqr (IQR of log1p(Y)) for manual mode')
    
    args = parser.parse_args()
    
    # ========================================================================
    # STEP 1: Get Y_median and Y_iqr
    # ========================================================================
    
    stats = None
    
    if args.mode == 'load':
        # Try to load from checkpoint
        if args.checkpoint_path:
            stats = load_preprocessor_stats(args.checkpoint_path)
        elif args.stats_json:
            stats = load_stats_from_json(args.stats_json)
        else:
            print("Error: --checkpoint_path or --stats_json required for load mode")
            return
    
    elif args.mode == 'estimate':
        # Try multiple methods to estimate
        
        # Method 1: Try loading from preprocessor.pt
        if args.preprocessor and Path(args.preprocessor).exists():
            print(f"\n🔍 Attempting to load from preprocessor: {args.preprocessor}")
            stats = load_preprocessor_stats(args.preprocessor)
            
            if stats is not None:
                print("✓ Successfully loaded stats from preprocessor!")
        
        # Method 2: Estimate from HiCARN norm and raw predictions
        if stats is None and args.hicarn_raw and Path(args.hicarn_raw).exists():
            print(f"\n🔍 Estimating from HiCARN norm and raw predictions")
            stats = estimate_stats_from_hicarn(args.hicarn_norm, args.hicarn_raw)
        
        # Method 3: Estimate from HiCARN norm and ground truth
        if stats is None:
            # Try to find ground_truth.npy in same directory as hicarn_norm
            if args.ground_truth and Path(args.ground_truth).exists():
                gt_path = args.ground_truth
            else:
                # Auto-detect ground_truth.npy in same directory
                norm_path = Path(args.hicarn_norm)
                gt_path = norm_path.parent / 'ground_truth.npy'
            
            if Path(gt_path).exists():
                print(f"\n🔍 Estimating from norm predictions and ground truth")
                print(f"Using ground truth: {gt_path}")
                stats = estimate_stats_from_norm_and_gt(args.hicarn_norm, str(gt_path))
            else:
                print(f"\n✗ No ground_truth.npy found at: {gt_path}")
    
    elif args.mode == 'manual':
        # Use provided parameters
        if args.Y_median is None or args.Y_iqr is None:
            print("Error: --Y_median and --Y_iqr required for manual mode")
            return
        stats = {'Y_median': args.Y_median, 'Y_iqr': args.Y_iqr}
        print(f"\nUsing manual parameters:")
        print(f"  Y_median: {args.Y_median}")
        print(f"  Y_iqr:    {args.Y_iqr}")
    
    if stats is None:
        print("\n✗ Failed to obtain normalization statistics!")
        print("\nTry one of these:")
        print("  1. Use --mode estimate to estimate from HiCARN data")
        print("  2. Use --mode load --stats_json preprocessing_stats.json")
        print("  3. Use --mode manual --Y_median <val> --Y_iqr <val>")
        return
    
    Y_median = stats['Y_median']
    Y_iqr = stats['Y_iqr']
    
    # ========================================================================
    # STEP 2: Denormalize all directories
    # ========================================================================
    
    print("\n" + "="*80)
    print("DENORMALIZING REFINED PREDICTIONS")
    print("="*80)
    print(f"Method: RobustHiCPreprocessor (median + IQR + log1p)")
    print(f"Parameters:")
    print(f"  Y_median (log space): {Y_median:.6f}")
    print(f"  Y_iqr (log space):    {Y_iqr:.6f}")
    print()
    
    for refined_dir in args.refined_dirs:
        denormalize_directory(
            refined_dir,
            Y_median=Y_median,
            Y_iqr=Y_iqr
        )
    
    # ========================================================================
    # STEP 3: Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Denormalization method:")
    print(f"  1. Clip normalized data to [-5, 5]")
    print(f"  2. Reverse normalization: log_data = norm * {Y_iqr:.4f} + {Y_median:.4f}")
    print(f"  3. Reverse log transform: raw = exp(log_data) - 1")
    print(f"  4. Ensure non-negative: raw = max(raw, 0)")
    print()
    print(f"Output files created:")
    for refined_dir in args.refined_dirs:
        output_file = Path(refined_dir) / 'refined_raw.npy'
        if output_file.exists():
            print(f"  ✓ {output_file}")
        else:
            print(f"  ✗ {output_file} (not created)")
    print()
    print("Ready for evaluation!")
    print("="*80)


if __name__ == '__main__':
    main()
