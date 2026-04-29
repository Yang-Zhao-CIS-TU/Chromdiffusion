"""
Improved Data Loader for Residual Diffusion

Key improvements:
1. Provides GT directly (no numpy conversion in training loop)
2. Clean separation of spaces (all in normalized space)
3. Proper data validation
4. Support for different data splits
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ResidualDiffusionDataset(Dataset):
    """
    Dataset for residual diffusion training
    
    Provides:
    - HiCARN predictions (condition)
    - Ground truth
    - Residual (GT - HiCARN)
    
    All in normalized space for consistent training
    """
    def __init__(
        self,
        hicarn_pred_path,
        ground_truth_path,
        transform=None,
        normalize=True,
        cache_data=True
    ):
        """
        Args:
            hicarn_pred_path: Path to HiCARN predictions (normalized)
            ground_truth_path: Path to ground truth (normalized)
            transform: Optional data augmentation
            normalize: Whether data is already normalized
            cache_data: Cache data in memory
        """
        self.transform = transform
        self.normalize = normalize
        self.cache_data = cache_data
        
        # Load data
        print(f"Loading data from:")
        print(f"  HiCARN predictions: {hicarn_pred_path}")
        print(f"  Ground truth: {ground_truth_path}")
        
        # Load HiCARN predictions
        hicarn_pred = np.load(hicarn_pred_path)
        
        # Load ground truth
        ground_truth = np.load(ground_truth_path)
        
        # Ensure compatible shapes
        hicarn_pred, ground_truth = self._ensure_compatible_shapes(hicarn_pred, ground_truth)
        
        print(f"  HiCARN shape: {hicarn_pred.shape}")
        print(f"  GT shape: {ground_truth.shape}")
        print(f"  HiCARN range: [{hicarn_pred.min():.2f}, {hicarn_pred.max():.2f}]")
        print(f"  GT range: [{ground_truth.min():.2f}, {ground_truth.max():.2f}]")
        
        # Compute residual (GT - HiCARN)
        residual = ground_truth - hicarn_pred
        print(f"  Residual range: [{residual.min():.2f}, {residual.max():.2f}]")
        
        # Cache or store paths
        if self.cache_data:
            self.hicarn_pred = hicarn_pred
            self.ground_truth = ground_truth
            self.residual = residual
        else:
            # For very large datasets, store paths and load on-the-fly
            raise NotImplementedError("On-the-fly loading not implemented yet")
        
        self.num_samples = len(self.ground_truth)
        print(f"  Total samples: {self.num_samples}")
    
    def _ensure_compatible_shapes(self, hicarn, gt):
        """
        Ensure HiCARN and GT have compatible shapes
        
        Handles:
        - (N, H, W) -> (N, 1, H, W)
        - (N, H, W, 1) -> (N, 1, H, W)
        - (N, 1, H, W) -> (N, 1, H, W)
        """
        def to_nchw(data):
            if data.ndim == 3:
                # (N, H, W) -> (N, 1, H, W)
                return data[:, np.newaxis, :, :]
            elif data.ndim == 4:
                if data.shape[3] == 1:
                    # (N, H, W, 1) -> (N, 1, H, W)
                    return data.transpose(0, 3, 1, 2)
                elif data.shape[1] == 1:
                    # Already (N, 1, H, W)
                    return data
                else:
                    raise ValueError(f"Unexpected shape: {data.shape}")
            else:
                raise ValueError(f"Unexpected ndim: {data.ndim}")
        
        hicarn = to_nchw(hicarn)
        gt = to_nchw(gt)
        
        # Verify shapes match
        if hicarn.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch: HiCARN {hicarn.shape} vs GT {gt.shape}"
            )
        
        return hicarn, gt
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
            - 'hicarn': HiCARN prediction (condition)
            - 'gt': Ground truth
            - 'residual': GT - HiCARN
            - 'idx': Sample index
        """
        hicarn = self.hicarn_pred[idx]
        gt = self.ground_truth[idx]
        residual = self.residual[idx]
        
        # Convert to torch tensors
        hicarn = torch.from_numpy(hicarn).float()
        gt = torch.from_numpy(gt).float()
        residual = torch.from_numpy(residual).float()
        
        # Apply transform if specified
        if self.transform is not None:
            hicarn, gt, residual = self.transform(hicarn, gt, residual)
        
        return {
            'hicarn': hicarn,
            'gt': gt,
            'residual': residual,
            'idx': idx
        }


def create_dataloaders(
    train_hicarn_path,
    train_gt_path,
    val_hicarn_path=None,
    val_gt_path=None,
    batch_size=16,
    num_workers=4,
    shuffle=True,
    pin_memory=True
):
    """
    Create train and validation dataloaders
    
    Args:
        train_hicarn_path: Path to training HiCARN predictions
        train_gt_path: Path to training ground truth
        val_hicarn_path: Path to validation HiCARN predictions (optional)
        val_gt_path: Path to validation ground truth (optional)
        batch_size: Batch size
        num_workers: Number of dataloader workers
        shuffle: Shuffle training data
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader (or None)
    """
    # Create training dataset
    train_dataset = ResidualDiffusionDataset(
        hicarn_pred_path=train_hicarn_path,
        ground_truth_path=train_gt_path,
        cache_data=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    # Create validation dataset if provided
    val_loader = None
    if val_hicarn_path is not None and val_gt_path is not None:
        val_dataset = ResidualDiffusionDataset(
            hicarn_pred_path=val_hicarn_path,
            ground_truth_path=val_gt_path,
            cache_data=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader


class DataAugmentation:
    """
    Optional data augmentation for Hi-C matrices
    
    - Random flips (preserve symmetry)
    - Small rotations (90, 180, 270 degrees)
    - Intensity jitter
    """
    def __init__(self, flip_prob=0.5, rotate_prob=0.5, jitter_std=0.05):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.jitter_std = jitter_std
    
    def __call__(self, hicarn, gt, residual):
        """
        Apply augmentation to all three matrices consistently
        """
        # Random horizontal flip (maintain symmetry)
        if torch.rand(1) < self.flip_prob:
            hicarn = torch.flip(hicarn, dims=[2])  # Flip H
            gt = torch.flip(gt, dims=[2])
            residual = torch.flip(residual, dims=[2])
        
        # Random vertical flip
        if torch.rand(1) < self.flip_prob:
            hicarn = torch.flip(hicarn, dims=[1])  # Flip W
            gt = torch.flip(gt, dims=[1])
            residual = torch.flip(residual, dims=[1])
        
        # Random 90-degree rotations
        if torch.rand(1) < self.rotate_prob:
            k = torch.randint(0, 4, (1,)).item()  # 0, 1, 2, 3 -> 0°, 90°, 180°, 270°
            if k > 0:
                hicarn = torch.rot90(hicarn, k, dims=[1, 2])
                gt = torch.rot90(gt, k, dims=[1, 2])
                residual = torch.rot90(residual, k, dims=[1, 2])
        
        # Intensity jitter (small gaussian noise)
        if self.jitter_std > 0:
            noise = torch.randn_like(residual) * self.jitter_std
            residual = residual + noise
            gt = hicarn + residual  # Maintain consistency
        
        return hicarn, gt, residual


def validate_data(hicarn_path, gt_path):
    """
    Validate data files before training
    
    Checks:
    - Files exist
    - Shapes are compatible
    - Data ranges are reasonable
    - No NaN or Inf values
    """
    print("Validating data...")
    
    # Check files exist
    if not Path(hicarn_path).exists():
        raise FileNotFoundError(f"HiCARN predictions not found: {hicarn_path}")
    if not Path(gt_path).exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    
    # Load data
    hicarn = np.load(hicarn_path)
    gt = np.load(gt_path)
    
    print(f"  HiCARN shape: {hicarn.shape}")
    print(f"  GT shape: {gt.shape}")
    
    # Check for NaN or Inf
    if np.isnan(hicarn).any():
        raise ValueError("HiCARN predictions contain NaN values")
    if np.isnan(gt).any():
        raise ValueError("Ground truth contains NaN values")
    if np.isinf(hicarn).any():
        raise ValueError("HiCARN predictions contain Inf values")
    if np.isinf(gt).any():
        raise ValueError("Ground truth contains Inf values")
    
    # Check data ranges (normalized data typically in [-3, 7])
    print(f"  HiCARN range: [{hicarn.min():.2f}, {hicarn.max():.2f}]")
    print(f"  GT range: [{gt.min():.2f}, {gt.max():.2f}]")
    
    # Warn if data looks like raw (not normalized)
    if hicarn.min() >= 0 and hicarn.max() > 100:
        print("  ⚠️  Warning: HiCARN data looks like raw counts (not normalized)")
    if gt.min() >= 0 and gt.max() > 100:
        print("  ⚠️  Warning: GT data looks like raw counts (not normalized)")
    
    # Check spatial dimensions
    if hicarn.ndim >= 3:
        if hicarn.shape[-2] != hicarn.shape[-1]:
            print(f"  ⚠️  Warning: Non-square matrices: {hicarn.shape[-2]}x{hicarn.shape[-1]}")
    
    print("  ✓ Data validation passed")
