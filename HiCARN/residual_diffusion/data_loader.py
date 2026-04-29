"""
Data Loader for Residual Diffusion

Handles:
1. Loading HiCARN predictions and ground truth
2. Computing residuals: Δ = x_GT - x_HiCARN
3. Normalizing residuals (critical for training)
4. Creating PyTorch datasets and dataloaders
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json


class ResidualNormalizer:
    """
    Normalize residuals for stable diffusion training
    
    Critical: μ and σ must be saved for inference denormalization
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, residuals):
        """
        Compute normalization statistics from residuals
        
        Args:
            residuals: (N, H, W) or (N, C, H, W) array of residuals
        """
        self.mean = np.mean(residuals)
        self.std = np.std(residuals)
        
        # Avoid division by zero
        if self.std < 1e-8:
            print(f"Warning: Very small std ({self.std}), setting to 1.0")
            self.std = 1.0
        
        self.fitted = True
        
        print(f"Residual normalization statistics:")
        print(f"  Mean: {self.mean:.6f}")
        print(f"  Std:  {self.std:.6f}")
        
        return self
    
    def transform(self, residuals):
        """
        Normalize residuals: (Δ - μ) / σ
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted! Call fit() first.")
        
        return (residuals - self.mean) / (self.std + 1e-8)
    
    def inverse_transform(self, residuals_norm):
        """
        Denormalize residuals: Δ = residuals_norm * σ + μ
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted! Call fit() first.")
        
        return residuals_norm * self.std + self.mean
    
    def save(self, filepath):
        """Save normalization statistics"""
        if not self.fitted:
            raise ValueError("Cannot save unfitted normalizer")
        
        stats = {
            'mean': float(self.mean),
            'std': float(self.std)
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved normalization stats to: {filepath}")
    
    def load(self, filepath):
        """Load normalization statistics"""
        with open(filepath, 'r') as f:
            stats = json.load(f)
        
        self.mean = stats['mean']
        self.std = stats['std']
        self.fitted = True
        
        print(f"Loaded normalization stats from: {filepath}")
        print(f"  Mean: {self.mean:.6f}")
        print(f"  Std:  {self.std:.6f}")
        
        return self


class ResidualDataset(Dataset):
    """
    Dataset for residual diffusion training
    
    Returns:
        - condition: HiCARN prediction (x̃)
        - residual_norm: normalized residual (Δ_norm)
        - residual_raw: raw residual (for evaluation)
    """
    
    def __init__(
        self,
        predictions,
        ground_truth,
        normalizer,
        transform=None
    ):
        """
        Args:
            predictions: HiCARN predictions (N, H, W) or (N, C, H, W)
            ground_truth: ground truth HR (N, H, W) or (N, C, H, W)
            normalizer: fitted ResidualNormalizer
            transform: optional data augmentation
        """
        assert predictions.shape == ground_truth.shape, \
            f"Shape mismatch: pred {predictions.shape} vs gt {ground_truth.shape}"
        
        # Ensure 4D: (N, C, H, W)
        if predictions.ndim == 3:
            predictions = predictions[:, None, :, :]
            ground_truth = ground_truth[:, None, :, :]
        
        self.predictions = predictions.astype(np.float32)
        self.ground_truth = ground_truth.astype(np.float32)
        
        # Compute residuals
        self.residuals_raw = ground_truth - predictions
        
        # Normalize
        if not normalizer.fitted:
            raise ValueError("Normalizer must be fitted before creating dataset")
        
        self.residuals_norm = normalizer.transform(self.residuals_raw).astype(np.float32)
        
        self.normalizer = normalizer
        self.transform = transform
        
        print(f"\nDataset created:")
        print(f"  Samples: {len(self)}")
        print(f"  Shape: {self.predictions.shape}")
        print(f"  Residual range (raw): [{self.residuals_raw.min():.2f}, {self.residuals_raw.max():.2f}]")
        print(f"  Residual range (norm): [{self.residuals_norm.min():.2f}, {self.residuals_norm.max():.2f}]")
    
    def __len__(self):
        return len(self.predictions)
    
    def __getitem__(self, idx):
        condition = torch.from_numpy(self.predictions[idx])
        residual_norm = torch.from_numpy(self.residuals_norm[idx])
        residual_raw = torch.from_numpy(self.residuals_raw[idx])
        
        if self.transform:
            condition, residual_norm = self.transform(condition, residual_norm)
        
        return {
            'condition': condition,
            'residual_norm': residual_norm,
            'residual_raw': residual_raw,
            'index': idx
        }


def load_hic_data(
    pred_path,
    gt_path,
    validate_data=True,
    check_symmetry=False
):
    """
    Load HiCARN predictions and ground truth
    
    Args:
        pred_path: path to predictions_norm.npy
        gt_path: path to ground_truth.npy
        validate_data: perform data validation
        check_symmetry: check if matrices are symmetric (Hi-C specific)
    
    Returns:
        predictions: (N, H, W) array
        ground_truth: (N, H, W) array
    """
    print("="*80)
    print("LOADING Hi-C DATA")
    print("="*80)
    
    # Load data
    print(f"Loading predictions from: {pred_path}")
    predictions = np.load(pred_path)
    
    print(f"Loading ground truth from: {gt_path}")
    ground_truth = np.load(gt_path)
    
    print(f"\nLoaded shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    
    # Validate
    if validate_data:
        print("\nValidating data...")
        
        # Check shapes
        assert predictions.shape == ground_truth.shape, \
            f"Shape mismatch: {predictions.shape} vs {ground_truth.shape}"
        
        # Check for NaN/Inf
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            raise ValueError("Predictions contain NaN or Inf!")
        if np.any(np.isnan(ground_truth)) or np.any(np.isinf(ground_truth)):
            raise ValueError("Ground truth contains NaN or Inf!")
        
        print("  ✓ No NaN/Inf")
        
        # Check value ranges
        print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"  Ground truth range: [{ground_truth.min():.4f}, {ground_truth.max():.4f}]")
        
        # Check symmetry (optional, for Hi-C matrices)
        if check_symmetry and predictions.ndim >= 2:
            # Check last two dimensions
            for i in range(min(10, len(predictions))):
                if predictions.ndim == 3:
                    mat = predictions[i]
                elif predictions.ndim == 4:
                    mat = predictions[i, 0]
                
                if not np.allclose(mat, mat.T, rtol=1e-3):
                    print(f"  ⚠ Warning: Sample {i} is not symmetric (Hi-C should be)")
        
        print("  ✓ Data validation passed")
    
    print("="*80)
    
    return predictions, ground_truth


def create_dataloaders(
    predictions,
    ground_truth,
    batch_size=16,
    train_split=0.9,
    num_workers=4,
    seed=42
):
    """
    Create train and validation dataloaders
    
    Args:
        predictions: HiCARN predictions
        ground_truth: ground truth HR
        batch_size: batch size
        train_split: fraction for training
        num_workers: number of data loading workers
        seed: random seed
    
    Returns:
        train_loader: training dataloader
        val_loader: validation dataloader
        normalizer: fitted ResidualNormalizer
    """
    print("\n" + "="*80)
    print("CREATING DATALOADERS")
    print("="*80)
    
    # Set seed
    np.random.seed(seed)
    
    # Split data
    n_total = len(predictions)
    n_train = int(n_total * train_split)
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    print(f"\nData split:")
    print(f"  Total samples: {n_total}")
    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {n_total - n_train}")
    
    # Split data
    pred_train = predictions[train_indices]
    pred_val = predictions[val_indices]
    gt_train = ground_truth[train_indices]
    gt_val = ground_truth[val_indices]
    
    # Fit normalizer on training residuals only
    print("\nFitting residual normalizer...")
    residuals_train = gt_train - pred_train
    normalizer = ResidualNormalizer()
    normalizer.fit(residuals_train)
    
    # Create datasets
    train_dataset = ResidualDataset(pred_train, gt_train, normalizer)
    val_dataset = ResidualDataset(pred_val, gt_val, normalizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    print("="*80)
    
    return train_loader, val_loader, normalizer


if __name__ == "__main__":
    # Test data loading
    
    # Create dummy data
    n_samples = 1000
    H, W = 40, 40
    
    predictions = np.random.randn(n_samples, H, W).astype(np.float32)
    ground_truth = predictions + np.random.randn(n_samples, H, W).astype(np.float32) * 0.1
    
    # Save dummy data
    os.makedirs('dummy_data', exist_ok=True)
    np.save('dummy_data/predictions_norm.npy', predictions)
    np.save('dummy_data/ground_truth.npy', ground_truth)
    
    # Load data
    pred, gt = load_hic_data(
        'dummy_data/predictions_norm.npy',
        'dummy_data/ground_truth.npy'
    )
    
    # Create dataloaders
    train_loader, val_loader, normalizer = create_dataloaders(
        pred, gt, batch_size=32, train_split=0.9
    )
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Condition shape: {batch['condition'].shape}")
    print(f"Residual (norm) shape: {batch['residual_norm'].shape}")
    print(f"Residual (raw) shape: {batch['residual_raw'].shape}")
    
    # Test normalizer save/load
    normalizer.save('dummy_data/normalizer.json')
    
    new_normalizer = ResidualNormalizer()
    new_normalizer.load('dummy_data/normalizer.json')
    
    # Clean up
    import shutil
    shutil.rmtree('dummy_data')
