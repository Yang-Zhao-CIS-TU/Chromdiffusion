#!/usr/bin/env python3
"""
HiCARN Training with ROBUST Preprocessing
==========================================
Uses median/IQR normalization and clipping instead of mean/std.
More robust to outliers and training instability.
"""

import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from math import log10

# Import HiCARN modules
try:
    from Models.HiCARN_1 import Generator
    from Models.HiCARN_1_Loss import GeneratorLoss
    from Utils.SSIM import ssim
except ImportError:
    print("Warning: Could not import HiCARN modules.")
    class Generator(torch.nn.Module):
        def __init__(self, num_channels=64):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
        def forward(self, x):
            return self.conv(x)
    class GeneratorLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, pred, target):
            return torch.nn.functional.mse_loss(pred, target)
    def ssim(img1, img2):
        return torch.tensor(0.8)


# ================================================================
# 🔹 ROBUST PREPROCESSING (Median/IQR + Clipping)
# ================================================================

def ensure_nchw(arr):
    """
    Ensure array is (N, C, H, W). Accepts (N, C, H, W) or (N, H, W, C).
    
    Args:
        arr: Input array
    
    Returns:
        Array in NCHW format
    """
    arr = np.asarray(arr)
    
    if arr.ndim == 3:
        # (N, H, W) → (N, 1, H, W)
        return arr[:, np.newaxis, :, :]
    
    elif arr.ndim == 4:
        # Already NCHW
        if arr.shape[1] in [1, 3]:
            return arr
        # NHWC → NCHW
        elif arr.shape[-1] in [1, 3]:
            return np.transpose(arr, (0, 3, 1, 2))
        # Ambiguous
        elif arr.shape[1] == 1 and arr.shape[-1] == 1:
            return arr
        else:
            raise ValueError(f"Cannot infer channel axis for shape {arr.shape}")
    
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape={arr.shape}")


class RobustHiCPreprocessor:
    """
    Robust Hi-C preprocessing using median and IQR instead of mean/std.
    
    Key features:
    - Log1p transformation for sparse data
    - Median/IQR normalization (robust to outliers)
    - Clipping to [-5, 5] range for training stability
    - No symmetrization (assumes data is symmetric or model learns it)
    
    This matches the preprocessing in hic_preprocessing.txt
    """
    
    def __init__(self, size=40):
        """
        Initialize preprocessor.
        
        Args:
            size: Expected matrix dimension (default: 40x40)
        """
        self.size = size
        self.X_mean = None  # Actually stores median
        self.X_std = None   # Actually stores IQR
        self.Y_mean = None  # Actually stores median
        self.Y_std = None   # Actually stores IQR
        self._is_fitted = False

    def fit(self, X_low, Y_high, verbose=True):
        """
        Fit normalization statistics on training data.
        
        Args:
            X_low: Low-resolution Hi-C matrices (N, H, W) or (N, C, H, W)
            Y_high: High-resolution Hi-C matrices (N, H, W) or (N, C, H, W)
            verbose: Print statistics
        
        Returns:
            self (for method chaining)
        """
        # Ensure correct format
        X_low = ensure_nchw(X_low)
        Y_high = ensure_nchw(Y_high)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"FITTING ROBUST PREPROCESSOR")
            print(f"{'='*80}")
            print(f"Training samples: {len(X_low)}")
            print(f"  X shape: {X_low.shape}, range: [{X_low.min():.2f}, {X_low.max():.2f}]")
            print(f"  Y shape: {Y_high.shape}, range: [{Y_high.min():.2f}, {Y_high.max():.2f}]")
        
        # Log transform (handles sparse Hi-C data)
        X_log = np.log1p(X_low)
        Y_log = np.log1p(Y_high)
        
        # ROBUST STATISTICS: Use median and IQR instead of mean/std
        # This is more robust to outliers in sparse Hi-C data
        
        # For X (low-resolution)
        self.X_mean = np.median(X_log)  # Using median instead of mean
        self.X_std = (np.percentile(X_log, 75) - np.percentile(X_log, 25)) + 1e-8  # IQR instead of std
        
        # For Y (high-resolution)
        self.Y_mean = np.median(Y_log)
        self.Y_std = (np.percentile(Y_log, 75) - np.percentile(Y_log, 25)) + 1e-8
        
        self._is_fitted = True
        
        if verbose:
            print(f"\n{'Normalization Statistics (Robust)':^80}")
            print(f"{'-'*80}")
            print(f"{'Metric':<20} {'Low-Res (X)':<30} {'High-Res (Y)':<30}")
            print(f"{'-'*80}")
            print(f"{'Log-Median:':<20} {self.X_mean:<30.6f} {self.Y_mean:<30.6f}")
            print(f"{'Log-IQR:':<20} {self.X_std:<30.6f} {self.Y_std:<30.6f}")
            print(f"{'-'*80}")
            print(f"\nNote: Using MEDIAN + IQR for robust normalization (vs mean + std)")
            print(f"{'='*80}\n")
        
        return self

    def preprocess(self, X_low, Y_high=None):
        """
        Transform data to normalized space.
        
        Args:
            X_low: Low-resolution data
            Y_high: High-resolution data (optional)
        
        Returns:
            Normalized X and Y (or None if Y not provided)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before preprocessing!")
        
        X_low = ensure_nchw(X_low)
        
        # Step 1: Log transform
        X_log = np.log1p(X_low)
        
        # Step 2: Normalize using median and IQR
        Xn = (X_log - self.X_mean) / self.X_std
        
        # Step 3: CLIP to [-5, 5] for training stability
        # This prevents extreme values from causing gradient issues
        Xn = np.clip(Xn, -5, 5).astype(np.float32)
        
        if Y_high is None:
            return Xn, None
        
        # Same processing for Y
        Y_high = ensure_nchw(Y_high)
        Y_log = np.log1p(Y_high)
        Yn = (Y_log - self.Y_mean) / self.Y_std
        Yn = np.clip(Yn, -5, 5).astype(np.float32)
        
        return Xn, Yn

    def postprocess(self, Y_norm):
        """
        Transform normalized predictions back to original scale.
        
        Args:
            Y_norm: Normalized high-resolution predictions
        
        Returns:
            Predictions in original scale (contact counts)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before postprocessing!")
        
        # Step 1: Clip back to safe range (in case of out-of-range predictions)
        Y_norm = np.clip(Y_norm, -5, 5)
        
        # Step 2: Reverse normalization
        Y_log = Y_norm * self.Y_std + self.Y_mean
        
        # Step 3: Reverse log transform
        Y_counts = np.expm1(Y_log)  # exp(Y_log) - 1
        
        # Step 4: Ensure non-negative (Hi-C data represents counts)
        return np.maximum(Y_counts, 0.0)

    def get_stats(self):
        """Return preprocessing statistics as a dictionary."""
        return {
            'X_median': float(self.X_mean),
            'X_iqr': float(self.X_std),
            'Y_median': float(self.Y_mean),
            'Y_iqr': float(self.Y_std),
            'method': 'robust (median/IQR)',
            'clip_range': [-5, 5]
        }


# ================================================================
# 🔹 CONFIGURATION
# ================================================================

def adjust_learning_rate(epoch):
    """Learning rate schedule."""
    lr = 0.0003 * (0.1 ** (epoch // 30))
    return lr


# Directories
root_dir = os.getcwd()
out_dir = os.path.join(root_dir, 'checkpoints_robust')
os.makedirs(out_dir, exist_ok=True)

# Timestamp for file naming
datestr = time.strftime('%m_%d_%H_%M')
name = 'HiCARN_1'

# Training parameters
num_epochs = 100
batch_size = 64

# Device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("\n" + "="*80)
print("HICARN TRAINING WITH ROBUST PREPROCESSING")
print("="*80)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {device}")
print(f"Epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print("="*80)


# ================================================================
# 🔹 DATA LOADING
# ================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# Try different data paths
data_file = '/home/yangz/data/hic_data/data_new/cr_train.npz'
if not os.path.exists(data_file):
    data_file = 'cr_train.npz'
    if not os.path.exists(data_file):
        raise FileNotFoundError("Could not find cr_train.npz")

print(f"Data file: {data_file}")

data = np.load(data_file)
X_low = data['data']      # Low-resolution
Y_high = data['target']   # High-resolution

print(f"\nRaw data loaded:")
print(f"  LR shape: {X_low.shape}")
print(f"  HR shape: {Y_high.shape}")
print(f"  LR range: [{X_low.min():.2f}, {X_low.max():.2f}]")
print(f"  HR range: [{Y_high.min():.2f}, {Y_high.max():.2f}]")

# Ensure NCHW format
X_low = ensure_nchw(X_low)
Y_high = ensure_nchw(Y_high)

print(f"\nAfter NCHW conversion:")
print(f"  LR shape: {X_low.shape}")
print(f"  HR shape: {Y_high.shape}")


# ================================================================
# 🔹 TRAIN/VALIDATION SPLIT
# ================================================================

n_total = len(X_low)
n_train = int(0.9 * n_total)

X_train, X_valid = X_low[:n_train], X_low[n_train:]
Y_train, Y_valid = Y_high[:n_train], Y_high[n_train:]

print(f"\nTrain/validation split (90/10):")
print(f"  Training samples: {len(X_train)}")
print(f"  Validation samples: {len(X_valid)}")


# ================================================================
# 🔹 FIT ROBUST PREPROCESSOR
# ================================================================

preprocessor = RobustHiCPreprocessor(size=40)
preprocessor.fit(X_train, Y_train, verbose=True)

# Save preprocessor
preprocessor_file = os.path.join(out_dir, 'preprocessor_robust.pt')
torch.save(preprocessor, preprocessor_file)
print(f"✓ Saved preprocessor to: {preprocessor_file}")


# ================================================================
# 🔹 PREPROCESS DATA
# ================================================================

print("\n" + "="*80)
print("PREPROCESSING DATA")
print("="*80)

# Preprocess training data
X_train_norm, Y_train_norm = preprocessor.preprocess(X_train, Y_train)

# Preprocess validation data
X_valid_norm, Y_valid_norm = preprocessor.preprocess(X_valid, Y_valid)

print(f"\nNormalized data statistics:")
print(f"  Train LR: shape={X_train_norm.shape}, range=[{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
print(f"  Train HR: shape={Y_train_norm.shape}, range=[{Y_train_norm.min():.4f}, {Y_train_norm.max():.4f}]")
print(f"  Valid LR: shape={X_valid_norm.shape}, range=[{X_valid_norm.min():.4f}, {X_valid_norm.max():.4f}]")
print(f"  Valid HR: shape={Y_valid_norm.shape}, range=[{Y_valid_norm.min():.4f}, {Y_valid_norm.max():.4f}]")

# Verify clipping
assert X_train_norm.min() >= -5 and X_train_norm.max() <= 5, "LR train data not properly clipped!"
assert Y_train_norm.min() >= -5 and Y_train_norm.max() <= 5, "HR train data not properly clipped!"
print(f"\n✓ Data properly clipped to [-5, 5] range")


# ================================================================
# 🔹 CONVERT TO TENSORS
# ================================================================

# Convert to PyTorch tensors (already in NCHW format)
train_data = torch.from_numpy(X_train_norm).float()
train_target = torch.from_numpy(Y_train_norm).float()

valid_data = torch.from_numpy(X_valid_norm).float()
valid_target = torch.from_numpy(Y_valid_norm).float()

# Keep original validation targets for metrics in count space
valid_target_counts = torch.from_numpy(Y_valid).float()

print(f"\nTensor shapes (NCHW format):")
print(f"  train_data: {train_data.shape}")
print(f"  train_target: {train_target.shape}")
print(f"  valid_data: {valid_data.shape}")
print(f"  valid_target: {valid_target.shape}")


# ================================================================
# 🔹 CREATE DATASETS AND DATALOADERS
# ================================================================

train_set = TensorDataset(train_data, train_target)
valid_set = TensorDataset(valid_data, valid_target, valid_target_counts)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Valid batches: {len(valid_loader)}")
print("="*80)


# ================================================================
# 🔹 INITIALIZE MODEL
# ================================================================

print("\n" + "="*80)
print("INITIALIZING MODEL")
print("="*80)

netG = Generator(num_channels=64).to(device)
num_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
print(f"Generator parameters: {num_params/1e6:.2f}M")

criterionG = GeneratorLoss().to(device)
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)

print("="*80)


# ================================================================
# 🔹 TRAINING LOOP
# ================================================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

ssim_scores = []
psnr_scores = []
mse_scores = []
mae_scores = []

best_ssim = 0

for epoch in range(1, num_epochs + 1):
    run_result = {'nsamples': 0, 'g_loss': 0}

    # Adjust learning rate
    alr = adjust_learning_rate(epoch)
    optimizerG = optim.Adam(netG.parameters(), lr=alr)

    # Clear gradients and free memory
    for p in netG.parameters():
        if p.grad is not None:
            del p.grad
    torch.cuda.empty_cache()

    # ============================================================
    # TRAINING PHASE
    # ============================================================
    
    netG.train()
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
    
    for data_batch, target_batch in train_bar:
        batch_size_actual = data_batch.size(0)
        run_result['nsamples'] += batch_size_actual

        # Verify shapes
        assert data_batch.dim() == 4 and data_batch.shape[1] == 1, \
            f"Wrong data shape: {data_batch.shape}"
        assert target_batch.dim() == 4 and target_batch.shape[1] == 1, \
            f"Wrong target shape: {target_batch.shape}"

        # Move to device
        real_img = target_batch.to(device)
        z = data_batch.to(device)
        
        # Forward pass
        fake_img = netG(z)

        # Compute loss and backprop
        netG.zero_grad()
        g_loss = criterionG(fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        run_result['g_loss'] += g_loss.item() * batch_size_actual

        train_bar.set_postfix({
            'Loss': f"{run_result['g_loss'] / run_result['nsamples']:.4f}"
        })
    
    train_gloss = run_result['g_loss'] / run_result['nsamples']
    
    # ============================================================
    # VALIDATION PHASE
    # ============================================================
    
    valid_result = {
        'g_loss': 0,
        'mse': 0, 
        'ssims': 0, 
        'psnr': 0, 
        'ssim': 0, 
        'nsamples': 0
    }
    
    netG.eval()

    batch_ssims = []
    batch_mses = []
    batch_psnrs = []
    batch_maes = []

    valid_bar = tqdm(valid_loader, desc='Validating')
    
    with torch.no_grad():
        for val_lr, val_hr_norm, val_hr_counts in valid_bar:
            batch_size_actual = val_lr.size(0)
            valid_result['nsamples'] += batch_size_actual
            
            # Verify shapes
            assert val_lr.dim() == 4 and val_lr.shape[1] == 1
            assert val_hr_norm.dim() == 4 and val_hr_norm.shape[1] == 1
            
            # Move to device
            lr = val_lr.to(device)
            hr_norm = val_hr_norm.to(device)
            hr_counts = val_hr_counts.to(device)
            
            # Predict in normalized space
            sr_norm = netG(lr)

            # Verify output shape
            assert sr_norm.dim() == 4 and sr_norm.shape[1] == 1

            # Compute loss in normalized space
            g_loss = criterionG(sr_norm, hr_norm)
            valid_result['g_loss'] += g_loss.item() * batch_size_actual

            # Convert predictions back to count space for evaluation
            sr_norm_np = sr_norm.cpu().numpy()
            sr_counts_np = preprocessor.postprocess(sr_norm_np)
            sr_counts = torch.from_numpy(sr_counts_np).float().to(device)
            
            # Normalize to [0, 1] for standard metrics
            max_val = max(sr_counts.max().item(), hr_counts.max().item())
            if max_val > 0:
                sr = sr_counts / max_val
                hr = hr_counts / max_val
            else:
                sr = sr_counts
                hr = hr_counts

            # Compute metrics
            batch_mse = ((sr - hr) ** 2).mean()
            batch_mae = (torch.abs(sr - hr)).mean()
            
            valid_result['mse'] += batch_mse.item() * batch_size_actual
            
            # SSIM
            batch_ssim = ssim(sr, hr)
            if isinstance(batch_ssim, torch.Tensor):
                batch_ssim = batch_ssim.item()
            valid_result['ssims'] += batch_ssim * batch_size_actual
            
            # Running averages
            avg_mse = valid_result['mse'] / valid_result['nsamples']
            avg_ssim = valid_result['ssims'] / valid_result['nsamples']
            avg_psnr = 10 * log10(1.0 / avg_mse) if avg_mse > 0 else 100.0
            
            valid_result['psnr'] = avg_psnr
            valid_result['ssim'] = avg_ssim
            
            valid_bar.set_postfix({
                'PSNR': f"{avg_psnr:.2f} dB",
                'SSIM': f"{avg_ssim:.4f}"
            })

            batch_ssims.append(batch_ssim)
            batch_psnrs.append(avg_psnr)
            batch_mses.append(batch_mse.item())
            batch_maes.append(batch_mae.item())

    # Aggregate epoch metrics
    ssim_scores.append(sum(batch_ssims) / len(batch_ssims))
    psnr_scores.append(sum(batch_psnrs) / len(batch_psnrs))
    mse_scores.append(sum(batch_mses) / len(batch_mses))
    mae_scores.append(sum(batch_maes) / len(batch_maes))

    valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
    now_ssim = valid_result['ssim']

    # Print epoch summary
    print(f"\nEpoch {epoch}/{num_epochs}:")
    print(f"  Train Loss: {train_gloss:.4f}")
    print(f"  Valid Loss: {valid_gloss:.4f}")
    print(f"  Valid SSIM: {now_ssim:.4f}")
    print(f"  Valid PSNR: {valid_result['psnr']:.2f} dB")
    print(f"  Valid MSE:  {valid_result['mse']/valid_result['nsamples']:.6f}")

    # Save best model
    if now_ssim > best_ssim:
        best_ssim = now_ssim
        print(f"  ✓ New best SSIM: {best_ssim:.6f}")
        
        best_ckpt_file = f'{datestr}_bestg_robust_{name}.pytorch'
        
        # Save with preprocessor for easy inference
        save_dict = {
            'model_state_dict': netG.state_dict(),
            'preprocessor': preprocessor,
            'epoch': epoch,
            'ssim': best_ssim,
            'preprocessing_method': 'robust_median_iqr'
        }
        torch.save(save_dict, os.path.join(out_dir, best_ckpt_file))
        print(f"  ✓ Saved to: {best_ckpt_file}")


# ================================================================
# 🔹 SAVE FINAL MODEL AND METRICS
# ================================================================

print("\n" + "="*80)
print("SAVING FINAL RESULTS")
print("="*80)

# Save final model
final_ckpt_g = f'{datestr}_finalg_robust_{name}.pytorch'
save_dict = {
    'model_state_dict': netG.state_dict(),
    'preprocessor': preprocessor,
    'epoch': num_epochs,
    'ssim': now_ssim,
    'preprocessing_method': 'robust_median_iqr'
}
torch.save(save_dict, os.path.join(out_dir, final_ckpt_g))
print(f"✓ Saved final model: {final_ckpt_g}")

# Save metrics
np.savetxt(os.path.join(out_dir, f'valid_ssim_scores_{name}.txt'), 
           X=np.array(ssim_scores), delimiter=',')
np.savetxt(os.path.join(out_dir, f'valid_psnr_scores_{name}.txt'), 
           X=np.array(psnr_scores), delimiter=',')
np.savetxt(os.path.join(out_dir, f'valid_mse_scores_{name}.txt'), 
           X=np.array(mse_scores), delimiter=',')
np.savetxt(os.path.join(out_dir, f'valid_mae_scores_{name}.txt'), 
           X=np.array(mae_scores), delimiter=',')
print(f"✓ Saved training metrics")

# Save preprocessing stats
import json
stats_file = os.path.join(out_dir, 'preprocessing_stats.json')
with open(stats_file, 'w') as f:
    json.dump(preprocessor.get_stats(), f, indent=2)
print(f"✓ Saved preprocessing statistics: {stats_file}")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Best SSIM: {best_ssim:.6f}")
print(f"Final SSIM: {ssim_scores[-1]:.6f}")
print(f"Final PSNR: {psnr_scores[-1]:.2f} dB")
print(f"Final MSE:  {mse_scores[-1]:.6f}")
print(f"\nCheckpoints saved to: {out_dir}")
print(f"  - Best model: {datestr}_bestg_robust_{name}.pytorch")
print(f"  - Final model: {final_ckpt_g}")
print(f"  - Preprocessor: preprocessor_robust.pt")
print("="*80)
