#!/usr/bin/env python3
"""
HiCARN training with EXACT SAME preprocessing as hic_imputer_with_validation.py
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


# ========== EXACT SAME PREPROCESSING AS HIC_IMPUTER ==========

def ensure_nchw(arr):
    """Ensure array is (N, 1, H, W). Accepts (N,1,H,W) or (N,H,W,1)."""
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape={arr.shape}")
    if arr.shape[1] == 1 and arr.shape[-1] != 1:
        return arr
    if arr.shape[-1] == 1 and arr.shape[1] != 1:
        return np.transpose(arr, (0, 3, 1, 2))
    if arr.shape[1] == 1 and arr.shape[-1] == 1:
        return arr
    raise ValueError(f"Cannot infer channel axis for shape {arr.shape}")


class HiCPreprocessor:
    """EXACT SAME preprocessing as in hic_imputer_with_validation.py"""
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self.band_means = None
        self.band_nonzero_rates = None

    def fit(self, X_low, Y_high):
        """Fit statistics."""
        print("Fitting preprocessor...")
        X_low = ensure_nchw(X_low)
        Y_high = ensure_nchw(Y_high)

        X_log = np.log1p(X_low)
        Y_log = np.log1p(Y_high)

        self.X_mean = X_log.mean()
        self.X_std = X_log.std() + 1e-8
        self.Y_mean = Y_log.mean()
        self.Y_std = Y_log.std() + 1e-8

        print(f"X_log: mean={self.X_mean:.4f}, std={self.X_std:.4f}")
        print(f"Y_log: mean={self.Y_mean:.4f}, std={self.Y_std:.4f}")

        # Band statistics
        self.band_means = np.zeros(self.size, dtype=np.float32)
        self.band_nonzero_rates = np.zeros(self.size, dtype=np.float32)

        H = W = self.size
        I, J = np.indices((H, W))
        Y2 = Y_log[:, 0, :, :]
        Y2f = Y2.reshape(Y2.shape[0], -1)

        for b in range(self.size):
            mask = (np.abs(I - J) == b).ravel()
            values = Y2f[:, mask]
            self.band_means[b] = values.mean()
            self.band_nonzero_rates[b] = (values > 1e-3).mean().astype(np.float32)

    def preprocess(self, X_low, Y_high=None):
        """Preprocess data."""
        X_low = ensure_nchw(X_low)
        X_log = np.log1p(X_low)
        X_norm = (X_log - self.X_mean) / self.X_std
        X_norm = (X_norm + np.transpose(X_norm, (0, 1, 3, 2))) / 2

        if Y_high is not None:
            Y_high = ensure_nchw(Y_high)
            Y_log = np.log1p(Y_high)
            Y_norm = (Y_log - self.Y_mean) / self.Y_std
            Y_norm = (Y_norm + np.transpose(Y_norm, (0, 1, 3, 2))) / 2
            return X_norm, Y_norm

        return X_norm, None

    def postprocess(self, Y_norm):
        """Postprocess predictions."""
        Y_log = Y_norm * self.Y_std + self.Y_mean
        Y_log = (Y_log + np.transpose(Y_log, (0, 1, 3, 2))) / 2
        Y_counts = np.expm1(Y_log)
        return np.maximum(Y_counts, 0.0)


# ========== CONFIGURATION ==========

def adjust_learning_rate(epoch):
    lr = 0.0003 * (0.1 ** (epoch // 30))
    return lr


root_dir = os.getcwd()
out_dir = os.path.join(root_dir, 'checkpoints_log')
os.makedirs(out_dir, exist_ok=True)

datestr = time.strftime('%m_%d_%H_%M')
name = 'HiCARN_1'

num_epochs = 100
batch_size = 64

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("CUDA available?", torch.cuda.is_available())
print("Device being used:", device)


# ========== DATA LOADING ==========

print("\n" + "="*60)
print("Loading and preprocessing data...")
print("="*60)

data_file = '/home/yangz/data/hic_data/data_new/cr_train.npz'
if not os.path.exists(data_file):
    data_file = 'cr_train.npz'
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Could not find cr_train.npz")

print(f"Loading data from: {data_file}")
data = np.load(data_file)

X_low = data['data']
Y_high = data['target']

print(f"Original data shape: {X_low.shape}")
print(f"Original target shape: {Y_high.shape}")

# Ensure NCHW format
X_low = ensure_nchw(X_low)
Y_high = ensure_nchw(Y_high)

print(f"After ensure_nchw:")
print(f"  Data shape: {X_low.shape}")
print(f"  Target shape: {Y_high.shape}")
print(f"  Data range: [{X_low.min():.2f}, {X_low.max():.2f}]")
print(f"  Target range: [{Y_high.min():.2f}, {Y_high.max():.2f}]")

# Split train/valid
n_total = len(X_low)
n_train = int(0.9 * n_total)

X_train, X_valid = X_low[:n_train], X_low[n_train:]
Y_train, Y_valid = Y_high[:n_train], Y_high[n_train:]

print(f"\nTrain samples: {len(X_train)}")
print(f"Valid samples: {len(X_valid)}")

# Fit preprocessor
preprocessor = HiCPreprocessor(size=40)
preprocessor.fit(X_train, Y_train)

# Preprocess
print("\nPreprocessing training data...")
X_train_norm, Y_train_norm = preprocessor.preprocess(X_train, Y_train)

print("Preprocessing validation data...")
X_valid_norm, Y_valid_norm = preprocessor.preprocess(X_valid, Y_valid)

print(f"\nNormalized shapes:")
print(f"  X_train_norm: {X_train_norm.shape}")
print(f"  Y_train_norm: {Y_train_norm.shape}")
print(f"  X_valid_norm: {X_valid_norm.shape}")
print(f"  Y_valid_norm: {Y_valid_norm.shape}")

print(f"\nNormalized ranges:")
print(f"  Train data: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
print(f"  Train target: [{Y_train_norm.min():.4f}, {Y_train_norm.max():.4f}]")

# Save preprocessor
preprocessor_file = os.path.join(out_dir, 'preprocessor.pt')
torch.save(preprocessor, preprocessor_file)
print(f"\nSaved preprocessor to: {preprocessor_file}")

# Convert to torch tensors (already in NCHW format)
train_data = torch.from_numpy(X_train_norm).float()
train_target = torch.from_numpy(Y_train_norm).float()

valid_data = torch.from_numpy(X_valid_norm).float()
valid_target = torch.from_numpy(Y_valid_norm).float()

# Keep original for metrics
valid_target_counts = torch.from_numpy(Y_valid).float()

print(f"\nTensor shapes (should be [N, 1, H, W]):")
print(f"  train_data: {train_data.shape}")
print(f"  train_target: {train_target.shape}")

# Create datasets
train_set = TensorDataset(train_data, train_target)
valid_set = TensorDataset(valid_data, valid_target, valid_target_counts)

# DataLoaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Valid batches: {len(valid_loader)}")


# ========== MODEL ==========

print("\n" + "="*60)
print("Initializing model...")
print("="*60)

netG = Generator(num_channels=64).to(device)
num_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
print(f"Model parameters: {num_params/1e6:.2f}M")

criterionG = GeneratorLoss().to(device)
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)


# ========== TRAINING ==========

print("\n" + "="*60)
print("Starting training...")
print("="*60)

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

    # Free memory
    for p in netG.parameters():
        if p.grad is not None:
            del p.grad
    torch.cuda.empty_cache()

    # ========== TRAINING PHASE ==========
    netG.train()
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
    
    for data_batch, target_batch in train_bar:
        batch_size_actual = data_batch.size(0)
        run_result['nsamples'] += batch_size_actual

        # Check shapes before moving to device
        assert data_batch.dim() == 4 and data_batch.shape[1] == 1, f"Wrong data shape: {data_batch.shape}"
        assert target_batch.dim() == 4 and target_batch.shape[1] == 1, f"Wrong target shape: {target_batch.shape}"

        real_img = target_batch.to(device)
        z = data_batch.to(device)
        fake_img = netG(z)

        # Train generator
        netG.zero_grad()
        g_loss = criterionG(fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        run_result['g_loss'] += g_loss.item() * batch_size_actual

        train_bar.set_postfix({
            'Loss': f"{run_result['g_loss'] / run_result['nsamples']:.4f}"
        })

    train_gloss = run_result['g_loss'] / run_result['nsamples']
    
    # ========== VALIDATION PHASE ==========
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
            
            # Check shapes
            assert val_lr.dim() == 4 and val_lr.shape[1] == 1, f"Wrong val_lr shape: {val_lr.shape}"
            assert val_hr_norm.dim() == 4 and val_hr_norm.shape[1] == 1, f"Wrong val_hr_norm shape: {val_hr_norm.shape}"
            
            lr = val_lr.to(device)
            hr_norm = val_hr_norm.to(device)
            hr_counts = val_hr_counts.to(device)
            
            # Predict in normalized space
            sr_norm = netG(lr)

            # Check output shape
            assert sr_norm.dim() == 4 and sr_norm.shape[1] == 1, f"Wrong sr_norm shape: {sr_norm.shape}"

            # Compute loss in normalized space
            g_loss = criterionG(sr_norm, hr_norm)
            valid_result['g_loss'] += g_loss.item() * batch_size_actual

            # Convert predictions back to count space
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
            
            # SSIM - ensure correct shape (B, C, H, W)
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
        best_ckpt_file = f'{datestr}_bestg_{name}.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, best_ckpt_file))
        print(f"  ✓ Saved to: {best_ckpt_file}")

# Save final model
final_ckpt_g = f'{datestr}_finalg_{name}.pytorch'
torch.save(netG.state_dict(), os.path.join(out_dir, final_ckpt_g))
print(f"\n✓ Saved final model to: {final_ckpt_g}")

# Save metrics
print("\nSaving training metrics...")
np.savetxt(os.path.join(out_dir, f'valid_ssim_scores_{name}.txt'), X=np.array(ssim_scores), delimiter=',')
np.savetxt(os.path.join(out_dir, f'valid_psnr_scores_{name}.txt'), X=np.array(psnr_scores), delimiter=',')
np.savetxt(os.path.join(out_dir, f'valid_mse_scores_{name}.txt'), X=np.array(mse_scores), delimiter=',')
np.savetxt(os.path.join(out_dir, f'valid_mae_scores_{name}.txt'), X=np.array(mae_scores), delimiter=',')

print("\n" + "="*60)
print("Training complete!")
print("="*60)
print(f"Best SSIM: {best_ssim:.6f}")
print(f"Final SSIM: {ssim_scores[-1]:.6f}")
print(f"Final PSNR: {psnr_scores[-1]:.2f} dB")
print(f"Final MSE:  {mse_scores[-1]:.6f}")
print("="*60)
