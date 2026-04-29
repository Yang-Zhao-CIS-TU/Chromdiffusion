#!/usr/bin/env python3
"""
HiCARN Training with ROBUST Preprocessing - MEMORY OPTIMIZED
=============================================================
Uses median/IQR normalization and clipping instead of mean/std.
Optimized for low GPU memory usage.
"""

import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from math import log10

# Additional imports for metrics
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

# Import HiCARN modules
try:
    from Models.HiCARN_1 import Generator
    from Models.HiCARN_1_Loss import GeneratorLoss
    from Utils.SSIM import ssim as ssim_torch
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
    def ssim_torch(img1, img2):
        return torch.tensor(0.8)


# ================================================================
# 🔹 CUSTOM METRICS CALCULATION
# ================================================================

def calculate_psnr(img1, img2):
    """Calculate PSNR using max value of ground truth"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = np.max(img2)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM using skimage with data range"""
    if img1.ndim > 2:
        img1 = img1.squeeze()
    if img2.ndim > 2:
        img2 = img2.squeeze()
    return structural_similarity(img1, img2, data_range=img2.max() - img2.min())

def calculate_pcc(img1, img2):
    """Calculate Pearson Correlation Coefficient"""
    return pearsonr(img1.flatten(), img2.flatten())[0]

def calculate_mse(img1, img2):
    """Calculate Mean Squared Error"""
    return np.mean((img1 - img2) ** 2)

def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(img1 - img2))

def calculate_snr(img1, img2):
    """Calculate Signal-to-Noise Ratio"""
    signal_power = np.mean(img2 ** 2)
    noise_power = np.mean((img1 - img2) ** 2)
    if noise_power == 0:
        return 100
    return 10 * np.log10(signal_power / noise_power)

class Metrics:
    """Class to accumulate and summarize metrics across samples"""
    def __init__(self):
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'pcc': [],
            'mse': [],
            'mae': [],
            'snr': [],
        }
    
    def compute_sample(self, pred_sample, hr_sample):
        """Compute metrics for a single sample"""
        pred = pred_sample.squeeze()
        hr = hr_sample.squeeze()
        
        self.metrics['psnr'].append(calculate_psnr(pred, hr))
        self.metrics['ssim'].append(calculate_ssim(pred, hr))
        self.metrics['pcc'].append(calculate_pcc(pred, hr))
        self.metrics['mse'].append(calculate_mse(pred, hr))
        self.metrics['mae'].append(calculate_mae(pred, hr))
        self.metrics['snr'].append(calculate_snr(pred, hr))
    
    def get_summary(self):
        """Get summary statistics for all metrics"""
        summary = {}
        for metric_name in self.metrics.keys():
            if len(self.metrics[metric_name]) > 0:
                summary[metric_name] = {
                    'mean': float(np.mean(self.metrics[metric_name])),
                    'std': float(np.std(self.metrics[metric_name])),
                    'median': float(np.median(self.metrics[metric_name])),
                    'min': float(np.min(self.metrics[metric_name])),
                    'max': float(np.max(self.metrics[metric_name])),
                }
            else:
                summary[metric_name] = {
                    'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0
                }
        return summary
    
    def reset(self):
        """Reset all metrics"""
        for key in self.metrics.keys():
            self.metrics[key] = []


# ================================================================
# 🔹 ROBUST PREPROCESSING (Median/IQR + Clipping)
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
    """Robust Hi-C preprocessing using median and IQR."""
    
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self._is_fitted = False

    def fit(self, X_low, Y_high, verbose=True):
        X_low = ensure_nchw(X_low)
        Y_high = ensure_nchw(Y_high)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"FITTING ROBUST PREPROCESSOR")
            print(f"{'='*80}")
            print(f"Training samples: {len(X_low)}")
            print(f"  X shape: {X_low.shape}, range: [{X_low.min():.2f}, {X_low.max():.2f}]")
            print(f"  Y shape: {Y_high.shape}, range: [{Y_high.min():.2f}, {Y_high.max():.2f}]")
        
        X_log = np.log1p(X_low)
        Y_log = np.log1p(Y_high)
        
        self.X_mean = np.median(X_log)
        self.X_std = (np.percentile(X_log, 75) - np.percentile(X_log, 25)) + 1e-8
        
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
            print(f"\nNote: Using MEDIAN + IQR for robust normalization")
            print(f"{'='*80}\n")
        
        return self

    def preprocess(self, X_low, Y_high=None):
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted first!")
        
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
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted first!")
        
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
            'method': 'robust (median/IQR)',
            'clip_range': [-5, 5]
        }


# ================================================================
# 🔹 CONFIGURATION
# ================================================================

def adjust_learning_rate(epoch):
    lr = 0.0003 * (0.1 ** (epoch // 30))
    return lr


def parse_args():
    parser = argparse.ArgumentParser(description='HiCARN Training with Robust Preprocessing')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size (reduce if OOM, default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--gpu', type=int, default=1,
                       help='GPU ID to use (default: 1)')
    parser.add_argument('--data_file', type=str, 
                       default='/home/yangz/data/hic_data/data_new/cr_train.npz',
                       help='Path to training data .npz file')
    parser.add_argument('--output_dir', type=str, default='checkpoints_robust',
                       help='Directory to save checkpoints (default: checkpoints_robust)')
    return parser.parse_args()


# ================================================================
# 🔹 MAIN
# ================================================================

if __name__ == '__main__':
    args = parse_args()
    
    # Directories
    root_dir = os.getcwd()
    out_dir = os.path.join(root_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    datestr = time.strftime('%m_%d_%H_%M')
    name = 'HiCARN_1'
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("HICARN TRAINING WITH ROBUST PREPROCESSING (MEMORY OPTIMIZED)")
    print("="*80)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"\nMetrics Computation:")
    print(f"  - Training Loss: Normalized space")
    print(f"  - Validation Loss: Normalized space")
    print(f"  - Validation Metrics (SSIM/PSNR/MSE/MAE): RAW space (contact counts)")
    print("="*80)
    
    # ================================================================
    # LOAD DATA
    # ================================================================
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    if not os.path.exists(args.data_file):
        alt_path = 'cr_train.npz'
        if os.path.exists(alt_path):
            args.data_file = alt_path
        else:
            raise FileNotFoundError(f"Could not find data file: {args.data_file}")
    
    print(f"Data file: {args.data_file}")
    
    data = np.load(args.data_file)
    X_low = data['data']
    Y_high = data['target']
    
    print(f"\nRaw data loaded:")
    print(f"  LR shape: {X_low.shape}")
    print(f"  HR shape: {Y_high.shape}")
    print(f"  LR range: [{X_low.min():.2f}, {X_low.max():.2f}]")
    print(f"  HR range: [{Y_high.min():.2f}, {Y_high.max():.2f}]")
    
    X_low = ensure_nchw(X_low)
    Y_high = ensure_nchw(Y_high)
    
    print(f"\nAfter NCHW conversion:")
    print(f"  LR shape: {X_low.shape}")
    print(f"  HR shape: {Y_high.shape}")
    
    # Train/val split
    n_total = len(X_low)
    n_train = int(0.9 * n_total)
    
    X_train, X_valid = X_low[:n_train], X_low[n_train:]
    Y_train, Y_valid = Y_high[:n_train], Y_high[n_train:]
    
    print(f"\nTrain/validation split (90/10):")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_valid)}")
    
    # ================================================================
    # FIT PREPROCESSOR
    # ================================================================
    
    preprocessor = RobustHiCPreprocessor(size=40)
    preprocessor.fit(X_train, Y_train, verbose=True)
    
    preprocessor_file = os.path.join(out_dir, 'preprocessor_robust.pt')
    torch.save(preprocessor, preprocessor_file)
    print(f"✓ Saved preprocessor to: {preprocessor_file}")
    
    # ================================================================
    # PREPROCESS DATA
    # ================================================================
    
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)
    
    X_train_norm, Y_train_norm = preprocessor.preprocess(X_train, Y_train)
    X_valid_norm, Y_valid_norm = preprocessor.preprocess(X_valid, Y_valid)
    
    print(f"\nNormalized data statistics:")
    print(f"  Train LR: shape={X_train_norm.shape}, range=[{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
    print(f"  Train HR: shape={Y_train_norm.shape}, range=[{Y_train_norm.min():.4f}, {Y_train_norm.max():.4f}]")
    
    assert X_train_norm.min() >= -5 and X_train_norm.max() <= 5
    assert Y_train_norm.min() >= -5 and Y_train_norm.max() <= 5
    print(f"\n✓ Data properly clipped to [-5, 5] range")
    
    # Convert to tensors
    train_data = torch.from_numpy(X_train_norm).float()
    train_target = torch.from_numpy(Y_train_norm).float()
    valid_data = torch.from_numpy(X_valid_norm).float()
    valid_target = torch.from_numpy(Y_valid_norm).float()
    valid_target_counts = torch.from_numpy(Y_valid).float()
    
    print(f"\nTensor shapes:")
    print(f"  train_data: {train_data.shape}")
    print(f"  train_target: {train_target.shape}")
    
    # Create datasets
    train_set = TensorDataset(train_data, train_target)
    valid_set = TensorDataset(valid_data, valid_target, valid_target_counts)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Valid batches: {len(valid_loader)}")
    print("="*80)
    
    # ================================================================
    # INITIALIZE MODEL
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
    # TRAINING LOOP
    # ================================================================
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    mae_scores = []
    pcc_scores = []
    snr_scores = []
    
    best_ssim = 0
    
    for epoch in range(1, args.epochs + 1):
        run_result = {'nsamples': 0, 'g_loss': 0}
        
        # Adjust learning rate (update existing optimizer instead of recreating)
        alr = adjust_learning_rate(epoch)
        for param_group in optimizerG.param_groups:
            param_group['lr'] = alr
        
        # Clear memory at start of epoch
        netG.zero_grad()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                # If cache clearing fails, continue anyway
                pass
        
        # ============================================================
        # TRAINING PHASE
        # ============================================================
        
        netG.train()
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        
        for data_batch, target_batch in train_bar:
            batch_size_actual = data_batch.size(0)
            run_result['nsamples'] += batch_size_actual
            
            real_img = target_batch.to(device)
            z = data_batch.to(device)
            fake_img = netG(z)
            
            netG.zero_grad()
            g_loss = criterionG(fake_img, real_img)
            g_loss.backward()
            optimizerG.step()
            
            run_result['g_loss'] += g_loss.item() * batch_size_actual
            
            train_bar.set_postfix({'Loss': f"{run_result['g_loss'] / run_result['nsamples']:.4f}"})
            
            # Clean up
            del real_img, z, fake_img, g_loss
        
        train_gloss = run_result['g_loss'] / run_result['nsamples']
        
        # ============================================================
        # VALIDATION PHASE (WITH CUSTOM METRICS)
        # ============================================================
        
        valid_result = {'g_loss': 0, 'nsamples': 0}
        netG.eval()
        
        # Initialize metrics tracker
        metrics_tracker = Metrics()
        
        valid_bar = tqdm(valid_loader, desc='Validating')
        
        with torch.no_grad():
            for val_lr, val_hr_norm, val_hr_counts in valid_bar:
                batch_size_actual = val_lr.size(0)
                valid_result['nsamples'] += batch_size_actual
                
                lr = val_lr.to(device)
                hr_norm = val_hr_norm.to(device)
                hr_counts = val_hr_counts.to(device)
                
                # Predict in normalized space
                sr_norm = netG(lr)
                
                # Loss in normalized space
                g_loss = criterionG(sr_norm, hr_norm)
                valid_result['g_loss'] += g_loss.item() * batch_size_actual
                
                # ============================================================
                # CONVERT TO RAW SPACE AND COMPUTE METRICS PER SAMPLE
                # ============================================================
                
                # Postprocess SR predictions to original contact count space
                sr_norm_np = sr_norm.detach().cpu().numpy()
                sr_counts_np = preprocessor.postprocess(sr_norm_np)
                hr_counts_np = hr_counts.cpu().numpy()
                
                # Compute metrics for each sample in the batch
                for i in range(batch_size_actual):
                    pred_sample = sr_counts_np[i]  # Shape: (1, H, W)
                    hr_sample = hr_counts_np[i]    # Shape: (1, H, W)
                    metrics_tracker.compute_sample(pred_sample, hr_sample)
                
                # Get running averages for progress bar
                current_summary = metrics_tracker.get_summary()
                
                valid_bar.set_postfix({
                    'SSIM': f"{current_summary['ssim']['mean']:.4f}",
                    'PCC': f"{current_summary['pcc']['mean']:.4f}",
                    'MAE': f"{current_summary['mae']['mean']:.2f}"
                })
                
                # CRITICAL: Free GPU memory after each batch
                del sr_norm, lr, hr_norm, hr_counts, g_loss
                del sr_norm_np, sr_counts_np, hr_counts_np
        
        # Get final summary statistics
        metrics_summary = metrics_tracker.get_summary()
        
        valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
        
        # Clean up
        del valid_result
        metrics_tracker.reset()
        del metrics_tracker
        
        # Clear GPU cache
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass
        
        # Print summary with statistics
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs} - Validation Results")
        print(f"{'='*80}")
        print(f"Train Loss (normalized): {train_gloss:.4f}")
        print(f"Valid Loss (normalized): {valid_gloss:.4f}")
        print(f"\n{'Metric':<8} {'Mean':<12} {'Std':<12} {'Median':<12} {'Min':<12} {'Max':<12}")
        print(f"{'-'*80}")
        
        for metric_name in ['psnr', 'ssim', 'pcc', 'mse', 'mae', 'snr']:
            stats = metrics_summary[metric_name]
            if metric_name in ['psnr', 'snr']:
                # dB metrics
                print(f"{metric_name.upper():<8} {stats['mean']:<12.2f} {stats['std']:<12.2f} "
                      f"{stats['median']:<12.2f} {stats['min']:<12.2f} {stats['max']:<12.2f}")
            elif metric_name in ['ssim', 'pcc']:
                # Correlation metrics (0-1)
                print(f"{metric_name.upper():<8} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
                      f"{stats['median']:<12.4f} {stats['min']:<12.4f} {stats['max']:<12.4f}")
            else:
                # Error metrics (MSE, MAE)
                print(f"{metric_name.upper():<8} {stats['mean']:<12.2f} {stats['std']:<12.2f} "
                      f"{stats['median']:<12.2f} {stats['min']:<12.2f} {stats['max']:<12.2f}")
        
        print(f"{'='*80}\n")
        
        # Use SSIM mean for best model selection
        now_ssim = metrics_summary['ssim']['mean']
        
        # Store metrics for history (use mean values)
        ssim_scores.append(metrics_summary['ssim']['mean'])
        psnr_scores.append(metrics_summary['psnr']['mean'])
        mse_scores.append(metrics_summary['mse']['mean'])
        mae_scores.append(metrics_summary['mae']['mean'])
        pcc_scores.append(metrics_summary['pcc']['mean'])
        snr_scores.append(metrics_summary['snr']['mean'])
        
        # Save best model
        if now_ssim > best_ssim:
            best_ssim = now_ssim
            print(f"  ✓ New best SSIM: {best_ssim:.6f}")
            
            best_ckpt_file = f'{datestr}_bestg_robust_{name}.pytorch'
            save_dict = {
                'model_state_dict': netG.state_dict(),
                'preprocessor': preprocessor,
                'epoch': epoch,
                'ssim': best_ssim,
                'preprocessing_method': 'robust_median_iqr',
                'batch_size': args.batch_size
            }
            torch.save(save_dict, os.path.join(out_dir, best_ckpt_file))
            print(f"  ✓ Saved to: {best_ckpt_file}")
            
            # Clean up after saving
            del save_dict
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
    
    # ================================================================
    # SAVE FINAL MODEL
    # ================================================================
    
    print("\n" + "="*80)
    print("SAVING FINAL RESULTS")
    print("="*80)
    
    final_ckpt_g = f'{datestr}_finalg_robust_{name}.pytorch'
    save_dict = {
        'model_state_dict': netG.state_dict(),
        'preprocessor': preprocessor,
        'epoch': args.epochs,
        'ssim': now_ssim,
        'preprocessing_method': 'robust_median_iqr',
        'batch_size': args.batch_size
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
    np.savetxt(os.path.join(out_dir, f'valid_pcc_scores_{name}.txt'), 
               X=np.array(pcc_scores), delimiter=',')
    np.savetxt(os.path.join(out_dir, f'valid_snr_scores_{name}.txt'), 
               X=np.array(snr_scores), delimiter=',')
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
    print(f"\nFinal Metrics (Mean across all validation samples):")
    print(f"  SSIM: {ssim_scores[-1]:.4f}")
    print(f"  PSNR: {psnr_scores[-1]:.2f} dB")
    print(f"  PCC:  {pcc_scores[-1]:.4f}")
    print(f"  SNR:  {snr_scores[-1]:.2f} dB")
    print(f"  MSE:  {mse_scores[-1]:.2f}")
    print(f"  MAE:  {mae_scores[-1]:.2f}")
    print(f"\nNote: All validation metrics computed in RAW space (contact counts)")
    print(f"      Loss computed in NORMALIZED space (for training)")
    print(f"      Metrics include per-sample statistics (mean/std/median/min/max)")
    print(f"\nCheckpoints saved to: {out_dir}")
    print("="*80)
