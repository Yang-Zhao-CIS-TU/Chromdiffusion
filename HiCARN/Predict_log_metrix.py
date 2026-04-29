import os
import time
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from math import log10
import torch
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

# ================================================================
# 🔹 IMPORT MODELS AND UTILITIES (Same as training)
# ================================================================

from Models.HiCARN_1 import Generator
from Utils.SSIM import ssim

# ================================================================
# 🔹 EVALUATION METRICS (Exactly matching diffusion model)
# ================================================================

def calculate_psnr(img1, img2):
    """PSNR calculation - matching diffusion model"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = np.max(img2)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """SSIM calculation - matching diffusion model"""
    if img1.ndim > 2:
        img1 = img1.squeeze()
    if img2.ndim > 2:
        img2 = img2.squeeze()
    return structural_similarity(img1, img2, data_range=img2.max() - img2.min())

def calculate_pcc(img1, img2):
    """Pearson correlation - matching diffusion model"""
    return pearsonr(img1.flatten(), img2.flatten())[0]

def calculate_mse(img1, img2):
    """MSE calculation - matching diffusion model"""
    return np.mean((img1 - img2) ** 2)

def calculate_mae(img1, img2):
    """MAE calculation - matching diffusion model"""
    return np.mean(np.abs(img1 - img2))

def calculate_snr(img1, img2):
    """SNR calculation - matching diffusion model"""
    signal_power = np.mean(img2 ** 2)
    noise_power = np.mean((img1 - img2) ** 2)
    if noise_power == 0:
        return 100
    return 10 * np.log10(signal_power / noise_power)

class Metrics:
    """Metrics class - matching diffusion model"""
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
            summary[metric_name] = {
                'mean': float(np.mean(self.metrics[metric_name])),
                'std': float(np.std(self.metrics[metric_name])),
                'median': float(np.median(self.metrics[metric_name])),
                'min': float(np.min(self.metrics[metric_name])),
                'max': float(np.max(self.metrics[metric_name])),
            }
        return summary

# ================================================================
# 🔹 PREPROCESSING - MATCHING TRAINING SCRIPT
# ================================================================

def ensure_nchw(x):
    """Ensure data is in NCHW format"""
    x = np.asarray(x)
    if x.ndim == 3:
        return x[:, None, :, :]
    elif x.ndim == 4 and x.shape[1] in [1,3]:
        return x
    elif x.ndim == 4 and x.shape[-1] in [1,3]:
        return np.transpose(x, (0,3,1,2))
    raise ValueError(f"Cannot convert to NCHW: shape={x.shape}")

class HiCPreprocessor:
    """
    Preprocessing matching diffusion model:
    - Log1p transformation
    - Robust normalization (median + IQR)
    - Clipping to [-5, 5]
    """
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        
    def fit(self, X, Y):
        """Fit preprocessor on training data"""
        X, Y = ensure_nchw(X), ensure_nchw(Y)
        
        # Log transform
        X_log = np.log1p(X)
        Y_log = np.log1p(Y)
        
        # Robust statistics (median + IQR)
        self.X_mean = np.median(X_log)
        self.X_std = np.percentile(X_log, 75) - np.percentile(X_log, 25) + 1e-8
        
        self.Y_mean = np.median(Y_log)
        self.Y_std = np.percentile(Y_log, 75) - np.percentile(Y_log, 25) + 1e-8
        
        print(f"\n{'='*80}")
        print(f"PREPROCESSING STATISTICS (Matching Diffusion Model)")
        print(f"{'='*80}")
        print(f"LR (Low Resolution):")
        print(f"  Log-median: {self.X_mean:.4f}")
        print(f"  Log-IQR: {self.X_std:.4f}")
        print(f"\nHR (High Resolution):")
        print(f"  Log-median: {self.Y_mean:.4f}")
        print(f"  Log-IQR: {self.Y_std:.4f}")
        print(f"{'='*80}\n")
        
    def preprocess(self, X, Y=None):
        """
        Preprocess data:
        1. Log1p transform
        2. Standardize using median and IQR
        3. Clip to [-5, 5]
        """
        X = ensure_nchw(X)
        
        # Log transform
        X_log = np.log1p(X)
        
        # Standardize
        Xn = (X_log - self.X_mean) / self.X_std
        
        # Clip
        Xn = np.clip(Xn, -5, 5).astype(np.float32)
        
        if Y is None:
            return Xn, None
        
        Y = ensure_nchw(Y)
        
        # Same for Y
        Y_log = np.log1p(Y)
        Yn = (Y_log - self.Y_mean) / self.Y_std
        Yn = np.clip(Yn, -5, 5).astype(np.float32)
        
        return Xn, Yn
    
    def postprocess(self, Yn):
        """
        Inverse preprocessing:
        1. Unclip (handled by clipping again)
        2. Unstandardize
        3. Inverse log1p (expm1)
        """
        # Clip back to safe range
        Yn = np.clip(Yn, -5, 5)
        
        # Unstandardize
        Ylog = Yn * self.Y_std + self.Y_mean
        
        # Inverse log1p
        Y = np.expm1(Ylog)
        
        # Ensure non-negative
        return np.maximum(Y, 0)

# ================================================================
# 🔹 TESTING FUNCTION
# ================================================================

def test_hicarn_npy(lr_input_path, hr_gt_path, ckpt_file, device, batch_size=64):
    """
    Test HiCARN model on separate .npy files
    Computes metrics EXACTLY matching diffusion model testing
    """
    print(f"\n{'='*80}")
    print(f"HiCARN TESTING ON NPY FILES")
    print(f"{'='*80}\n")
    
    # ================================================================
    # LOAD DATA
    # ================================================================
    
    print(f"Loading data:")
    print(f"  LR: {lr_input_path}")
    print(f"  HR: {hr_gt_path}")
    
    lr_raw = np.load(lr_input_path)
    hr_raw = np.load(hr_gt_path)
    
    # Ensure NCHW format and float64 (matching diffusion)
    lr_raw = ensure_nchw(lr_raw).astype(np.float64)
    hr_raw = ensure_nchw(hr_raw).astype(np.float64)
    
    print(f"\nData loaded:")
    print(f"  LR shape: {lr_raw.shape}")
    print(f"  HR shape: {hr_raw.shape}")
    print(f"  LR range: [{lr_raw.min():.2f}, {lr_raw.max():.2f}]")
    print(f"  HR range: [{hr_raw.min():.2f}, {hr_raw.max():.2f}]")
    
    # ================================================================
    # LOAD MODEL AND PREPROCESSOR
    # ================================================================
    
    print(f"\n{'='*80}")
    print(f"LOADING MODEL AND PREPROCESSOR")
    print(f"{'='*80}")
    
    # Load checkpoint
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    
    checkpoint = torch.load(ckpt_file, map_location='cpu')
    print(f'✓ Loaded checkpoint from: {ckpt_file}')
    
    # Check checkpoint format (matching training script save format)
    if 'preprocessor' in checkpoint:
        pre = checkpoint['preprocessor']
        model_state = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        ssim_train = checkpoint.get('ssim', 'unknown')
        config = checkpoint.get('config', {})
        
        print(f'✓ Using saved preprocessor from checkpoint')
        print(f'  Epoch: {epoch}')
        print(f'  Training SSIM: {ssim_train}')
        if config:
            print(f'  Config: {config}')
        use_preprocessing = True
    else:
        # Old checkpoint format - create preprocessor from data
        print(f'⚠️  Old checkpoint format detected')
        print(f'   Creating preprocessor from test data')
        model_state = checkpoint
        pre = HiCPreprocessor()
        pre.fit(lr_raw, hr_raw)
        use_preprocessing = True
    
    # ✅ Initialize HiCARN Generator (matching training)
    print(f'\nInitializing HiCARN_1 Generator...')
    netG = Generator(num_channels=64).to(device)
    netG.load_state_dict(model_state)
    netG.eval()
    
    n_params = sum(p.numel() for p in netG.parameters())
    print(f'✓ Model loaded successfully')
    print(f'  Parameters: {n_params/1e6:.2f}M')
    
    if use_preprocessing:
        print(f'\nPreprocessor statistics:')
        print(f'  LR log-median: {pre.X_mean:.4f}')
        print(f'  LR log-IQR: {pre.X_std:.4f}')
        print(f'  HR log-median: {pre.Y_mean:.4f}')
        print(f'  HR log-IQR: {pre.Y_std:.4f}')
    
    print(f"{'='*80}\n")
    
    # ================================================================
    # PREPROCESS DATA
    # ================================================================
    
    print("Preprocessing data...")
    lr_norm, hr_norm = pre.preprocess(lr_raw, hr_raw)
    
    print(f"  Normalized LR range: [{lr_norm.min():.2f}, {lr_norm.max():.2f}]")
    print(f"  Normalized HR range: [{hr_norm.min():.2f}, {hr_norm.max():.2f}]")
    
    # Create DataLoader
    lr_tensor = torch.tensor(lr_norm, dtype=torch.float32)
    
    print(f"  Total samples: {len(lr_tensor)}")
    print(f"  Batch size: {batch_size}\n")
    
    # ================================================================
    # RUN INFERENCE
    # ================================================================
    
    print(f"{'='*80}")
    print(f"RUNNING INFERENCE")
    print(f"{'='*80}\n")
    
    all_preds_norm = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(lr_tensor), batch_size), desc='Predicting'):
            lr_batch = lr_tensor[i:i+batch_size].to(device)
            
            # Get predictions (in normalized space)
            sr = netG(lr_batch)
            
            # Store normalized predictions
            all_preds_norm.append(sr.cpu().numpy())
    
    # Concatenate all predictions
    all_preds_norm = np.concatenate(all_preds_norm, axis=0)
    
    # ================================================================
    # POSTPROCESSING
    # ================================================================
    
    print(f"\n{'='*80}")
    print(f"POSTPROCESSING")
    print(f"{'='*80}\n")
    
    # Convert to raw scale
    Y_pred = pre.postprocess(all_preds_norm).astype(np.float64)
    
    print(f"Predictions:")
    print(f"  Shape: {Y_pred.shape}")
    print(f"  Range: [{Y_pred.min():.2f}, {Y_pred.max():.2f}]")
    print(f"  Mean: {Y_pred.mean():.2f}")
    print(f"  Median: {np.median(Y_pred):.2f}")
    print(f"  Sparsity: {(Y_pred == 0).mean()*100:.2f}% zeros")
    
    # ================================================================
    # COMPUTE METRICS (EXACTLY MATCHING DIFFUSION MODEL)
    # ================================================================
    
    print(f"\n{'='*80}")
    print(f"COMPUTING METRICS ON RAW DATA")
    print(f"{'='*80}\n")
    
    n = min(len(hr_raw), len(Y_pred))
    metrics = Metrics()
    
    for i in tqdm(range(n), desc="Computing metrics"):
        metrics.compute_sample(Y_pred[i], hr_raw[i])
    
    summary = metrics.get_summary()
    
    # ================================================================
    # DISPLAY RESULTS (MATCHING DIFFUSION FORMAT)
    # ================================================================
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<10} {'Mean':<12} {'Std':<12} {'Median':<12} {'Status'}")
    print("-"*70)
    
    # Status indicators (matching diffusion)
    pcc_status = "✅ EXCELLENT" if summary['pcc']['mean'] > 0.90 else \
                 ("✅ GOOD" if summary['pcc']['mean'] > 0.80 else \
                  ("⚠️  FAIR" if summary['pcc']['mean'] > 0.70 else "❌ POOR"))
    ssim_status = "✅ EXCELLENT" if summary['ssim']['mean'] > 0.85 else \
                  ("✅ GOOD" if summary['ssim']['mean'] > 0.75 else \
                   ("⚠️  FAIR" if summary['ssim']['mean'] > 0.65 else "❌ POOR"))
    psnr_status = "✅ EXCELLENT" if summary['psnr']['mean'] > 35 else \
                  ("✅ GOOD" if summary['psnr']['mean'] > 30 else \
                   ("⚠️  FAIR" if summary['psnr']['mean'] > 25 else "❌ POOR"))
    
    print(f"{'PCC':<10} {summary['pcc']['mean']:<12.4f} {summary['pcc']['std']:<12.4f} {summary['pcc']['median']:<12.4f} {pcc_status}")
    print(f"{'SSIM':<10} {summary['ssim']['mean']:<12.4f} {summary['ssim']['std']:<12.4f} {summary['ssim']['median']:<12.4f} {ssim_status}")
    print(f"{'PSNR':<10} {summary['psnr']['mean']:<12.2f} {summary['psnr']['std']:<12.2f} {summary['psnr']['median']:<12.2f} {psnr_status}")
    print(f"{'MSE':<10} {summary['mse']['mean']:<12.2f} {summary['mse']['std']:<12.2f} {summary['mse']['median']:<12.2f}")
    print(f"{'MAE':<10} {summary['mae']['mean']:<12.2f} {summary['mae']['std']:<12.2f} {summary['mae']['median']:<12.2f}")
    print(f"{'SNR':<10} {summary['snr']['mean']:<12.2f} {summary['snr']['std']:<12.2f} {summary['snr']['median']:<12.2f}")
    
    print("="*70)
    
    # ================================================================
    # FINAL VERDICT (MATCHING DIFFUSION)
    # ================================================================
    
    print("\n" + "="*80)
    print("HiCARN MODEL VERDICT")
    print("="*80)
    
    if summary['pcc']['mean'] > 0.90 and summary['ssim']['mean'] > 0.85:
        print("✅ ✅ ✅ EXCELLENT PERFORMANCE! ✅ ✅ ✅")
        print(f"   PCC = {summary['pcc']['mean']:.4f} is excellent!")
        print(f"   SSIM = {summary['ssim']['mean']:.4f} is excellent!")
        print("   Model successfully performs LR→HR super-resolution!")
    elif summary['pcc']['mean'] > 0.80 and summary['ssim']['mean'] > 0.75:
        print("✅ GOOD PERFORMANCE")
        print(f"   PCC = {summary['pcc']['mean']:.4f} is good")
        print(f"   SSIM = {summary['ssim']['mean']:.4f} is good")
        print("   Model works well for super-resolution")
    elif summary['pcc']['mean'] > 0.70:
        print("⚠️  FAIR PERFORMANCE")
        print(f"   PCC = {summary['pcc']['mean']:.4f} is acceptable but could be better")
        print("   Consider longer training or hyperparameter tuning")
    else:
        print("❌ POOR PERFORMANCE")
        print(f"   PCC = {summary['pcc']['mean']:.4f} is too low")
        print("   Model needs retraining or debugging")
    
    print("="*80 + "\n")
    
    # ================================================================
    # RETURN RESULTS
    # ================================================================
    
    results = {
        'predictions_raw': Y_pred,
        'ground_truth_raw': hr_raw,
        'metrics': summary,
        'checkpoint_info': {
            'epoch': epoch if 'preprocessor' in checkpoint else 'unknown',
            'training_ssim': ssim_train if 'preprocessor' in checkpoint else 'unknown'
        }
    }
    
    return results

# ================================================================
# 🔹 ARGUMENT PARSER
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test HiCARN (metrics matching diffusion model)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python Predict_log_metrix.py \\
      --lr_input /data/.../lr_test_chr22_ratio16.npy \\
      --hr_gt /data/.../hr_test_chr22.npy \\
      --checkpoint checkpoints/best_model.pytorch \\
      --cuda 0 \\
      --batch_size 64 \\
      --output results_chr22
        """
    )
    
    parser.add_argument('--lr_input', type=str, required=True,
                       help='Path to low-resolution input .npy file')
    parser.add_argument('--hr_gt', type=str, required=True,
                       help='Path to high-resolution ground truth .npy file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pytorch file)')
    parser.add_argument('--cuda', type=int, default=0,
                       help='CUDA device ID (-1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Output directory for results')
    
    return parser.parse_args()

# ================================================================
# 🔹 MAIN
# ================================================================

def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"HiCARN TESTING (METRICS MATCHING DIFFUSION MODEL)")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  LR input: {args.lr_input}")
    print(f"  HR ground truth: {args.hr_gt}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  CUDA device: {args.cuda}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {args.output}")
    print(f"{'='*80}\n")
    
    # Validate inputs
    if not os.path.exists(args.lr_input):
        raise FileNotFoundError(f"LR input not found: {args.lr_input}")
    if not os.path.exists(args.hr_gt):
        raise FileNotFoundError(f"HR ground truth not found: {args.hr_gt}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Setup device (matching training)
    device = torch.device(
        f'cuda:{args.cuda}' if (torch.cuda.is_available() and args.cuda >= 0) else 'cpu'
    )
    print(f"Using device: {device}\n")
    
    # Run testing
    start = time.time()
    
    results = test_hicarn_npy(
        lr_input_path=args.lr_input,
        hr_gt_path=args.hr_gt,
        ckpt_file=args.checkpoint,
        device=device,
        batch_size=args.batch_size
    )
    
    elapsed = (time.time() - start) / 60
    
    # ================================================================
    # SAVE RESULTS
    # ================================================================
    
    os.makedirs(args.output, exist_ok=True)
    
    # Save predictions
    pred_path = os.path.join(args.output, 'predictions.npy')
    np.save(pred_path, results['predictions_raw'])
    print(f"✓ Saved predictions: {pred_path}")
    
    # Save ground truth for reference
    np.save(f"{args.output}/ground_truth.npy", results['ground_truth_raw'])
    print(f"✓ Saved ground truth: {args.output}/ground_truth.npy")
    
    # Save metrics
    import json
    metrics_dict = {
        'metrics': results['metrics'],
        'checkpoint_info': results['checkpoint_info'],
        'test_info': {
            'lr_input': args.lr_input,
            'hr_gt': args.hr_gt,
            'checkpoint': args.checkpoint,
            'num_samples': int(results['predictions_raw'].shape[0]),
            'testing_time_minutes': float(elapsed)
        }
    }
    
    metrics_path = os.path.join(args.output, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✓ Saved metrics: {metrics_path}")
    
    # Save summary text file (matching diffusion format)
    with open(f"{args.output}/summary.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("HiCARN TEST RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("METRICS ON RAW DATA (Matching Diffusion Model):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<10} {'Mean':<12} {'Std':<12} {'Median':<12}\n")
        f.write("-"*80 + "\n")
        
        for metric_name in ['pcc', 'ssim', 'psnr', 'mse', 'mae', 'snr']:
            m = results['metrics'][metric_name]
            f.write(f"{metric_name.upper():<10} {m['mean']:<12.4f} {m['std']:<12.4f} {m['median']:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved summary: {args.output}/summary.txt")
    
    print(f"\n{'='*80}")
    print(f"TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed:.2f} minutes")
    print(f"Results saved to: {args.output}/")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()