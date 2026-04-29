"""
Evaluate HiCARN/Diffusion Predictions in RAW Space

This script:
  1. Takes NORMALIZED predictions as input
  2. Denormalizes them to RAW contact counts
  3. Compares against RAW high-resolution ground truth
  4. Computes metrics: PSNR, SNR, SSIM, PCC, SPC, MSE, GDS

Key difference from v1:
  - Handles normalized predictions automatically
  - Supports both normalized and raw ground truth
  - Clearer workflow for Hi-C evaluation

Usage:
    # With normalized predictions and normalized ground truth
    python evaluate_raw_space_v2.py \
        --pred-norm hicarn_predictions/predictions_norm.npy \
        --gt-norm hicarn_predictions/ground_truth.npy \
        --preprocessor hicarn_predictions/hicarn_preprocessor.pt \
        --output results_chr18.json

    # With normalized predictions and RAW ground truth
    python evaluate_raw_space_v2.py \
        --pred-norm hicarn_predictions/predictions_norm.npy \
        --gt-raw /data/hr_test_chr18.npy \
        --preprocessor hicarn_predictions/hicarn_preprocessor.pt \
        --output results_chr18.json
"""

import argparse
import numpy as np
import torch
import json
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from math import exp
import torch.nn as nn
import torch.nn.functional as F
import os

# Import GenomeDISCO if available
try:
    from Utils.GenomeDISCO import compute_reproducibility
    HAS_GENOMEDISCO = True
except ImportError:
    HAS_GENOMEDISCO = False
    print("⚠ Warning: GenomeDISCO not available, GDS metric will be skipped")


class RobustHiCPreprocessor:
    """
    Robust HiC Preprocessor using Median + IQR
    
    This class is needed to unpickle the preprocessor saved during HiCARN training.
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
        # Clip first
        Y_norm = np.clip(Y_norm, -5, 5)
        
        # Denormalize
        Y_log = Y_norm * self.Y_std + self.Y_mean
        
        # Inverse log
        Y_counts = np.expm1(Y_log)
        
        # Ensure non-negative
        Y_counts = np.maximum(Y_counts, 0.0)
        
        return Y_counts


class SSIM(nn.Module):
    """SSIM implementation from original evaluation script"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def _toimg(self, mat):
        m = torch.tensor(mat)
        return m.float().unsqueeze(0)

    def _tohic(self, mat):
        mat.squeeze_()
        return mat.numpy()

    def gaussian(self, width, sigma):
        gauss = torch.Tensor([exp(-(x - width // 2) ** 2 / float(2 * sigma ** 2)) for x in range(width)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel, sigma=3):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.nanmean()
        else:
            return ssim_map.nanmean(1).nanmean(1).nanmean(1)

    def ssim(self, img1, img2, window_size=11, size_average=True):
        img1 = self._toimg(img1).unsqueeze(0)
        img2 = self._toimg(img2).unsqueeze(0)
        _, channel, _, _ = img1.size()
        window = self.create_window(window_size, channel)
        window = window.type_as(img1)

        return self._ssim(img1, img2, window, window_size, channel, size_average)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class VisionMetrics:
    """Vision metrics calculator from original evaluation script"""
    def __init__(self):
        self.ssim = SSIM()
        self.metric_logs = {
            "pas_psnr": [],
            "pas_snr": [],
            "pas_spc": [],
            "pas_pcc": [],
            "pas_gds": [],
            "pas_ssim": [],
            "pas_mse": [],
        }

    def _logSSIM(self, target, output):
        self.metric_logs['pas_ssim'].append(self.compareSSIM(output, target))

    def _logPSNR(self, target, output):
        self.metric_logs['pas_psnr'].append(self.comparePSNR(output, target))

    def _logPCC(self, target, output):
        self.metric_logs['pas_pcc'].append(self.comparePCC(output, target))

    def _logSPC(self, target, output):
        self.metric_logs['pas_spc'].append(self.compareSPC(output, target))

    def _logMSE(self, target, output):
        self.metric_logs['pas_mse'].append(self.compareMSE(output, target))

    def _logSNR(self, target, output):
        self.metric_logs['pas_snr'].append(self.compareSNR(output, target))

    def _logGDS(self, target, output):
        if HAS_GENOMEDISCO:
            self.metric_logs['pas_gds'].append(self.compareGDS(output, target))

    def compareGDS(self, a, b):
        """GenomeDISCO score"""
        # Convert to numpy if torch tensor
        a_data = a[0][0].cpu().numpy() if torch.is_tensor(a[0][0]) else a[0][0]
        b_data = b[0][0].cpu().numpy() if torch.is_tensor(b[0][0]) else b[0][0]
        return compute_reproducibility(a_data, b_data, transition=True)

    def compareSPC(self, a, b):
        """Spearman correlation"""
        # Convert to numpy if torch tensor
        a_data = a[0][0].cpu().numpy() if torch.is_tensor(a[0][0]) else a[0][0]
        b_data = b[0][0].cpu().numpy() if torch.is_tensor(b[0][0]) else b[0][0]
        return spearmanr(a_data, b_data, axis=None)[0]

    def comparePCC(self, a, b):
        """Pearson correlation"""
        # Convert to numpy if torch tensor
        a_data = a[0][0].cpu().numpy() if torch.is_tensor(a[0][0]) else a[0][0]
        b_data = b[0][0].cpu().numpy() if torch.is_tensor(b[0][0]) else b[0][0]
        return pearsonr(a_data.flatten(), b_data.flatten())[0]

    def comparePSNR(self, a, b):
        """Peak Signal-to-Noise Ratio"""
        MSE = np.square(a[0][0] - b[0][0]).nanmean().item()
        MAX = torch.max(b).item()
        return 20 * np.log10(MAX, where=MAX > 1e-6) - 10 * np.log10(MSE, where=MSE > 1e-6)

    def compareSNR(self, a, b):
        """Signal-to-Noise Ratio"""
        return torch.sum(b[0][0]).item() / (torch.sqrt(torch.sum((b[0][0] - a[0][0]) ** 2)).item())

    def compareSSIM(self, a, b):
        """Structural Similarity Index"""
        return self.ssim(a, b).item()

    def compareMSE(self, a, b):
        """Mean Squared Error"""
        return np.square(a[0][0] - b[0][0]).nanmean().item()

    def log_means(self, name):
        """Get mean and std for a metric"""
        return (name, np.nanmean(self.metric_logs[name]), np.std(self.metric_logs[name]))

    def setDataset(self, model_output, target):
        """Set the datasets to compare"""
        self.model_output = model_output
        self.target = target

    def getMetrics(self):
        """Compute all metrics"""
        self.metric_logs = {
            "pas_psnr": [],
            "pas_snr": [],
            "pas_spc": [],
            "pas_pcc": [],
            "pas_gds": [],
            "pas_ssim": [],
            "pas_mse": [],
        }

        print("Computing metrics...")
        for e in tqdm(range(len(self.model_output)), desc="Evaluating samples"):
            self._logPCC(target=self.target[e:e+1], output=self.model_output[e:e+1])
            self._logSPC(target=self.target[e:e+1], output=self.model_output[e:e+1])
            self._logMSE(target=self.target[e:e+1], output=self.model_output[e:e+1])
            self._logPSNR(target=self.target[e:e+1], output=self.model_output[e:e+1])
            self._logSNR(target=self.target[e:e+1], output=self.model_output[e:e+1])
            self._logSSIM(target=self.target[e:e+1], output=self.model_output[e:e+1])
            if HAS_GENOMEDISCO:
                self._logGDS(target=self.target[e:e+1], output=self.model_output[e:e+1])

        return list(map(self.log_means, self.metric_logs.keys()))


def load_preprocessor(preprocessor_path):
    """Load the HiCARN preprocessor"""
    if os.path.exists(preprocessor_path):
        preprocessor = torch.load(preprocessor_path, map_location='cpu')
        print(f"✓ Loaded preprocessor from: {preprocessor_path}")
        return preprocessor
    else:
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")


def denormalize_to_raw(data_norm, preprocessor, data_type="predictions"):
    """
    Denormalize data from normalized space to raw contact counts
    
    Args:
        data_norm: normalized data (N, H, W) or (N, C, H, W) in range ~[-5, 5]
        preprocessor: HiCARN preprocessor with Y_mean and Y_std
        data_type: description for logging
    
    Returns:
        data_raw: raw contact counts (N, C, H, W) in NCHW format for metrics
    """
    print(f"\nDenormalizing {data_type} to raw space...")
    print(f"  Input shape: {data_norm.shape}")
    print(f"  Input range: [{data_norm.min():.4f}, {data_norm.max():.4f}]")
    
    # Add channel dimension if needed
    if data_norm.ndim == 3:
        data_norm = data_norm[:, None, :, :]
        print(f"  Added channel dimension: {data_norm.shape}")
    
    # Denormalize using preprocessor
    data_raw = preprocessor.postprocess(data_norm)
    
    # Ensure NCHW format (N, C, H, W)
    if data_raw.ndim == 3:
        data_raw = data_raw[:, None, :, :]
    elif data_raw.shape[1] == 1:
        pass  # Already correct
    else:
        # If somehow in NHWC format, convert to NCHW
        if data_raw.shape[-1] == 1:
            data_raw = data_raw.transpose(0, 3, 1, 2)
    
    print(f"  Output shape: {data_raw.shape}")
    print(f"  Output range: [{data_raw.min():.2f}, {data_raw.max():.2f}]")
    
    return data_raw


def load_raw_ground_truth(gt_raw_path):
    """
    Load raw ground truth data
    
    Args:
        gt_raw_path: Path to raw HR ground truth (.npy)
    
    Returns:
        gt_raw: ground truth in NCHW format (N, C, H, W)
    """
    print(f"\nLoading RAW ground truth from: {gt_raw_path}")
    gt_raw = np.load(gt_raw_path)
    
    print(f"  Loaded shape: {gt_raw.shape}")
    print(f"  Loaded range: [{gt_raw.min():.2f}, {gt_raw.max():.2f}]")
    
    # Ensure NCHW format
    if gt_raw.ndim == 3:
        # (N, H, W) → (N, 1, H, W)
        gt_raw = gt_raw[:, None, :, :]
        print(f"  Converted to NCHW: {gt_raw.shape}")
    elif gt_raw.ndim == 4:
        # Check if NHWC (N, H, W, C) and convert to NCHW
        if gt_raw.shape[-1] == 1:
            gt_raw = gt_raw.transpose(0, 3, 1, 2)
            print(f"  Converted NHWC → NCHW: {gt_raw.shape}")
    
    return gt_raw


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate HiCARN/Diffusion predictions in RAW space',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Normalized predictions (required)
    parser.add_argument('--pred-norm', type=str, required=True,
                       help='Normalized predictions (predictions_norm.npy or refined_norm.npy)')
    
    # Ground truth (one of these required)
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument('--gt-norm', type=str,
                         help='Normalized ground truth (ground_truth.npy)')
    gt_group.add_argument('--gt-raw', type=str,
                         help='RAW ground truth (hr_test_chr*.npy)')
    
    # Preprocessor (required for denormalization)
    parser.add_argument('--preprocessor', type=str, required=True,
                       help='Path to HiCARN preprocessor (.pt file)')
    
    # Output
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output JSON file (default: %(default)s)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("HiC SUPER-RESOLUTION EVALUATION IN RAW SPACE")
    print("="*80)
    print(f"Normalized predictions: {args.pred_norm}")
    if args.gt_norm:
        print(f"Normalized ground truth: {args.gt_norm}")
    else:
        print(f"RAW ground truth: {args.gt_raw}")
    print(f"Preprocessor: {args.preprocessor}")
    print("="*80)
    
    # Load preprocessor
    print("\n[1/4] Loading preprocessor...")
    preprocessor = load_preprocessor(args.preprocessor)
    
    # Load normalized predictions
    print("\n[2/4] Loading predictions...")
    print(f"Loading normalized predictions from: {args.pred_norm}")
    pred_norm = np.load(args.pred_norm)
    print(f"  Shape: {pred_norm.shape}")
    print(f"  Range: [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
    
    # Denormalize predictions to raw space
    pred_raw = denormalize_to_raw(pred_norm, preprocessor, "predictions")
    
    # Load and prepare ground truth
    print("\n[3/4] Loading ground truth...")
    if args.gt_norm:
        # Load normalized GT and denormalize
        print(f"Loading normalized ground truth from: {args.gt_norm}")
        gt_norm = np.load(args.gt_norm)
        print(f"  Shape: {gt_norm.shape}")
        print(f"  Range: [{gt_norm.min():.4f}, {gt_norm.max():.4f}]")
        
        gt_raw = denormalize_to_raw(gt_norm, preprocessor, "ground truth")
    else:
        # Load raw GT directly
        gt_raw = load_raw_ground_truth(args.gt_raw)
    
    # Validate shapes match
    if pred_raw.shape != gt_raw.shape:
        raise ValueError(
            f"Shape mismatch! Predictions: {pred_raw.shape}, "
            f"Ground truth: {gt_raw.shape}"
        )
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Predictions (raw): {pred_raw.shape}, range [{pred_raw.min():.2f}, {pred_raw.max():.2f}]")
    print(f"  Ground truth (raw): {gt_raw.shape}, range [{gt_raw.min():.2f}, {gt_raw.max():.2f}]")
    
    # Convert to torch tensors
    print("\n[4/4] Computing metrics...")
    pred_tensor = torch.from_numpy(pred_raw).float()
    gt_tensor = torch.from_numpy(gt_raw).float()
    
    # Compute metrics using VisionMetrics
    visionMetrics = VisionMetrics()
    visionMetrics.setDataset(pred_tensor, gt_tensor)
    results = visionMetrics.getMetrics()
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS (RAW CONTACT COUNT SPACE)")
    print("="*80)
    print(f"\nTotal samples evaluated: {len(pred_raw)}")
    
    print("\n" + "-"*80)
    print("METRIC SUMMARY (Mean ± Std)")
    print("-"*80)
    
    # Parse results
    results_dict = {}
    for metric_name, mean_val, std_val in results:
        # Remove "pas_" prefix for display
        clean_name = metric_name.replace("pas_", "").upper()
        results_dict[metric_name] = {
            'mean': float(mean_val),
            'std': float(std_val)
        }
        print(f"{clean_name:6s}: {mean_val:8.4f} ± {std_val:7.4f}")
    
    print("="*80)
    
    # Save results
    output_results = {
        'evaluation_space': 'raw_contact_counts',
        'num_samples': int(len(pred_raw)),
        'pred_file': args.pred_norm,
        'gt_file': args.gt_norm or args.gt_raw,
        'pred_range': [float(pred_raw.min()), float(pred_raw.max())],
        'gt_range': [float(gt_raw.min()), float(gt_raw.max())],
        'metrics': results_dict,
        'tab_separated_values': "\t".join([f"{x[1]:.6f}" for x in results])
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    
    # Print tab-separated for easy copying
    print("\n" + "-"*80)
    print("TAB-SEPARATED VALUES (for spreadsheet):")
    print("-"*80)
    print("\t".join([x[0].replace("pas_", "").upper() for x in results]))
    print("\t".join([f"{x[1]:.6f}" for x in results]))
    print("="*80)


if __name__ == "__main__":
    main()
