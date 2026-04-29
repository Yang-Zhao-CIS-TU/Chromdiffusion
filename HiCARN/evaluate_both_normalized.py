"""
Evaluate Predictions in NORMALIZED Space (Both Already Normalized)

This script evaluates when BOTH predictions and ground truth 
are already in normalized space - no normalization needed.
"""

import os
import argparse
import numpy as np
import torch
import json
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from math import exp
import torch.nn as nn
import torch.nn.functional as F

# Import GenomeDISCO if available
try:
    from Utils.GenomeDISCO import compute_reproducibility
    HAS_GENOMEDISCO = True
except ImportError:
    HAS_GENOMEDISCO = False
    print("⚠ Warning: GenomeDISCO not available, GDS metric will be skipped")


class SSIM(nn.Module):
    """SSIM implementation"""
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
    """Vision metrics calculator"""
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
        a_data = a[0][0].cpu().numpy() if torch.is_tensor(a[0][0]) else a[0][0]
        b_data = b[0][0].cpu().numpy() if torch.is_tensor(b[0][0]) else b[0][0]
        return compute_reproducibility(a_data, b_data, transition=True)

    def compareSPC(self, a, b):
        """Spearman correlation"""
        a_data = a[0][0].cpu().numpy() if torch.is_tensor(a[0][0]) else a[0][0]
        b_data = b[0][0].cpu().numpy() if torch.is_tensor(b[0][0]) else b[0][0]
        
        # Check for constant arrays
        if np.std(a_data) < 1e-10 or np.std(b_data) < 1e-10:
            return np.nan
        
        try:
            return spearmanr(a_data, b_data, axis=None)[0]
        except:
            return np.nan

    def comparePCC(self, a, b):
        """Pearson correlation"""
        a_data = a[0][0].cpu().numpy() if torch.is_tensor(a[0][0]) else a[0][0]
        b_data = b[0][0].cpu().numpy() if torch.is_tensor(b[0][0]) else b[0][0]
        
        # Check for constant arrays
        if np.std(a_data) < 1e-10 or np.std(b_data) < 1e-10:
            return np.nan
        
        try:
            return pearsonr(a_data.flatten(), b_data.flatten())[0]
        except:
            return np.nan

    def comparePSNR(self, a, b):
        """Peak Signal-to-Noise Ratio"""
        MSE = np.square(a[0][0] - b[0][0]).nanmean().item()
        MAX = torch.max(b).item()
        
        # Add epsilon to prevent log of zero
        epsilon = 1e-10
        MSE = max(MSE, epsilon)
        MAX = max(MAX, epsilon)
        
        return 20 * np.log10(MAX) - 10 * np.log10(MSE)

    def compareSNR(self, a, b):
        """Signal-to-Noise Ratio"""
        # Add epsilon to prevent division by zero
        denominator = torch.sqrt(torch.sum((b[0][0] - a[0][0]) ** 2)).item()
        epsilon = 1e-10
        return torch.sum(b[0][0]).item() / (denominator + epsilon)

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

        # Add progress bar
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


def convert_to_nchw(data):
    """
    Convert input data to NCHW format (N, 1, H, W)
    
    Handles multiple input formats:
    - (N, H, W) → (N, 1, H, W)
    - (N, H, W, 1) → (N, 1, H, W)
    - (N, 1, H, W) → (N, 1, H, W)
    """
    if data.ndim == 3:
        # (N, H, W) → (N, 1, H, W)
        return data[:, None, :, :]
    elif data.ndim == 4:
        if data.shape[3] == 1:
            # (N, H, W, 1) → (N, 1, H, W)
            return data.transpose(0, 3, 1, 2)
        elif data.shape[1] == 1:
            # Already (N, 1, H, W)
            return data
        else:
            raise ValueError(f"Unexpected shape: {data.shape}")
    else:
        raise ValueError(f"Unexpected ndim: {data.ndim}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate predictions in NORMALIZED space (both already normalized)'
    )
    
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to predictions (normalized)')
    parser.add_argument('--gt_path', type=str, required=True,
                       help='Path to ground truth (normalized)')
    parser.add_argument('--output_json', type=str, default='evaluation_results_norm.json',
                       help='Output JSON file for results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("EVALUATION IN NORMALIZED SPACE (BOTH ALREADY NORMALIZED)")
    print("="*80)
    
    # Load data
    print(f"\nLoading normalized predictions from: {args.pred_path}")
    pred_norm = np.load(args.pred_path)
    
    print(f"Loading normalized ground truth from: {args.gt_path}")
    gt_norm = np.load(args.gt_path)
    
    print(f"\nData shapes:")
    print(f"  Predictions: {pred_norm.shape}")
    print(f"  Ground truth: {gt_norm.shape}")
    print(f"  Predictions range: [{pred_norm.min():.2f}, {pred_norm.max():.2f}]")
    print(f"  Ground truth range: [{gt_norm.min():.2f}, {gt_norm.max():.2f}]")
    
    # Check if data looks normalized (typical range: -3 to 7)
    if gt_norm.min() >= 0 and gt_norm.max() > 1000:
        print("\n⚠️  WARNING: Ground truth looks like RAW data (large positive values)")
        print("    This script expects NORMALIZED ground truth.")
        print("    Please use evaluate_normalized_space.py if GT is raw.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Convert to NCHW format
    print("\nConverting to NCHW format (N, 1, H, W)...")
    pred_nchw = convert_to_nchw(pred_norm)
    gt_nchw = convert_to_nchw(gt_norm)
    
    print(f"\nNCHW format shapes:")
    print(f"  Predictions: {pred_nchw.shape}")
    print(f"  Ground truth: {gt_nchw.shape}")
    
    # Verify shapes match
    if pred_nchw.shape != gt_nchw.shape:
        print(f"\n❌ ERROR: Shape mismatch!")
        print(f"  Predictions: {pred_nchw.shape}")
        print(f"  Ground truth: {gt_nchw.shape}")
        raise ValueError(f"Shape mismatch: pred={pred_nchw.shape}, gt={gt_nchw.shape}")
    
    # Verify NCHW format
    assert pred_nchw.ndim == 4, f"Expected 4D tensor, got {pred_nchw.ndim}D"
    assert pred_nchw.shape[1] == 1, f"Expected 1 channel, got {pred_nchw.shape[1]} channels"
    print(f"✓ Shape verification passed: {pred_nchw.shape}")
    
    # Convert to torch tensors
    print("\nConverting to torch tensors...")
    pred_tensor = torch.from_numpy(pred_nchw).float()
    gt_tensor = torch.from_numpy(gt_nchw).float()
    
    # Compute metrics
    print("\nComputing metrics on normalized data...")
    visionMetrics = VisionMetrics()
    visionMetrics.setDataset(pred_tensor, gt_tensor)
    
    results = visionMetrics.getMetrics()
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS (NORMALIZED SPACE)")
    print("="*80)
    print(f"\nTotal samples evaluated: {len(pred_norm)}")
    
    print("\n" + "-"*80)
    print("METRIC SUMMARY (Mean ± Std)")
    print("-"*80)
    
    # Parse results
    results_dict = {}
    for metric_name, mean_val, std_val in results:
        # Remove "pas_" prefix
        clean_name = metric_name.replace("pas_", "").upper()
        results_dict[metric_name] = {
            'mean': float(mean_val),
            'std': float(std_val)
        }
        print(f"{clean_name:6s}: {mean_val:8.4f} ± {std_val:7.4f}")
    
    print("="*80)
    
    # Save results
    output_results = {
        'num_samples': int(len(pred_norm)),
        'data_space': 'normalized',
        'note': 'Both predictions and ground truth are already in normalized space',
        'pred_range': [float(pred_norm.min()), float(pred_norm.max())],
        'gt_range': [float(gt_norm.min()), float(gt_norm.max())],
        'metrics': results_dict,
        'tab_separated_values': "\t".join([str(x[1]) for x in results])
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output_json}")
    
    # Print tab-separated for easy copying
    print("\n" + "-"*80)
    print("TAB-SEPARATED VALUES (for spreadsheet):")
    print("-"*80)
    print("\t".join([x[0].replace("pas_", "").upper() for x in results]))
    print("\t".join([f"{x[1]:.6f}" for x in results]))
    print("="*80)


if __name__ == "__main__":
    main()
