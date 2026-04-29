"""
Evaluate HiCARN Predictions in RAW Space

Uses the same metric calculation methods as the original evaluation script.
Computes: PSNR, SNR, SSIM, PCC, SPC (Spearman), MSE, GDS (GenomeDISCO)
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
        a_np = a[0][0].numpy() if torch.is_tensor(a[0][0]) else a[0][0]
        b_np = b[0][0].numpy() if torch.is_tensor(b[0][0]) else b[0][0]
        return compute_reproducibility(a_np, b_np, transition=True)

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
        
        # Check for constant arrays (would cause correlation to be undefined)
        if np.std(a_data) < 1e-10 or np.std(b_data) < 1e-10:
            return np.nan
        
        try:
            return spearmanr(a_data, b_data, axis=None)[0]
        except:
            return np.nan

    def comparePCC(self, a, b):
        """Pearson correlation"""
        # Convert to numpy if torch tensor
        a_data = a[0][0].cpu().numpy() if torch.is_tensor(a[0][0]) else a[0][0]
        b_data = b[0][0].cpu().numpy() if torch.is_tensor(b[0][0]) else b[0][0]
        
        # Check for constant arrays (would cause correlation to be undefined)
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
        # Add epsilon to prevent division by zero when prediction matches ground truth perfectly
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


def load_preprocessor(preprocessor_path):
    """Load the HiCARN preprocessor"""
    if os.path.exists(preprocessor_path):
        preprocessor = torch.load(preprocessor_path, map_location='cpu')
        print(f"✓ Loaded preprocessor from: {preprocessor_path}")
        return preprocessor
    else:
        print(f"⚠ Warning: Preprocessor not found at {preprocessor_path}")
        return None


def denormalize_data(data_norm, preprocessor):
    """
    Denormalize data from normalized space back to raw contact counts
    
    Args:
        data_norm: normalized data 
                   - (N, H, W) or 
                   - (N, H, W, 1) or 
                   - (N, 1, H, W)
        preprocessor: HiCARN preprocessor with Y_mean and Y_std
    
    Returns:
        data_raw: raw contact counts in NCHW format (N, 1, H, W)
    """
    if preprocessor is None:
        print("⚠ Warning: No preprocessor available, returning normalized data")
        # Convert to NCHW format
        if data_norm.ndim == 3:
            return data_norm[:, None, :, :]  # (N, H, W) → (N, 1, H, W)
        elif data_norm.ndim == 4:
            if data_norm.shape[3] == 1:
                # (N, H, W, 1) → (N, 1, H, W)
                return data_norm.transpose(0, 3, 1, 2)
        return data_norm
    
    # Convert input to (N, 1, H, W) format for processing
    original_shape = data_norm.shape
    if data_norm.ndim == 3:
        # (N, H, W) → (N, 1, H, W)
        data_norm = data_norm[:, None, :, :]
    elif data_norm.ndim == 4:
        if data_norm.shape[3] == 1:
            # (N, H, W, 1) → (N, 1, H, W)
            data_norm = data_norm.transpose(0, 3, 1, 2)
        # else already (N, 1, H, W) or (N, C, H, W)
    
    # Denormalize using preprocessor's postprocess method
    if hasattr(preprocessor, 'postprocess'):
        data_raw = preprocessor.postprocess(data_norm)
    else:
        # Manual denormalization
        Y_mean = preprocessor.Y_mean
        Y_std = preprocessor.Y_std
        
        # Clip
        data_norm_clipped = np.clip(data_norm, -5, 5)
        
        # Denormalize: Y_log = Y_norm * std + mean
        data_log = data_norm_clipped * Y_std + Y_mean
        
        # Inverse log: Y = exp(Y_log) - 1
        data_raw = np.expm1(data_log)
        
        # Ensure non-negative
        data_raw = np.maximum(data_raw, 0.0)
    
    # Ensure output is in NCHW format (N, 1, H, W)
    if data_raw.ndim == 3:
        data_raw = data_raw[:, None, :, :]
    elif data_raw.ndim == 4 and data_raw.shape[3] == 1:
        # (N, H, W, 1) → (N, 1, H, W)
        data_raw = data_raw.transpose(0, 3, 1, 2)
    
    return data_raw


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HiCARN predictions in RAW space')
    
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to predictions (predictions_norm.npy)')
    parser.add_argument('--gt_path', type=str, required=True,
                       help='Path to ground truth (ground_truth.npy)')
    parser.add_argument('--preprocessor_path', type=str, 
                       default='hicarn_predictions/hicarn_preprocessor.pt',
                       help='Path to HiCARN preprocessor')
    parser.add_argument('--output_json', type=str, default='evaluation_results_raw.json',
                       help='Output JSON file for results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("HICARN EVALUATION IN RAW SPACE")
    print("="*80)
    
    # Load normalized data
    print(f"\nLoading normalized predictions from: {args.pred_path}")
    pred_norm = np.load(args.pred_path)
    
    print(f"Loading normalized ground truth from: {args.gt_path}")
    gt_norm = np.load(args.gt_path)
    
    print(f"\nNormalized data shapes:")
    print(f"  Predictions: {pred_norm.shape}")
    print(f"  Ground truth: {gt_norm.shape}")
    print(f"  Normalized range: [{pred_norm.min():.2f}, {pred_norm.max():.2f}]")
    
    # Load preprocessor
    print(f"\nLoading preprocessor from: {args.preprocessor_path}")
    preprocessor = load_preprocessor(args.preprocessor_path)
    
    # Denormalize to raw space
    print("\nDenormalizing to raw contact counts...")
    pred_raw = denormalize_data(pred_norm, preprocessor)
    gt_raw = denormalize_data(gt_norm, preprocessor)
    
    print(f"\nRaw data shapes (NCHW format):")
    print(f"  Predictions: {pred_raw.shape}")
    print(f"  Ground truth: {gt_raw.shape}")
    print(f"  Raw range - Pred: [{pred_raw.min():.2f}, {pred_raw.max():.2f}]")
    print(f"  Raw range - GT:   [{gt_raw.min():.2f}, {gt_raw.max():.2f}]")
    
    # Verify shapes match
    if pred_raw.shape != gt_raw.shape:
        print(f"\n⚠️  WARNING: Shape mismatch detected!")
        print(f"  Predictions: {pred_raw.shape}")
        print(f"  Ground truth: {gt_raw.shape}")
        raise ValueError(f"Shape mismatch: pred={pred_raw.shape}, gt={gt_raw.shape}")
    
    # Verify NCHW format
    assert pred_raw.ndim == 4, f"Expected 4D tensor, got {pred_raw.ndim}D"
    assert pred_raw.shape[1] == 1, f"Expected 1 channel, got {pred_raw.shape[1]} channels"
    print(f"✓ Shape verification passed: {pred_raw.shape}")
    
    # Convert to torch tensors
    print("\nConverting to torch tensors...")
    pred_tensor = torch.from_numpy(pred_raw).float()
    gt_tensor = torch.from_numpy(gt_raw).float()
    
    # Compute metrics using VisionMetrics
    print("\nComputing metrics on raw data...")
    visionMetrics = VisionMetrics()
    visionMetrics.setDataset(pred_tensor, gt_tensor)
    
    print("Evaluating samples...")
    results = visionMetrics.getMetrics()
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS (RAW SPACE)")
    print("="*80)
    print(f"\nTotal samples evaluated: {len(pred_raw)}")
    
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
        'num_samples': int(len(pred_raw)),
        'data_space': 'raw_contact_counts',
        'pred_range': [float(pred_raw.min()), float(pred_raw.max())],
        'gt_range': [float(gt_raw.min()), float(gt_raw.max())],
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
