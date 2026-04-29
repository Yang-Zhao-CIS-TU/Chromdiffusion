# eval_norm.py
"""
Evaluate predictions in NORMALIZED space only.
Used to measure reconstruction fidelity (NOT biological validity).
"""

import argparse
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from vision_metrics import VisionMetrics   # 你已有的 VisionMetrics / SSIM 实现

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-norm', required=True)
    p.add_argument('--gt-norm', required=True)
    return p.parse_args()

def main():
    args = parse_args()

    pred = np.load(args.pred_norm)
    gt   = np.load(args.gt_norm)

    assert pred.shape == gt.shape, "Shape mismatch"

    # Ensure NCHW
    if pred.ndim == 3:
        pred = pred[:, None]
        gt   = gt[:, None]

    pred_t = torch.from_numpy(pred).float()
    gt_t   = torch.from_numpy(gt).float()

    metrics = VisionMetrics()
    metrics.setDataset(pred_t, gt_t)
    results = metrics.getMetrics()

    print("\n=== NORMALIZED SPACE METRICS ===")
    for name, mean, std in results:
        print(f"{name.replace('pas_','').upper():6s}: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
