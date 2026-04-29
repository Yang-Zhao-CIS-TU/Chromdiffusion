#!/usr/bin/env python3
"""
Peak local-window diagnostics for Hi-C matrices (raw space)

What it does (per sample):
- Find GT peak locations (top M peaks in GT within the valid mask)
- For each GT peak, look at a local window (radius r) in pred and GT
- Report:
  - peak_max_ratio = pred_window_max / gt_window_max
  - peak_sum_ratio = pred_window_sum / gt_window_sum
  - center_of_mass_offset (pred vs GT) inside the window
  - pred_max_offset: distance between pred window argmax and GT peak coord

This tells you:
- If ratios ~1 but offsets large => "强度对了但位置错"
- If ratios <1 => "强度没补够"
- If sum_ratio >> max_ratio => "变胖/扩散"

Usage:
  python diag_peak_window_energy.py \
    --pred residual_diffusion/chr19/refined_raw.npy \
    --gt /data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40/hr_test_chr19.npy \
    --min_diag 2 --upper \
    --top_peaks 20 \
    --radius 2 \
    --out_json peak_window_chr19.json
"""

import argparse, json
import numpy as np

def load_any(path: str) -> np.ndarray:
    x = np.load(path)
    if x.ndim == 4:
        if x.shape[-1] == 1:
            x = x[..., 0]
        elif x.shape[1] == 1:
            x = x[:, 0, :, :]
        else:
            raise ValueError(f"Unsupported 4D shape: {x.shape}")
    if x.ndim != 3:
        raise ValueError(f"Expected 3D (N,H,W) after squeeze, got {x.shape}")
    return x.astype(np.float64)

def make_mask(H: int, W: int, min_diag: int, upper: bool) -> np.ndarray:
    rr, cc = np.indices((H, W))
    mask = np.ones((H, W), dtype=bool)
    if min_diag > 0:
        mask &= (np.abs(rr - cc) >= min_diag)
    if upper:
        mask &= (rr < cc)
    return mask

def topk_coords(mat: np.ndarray, mask: np.ndarray, k: int):
    # returns list of (r,c) of top-k in masked region
    H, W = mat.shape
    flat = mat.reshape(-1)
    mflat = mask.reshape(-1)
    idx_valid = np.flatnonzero(mflat)
    vec = flat[idx_valid]
    k = min(max(1, k), vec.size)
    sel = np.argpartition(vec, -k)[-k:]
    flat_idx = idx_valid[sel]
    rr = flat_idx // W
    cc = flat_idx % W
    return list(zip(rr.tolist(), cc.tolist()))

def window_slice(r, c, H, W, rad):
    r0 = max(0, r - rad); r1 = min(H, r + rad + 1)
    c0 = max(0, c - rad); c1 = min(W, c + rad + 1)
    return slice(r0, r1), slice(c0, c1), (r0, c0)

def center_of_mass(patch: np.ndarray):
    # patch nonnegative assumed; add eps
    eps = 1e-12
    w = np.maximum(patch, 0.0) + eps
    total = w.sum()
    rr, cc = np.indices(patch.shape)
    r_cm = (rr * w).sum() / total
    c_cm = (cc * w).sum() / total
    return float(r_cm), float(c_cm)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--min_diag", type=int, default=2)
    ap.add_argument("--upper", action="store_true")
    ap.add_argument("--top_peaks", type=int, default=20, help="Number of GT peaks per sample")
    ap.add_argument("--radius", type=int, default=2, help="Window radius r (window size = 2r+1)")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    pred = load_any(args.pred)
    gt   = load_any(args.gt)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch pred={pred.shape}, gt={gt.shape}")

    N, H, W = pred.shape
    mask2d = make_mask(H, W, args.min_diag, args.upper)

    print("="*60)
    print("Peak local-window diagnostics")
    print(f"pred={args.pred}\ngt  ={args.gt}")
    print(f"shape={pred.shape}  min_diag={args.min_diag}  upper={args.upper}")
    print(f"top_peaks={args.top_peaks}  radius={args.radius}")
    print("="*60)

    # collect per-peak stats across all samples
    peak_max_ratio = []
    peak_sum_ratio = []
    pred_max_offset = []
    com_offset = []

    # optional: per-sample summaries
    per_sample = []

    for i in range(N):
        gt_i = gt[i]
        pr_i = pred[i]

        coords = topk_coords(gt_i, mask2d, args.top_peaks)
        # (optional) remove near-duplicates by enforcing min separation
        # Here keep as-is for simplicity.

        ratios_max, ratios_sum, offsets_max, offsets_com = [], [], [], []

        for (r, c) in coords:
            rs, cs, (r0, c0) = window_slice(r, c, H, W, args.radius)
            gt_win = gt_i[rs, cs]
            pr_win = pr_i[rs, cs]

            gt_max = float(np.max(gt_win))
            pr_max = float(np.max(pr_win))
            gt_sum = float(np.sum(gt_win))
            pr_sum = float(np.sum(pr_win))

            # ratios
            rm = pr_max / max(1e-12, gt_max)
            rsu = pr_sum / max(1e-12, gt_sum)

            # pred argmax offset to GT peak
            pr_arg = np.argmax(pr_win)
            wr, wc = np.unravel_index(pr_arg, pr_win.shape)
            pr_r = r0 + wr
            pr_c = c0 + wc
            dmax = float(np.sqrt((pr_r - r)**2 + (pr_c - c)**2))

            # center of mass offsets within window
            gt_cm_r, gt_cm_c = center_of_mass(gt_win)
            pr_cm_r, pr_cm_c = center_of_mass(pr_win)
            # convert cm coords to global
            gt_cm_rg = r0 + gt_cm_r; gt_cm_cg = c0 + gt_cm_c
            pr_cm_rg = r0 + pr_cm_r; pr_cm_cg = c0 + pr_cm_c
            dcom = float(np.sqrt((pr_cm_rg - gt_cm_rg)**2 + (pr_cm_cg - gt_cm_cg)**2))

            ratios_max.append(rm)
            ratios_sum.append(rsu)
            offsets_max.append(dmax)
            offsets_com.append(dcom)

        # aggregate per sample
        if len(ratios_max) > 0:
            peak_max_ratio.extend(ratios_max)
            peak_sum_ratio.extend(ratios_sum)
            pred_max_offset.extend(offsets_max)
            com_offset.extend(offsets_com)

            per_sample.append({
                "sample": int(i),
                "peak_max_ratio_mean": float(np.mean(ratios_max)),
                "peak_sum_ratio_mean": float(np.mean(ratios_sum)),
                "pred_max_offset_mean": float(np.mean(offsets_max)),
                "com_offset_mean": float(np.mean(offsets_com)),
            })

    def summarize(x):
        x = np.asarray(x, dtype=np.float64)
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)),
        }

    summary = {
        "pred": args.pred, "gt": args.gt,
        "shape": list(pred.shape),
        "min_diag": args.min_diag, "upper": args.upper,
        "top_peaks": args.top_peaks, "radius": args.radius,
        "metrics": {
            "peak_max_ratio": summarize(peak_max_ratio),
            "peak_sum_ratio": summarize(peak_sum_ratio),
            "pred_max_offset_pixels": summarize(pred_max_offset),
            "center_of_mass_offset_pixels": summarize(com_offset),
        }
    }

    print("\n" + "-"*60)
    print("Aggregated across ALL peaks (all samples):")
    print(f"peak_max_ratio     mean/med: {summary['metrics']['peak_max_ratio']['mean']:.4f} / {summary['metrics']['peak_max_ratio']['median']:.4f}")
    print(f"peak_sum_ratio     mean/med: {summary['metrics']['peak_sum_ratio']['mean']:.4f} / {summary['metrics']['peak_sum_ratio']['median']:.4f}")
    print(f"pred_max_offset(px) mean/med: {summary['metrics']['pred_max_offset_pixels']['mean']:.4f} / {summary['metrics']['pred_max_offset_pixels']['median']:.4f}")
    print(f"com_offset(px)      mean/med: {summary['metrics']['center_of_mass_offset_pixels']['mean']:.4f} / {summary['metrics']['center_of_mass_offset_pixels']['median']:.4f}")

    # quick interpretation hints
    print("\nInterpretation hints:")
    print("- peak_max_ratio ~1 and peak_sum_ratio ~1: 强度补得对")
    print("- pred_max_offset/com_offset large: 位置错（peak 在附近但不在正确格子）")
    print("- peak_sum_ratio >> peak_max_ratio: 峰变胖/扩散（能量在但尖锐度不足）")

    if args.out_json:
        out = dict(summary)
        out["per_sample"] = per_sample[:2000]  # cap
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n✓ Saved: {args.out_json}")

if __name__ == "__main__":
    main()
