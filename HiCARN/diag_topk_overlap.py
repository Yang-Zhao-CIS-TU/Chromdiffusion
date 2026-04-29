#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_topk_overlap.py

Top-K overlap diagnostics for Hi-C contact maps (raw space).

Given prediction and ground truth tensors (N, H, W) (or NHWC / NCHW variants),
this script computes, per sample and aggregated:

- Precision@K / Recall@K / IoU@K between Top-K pixels in pred vs gt
- Hit@1: whether pred top-1 pixel is also in gt Top-K set (same K)

Supports masking:
- ignore pixels close to diagonal (min_diag)
- optionally only use upper triangle (upper=True)

Outputs:
- summary stats per K (mean/std/median/p25/p75)
- per-sample arrays for hit@1 / iou / precision / recall (for hard-mining)

Example:
python diag_topk_overlap.py \
  --pred residual_diffusion/chr19/refined_raw.npy \
  --gt /data/.../hr_test_chr19.npy \
  --min_diag 2 --upper \
  --k_perc 1.0 0.5 0.1 \
  --out_json diag_topk_resdiff_chr19.json
"""

import argparse
import json
import os
from typing import Dict, Tuple, List

import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def to_nhw(x: np.ndarray, name: str) -> np.ndarray:
    """
    Convert input to (N, H, W) float32.

    Accepts:
      - (N, H, W)
      - (N, 1, H, W)  (NCHW)
      - (N, H, W, 1)  (NHWC)
    """
    if x.ndim == 3:
        out = x
    elif x.ndim == 4:
        if x.shape[1] == 1:          # NCHW
            out = x[:, 0, :, :]
        elif x.shape[-1] == 1:       # NHWC
            out = x[:, :, :, 0]
        else:
            raise ValueError(f"[{name}] Unsupported 4D shape (expect channel=1): {x.shape}")
    else:
        raise ValueError(f"[{name}] Unsupported ndim={x.ndim}, shape={x.shape}")
    return out.astype(np.float32)


def build_mask(h: int, w: int, min_diag: int, upper: bool) -> np.ndarray:
    """
    Mask shape (H, W). True = valid.
    - excludes |i-j| < min_diag
    - if upper: only keep i < j (strict upper); otherwise keep both sides
    """
    ii, jj = np.indices((h, w))
    valid = (np.abs(ii - jj) >= int(min_diag))
    if upper:
        valid = valid & (ii < jj)
    return valid


def quantiles(arr: np.ndarray) -> Dict[str, float]:
    """Compute mean/std/median/p25/p75 for 1D array."""
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "p25": float(np.percentile(arr, 25)) if arr.size else float("nan"),
        "p75": float(np.percentile(arr, 75)) if arr.size else float("nan"),
    }


def topk_indices_1d(values_1d: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of top-k elements in values_1d (descending), as 1D indices.
    More stable + faster than full sort: argpartition then sort within top-k.
    """
    n = values_1d.size
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k >= n:
        return np.argsort(values_1d)[::-1].astype(np.int64)

    # get k largest (unsorted)
    idx = np.argpartition(values_1d, -k)[-k:]
    # sort those by value descending
    idx = idx[np.argsort(values_1d[idx])[::-1]]
    return idx.astype(np.int64)


# -----------------------------
# Core metrics
# -----------------------------
def compute_topk_overlap(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    k_list: List[int],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    pred, gt: (N, H, W)
    mask: (H, W) boolean
    k_list: list of K integers

    Returns:
      per_sample: dict[k_key][metric] -> np.ndarray shape (N,)
      summary: dict[k_key][metric] -> stats dict
    """
    n, h, w = pred.shape
    valid_flat = mask.reshape(-1)
    n_valid = int(valid_flat.sum())
    if n_valid <= 0:
        raise ValueError("Mask has 0 valid pixels. Check min_diag/upper.")

    # Flatten and select valid pixels only
    pred_v = pred.reshape(n, -1)[:, valid_flat]  # (N, n_valid)
    gt_v = gt.reshape(n, -1)[:, valid_flat]      # (N, n_valid)

    per_sample: Dict[str, Dict[str, np.ndarray]] = {}
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for K in k_list:
        k = int(K)
        k = max(1, min(k, n_valid))

        prec = np.zeros((n,), dtype=np.float32)
        rec = np.zeros((n,), dtype=np.float32)
        iou = np.zeros((n,), dtype=np.float32)
        hit1 = np.zeros((n,), dtype=np.float32)

        # loop samples: N~1216, n_valid~741 => fine
        for i in range(n):
            p = pred_v[i]
            g = gt_v[i]

            p_idx = topk_indices_1d(p, k)
            g_idx = topk_indices_1d(g, k)

            p_set = set(p_idx.tolist())
            g_set = set(g_idx.tolist())

            inter = len(p_set & g_set)
            union = len(p_set | g_set)

            # Precision/Recall for equal K are identical numerically,
            # but we keep both for clarity.
            prec[i] = inter / float(k)
            rec[i] = inter / float(k)
            iou[i] = (inter / float(union)) if union > 0 else 0.0

            # Hit@1: pred top1 is inside GT top-K
            pred_top1 = p_idx[0]
            hit1[i] = 1.0 if (pred_top1 in g_set) else 0.0

        k_key = f"{(100.0 * k / n_valid):.1f}%"
        per_sample[k_key] = {
            "K": np.array([k], dtype=np.int64),  # convenience
            "precision@K": prec,
            "recall@K": rec,
            "iou@K": iou,
            "hit@1": hit1,
        }
        summary[k_key] = {
            "K": int(k),
            "precision@K": quantiles(prec),
            "recall@K": quantiles(rec),
            "iou@K": quantiles(iou),
            "hit@1": quantiles(hit1),
        }

    return per_sample, summary


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Top-K overlap diagnostics for Hi-C.")
    p.add_argument("--pred", required=True, type=str, help="Prediction .npy (raw space).")
    p.add_argument("--gt", required=True, type=str, help="Ground truth .npy (raw space).")
    p.add_argument("--min_diag", type=int, default=2, help="Exclude |i-j| < min_diag.")
    p.add_argument("--upper", action="store_true", help="Only evaluate strict upper triangle (i<j).")
    p.add_argument(
        "--k_perc",
        nargs="+",
        type=float,
        default=[1.0, 0.5, 0.1],
        help="Top-K percentage(s) of valid pixels (e.g., 1.0 0.5 0.1).",
    )
    p.add_argument("--out_json", required=True, type=str, help="Output JSON file.")
    p.add_argument("--no_per_sample", action="store_true", help="Only save summary stats (smaller JSON).")
    return p.parse_args()


def main():
    args = parse_args()

    pred = np.load(args.pred)
    gt = np.load(args.gt)

    pred = to_nhw(pred, "pred")
    gt = to_nhw(gt, "gt")

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    n, h, w = pred.shape
    mask = build_mask(h, w, min_diag=args.min_diag, upper=args.upper)
    n_valid = int(mask.sum())

    # Convert perc -> K list (ensure deterministic)
    k_list = []
    for perc in args.k_perc:
        k = int(round((perc / 100.0) * n_valid))
        k = max(1, min(k, n_valid))
        k_list.append(k)

    # remove duplicates while preserving order
    seen = set()
    k_list_unique = []
    for k in k_list:
        if k not in seen:
            seen.add(k)
            k_list_unique.append(k)
    k_list = k_list_unique

    print("=" * 60)
    print("Top-K overlap diagnostics")
    print(f"pred={args.pred}")
    print(f"gt  ={args.gt}")
    print(f"shape={pred.shape}  min_diag={args.min_diag}  upper={args.upper}")
    print(f"valid pixels per sample = {n_valid} / {h*w}")
    print("=" * 60)
    print()

    per_sample, summary = compute_topk_overlap(pred, gt, mask, k_list)

    # Pretty print (mean/median)
    for k_key, stats in summary.items():
        k = stats["K"]
        print(f"K = {k} ({k_key} of valid pixels)")
        print(f"  Precision@K mean/med: {stats['precision@K']['mean']:.4f} / {stats['precision@K']['median']:.4f}")
        print(f"  Recall@K     mean/med: {stats['recall@K']['mean']:.4f} / {stats['recall@K']['median']:.4f}")
        print(f"  IoU@K        mean/med: {stats['iou@K']['mean']:.4f} / {stats['iou@K']['median']:.4f}")
        print(f"  Hit@1        mean/med: {stats['hit@1']['mean']:.4f} / {stats['hit@1']['median']:.4f}")
        print()

    out = {
        "pred": args.pred,
        "gt": args.gt,
        "shape": list(map(int, pred.shape)),
        "min_diag": int(args.min_diag),
        "upper": bool(args.upper),
        "n_valid": int(n_valid),
        "k_perc": [float(x) for x in args.k_perc],
        "per_k": summary,
    }

    if not args.no_per_sample:
        # Convert numpy arrays to lists for JSON
        out["per_sample"] = {}
        for k_key, d in per_sample.items():
            out["per_sample"][k_key] = {
                "K": int(d["K"][0]),
                "precision@K": d["precision@K"].astype(float).tolist(),
                "recall@K": d["recall@K"].astype(float).tolist(),
                "iou@K": d["iou@K"].astype(float).tolist(),
                "hit@1": d["hit@1"].astype(float).tolist(),
            }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"✓ Saved: {args.out_json}")


if __name__ == "__main__":
    main()
