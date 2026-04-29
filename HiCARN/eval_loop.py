# quantile_compare.py
import argparse
import numpy as np

def to_nhw(mat):
    """Accept (N,H,W), (N,1,H,W), (N,H,W,1) -> (N,H,W)"""
    x = np.load(mat) if isinstance(mat, str) else mat
    if x.ndim == 4 and x.shape[1] == 1:      # NCHW
        x = x[:, 0]
    elif x.ndim == 4 and x.shape[-1] == 1:   # NHWC
        x = x[:, :, :, 0]
    if x.ndim != 3:
        raise ValueError(f"Unsupported shape: {x.shape}")
    return x

def make_mask(H, W, min_diag=0, upper=True):
    """mask for upper triangle and excluding |i-j| < min_diag."""
    yy, xx = np.ogrid[:H, :W]
    diag_dist = np.abs(yy - xx)
    m = diag_dist >= min_diag
    if upper:
        m = m & (yy <= xx)  # include diagonal if min_diag==0, otherwise excluded by diag_dist
    return m

def per_sample_quantiles(x3d, mask=None, qs=(0.5, 0.9, 0.99, 0.999)):
    """Return arrays of shape (N, len(qs)) and (N,) max."""
    N, H, W = x3d.shape
    out_q = np.zeros((N, len(qs)), dtype=np.float64)
    out_max = np.zeros((N,), dtype=np.float64)
    for i in range(N):
        v = x3d[i]
        if mask is not None:
            v = v[mask]
        else:
            v = v.reshape(-1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out_q[i] = np.nan
            out_max[i] = np.nan
            continue
        out_q[i] = np.quantile(v, qs)
        out_max[i] = np.max(v)
    return out_q, out_max

def summarize(name, qvals, maxvals, qs):
    print(f"\n[{name}]  (across samples)")
    for j, q in enumerate(qs):
        arr = qvals[:, j]
        print(f"  p{q*100:6.2f}: mean={np.nanmean(arr):10.2f}  median={np.nanmedian(arr):10.2f}")
    print(f"  max     : mean={np.nanmean(maxvals):10.2f}  median={np.nanmedian(maxvals):10.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="pred raw .npy")
    ap.add_argument("--gt", required=True, help="gt raw .npy")
    ap.add_argument("--min_diag", type=int, default=2, help="exclude |i-j| < min_diag (default 2)")
    ap.add_argument("--upper", action="store_true", help="use only upper triangle (default False)")
    args = ap.parse_args()

    pred = to_nhw(args.pred)
    gt = to_nhw(args.gt)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    N, H, W = pred.shape
    mask = make_mask(H, W, min_diag=args.min_diag, upper=args.upper)

    qs = (0.5, 0.9, 0.99, 0.999)

    pred_q, pred_mx = per_sample_quantiles(pred, mask=mask, qs=qs)
    gt_q, gt_mx = per_sample_quantiles(gt, mask=mask, qs=qs)

    print("============================================================")
    print("Quantile comparison (raw space)")
    print(f"Files: pred={args.pred}  gt={args.gt}")
    print(f"Shape: {pred.shape}  mask: min_diag={args.min_diag}, upper={args.upper}")
    print("============================================================")

    summarize("PRED", pred_q, pred_mx, qs)
    summarize("GT  ", gt_q, gt_mx, qs)

    # Helpful ratio report (median ratios)
    print("\n[Ratio PRED/GT] (median across samples)")
    for j, q in enumerate(qs):
        r = pred_q[:, j] / (gt_q[:, j] + 1e-12)
        print(f"  p{q*100:6.2f}: median ratio={np.nanmedian(r):8.4f}")
    rmax = pred_mx / (gt_mx + 1e-12)
    print(f"  max     : median ratio={np.nanmedian(rmax):8.4f}")

if __name__ == "__main__":
    main()
