#!/usr/bin/env python3
# diag_topk_radius.py
import argparse, json, os
import numpy as np

def load_npy_anyshape(path: str) -> np.ndarray:
    x = np.load(path)
    # Accept (N,H,W), (N,H,W,1), (N,1,H,W)
    if x.ndim == 4:
        if x.shape[1] == 1:          # NCHW -> NHW
            x = x[:, 0, :, :]
        elif x.shape[-1] == 1:       # NHWC -> NHW
            x = x[:, :, :, 0]
        else:
            raise ValueError(f"Unsupported 4D shape {x.shape} for {path}")
    if x.ndim != 3:
        raise ValueError(f"Expected 3D after squeeze, got {x.ndim} with shape {x.shape} for {path}")
    return x.astype(np.float32, copy=False)

def build_mask(H: int, W: int, min_diag: int, upper: bool) -> np.ndarray:
    rr, cc = np.indices((H, W))
    diag = np.abs(rr - cc)
    m = diag >= min_diag
    if upper:
        m &= (rr < cc)
    return m

def topk_indices_flat(x_flat: np.ndarray, k: int) -> np.ndarray:
    # returns indices (flat) of top-k values
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k >= x_flat.size:
        return np.argsort(-x_flat)
    idx = np.argpartition(-x_flat, k-1)[:k]
    # sort these k indices by value desc for deterministic
    idx = idx[np.argsort(-x_flat[idx])]
    return idx

def idx_to_rc(idx_flat: np.ndarray, W: int):
    r = idx_flat // W
    c = idx_flat % W
    return r.astype(np.int32), c.astype(np.int32)

def dilate_points_to_mask(r: np.ndarray, c: np.ndarray, H: int, W: int, radius: int) -> np.ndarray:
    """Return boolean mask with True in (2r+1)^2 neighborhood around each point."""
    out = np.zeros((H, W), dtype=bool)
    if r.size == 0:
        return out
    for rr, cc in zip(r, c):
        r0 = max(0, rr - radius); r1 = min(H, rr + radius + 1)
        c0 = max(0, cc - radius); c1 = min(W, cc + radius + 1)
        out[r0:r1, c0:c1] = True
    return out

def stats(arr: np.ndarray):
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="pred .npy (N,H,W) or (N,1,H,W) or (N,H,W,1)")
    ap.add_argument("--gt", required=True, help="gt .npy (same shape conventions)")
    ap.add_argument("--min_diag", type=int, default=2)
    ap.add_argument("--upper", action="store_true")
    ap.add_argument("--radius", type=int, default=2, help="radius for hit@1 and IoU@K")
    ap.add_argument("--k_perc", type=float, nargs="*", default=[1.0, 0.5, 0.1],
                    help="K as percent of valid pixels (e.g. 1.0 0.5 0.1)")
    ap.add_argument("--k_abs", type=int, nargs="*", default=[],
                    help="Optional absolute K list (e.g. 7 4 1). If set, overrides k_perc.")
    ap.add_argument("--out_npz", required=True, help="save per-sample arrays to .npz")
    ap.add_argument("--out_json", default=None, help="save summary to .json")
    args = ap.parse_args()

    pred = load_npy_anyshape(args.pred)
    gt   = load_npy_anyshape(args.gt)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch pred {pred.shape} vs gt {gt.shape}")
    N, H, W = pred.shape

    mask2d = build_mask(H, W, args.min_diag, args.upper)
    valid_idx_flat = np.flatnonzero(mask2d.reshape(-1))
    n_valid = valid_idx_flat.size
    if n_valid == 0:
        raise ValueError("No valid pixels after masking. Check min_diag/upper.")

    # Determine K list
    K_list = []
    K_keys = []
    if args.k_abs and len(args.k_abs) > 0:
        for k in args.k_abs:
            K_list.append(int(k))
            K_keys.append(f"K={int(k)}")
    else:
        for p in args.k_perc:
            k = max(1, int(round(n_valid * (p / 100.0))))
            K_list.append(k)
            K_keys.append(f"{p:.1f}%")
    # storage
    hit1 = np.zeros((N,), dtype=np.float32)  # hit@1@radius
    iouK = {kkey: np.zeros((N,), dtype=np.float32) for kkey in K_keys}

    # Pre-alloc small arrays
    for i in range(N):
        # masked vectors
        p_vec = pred[i].reshape(-1)[valid_idx_flat]
        g_vec = gt[i].reshape(-1)[valid_idx_flat]

        # --- hit@1@radius ---
        p1_local = int(np.argmax(p_vec))
        g1_local = int(np.argmax(g_vec))
        p1_flat = int(valid_idx_flat[p1_local])
        g1_flat = int(valid_idx_flat[g1_local])
        pr, pc = divmod(p1_flat, W)
        gr, gc = divmod(g1_flat, W)
        hit1[i] = 1.0 if (abs(pr - gr) <= args.radius and abs(pc - gc) <= args.radius) else 0.0

        # --- IoU@K@radius for each K ---
        for k, kkey in zip(K_list, K_keys):
            p_idx_local = topk_indices_flat(p_vec, k)
            g_idx_local = topk_indices_flat(g_vec, k)
            p_idx_flat = valid_idx_flat[p_idx_local]
            g_idx_flat = valid_idx_flat[g_idx_local]

            pr, pc = idx_to_rc(p_idx_flat, W)
            grr, gcc = idx_to_rc(g_idx_flat, W)

            # Make boolean masks of exact topK points
            pred_pts = np.zeros((H, W), dtype=bool)
            gt_pts   = np.zeros((H, W), dtype=bool)
            pred_pts[pr, pc] = True
            gt_pts[grr, gcc] = True

            # Allow radius match by dilating pred points
            pred_dil = dilate_points_to_mask(pr, pc, H, W, args.radius)

            inter = np.logical_and(pred_dil, gt_pts).sum()
            union = np.logical_or(pred_pts, gt_pts).sum()  # union of original sets (not dilated) OR use dilated? choose consistent:
            # Better: union in the same "tolerance space": use dilated pred OR gt? We'll use (pred_dil OR gt_pts)
            union_tol = np.logical_or(pred_dil, gt_pts).sum()
            iou = (inter / union_tol) if union_tol > 0 else 0.0
            iouK[kkey][i] = iou

    # save
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    npz_dict = {"hit1_radius": hit1}
    for kkey in K_keys:
        npz_dict[f"iou_radius_{kkey}"] = iouK[kkey]
    np.savez_compressed(args.out_npz, **npz_dict)
    print(f"✓ Saved per-sample arrays: {args.out_npz}")

    summary = {
        "pred": args.pred,
        "gt": args.gt,
        "shape": [int(N), int(H), int(W)],
        "min_diag": int(args.min_diag),
        "upper": bool(args.upper),
        "radius": int(args.radius),
        "n_valid": int(n_valid),
        "K_list": {kkey: int(k) for kkey, k in zip(K_keys, K_list)},
        "hit1_radius": stats(hit1),
        "iou_radius": {kkey: stats(iouK[kkey]) for kkey in K_keys},
        "hit1_zero_frac": float(np.mean(hit1 == 0.0)),
    }

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary json: {args.out_json}")

    # print brief
    print("============================================================")
    print("Per-sample diagnostics (tolerant)")
    print(f"hit@1@r={args.radius}: mean={hit1.mean():.4f}  zero_frac={(hit1==0).mean():.4f}")
    for kkey in K_keys:
        arr = iouK[kkey]
        print(f"IoU@{kkey}@r={args.radius}: mean={arr.mean():.4f}  median={np.median(arr):.4f}  p25={np.percentile(arr,25):.4f}  p75={np.percentile(arr,75):.4f}")
    print("============================================================")

if __name__ == "__main__":
    main()
