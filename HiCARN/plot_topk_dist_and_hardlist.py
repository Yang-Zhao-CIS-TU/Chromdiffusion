import argparse
import json
import numpy as np
import os
import matplotlib.pyplot as plt

def load_per_sample(path: str):
    if path.endswith(".npz"):
        z = np.load(path)
        # keys like "1.0%_iou", "1.0%_hit1"
        out = {}
        for k in z.files:
            # split "1.0%_iou" -> kperc="1.0%", metric="iou"
            if "_" not in k:
                continue
            kperc, metric = k.split("_", 1)
            out.setdefault(kperc, {})[metric] = z[k]
        return out

    if path.endswith(".json"):
        with open(path, "r") as f:
            j = json.load(f)
        if "per_sample" not in j:
            raise ValueError(f"{path} has no per_sample. Please re-run diag_topk_overlap.py with per-sample saving.")
        out = {}
        for kperc, d in j["per_sample"].items():
            out[kperc] = {
                "precision": np.asarray(d["precision@K"], dtype=np.float32),
                "recall":    np.asarray(d["recall@K"], dtype=np.float32),
                "iou":       np.asarray(d["iou@K"], dtype=np.float32),
                "hit1":      np.asarray(d["hit@1"], dtype=np.int32),
            }
        return out

    raise ValueError("Unsupported file type. Use .npz or .json")

def summarize(arr):
    q = np.quantile(arr, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(q[0]),
        "p10": float(q[1]),
        "p25": float(q[2]),
        "p50": float(q[3]),
        "p75": float(q[4]),
        "p90": float(q[5]),
        "max": float(q[6]),
    }

def hist_save(values_a, values_b, label_a, label_b, title, out_png, bins=40):
    plt.figure()
    plt.hist(values_a, bins=bins, alpha=0.6, label=label_a)
    plt.hist(values_b, bins=bins, alpha=0.6, label=label_b)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="model A per-sample file (.npz or .json)")
    ap.add_argument("--b", required=True, help="model B per-sample file (.npz or .json)")
    ap.add_argument("--kperc", default="1.0%", help='which k bucket, e.g. "1.0%"')
    ap.add_argument("--out_dir", default="topk_dist_out")
    ap.add_argument("--hard_iou_quantile", type=float, default=0.2, help="hard if IoU < this quantile (A)")
    ap.add_argument("--hard_hit1", action="store_true", help="hard also requires hit@1==0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    A = load_per_sample(args.a)
    B = load_per_sample(args.b)
    if args.kperc not in A or args.kperc not in B:
        raise ValueError(f"kperc={args.kperc} not found. A has {list(A.keys())}, B has {list(B.keys())}")

    # support both npz metric naming and json naming
    def get_metrics(D):
        d = D[args.kperc]
        if "iou" in d:   # json-style
            return d["hit1"], d["iou"]
        # npz-style: metric names are "hit1"/"iou" already
        return d.get("hit1"), d.get("iou")

    hitA, iouA = get_metrics(A)
    hitB, iouB = get_metrics(B)

    # basic checks
    if hitA is None or iouA is None or hitB is None or iouB is None:
        raise ValueError("Missing hit1/iou arrays. Check your saved keys.")

    # summary
    print("============================================================")
    print(f"Compare distributions @ {args.kperc}")
    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print("============================================================")

    print("\n[A] hit@1 success rate:", float(np.mean(hitA)))
    print("[B] hit@1 success rate:", float(np.mean(hitB)))

    print("\n[A] IoU summary:", summarize(iouA))
    print("[B] IoU summary:", summarize(iouB))

    # plots
    hist_save(iouA, iouB, "A", "B",
              f"IoU@K distribution ({args.kperc})",
              os.path.join(args.out_dir, f"iou_{args.kperc.replace('%','pct')}.png"))

    hist_save(hitA, hitB, "A", "B",
              f"Hit@1 distribution ({args.kperc})",
              os.path.join(args.out_dir, f"hit1_{args.kperc.replace('%','pct')}.png"),
              bins=2)

    # hard list from A
    thr = np.quantile(iouA, args.hard_iou_quantile)
    hard_mask = (iouA < thr)
    if args.hard_hit1:
        hard_mask = hard_mask & (hitA == 0)

    hard_idx = np.where(hard_mask)[0]
    out_txt = os.path.join(args.out_dir, f"hard_idx_A_{args.kperc.replace('%','pct')}.txt")
    np.savetxt(out_txt, hard_idx, fmt="%d")
    print(f"\nHard threshold (A IoU q={args.hard_iou_quantile}): {thr:.4f}")
    print(f"Hard samples: {len(hard_idx)} / {len(iouA)}")
    print(f"Saved hard indices: {out_txt}")
    print(f"Saved plots to: {args.out_dir}")

if __name__ == "__main__":
    main()
