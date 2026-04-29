import json
import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="diag_topk_*.json")
    parser.add_argument("--out", required=True, help="output .npz")
    parser.add_argument("--k", default="1.0%", help="which K bucket, e.g. 1.0%, 0.5%, 0.1%")
    args = parser.parse_args()

    with open(args.json, "r") as f:
        data = json.load(f)

    if "per_sample" not in data:
        raise ValueError("JSON does not contain per_sample (re-run diag_topk_overlap.py without --no_per_sample)")

    if args.k not in data["per_sample"]:
        raise ValueError(f"K={args.k} not found. Available: {list(data['per_sample'].keys())}")

    d = data["per_sample"][args.k]

    out = {
        "precision": np.array(d["precision@K"], dtype=np.float32),
        "recall":    np.array(d["recall@K"], dtype=np.float32),
        "iou":       np.array(d["iou@K"], dtype=np.float32),
        "hit1":      np.array(d["hit@1"], dtype=np.float32),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **out)

    print(f"✓ Saved per-sample Top-K metrics to {args.out}")
    for k, v in out.items():
        print(f"  {k}: shape={v.shape}, mean={v.mean():.4f}")

if __name__ == "__main__":
    main()
