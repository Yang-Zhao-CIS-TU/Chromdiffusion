#!/usr/bin/env python
"""Pixel fidelity (PSNR and SSIM) of prediction tiles against the HR tiles
(Supplementary Table S5 of the ChromDiffusion manuscript).

Metrics are computed in log1p space with ONE global data_range = log1p(HR max)
pooled over the chr18-22 HR test tiles, Wang-convention SSIM (11x11 Gaussian
window, sigma 1.5), per-tile metrics averaged within chromosome, and mean +/- SD
across the five chromosomes. "Scale-matched" PSNR first rescales each predicted
tile to the total counts of its HR tile.

Usage:
    python fidelity_psnr_ssim.py --gt-dir <tensors/data/GM12878/40x40> \
        --pred-root <tensors/predictions> [--nproc 8]

--gt-dir holds hr_test_chr{N}.npy from preprocessing/NPYtensorBuilderTKcustom.py;
--pred-root holds one folder per condition with chr{N}.npy predictions, using the
folder names in CONDITIONS below (our run's names; edit them for your own).
Outputs fidelity_psnr_ssim.csv, fidelity_summary.csv and fidelity_meta.json in
the working directory.
"""
import argparse
import os
import sys
import json
import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import skimage

GT_DIR = "/data/tomoya/ExperimentsTK/GM12878_test/tensors/data/GM12878/40x40_200_loc"  # set by --gt-dir
PRED_ROOT = "/data/tomoya/ExperimentsTK/GM12878_test/tensors/predictions"  # set by --pred-root
CHROMS = [18, 19, 20, 21, 22]
EXPECTED_TILES = {18: 1650, 19: 1216, 20: 1302, 21: 755, 22: 764}  # our GM12878 test build

# directory -> (manuscript_label, backbone, variant)
CONDITIONS = [
    ("Jan19_SRHiC_baseline",           "SRHiC Baseline",     "SRHiC",   "Baseline"),
    ("Feb14_srhic_refine1",            "SRHiC Cross-Diff",   "SRHiC",   "Cross-Diff"),
    ("Feb14_srhic_refine2",            "SRHiC Struct-Diff",  "SRHiC",   "Struct-Diff"),
    ("Feb14_srhic_refine3",            "SRHiC Vanilla-Diff", "SRHiC",   "Vanilla-Diff"),
    ("Jan19_HiCNN_baseline",           "HiCNN Baseline",     "HiCNN",   "Baseline"),
    ("Feb14_HiCNN_refine1",            "HiCNN Cross-Diff",   "HiCNN",   "Cross-Diff"),
    ("Feb14_HiCNN_refine2",            "HiCNN Struct-Diff",  "HiCNN",   "Struct-Diff"),
    ("Feb14_HiCNN_refine3",            "HiCNN Vanilla-Diff", "HiCNN",   "Vanilla-Diff"),
    ("Jan19_DeepHiC_baseline",         "DeepHiC Baseline",   "DeepHiC", "Baseline"),
    ("Jan19_DeepHiC_refined",          "DeepHiC Cross-Diff", "DeepHiC", "Cross-Diff"),
    ("Jan22_deephic_refine2",          "DeepHiC Struct-Diff","DeepHiC", "Struct-Diff"),
    ("Jan22_deephic_refine3",          "DeepHiC Vanilla-Diff","DeepHiC","Vanilla-Diff"),
    ("res+diff_40_Jan08/hicarn",       "HiCARN Baseline",    "HiCARN",  "Baseline"),
    ("CD_40Loc_resdiff_15_alpa8_iou",  "HiCARN Run A",       "HiCARN",  "Run A"),
    ("CD_40Loc_resdiff_15_alpa8_pcc",  "HiCARN Run B",       "HiCARN",  "Run B"),
    ("CD_40Loc_resdiff_15_iou",        "HiCARN Run C",       "HiCARN",  "Run C"),
]


def load_gt(ch):
    a = np.load(os.path.join(GT_DIR, "hr_test_chr%d.npy" % ch))
    raw_shape = a.shape
    a = np.squeeze(a).astype(np.float64)
    assert a.ndim == 3 and a.shape[1:] == (40, 40), "GT chr%d bad shape %s" % (ch, a.shape)
    return a, raw_shape


def load_pred(cond, ch):
    a = np.load(os.path.join(PRED_ROOT, cond, "chr%d.npy" % ch))
    raw_shape = a.shape
    a = np.squeeze(a).astype(np.float64)
    assert a.ndim == 3 and a.shape[1:] == (40, 40), "%s chr%d bad shape %s" % (cond, ch, a.shape)
    return a, raw_shape


def task(args):
    cond, label, backbone, variant, ch, DR, gt_dir, pred_root = args
    global GT_DIR, PRED_ROOT
    GT_DIR, PRED_ROOT = gt_dir, pred_root
    gt, gt_raw_shape = load_gt(ch)
    pred, pred_raw_shape = load_pred(cond, ch)

    # HARD assertion: tile counts must match exactly, and match the verified manifest
    if gt.shape[0] != EXPECTED_TILES[ch]:
        raise AssertionError("GT chr%d has %d tiles, manifest says %d"
                             % (ch, gt.shape[0], EXPECTED_TILES[ch]))
    if pred.shape[0] != gt.shape[0]:
        raise AssertionError("TILE COUNT MISMATCH %s chr%d: pred %d vs GT %d"
                             % (cond, ch, pred.shape[0], gt.shape[0]))

    n = gt.shape[0]
    neg_pix = int(np.sum(pred < 0.0))
    pred_min = float(pred.min())

    # ---- GENERAL clip detector: no hard-coded ceiling value ----
    # A clipped export piles hundreds of pixels at exactly its denormalizer cap;
    # an unclipped one has a unique argmax. Judged per (condition, chromosome).
    m = float(pred.max())
    n_at_max = int(np.sum(np.abs(pred - m) <= 1e-6 * abs(m)))
    n_distinct_above_80pct = int(np.unique(pred[pred > 0.8 * m]).size)
    clipped_bool = bool(n_at_max >= 10)

    # numerical guard: log1p is undefined for values <= -1; floor raw counts at 0.
    pred_nn = np.maximum(pred, 0.0)
    gt_nn = np.maximum(gt, 0.0)

    lg_gt = np.log1p(gt_nn)
    lg_pred = np.log1p(pred_nn)

    psnrs = np.empty(n)
    ssims = np.empty(n)
    psnrs_sm = np.empty(n)
    n_zero_sum = 0

    for i in range(n):
        g = lg_gt[i]
        p = lg_pred[i]
        psnrs[i] = peak_signal_noise_ratio(g, p, data_range=DR)
        ssims[i] = structural_similarity(g, p, data_range=DR, win_size=11,
                                         gaussian_weights=True, sigma=1.5,
                                         use_sample_covariance=False)
        # scale-matched: per-tile total matching in RAW space, then log1p
        ps = pred_nn[i].sum()
        gs = gt_nn[i].sum()
        if ps > 0:
            fac = gs / ps
        else:
            fac = 1.0
            n_zero_sum += 1
        psnrs_sm[i] = peak_signal_noise_ratio(np.log1p(gt_nn[i]),
                                             np.log1p(pred_nn[i] * fac),
                                             data_range=DR)

    return dict(directory=cond, manuscript_label=label, backbone=backbone,
                variant=variant, chrom="chr%d" % ch,
                psnr_db=float(psnrs.mean()),
                psnr_scalematched_db=float(psnrs_sm.mean()),
                ssim=float(ssims.mean()),
                n_tiles=n,
                pred_max=m, n_at_max=n_at_max,
                n_distinct_above_80pct=n_distinct_above_80pct,
                clipped_bool=clipped_bool,
                neg_pixel_count=neg_pix, pred_min=pred_min,
                zero_sum_tiles=n_zero_sum,
                gt_raw_shape=str(gt_raw_shape), pred_raw_shape=str(pred_raw_shape))


def main():
    global GT_DIR, PRED_ROOT
    ap = argparse.ArgumentParser(description="PSNR/SSIM of prediction tiles vs HR tiles")
    ap.add_argument("--gt-dir", default=GT_DIR, help="folder with hr_test_chr{N}.npy")
    ap.add_argument("--pred-root", default=PRED_ROOT, help="folder with one subfolder per condition")
    ap.add_argument("--nproc", type=int, default=8, help="worker processes (default: 8)")
    args = ap.parse_args()
    GT_DIR, PRED_ROOT, nproc = args.gt_dir, args.pred_root, args.nproc
    print("skimage", skimage.__version__, "numpy", np.__version__, flush=True)

    # ---- step 1: verify GT build + compute the ONE global data_range ----
    gt_max = -np.inf
    gt_min = np.inf
    counts = {}
    for ch in CHROMS:
        g, shp = load_gt(ch)
        counts[ch] = g.shape[0]
        gt_max = max(gt_max, float(g.max()))
        gt_min = min(gt_min, float(g.min()))
        print("GT chr%d raw_shape=%s tiles=%d expected=%d max=%.6f"
              % (ch, shp, g.shape[0], EXPECTED_TILES[ch], g.max()), flush=True)
        assert g.shape[0] == EXPECTED_TILES[ch], "GT tile-count manifest mismatch chr%d" % ch
    DR = float(np.log1p(gt_max))
    print("POOLED_GT_MAX_RAW=%.10f" % gt_max, flush=True)
    print("POOLED_GT_MIN_RAW=%.10f" % gt_min, flush=True)
    print("DATA_RANGE_log1p=%.10f" % DR, flush=True)

    # ---- step 2: pre-flight tile-count assertion for every condition ----
    bad = []
    for cond, label, bb, var in CONDITIONS:
        for ch in CHROMS:
            p = os.path.join(PRED_ROOT, cond, "chr%d.npy" % ch)
            if not os.path.exists(p):
                bad.append("MISSING %s" % p)
                continue
            a = np.load(p, mmap_mode="r")
            nt = np.squeeze(np.asarray(a[:1])).shape  # shape check on first tile
            ntiles = a.shape[0]
            if ntiles != EXPECTED_TILES[ch]:
                bad.append("COUNT %s chr%d pred=%d gt=%d" % (cond, ch, ntiles, EXPECTED_TILES[ch]))
    if bad:
        print("PREFLIGHT_FAILURES:", flush=True)
        for b in bad:
            print("  ", b, flush=True)
        raise SystemExit("ABORT: tile-count / existence preflight failed")
    print("PREFLIGHT_OK: all 16 conditions x 5 chroms match GT tile counts", flush=True)

    # ---- step 3: metrics ----
    jobs = [(cond, label, bb, var, ch, DR, GT_DIR, PRED_ROOT)
            for cond, label, bb, var in CONDITIONS for ch in CHROMS]
    rows = []
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for i, res in enumerate(ex.map(task, jobs), 1):
            rows.append(res)
            print("done %d/%d %s %s" % (i, len(jobs), res["directory"], res["chrom"]), flush=True)

    order = {(c[0], "chr%d" % ch): k * 10 + j
             for k, c in enumerate(CONDITIONS) for j, ch in enumerate(CHROMS)}
    rows.sort(key=lambda r: order[(r["directory"], r["chrom"])])

    # Per condition: is the detected ceiling the same on all five chromosomes?
    # Relative tolerance 1e-6, same as the plateau test. Only meaningful where clipped.
    for cond, label, bb, var in CONDITIONS:
        rs = [r for r in rows if r["directory"] == cond]
        mx = np.array([r["pred_max"] for r in rs], dtype=float)
        same = bool(np.all(np.abs(mx - mx[0]) <= 1e-6 * np.abs(mx[0])))
        anyclip = any(r["clipped_bool"] for r in rs)
        for r in rs:
            r["ceiling_same_all_chroms"] = same
            if not anyclip:
                r["ceiling_note"] = "unclipped"
            elif same:
                r["ceiling_note"] = "clipped, ceiling invariant across chr18-22 (%.5f)" % mx[0]
            else:
                r["ceiling_note"] = ("clipped, ceiling VARIES by chromosome (%.5f-%.5f)"
                                     % (mx.min(), mx.max()))

    percol = ["directory", "manuscript_label", "backbone", "variant", "chrom",
              "psnr_db", "psnr_scalematched_db", "ssim", "n_tiles",
              "pred_max", "n_at_max", "n_distinct_above_80pct", "clipped_bool",
              "ceiling_same_all_chroms", "ceiling_note"]
    with open("fidelity_psnr_ssim.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=percol, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- step 4: summary, mean +/- SD over the 5 chromosomes (n=5, ddof=1) ----
    summ = []
    for cond, label, bb, var in CONDITIONS:
        rs = [r for r in rows if r["directory"] == cond]
        assert len(rs) == 5
        mx = np.array([r["pred_max"] for r in rs], dtype=float)
        d = dict(directory=cond, manuscript_label=label, backbone=bb, variant=var,
                 n_chrom=5, n_tiles_total=sum(r["n_tiles"] for r in rs),
                 clipped_bool=any(r["clipped_bool"] for r in rs),
                 n_chrom_clipped=sum(1 for r in rs if r["clipped_bool"]),
                 plateau_px_total=sum(r["n_at_max"] for r in rs if r["clipped_bool"]),
                 ceiling_min=float(mx.min()), ceiling_max=float(mx.max()),
                 ceiling_same_all_chroms=rs[0]["ceiling_same_all_chroms"],
                 ceiling_note=rs[0]["ceiling_note"])
        for key, out in (("psnr_db", "psnr"), ("psnr_scalematched_db", "psnr_scalematched"),
                         ("ssim", "ssim")):
            v = np.array([r[key] for r in rs], dtype=float)
            d[out + "_mean"] = float(v.mean())
            d[out + "_sd"] = float(v.std(ddof=1))
        summ.append(d)

    base = {s["backbone"]: s for s in summ if s["variant"] == "Baseline"}
    for s in summ:
        b = base[s["backbone"]]
        for out in ("psnr", "psnr_scalematched", "ssim"):
            s["delta_" + out + "_vs_baseline"] = s[out + "_mean"] - b[out + "_mean"]

    scols = ["directory", "manuscript_label", "backbone", "variant", "n_chrom", "n_tiles_total",
             "psnr_mean", "psnr_sd", "delta_psnr_vs_baseline",
             "psnr_scalematched_mean", "psnr_scalematched_sd", "delta_psnr_scalematched_vs_baseline",
             "ssim_mean", "ssim_sd", "delta_ssim_vs_baseline",
             "clipped_bool", "n_chrom_clipped", "plateau_px_total",
             "ceiling_min", "ceiling_max", "ceiling_same_all_chroms", "ceiling_note"]
    with open("fidelity_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scols, extrasaction="ignore")
        w.writeheader()
        for s in summ:
            w.writerow(s)

    meta = dict(data_range=DR, pooled_gt_max_raw=gt_max, pooled_gt_min_raw=gt_min,
                gt_dir=GT_DIR, pred_root=PRED_ROOT, tile_counts=counts,
                skimage=skimage.__version__, numpy=np.__version__,
                clip_detector="n_at_max within 1e-6 rel tol of pred.max(); clipped if >=10",
                diagnostics=[{k: r[k] for k in ("directory", "chrom", "neg_pixel_count",
                                                "pred_min", "pred_max", "n_at_max",
                                                "n_distinct_above_80pct", "clipped_bool",
                                                "zero_sum_tiles",
                                                "gt_raw_shape", "pred_raw_shape")}
                             for r in rows])
    with open("fidelity_meta.json", "w") as f:
        json.dump(meta, f, indent=1)

    print("WROTE fidelity_psnr_ssim.csv (%d rows), fidelity_summary.csv (%d rows), fidelity_meta.json"
          % (len(rows), len(summ)), flush=True)


if __name__ == "__main__":
    main()
