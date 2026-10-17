"""Recovery of HR loops by the LR map, and the HiCCUPS donut enrichment of
recovered versus missed loops (Supplementary Tables S6-S7 of the ChromDiffusion
manuscript).

An HR loop counts as recovered when an LR loop has both anchor centres within
--tolerance bp of the HR loop's anchor centres (5 kb by default; at 10 kb bins
this means the same anchor bins). Enrichment is HiCCUPS observed / expectedDonut
of the HR loop. Inputs are the stage 0 calls from ground_truth_tad_loop.sh.

Usage:
    python loop_recovery_enrichment.py --gt-dir <tensors/predictions/ground_truth>

Outputs loop_recovery_by_chrom.csv and loop_recovery_summary.json in the
working directory.
"""
import argparse
import os, sys, csv
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

_ap = argparse.ArgumentParser(description="HR loop recovery by the LR map and loop enrichment")
_ap.add_argument("--gt-dir", default="/data/tomoya/ExperimentsTK/GM12878_test/tensors/predictions/ground_truth",
                 help="folder with HR_hiccups_chr{N}/ and LR_hiccups_chr{N}/ from stage 0")
_ap.add_argument("--tolerance", type=int, default=5000, help="anchor-centre tolerance in bp (default: 5000)")
_args = _ap.parse_args()

BASE = _args.gt_dir
CHROMS = ["chr18", "chr19", "chr20", "chr21", "chr22"]
TOL = _args.tolerance

def read_bedpe(path):
    lines = open(path).read().splitlines()
    assert lines[0].startswith("#"), f"expected '#' header in {path}, got: {lines[0][:80]}"
    names = lines[0].lstrip("#").split("\t")
    data = [l.split("\t") for l in lines[1:] if l and not l.startswith("#")]
    for i, row in enumerate(data):
        if len(row) != len(names):
            raise SystemExit(f"COLUMN MISMATCH {path} row {i}: {len(row)} vs header {len(names)} {names}")
    df = pd.DataFrame(data, columns=names)
    return df

# juicer 1.22 names them observed/expectedDonut; older HiCCUPS output uses o/e_donut
OBS_ALIASES = ("o", "observed")
EXP_ALIASES = ("e_donut", "expectedDonut")

def pick(df, aliases, path):
    hits = [a for a in aliases if a in df.columns]
    if len(hits) != 1:
        raise SystemExit(f"cannot resolve {aliases} in {path}: header={list(df.columns)}")
    return hits[0]

def anchors(df):
    # anchor midpoints from the two bin intervals (col 0-2 = x1/x2, 3-5 = y1/y2 by bedpe convention)
    c = list(df.columns)
    x1, x2, y1, y2 = c[1], c[2], c[4], c[5]
    return df[x1].to_numpy(float), df[x2].to_numpy(float), df[y1].to_numpy(float), df[y2].to_numpy(float)

rows = []
pooled_rec, pooled_mis = [], []
hdr_report = {}

for ch in CHROMS:
    hr_p = f"{BASE}/HR_hiccups_{ch}/merged_loops.bedpe"
    lr_p = f"{BASE}/LR_hiccups_{ch}/merged_loops.bedpe"
    hr = read_bedpe(hr_p); lr = read_bedpe(lr_p)
    hdr_report[ch] = (list(hr.columns), len(hr), len(lr))
    obs_col = pick(hr, OBS_ALIASES, hr_p); exp_col = pick(hr, EXP_ALIASES, hr_p)

    hx1, hx2, hy1, hy2 = anchors(hr)
    lx1, lx2, ly1, ly2 = anchors(lr)
    # anchor centers
    hxc = (hx1 + hx2) / 2.0; hyc = (hy1 + hy2) / 2.0
    lxc = (lx1 + lx2) / 2.0; lyc = (ly1 + ly2) / 2.0

    if len(lr) > 0:
        dx = np.abs(hxc[:, None] - lxc[None, :])
        dy = np.abs(hyc[:, None] - lyc[None, :])
        matched = ((dx <= TOL) & (dy <= TOL)).any(axis=1)   # each HR loop counted once
    else:
        matched = np.zeros(len(hr), bool)

    enr = hr[obs_col].to_numpy(float) / hr[exp_col].to_numpy(float)
    rec = enr[matched]; mis = enr[~matched]
    pooled_rec.append(rec); pooled_mis.append(mis)

    med_r = float(np.median(rec)) if rec.size else float("nan")
    med_m = float(np.median(mis)) if mis.size else float("nan")
    rows.append(dict(chrom=ch, n_hr_loops=int(len(hr)), n_recovered=int(matched.sum()),
                     pct_recovered=round(100.0 * matched.sum() / len(hr), 4),
                     median_enrichment_recovered=round(med_r, 4),
                     median_enrichment_missed=round(med_m, 4),
                     ratio=round(med_r / med_m, 4)))

R = np.concatenate(pooled_rec); M = np.concatenate(pooled_mis)
med_R, med_M = float(np.median(R)), float(np.median(M))
U, p = mannwhitneyu(R, M, alternative="greater")
n_tot = int(sum(r["n_hr_loops"] for r in rows)); n_rec = int(R.size)
rows.append(dict(chrom="all", n_hr_loops=n_tot, n_recovered=n_rec,
                 pct_recovered=round(100.0 * n_rec / n_tot, 4),
                 median_enrichment_recovered=round(med_R, 4),
                 median_enrichment_missed=round(med_M, 4),
                 ratio=round(med_R / med_M, 4)))

with open("loop_recovery_by_chrom.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

qs = [10, 25, 50, 75, 90]
pct = {"recovered": {f"p{q}": round(float(np.percentile(R, q)), 4) for q in qs},
       "missed":    {f"p{q}": round(float(np.percentile(M, q)), 4) for q in qs}}

import json
summary = dict(n_hr_total=n_tot, n_recovered=n_rec, n_missed=int(M.size),
               median_recovered=round(med_R, 4), median_missed=round(med_M, 4),
               ratio=round(med_R / med_M, 4), U=float(U), p=float(p),
               percentiles=pct, per_chrom=rows,
               columns_used={"observed": obs_col, "expectedDonut": exp_col},
               headers={k: v[0] for k, v in hdr_report.items()},
               file_counts={k: {"hr": v[1], "lr": v[2]} for k, v in hdr_report.items()})
with open("loop_recovery_summary.json", "w") as f:
    json.dump(summary, f, indent=1)

print(json.dumps(summary, indent=1))
