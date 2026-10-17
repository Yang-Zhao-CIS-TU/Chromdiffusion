# Structural benchmarking pipeline (TAD and loop recovery)

Scripts used to produce the loop and TAD detection results in the ChromDiffusion
manuscript (Tables 1–3 and Supplementary Table S1). The pipeline converts
model prediction tiles back into Juicer-compatible contact matrices, calls TADs
(Arrowhead) and loops (HiCCUPS) on the reconstructed maps, and scores the calls
against calls made on the high-resolution reference map. The loop recovery and
enrichment analysis in Supplementary Tables S6–S7 was computed separately and is
not part of these scripts.

```
                 HR / LR .hic                     prediction tiles (.npz / .npy)
                      │                                        │
   [0] ground_truth_tad_loop.sh              [1] prediction_convert_combined_TK.py
       Arrowhead + HiCCUPS on HR and LR          tiles → genome coords → Juicer `pre` text
                      │                                        │
                      │                          [2] feature_calling_combined_TK.sh
                      │                              `pre` → .hic → Arrowhead TADs, HiCCUPS loops
                      │                                        │
                      └──────────────► [3] benchmark_combined_TK.py ◄──────────┘
                                          loop F1 / Jaccard (±5 kb), TAD F1 / Jaccard,
                                          optional ChIP-seq loop validation → TSV
```

## Requirements

* Python 3.9 with `numpy` and `pandas` (create the environment from `environment.yml`)
* Java 11 and **juicer_tools 1.22.01** (`juicer_tools_1.22.01.jar`) — set `JUICER_JAR=/path/to/jar`
  or pass `--jar`; the default points at the lab server copy
* `bedtools` ≥ 2.31 (TAD Jaccard)
* HiCCUPS requires a CUDA-capable GPU. On the lab server:
  `export PATH=/usr/local/cuda-11.7/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64`
* Optional: FAN-C (`--mode fanc` insulation boundaries; not used in the manuscript)

## Conventions

| Parameter | Default | Flag |
|---|---|---|
| Resolution | 10 kb | `--resolution` (stages 1 and 2) |
| Test chromosomes | chr18–22 | `--chroms`: stages 1 and 3 take `chr18 chr19 …`; stage 2 takes one quoted string of numbers, e.g. `"18 19"` |
| Downsampling ratio (filename tag) | 16 | `--ratio` (stages 1–3) |
| Tile size | 40 × 40 bins | `--chunk-size` (stage 1) |
| Max diagonal distance kept | 200 bins (2 Mb) | `--thresh-bins` (stage 1) |
| Minimum nonzero fraction of an LR tile | 0.05 | `--min-nonzero-frac` (stage 1) |
| Genome for `.hic` conversion | hg19 | `--genome` (stage 2) |
| Normalisation for calling | KR | `--norm` (stage 2) |
| HiCCUPS FDR | HiCCUPS default (0.10); `--fdr` tags the output folder when set explicitly | `--fdr` (stages 2 and 3) |
| Loop anchor padding | ±5 kb | `--tolerance` (stage 3) |

Stage 0 takes positional arguments only and always uses chr18–22, 10 kb and KR.
Tile size, maximum distance and nonzero threshold in stage 1 must match the values
used to build the model's input tensors, so that each prediction tile is placed
where its input tile came from.

Prediction files are expected as `preds_lr_test_chr{N}_ratio{R}.npy` (one per chromosome,
tiles × 40 × 40 × 1) or the raw model output `hic_impute_chr{N}_pred_real.npz`. If your model writes
`chr{N}.npy`, link or rename the files to this pattern first.

## Stage 0 — reference calls

```bash
bash scripts/ground_truth_tad_loop.sh <cell_test_dir> <cell> [<lr_label>] [<ratio>]
# expects <cell_test_dir>/juicer_ready/<cell>/total_merged.hic and
#         <cell_test_dir>/juicer_ready/<lr_label>/total_merged_downsample_ratio_<ratio>.hic
# writes  <cell_test_dir>/tensors/predictions/ground_truth/{HR,LR}_{arrowhead,hiccups}_chr{N}/
```

## Stage 1 — tiles to Juicer text

```bash
python scripts/prediction_convert_combined_TK.py --input-dir <pred_dir> --input-format npy \
    --juicer-root <cell_test_dir>/juicer_ready --hr-cell GM12878 \
    --chrom-sizes <hg19.chrom.sizes> --ratio 16
# writes <pred_dir>/preds_lr_test_chr{N}_ratio16.txt and *_convert.txt (Juicer `pre` input)
```

Tile placement uses the same LR `intra_NONE` dumps that built the tensors, so the
`--juicer-root`/`--lr-cell` arguments must point at the data the model was run on.
Overlapping tiles are averaged.

## Stage 2 — reconstruct .hic and call features

```bash
bash scripts/feature_calling_combined_TK.sh --mode all --pred-dir <pred_dir>
# writes <pred_dir>/preds_lr_test_chr{N}_ratio16_convert.hic
#        <pred_dir>/preds_lr_test_chr{N}_ratio16_convert_10kb/10000_blocks.bedpe   (Arrowhead)
#        <pred_dir>/hiccups_results_KR_chr{N}[_fdr0.10]/merged_loops.bedpe          (HiCCUPS)
```

## Stage 3 — score against the reference

```bash
python scripts/benchmark_combined_TK.py --mode all --pred-dir <pred_dir> \
    --gt-dir <cell_test_dir>/tensors/predictions/ground_truth \
    --output benchmark_results.tsv
```

Output is a tab-separated table with one row per (Type ∈ {Loop, TAD}, chromosome):
`Type, Comparison, Chromosome, pred_count, hr_count, TP, FP, FN, F1, Jaccard`.
`--include-lr` adds LR-vs-HR rows as the unrefined reference point.

*Loops.* The anchor intervals of every predicted and every reference loop are padded
by ±5 kb (`--tolerance`), and a predicted loop is a true positive (TP) when its padded
anchors overlap those of a reference loop on both axes. FP counts predicted loops with
no match and FN counts reference loops with no match.
F1 = 2TP / (2TP + FP + FN) and Jaccard = TP / (TP + FP + FN).

*TADs.* A predicted domain is a TP if it overlaps any reference domain, so TP can exceed
the reference count. F1 uses the same formula as for loops. Jaccard is the bedtools
interval Jaccard (base-pair overlap) between predicted and reference domains.

Table 1 of the manuscript sums TP and FP over chr18–22 and averages F1 and Jaccard
over the five chromosomes; Tables 2–3 show the per-chromosome values. Supplementary
Table S1 instead sums TP, FP and FN over the five chromosomes and computes F1 and
Jaccard from those totals.

Reference calls are made on the full HR map, while a reconstructed map contains only
tiles within 2 Mb of the diagonal that pass the nonzero filter. Features outside that
coverage count as false negatives for every method.

## Example

`examples/run_example.sh` runs stages 0–3 for one prediction directory. Edit the paths
at the top of the script before running it.

## Provenance

`_TK` scripts were written by Tomoya Kanno (2025–2026). The three combined scripts
supersede the single-purpose scripts named in their headers
(`convert_npz_to_npy.py`, `convert_predictions_to_hic_input_TK.py`, `convert_to_hic.sh`,
`hiccups_loop.sh`, `fanc_boundary.sh`, `Loop_jaccard_F1_TK.py`, `TAD_jaccard_F1_TK.py`,
`Loop_validate.py`). `--ratio` accepts a string so fractional depth-matching ratios
(e.g. `7.5`) can be used as filename tags; integer values behave exactly as before.
