# Structural benchmarking pipeline (TAD and loop recovery)

Scripts used to produce the loop and TAD detection results in the ChromDiffusion
manuscript (Tables 1–3, Supplementary Tables S1–S2, S6–S7). The pipeline converts
model prediction tiles back into Juicer-compatible contact matrices, calls TADs
(Arrowhead) and loops (HiCCUPS) on the reconstructed maps, and scores the calls
against calls made on the high-resolution reference map.

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

* Python 3.9 with `numpy` and `pandas` (see `environment.yml`; the lab environment is `truhic`)
* Java 11 and **juicer_tools 1.22.01** (`juicer_tools_1.22.01.jar`) — set `JUICER_JAR=/path/to/jar`
  or pass `--jar`; the default points at the lab server copy
* `bedtools` ≥ 2.31 (TAD Jaccard)
* HiCCUPS requires a CUDA-capable GPU. On the lab server:
  `export PATH=/usr/local/cuda-11.7/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64`
* Optional: FAN-C (`--mode fanc` insulation boundaries; not used in the manuscript)

## Conventions (all stages)

| Parameter | Default | Flag |
|---|---|---|
| Resolution | 10 kb | `--resolution` |
| Test chromosomes | chr18–22 | `--chroms` |
| Downsampling ratio (filename tag) | 16 | `--ratio` |
| Tile size | 40 × 40 bins | `--chunk-size` (stage 1) |
| Max diagonal distance kept | 200 bins (2 Mb) | `--thresh-bins` (stage 1) |
| Normalisation for calling | KR | `--norm` (stage 2) |
| HiCCUPS FDR | HiCCUPS default (0.10); `--fdr` tags the output folder when set explicitly | `--fdr` (stages 2 and 3) |
| Loop match tolerance | ±5 kb on both anchors | `--tolerance` (stage 3) |

Prediction files are expected as `preds_lr_test_chr{N}_ratio{R}.npy` (one per chromosome,
tiles × 40 × 40 × 1) or the raw model output `hic_impute_chr{N}_pred_real.npz`. The example
driver shows the symlink step used when model outputs are named `chr{N}.npy`.

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
Loops match when both anchors fall within ±5 kb of a reference loop. TAD Jaccard is the
bedtools interval Jaccard between predicted and reference domains; TAD TP/FP count
predicted domains that overlap a reference domain, so TP can exceed the reference count.
`--include-lr` adds LR-vs-HR rows as the unrefined reference point.

## Example

`examples/run_CD40_pipeline.sh` is the driver that produced the HiCARN rows of Tables 1–3
(stages 1–3 over five prediction batches with per-batch logs). Paths inside it are the lab
server's; treat it as a template.

## Provenance

`_TK` scripts were written by Tomoya Kanno (2025–2026). The three combined scripts
supersede the single-purpose scripts named in their headers
(`convert_npz_to_npy.py`, `convert_predictions_to_hic_input_TK.py`, `convert_to_hic.sh`,
`hiccups_loop.sh`, `fanc_boundary.sh`, `Loop_jaccard_F1_TK.py`, `TAD_jaccard_F1_TK.py`,
`Loop_validate.py`). `--ratio` accepts a string so fractional depth-matching ratios
(e.g. `7.5`) can be used as filename tags; integer values behave exactly as before.
