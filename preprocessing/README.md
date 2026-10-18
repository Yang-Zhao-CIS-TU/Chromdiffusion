# Data preprocessing (GEO contact lists to model input tiles)

Scripts used to build the high-resolution (HR) and 1/16 downsampled low-resolution (LR)
inputs for the ChromDiffusion manuscript: from the GSE63525 contact lists to the 40 × 40
tiles that every backbone and refinement model was trained and tested on. The same
scripts handle K562 and NHEK. The stages follow DFHiC's preprocessing (see Provenance).

```
   GEO GSE63525: per-replicate *_merged_nodups.txt.gz
                        │
   [1] raw_data_download_script.sh     download one cell line
                        │
   [2] merge_contacts.sh               merge replicates, sort chr1 → chr2 → pos1 → pos2
                        │              → juicer_ready/<CELL>/total_merged_nodups.txt
   [3] preprocess_rep_TK_25.sh         1/16 random downsample, HR and LR .hic,
                        │              per-chromosome contact dumps (chr1–22)
   [4] NPYtensorBuilderTKcustom.py     40 × 40 HR/LR tiles: train chr1–17, test chr18–22
                        │
          model training and ../benchmark
```

## Requirements

* Java 11 and **juicer_tools 1.22.01** (stage 3)
* GNU coreutils (`sort` with `-V`, `shuf`), `gzip` and `wget`
* Python 3.9 with `numpy` (stage 4). The environment in `../benchmark/environment.yml`
  covers stages 3–4 except juicer_tools itself.

Resources for GM12878, the largest of the three cell lines (sizes from our run):

| Item | Size |
|---|---|
| Downloaded `*_merged_nodups.txt.gz` (18 replicates) | 63 GB |
| Merged, sorted contact list | 239 GB, plus about as much temporary space for `sort` |
| HR `.hic` / LR `.hic` | 13 GB / 1.8 GB |
| Per-chromosome dumps (`intra_NONE`, HR / LR) | 8.7 GB / 1.1 GB |
| Test tiles (chr18–22) plus train and test NPZ files | 810 MB |

Stage 3 runs Juicer `pre` with `-Xmx200g`; lower it in the script on a smaller machine.
Stage 4 holds the HR and LR matrices of chr1–22 in memory at once, about 40 GB.

## Data

All three cell lines come from GEO GSE63525 (Rao et al., 2014). The pipeline starts from the
per-replicate contact lists published there (`*_merged_nodups.txt.gz`), which the data's
authors produced with Juicer; it does not realign reads.

| Cell line | Replicates | GEO samples |
|---|---|---|
| GM12878 | HIC001–HIC018 | GSM1551550–GSM1551567 |
| K562 | HIC069–HIC074 | GSM1551618–GSM1551623 |
| NHEK | HIC065–HIC067 | GSM1551614–GSM1551616 |

`metadata/hg19.chrom.sizes` is the chromosome sizes file used by stage 4 and by
`../benchmark` stage 1.

## Stage 1 — download

```bash
bash scripts/raw_data_download_script.sh GM12878 <cell_test_dir>/raw/merged_nodups_gz
```

`wget -c` resumes interrupted downloads, so the command can be rerun safely.

## Stage 2 — merge and sort

```bash
bash scripts/merge_contacts.sh <cell_test_dir>/raw/merged_nodups_gz \
    <cell_test_dir>/juicer_ready/GM12878/total_merged_nodups.txt <sort_tmp_dir>
```

The files have 11 columns (`readname str1 chr1 pos1 frag1 str2 chr2 pos2 frag2 mapq1 mapq2`),
and the script sorts on columns 3, 7, 4 and 8. Juicer `pre` needs this full four-key order;
with a partial sort it stops with "chromosome combination … appears in multiple blocks".

## Stage 3 — downsample, .hic files and dumps

```bash
SORT_TMPDIR=<sort_tmp_dir> bash scripts/preprocess_rep_TK_25.sh GM12878 10000 \
    <juicer_tools_1.22.01.jar> GM12878_LR_rep <cell_test_dir>/juicer_ready hg19
```

Arguments: cell label, resolution in bp, path to juicer_tools, LR folder name, the
`juicer_ready` folder, genome. Outputs, under `<cell_test_dir>/juicer_ready/`:

```
GM12878/total_merged.hic
GM12878/intra_NONE/chr{N}_10k_intra_NONE.txt
GM12878_LR_rep/total_merged_nodups_downsample_ratio_16.txt
GM12878_LR_rep/total_merged_downsample_ratio_16.hic
GM12878_LR_rep/intra_NONE/chr{N}_10k_intra_NONE_downsample_ratio16.txt
```

* The LR contact list keeps a random 1/16 of the HR contacts (`shuf`, unseeded), so a
  rerun draws a different sample and the LR map will differ slightly from ours.
* The divisor is fixed at 16 in the script (`ratio=16`).
* The script also writes dense dumps to `intra_HR/` and `intra_LR/`, a DFHiC format that
  the later stages do not use.
* A log goes to `<cell_test_dir>/logs/preprocess_<timestamp>.log`.

## Stage 4 — tiles

```bash
python scripts/NPYtensorBuilderTKcustom.py --workspace-root <cell_test_dir>/juicer_ready \
    --hr-cell GM12878 --chrom-sizes metadata/hg19.chrom.sizes
```

| Parameter | Default | Flag |
|---|---|---|
| Tile size | 40 × 40 bins | `--chunk-size` |
| Maximum diagonal distance | 200 bins (2 Mb) | `--thresh-bins` |
| Minimum nonzero fraction of the LR tile | 0.05 | `--min-nonzero-frac` |
| Resolution | 10 kb | `--resolution` |
| Downsampling divisor (filename tag) | 16 | `--ratio` |

Tiles are taken on a non-overlapping 40-bin grid. A tile is kept when its offset from the
diagonal is under 200 bins and at least 5% of its LR entries are nonzero; the HR tile at
the same position is kept with it. Values are raw contact counts; the log transform and
standardization happen in the model scripts.

Outputs, in `<cell_test_dir>/tensors/data/GM12878/40x40/`:

* `train_data_raw_ratio16.npz`: `train_lr`, `train_hr` (tiles × 40 × 40 × 1), `distance`
  (diagonal offset and chromosome per tile) and `location` (tile origin in bins) for chr1–17.
* `hr_test_chr{N}.npy`, `lr_test_chr{N}_ratio16.npy`, `distance_test_chr{N}.npy` and
  `location_test_chr{N}.npy` for chr18–22, plus the combined `test_data_raw_ratio16.npz`.
* `build_report.txt`: tiles considered and kept per chromosome.

`../benchmark` stage 1 applies the same grid and filter to the LR dumps to put prediction
tiles back in place, so it must use the same parameters and the same LR dumps.

Our GM12878 build kept 1,650, 1,216, 1,302, 755 and 764 test tiles on chr18–22. Rerunning
this script on our dumps reproduces those test tensors exactly; with a newly drawn LR
sample the counts will differ slightly.

## Example

`examples/run_example.sh` runs stages 1–4 for one cell line. Edit the paths at the top of
the script before running it.

## Not included

Supplementary Tables S1–S2 also use inputs at divisors other than 16 (8 for all three cell
lines, 7.5 for K562 and 6.8 for NHEK). These scripts fix the divisor at 16, so those inputs
are not covered here yet.

## Provenance

`preprocess_rep_TK_25.sh` and `NPYtensorBuilderTKcustom.py` were written by Tomoya Kanno,
adapted from DFHiC's `preprocess.sh` and `generate_train_data.py`.
`raw_data_download_script.sh` comes from hicGAN by way of DFHiC; the changes are listed in
its header. `merge_contacts.sh` packages the merge-and-sort commands used for our data.
Both upstream projects are MIT-licensed; their license texts are in
`THIRD_PARTY_LICENSES.md`.
