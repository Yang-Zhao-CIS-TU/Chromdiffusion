#!/usr/bin/env bash
# Run the structural benchmark (stages 0-3) for one directory of model predictions.
#
# Before running:
#   1. Activate the environment from ../environment.yml (numpy, pandas, bedtools, Java).
#   2. Edit the paths in the block below.
#
# Usage: bash examples/run_example.sh

set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")/../scripts" && pwd)"

# ---- Edit these ---------------------------------------------------------------
# Cell-line folder with juicer_ready/<CELL>/total_merged.hic, the LR map
# juicer_ready/<CELL>_LR_rep/total_merged_downsample_ratio_<RATIO>.hic and the
# LR intra_NONE dumps used to build the model's input tiles.
CELL_TEST_DIR=/path/to/GM12878_test
CELL=GM12878
RATIO=16
# Model output: preds_lr_test_chr{N}_ratio<RATIO>.npy for chr18-chr22.
PRED_DIR=/path/to/predictions/my_model
CHROM_SIZES=/path/to/hg19.chrom.sizes
export JUICER_JAR=/path/to/juicer_tools_1.22.01.jar
# HiCCUPS needs a CUDA installation on PATH.
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# -------------------------------------------------------------------------------

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
GT_DIR="${CELL_TEST_DIR}/tensors/predictions/ground_truth"

# Stage 0: Arrowhead and HiCCUPS on the HR and LR maps. Needed once per cell line and ratio.
if [[ ! -f "${GT_DIR}/HR_hiccups_chr22/merged_loops.bedpe" ]]; then
    bash "${SCRIPTS}/ground_truth_tad_loop.sh" "$CELL_TEST_DIR" "$CELL" "${CELL}_LR_rep" "$RATIO"
fi

# Stage 1: place prediction tiles back on the genome and write Juicer `pre` input.
python "${SCRIPTS}/prediction_convert_combined_TK.py" \
    --input-dir "$PRED_DIR" --input-format npy \
    --juicer-root "${CELL_TEST_DIR}/juicer_ready" --hr-cell "$CELL" \
    --chrom-sizes "$CHROM_SIZES" --ratio "$RATIO"

# Stage 2: build .hic files, call TADs (Arrowhead) and loops (HiCCUPS).
bash "${SCRIPTS}/feature_calling_combined_TK.sh" --mode all --pred-dir "$PRED_DIR" --ratio "$RATIO"

# Stage 3: score the calls against the HR reference.
python "${SCRIPTS}/benchmark_combined_TK.py" --mode all \
    --pred-dir "$PRED_DIR" --gt-dir "$GT_DIR" --ratio "$RATIO" \
    --output "${PRED_DIR}/benchmark_results.tsv"

echo "Results: ${PRED_DIR}/benchmark_results.tsv"
