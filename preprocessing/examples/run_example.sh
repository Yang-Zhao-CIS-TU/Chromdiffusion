#!/usr/bin/env bash
# Build the HR and 1/16 LR inputs for one cell line, from the GEO download to 40x40 tiles.
#
# Before running:
#   1. Activate the environment from ../../benchmark/environment.yml (Python, numpy, Java 11).
#   2. Edit the paths in the block below.
#
# Usage: bash examples/run_example.sh

set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")/../scripts" && pwd)"
METADATA="$(cd "$(dirname "$0")/../metadata" && pwd)"

# ---- Edit these ---------------------------------------------------------------
CELL=GM12878                               # GM12878, K562 or NHEK
CELL_TEST_DIR=/path/to/GM12878_test        # working folder for this cell line
JUICER_JAR=/path/to/juicer_tools_1.22.01.jar
export SORT_TMPDIR=/path/to/large/tmp      # temporary space for sort
RESOLUTION=10000
# -------------------------------------------------------------------------------

JUICER_READY="${CELL_TEST_DIR}/juicer_ready"

# Stage 1: download the per-replicate contact lists from GEO GSE63525.
bash "${SCRIPTS}/raw_data_download_script.sh" "$CELL" "${CELL_TEST_DIR}/raw/merged_nodups_gz"

# Stage 2: merge the replicates and sort for Juicer.
bash "${SCRIPTS}/merge_contacts.sh" "${CELL_TEST_DIR}/raw/merged_nodups_gz" \
    "${JUICER_READY}/${CELL}/total_merged_nodups.txt" "$SORT_TMPDIR"

# Stage 3: 1/16 downsample, HR and LR .hic files, per-chromosome dumps.
bash "${SCRIPTS}/preprocess_rep_TK_25.sh" "$CELL" "$RESOLUTION" "$JUICER_JAR" \
    "${CELL}_LR_rep" "$JUICER_READY" hg19

# Stage 4: 40x40 HR/LR tiles (train chr1-17, test chr18-22).
python "${SCRIPTS}/NPYtensorBuilderTKcustom.py" \
    --workspace-root "$JUICER_READY" --hr-cell "$CELL" \
    --chrom-sizes "${METADATA}/hg19.chrom.sizes"

echo "Tiles: ${CELL_TEST_DIR}/tensors/data/${CELL}/40x40/"
