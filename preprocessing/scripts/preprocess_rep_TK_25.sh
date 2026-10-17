#!/bin/bash

# Preprocess Hi-C data for ExperimentsTK.
# Creates .hic files and per-chromosome contact matrices from merged contact lists.
# High-coverage contacts → $WORKSPACE_ROOT/<CELL>
# Downsampled (1/16) contacts → $WORKSPACE_ROOT/<LR_CELL>
#
# IMPORTANT: Juicer pre requires full 4-key sort (chr1 → chr2 → pos1 → pos2).
#
# Usage: preprocess_rep_TK_25.sh <cell_label> <resolution_bp> <juicer_tools.jar> [lr_cell_label] [workspace_root] [genome]
# Examples:
#   preprocess_rep_TK_25.sh GM12878 10000 juicer_tools.jar
#   preprocess_rep_TK_25.sh K562 10000 juicer_tools.jar K562_LR_rep /data/tomoya/ExperimentsTK/K562_test/juicer_ready hg19
#
# Set SORT_TMPDIR to a folder with room for sorting the 1/16 contact list (default: /data/tomoya/tmp).

set -euo pipefail

CELL=${1:? "Usage: preprocess_rep_TK_25.sh <cell_label> <resolution_bp> <juicer_tools.jar> [lr_cell_label] [workspace_root] [genome]"}
resolution=${2:? "Provide the resolution (e.g., 10000)"}
juicer_tool=${3:? "Provide the path to juicer_tools .jar"}
LR_CELL=${4:-${CELL}_LR_rep}
WORKSPACE_ROOT=${5:-"/data/tomoya/ExperimentsTK/GM12878_test/juicer_ready"}
GENOME=${6:-"hg19"}

# Set up timestamped logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$(dirname "${WORKSPACE_ROOT}")/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/preprocess_${TIMESTAMP}.log"

# Redirect all output to both terminal and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

# Log header
echo "=== Preprocessing started at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo ""
ratio=16  # DFHiC-compatible downsampling ratio
res_label="$((resolution / 1000))k"

HR_DIR="${WORKSPACE_ROOT}/${CELL}"
LR_DIR="${WORKSPACE_ROOT}/${LR_CELL}"
HR_INPUT="${HR_DIR}/total_merged_nodups.txt"
LR_INPUT="${LR_DIR}/total_merged_nodups_downsample_ratio_${ratio}.txt"

chromes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)

echo "[INFO] Workspace root  : ${WORKSPACE_ROOT}"
echo "[INFO] HR cell folder  : ${HR_DIR}"
echo "[INFO] LR cell folder  : ${LR_DIR}"
echo "[INFO] Resolution (bp) : ${resolution}"
echo "[INFO] Genome assembly : ${GENOME}"
echo "[INFO] Juicer jar      : ${juicer_tool}"

# Create output directories
mkdir -p "${HR_DIR}/intra_NONE" "${HR_DIR}/intra_HR" "${HR_DIR}/intra_LR"
mkdir -p "${LR_DIR}/intra_NONE" "${LR_DIR}/intra_HR" "${LR_DIR}/intra_LR"

# Validate HR input file exists
if [[ ! -f "${HR_INPUT}" ]]; then
    echo "[ERROR] Missing ${HR_INPUT}. Copy the merged/highcov file first." >&2
    exit 1
fi

# Create downsampled contact list if needed
if [[ ! -f "${LR_INPUT}" ]]; then
    echo "[INFO] Building downsampled contact list at ${LR_INPUT}"
    total_lines=$(wc -l < "${HR_INPUT}")
    num_downsample=$(( total_lines / ratio ))
    shuf -n "${num_downsample}" "${HR_INPUT}" | sort -k3,3V -k7,7V -k4,4n -k8,8n -T "${SORT_TMPDIR:-/data/tomoya/tmp}" > "${LR_INPUT}"
    echo "[INFO] Downsample complete (kept ${num_downsample} of ${total_lines} rows)."
else
    echo "[INFO] Found existing downsampled file at ${LR_INPUT}; reusing it."
fi

# Generate .hic files using Juicer tools
echo "[INFO] Generating high-coverage .hic at ${HR_DIR}/total_merged.hic"
java -Xmx200g -jar "${juicer_tool}" pre "${HR_INPUT}" "${HR_DIR}/total_merged.hic" "${GENOME}"

echo "[INFO] Generating downsampled .hic at ${LR_DIR}/total_merged_downsample_ratio_${ratio}.hic"
java -Xmx200g -jar "${juicer_tool}" pre "${LR_INPUT}" "${LR_DIR}/total_merged_downsample_ratio_${ratio}.hic" "${GENOME}"

# Dump per-chromosome contact maps (HR and LR)
for chrom in "${chromes[@]}"; do
    echo "[INFO] Processing chr${chrom}"

    # HR observed counts
    java -jar "${juicer_tool}" dump observed NONE "${HR_DIR}/total_merged.hic" \
        "${chrom}" "${chrom}" BP "${resolution}" \
        "${HR_DIR}/intra_NONE/chr${chrom}_${res_label}_intra_NONE.txt"

    # HR dense tiles (for DFHiC)
    java -jar "${juicer_tool}" dump observed NONE "${HR_DIR}/total_merged.hic" \
        "${chrom}" "${chrom}" BP "${resolution}" \
        "${HR_DIR}/intra_HR/HR_${res_label}_NONE.chr${chrom}" -d

    # LR observed counts
    java -jar "${juicer_tool}" dump observed NONE \
        "${LR_DIR}/total_merged_downsample_ratio_${ratio}.hic" \
        "${chrom}" "${chrom}" BP "${resolution}" \
        "${LR_DIR}/intra_NONE/chr${chrom}_${res_label}_intra_NONE_downsample_ratio${ratio}.txt"

    # LR dense tiles
    java -jar "${juicer_tool}" dump observed NONE \
        "${LR_DIR}/total_merged_downsample_ratio_${ratio}.hic" \
        "${chrom}" "${chrom}" BP "${resolution}" \
        "${LR_DIR}/intra_LR/LR_${res_label}_NONE.chr${chrom}" -d
done

echo "[INFO] Completed preprocessing for ${CELL} (HR) and ${LR_CELL} (LR)."
echo ""
echo "=== Preprocessing completed at $(date '+%Y-%m-%d %H:%M:%S') ==="
