#!/bin/bash
# Ground truth TAD and Loop calling for HR and LR Hi-C data (chr18-22)
# Outputs can be compared against model predictions
#
# Usage: ground_truth_tad_loop.sh [cell_test_dir] [cell_label] [lr_label] [ratio]
# Examples:
#   ground_truth_tad_loop.sh                                                    # GM12878 defaults
#   ground_truth_tad_loop.sh /data/tomoya/ExperimentsTK/K562_test K562
#   ground_truth_tad_loop.sh /data/tomoya/ExperimentsTK/NHEK_test NHEK NHEK_LR_rep 16

CELL_TEST_DIR=${1:-"/data/tomoya/ExperimentsTK/GM12878_test"}
CELL=${2:-"GM12878"}
LR_CELL=${3:-"${CELL}_LR_rep"}
RATIO=${4:-16}

JAR="${JUICER_JAR:-/data/tomoya/ExperimentsTK/juicer_tools_1.22.01.jar}"
HR_HIC="${CELL_TEST_DIR}/juicer_ready/${CELL}/total_merged.hic"
LR_HIC="${CELL_TEST_DIR}/juicer_ready/${LR_CELL}/total_merged_downsample_ratio_${RATIO}.hic"
OUTPUT_DIR="${CELL_TEST_DIR}/tensors/predictions/ground_truth"

CHR_NUMS=(18 19 20 21 22)

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=== Ground Truth TAD and Loop Calling (${CELL}) ==="
echo "HR: $HR_HIC"
echo "LR: $LR_HIC"
echo "Output: $OUTPUT_DIR"
echo ""

for CHR_NUM in "${CHR_NUMS[@]}"; do
    echo "--- Processing chr${CHR_NUM} ---"

    # HR Arrowhead (TADs)
    echo "  [HR] Arrowhead TAD calling..."
    java -Xmx200g -jar "$JAR" arrowhead -r 10000 -k KR \
        "$HR_HIC" -c chr${CHR_NUM} \
        HR_arrowhead_chr${CHR_NUM} --threads 16

    # HR HiCCUPS (Loops)
    echo "  [HR] HiCCUPS loop calling..."
    java -Xmx200g -jar "$JAR" hiccups -r 10000 -k KR \
        "$HR_HIC" -c chr${CHR_NUM} \
        HR_hiccups_chr${CHR_NUM}

    # LR Arrowhead (TADs)
    echo "  [LR] Arrowhead TAD calling..."
    java -Xmx200g -jar "$JAR" arrowhead -r 10000 -k KR \
        "$LR_HIC" -c chr${CHR_NUM} \
        LR_arrowhead_chr${CHR_NUM} --threads 16

    # LR HiCCUPS (Loops)
    echo "  [LR] HiCCUPS loop calling..."
    java -Xmx200g -jar "$JAR" hiccups -r 10000 -k KR \
        "$LR_HIC" -c chr${CHR_NUM} \
        LR_hiccups_chr${CHR_NUM}

    echo "  Done with chr${CHR_NUM}"
    echo ""
done

echo "=== All chromosomes processed ==="
echo "HR TADs:  HR_arrowhead_chr*/10000_blocks.bedpe"
echo "HR Loops: HR_hiccups_chr*/merged_loops.bedpe"
echo "LR TADs:  LR_arrowhead_chr*/10000_blocks.bedpe"
echo "LR Loops: LR_hiccups_chr*/merged_loops.bedpe"
