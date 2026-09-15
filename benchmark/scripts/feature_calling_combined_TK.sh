#!/bin/bash
#===============================================================================
# COMBINED HI-C FEATURE CALLING SCRIPT
#===============================================================================
#
# Converts prediction matrices to .hic format and calls chromatin features
# (TADs and loops) using Juicer tools and optionally FAN-C.
#
# THREE FEATURE CALLING MODULES:
#   1. HIC CONVERSION  - Convert prediction text → .hic format (Juicer pre)
#   2. TAD CALLING     - Arrowhead TAD detection on .hic files
#   3. LOOP CALLING    - HiCCUPS loop detection on .hic files
#   4. FANC BOUNDARIES - Alternative TAD calling via FAN-C insulation (optional)
#
# Usage:
#     ./feature_calling_combined_TK.sh --mode all --pred-dir <predictions>
#     ./feature_calling_combined_TK.sh --mode tad --pred-dir <predictions>
#     ./feature_calling_combined_TK.sh --mode loop --pred-dir <predictions>
#     ./feature_calling_combined_TK.sh --mode fanc --pred-dir <predictions>
#
# Author: Combined from convert_to_hic.sh, hiccups_loop.sh, fanc_boundary.sh
#===============================================================================

set -e  # Exit on error

#===============================================================================
# SECTION 1: DEFAULT PARAMETERS
#===============================================================================

# Chromosomes to process (default: test set)
CHROMOSOMES="18 19 20 21 22"

# Downsampling ratio (must match tensor builder)
RATIO=16

# Resolution in base pairs
RESOLUTION=10000

# Normalization method
NORMALIZATION="KR"

# Reference genome (hg19 for human, mm9 for mouse)
GENOME="hg19"

# Juicer tools JAR path
JUICER_JAR="${JUICER_JAR:-/data/tomoya/ExperimentsTK/juicer_tools_1.22.01.jar}"

# Java memory allocation
JAVA_MEM="200g"

# Arrowhead threads
ARROWHEAD_THREADS=16

# FAN-C insulation window size
FANC_WINDOW="100000"

# FAN-C boundary score threshold
FANC_SCORE="0.2"

# HiCCUPS FDR threshold (empty = use HiCCUPS default)
HICCUPS_FDR=""

# Processing mode (all, hic, tad, loop, fanc)
MODE="all"

# Prediction directory
PRED_DIR=""

#===============================================================================
# SECTION 2: ARGUMENT PARSING
#===============================================================================

print_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Required:
    --pred-dir DIR      Prediction directory containing .txt files

Options:
    --mode MODE         Processing mode: all, hic, tad, loop, fanc (default: all)
                          hic  = Convert text to .hic only
                          tad  = Arrowhead TAD calling (requires .hic)
                          loop = HiCCUPS loop calling (requires .hic)
                          fanc = FAN-C insulation boundaries (requires .hic)
                          all  = hic + tad + loop (not fanc)
    --chroms LIST       Space-separated chromosome numbers (default: "18 19 20 21 22")
    --ratio N           Downsample ratio (default: 16)
    --resolution N      Resolution in bp (default: 10000)
    --norm METHOD       Normalization: KR, VC, NONE (default: KR)
    --genome REF        Reference genome: hg19, hg38, mm9, mm10 (default: hg19)
    --jar PATH          Path to juicer_tools.jar
    --mem SIZE          Java memory allocation (default: 200g)
    --threads N         Arrowhead threads (default: 16)
    --fdr VALUE         HiCCUPS FDR threshold (default: HiCCUPS default ~0.1)
    --help              Show this help message

Examples:
    # Run full pipeline (hic + tad + loop)
    $(basename "$0") --mode all --pred-dir /path/to/predictions

    # Only convert to .hic format
    $(basename "$0") --mode hic --pred-dir /path/to/predictions

    # Run HiCCUPS on existing .hic files
    $(basename "$0") --mode loop --pred-dir /path/to/predictions

    # Run FAN-C alternative TAD calling
    $(basename "$0") --mode fanc --pred-dir /path/to/predictions
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pred-dir)
            PRED_DIR="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --chroms)
            CHROMOSOMES="$2"
            shift 2
            ;;
        --ratio)
            RATIO="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --norm)
            NORMALIZATION="$2"
            shift 2
            ;;
        --genome)
            GENOME="$2"
            shift 2
            ;;
        --jar)
            JUICER_JAR="$2"
            shift 2
            ;;
        --mem)
            JAVA_MEM="$2"
            shift 2
            ;;
        --threads)
            ARROWHEAD_THREADS="$2"
            shift 2
            ;;
        --fdr)
            HICCUPS_FDR="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$PRED_DIR" ]]; then
    echo "Error: --pred-dir is required"
    print_usage
    exit 1
fi

if [[ ! -d "$PRED_DIR" ]]; then
    echo "Error: Prediction directory not found: $PRED_DIR"
    exit 1
fi

# Validate mode
case $MODE in
    all|hic|tad|loop|fanc)
        ;;
    *)
        echo "Error: Invalid mode '$MODE'. Must be: all, hic, tad, loop, fanc"
        exit 1
        ;;
esac

#===============================================================================
# SECTION 3: HIC CONVERSION (Juicer pre)
#===============================================================================
# Converts prediction text files to .hic format for downstream analysis.
# Input:  preds_lr_test_chr${CHR}_ratio${RATIO}_convert.txt
# Output: preds_lr_test_chr${CHR}_ratio${RATIO}_convert.hic
#===============================================================================

run_hic_conversion() {
    local chr_num=$1
    local input_txt="${PRED_DIR}/preds_lr_test_chr${chr_num}_ratio${RATIO}_convert.txt"
    local output_hic="${PRED_DIR}/preds_lr_test_chr${chr_num}_ratio${RATIO}_convert.hic"

    if [[ ! -f "$input_txt" ]]; then
        echo "  Warning: Input file not found: $input_txt"
        return 1
    fi

    if [[ -f "$output_hic" ]]; then
        echo "  Skipping: .hic file already exists: $output_hic"
        return 0
    fi

    echo "  Converting to .hic format..."
    java -Xmx${JAVA_MEM} -jar "$JUICER_JAR" pre \
        -r "$RESOLUTION" \
        "$input_txt" \
        "$output_hic" \
        "$GENOME"

    echo "  Created: $output_hic"
}

#===============================================================================
# SECTION 4: TAD CALLING (Arrowhead)
#===============================================================================
# Detects Topologically Associating Domains using Arrowhead algorithm.
# Input:  preds_lr_test_chr${CHR}_ratio${RATIO}_convert.hic
# Output: preds_lr_test_chr${CHR}_ratio${RATIO}_convert_10kb/10000_blocks.bedpe
#===============================================================================

run_arrowhead() {
    local chr_num=$1
    local input_hic="${PRED_DIR}/preds_lr_test_chr${chr_num}_ratio${RATIO}_convert.hic"
    local output_dir="${PRED_DIR}/preds_lr_test_chr${chr_num}_ratio${RATIO}_convert_10kb"

    if [[ ! -f "$input_hic" ]]; then
        echo "  Warning: .hic file not found: $input_hic"
        echo "  Run with --mode hic first to generate .hic files"
        return 1
    fi

    if [[ -f "${output_dir}/10000_blocks.bedpe" ]]; then
        echo "  Skipping: Arrowhead output already exists: ${output_dir}"
        return 0
    fi

    echo "  Running Arrowhead TAD detection..."
    java -Xmx${JAVA_MEM} -jar "$JUICER_JAR" arrowhead \
        -r "$RESOLUTION" \
        -k "$NORMALIZATION" \
        --threads "$ARROWHEAD_THREADS" \
        "$input_hic" \
        "$output_dir"

    echo "  Created: ${output_dir}/10000_blocks.bedpe"
}

#===============================================================================
# SECTION 5: LOOP CALLING (HiCCUPS)
#===============================================================================
# Detects chromatin loops using HiCCUPS algorithm.
# Input:  preds_lr_test_chr${CHR}_ratio${RATIO}_convert.hic
# Output: hiccups_results_KR_chr${CHR}/merged_loops.bedpe
#===============================================================================

run_hiccups() {
    local chr_num=$1
    local input_hic="${PRED_DIR}/preds_lr_test_chr${chr_num}_ratio${RATIO}_convert.hic"

    # Adjust output dir name if custom FDR specified
    local output_dir="${PRED_DIR}/hiccups_results_${NORMALIZATION}_chr${chr_num}"
    if [[ -n "$HICCUPS_FDR" ]]; then
        output_dir="${output_dir}_fdr${HICCUPS_FDR}"
    fi

    if [[ ! -f "$input_hic" ]]; then
        echo "  Warning: .hic file not found: $input_hic"
        echo "  Run with --mode hic first to generate .hic files"
        return 1
    fi

    if [[ -f "${output_dir}/merged_loops.bedpe" ]]; then
        echo "  Skipping: HiCCUPS output already exists: ${output_dir}"
        return 0
    fi

    echo "  Running HiCCUPS loop detection..."
    if [[ -n "$HICCUPS_FDR" ]]; then
        java -Xmx${JAVA_MEM} -jar "$JUICER_JAR" hiccups \
            -r "$RESOLUTION" \
            -k "$NORMALIZATION" \
            -f "$HICCUPS_FDR" \
            -c "chr${chr_num}" \
            "$input_hic" \
            "$output_dir"
    else
        java -Xmx${JAVA_MEM} -jar "$JUICER_JAR" hiccups \
            -r "$RESOLUTION" \
            -k "$NORMALIZATION" \
            -c "chr${chr_num}" \
            "$input_hic" \
            "$output_dir"
    fi

    echo "  Created: ${output_dir}/merged_loops.bedpe"
}

#===============================================================================
# SECTION 6: FANC BOUNDARY CALLING (Alternative TAD method)
#===============================================================================
# Detects TAD boundaries using FAN-C insulation score method.
# Input:  preds_lr_test_chr${CHR}_ratio${RATIO}_convert.hic
# Output: fanc_insulation_KR/inter30_10kb_boundaries_100kb_chr${CHR}
#===============================================================================

run_fanc_boundaries() {
    local chr_num=$1
    local input_hic="${PRED_DIR}/preds_lr_test_chr${chr_num}_ratio${RATIO}_convert.hic"
    local fanc_dir="${PRED_DIR}/fanc_insulation_${NORMALIZATION}"
    local insulation_file="${fanc_dir}/inter30_10kb_insulation_chr${chr_num}"
    local boundary_file="${fanc_dir}/inter30_10kb_boundaries_100kb_chr${chr_num}"

    if [[ ! -f "$input_hic" ]]; then
        echo "  Warning: .hic file not found: $input_hic"
        echo "  Run with --mode hic first to generate .hic files"
        return 1
    fi

    # Create output directory
    mkdir -p "$fanc_dir"

    # Step 1: Calculate insulation scores (if not exists)
    if [[ -f "$insulation_file" ]]; then
        echo "  Skipping: Insulation file already exists"
    else
        echo "  Calculating insulation scores..."
        fanc insulation \
            "${input_hic}@10kb@${NORMALIZATION}" \
            "$insulation_file" \
            -w "$FANC_WINDOW" \
            -o bed
    fi

    # Step 2: Call boundaries
    echo "  Detecting TAD boundaries..."
    fanc boundaries \
        "$insulation_file" \
        "$boundary_file" \
        -w 100kb \
        -s "$FANC_SCORE"

    echo "  Created: $boundary_file"
}

#===============================================================================
# SECTION 7: MAIN EXECUTION
#===============================================================================

echo "========================================================================"
echo "COMBINED FEATURE CALLING PIPELINE"
echo "========================================================================"
echo "Mode:        $MODE"
echo "Pred dir:    $PRED_DIR"
echo "Chromosomes: $CHROMOSOMES"
echo "Ratio:       $RATIO"
echo "Resolution:  $RESOLUTION"
echo "Genome:      $GENOME"
echo "========================================================================"

# Change to prediction directory and update PRED_DIR to current dir
cd "$PRED_DIR" || exit 1
PRED_DIR="."

# Process each chromosome
for CHR_NUM in $CHROMOSOMES; do
    echo ""
    echo "Processing chromosome ${CHR_NUM}..."
    echo "----------------------------------------"

    # HIC CONVERSION
    if [[ "$MODE" == "all" || "$MODE" == "hic" ]]; then
        run_hic_conversion "$CHR_NUM"
    fi

    # TAD CALLING (Arrowhead)
    if [[ "$MODE" == "all" || "$MODE" == "tad" ]]; then
        run_arrowhead "$CHR_NUM"
    fi

    # LOOP CALLING (HiCCUPS)
    if [[ "$MODE" == "all" || "$MODE" == "loop" ]]; then
        run_hiccups "$CHR_NUM"
    fi

    # FANC BOUNDARIES (optional, not included in "all")
    if [[ "$MODE" == "fanc" ]]; then
        run_fanc_boundaries "$CHR_NUM"
    fi
done

echo ""
echo "========================================================================"
echo "FEATURE CALLING COMPLETE"
echo "========================================================================"
echo "Output location: $PRED_DIR"
echo ""

# Print summary of outputs based on mode
case $MODE in
    all)
        echo "Generated files:"
        echo "  - .hic files (Juicer format)"
        echo "  - *_10kb/10000_blocks.bedpe (Arrowhead TADs)"
        echo "  - hiccups_results_*/merged_loops.bedpe (HiCCUPS loops)"
        ;;
    hic)
        echo "Generated files:"
        echo "  - .hic files (Juicer format)"
        ;;
    tad)
        echo "Generated files:"
        echo "  - *_10kb/10000_blocks.bedpe (Arrowhead TADs)"
        ;;
    loop)
        echo "Generated files:"
        echo "  - hiccups_results_*/merged_loops.bedpe (HiCCUPS loops)"
        ;;
    fanc)
        echo "Generated files:"
        echo "  - fanc_insulation_*/inter30_10kb_insulation_* (insulation scores)"
        echo "  - fanc_insulation_*/inter30_10kb_boundaries_* (TAD boundaries)"
        ;;
esac

echo ""
echo "Next step: Run benchmark_combined_TK.py to evaluate results"
