#!/bin/bash
################################################################################
# HiCARN vs HiCARN+Diffusion TAD/Loop Evaluation Workflow
################################################################################
#
# This script runs the complete evaluation pipeline:
#   1. Denormalize predictions (normalized → raw counts)
#   2. Convert to Juicer format (using prediction_convert_combined_TK.py)
#   3. Build .hic files (using juicer_tools pre)
#   4. Run TAD calling (Arrowhead) and loop calling (HiCCUPS)
#   5. Benchmark against ground truth (using benchmark_combined_TK.py)
#   6. Generate comparison report
#
# Usage:
#   bash run_tad_loop_evaluation.sh
#
# Prerequisites:
#   - prediction_convert_combined_TK.py
#   - feature_calling_combined_TK.sh (or equivalent)
#   - benchmark_combined_TK.py
#   - juicer_tools.jar
#   - evaluate_raw_space_v2.py (for denormalization)
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

################################################################################
# CONFIGURATION
################################################################################

# Input files
HICARN_PRED="hicarn_predictions/predictions_norm.npy"
REFINED_PRED="refined_predictions/refined_chr22.npy"
GT_PRED="hicarn_predictions/ground_truth.npy"
PREPROCESSOR="hicarn_predictions/hicarn_preprocessor.pt"

# Reference data
JUICER_ROOT="/path/to/juicer_ready"  # Update this path
CHROM_SIZES="/path/to/hg19.chrom.sizes"  # Update this path
JUICER_TOOLS="/path/to/juicer_tools.jar"  # Update this path

# Output directory
OUTPUT_DIR="tad_loop_evaluation"

# Chromosomes to process
CHROMS="chr18 chr19 chr20 chr21 chr22"

# Parameters
RESOLUTION=10000
RATIO=16
CHUNK_SIZE=40
LOOP_TOLERANCE=5000

################################################################################
# SETUP
################################################################################

echo "========================================================================"
echo "HiCARN vs HiCARN+DIFFUSION TAD/LOOP EVALUATION"
echo "========================================================================"
echo "HiCARN predictions:  $HICARN_PRED"
echo "Refined predictions: $REFINED_PRED"
echo "Ground truth:        $GT_PRED"
echo "Output directory:    $OUTPUT_DIR"
echo "Chromosomes:         $CHROMS"
echo "========================================================================"

# Create output directories
mkdir -p $OUTPUT_DIR/{hicarn,refined,hr_gt,lr_baseline,results}
mkdir -p $OUTPUT_DIR/{hicarn,refined}/converted
mkdir -p $OUTPUT_DIR/{hicarn,refined}/hic_files
mkdir -p $OUTPUT_DIR/{hicarn,refined}/features

################################################################################
# STEP 1: DENORMALIZE PREDICTIONS
################################################################################

echo ""
echo "========================================================================"
echo "STEP 1: DENORMALIZING PREDICTIONS TO RAW SPACE"
echo "========================================================================"

# Denormalize HiCARN predictions
python3 << 'EOF'
import numpy as np
import torch
import sys

def denormalize(pred_path, preprocessor_path, output_path):
    # Load preprocessor
    preprocessor = torch.load(preprocessor_path, map_location='cpu')
    
    # Load normalized predictions
    pred_norm = np.load(pred_path)
    
    # Add channel dimension if needed
    if pred_norm.ndim == 3:
        pred_norm = pred_norm[:, None, :, :]
    
    # Denormalize
    pred_raw = preprocessor.postprocess(pred_norm)
    
    # Remove channel dimension
    if pred_raw.shape[1] == 1:
        pred_raw = pred_raw[:, 0, :, :]
    
    # Save - IMPORTANT: Save in NHWC format (N, H, W, 1) for conversion script
    pred_raw = pred_raw[:, :, :, np.newaxis]
    
    np.save(output_path, pred_raw)
    print(f"Saved: {output_path}")
    print(f"Shape: {pred_raw.shape}")
    print(f"Range: [{pred_raw.min():.2f}, {pred_raw.max():.2f}]")

# Denormalize all predictions
denormalize("$HICARN_PRED", "$PREPROCESSOR", "$OUTPUT_DIR/hicarn/hicarn_raw.npy")
denormalize("$REFINED_PRED", "$PREPROCESSOR", "$OUTPUT_DIR/refined/refined_raw.npy")
denormalize("$GT_PRED", "$PREPROCESSOR", "$OUTPUT_DIR/hr_gt/gt_raw.npy")
EOF

echo "✓ Denormalization complete"

################################################################################
# STEP 2: CONVERT TO JUICER FORMAT
################################################################################

echo ""
echo "========================================================================"
echo "STEP 2: CONVERTING TO JUICER FORMAT"
echo "========================================================================"

# Convert HiCARN predictions
echo "[HiCARN] Converting predictions..."
python prediction_convert_combined_TK.py \
    --input-dir $OUTPUT_DIR/hicarn \
    --input-format npy \
    --output-dir $OUTPUT_DIR/hicarn/converted \
    --juicer-root $JUICER_ROOT \
    --chrom-sizes $CHROM_SIZES \
    --chroms $CHROMS \
    --ratio $RATIO \
    --resolution $RESOLUTION \
    --chunk-size $CHUNK_SIZE

echo ""
echo "[Refined] Converting predictions..."
python prediction_convert_combined_TK.py \
    --input-dir $OUTPUT_DIR/refined \
    --input-format npy \
    --output-dir $OUTPUT_DIR/refined/converted \
    --juicer-root $JUICER_ROOT \
    --chrom-sizes $CHROM_SIZES \
    --chroms $CHROMS \
    --ratio $RATIO \
    --resolution $RESOLUTION \
    --chunk-size $CHUNK_SIZE

echo "✓ Conversion complete"

################################################################################
# STEP 3: BUILD .HIC FILES
################################################################################

echo ""
echo "========================================================================"
echo "STEP 3: BUILDING .HIC FILES"
echo "========================================================================"

# Function to build .hic file
build_hic() {
    local bedpe=$1
    local output_hic=$2
    
    if [ -f "$bedpe" ]; then
        echo "Building: $output_hic"
        java -Xmx8g -jar $JUICER_TOOLS pre \
            -r $RESOLUTION \
            $bedpe \
            $output_hic \
            $CHROM_SIZES
        
        if [ -f "$output_hic" ]; then
            echo "  ✓ Created: $output_hic"
        else
            echo "  ✗ Failed to create .hic file"
        fi
    else
        echo "  ✗ BEDPE not found: $bedpe"
    fi
}

# Build .hic files for each model and chromosome
for CHROM in $CHROMS; do
    echo ""
    echo "Processing $CHROM..."
    
    # HiCARN
    build_hic \
        "$OUTPUT_DIR/hicarn/converted/preds_lr_test_${CHROM}_ratio${RATIO}_convert.txt" \
        "$OUTPUT_DIR/hicarn/hic_files/hicarn_${CHROM}.hic"
    
    # Refined
    build_hic \
        "$OUTPUT_DIR/refined/converted/preds_lr_test_${CHROM}_ratio${RATIO}_convert.txt" \
        "$OUTPUT_DIR/refined/hic_files/refined_${CHROM}.hic"
done

echo "✓ .hic files built"

################################################################################
# STEP 4: RUN TAD AND LOOP CALLING
################################################################################

echo ""
echo "========================================================================"
echo "STEP 4: RUNNING TAD AND LOOP CALLING"
echo "========================================================================"

# Function to run Arrowhead (TAD calling)
run_arrowhead() {
    local hic_file=$1
    local output_dir=$2
    local chrom=$3
    
    echo "  [Arrowhead] $chrom"
    java -Xmx8g -jar $JUICER_TOOLS arrowhead \
        -r $RESOLUTION \
        -k KR \
        $hic_file \
        $output_dir/arrowhead_${chrom}
}

# Function to run HiCCUPS (loop calling)
run_hiccups() {
    local hic_file=$1
    local output_dir=$2
    local chrom=$3
    
    echo "  [HiCCUPS] $chrom"
    java -Xmx8g -jar $JUICER_TOOLS hiccups \
        -r $RESOLUTION \
        -k KR \
        $hic_file \
        $output_dir/hiccups_results_KR_${chrom}
}

# Run feature calling for each model
for MODEL in "hicarn" "refined"; do
    echo ""
    echo "[$MODEL] Running feature calling..."
    
    for CHROM in $CHROMS; do
        HIC_FILE="$OUTPUT_DIR/$MODEL/hic_files/${MODEL}_${CHROM}.hic"
        
        if [ -f "$HIC_FILE" ]; then
            echo "  Processing $CHROM..."
            run_arrowhead "$HIC_FILE" "$OUTPUT_DIR/$MODEL/features" "$CHROM" 2>/dev/null || echo "    Warning: Arrowhead failed"
            run_hiccups "$HIC_FILE" "$OUTPUT_DIR/$MODEL/features" "$CHROM" 2>/dev/null || echo "    Warning: HiCCUPS failed"
        else
            echo "  ✗ .hic file not found: $HIC_FILE"
        fi
    done
done

echo "✓ Feature calling complete"

################################################################################
# STEP 5: RUN BENCHMARKING
################################################################################

echo ""
echo "========================================================================"
echo "STEP 5: BENCHMARKING AGAINST GROUND TRUTH"
echo "========================================================================"

# You need to have ground truth TAD/loop calls
# This assumes you have run the same pipeline on HR ground truth data
GT_DIR="/path/to/ground_truth/features"  # Update this path

# Benchmark HiCARN
echo "[HiCARN] Running benchmark..."
python benchmark_combined_TK.py \
    --mode all \
    --pred-dir $OUTPUT_DIR/hicarn/features \
    --gt-dir $GT_DIR \
    --chroms $CHROMS \
    --tolerance $LOOP_TOLERANCE \
    --ratio $RATIO \
    --output $OUTPUT_DIR/results/hicarn_benchmark.csv

echo ""
echo "[Refined] Running benchmark..."
python benchmark_combined_TK.py \
    --mode all \
    --pred-dir $OUTPUT_DIR/refined/features \
    --gt-dir $GT_DIR \
    --chroms $CHROMS \
    --tolerance $LOOP_TOLERANCE \
    --ratio $RATIO \
    --output $OUTPUT_DIR/results/refined_benchmark.csv

echo "✓ Benchmarking complete"

################################################################################
# STEP 6: GENERATE COMPARISON REPORT
################################################################################

echo ""
echo "========================================================================"
echo "STEP 6: GENERATING COMPARISON REPORT"
echo "========================================================================"

python3 << 'EOF'
import pandas as pd
import numpy as np

# Load results
hicarn_df = pd.read_csv('$OUTPUT_DIR/results/hicarn_benchmark.csv', sep='\t')
refined_df = pd.read_csv('$OUTPUT_DIR/results/refined_benchmark.csv', sep='\t')

# Add model column
hicarn_df['Model'] = 'HiCARN'
refined_df['Model'] = 'HiCARN+Diffusion'

# Combine
combined = pd.concat([hicarn_df, refined_df], ignore_index=True)
combined.to_csv('$OUTPUT_DIR/results/comparison.csv', index=False, sep='\t')

print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)

# Calculate improvements for each metric type
for metric_type in ['Loop', 'TAD']:
    hicarn_rows = hicarn_df[hicarn_df['Type'].str.contains(metric_type, na=False)]
    refined_rows = refined_df[refined_df['Type'].str.contains(metric_type, na=False)]
    
    if len(hicarn_rows) == 0 or len(refined_rows) == 0:
        continue
    
    print(f"\n{metric_type} Performance:")
    print("-" * 70)
    
    for metric in ['F1', 'Jaccard']:
        if metric in hicarn_rows.columns:
            hicarn_avg = hicarn_rows[metric].mean()
            refined_avg = refined_rows[metric].mean()
            improvement = ((refined_avg - hicarn_avg) / hicarn_avg * 100) if hicarn_avg > 0 else 0
            
            print(f"{metric:10s}: HiCARN = {hicarn_avg:.4f}, "
                  f"Refined = {refined_avg:.4f}, "
                  f"Improvement = {improvement:+.2f}%")

print("\n" + "="*70)
print("✓ Comparison report saved to: $OUTPUT_DIR/results/comparison.csv")
print("="*70)
EOF

################################################################################
# DONE
################################################################################

echo ""
echo "========================================================================"
echo "EVALUATION COMPLETE!"
echo "========================================================================"
echo ""
echo "Results location: $OUTPUT_DIR/results/"
echo ""
echo "Files generated:"
echo "  - hicarn_benchmark.csv    (HiCARN performance)"
echo "  - refined_benchmark.csv   (HiCARN+Diffusion performance)"
echo "  - comparison.csv          (Side-by-side comparison)"
echo ""
echo "========================================================================"
