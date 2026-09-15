#!/bin/bash
#===============================================================================
# CD_40 BATCH PROCESSING PIPELINE
#===============================================================================
# Runs the 3-stage post-processing pipeline on all 5 CD_40 prediction sets:
#   1. Create symlinks (chr{N}.npy → preds_lr_test_chr{N}_ratio16.npy)
#   2. Stage 6: prediction_convert_combined_TK.py (NPY → Juicer text)
#   3. Stage 7: feature_calling_combined_TK.sh (text → .hic → TADs/loops)
#   4. Stage 8: benchmark_combined_TK.py (benchmark vs ground truth)
#===============================================================================

set -euo pipefail  # Exit on error, unset vars, and pipeline failures

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREDICTIONS_DIR="/data/tomoya/ExperimentsTK/GM12878_test/tensors/predictions"
GT_DIR="${PREDICTIONS_DIR}/ground_truth"

# CD_40 batch directories
BATCHES=(
    "CD_40_loss_Jan16"
    "CD_40Loc_resdiff_15_pcc"
    "CD_40Loc_resdiff_15_iou"
    "CD_40Loc_resdiff_15_alpa8_pcc"
    "CD_40Loc_resdiff_15_alpa8_iou"
)

# Chromosomes
CHROMS="18 19 20 21 22"
RATIO=16

# Logging
LOG_DIR="${PREDICTIONS_DIR}/pipeline_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/CD40_pipeline_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

#===============================================================================
# STEP 1: CREATE SYMLINKS
#===============================================================================
create_symlinks() {
    local batch_dir="$1"
    log "Creating symlinks in ${batch_dir}..."

    cd "${PREDICTIONS_DIR}/${batch_dir}"

    for c in $CHROMS; do
        local src="chr${c}.npy"
        local dst="preds_lr_test_chr${c}_ratio${RATIO}.npy"

        if [[ -f "$src" ]]; then
            if [[ -L "$dst" ]]; then
                ln -sf "$src" "$dst"
                log "  Refreshed symlink: $dst -> $src"
            elif [[ -f "$dst" ]]; then
                local backup="${dst}.bak_${TIMESTAMP}"
                mv "$dst" "$backup"
                ln -s "$src" "$dst"
                log "  Replaced file with symlink: $dst -> $src (backup: $backup)"
            else
                ln -s "$src" "$dst"
                log "  Created: $dst -> $src"
            fi
        else
            log "  WARNING: Source file not found: $src"
        fi
    done
}

#===============================================================================
# STEP 2: STAGE 6 - PREDICTION CONVERSION
#===============================================================================
run_stage6() {
    local batch_dir="$1"
    local pred_path="${PREDICTIONS_DIR}/${batch_dir}"

    log "Stage 6: Converting predictions for ${batch_dir}..."

    if python "${SCRIPT_DIR}/prediction_convert_combined_TK.py" \
        --input-dir "$pred_path" \
        --input-format npy \
        --ratio "$RATIO" \
        2>&1 | tee -a "$MAIN_LOG"; then
        log "Stage 6 complete for ${batch_dir}"
    else
        local status=${PIPESTATUS[0]}
        log "ERROR: Stage 6 failed for ${batch_dir} (exit ${status})"
        return "$status"
    fi
}

#===============================================================================
# STEP 3: STAGE 7 - FEATURE CALLING
#===============================================================================
run_stage7() {
    local batch_dir="$1"
    local pred_path="${PREDICTIONS_DIR}/${batch_dir}"

    log "Stage 7: Calling features for ${batch_dir}..."

    if "${SCRIPT_DIR}/feature_calling_combined_TK.sh" \
        --mode all \
        --pred-dir "$pred_path" \
        --chroms "$CHROMS" \
        --ratio "$RATIO" \
        2>&1 | tee -a "$MAIN_LOG"; then
        log "Stage 7 complete for ${batch_dir}"
    else
        local status=${PIPESTATUS[0]}
        log "ERROR: Stage 7 failed for ${batch_dir} (exit ${status})"
        return "$status"
    fi
}

#===============================================================================
# STEP 4: STAGE 8 - BENCHMARKING
#===============================================================================
run_stage8() {
    local batch_dir="$1"
    local pred_path="${PREDICTIONS_DIR}/${batch_dir}"
    local output_file="${pred_path}/benchmark_results_${TIMESTAMP}.tsv"

    log "Stage 8: Benchmarking ${batch_dir}..."

    if python "${SCRIPT_DIR}/benchmark_combined_TK.py" \
        --mode all \
        --pred-dir "$pred_path" \
        --gt-dir "$GT_DIR" \
        --ratio "$RATIO" \
        --output "$output_file" \
        2>&1 | tee -a "$MAIN_LOG"; then
        log "Stage 8 complete for ${batch_dir}"
        log "Results saved to: ${output_file}"
    else
        local status=${PIPESTATUS[0]}
        log "ERROR: Stage 8 failed for ${batch_dir} (exit ${status})"
        return "$status"
    fi
}

#===============================================================================
# MAIN EXECUTION
#===============================================================================
log "========================================================================"
log "CD_40 BATCH PROCESSING PIPELINE"
log "========================================================================"
log "Processing ${#BATCHES[@]} batches: ${BATCHES[*]}"
log "Ground truth dir: ${GT_DIR}"
log "Log file: ${MAIN_LOG}"
log "========================================================================"

# Process each batch sequentially
for batch in "${BATCHES[@]}"; do
    log ""
    log "========================================================================"
    log "PROCESSING: ${batch}"
    log "========================================================================"

    # Check if directory exists
    if [[ ! -d "${PREDICTIONS_DIR}/${batch}" ]]; then
        log "ERROR: Directory not found: ${PREDICTIONS_DIR}/${batch}"
        continue
    fi

    # Run all stages
    create_symlinks "$batch"
    run_stage6 "$batch"
    run_stage7 "$batch"
    run_stage8 "$batch"

    log "COMPLETED: ${batch}"
done

log ""
log "========================================================================"
log "ALL BATCHES COMPLETE"
log "========================================================================"
log "Results are in each batch directory as benchmark_results_${TIMESTAMP}.tsv"
log "Full log: ${MAIN_LOG}"
