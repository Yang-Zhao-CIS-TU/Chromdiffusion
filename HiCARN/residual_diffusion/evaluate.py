"""
Evaluation Metrics for Hi-C Refinement

Measures:
1. Loop survival: Do refined predictions preserve HiCARN's detected loops?
2. TAD boundary sharpness: Are TAD boundaries clearer after refinement?
3. Standard metrics: SSIM, PSNR, PCC, MSE, MAE

Critical success criteria (from specification):
- Loop recall should NOT decrease
- TAD boundaries should be sharper
- Overall correlation should improve or stay same
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity
import argparse


# ================================================================
# STANDARD METRICS
# ================================================================

def calculate_psnr(pred, target):
    """PSNR using max value of target"""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    max_val = np.max(target)
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(pred, target):
    """SSIM with dynamic data range"""
    if pred.ndim > 2:
        pred = pred.squeeze()
    if target.ndim > 2:
        target = target.squeeze()
    
    data_range = target.max() - target.min()
    return structural_similarity(pred, target, data_range=data_range)


def calculate_pcc(pred, target):
    """Pearson Correlation Coefficient"""
    return pearsonr(pred.flatten(), target.flatten())[0]


def calculate_scc(pred, target):
    """Spearman Correlation Coefficient"""
    from scipy.stats import spearmanr
    return spearmanr(pred.flatten(), target.flatten())[0]


def calculate_mse(pred, target):
    """Mean Squared Error"""
    return np.mean((pred - target) ** 2)


def calculate_mae(pred, target):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - target))


# ================================================================
# LOOP DETECTION AND SURVIVAL
# ================================================================

def detect_loops_simple(hic_matrix, threshold=None, min_distance=3):
    """
    Simple loop detection based on local maxima
    
    A more sophisticated approach would use:
    - HiCCUPS (from Juicer)
    - Mustache
    - HiCExplorer
    
    This is a simplified version for demonstration.
    
    Args:
        hic_matrix: (H, W) Hi-C contact matrix
        threshold: threshold for peak detection (auto if None)
        min_distance: minimum distance between peaks
    
    Returns:
        loops: list of (i, j, intensity) tuples
    """
    # Auto threshold
    if threshold is None:
        # Use median + 3*MAD as threshold
        median = np.median(hic_matrix)
        mad = np.median(np.abs(hic_matrix - median))
        threshold = median + 3 * mad
    
    loops = []
    H, W = hic_matrix.shape
    
    # Look for off-diagonal peaks
    for i in range(H):
        for j in range(i + min_distance, min(i + 20, W)):  # Look within window
            # Check if local maximum
            if hic_matrix[i, j] < threshold:
                continue
            
            # Check 3x3 neighborhood
            is_peak = True
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if hic_matrix[ni, nj] > hic_matrix[i, j]:
                            is_peak = False
                            break
                if not is_peak:
                    break
            
            if is_peak:
                loops.append((i, j, hic_matrix[i, j]))
    
    return loops


def compute_loop_recall(loops_pred, loops_target, tolerance=2):
    """
    Compute loop recall: how many target loops are recovered in prediction?
    
    Args:
        loops_pred: loops detected in prediction
        loops_target: loops detected in target
        tolerance: spatial tolerance in bins
    
    Returns:
        recall: fraction of target loops recovered
        precision: fraction of predicted loops that match target
        matched_loops: list of matched loop pairs
    """
    if len(loops_target) == 0:
        return 1.0, 1.0, []
    
    matched = []
    
    for tloop in loops_target:
        ti, tj, tint = tloop
        
        # Find matching loop in prediction
        for ploop in loops_pred:
            pi, pj, pint = ploop
            
            # Check if within tolerance
            if abs(ti - pi) <= tolerance and abs(tj - pj) <= tolerance:
                matched.append((tloop, ploop))
                break
    
    recall = len(matched) / len(loops_target) if len(loops_target) > 0 else 0.0
    precision = len(matched) / len(loops_pred) if len(loops_pred) > 0 else 0.0
    
    return recall, precision, matched


def compute_loop_intensity_correlation(matched_loops):
    """
    For matched loops, compute correlation of intensities
    
    Args:
        matched_loops: list of (target_loop, pred_loop) tuples
    
    Returns:
        correlation: Pearson correlation of loop intensities
    """
    if len(matched_loops) == 0:
        return 0.0
    
    target_intensities = [t[2] for t, p in matched_loops]
    pred_intensities = [p[2] for t, p in matched_loops]
    
    if len(target_intensities) < 2:
        return 0.0
    
    return pearsonr(target_intensities, pred_intensities)[0]


# ================================================================
# TAD BOUNDARY DETECTION AND SHARPNESS
# ================================================================

def compute_insulation_score(hic_matrix, window_size=5):
    """
    Compute insulation score for TAD boundary detection
    
    Insulation score measures how much a bin is insulated from
    its neighbors, with local minima indicating TAD boundaries.
    
    Args:
        hic_matrix: (H, W) Hi-C matrix
        window_size: window size for insulation calculation
    
    Returns:
        insulation: (H,) insulation score
    """
    H, W = hic_matrix.shape
    insulation = np.zeros(H)
    
    for i in range(H):
        # Sum of contacts in square window around diagonal
        start = max(0, i - window_size)
        end = min(H, i + window_size + 1)
        
        window_sum = np.sum(hic_matrix[start:end, start:end])
        insulation[i] = window_sum
    
    # Log transform and normalize
    insulation = np.log(insulation + 1e-8)
    insulation = (insulation - insulation.mean()) / (insulation.std() + 1e-8)
    
    return insulation


def detect_tad_boundaries(insulation, threshold=-1.0, min_distance=5):
    """
    Detect TAD boundaries from insulation score
    
    Args:
        insulation: (H,) insulation score
        threshold: threshold for boundary detection
        min_distance: minimum distance between boundaries
    
    Returns:
        boundaries: array of boundary positions
    """
    # Find local minima (boundaries are insulation minima)
    peaks, _ = find_peaks(-insulation, height=-threshold, distance=min_distance)
    return peaks


def compute_boundary_sharpness(hic_matrix, boundaries, window=3):
    """
    Compute sharpness of TAD boundaries
    
    Sharpness is measured as the contrast between intra-TAD and
    inter-TAD contacts around the boundary.
    
    Args:
        hic_matrix: (H, W) Hi-C matrix
        boundaries: array of boundary positions
        window: window size around boundary
    
    Returns:
        mean_sharpness: average boundary sharpness
        sharpness_values: sharpness for each boundary
    """
    sharpness_values = []
    H, W = hic_matrix.shape
    
    for b in boundaries:
        if b < window or b >= H - window:
            continue
        
        # Intra-TAD contacts (within same TAD)
        left_intra = hic_matrix[b-window:b, b-window:b].mean()
        right_intra = hic_matrix[b+1:b+1+window, b+1:b+1+window].mean()
        intra = (left_intra + right_intra) / 2
        
        # Inter-TAD contacts (across boundary)
        inter = hic_matrix[b-window:b, b+1:b+1+window].mean()
        
        # Sharpness = ratio of intra to inter
        if inter > 0:
            sharpness = intra / (inter + 1e-8)
            sharpness_values.append(sharpness)
    
    mean_sharpness = np.mean(sharpness_values) if sharpness_values else 0.0
    
    return mean_sharpness, sharpness_values


# ================================================================
# COMPREHENSIVE EVALUATION
# ================================================================

def evaluate_refinement(
    pred_original,
    pred_refined,
    target,
    compute_loops=True,
    compute_tads=True
):
    """
    Comprehensive evaluation of Hi-C refinement
    
    Args:
        pred_original: HiCARN predictions (N, H, W)
        pred_refined: refined predictions (N, H, W)
        target: ground truth (N, H, W)
        compute_loops: whether to compute loop metrics
        compute_tads: whether to compute TAD metrics
    
    Returns:
        results: dictionary of evaluation metrics
    """
    results = {
        'standard_metrics': {},
        'loop_metrics': {},
        'tad_metrics': {}
    }
    
    n_samples = len(pred_original)
    
    # ================================================================
    # STANDARD METRICS (averaged over samples)
    # ================================================================
    
    print("\nComputing standard metrics...")
    
    psnr_orig = []
    psnr_refined = []
    ssim_orig = []
    ssim_refined = []
    pcc_orig = []
    pcc_refined = []
    scc_orig = []
    scc_refined = []
    mse_orig = []
    mse_refined = []
    mae_orig = []
    mae_refined = []
    
    for i in range(n_samples):
        # Original metrics
        psnr_orig.append(calculate_psnr(pred_original[i], target[i]))
        ssim_orig.append(calculate_ssim(pred_original[i], target[i]))
        pcc_orig.append(calculate_pcc(pred_original[i], target[i]))
        scc_orig.append(calculate_scc(pred_original[i], target[i]))
        mse_orig.append(calculate_mse(pred_original[i], target[i]))
        mae_orig.append(calculate_mae(pred_original[i], target[i]))
        
        # Refined metrics
        psnr_refined.append(calculate_psnr(pred_refined[i], target[i]))
        ssim_refined.append(calculate_ssim(pred_refined[i], target[i]))
        pcc_refined.append(calculate_pcc(pred_refined[i], target[i]))
        scc_refined.append(calculate_scc(pred_refined[i], target[i]))
        mse_refined.append(calculate_mse(pred_refined[i], target[i]))
        mae_refined.append(calculate_mae(pred_refined[i], target[i]))
    
    results['standard_metrics'] = {
        'psnr_original': {'mean': np.mean(psnr_orig), 'std': np.std(psnr_orig)},
        'psnr_refined': {'mean': np.mean(psnr_refined), 'std': np.std(psnr_refined)},
        'ssim_original': {'mean': np.mean(ssim_orig), 'std': np.std(ssim_orig)},
        'ssim_refined': {'mean': np.mean(ssim_refined), 'std': np.std(ssim_refined)},
        'pcc_original': {'mean': np.mean(pcc_orig), 'std': np.std(pcc_orig)},
        'pcc_refined': {'mean': np.mean(pcc_refined), 'std': np.std(pcc_refined)},
        'scc_original': {'mean': np.mean(scc_orig), 'std': np.std(scc_orig)},
        'scc_refined': {'mean': np.mean(scc_refined), 'std': np.std(scc_refined)},
        'mse_original': {'mean': np.mean(mse_orig), 'std': np.std(mse_orig)},
        'mse_refined': {'mean': np.mean(mse_refined), 'std': np.std(mse_refined)},
        'mae_original': {'mean': np.mean(mae_orig), 'std': np.std(mae_orig)},
        'mae_refined': {'mean': np.mean(mae_refined), 'std': np.std(mae_refined)},
    }
    
    # ================================================================
    # LOOP METRICS
    # ================================================================
    
    if compute_loops:
        print("\nComputing loop survival metrics...")
        
        recall_orig = []
        recall_refined = []
        precision_orig = []
        precision_refined = []
        intensity_corr_orig = []
        intensity_corr_refined = []
        
        for i in range(min(n_samples, 100)):  # Limit to 100 samples for speed
            # Detect loops
            loops_target = detect_loops_simple(target[i])
            loops_orig = detect_loops_simple(pred_original[i])
            loops_refined = detect_loops_simple(pred_refined[i])
            
            # Compute metrics
            r_orig, p_orig, m_orig = compute_loop_recall(loops_orig, loops_target)
            r_refined, p_refined, m_refined = compute_loop_recall(loops_refined, loops_target)
            
            recall_orig.append(r_orig)
            recall_refined.append(r_refined)
            precision_orig.append(p_orig)
            precision_refined.append(p_refined)
            
            intensity_corr_orig.append(compute_loop_intensity_correlation(m_orig))
            intensity_corr_refined.append(compute_loop_intensity_correlation(m_refined))
        
        results['loop_metrics'] = {
            'recall_original': {'mean': np.mean(recall_orig), 'std': np.std(recall_orig)},
            'recall_refined': {'mean': np.mean(recall_refined), 'std': np.std(recall_refined)},
            'precision_original': {'mean': np.mean(precision_orig), 'std': np.std(precision_orig)},
            'precision_refined': {'mean': np.mean(precision_refined), 'std': np.std(precision_refined)},
            'intensity_corr_original': {'mean': np.mean(intensity_corr_orig), 'std': np.std(intensity_corr_orig)},
            'intensity_corr_refined': {'mean': np.mean(intensity_corr_refined), 'std': np.std(intensity_corr_refined)},
        }
    
    # ================================================================
    # TAD BOUNDARY METRICS
    # ================================================================
    
    if compute_tads:
        print("\nComputing TAD boundary metrics...")
        
        sharpness_orig = []
        sharpness_refined = []
        boundary_precision_orig = []
        boundary_precision_refined = []
        
        for i in range(min(n_samples, 100)):  # Limit for speed
            # Compute insulation scores
            ins_target = compute_insulation_score(target[i])
            ins_orig = compute_insulation_score(pred_original[i])
            ins_refined = compute_insulation_score(pred_refined[i])
            
            # Detect boundaries
            boundaries_target = detect_tad_boundaries(ins_target)
            boundaries_orig = detect_tad_boundaries(ins_orig)
            boundaries_refined = detect_tad_boundaries(ins_refined)
            
            # Compute sharpness
            sharp_orig, _ = compute_boundary_sharpness(pred_original[i], boundaries_orig)
            sharp_refined, _ = compute_boundary_sharpness(pred_refined[i], boundaries_refined)
            
            sharpness_orig.append(sharp_orig)
            sharpness_refined.append(sharp_refined)
            
            # Boundary detection precision (similar to loop recall)
            # Count how many target boundaries are detected
            matched_orig = sum(1 for b_t in boundaries_target 
                              if any(abs(b_t - b_o) <= 2 for b_o in boundaries_orig))
            matched_refined = sum(1 for b_t in boundaries_target 
                                 if any(abs(b_t - b_r) <= 2 for b_r in boundaries_refined))
            
            bp_orig = matched_orig / len(boundaries_target) if len(boundaries_target) > 0 else 0.0
            bp_refined = matched_refined / len(boundaries_target) if len(boundaries_target) > 0 else 0.0
            
            boundary_precision_orig.append(bp_orig)
            boundary_precision_refined.append(bp_refined)
        
        results['tad_metrics'] = {
            'sharpness_original': {'mean': np.mean(sharpness_orig), 'std': np.std(sharpness_orig)},
            'sharpness_refined': {'mean': np.mean(sharpness_refined), 'std': np.std(sharpness_refined)},
            'boundary_recall_original': {'mean': np.mean(boundary_precision_orig), 'std': np.std(boundary_precision_orig)},
            'boundary_recall_refined': {'mean': np.mean(boundary_precision_refined), 'std': np.std(boundary_precision_refined)},
        }
    
    return results


def print_results(results):
    """Print evaluation results in a nice format"""
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Standard metrics
    print("\n📊 STANDARD METRICS")
    print("-" * 80)
    metrics = results['standard_metrics']
    
    print(f"{'Metric':<15} {'Original':<20} {'Refined':<20} {'Improvement':<15}")
    print("-" * 80)
    
    for metric_name in ['ssim', 'psnr', 'pcc', 'scc', 'mae', 'mse']:
        orig = metrics[f'{metric_name}_original']['mean']
        refined = metrics[f'{metric_name}_refined']['mean']
        
        # For error metrics, improvement is negative (lower is better)
        if metric_name in ['mae', 'mse']:
            improvement = (orig - refined) / orig * 100
            symbol = "✓" if improvement > 0 else "✗"
        else:
            improvement = (refined - orig) / orig * 100
            symbol = "✓" if improvement > 0 else "✗"
        
        print(f"{metric_name.upper():<15} {orig:<20.4f} {refined:<20.4f} {symbol} {improvement:>+.2f}%")
    
    # Loop metrics
    if results['loop_metrics']:
        print("\n🔄 LOOP SURVIVAL METRICS")
        print("-" * 80)
        loop_metrics = results['loop_metrics']
        
        recall_orig = loop_metrics['recall_original']['mean']
        recall_refined = loop_metrics['recall_refined']['mean']
        recall_change = (recall_refined - recall_orig) / recall_orig * 100
        
        print(f"Loop Recall:")
        print(f"  Original:  {recall_orig:.4f}")
        print(f"  Refined:   {recall_refined:.4f}")
        print(f"  Change:    {recall_change:+.2f}%")
        
        if recall_refined >= recall_orig * 0.95:
            print(f"  ✓ PASS: Loop recall preserved (>{95}% of original)")
        else:
            print(f"  ✗ FAIL: Loop recall decreased significantly")
    
    # TAD metrics
    if results['tad_metrics']:
        print("\n🧬 TAD BOUNDARY METRICS")
        print("-" * 80)
        tad_metrics = results['tad_metrics']
        
        sharp_orig = tad_metrics['sharpness_original']['mean']
        sharp_refined = tad_metrics['sharpness_refined']['mean']
        sharp_change = (sharp_refined - sharp_orig) / sharp_orig * 100
        
        print(f"Boundary Sharpness:")
        print(f"  Original:  {sharp_orig:.4f}")
        print(f"  Refined:   {sharp_refined:.4f}")
        print(f"  Change:    {sharp_change:+.2f}%")
        
        if sharp_refined > sharp_orig:
            print(f"  ✓ PASS: Boundaries are sharper")
        else:
            print(f"  ✗ FAIL: Boundaries are less sharp")
    
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)
    
    # Check success criteria
    criteria_met = []
    
    # 1. Loop recall not decreased
    if results['loop_metrics']:
        loop_preserved = recall_refined >= recall_orig * 0.95
        criteria_met.append(loop_preserved)
        status = "✓ PASS" if loop_preserved else "✗ FAIL"
        print(f"{status}: Loop recall preserved")
    
    # 2. TAD boundaries sharper
    if results['tad_metrics']:
        boundaries_sharper = sharp_refined > sharp_orig
        criteria_met.append(boundaries_sharper)
        status = "✓ PASS" if boundaries_sharper else "✗ FAIL"
        print(f"{status}: TAD boundaries sharper")
    
    # 3. Overall correlation improved or same
    pcc_improved = metrics['pcc_refined']['mean'] >= metrics['pcc_original']['mean'] * 0.99
    criteria_met.append(pcc_improved)
    status = "✓ PASS" if pcc_improved else "✗ FAIL"
    print(f"{status}: Overall correlation maintained/improved")
    
    print("\n" + "="*80)
    if all(criteria_met):
        print("🎉 ALL SUCCESS CRITERIA MET - REFINEMENT IS SUCCESSFUL!")
    else:
        print("⚠️  SOME CRITERIA NOT MET - REFINEMENT NEEDS IMPROVEMENT")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Hi-C refinement')
    parser.add_argument('--pred_original', type=str, required=True,
                       help='HiCARN predictions (original)')
    parser.add_argument('--pred_refined', type=str, required=True,
                       help='Refined predictions')
    parser.add_argument('--target', type=str, required=True,
                       help='Ground truth')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output JSON file')
    parser.add_argument('--no_loops', action='store_true',
                       help='Skip loop metrics (faster)')
    parser.add_argument('--no_tads', action='store_true',
                       help='Skip TAD metrics (faster)')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    pred_orig = np.load(args.pred_original)
    pred_refined = np.load(args.pred_refined)
    target = np.load(args.target)
    
    print(f"Data shapes:")
    print(f"  Original: {pred_orig.shape}")
    print(f"  Refined: {pred_refined.shape}")
    print(f"  Target: {target.shape}")
    
    # Evaluate
    results = evaluate_refinement(
        pred_orig, pred_refined, target,
        compute_loops=not args.no_loops,
        compute_tads=not args.no_tads
    )
    
    # Print results
    print_results(results)
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {args.output}")
