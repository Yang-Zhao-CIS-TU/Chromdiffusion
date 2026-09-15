#!/usr/bin/env python3
"""
================================================================================
COMBINED HI-C SUPER-RESOLUTION BENCHMARKING SCRIPT
================================================================================

Evaluates Hi-C super-resolution predictions by measuring preservation of
biologically meaningful chromatin structures.

THREE BENCHMARKING MODULES:
  1. LOOP BENCHMARKING   - Compares predicted loops vs HR ground truth
  2. TAD BENCHMARKING    - Compares predicted TADs vs HR ground truth
  3. LOOP VALIDATION     - Validates loops against ChIP-seq peaks (CTCF, RAD21, SMC3)

Usage:
    python benchmark_combined_TK.py --mode all \
        --pred-dir <predictions> --gt-dir <ground_truth> \
        --chipseq-dir <chipseq_peaks> --output <results.csv>

Author: Combined from Loop_jaccard_F1_TK.py, TAD_jaccard_F1_TK.py, Loop_validate.py
================================================================================
"""

import argparse
import os
import subprocess
import pandas as pd
from typing import Optional, List, Dict


# ==============================================================================
# SECTION 1: CORE UTILITY FUNCTIONS
# ==============================================================================

def load_loops(bedpe_path: str) -> pd.DataFrame:
    """
    Load loops from HiCCUPS BEDPE output.

    HiCCUPS outputs loops in BEDPE format with columns:
      #chr1, x1, x2, chr2, y1, y2, name, score, strand1, strand2, ...

    We extract the core coordinates (chr, x1, x2, y1, y2) for comparison.
    Row index 1 is skipped because it contains version/header info.

    Args:
        bedpe_path: Path to merged_loops.bedpe from HiCCUPS

    Returns:
        DataFrame with columns: chr, x1, x2, y1, y2 (sorted by position)
    """
    df = pd.read_csv(bedpe_path, sep='\t', skiprows=[1])
    loops = df[['#chr1', 'x1', 'x2', 'y1', 'y2']].copy()
    loops.columns = ['chr', 'x1', 'x2', 'y1', 'y2']
    loops = loops.sort_values(['chr', 'x1', 'y1'], ignore_index=True)
    return loops


def load_tads(bedpe_path: str) -> pd.DataFrame:
    """
    Load TADs from Arrowhead BEDPE output.

    Arrowhead outputs TADs as diagonal blocks in BEDPE format.
    For TADs, x1/x2 == y1/y2 since they're on the diagonal.
    We only need chr, start, end for TAD comparison.

    Args:
        bedpe_path: Path to 10000_blocks.bedpe from Arrowhead

    Returns:
        DataFrame with columns: chr, start, end (sorted by position)
    """
    df = pd.read_csv(bedpe_path, sep='\t', skiprows=[1])
    tads = df[['#chr1', 'x1', 'x2']].copy()
    tads.columns = ['chr', 'start', 'end']
    tads = tads.sort_values(['chr', 'start'], ignore_index=True)
    return tads


def expand_loop_coords(loops: pd.DataFrame, tolerance: int = 5000) -> pd.DataFrame:
    """
    Expand loop coordinates by ±tolerance for fuzzy matching.

    Args:
        loops: DataFrame with x1, x2, y1, y2 columns
        tolerance: Expansion in base pairs (default: 5000 = 5kb)

    Returns:
        DataFrame with additional min_x, max_x, min_y, max_y columns
    """
    loops = loops.copy()
    loops['min_x'] = loops['x1'] - tolerance
    loops['max_x'] = loops['x2'] + tolerance
    loops['min_y'] = loops['y1'] - tolerance
    loops['max_y'] = loops['y2'] + tolerance
    return loops


def save_as_bed(tads: pd.DataFrame, output_path: str) -> str:
    """
    Save TADs as BED file for bedtools compatibility.

    BED format: chr  start  end (tab-separated, no header)
    Required for bedtools jaccard calculation.
    """
    tads.to_csv(output_path, sep='\t', index=False, header=False)
    return output_path


# ==============================================================================
# SECTION 2: LOOP BENCHMARKING
# ==============================================================================
# Compares predicted loops against HR ground truth loops.
# Uses spatial overlap within tolerance window to define matching.
# ==============================================================================

def loop_has_overlap(row: pd.Series, other_df: pd.DataFrame) -> bool:
    """
    Check if a loop overlaps with any loop in another dataset.

    MATCHING LOGIC:
    Two loops match if their expanded bounding boxes overlap in BOTH dimensions.
    This is a 2D overlap check since loops are points in the contact matrix.

    For loop at (x, y) and candidate at (x', y'):
      Match if: |x - x'| <= tolerance AND |y - y'| <= tolerance

    Implemented as bounding box intersection check.
    """
    overlaps = other_df[
        (other_df['chr'] == row['chr']) &
        (other_df['min_x'] <= row['max_x']) & (other_df['max_x'] >= row['min_x']) &
        (other_df['min_y'] <= row['max_y']) & (other_df['max_y'] >= row['min_y'])
    ]
    return len(overlaps) > 0


def calculate_loop_metrics(pred_loops: pd.DataFrame,
                           hr_loops: pd.DataFrame,
                           tolerance: int = 5000) -> Dict:
    """
    Calculate F1 and Jaccard for loop predictions.

    Args:
        pred_loops: Predicted loops DataFrame
        hr_loops: Ground truth HR loops DataFrame
        tolerance: Matching tolerance in bp (default: 5000)

    Returns:
        Dict with TP, FP, FN, F1, Jaccard, and counts
    """
    # Expand coordinates for fuzzy matching
    pred_exp = expand_loop_coords(pred_loops, tolerance)
    hr_exp = expand_loop_coords(hr_loops, tolerance)

    # Count True Positives: predicted loops matching HR
    pred_has_match = pred_exp.apply(lambda r: loop_has_overlap(r, hr_exp), axis=1)
    tp = pred_has_match.sum()

    # Count False Positives: predicted loops NOT matching HR
    fp = len(pred_exp) - tp

    # Count False Negatives: HR loops NOT matched by predictions
    hr_has_match = hr_exp.apply(lambda r: loop_has_overlap(r, pred_exp), axis=1)
    fn = (~hr_has_match).sum()

    # Calculate F1 Score
    denominator = tp + 0.5 * (fp + fn)
    f1 = tp / denominator if denominator > 0 else 0.0

    # Calculate Jaccard Index (TK formula: pred-centric intersection)
    intersection = tp
    union = len(pred_loops) + len(hr_loops) - intersection
    jaccard = intersection / union if union > 0 else 0.0

    return {
        'pred_count': len(pred_loops),
        'hr_count': len(hr_loops),
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'F1': round(f1, 4),
        'Jaccard': round(jaccard, 4)
    }


def run_loop_benchmark(pred_dir: str, gt_dir: str,
                       chroms: List[str], tolerance: int = 5000,
                       include_lr: bool = False, fdr: str = None) -> pd.DataFrame:
    """
    Run loop benchmarking for all chromosomes.

    Args:
        pred_dir: Directory containing prediction HiCCUPS results
        gt_dir: Directory containing ground truth HiCCUPS results
        chroms: List of chromosomes to process
        tolerance: Matching tolerance in bp
        include_lr: Also compare LR baseline against HR

    Returns:
        DataFrame with per-chromosome metrics
    """
    results = []

    for chrom in chroms:
        print(f"[LOOP] Processing {chrom}...")

        # Build file paths (adjust for FDR suffix if specified)
        if fdr:
            pred_bedpe = os.path.join(pred_dir, f"hiccups_results_KR_{chrom}_fdr{fdr}/merged_loops.bedpe")
        else:
            pred_bedpe = os.path.join(pred_dir, f"hiccups_results_KR_{chrom}/merged_loops.bedpe")
        hr_bedpe = os.path.join(gt_dir, f"HR_hiccups_{chrom}/merged_loops.bedpe")
        lr_bedpe = os.path.join(gt_dir, f"LR_hiccups_{chrom}/merged_loops.bedpe")

        # Validate files exist
        if not os.path.isfile(pred_bedpe):
            print(f"  Warning: prediction not found: {pred_bedpe}")
            continue
        if not os.path.isfile(hr_bedpe):
            print(f"  Warning: HR ground truth not found: {hr_bedpe}")
            continue

        # Load loops
        pred_loops = load_loops(pred_bedpe)
        hr_loops = load_loops(hr_bedpe)

        # Calculate metrics: Prediction vs HR
        metrics = calculate_loop_metrics(pred_loops, hr_loops, tolerance)
        results.append({
            'Type': 'Loop',
            'Comparison': 'Pred_vs_HR',
            'Chromosome': chrom,
            **metrics
        })
        print(f"  Pred vs HR: F1={metrics['F1']:.4f}, Jaccard={metrics['Jaccard']:.4f}")

        # Optional: LR baseline comparison
        if include_lr and os.path.isfile(lr_bedpe):
            lr_loops = load_loops(lr_bedpe)
            if len(lr_loops) > 0:
                lr_metrics = calculate_loop_metrics(lr_loops, hr_loops, tolerance)
                results.append({
                    'Type': 'Loop',
                    'Comparison': 'LR_vs_HR',
                    'Chromosome': chrom,
                    **lr_metrics
                })
                print(f"  LR vs HR:   F1={lr_metrics['F1']:.4f}, Jaccard={lr_metrics['Jaccard']:.4f}")

    return pd.DataFrame(results)


# ==============================================================================
# SECTION 3: TAD BENCHMARKING
# ==============================================================================
# Compares predicted TADs against HR ground truth TADs.
# Uses both overlap-based F1 and bedtools Jaccard (genomic interval overlap).
# ==============================================================================

def tad_has_overlap(row: pd.Series, other_df: pd.DataFrame) -> bool:
    """
    Check if a TAD overlaps with any TAD in another dataset.

    TAD MATCHING LOGIC:
    Unlike loops (2D points), TADs are 1D genomic intervals.
    Two TADs match if their intervals overlap on the same chromosome.

    Standard interval overlap: A overlaps B if A.start <= B.end AND A.end >= B.start
    """
    overlaps = other_df[
        (other_df['chr'] == row['chr']) &
        (other_df['start'] <= row['end']) &
        (other_df['end'] >= row['start'])
    ]
    return len(overlaps) > 0


def calculate_tad_jaccard_bedtools(file_a: str, file_b: str) -> Optional[float]:
    """
    Calculate Jaccard index using bedtools (base-pair level overlap).

    Args:
        file_a: Path to first BED file (predictions)
        file_b: Path to second BED file (ground truth)

    Returns:
        Jaccard index as float, or None on error
    """
    # Sort files for bedtools compatibility
    sorted_a = file_a.replace(".bed", "_sorted.bed")
    sorted_b = file_b.replace(".bed", "_sorted.bed")

    subprocess.run(f"sort -k1,1 -k2,2n {file_a} > {sorted_a}", shell=True, check=True)
    subprocess.run(f"sort -k1,1 -k2,2n {file_b} > {sorted_b}", shell=True, check=True)

    # Run bedtools jaccard
    result = subprocess.run(
        f"bedtools jaccard -a {sorted_a} -b {sorted_b}",
        shell=True, capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  bedtools error: {result.stderr}")
        return None

    # Parse output: "intersection  union  jaccard  n_intersections"
    lines = result.stdout.strip().split("\n")
    if len(lines) >= 2:
        values = lines[1].split("\t")
        return float(values[2])  # Jaccard is 3rd column
    return None


def calculate_tad_f1(pred_tads: pd.DataFrame, hr_tads: pd.DataFrame) -> Dict:
    """
    Calculate F1 score for TAD predictions.

    Uses overlap-based matching (same logic as loops but 1D).
    A predicted TAD is a TP if it overlaps any HR TAD.
    """
    # Count True Positives
    pred_has_match = pred_tads.apply(lambda r: tad_has_overlap(r, hr_tads), axis=1)
    tp = pred_has_match.sum()

    # Count False Positives
    fp = len(pred_tads) - tp

    # Count False Negatives
    hr_has_match = hr_tads.apply(lambda r: tad_has_overlap(r, pred_tads), axis=1)
    fn = (~hr_has_match).sum()

    # Calculate F1
    denominator = tp + 0.5 * (fp + fn)
    f1 = tp / denominator if denominator > 0 else 0.0

    return {
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'F1': round(f1, 4)
    }


def run_tad_benchmark(pred_dir: str, gt_dir: str,
                      chroms: List[str], ratio: int = 16,
                      include_lr: bool = False) -> pd.DataFrame:
    """
    Run TAD benchmarking for all chromosomes.

    Args:
        pred_dir: Directory containing prediction Arrowhead results
        gt_dir: Directory containing ground truth Arrowhead results
        chroms: List of chromosomes to process
        ratio: Downsample ratio used in filenames
        include_lr: Also compare LR baseline against HR

    Returns:
        DataFrame with per-chromosome metrics
    """
    results = []

    for chrom in chroms:
        print(f"[TAD] Processing {chrom}...")

        # Build file paths
        pred_bedpe = os.path.join(
            pred_dir,
            f"preds_lr_test_{chrom}_ratio{ratio}_convert_10kb/10000_blocks.bedpe"
        )
        hr_bedpe = os.path.join(gt_dir, f"HR_arrowhead_{chrom}/10000_blocks.bedpe")
        lr_bedpe = os.path.join(gt_dir, f"LR_arrowhead_{chrom}/10000_blocks.bedpe")

        # Validate files exist
        if not os.path.isfile(pred_bedpe):
            print(f"  Warning: prediction not found: {pred_bedpe}")
            continue
        if not os.path.isfile(hr_bedpe):
            print(f"  Warning: HR ground truth not found: {hr_bedpe}")
            continue

        # Load TADs
        pred_tads = load_tads(pred_bedpe)
        hr_tads = load_tads(hr_bedpe)

        # Save as BED for bedtools
        pred_bed = os.path.join(pred_dir, f"tmp_pred_{chrom}.bed")
        hr_bed = os.path.join(gt_dir, f"tmp_hr_{chrom}.bed")
        save_as_bed(pred_tads, pred_bed)
        save_as_bed(hr_tads, hr_bed)

        # Calculate metrics
        f1_metrics = calculate_tad_f1(pred_tads, hr_tads)
        jaccard = calculate_tad_jaccard_bedtools(pred_bed, hr_bed)

        results.append({
            'Type': 'TAD',
            'Comparison': 'Pred_vs_HR',
            'Chromosome': chrom,
            'pred_count': len(pred_tads),
            'hr_count': len(hr_tads),
            **f1_metrics,
            'Jaccard': round(jaccard, 4) if jaccard else None
        })
        print(f"  Pred vs HR: F1={f1_metrics['F1']:.4f}, Jaccard={f'{jaccard:.4f}' if jaccard else 'N/A'}")

        # Optional: LR baseline comparison
        if include_lr and os.path.isfile(lr_bedpe):
            lr_tads = load_tads(lr_bedpe)
            lr_bed = os.path.join(gt_dir, f"tmp_lr_{chrom}.bed")
            save_as_bed(lr_tads, lr_bed)

            lr_f1_metrics = calculate_tad_f1(lr_tads, hr_tads)
            lr_jaccard = calculate_tad_jaccard_bedtools(lr_bed, hr_bed)

            results.append({
                'Type': 'TAD',
                'Comparison': 'LR_vs_HR',
                'Chromosome': chrom,
                'pred_count': len(lr_tads),
                'hr_count': len(hr_tads),
                **lr_f1_metrics,
                'Jaccard': round(lr_jaccard, 4) if lr_jaccard else None
            })
            print(f"  LR vs HR:   F1={lr_f1_metrics['F1']:.4f}, Jaccard={f'{lr_jaccard:.4f}' if lr_jaccard else 'N/A'}")

    return pd.DataFrame(results)


# ==============================================================================
# SECTION 4: LOOP VALIDATION (ChIP-seq)
# ==============================================================================
# Validates predicted loops against ChIP-seq peaks (CTCF, RAD21, SMC3).
# Measures what fraction of loop anchors contain expected binding sites.
# ==============================================================================

def convert_chr_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert chromosome names to integers for consistent matching.

    Handles: "chr18" -> 18, "chrX" -> dropped (or handled separately)
    """
    df = df.copy()
    df['chr'] = df['chr'].astype(str).str.replace('chr', '', regex=True)
    df['chr'] = pd.to_numeric(df['chr'], errors='coerce')
    return df.dropna(subset=['chr']).astype({'chr': 'int'})


def expand_and_merge_intervals(df: pd.DataFrame, expansion: int = 5000) -> pd.DataFrame:
    """
    Expand loop anchor intervals by ±expansion and merge overlapping ones.

    Args:
        df: DataFrame with chr, start, end columns
        expansion: Expansion in bp (default: 5000)

    Returns:
        DataFrame of merged intervals
    """
    if df.empty:
        return pd.DataFrame(columns=['chr', 'start', 'end'])

    df = df.copy()
    df['start'] = df['start'] - expansion
    df['end'] = df['end'] + expansion
    df = df.sort_values(['chr', 'start'])

    # Merge overlapping intervals
    merged = []
    curr_chr, curr_start, curr_end = df.iloc[0]

    for i in range(1, len(df)):
        row_chr, row_start, row_end = df.iloc[i][['chr', 'start', 'end']]

        # If same chromosome and overlapping/adjacent, extend current interval
        if row_chr == curr_chr and row_start <= curr_end + 1:
            curr_end = max(curr_end, row_end)
        else:
            merged.append([curr_chr, curr_start, curr_end])
            curr_chr, curr_start, curr_end = row_chr, row_start, row_end

    merged.append([curr_chr, curr_start, curr_end])
    return pd.DataFrame(merged, columns=['chr', 'start', 'end'])


def check_chipseq_overlap(chipseq_df: pd.DataFrame,
                          chrom: int, start: int, end: int) -> bool:
    """
    Check if any ChIP-seq peak is fully contained within a genomic interval.

    Args:
        chipseq_df: ChIP-seq peaks DataFrame (chr, start, end)
        chrom: Chromosome (integer)
        start: Interval start
        end: Interval end

    Returns:
        True if at least one peak is contained in the interval
    """
    if chipseq_df is None or chipseq_df.empty:
        return False

    contained = chipseq_df[
        (chipseq_df['chr'] == chrom) &
        (chipseq_df['start'] >= start) &
        (chipseq_df['end'] <= end)
    ]
    return len(contained) > 0


def run_loop_validation(pred_dir: str, chipseq_dir: str,
                        chroms: List[str],
                        factors: List[str] = ["CTCF", "RAD21", "SMC3"]) -> pd.DataFrame:
    """
    Validate loops against ChIP-seq peaks.

    Measures what fraction of loop anchors contain ALL expected binding factors.
    Loop anchors should be enriched for CTCF and cohesin (RAD21, SMC3).

    Args:
        pred_dir: Directory with HiCCUPS loop results
        chipseq_dir: Directory with ChIP-seq peak files
        chroms: Chromosomes to process
        factors: ChIP-seq factors to check (default: CTCF, RAD21, SMC3)

    Returns:
        DataFrame with per-chromosome validation rates
    """
    results = []

    # Load ChIP-seq data for all factors
    chipseq_data = {}
    for factor in factors:
        chipseq_file = os.path.join(chipseq_dir, factor, "merged_output.txt")
        if os.path.isfile(chipseq_file) and os.path.getsize(chipseq_file) > 0:
            df = pd.read_csv(chipseq_file, sep='\t', header=None,
                           usecols=[0, 1, 2], names=['chr', 'start', 'end'])
            chipseq_data[factor] = convert_chr_to_int(df)
            print(f"[VALIDATION] Loaded {factor}: {len(chipseq_data[factor])} peaks")
        else:
            print(f"[VALIDATION] Warning: {factor} data not found")

    available_factors = list(chipseq_data.keys())
    if not available_factors:
        print("[VALIDATION] No ChIP-seq data available, skipping validation")
        return pd.DataFrame()

    for chrom in chroms:
        print(f"[VALIDATION] Processing {chrom}...")

        # Build file path
        loop_file = os.path.join(pred_dir, f"hiccups_results_{chrom}/merged_loops.bedpe")

        if not os.path.isfile(loop_file):
            print(f"  Warning: loop file not found: {loop_file}")
            continue

        # Load loops
        try:
            loops = pd.read_csv(loop_file, sep='\t', skiprows=[1],
                               usecols=[0, 1, 2, 3, 4, 5])
            loops.columns = ['chr1', 'x1', 'x2', 'chr2', 'y1', 'y2']
        except Exception as e:
            print(f"  Error reading {loop_file}: {e}")
            continue

        # Extract both anchors of each loop
        upstream = loops[['chr1', 'x1', 'x2']].copy()
        upstream.columns = ['chr', 'start', 'end']
        downstream = loops[['chr2', 'y1', 'y2']].copy()
        downstream.columns = ['chr', 'start', 'end']

        # Combine and deduplicate anchors
        all_anchors = pd.concat([upstream, downstream], ignore_index=True)
        all_anchors = all_anchors.drop_duplicates()
        all_anchors = convert_chr_to_int(all_anchors)

        # Expand and merge overlapping anchors
        merged_anchors = expand_and_merge_intervals(all_anchors)

        if merged_anchors.empty:
            continue

        # Count validated anchors (contain ALL available factors)
        validated_count = 0
        for _, row in merged_anchors.iterrows():
            chrom_int, start, end = int(row['chr']), row['start'], row['end']

            # Check if ALL factors have peaks in this anchor
            all_present = all(
                check_chipseq_overlap(chipseq_data[f], chrom_int, start, end)
                for f in available_factors
            )

            if all_present:
                validated_count += 1

        total_anchors = len(merged_anchors)
        validation_rate = (validated_count / total_anchors * 100) if total_anchors > 0 else 0

        results.append({
            'Type': 'Loop_Validation',
            'Chromosome': chrom,
            'Total_Anchors': total_anchors,
            'Validated_Anchors': validated_count,
            'Validation_Rate_%': round(validation_rate, 2),
            'Factors_Checked': ','.join(available_factors)
        })
        print(f"  Validated: {validated_count}/{total_anchors} ({validation_rate:.1f}%)")

    return pd.DataFrame(results)


# ==============================================================================
# SECTION 5: MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection
    parser.add_argument("--mode", choices=["loop", "tad", "validation", "all"],
                        default="all", help="Benchmarking mode (default: all)")

    # Directory arguments
    parser.add_argument("--pred-dir", required=True,
                        help="Prediction directory (contains HiCCUPS/Arrowhead outputs)")
    parser.add_argument("--gt-dir", required=True,
                        help="Ground truth directory (from ground_truth_tad_loop.sh)")
    parser.add_argument("--chipseq-dir", default=None,
                        help="ChIP-seq peaks directory (for validation mode)")

    # Parameters
    parser.add_argument("--chroms", nargs="+",
                        default=["chr18", "chr19", "chr20", "chr21", "chr22"],
                        help="Chromosomes to process (default: chr18-22)")
    parser.add_argument("--tolerance", type=int, default=5000,
                        help="Loop matching tolerance in bp (default: 5000)")
    parser.add_argument("--ratio", type=str, default=16,
                        help="Downsample ratio for TAD file paths (default: 16)")
    parser.add_argument("--fdr", default=None,
                        help="HiCCUPS FDR threshold (matches feature_calling --fdr output)")
    parser.add_argument("--include-lr", action="store_true",
                        help="Include LR baseline comparison")

    # Output
    parser.add_argument("--output", default="benchmark_results.csv",
                        help="Output CSV file (default: benchmark_results.csv)")

    args = parser.parse_args()

    all_results = []

    # Run requested benchmarks
    if args.mode in ["loop", "all"]:
        print("\n" + "="*60)
        print("LOOP BENCHMARKING")
        print("="*60)
        loop_results = run_loop_benchmark(
            args.pred_dir, args.gt_dir, args.chroms,
            args.tolerance, args.include_lr, args.fdr
        )
        all_results.append(loop_results)

    if args.mode in ["tad", "all"]:
        print("\n" + "="*60)
        print("TAD BENCHMARKING")
        print("="*60)
        tad_results = run_tad_benchmark(
            args.pred_dir, args.gt_dir, args.chroms,
            args.ratio, args.include_lr
        )
        all_results.append(tad_results)

    if args.mode in ["validation", "all"] and args.chipseq_dir:
        print("\n" + "="*60)
        print("LOOP VALIDATION (ChIP-seq)")
        print("="*60)
        val_results = run_loop_validation(
            args.pred_dir, args.chipseq_dir, args.chroms
        )
        all_results.append(val_results)
    elif args.mode in ["validation", "all"] and not args.chipseq_dir:
        print("\n[VALIDATION] Skipped: --chipseq-dir not provided")

    # Combine and save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.output, index=False, sep='\t')
        print(f"\n{'='*60}")
        print(f"Results saved to: {args.output}")
        print(f"{'='*60}")

        # Print summary
        print("\nSUMMARY:")
        print(combined.to_string(index=False))
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
