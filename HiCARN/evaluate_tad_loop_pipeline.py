#!/usr/bin/env python3
"""
================================================================================
HiCARN vs HiCARN+Diffusion TAD/Loop Evaluation Pipeline
================================================================================

Complete evaluation pipeline for comparing TAD and loop detection performance:
  - HiCARN baseline predictions
  - HiCARN + Diffusion refined predictions
  - Low-resolution baseline
  - High-resolution ground truth

WORKFLOW:
  1. Convert predictions to Juicer format
  2. Build .hic files using juicer_tools
  3. Run TAD calling (Arrowhead) and loop calling (HiCCUPS)
  4. Benchmark against ground truth (F1, Jaccard, validation)

This script orchestrates the entire pipeline and generates comparison reports.

Usage:
    python evaluate_tad_loop_pipeline.py \
        --hicarn-pred hicarn_predictions/predictions_norm.npy \
        --refined-pred refined_predictions/refined_chr22.npy \
        --gt-path hicarn_predictions/ground_truth.npy \
        --preprocessor hicarn_predictions/hicarn_preprocessor.pt \
        --chroms chr18 chr19 chr20 chr21 chr22 \
        --output-dir tad_loop_evaluation

Author: Adapted for HiCARN + Residual Diffusion evaluation
================================================================================
"""

import argparse
import os
import sys
import subprocess
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_CHROMS = ["chr18", "chr19", "chr20", "chr21", "chr22"]
DEFAULT_RESOLUTION = 10000
DEFAULT_RATIO = 16
DEFAULT_CHUNK_SIZE = 40


class EvaluationConfig:
    """Configuration for TAD/loop evaluation pipeline"""
    
    def __init__(self, args):
        self.hicarn_pred = args.hicarn_pred
        self.refined_pred = args.refined_pred
        self.gt_path = args.gt_path
        self.preprocessor_path = args.preprocessor
        
        # Directories
        self.output_dir = Path(args.output_dir)
        self.juicer_root = args.juicer_root
        self.chrom_sizes = args.chrom_sizes
        
        # Processing parameters
        self.chroms = args.chroms
        self.resolution = args.resolution
        self.ratio = args.ratio
        self.chunk_size = args.chunk_size
        
        # Tools
        self.juicer_tools = args.juicer_tools
        self.bedtools = args.bedtools or "bedtools"
        
        # Evaluation parameters
        self.loop_tolerance = args.loop_tolerance
        
        # Create subdirectories
        self.setup_directories()
    
    def setup_directories(self):
        """Create all necessary output directories"""
        self.hicarn_dir = self.output_dir / "hicarn"
        self.refined_dir = self.output_dir / "refined"
        self.lr_dir = self.output_dir / "lr_baseline"
        self.hr_dir = self.output_dir / "hr_groundtruth"
        self.results_dir = self.output_dir / "results"
        
        for d in [self.hicarn_dir, self.refined_dir, self.lr_dir, 
                  self.hr_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# STEP 1: DENORMALIZE PREDICTIONS
# ==============================================================================

def denormalize_predictions(pred_norm_path: str, preprocessor_path: str, 
                            output_path: str) -> str:
    """
    Denormalize predictions from normalized space to raw contact counts
    
    Args:
        pred_norm_path: Path to normalized predictions
        preprocessor_path: Path to HiCARN preprocessor
        output_path: Where to save denormalized predictions
    
    Returns:
        Path to denormalized predictions
    """
    print(f"\nDenormalizing: {pred_norm_path}")
    
    import torch
    
    # Load preprocessor
    preprocessor = torch.load(preprocessor_path, map_location='cpu')
    
    # Load normalized predictions
    pred_norm = np.load(pred_norm_path)
    
    # Add channel dimension if needed
    if pred_norm.ndim == 3:
        pred_norm = pred_norm[:, None, :, :]
    
    # Denormalize
    pred_raw = preprocessor.postprocess(pred_norm)
    
    # Remove channel dimension
    if pred_raw.shape[1] == 1:
        pred_raw = pred_raw[:, 0, :, :]
    
    # Save
    np.save(output_path, pred_raw)
    print(f"  Saved: {output_path}")
    print(f"  Shape: {pred_raw.shape}")
    print(f"  Range: [{pred_raw.min():.2f}, {pred_raw.max():.2f}]")
    
    return output_path


# ==============================================================================
# STEP 2: CONVERT TO JUICER FORMAT
# ==============================================================================

def convert_predictions_to_juicer(
    pred_path: str,
    output_dir: Path,
    chroms: List[str],
    config: EvaluationConfig
) -> Dict[str, str]:
    """
    Convert model predictions to Juicer format using prediction_convert_combined_TK.py
    
    Returns:
        Dict mapping chromosome to converted file path
    """
    print(f"\n{'='*70}")
    print(f"Converting predictions to Juicer format")
    print(f"{'='*70}")
    
    # This assumes you have the prediction_convert_combined_TK.py script
    convert_script = "prediction_convert_combined_TK.py"
    
    if not os.path.exists(convert_script):
        print(f"WARNING: {convert_script} not found!")
        print("Please ensure prediction_convert_combined_TK.py is in the current directory")
        return {}
    
    # Run conversion
    cmd = [
        "python", convert_script,
        "--input-dir", os.path.dirname(pred_path),
        "--input-format", "npy",
        "--output-dir", str(output_dir),
        "--juicer-root", config.juicer_root,
        "--chrom-sizes", config.chrom_sizes,
        "--chroms", *chroms,
        "--ratio", str(config.ratio),
        "--resolution", str(config.resolution),
        "--chunk-size", str(config.chunk_size),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR during conversion:")
        print(result.stderr)
        return {}
    
    print(result.stdout)
    
    # Return paths to converted files
    converted_files = {}
    for chrom in chroms:
        txt_file = output_dir / f"preds_lr_test_{chrom}_ratio{config.ratio}_convert.txt"
        if txt_file.exists():
            converted_files[chrom] = str(txt_file)
    
    return converted_files


# ==============================================================================
# STEP 3: BUILD .HIC FILES
# ==============================================================================

def build_hic_file(
    bedpe_path: str,
    output_hic: str,
    chrom_sizes: str,
    resolution: int,
    juicer_tools: str
) -> str:
    """
    Build .hic file from BEDPE using juicer_tools pre
    
    Args:
        bedpe_path: Path to BEDPE format file
        output_hic: Output .hic file path
        chrom_sizes: Path to chromosome sizes file
        resolution: Resolution in bp
        juicer_tools: Path to juicer_tools.jar
    
    Returns:
        Path to created .hic file
    """
    print(f"\nBuilding .hic file: {output_hic}")
    
    cmd = [
        "java", "-Xmx8g", "-jar", juicer_tools,
        "pre",
        "-r", str(resolution),
        bedpe_path,
        output_hic,
        chrom_sizes
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR building .hic file:")
        print(result.stderr)
        return None
    
    print(f"  ✓ Created: {output_hic}")
    return output_hic


# ==============================================================================
# STEP 4: RUN TAD AND LOOP CALLING
# ==============================================================================

def run_arrowhead(
    hic_file: str,
    output_dir: Path,
    chrom: str,
    resolution: int,
    juicer_tools: str
) -> str:
    """
    Run Arrowhead TAD calling
    
    Returns:
        Path to TAD bedpe file
    """
    print(f"\n  Running Arrowhead for {chrom}...")
    
    cmd = [
        "java", "-Xmx8g", "-jar", juicer_tools,
        "arrowhead",
        "-r", str(resolution),
        "-k", "KR",
        hic_file,
        str(output_dir / f"arrowhead_{chrom}")
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    WARNING: Arrowhead failed for {chrom}")
        return None
    
    tad_file = output_dir / f"arrowhead_{chrom}" / f"{resolution}_blocks.bedpe"
    if tad_file.exists():
        print(f"    ✓ TADs: {tad_file}")
        return str(tad_file)
    return None


def run_hiccups(
    hic_file: str,
    output_dir: Path,
    chrom: str,
    resolution: int,
    juicer_tools: str
) -> str:
    """
    Run HiCCUPS loop calling
    
    Returns:
        Path to loops bedpe file
    """
    print(f"\n  Running HiCCUPS for {chrom}...")
    
    cmd = [
        "java", "-Xmx8g", "-jar", juicer_tools,
        "hiccups",
        "-r", str(resolution),
        "-k", "KR",
        hic_file,
        str(output_dir / f"hiccups_results_KR_{chrom}")
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    WARNING: HiCCUPS failed for {chrom}")
        return None
    
    loop_file = output_dir / f"hiccups_results_KR_{chrom}" / "merged_loops.bedpe"
    if loop_file.exists():
        print(f"    ✓ Loops: {loop_file}")
        return str(loop_file)
    return None


def run_feature_calling(
    hic_files: Dict[str, str],
    output_dir: Path,
    chroms: List[str],
    config: EvaluationConfig
) -> Dict:
    """
    Run TAD and loop calling for all chromosomes
    
    Returns:
        Dict with TAD and loop file paths
    """
    print(f"\n{'='*70}")
    print("RUNNING TAD AND LOOP CALLING")
    print(f"{'='*70}")
    
    results = {
        'tads': {},
        'loops': {}
    }
    
    for chrom in chroms:
        if chrom not in hic_files:
            continue
        
        hic_file = hic_files[chrom]
        print(f"\nProcessing {chrom}...")
        
        # Run Arrowhead (TADs)
        tad_file = run_arrowhead(
            hic_file, output_dir, chrom, 
            config.resolution, config.juicer_tools
        )
        if tad_file:
            results['tads'][chrom] = tad_file
        
        # Run HiCCUPS (loops)
        loop_file = run_hiccups(
            hic_file, output_dir, chrom,
            config.resolution, config.juicer_tools
        )
        if loop_file:
            results['loops'][chrom] = loop_file
    
    return results


# ==============================================================================
# STEP 5: RUN BENCHMARKING
# ==============================================================================

def run_benchmark(
    pred_dir: Path,
    gt_dir: Path,
    output_file: str,
    chroms: List[str],
    config: EvaluationConfig
) -> pd.DataFrame:
    """
    Run benchmarking using benchmark_combined_TK.py
    
    Returns:
        DataFrame with benchmark results
    """
    print(f"\n{'='*70}")
    print("RUNNING BENCHMARK")
    print(f"{'='*70}")
    
    benchmark_script = "benchmark_combined_TK.py"
    
    if not os.path.exists(benchmark_script):
        print(f"WARNING: {benchmark_script} not found!")
        print("Please ensure benchmark_combined_TK.py is in the current directory")
        return pd.DataFrame()
    
    cmd = [
        "python", benchmark_script,
        "--mode", "all",
        "--pred-dir", str(pred_dir),
        "--gt-dir", str(gt_dir),
        "--chroms", *chroms,
        "--tolerance", str(config.loop_tolerance),
        "--ratio", str(config.ratio),
        "--output", output_file
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"ERROR during benchmarking:")
        print(result.stderr)
        return pd.DataFrame()
    
    # Load results
    if os.path.exists(output_file):
        return pd.read_csv(output_file, sep='\t')
    return pd.DataFrame()


# ==============================================================================
# STEP 6: COMPARISON REPORT
# ==============================================================================

def generate_comparison_report(
    hicarn_results: pd.DataFrame,
    refined_results: pd.DataFrame,
    output_path: str
):
    """
    Generate comparison report between HiCARN and HiCARN+Diffusion
    """
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*70}")
    
    # Combine results
    hicarn_results['Model'] = 'HiCARN'
    refined_results['Model'] = 'HiCARN+Diffusion'
    
    combined = pd.concat([hicarn_results, refined_results], ignore_index=True)
    combined.to_csv(output_path, index=False, sep='\t')
    
    print(f"\n✓ Saved comparison: {output_path}")
    
    # Calculate improvements
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    
    # For each metric type
    for metric_type in ['Loop', 'TAD']:
        hicarn_rows = hicarn_results[hicarn_results['Type'].str.contains(metric_type, na=False)]
        refined_rows = refined_results[refined_results['Type'].str.contains(metric_type, na=False)]
        
        if len(hicarn_rows) == 0 or len(refined_rows) == 0:
            continue
        
        print(f"\n{metric_type} Performance:")
        print("-" * 70)
        
        # Calculate average F1 and Jaccard
        for metric in ['F1', 'Jaccard']:
            if metric in hicarn_rows.columns and metric in refined_rows.columns:
                hicarn_avg = hicarn_rows[metric].mean()
                refined_avg = refined_rows[metric].mean()
                improvement = ((refined_avg - hicarn_avg) / hicarn_avg * 100) if hicarn_avg > 0 else 0
                
                print(f"{metric:10s}: HiCARN = {hicarn_avg:.4f}, "
                      f"Refined = {refined_avg:.4f}, "
                      f"Δ = {improvement:+.2f}%")
    
    print(f"\n{'='*70}")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input predictions
    parser.add_argument('--hicarn-pred', required=True,
                       help='HiCARN predictions (predictions_norm.npy)')
    parser.add_argument('--refined-pred', required=True,
                       help='HiCARN+Diffusion refined predictions (refined_norm.npy)')
    parser.add_argument('--gt-path', required=True,
                       help='Ground truth (ground_truth.npy)')
    parser.add_argument('--preprocessor', required=True,
                       help='HiCARN preprocessor (.pt file)')
    
    # Directories and references
    parser.add_argument('--output-dir', default='tad_loop_evaluation',
                       help='Output directory (default: %(default)s)')
    parser.add_argument('--juicer-root', required=True,
                       help='Juicer root directory with HR/LR intra_NONE dumps')
    parser.add_argument('--chrom-sizes', required=True,
                       help='Chromosome sizes file (hg19.chrom.sizes)')
    
    # Processing parameters
    parser.add_argument('--chroms', nargs='+', default=DEFAULT_CHROMS,
                       help='Chromosomes to process (default: chr18-22)')
    parser.add_argument('--resolution', type=int, default=DEFAULT_RESOLUTION,
                       help='Resolution in bp (default: %(default)s)')
    parser.add_argument('--ratio', type=int, default=DEFAULT_RATIO,
                       help='Downsample ratio (default: %(default)s)')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                       help='Tile size (default: %(default)s)')
    
    # Tools
    parser.add_argument('--juicer-tools', required=True,
                       help='Path to juicer_tools.jar')
    parser.add_argument('--bedtools', default=None,
                       help='Path to bedtools (default: use system bedtools)')
    
    # Evaluation parameters
    parser.add_argument('--loop-tolerance', type=int, default=5000,
                       help='Loop matching tolerance in bp (default: %(default)s)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = EvaluationConfig(args)
    
    print("="*70)
    print("HiCARN vs HiCARN+DIFFUSION TAD/LOOP EVALUATION")
    print("="*70)
    print(f"HiCARN predictions: {config.hicarn_pred}")
    print(f"Refined predictions: {config.refined_pred}")
    print(f"Ground truth: {config.gt_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Chromosomes: {', '.join(config.chroms)}")
    print("="*70)
    
    # STEP 1: Denormalize predictions
    print(f"\n{'='*70}")
    print("STEP 1: DENORMALIZING PREDICTIONS")
    print(f"{'='*70}")
    
    hicarn_raw = denormalize_predictions(
        config.hicarn_pred,
        config.preprocessor_path,
        str(config.hicarn_dir / "hicarn_raw.npy")
    )
    
    refined_raw = denormalize_predictions(
        config.refined_pred,
        config.preprocessor_path,
        str(config.refined_dir / "refined_raw.npy")
    )
    
    gt_raw = denormalize_predictions(
        config.gt_path,
        config.preprocessor_path,
        str(config.hr_dir / "gt_raw.npy")
    )
    
    # STEP 2-5: Process both models
    results_summary = {}
    
    for model_name, pred_path, output_dir in [
        ("HiCARN", hicarn_raw, config.hicarn_dir),
        ("HiCARN+Diffusion", refined_raw, config.refined_dir)
    ]:
        print(f"\n{'='*70}")
        print(f"PROCESSING: {model_name}")
        print(f"{'='*70}")
        
        # Step 2: Convert to Juicer format
        # (Note: You'll need to adapt this based on your actual conversion script)
        print("\nStep 2: Conversion to Juicer format")
        print("Please run prediction_convert_combined_TK.py manually for:")
        print(f"  Input: {pred_path}")
        print(f"  Output: {output_dir}")
        
        # Step 3-5: Build .hic, run feature calling, benchmark
        # (This requires the converted files - implementation depends on your setup)
    
    print(f"\n{'='*70}")
    print("EVALUATION PIPELINE SETUP COMPLETE")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Run prediction_convert_combined_TK.py for HiCARN and refined predictions")
    print("2. Run feature_calling to generate TADs and loops")
    print("3. Run benchmark_combined_TK.py to compare performance")
    print(f"\nAll outputs will be in: {config.output_dir}")


if __name__ == "__main__":
    main()
