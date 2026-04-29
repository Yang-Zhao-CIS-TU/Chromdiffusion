#!/usr/bin/env python3
"""
Fix denormalization by removing the [-5, 5] clipping

The issue: RobustHiCPreprocessor clips values to [-5, 5] before denormalization,
which limits the maximum output value to expm1(5 * IQR + median).

This script removes the clipping to allow full range of values.
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Define class for torch.load
class RobustHiCPreprocessor:
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self._is_fitted = False

sys.modules['__main__'].RobustHiCPreprocessor = RobustHiCPreprocessor


def load_preprocessor_stats(preprocessor_path):
    """Load Y_median and Y_iqr from preprocessor"""
    try:
        checkpoint = torch.load(preprocessor_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(preprocessor_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'preprocessor' in checkpoint:
        prep = checkpoint['preprocessor']
    else:
        prep = checkpoint
    
    if hasattr(prep, 'Y_mean') and hasattr(prep, 'Y_std'):
        return prep.Y_mean, prep.Y_std
    else:
        raise ValueError("Could not extract Y_mean/Y_std from preprocessor")


def denormalize_no_clip(normalized_data, Y_median, Y_iqr):
    """
    Denormalize WITHOUT clipping to [-5, 5]
    
    This allows the full range of values to be preserved.
    """
    # NO clipping - allow full range
    Y_log = normalized_data * Y_iqr + Y_median
    
    # Reverse log transform
    Y_raw = np.expm1(Y_log)  # exp(Y_log) - 1
    
    # Ensure non-negative
    Y_raw = np.maximum(Y_raw, 0.0)
    
    return Y_raw.astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Denormalize without clipping')
    parser.add_argument('--preprocessor_path', type=str, required=True,
                       help='Path to preprocessor.pt')
    parser.add_argument('--refined_dirs', type=str, nargs='+',
                       default=['refined_chr18', 'refined_chr19', 'refined_chr20', 
                               'refined_chr21', 'refined_chr22'],
                       help='Directories containing refined_norm.npy')
    parser.add_argument('--output_suffix', type=str, default='_noclip',
                       help='Suffix for output files (default: _noclip)')
    args = parser.parse_args()
    
    print("="*80)
    print("DENORMALIZE WITHOUT CLIPPING")
    print("="*80)
    
    # Load preprocessor stats
    print(f"\nLoading preprocessor: {args.preprocessor_path}")
    Y_median, Y_iqr = load_preprocessor_stats(args.preprocessor_path)
    print(f"  Y_median: {Y_median:.6f}")
    print(f"  Y_iqr:    {Y_iqr:.6f}")
    
    # Calculate theoretical max WITH clipping
    clipped_max = np.expm1(5 * Y_iqr + Y_median)
    print(f"\nWith clipping [-5, 5]:")
    print(f"  Max possible value: {clipped_max:.0f}")
    
    # Process each directory
    print("\n" + "="*80)
    print("PROCESSING")
    print("="*80)
    
    for refined_dir in args.refined_dirs:
        norm_path = Path(refined_dir) / 'refined_norm.npy'
        
        if not norm_path.exists():
            print(f"\n⚠️  {norm_path} not found, skipping")
            continue
        
        # Load normalized data
        refined_norm = np.load(norm_path)
        
        print(f"\n{refined_dir}:")
        print(f"  Norm shape: {refined_norm.shape}")
        print(f"  Norm range: [{refined_norm.min():.4f}, {refined_norm.max():.4f}]")
        
        # Denormalize WITHOUT clipping
        refined_raw = denormalize_no_clip(refined_norm, Y_median, Y_iqr)
        
        print(f"  Raw range:  [{refined_raw.min():.0f}, {refined_raw.max():.0f}]")
        
        # Save
        output_path = Path(refined_dir) / f'refined_raw{args.output_suffix}.npy'
        np.save(output_path, refined_raw)
        print(f"  ✓ Saved: {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nKey difference:")
    print(f"  With clipping:    max = expm1(5 * {Y_iqr:.4f} + {Y_median:.4f}) = {clipped_max:.0f}")
    print(f"  Without clipping: max = expm1(norm_max * {Y_iqr:.4f} + {Y_median:.4f}) = varies by chromosome")
    print(f"\nOutput files created with suffix: {args.output_suffix}")
    print(f"Use these for evaluation:")
    for refined_dir in args.refined_dirs:
        print(f"  - {refined_dir}/refined_raw{args.output_suffix}.npy")
    print("="*80)


if __name__ == '__main__':
    main()
