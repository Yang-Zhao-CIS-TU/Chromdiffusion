#!/usr/bin/env python3
"""Check statistics of saved results"""

import numpy as np
from pathlib import Path
import sys

def check_stats(path):
    """Load and print stats for a numpy file"""
    if not path.exists():
        return None
    data = np.load(path)
    return {
        'shape': data.shape,
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(data.mean()),
        'std': float(data.std())
    }

def main(output_dir):
    out = Path(output_dir)
    
    if not out.exists():
        print(f"Directory not found: {out}")
        return
    
    print(f"\n{'='*80}")
    print(f"Results Directory: {out}")
    print(f"{'='*80}")
    
    # Check both spaces
    for space in ['norm', 'raw']:
        space_dir = out / space
        if not space_dir.exists():
            continue
        
        print(f"\n{'='*80}")
        print(f"{space.upper()} SPACE")
        print(f"{'='*80}")
        
        # Find all chromosomes
        chroms = set()
        for f in space_dir.glob("*.npy"):
            name = f.stem
            for prefix in ['base_', 'refined_', 'gt_']:
                if name.startswith(prefix):
                    chroms.add(name[len(prefix):])
        
        chroms = sorted(chroms)
        
        for chrom in chroms:
            print(f"\n>>> {chrom}")
            print(f"{'File':<20} {'Shape':<20} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
            print(f"{'-'*88}")
            
            for prefix in ['base', 'refined', 'gt']:
                path = space_dir / f"{prefix}_{chrom}.npy"
                stats = check_stats(path)
                if stats:
                    print(f"{prefix:<20} {str(stats['shape']):<20} {stats['min']:>12.2f} {stats['max']:>12.2f} {stats['mean']:>12.2f} {stats['std']:>12.2f}")
                else:
                    print(f"{prefix:<20} NOT FOUND")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_results.py <output_dir>")
        print("Example: python check_results.py test_results_diffusion_deephic")
        sys.exit(1)
    
    main(sys.argv[1])
