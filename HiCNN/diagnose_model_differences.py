#!/usr/bin/env python3
"""Compare results across different diffusion models"""

import numpy as np
from pathlib import Path

models = [
    'test_results_diffusion_hicnn',
    'test_results_v15_hicnn',
    'test_results_hicarn_v15_on_hicnn',
]

chroms = ['chr18', 'chr19', 'chr20', 'chr21', 'chr22']

print("="*100)
print("MODEL COMPARISON - What's Actually Different?")
print("="*100)

for chrom in chroms:
    print(f"\n{'='*100}")
    print(f"{chrom}")
    print(f"{'='*100}")
    print(f"{'Model':<35} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Median':>10} {'P95':>10}")
    print(f"{'-'*100}")
    
    for model_dir in models:
        model_name = model_dir.replace('test_results_', '').replace('_hicnn', '').replace('_on', ' on')
        
        # Check refined predictions
        refined_path = Path(model_dir) / 'raw' / f'refined_{chrom}.npy'
        
        if refined_path.exists():
            data = np.load(refined_path)
            print(f"{model_name:<35} {data.min():>10.2f} {data.max():>10.2f} {data.mean():>10.2f} "
                  f"{data.std():>10.2f} {np.median(data):>10.2f} {np.percentile(data, 95):>10.2f}")
        else:
            print(f"{model_name:<35} NOT FOUND")
    
    # Show GT for reference
    gt_path = Path(models[0]) / 'raw' / f'gt_{chrom}.npy'
    if gt_path.exists():
        gt = np.load(gt_path)
        print(f"{'Ground Truth':<35} {gt.min():>10.2f} {gt.max():>10.2f} {gt.mean():>10.2f} "
              f"{gt.std():>10.2f} {np.median(gt):>10.2f} {np.percentile(gt, 95):>10.2f}")

print(f"\n{'='*100}")
print("KEY INSIGHT:")
print("  - Max values may be similar (all hit clipping ceiling)")
print("  - Look at: Mean, Std, Median, P95 - these should differ between models")
print("  - Better models: Mean/Median closer to GT, better distribution match")
print("="*100)
