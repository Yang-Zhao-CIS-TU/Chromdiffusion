#!/usr/bin/env python3
"""Diagnose which training script was used for each checkpoint"""

import torch
from pathlib import Path
import json

def check_checkpoint(ckpt_path):
    """Check checkpoint details"""
    print(f"\n{'='*70}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*70}")
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Check keys
        print(f"Keys in checkpoint: {list(ckpt.keys())}")
        
        # Check config
        config = ckpt.get('config', {})
        print(f"\nConfig:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        # Check model architecture indicators
        model_keys = list(ckpt['model_state_dict'].keys())
        print(f"\nModel architecture indicators:")
        print(f"  Total parameters: {len(model_keys)}")
        print(f"  Has 'cond_transform': {'cond_transform' in str(model_keys)}")
        print(f"  Has 'gate_conv': {'gate_conv' in str(model_keys)}")
        print(f"  Has 'encoder_blocks': {'encoder_blocks' in str(model_keys)}")
        print(f"  Has 'encoder.0': {'encoder.0' in str(model_keys)}")
        
        # Check training info
        print(f"\nTraining info:")
        for key in ['epoch', 'loss', 'res_mean', 'res_std', 'alpha']:
            if key in ckpt:
                print(f"  {key}: {ckpt[key]}")
        
        # Determine architecture
        if 'cond_transform' in str(model_keys) and 'gate_conv' in str(model_keys):
            print(f"\n→ Architecture: V15 Gated Conditioned U-Net")
            print(f"→ Training script: train_v15.py")
            print(f"→ Test script: test_v15.py")
        elif 'encoder_blocks' in str(model_keys):
            print(f"\n→ Architecture: Vanilla Residual Diffusion U-Net")
            print(f"→ Training script: train_residual_diffusion.py")
            print(f"→ Test script: test_residual_diffusion.py")
        else:
            print(f"\n→ Architecture: Unknown")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

# Check all checkpoints
checkpoints = [
    'checkpoints_diffusion_hicnn/best_model.pt',
    'checkpoints_v15_hicnn/best_model_pcc.pt',
]

for ckpt_path in checkpoints:
    if Path(ckpt_path).exists():
        check_checkpoint(ckpt_path)
    else:
        print(f"\n{ckpt_path}: NOT FOUND")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print("""
Checkpoint Directory          → Training Script              → Test Script
--------------------------------------------------------------------------------
checkpoints_diffusion_hicnn   → train_residual_diffusion.py → test_residual_diffusion.py
checkpoints_v15_hicnn         → train_v15.py                → test_v15.py
""")
