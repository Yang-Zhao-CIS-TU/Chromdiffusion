"""
Quick Shape Fix for Evaluation Compatibility

Problem: 
  Refined output: (N, H, W) or (N, 1, H, W)
  Ground truth:   (N, H, W, 1)
  
Solution: Convert refined output to match GT format
"""

import numpy as np
import argparse
from pathlib import Path


def fix_shape_for_evaluation(refined_path, output_path=None):
    """
    Convert refined predictions to match ground truth shape format
    
    Args:
        refined_path: Path to refined predictions
        output_path: Output path (if None, overwrites original)
    """
    print(f"\nLoading refined predictions: {refined_path}")
    refined = np.load(refined_path)
    print(f"  Original shape: {refined.shape}")
    print(f"  Original range: [{refined.min():.2f}, {refined.max():.2f}]")
    
    # Convert to (N, H, W, 1) format
    if refined.ndim == 3:
        # (N, H, W) → (N, H, W, 1)
        refined_fixed = refined[:, :, :, np.newaxis]
    elif refined.ndim == 4:
        if refined.shape[1] == 1:
            # (N, 1, H, W) → (N, H, W, 1)
            refined_fixed = refined.squeeze(1)[:, :, :, np.newaxis]
        elif refined.shape[3] == 1:
            # Already (N, H, W, 1)
            refined_fixed = refined
        else:
            raise ValueError(f"Unexpected shape: {refined.shape}")
    else:
        raise ValueError(f"Unexpected ndim: {refined.ndim}")
    
    print(f"\nFixed shape: {refined_fixed.shape}")
    print(f"Fixed range: [{refined_fixed.min():.2f}, {refined_fixed.max():.2f}]")
    
    # Save
    if output_path is None:
        output_path = refined_path
    
    np.save(output_path, refined_fixed)
    print(f"\n✅ Saved to: {output_path}")
    
    return refined_fixed


def main():
    parser = argparse.ArgumentParser(description='Fix shape for evaluation')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to refined predictions')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (if None, overwrites input)')
    
    args = parser.parse_args()
    
    fix_shape_for_evaluation(args.input, args.output)


if __name__ == '__main__':
    main()
