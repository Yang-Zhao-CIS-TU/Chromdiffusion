"""
Quick Validation Check

Usage:
  python quick_validation.py --checkpoint checkpoints_v2/checkpoint_epoch_12.pt
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json

from data_loader import ResidualDiffusionDataset
from model import ImprovedResidualDiffusionUNet
from scheduler import ImprovedDDPMScheduler
from tqdm import tqdm


@torch.no_grad()
def sample_and_evaluate(checkpoint_path, val_hicarn, val_gt, device='cuda', num_samples=100):
    """Quick validation on subset of data"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')
    
    # Load model
    model = ImprovedResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        cond_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_levels=(2, 3),
        parameterization='v'
    ).to(device)
    
    # Load weights (try EMA first, then regular)
    if 'ema_shadow' in checkpoint and checkpoint['ema_shadow']:
        print("  Using EMA weights")
        ema_state = checkpoint['ema_shadow']
        model_state = {k: v for k, v in ema_state.items()}
        model.load_state_dict(model_state)
    else:
        print("  Using regular weights")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Load scheduler
    scheduler = ImprovedDDPMScheduler(
        num_train_timesteps=1000,
        parameterization='v'
    )
    scheduler.set_timesteps(50, device=device, method='uniform')
    
    # Load data
    print(f"Loading validation data...")
    dataset = ResidualDiffusionDataset(val_hicarn, val_gt)
    
    # Sample subset
    if len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        indices = range(len(dataset))
        num_samples = len(dataset)
    
    print(f"  Evaluating on {num_samples} samples...")
    
    # Collect predictions
    predictions = []
    ground_truths = []
    
    for idx in tqdm(indices, desc="Sampling"):
        batch_data = dataset[idx]
        condition = batch_data['hicarn'].unsqueeze(0).to(device)
        gt = batch_data['gt'].unsqueeze(0).to(device)
        
        # Start from noise
        residual = torch.randn_like(condition)
        
        # DDIM denoising
        for t in scheduler.timesteps:
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            
            # Predict
            model_output = model(residual, t_batch, condition)
            
            # Denoise one step
            residual, _ = scheduler.step(
                model_output, t, residual,
                eta=0.0, use_ddim=True
            )
        
        # Final prediction
        refined = condition + residual
        
        predictions.append(refined.cpu().numpy()[0, 0])
        ground_truths.append(gt.cpu().numpy()[0, 0])
    
    # Compute metrics
    print(f"\nComputing metrics...")
    from scipy.stats import pearsonr, spearmanr
    
    all_pred = np.array(predictions)
    all_gt = np.array(ground_truths)
    
    # Overall correlation
    pred_flat = all_pred.flatten()
    gt_flat = all_gt.flatten()
    
    pcc = pearsonr(pred_flat, gt_flat)[0]
    spc = spearmanr(pred_flat, gt_flat)[0]
    
    mse = np.mean((pred_flat - gt_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - gt_flat))
    
    # Quantiles
    quantiles = [0.5, 0.9, 0.95, 0.99, 0.999]
    pred_quantiles = {f"p{int(q*1000)}": np.quantile(pred_flat, q) for q in quantiles}
    gt_quantiles = {f"p{int(q*1000)}": np.quantile(gt_flat, q) for q in quantiles}
    
    # Top-K overlap (per sample)
    top_k_overlaps = {10: [], 20: [], 50: []}
    
    for pred_sample, gt_sample in zip(all_pred, all_gt):
        pred_flat_sample = pred_sample.flatten()
        gt_flat_sample = gt_sample.flatten()
        
        for k in [10, 20, 50]:
            pred_topk_idx = set(np.argsort(pred_flat_sample)[-k:])
            gt_topk_idx = set(np.argsort(gt_flat_sample)[-k:])
            
            overlap = len(pred_topk_idx & gt_topk_idx) / k
            top_k_overlaps[k].append(overlap)
    
    avg_overlaps = {k: np.mean(v) for k, v in top_k_overlaps.items()}
    
    # Print results
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS - Checkpoint Epoch {epoch}")
    print(f"{'='*60}")
    
    print(f"\n📊 Standard Metrics:")
    print(f"  PCC:  {pcc:.4f}")
    print(f"  SPC:  {spc:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    
    print(f"\n📈 Quantiles:")
    print(f"  {'Quantile':<10} {'Pred':<10} {'GT':<10} {'Ratio':<10}")
    for q_name in ['p500', 'p900', 'p950', 'p990', 'p999']:
        pred_val = pred_quantiles[q_name]
        gt_val = gt_quantiles[q_name]
        ratio = pred_val / gt_val if gt_val != 0 else 0
        print(f"  {q_name:<10} {pred_val:<10.3f} {gt_val:<10.3f} {ratio:<10.3f}")
    
    print(f"\n🎯 Top-K Overlap (Peak Localization):")
    for k, overlap in avg_overlaps.items():
        print(f"  Top-{k:2d}: {overlap:.4f} ({overlap*100:.1f}%)")
    
    # Summary
    metrics = {
        'epoch': epoch,
        'num_samples': num_samples,
        'pcc': float(pcc),
        'spc': float(spc),
        'mse': float(mse),
        'mae': float(mae),
        'pred_quantiles': {k: float(v) for k, v in pred_quantiles.items()},
        'gt_quantiles': {k: float(v) for k, v in gt_quantiles.items()},
        'top_k_overlap': {f'top{k}': float(v) for k, v in avg_overlaps.items()}
    }
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    if avg_overlaps[20] < 0.4:
        print(f"  ⚠️  Top-20 overlap is low ({avg_overlaps[20]:.2f}) - peak localization needs improvement")
    elif avg_overlaps[20] < 0.6:
        print(f"  ⚠️  Top-20 overlap is moderate ({avg_overlaps[20]:.2f}) - room for improvement")
    else:
        print(f"  ✅ Top-20 overlap is good ({avg_overlaps[20]:.2f})")
    
    if abs(pred_quantiles['p990'] / gt_quantiles['p990'] - 1.0) < 0.1:
        print(f"  ✅ High quantiles match GT well (ratio={pred_quantiles['p990']/gt_quantiles['p990']:.3f})")
    else:
        print(f"  ⚠️  High quantiles mismatch (ratio={pred_quantiles['p990']/gt_quantiles['p990']:.3f})")
    
    if pcc > 0.9:
        print(f"  ✅ Strong correlation (PCC={pcc:.3f})")
    elif pcc > 0.8:
        print(f"  ⚠️  Moderate correlation (PCC={pcc:.3f})")
    else:
        print(f"  ⚠️  Weak correlation (PCC={pcc:.3f})")
    
    print(f"\n{'='*60}")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--val_hicarn', type=str,
                        default='/home/yangz/data/hic_data/HiCARN/hicarn_predictions/predictions_norm.npy',
                        help='Validation HiCARN predictions')
    parser.add_argument('--val_gt', type=str,
                        default='/home/yangz/data/hic_data/HiCARN/hicarn_predictions/ground_truth.npy',
                        help='Validation ground truth')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of validation samples')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for metrics')
    
    args = parser.parse_args()
    
    metrics = sample_and_evaluate(
        args.checkpoint,
        args.val_hicarn,
        args.val_gt,
        args.device,
        args.num_samples
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Metrics saved to: {args.output}")
