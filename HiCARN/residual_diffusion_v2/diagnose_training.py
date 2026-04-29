"""
Quick Diagnostic Script - Check Training Health

Run this during training to diagnose:
1. Model collapse (outputting near-zero residuals)
2. Loss balance (diff vs recon contribution)
3. Prediction quality

Usage:
  python diagnose_training.py --checkpoint checkpoints_v2/checkpoint_epoch_10.pt
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from data_loader import ResidualDiffusionDataset
from model import ImprovedResidualDiffusionUNet
from scheduler import ImprovedDDPMScheduler


def diagnose_checkpoint(checkpoint_path, data_hicarn, data_gt, device='cuda'):
    """Diagnose training health from checkpoint"""
    
    print("=" * 80)
    print("TRAINING DIAGNOSTIC")
    print("=" * 80)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')
    train_loss = checkpoint.get('train_loss', 'unknown')
    
    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {epoch}")
    print(f"  Train Loss: {train_loss}")
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scheduler
    scheduler = ImprovedDDPMScheduler(
        num_train_timesteps=1000,
        parameterization='v'
    )
    
    # Load data
    print(f"\nLoading data...")
    dataset = ResidualDiffusionDataset(data_hicarn, data_gt)
    
    # Sample a batch
    batch_size = 16
    indices = np.random.choice(len(dataset), batch_size, replace=False)
    
    batch_data = [dataset[i] for i in indices]
    condition = torch.stack([b['hicarn'] for b in batch_data]).to(device)
    gt = torch.stack([b['gt'] for b in batch_data]).to(device)
    residual_clean = torch.stack([b['residual'] for b in batch_data]).to(device)
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS 1: Model Collapse Check")
    print("="*80)
    
    with torch.no_grad():
        # Sample various timesteps
        timesteps_to_check = [10, 100, 500, 900]
        
        for t_val in timesteps_to_check:
            timesteps = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            
            # Add noise
            noise = torch.randn_like(residual_clean)
            residual_noisy = scheduler.add_noise(residual_clean, noise, timesteps)
            
            # Predict
            model_output = model(residual_noisy, timesteps, condition)
            
            # Get predicted clean residual
            if scheduler.parameterization == 'v':
                pred_residual_clean = scheduler.predict_start_from_v(
                    residual_noisy, timesteps[0], model_output
                )
            else:
                pred_residual_clean = scheduler.predict_start_from_noise(
                    residual_noisy, timesteps[0], model_output
                )
            
            # Statistics
            gt_mean = residual_clean.abs().mean().item()
            gt_std = residual_clean.std().item()
            gt_max = residual_clean.abs().max().item()
            
            pred_mean = pred_residual_clean.abs().mean().item()
            pred_std = pred_residual_clean.std().item()
            pred_max = pred_residual_clean.abs().max().item()
            
            print(f"\nTimestep t={t_val}:")
            print(f"  GT Residual:   mean_abs={gt_mean:.4f}, std={gt_std:.4f}, max_abs={gt_max:.4f}")
            print(f"  Pred Residual: mean_abs={pred_mean:.4f}, std={pred_std:.4f}, max_abs={pred_max:.4f}")
            print(f"  Ratio (pred/gt): mean={pred_mean/gt_mean:.4f}, std={pred_std/gt_std:.4f}")
            
            if pred_mean < 0.1 * gt_mean:
                print(f"  ⚠️  WARNING: Model collapse! Predictions too small!")
            elif pred_mean < 0.5 * gt_mean:
                print(f"  ⚠️  Predictions weaker than GT (may be learning to be conservative)")
            elif pred_mean > 2.0 * gt_mean:
                print(f"  ⚠️  Predictions too large (may be overshooting)")
            else:
                print(f"  ✅ Magnitude looks reasonable")
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS 2: Loss Component Balance")
    print("="*80)
    
    # Load loss function
    from losses import CombinedResidualLoss, get_loss_weights
    
    loss_weights = get_loss_weights('peak_focused')
    criterion = CombinedResidualLoss(**loss_weights).to(device)
    
    with torch.no_grad():
        # Use t=500 as representative
        timesteps = torch.full((batch_size,), 500, device=device, dtype=torch.long)
        noise = torch.randn_like(residual_clean)
        residual_noisy = scheduler.add_noise(residual_clean, noise, timesteps)
        
        # Predict
        model_output = model(residual_noisy, timesteps, condition)
        
        # Get target
        if scheduler.parameterization == 'v':
            target = scheduler.get_v(residual_clean, noise, timesteps)
            pred_residual_clean = scheduler.predict_start_from_v(
                residual_noisy, timesteps[0], model_output
            )
        else:
            target = noise
            pred_residual_clean = scheduler.predict_start_from_noise(
                residual_noisy, timesteps[0], model_output
            )
        
        # Diffusion loss
        diffusion_loss = torch.nn.functional.mse_loss(model_output, target)
        
        # Reconstruction loss
        pred_gt = condition + pred_residual_clean
        recon_loss, loss_components = criterion(pred_gt, gt, return_components=True)
        
        # Combined
        total_loss = diffusion_loss + 0.5 * recon_loss
        
        print(f"\nLoss Components (current weight=0.5 for recon):")
        print(f"  Diffusion loss:      {diffusion_loss.item():.4f}")
        print(f"  Recon loss (raw):    {recon_loss.item():.4f}")
        print(f"  Recon loss (0.5*):   {0.5*recon_loss.item():.4f}")
        print(f"  Total loss:          {total_loss.item():.4f}")
        print(f"\n  Contribution to total:")
        print(f"    Diffusion: {diffusion_loss.item()/total_loss.item()*100:.1f}%")
        print(f"    Recon:     {0.5*recon_loss.item()/total_loss.item()*100:.1f}%")
        
        if loss_components:
            print(f"\n  Recon loss breakdown:")
            for key, value in loss_components.items():
                if key != 'total':
                    print(f"    {key}: {value:.4f}")
        
        # Recommendations
        print(f"\n  Recommendations:")
        recon_ratio = (0.5*recon_loss.item()) / diffusion_loss.item()
        if recon_ratio > 4.0:
            print(f"    ⚠️  Recon loss dominates (ratio={recon_ratio:.1f}:1)")
            print(f"    → Try reducing recon weight to 0.2 or 0.1")
        elif recon_ratio > 2.0:
            print(f"    ⚠️  Recon loss is strong (ratio={recon_ratio:.1f}:1)")
            print(f"    → Consider reducing to 0.3")
        else:
            print(f"    ✅ Loss balance looks reasonable (ratio={recon_ratio:.1f}:1)")
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS 3: Prediction Quality on Sample")
    print("="*80)
    
    # Take first sample and visualize statistics
    sample_idx = 0
    with torch.no_grad():
        cond_sample = condition[sample_idx:sample_idx+1]
        gt_sample = gt[sample_idx:sample_idx+1]
        res_sample = residual_clean[sample_idx:sample_idx+1]
        
        # Denoise from pure noise
        timesteps = torch.full((1,), 500, device=device, dtype=torch.long)
        noise = torch.randn_like(res_sample)
        residual_noisy = scheduler.add_noise(res_sample, noise, timesteps)
        
        model_output = model(residual_noisy, timesteps, cond_sample)
        
        if scheduler.parameterization == 'v':
            pred_res = scheduler.predict_start_from_v(
                residual_noisy, timesteps[0], model_output
            )
        else:
            pred_res = scheduler.predict_start_from_noise(
                residual_noisy, timesteps[0], model_output
            )
        
        pred_gt_sample = cond_sample + pred_res
        
        # Convert to numpy for stats
        gt_np = gt_sample.cpu().numpy()[0, 0]
        pred_np = pred_gt_sample.cpu().numpy()[0, 0]
        res_gt_np = res_sample.cpu().numpy()[0, 0]
        res_pred_np = pred_res.cpu().numpy()[0, 0]
        
        print(f"\nSample statistics:")
        print(f"  GT range: [{gt_np.min():.3f}, {gt_np.max():.3f}]")
        print(f"  Pred range: [{pred_np.min():.3f}, {pred_np.max():.3f}]")
        print(f"  GT residual range: [{res_gt_np.min():.3f}, {res_gt_np.max():.3f}]")
        print(f"  Pred residual range: [{res_pred_np.min():.3f}, {res_pred_np.max():.3f}]")
        
        # Correlation
        from scipy.stats import pearsonr, spearmanr
        pcc = pearsonr(gt_np.flatten(), pred_np.flatten())[0]
        spc = spearmanr(gt_np.flatten(), pred_np.flatten())[0]
        
        print(f"\n  Correlation (GT vs Pred):")
        print(f"    PCC: {pcc:.4f}")
        print(f"    SPC: {spc:.4f}")
        
        # Top-K overlap (simple version)
        k = 20
        gt_flat = gt_np.flatten()
        pred_flat = pred_np.flatten()
        
        gt_topk_idx = np.argsort(gt_flat)[-k:]
        pred_topk_idx = np.argsort(pred_flat)[-k:]
        
        overlap = len(set(gt_topk_idx) & set(pred_topk_idx))
        
        print(f"\n  Top-{k} overlap: {overlap}/{k} = {overlap/k*100:.1f}%")
        
        if overlap < k * 0.3:
            print(f"    ⚠️  Low overlap - peak positions may be off")
        elif overlap < k * 0.5:
            print(f"    ⚠️  Moderate overlap - room for improvement")
        else:
            print(f"    ✅ Good overlap")
    
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n1. Check validation metrics on held-out chr")
    print(f"2. If plateau continues, try:")
    print(f"   - Reduce recon weight: 0.5 → 0.2 or 0.1")
    print(f"   - Increase learning rate: 1e-4 → 2e-4")
    print(f"   - Add learning rate scheduler (cosine annealing)")
    print(f"   - Adjust loss strategy weights")
    print(f"3. Monitor for overfitting (train vs val gap)")
    print(f"4. Consider early stopping if val metrics plateau")
    
    print(f"\n{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint to diagnose')
    parser.add_argument('--data_hicarn', type=str, 
                        default='/home/yangz/data/hic_data/HiCARN/hicarn_predictions/predictions_norm.npy')
    parser.add_argument('--data_gt', type=str,
                        default='/home/yangz/data/hic_data/HiCARN/hicarn_predictions/ground_truth.npy')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    diagnose_checkpoint(args.checkpoint, args.data_hicarn, args.data_gt, args.device)
