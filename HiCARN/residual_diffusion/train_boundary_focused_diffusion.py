"""
Boundary-Focused Diffusion Training Script

PHILOSOPHY CHANGE:
  - Goal: Improve TAD boundaries (NOT loops, NOT global metrics)
  - Reference: HiCARN (NOT ground truth)
  - Checkpointing: Insulation sharpness (NOT total loss)
  
Expected Results:
  - TAD F1: +1.5-3.5% (stable across chromosomes)
  - Insulation sharpness: +3-8%
  - Loop F1: ±1% (ACCEPT this, don't optimize)
  - PSNR may drop slightly (ACCEPT this if structure improves)

Usage:
    python train_boundary_focused_diffusion.py \
        --pred_path hicarn_predictions/predictions_norm.npy \
        --gt_path hicarn_predictions/ground_truth.npy \
        --output_dir checkpoints_boundary_focused \
        --epochs 50 \
        --batch_size 16 \
        --gpus 0 1 2 3
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import json
from pathlib import Path

# Import boundary-focused losses
from structure_losses_boundary_focused import (
    BoundaryFocusedLossCalculator,
    ResidualClipper
)

# Import model and scheduler
import sys
sys.path.insert(0, 'residual_diffusion')
from model import ResidualDiffusionUNet
from scheduler import DDPMScheduler
from data_loader import ResidualNormalizer


class ResidualDataset(Dataset):
    """Dataset for boundary-focused diffusion"""
    def __init__(self, pred_path, gt_path):
        self.pred_norm = np.load(pred_path)
        self.gt_norm = np.load(gt_path)
        self.residual = self.gt_norm - self.pred_norm
        
        print(f"Loaded data:")
        print(f"  HiCARN predictions: {self.pred_norm.shape}")
        print(f"  Ground truth: {self.gt_norm.shape}")
        print(f"  Residuals: {self.residual.shape}")
    
    def __len__(self):
        return len(self.pred_norm)
    
    def __getitem__(self, idx):
        hicarn_pred = self.pred_norm[idx]
        gt = self.gt_norm[idx]
        residual = self.residual[idx]
        
        if hicarn_pred.ndim == 2:
            hicarn_pred = hicarn_pred[None, :, :]
            gt = gt[None, :, :]
            residual = residual[None, :, :]
        
        return {
            'hicarn_pred': torch.from_numpy(hicarn_pred).float(),
            'gt': torch.from_numpy(gt).float(),
            'residual': torch.from_numpy(residual).float()
        }


def compute_insulation_sharpness(hic_matrix, window_size=5):
    """
    Compute insulation sharpness metric (for checkpoint selection)
    
    Higher = sharper boundaries = better
    """
    H = hic_matrix.shape[-1]
    w = min(window_size, (H - 1) // 2)
    
    if w < 2:
        return 0.0
    
    # Compute insulation
    scores = []
    for i in range(w, H - w):
        block = hic_matrix[..., i-w:i, i:i+w]
        score = block.mean()
        scores.append(score)
    
    if len(scores) < 2:
        return 0.0
    
    insulation = np.array(scores)
    insulation = np.log(insulation + 1e-4)
    
    # Sharpness = variance of gradient
    gradient = np.abs(np.diff(insulation))
    sharpness = np.var(gradient)
    
    return sharpness


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler_diff,
    loss_calculator,
    residual_clipper,
    normalizer,
    device,
    epoch,
    rank=0,
    world_size=1
):
    """Train for one epoch with boundary-focused losses"""
    model.train()
    
    # Track losses with counters
    epoch_losses = {
        'total_sum': 0.0,
        'diffusion_sum': 0.0,
        'insulation_sum': 0.0,
        'boundary_sum': 0.0,
        'low_freq_sum': 0.0,
        'n_total': 0,
        'n_diffusion': 0,
        'n_insulation': 0,
        'n_boundary': 0,
        'n_lf': 0
    }
    
    # Track boundary mask coverage
    mask_coverage_sum = 0.0
    n_masks = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for batch in pbar:
        hicarn_pred = batch['hicarn_pred'].to(device)
        gt = batch['gt'].to(device)
        target_residual = batch['residual'].to(device)
        
        batch_size = hicarn_pred.shape[0]
        
        # Normalize residual
        target_residual_norm = normalizer.transform(target_residual.cpu().numpy())
        target_residual_norm = torch.from_numpy(target_residual_norm).to(device)
        
        # Sample timestep
        t = torch.randint(0, scheduler_diff.num_train_timesteps,
                         (batch_size,), device=device).long()
        
        # Add noise
        noise = torch.randn_like(target_residual_norm)
        noisy_residual = scheduler_diff.add_noise(target_residual_norm, noise, t)
        
        # Predict noise
        noise_pred = model(noisy_residual, t, hicarn_pred)
        
        # Diffusion loss
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        # Estimate predicted residual
        with torch.no_grad():
            alpha_bar_t = scheduler_diff.alphas_cumprod[t].view(-1, 1, 1, 1)
            pred_residual_norm = (noisy_residual - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            pred_residual = normalizer.inverse_transform(pred_residual_norm.cpu().numpy())
            pred_residual = torch.from_numpy(pred_residual).to(device)
            pred_residual = residual_clipper.clip_residual(pred_residual, hicarn_pred)
        
        # Get boundary mask
        boundary_mask = loss_calculator.get_boundary_mask(hicarn_pred)
        
        # Track mask coverage
        mask_coverage_sum += boundary_mask.mean().item()
        n_masks += 1
        
        # Apply mask to predicted residual
        pred_residual_masked = pred_residual * boundary_mask
        pred_hic = hicarn_pred + pred_residual_masked
        
        # Structure loss (boundary-focused, HiCARN-relative)
        structure_loss, loss_dict = loss_calculator(
            pred_residual_masked, target_residual,
            pred_hic, gt, hicarn_pred
        )
        
        # Total loss
        total_loss = diffusion_loss + structure_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        epoch_losses['total_sum'] += total_loss.item()
        epoch_losses['n_total'] += 1
        
        epoch_losses['diffusion_sum'] += diffusion_loss.item()
        epoch_losses['n_diffusion'] += 1
        
        if loss_dict['insulation'] is not None:
            epoch_losses['insulation_sum'] += loss_dict['insulation']
            epoch_losses['n_insulation'] += 1
        
        if loss_dict['boundary'] is not None:
            epoch_losses['boundary_sum'] += loss_dict['boundary']
            epoch_losses['n_boundary'] += 1
        
        if loss_dict['low_freq'] is not None:
            epoch_losses['low_freq_sum'] += loss_dict['low_freq']
            epoch_losses['n_lf'] += 1
        
        # Update progress bar
        if rank == 0:
            pbar.set_postfix({
                'total': f"{total_loss.item():.4f}",
                'diff': f"{diffusion_loss.item():.4f}",
                'ins': f"{loss_dict['insulation']:.4f}" if loss_dict['insulation'] else "N/A",
                'bnd': f"{loss_dict['boundary']:.4f}" if loss_dict['boundary'] else "N/A",
                'mask%': f"{boundary_mask.mean().item()*100:.1f}"
            })
    
    # Compute averages
    avg_losses = {
        'total': epoch_losses['total_sum'] / max(epoch_losses['n_total'], 1),
        'diffusion': epoch_losses['diffusion_sum'] / max(epoch_losses['n_diffusion'], 1),
        'insulation': epoch_losses['insulation_sum'] / max(epoch_losses['n_insulation'], 1),
        'boundary': epoch_losses['boundary_sum'] / max(epoch_losses['n_boundary'], 1),
        'low_freq': epoch_losses['low_freq_sum'] / max(epoch_losses['n_lf'], 1),
        'valid_insulation_rate': epoch_losses['n_insulation'] / max(epoch_losses['n_total'], 1),
        'valid_boundary_rate': epoch_losses['n_boundary'] / max(epoch_losses['n_total'], 1),
        'avg_mask_coverage': mask_coverage_sum / max(n_masks, 1)
    }
    
    # Sync across GPUs
    if world_size > 1:
        for key in avg_losses.keys():
            loss_tensor = torch.tensor(avg_losses[key], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_losses[key] = loss_tensor.item()
    
    return avg_losses


@torch.no_grad()
def validate(
    model,
    dataloader,
    scheduler_diff,
    loss_calculator,
    residual_clipper,
    normalizer,
    device,
    rank=0,
    world_size=1
):
    """
    Validation with structure-aware metrics
    
    CRITICAL: Track insulation sharpness (for checkpoint selection)
    """
    model.eval()
    
    val_losses = {
        'total_sum': 0.0,
        'insulation_sum': 0.0,
        'boundary_sum': 0.0,
        'n_total': 0,
        'n_insulation': 0,
        'n_boundary': 0
    }
    
    # Track insulation sharpness
    sharpness_hicarn_sum = 0.0
    sharpness_refined_sum = 0.0
    n_samples = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc="Validating", leave=False)
    else:
        pbar = dataloader
    
    for batch in pbar:
        hicarn_pred = batch['hicarn_pred'].to(device)
        gt = batch['gt'].to(device)
        target_residual = batch['residual'].to(device)
        
        batch_size = hicarn_pred.shape[0]
        
        # Denoise from random noise (fast validation with 10 steps)
        residual_t = torch.randn_like(hicarn_pred)
        
        for t_val in reversed(range(0, scheduler_diff.num_train_timesteps, 100)):
            t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            noise_pred = model(residual_t, t_tensor, hicarn_pred)
            residual_t, _ = scheduler_diff.step(noise_pred, t_val, residual_t)
        
        # Denormalize
        pred_residual = normalizer.inverse_transform(residual_t.cpu().numpy())
        pred_residual = torch.from_numpy(pred_residual).to(device)
        pred_residual = residual_clipper.clip_residual(pred_residual, hicarn_pred)
        
        # Apply boundary mask
        boundary_mask = loss_calculator.get_boundary_mask(hicarn_pred)
        pred_residual_masked = pred_residual * boundary_mask
        pred_hic = hicarn_pred + pred_residual_masked
        
        # Compute losses
        structure_loss, loss_dict = loss_calculator(
            pred_residual_masked, target_residual,
            pred_hic, gt, hicarn_pred
        )
        
        # Track losses
        val_losses['total_sum'] += structure_loss.item()
        val_losses['n_total'] += 1
        
        if loss_dict['insulation'] is not None:
            val_losses['insulation_sum'] += loss_dict['insulation']
            val_losses['n_insulation'] += 1
        
        if loss_dict['boundary'] is not None:
            val_losses['boundary_sum'] += loss_dict['boundary']
            val_losses['n_boundary'] += 1
        
        # Compute insulation sharpness
        for i in range(batch_size):
            sharp_hicarn = compute_insulation_sharpness(hicarn_pred[i, 0].cpu().numpy())
            sharp_refined = compute_insulation_sharpness(pred_hic[i, 0].cpu().numpy())
            
            sharpness_hicarn_sum += sharp_hicarn
            sharpness_refined_sum += sharp_refined
            n_samples += 1
    
    # Compute averages
    avg_val = {
        'total': val_losses['total_sum'] / max(val_losses['n_total'], 1),
        'insulation': val_losses['insulation_sum'] / max(val_losses['n_insulation'], 1),
        'boundary': val_losses['boundary_sum'] / max(val_losses['n_boundary'], 1),
        'sharpness_hicarn': sharpness_hicarn_sum / max(n_samples, 1),
        'sharpness_refined': sharpness_refined_sum / max(n_samples, 1),
        'sharpness_improvement': (sharpness_refined_sum - sharpness_hicarn_sum) / max(sharpness_hicarn_sum, 1e-6) * 100
    }
    
    # Sync across GPUs
    if world_size > 1:
        for key in avg_val.keys():
            val_tensor = torch.tensor(avg_val[key], device=device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            avg_val[key] = val_tensor.item()
    
    return avg_val


def main_worker(rank, world_size, args, gpu_ids):
    """Main training worker"""
    
    # Setup distributed
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Device
    if world_size > 1:
        device_idx = gpu_ids[rank]
        device = torch.device(f'cuda:{device_idx}')
        torch.cuda.set_device(device)
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    is_main_process = (rank == 0)
    
    if is_main_process:
        print("\n" + "="*80)
        print("BOUNDARY-FOCUSED DIFFUSION TRAINING")
        print("PHILOSOPHY: TAD refiner (NOT loop enhancer)")
        print("="*80)
        print(f"\nLoss Weights:")
        print(f"  λ_diffusion:   {args.lambda_diffusion}")
        print(f"  λ_insulation:  {args.lambda_insulation} (HiCARN-relative)")
        print(f"  λ_boundary:    {args.lambda_boundary} (boundary-only)")
        print(f"  λ_low_freq:    {args.lambda_low_freq}")
        print("="*80)
    
    # Load dataset
    if is_main_process:
        print("\n[1/6] Loading dataset...")
    
    dataset = ResidualDataset(args.pred_path, args.gt_path)
    
    # Split
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    if is_main_process:
        print(f"  Train samples: {n_train}")
        print(f"  Val samples: {n_val}")
    
    # Dataloaders
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    
    # Create model
    if is_main_process:
        print("\n[2/6] Creating model...")
    
    model = ResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index,
                   find_unused_parameters=(args.freeze_backbone_epochs > 0))
    
    if is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")
    
    # Create scheduler
    if is_main_process:
        print("\n[3/6] Creating diffusion scheduler...")
    scheduler_diff = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear')
    
    # Create loss calculator (boundary-focused!)
    if is_main_process:
        print("\n[4/6] Creating BOUNDARY-FOCUSED loss calculator...")
        print(f"  Using HiCARN as reference (not GT)")
        print(f"  Boundary mask coverage: ~{args.dilation_radius*10}%")
    
    loss_calculator = BoundaryFocusedLossCalculator(
        lambda_diffusion=args.lambda_diffusion,
        lambda_insulation=args.lambda_insulation,
        lambda_boundary=args.lambda_boundary,
        lambda_low_freq=args.lambda_low_freq,
        insulation_window=args.insulation_window,
        boundary_alpha=args.boundary_alpha,
        use_boundary_mask=True,
        dilation_radius=args.dilation_radius
    ).to(device)
    
    residual_clipper = ResidualClipper(clip_factor=args.clip_factor)
    
    # Fit normalizer
    if is_main_process:
        print("\n[5/6] Fitting residual normalizer...")
    normalizer = ResidualNormalizer()
    all_residuals = dataset.dataset.residual if hasattr(dataset, 'dataset') else dataset.residual
    normalizer.fit(all_residuals)
    if is_main_process:
        print(f"  Residual mean: {normalizer.mean:.6f}")
        print(f"  Residual std:  {normalizer.std:.6f}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Training
    if is_main_process:
        print("\n[6/6] Starting boundary-focused training...")
        print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_history = []
    best_sharpness_improvement = -float('inf')  # CRITICAL: Use sharpness, not loss!
    
    for epoch in range(args.epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, scheduler_diff,
            loss_calculator, residual_clipper, normalizer,
            device, epoch, rank, world_size
        )
        
        # Validate (every N epochs)
        should_validate = (epoch % args.val_frequency == 0) or (epoch == args.epochs - 1)
        
        if should_validate:
            val_losses = validate(
                model, val_loader, scheduler_diff,
                loss_calculator, residual_clipper, normalizer,
                device, rank, world_size
            )
        else:
            if epoch > 0 and len(training_history) > 0:
                val_losses = training_history[-1]['val']
            else:
                val_losses = {k: 0.0 for k in ['total', 'insulation', 'boundary', 'sharpness_improvement']}
        
        # Log (only on rank 0)
        if is_main_process:
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Total: {train_losses['total']:.4f}, "
                  f"Ins: {train_losses['insulation']:.4f}, "
                  f"Bnd: {train_losses['boundary']:.4f}")
            print(f"  Valid - Ins: {train_losses['valid_insulation_rate']*100:.1f}%, "
                  f"Bnd: {train_losses['valid_boundary_rate']*100:.1f}%, "
                  f"Mask: {train_losses['avg_mask_coverage']*100:.1f}%")
            
            if should_validate:
                print(f"  Val   - Total: {val_losses['total']:.4f}, "
                      f"Ins: {val_losses['insulation']:.4f}, "
                      f"Bnd: {val_losses['boundary']:.4f}")
                print(f"  Sharpness - HiCARN: {val_losses['sharpness_hicarn']:.6f}, "
                      f"Refined: {val_losses['sharpness_refined']:.6f}, "
                      f"Improvement: {val_losses['sharpness_improvement']:+.2f}%")
            
            # Save history
            training_history.append({
                'epoch': epoch,
                'train': train_losses,
                'val': val_losses if should_validate else {}
            })
            
            with open(output_dir / 'training_history.json', 'w') as f:
                json.dump(training_history, f, indent=2)
            
            # CRITICAL: Save checkpoint based on SHARPNESS, not loss!
            if should_validate and val_losses['sharpness_improvement'] > best_sharpness_improvement:
                best_sharpness_improvement = val_losses['sharpness_improvement']
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'sharpness_improvement': best_sharpness_improvement,
                    'normalizer_mean': normalizer.mean,
                    'normalizer_std': normalizer.std,
                    'config': {
                        'base_channels': 64,
                        'channel_multipliers': [1, 2, 4, 8],
                        'num_res_blocks': 2
                    }
                }
                
                torch.save(checkpoint, output_dir / 'best_boundary_focused.pt')
                print(f"  ✓ Saved best model (sharpness improvement: {best_sharpness_improvement:+.2f}%)")
            
            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'normalizer_mean': normalizer.mean,
                    'normalizer_std': normalizer.std
                }
                torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model
    if is_main_process:
        final_checkpoint = {
            'epoch': args.epochs - 1,
            'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'normalizer_mean': normalizer.mean,
            'normalizer_std': normalizer.std
        }
        torch.save(final_checkpoint, output_dir / 'final_boundary_focused.pt')
        print("\n✓ Training complete!")
        print(f"  Best sharpness improvement: {best_sharpness_improvement:+.2f}%")
    
    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Boundary-Focused Diffusion Training")
    
    # Data
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints_boundary_focused')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='50 epochs recommended (not 100)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--val_frequency', type=int, default=5)
    
    # Loss weights (boundary-focused)
    parser.add_argument('--lambda_diffusion', type=float, default=1.0)
    parser.add_argument('--lambda_insulation', type=float, default=2.0,
                       help='Increased from 1.0 (main target)')
    parser.add_argument('--lambda_boundary', type=float, default=1.5,
                       help='New: boundary-specific loss')
    parser.add_argument('--lambda_low_freq', type=float, default=0.3,
                       help='Decreased from 0.5 (less important)')
    
    # Boundary-focused parameters
    parser.add_argument('--insulation_window', type=int, default=5)
    parser.add_argument('--boundary_alpha', type=float, default=0.5,
                       help='Weight for HiCARN baseline in relative loss')
    parser.add_argument('--dilation_radius', type=int, default=2,
                       help='Boundary mask dilation (2-3 recommended)')
    parser.add_argument('--clip_factor', type=float, default=0.1)
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    
    # Backbone freezing
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0)
    
    args = parser.parse_args()
    
    # Multi-GPU setup
    if args.gpus is not None:
        gpu_ids = args.gpus
        world_size = len(gpu_ids)
        print(f"Starting multi-GPU training on GPUs: {gpu_ids}")
        print(f"World size: {world_size}")
        mp.spawn(main_worker, args=(world_size, args, gpu_ids), nprocs=world_size)
    else:
        main_worker(0, 1, args, [args.gpu])


if __name__ == "__main__":
    main()
