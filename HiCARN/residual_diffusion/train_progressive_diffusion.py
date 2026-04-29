"""
Progressive Structure Refinement Training

PHILOSOPHY: "先活下来 → 再慢慢变好"

Stage 1 (0-20 epochs): Force diffusion to DO SOMETHING
  - Enable: residual, consistency, low-freq, smooth
  - Disable: improvement
  - Goal: Residual variance > 0

Stage 2 (20+ epochs): Gently push for improvement
  - Enable: ALL losses
  - Gradually increase λ_improvement
  - Goal: TAD metrics improve

Expected Training Pattern:
  Epochs 0-10:   Residual var increases (0.001 → 0.01)
  Epochs 10-20:  Structure maintained, var stabilizes
  Epochs 20-30:  Gradual TAD improvement begins
  Epochs 30-50:  Steady improvement continues

Usage:
    python train_progressive_diffusion.py \
        --pred_path hicarn_predictions/predictions_norm.npy \
        --gt_path hicarn_predictions/ground_truth.npy \
        --output_dir checkpoints_progressive \
        --epochs 50 \
        --batch_size 16 \
        --gpus 0 1 2 3 \
        --stage1_epochs 20
"""

import argparse
import os
import sys
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

# Import progressive losses (ROBUST version)
from structure_losses_progressive_robust import (
    ProgressiveStructureLossCalculator,
    ResidualClipper
)

# Import TAD-only residual constructor
from tad_only_residual import construct_tad_only_residual_numpy

# Import model and scheduler
import sys
sys.path.insert(0, 'residual_diffusion')
from model import ResidualDiffusionUNet
from scheduler import DDPMScheduler
from data_loader import ResidualNormalizer


class ResidualDataset(Dataset):
    """
    Dataset for progressive diffusion with TAD-only residual
    
    CRITICAL CHANGE:
        Instead of full residual = gt - pred,
        We construct TAD-only residual:
            1. Low-frequency extraction (removes loop peaks)
            2. Loop masking (excludes high-intensity regions)
            3. residual_tad = residual_lf * (1 - loop_mask)
        
        This ensures:
            - Diffusion only modifies TAD structure
            - Loop peaks are frozen (handled by HiCARN)
            - Goal: TAD↑, Loop≈, PSNR/SSIM≈
    """
    def __init__(self, pred_path, gt_path, kernel_size=5, loop_percentile=90):
        self.pred_norm = np.load(pred_path)
        self.gt_norm = np.load(gt_path)
        
        # Original residual (for comparison)
        self.residual_raw = self.gt_norm - self.pred_norm
        
        # Construct TAD-only residual for each sample
        print(f"\nConstructing TAD-only residuals...")
        print(f"  Kernel size: {kernel_size} (low-pass filter)")
        print(f"  Loop percentile: {loop_percentile} (masking threshold)")
        
        residuals_tad = []
        loop_masks = []
        
        for i in range(len(self.pred_norm)):
            pred_sample = self.pred_norm[i]
            gt_sample = self.gt_norm[i]
            
            # Construct TAD-only residual
            residual_tad, loop_mask = construct_tad_only_residual_numpy(
                pred_sample, gt_sample, kernel_size, loop_percentile
            )
            
            residuals_tad.append(residual_tad)
            loop_masks.append(loop_mask)
        
        self.residual = np.array(residuals_tad)
        self.loop_masks = np.array(loop_masks)
        
        # Statistics
        loop_fraction = self.loop_masks.mean()
        raw_std = self.residual_raw.std()
        tad_std = self.residual.std()
        
        print(f"\nLoaded data:")
        print(f"  HiCARN predictions: {self.pred_norm.shape}")
        print(f"  Ground truth: {self.gt_norm.shape}")
        print(f"  TAD-only residuals: {self.residual.shape}")
        print(f"\nTAD-only residual statistics:")
        print(f"  Loop regions masked: {loop_fraction*100:.1f}%")
        print(f"  Raw residual std: {raw_std:.6f}")
        print(f"  TAD residual std: {tad_std:.6f}")
        print(f"  Std reduction: {(raw_std - tad_std)/raw_std*100:.1f}%")
        print(f"  TAD residual range: [{self.residual.min():.4f}, {self.residual.max():.4f}]")
        print(f"  TAD residual variance: {self.residual.var():.6f}")
        print(f"\n✅ TAD-only residual: Diffusion will only modify TAD, not loops!")
    
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
            'residual': torch.from_numpy(residual).float()  # TAD-only residual
        }


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
    """Train for one epoch with progressive losses"""
    model.train()
    
    # Track losses with counters
    epoch_losses = {
        'total_sum': 0.0,
        'residual_sum': 0.0,
        'directional_sum': 0.0,  # NEW
        'consistency_sum': 0.0,
        'improvement_sum': 0.0,
        'low_freq_sum': 0.0,
        'boundary_sum': 0.0,  # NEW: TAD boundary
        'n_total': 0,
        'n_residual': 0,
        'n_directional': 0,
        'n_consistency': 0,
        'n_improvement': 0,
        'n_low_freq': 0,
        'n_boundary': 0  # NEW
    }
    
    # Track residual variance, std, and cosine similarity (critical metrics!)
    pred_residual_var_sum = 0.0
    target_residual_var_sum = 0.0
    pred_residual_std_sum = 0.0
    target_residual_std_sum = 0.0
    rsr_sum = 0.0  # Residual-to-Signal Ratio
    cos_sim_sum = 0.0  # NEW: Directional alignment
    n_var_samples = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Stage {1 if epoch < loss_calculator.stage1_epochs else 2}]")
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
        
        # Estimate predicted residual
        with torch.no_grad():
            alpha_bar_t = scheduler_diff.alphas_cumprod[t].view(-1, 1, 1, 1)
            pred_residual_norm = (noisy_residual - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            pred_residual = normalizer.inverse_transform(pred_residual_norm.cpu().numpy())
            pred_residual = torch.from_numpy(pred_residual).to(device)
            pred_residual = residual_clipper.clip_residual(pred_residual, hicarn_pred)
        
        # Predicted Hi-C
        pred_hic = hicarn_pred + pred_residual
        
        # Progressive structure loss
        structure_loss, loss_dict = loss_calculator(
            pred_residual, target_residual,
            pred_hic, hicarn_pred, gt,
            epoch=epoch
        )
        
        # Diffusion loss
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
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
        
        if loss_dict['residual'] is not None:
            epoch_losses['residual_sum'] += loss_dict['residual']
            epoch_losses['n_residual'] += 1
        
        if loss_dict['directional'] is not None:
            epoch_losses['directional_sum'] += loss_dict['directional']
            epoch_losses['n_directional'] += 1
        
        if loss_dict['consistency'] is not None:
            epoch_losses['consistency_sum'] += loss_dict['consistency']
            epoch_losses['n_consistency'] += 1
        
        if loss_dict['improvement'] is not None:
            epoch_losses['improvement_sum'] += loss_dict['improvement']
            epoch_losses['n_improvement'] += 1
        
        if loss_dict['low_freq'] is not None:
            epoch_losses['low_freq_sum'] += loss_dict['low_freq']
            epoch_losses['n_low_freq'] += 1
        
        if loss_dict['boundary'] is not None:
            epoch_losses['boundary_sum'] += loss_dict['boundary']
            epoch_losses['n_boundary'] += 1
        
        # Track residual variance, std, RSR, and cosine similarity (CRITICAL METRICS!)
        pred_var = torch.var(pred_residual).item()
        target_var = torch.var(target_residual).item()
        pred_std = torch.std(pred_residual).item()
        target_std = torch.std(target_residual).item()
        
        # RSR: Residual-to-Signal Ratio
        # < 1%: identity trick
        # 1-5%: barely alive
        # 5-15%: healthy structure refiner
        # > 20%: structure破坏风险
        residual_norm = torch.norm(pred_residual).item()
        signal_norm = torch.norm(hicarn_pred).item()
        rsr = (residual_norm / (signal_norm + 1e-8)) * 100  # As percentage
        
        # NEW: Cosine similarity (directional alignment)
        pred_flat = pred_residual.flatten()
        target_flat = target_residual.flatten()
        cos_sim = F.cosine_similarity(
            pred_flat.unsqueeze(0),
            target_flat.unsqueeze(0),
            dim=1
        ).mean().item()
        
        pred_residual_var_sum += pred_var
        target_residual_var_sum += target_var
        pred_residual_std_sum += pred_std
        target_residual_std_sum += target_std
        rsr_sum += rsr
        cos_sim_sum += cos_sim  # NEW
        n_var_samples += 1
        
        # Update progress bar
        if rank == 0:
            pbar.set_postfix({
                'total': f"{total_loss.item():.4f}",
                'diff': f"{diffusion_loss.item():.4f}",
                'res': f"{loss_dict['residual']:.4f}" if loss_dict['residual'] else "N/A",
                'cons': f"{loss_dict['consistency']:.4f}" if loss_dict['consistency'] else "N/A",
                'imp': f"{loss_dict['improvement']:.4f}" if loss_dict['improvement'] else "OFF",
                'var': f"{pred_var:.6f}"
            })
    
    # Compute averages
    avg_losses = {
        'total': epoch_losses['total_sum'] / max(epoch_losses['n_total'], 1),
        'residual': epoch_losses['residual_sum'] / max(epoch_losses['n_residual'], 1) if epoch_losses['n_residual'] > 0 else None,
        'directional': epoch_losses['directional_sum'] / max(epoch_losses['n_directional'], 1) if epoch_losses['n_directional'] > 0 else None,
        'consistency': epoch_losses['consistency_sum'] / max(epoch_losses['n_consistency'], 1) if epoch_losses['n_consistency'] > 0 else None,
        'improvement': epoch_losses['improvement_sum'] / max(epoch_losses['n_improvement'], 1) if epoch_losses['n_improvement'] > 0 else 0.0,
        'low_freq': epoch_losses['low_freq_sum'] / max(epoch_losses['n_low_freq'], 1) if epoch_losses['n_low_freq'] > 0 else None,
        'boundary': epoch_losses['boundary_sum'] / max(epoch_losses['n_boundary'], 1) if epoch_losses['n_boundary'] > 0 else None,  # NEW
        'pred_residual_var': pred_residual_var_sum / max(n_var_samples, 1),
        'target_residual_var': target_residual_var_sum / max(n_var_samples, 1),
        'pred_residual_std': pred_residual_std_sum / max(n_var_samples, 1),
        'target_residual_std': target_residual_std_sum / max(n_var_samples, 1),
        'rsr': rsr_sum / max(n_var_samples, 1),  # Residual-to-Signal Ratio %
        'cos_sim': cos_sim_sum / max(n_var_samples, 1),  # Directional alignment
        'stage': 1 if epoch < loss_calculator.stage1_epochs else 2,
        'n_valid_losses': int(
            (epoch_losses['n_residual'] > 0) + 
            (epoch_losses['n_directional'] > 0) + 
            (epoch_losses['n_consistency'] > 0) + 
            (epoch_losses['n_improvement'] > 0) + 
            (epoch_losses['n_low_freq'] > 0) + 
            (epoch_losses['n_boundary'] > 0)  # Changed from smooth
        )
    }
    
    # Sync across GPUs (skip None values)
    if world_size > 1:
        for key in avg_losses.keys():
            if key != 'stage' and key != 'n_valid_losses' and avg_losses[key] is not None:
                loss_tensor = torch.tensor(avg_losses[key], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                avg_losses[key] = loss_tensor.item()
    
    # NO barrier here - will sync in main loop
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
    epoch,
    rank=0,
    world_size=1
):
    """
    Fast validation - just compute losses on a few samples
    Skip the expensive denoising process
    """
    model.eval()
    
    val_losses = {
        'total_sum': 0.0,
        'residual_sum': 0.0,
        'consistency_sum': 0.0,
        'n_total': 0,
        'n_residual': 0,
        'n_consistency': 0
    }
    
    pred_var_sum = 0.0
    n_samples = 0
    
    # Only validate on first 10 batches for speed
    max_batches = 10
    
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        
        hicarn_pred = batch['hicarn_pred'].to(device)
        gt = batch['gt'].to(device)
        target_residual = batch['residual'].to(device)
        
        # Just use the target residual directly (no denoising)
        pred_residual = target_residual
        pred_hic = hicarn_pred + pred_residual
        
        # Compute losses
        structure_loss, loss_dict = loss_calculator(
            pred_residual, target_residual,
            pred_hic, hicarn_pred, gt,
            epoch=epoch
        )
        
        # Track
        val_losses['total_sum'] += structure_loss.item()
        val_losses['n_total'] += 1
        
        if loss_dict['residual'] is not None:
            val_losses['residual_sum'] += loss_dict['residual']
            val_losses['n_residual'] += 1
        
        if loss_dict['consistency'] is not None:
            val_losses['consistency_sum'] += loss_dict['consistency']
            val_losses['n_consistency'] += 1
        
        # Track variance
        pred_var_sum += torch.var(pred_residual).item()
        n_samples += 1
    
    # Averages
    avg_val = {
        'total': val_losses['total_sum'] / max(val_losses['n_total'], 1),
        'residual': val_losses['residual_sum'] / max(val_losses['n_residual'], 1) if val_losses['n_residual'] > 0 else None,
        'consistency': val_losses['consistency_sum'] / max(val_losses['n_consistency'], 1) if val_losses['n_consistency'] > 0 else None,
        'pred_residual_var': pred_var_sum / max(n_samples, 1)
    }
    
    # Sync (skip None values)
    if world_size > 1:
        for key in avg_val.keys():
            if avg_val[key] is not None:
                val_tensor = torch.tensor(avg_val[key], device=device)
                dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
                avg_val[key] = val_tensor.item()
    
    # NO barrier here - will sync in main loop
    return avg_val


def main_worker(rank, world_size, args, gpu_ids):
    """Main training worker"""
    
    # Setup distributed
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
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
        print("TAD-ONLY PROGRESSIVE REFINEMENT TRAINING")
        print("PHILOSOPHY: Only modify TAD, preserve loops")
        print("GOAL: TAD↑, Loop≈, PSNR/SSIM≈")
        print("="*80)
        print(f"\nTAD-only residual construction:")
        print(f"  - Low-frequency extraction (removes loop peaks)")
        print(f"  - Loop masking (excludes high-intensity regions)")
        print(f"  - Diffusion sees ONLY domain-scale structure")
        print(f"\nStage 1 (0-{args.stage1_epochs} epochs): Force diffusion to MOVE")
        print(f"  - Residual activation: ON (interval [50%, 90%])")
        print(f"  - Directional alignment: ON (prevent oscillation)")
        print(f"  - Structure consistency: ON")
        print(f"  - Structure improvement: OFF")
        print(f"  - Low-freq consistency: ON")
        print(f"  - Boundary-aware: ON (reinforce TAD transitions)")
        print(f"  - TV/Smooth: OFF (removed for TAD approach)")
        print(f"  - Goal: Residual std ≥ 50% GT, RSR ≥ 5%, Alignment ≥ 0.3")
        print(f"\nStage 2 ({args.stage1_epochs}+ epochs): Gentle improvement")
        print(f"  - All losses: ON except Smooth")
        print(f"  - λ_improvement gradually increases")
        print(f"  - Boundary-aware continues")
        print(f"  - Goal: TAD metrics improve without hurting loops")
        print(f"  - All losses: ON")
        print(f"  - λ_improvement gradually increases")
        print(f"  - TV/Smooth: ON (now safe)")
        print(f"  - Goal: TAD metrics improve")
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
        model = DDP(model, device_ids=[device.index], output_device=device.index)
    
    if is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")
    
    # Create scheduler
    if is_main_process:
        print("\n[3/6] Creating diffusion scheduler...")
    scheduler_diff = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear')
    
    # Create loss calculator (PROGRESSIVE!)
    if is_main_process:
        print("\n[4/6] Creating PROGRESSIVE loss calculator...")
        print(f"  Initial weights:")
        print(f"    λ_residual:     {args.lambda_residual}")
        print(f"    λ_consistency:  {args.lambda_consistency}")
        print(f"    λ_improvement:  {args.lambda_improvement} (stage 2 only)")
        print(f"    λ_low_freq:     {args.lambda_low_freq}")
        print(f"    λ_smooth:       {args.lambda_smooth}")
    
    loss_calculator = ProgressiveStructureLossCalculator(
        lambda_residual=args.lambda_residual,
        lambda_consistency=args.lambda_consistency,
        lambda_improvement=args.lambda_improvement,
        lambda_low_freq=args.lambda_low_freq,
        lambda_smooth=args.lambda_smooth,
        insulation_window=args.insulation_window,
        stage1_epochs=args.stage1_epochs
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
        print("\n[6/6] Starting progressive training...")
        print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_history = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        if is_main_process:
            print(f"\n[DEBUG] Starting epoch {epoch}")
            sys.stdout.flush()
        
        # Set epoch for distributed sampler
        if world_size > 1:
            if is_main_process:
                print(f"[DEBUG] Setting sampler epoch to {epoch}")
                sys.stdout.flush()
            train_loader.sampler.set_epoch(epoch)
            if is_main_process:
                print(f"[DEBUG] Sampler epoch set")
                sys.stdout.flush()
        
        # Update weights at stage 2
        if epoch == args.stage1_epochs and is_main_process:
            print("\n" + "="*80)
            print(f"🔄 SWITCHING TO STAGE 2 at epoch {epoch}")
            print("="*80)
            loss_calculator.update_weights_for_stage2(epoch)
            print(f"  Updated weights:")
            print(f"    λ_residual:     {loss_calculator.lambda_residual}")
            print(f"    λ_improvement:  {loss_calculator.lambda_improvement}")
            print(f"    λ_low_freq:     {loss_calculator.lambda_low_freq}")
            print("="*80)
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, scheduler_diff,
            loss_calculator, residual_clipper, normalizer,
            device, epoch, rank, world_size
        )
        
        # Validate  
        should_validate = (epoch % args.val_frequency == 0) or (epoch == args.epochs - 1)
        
        if should_validate:
            val_losses = validate(
                model, val_loader, scheduler_diff,
                loss_calculator, residual_clipper, normalizer,
                device, epoch, rank, world_size
            )
        else:
            if epoch > 0 and len(training_history) > 0:
                val_losses = training_history[-1].get('val', {k: 0.0 for k in ['total', 'residual', 'consistency', 'pred_residual_var']})
            else:
                val_losses = {k: 0.0 for k in ['total', 'residual', 'consistency', 'pred_residual_var']}
        
        # NO barrier - DDP handles synchronization automatically
        
        # Log (only on rank 0)
        if is_main_process:
            imp_str = f"{train_losses['improvement']:.4f}" if train_losses['stage'] == 2 and train_losses['improvement'] > 0 else 'OFF'
            
            # Format with validity indicators
            res_str = f"{train_losses['residual']:.4f}" if train_losses['residual'] is not None and train_losses['residual'] > 0 else "INVALID"
            cons_str = f"{train_losses['consistency']:.4f}" if train_losses['consistency'] is not None and train_losses['consistency'] > 0 else "INVALID"
            
            print(f"\nEpoch {epoch} [Stage {train_losses['stage']}]:")
            print(f"  Train - Total: {train_losses['total']:.4f}, "
                  f"Res: {res_str}, "
                  f"Cons: {cons_str}, "
                  f"Imp: {imp_str}")
            print(f"  Residual Std - Pred: {train_losses['pred_residual_std']:.6f}, "
                  f"Target: {train_losses['target_residual_std']:.6f} "
                  f"(Ratio: {train_losses['pred_residual_std']/train_losses['target_residual_std']*100:.1f}%)")
            print(f"  🎯 RSR (Residual-to-Signal): {train_losses['rsr']:.2f}% "
                  f"{'[IDENTITY RISK]' if train_losses['rsr'] < 1.0 else ''}"
                  f"{'[BARELY ALIVE]' if 1.0 <= train_losses['rsr'] < 5.0 else ''}"
                  f"{'[HEALTHY ✓]' if 5.0 <= train_losses['rsr'] <= 15.0 else ''}"
                  f"{'[DESTROY RISK]' if train_losses['rsr'] > 20.0 else ''}")
            print(f"  🧭 Directional Alignment (cos_sim): {train_losses['cos_sim']:.3f} "
                  f"{'[RANDOM]' if train_losses['cos_sim'] < 0.3 else ''}"
                  f"{'[WEAK]' if 0.3 <= train_losses['cos_sim'] < 0.5 else ''}"
                  f"{'[GOOD ✓]' if 0.5 <= train_losses['cos_sim'] < 0.7 else ''}"
                  f"{'[STRONG ✓✓]' if train_losses['cos_sim'] >= 0.7 else ''}")
            
            if should_validate:
                val_res_str = f"{val_losses['residual']:.4f}" if val_losses['residual'] is not None else "INVALID"
                val_cons_str = f"{val_losses['consistency']:.4f}" if val_losses['consistency'] is not None else "INVALID"
                
                print(f"  Val   - Total: {val_losses['total']:.4f}, "
                      f"Res: {val_res_str}, "
                      f"Cons: {val_cons_str}")
                print(f"  Val Residual Var: {val_losses['pred_residual_var']:.6f}")
            
            # CRITICAL: Check Stage 1 progress
            if epoch < args.stage1_epochs:
                # NEW: Get current λ_residual from loss_calculator
                current_lambda_residual = loss_calculator.get_dynamic_lambda_residual(epoch)
                
                # Check multiple criteria
                std_ratio = train_losses['pred_residual_std'] / train_losses['target_residual_std']
                rsr = train_losses['rsr']
                cos_sim = train_losses['cos_sim']
                
                print(f"  📊 λ_residual = {current_lambda_residual:.2f} (dynamic growth)")
                
                if std_ratio >= 0.5 and rsr >= 5.0 and cos_sim >= 0.3:
                    print(f"  ✅ Stage 1 SUCCESS: Diffusion activated!")
                    print(f"     Std ratio = {std_ratio*100:.1f}%, RSR = {rsr:.2f}%, Alignment = {cos_sim:.3f}")
                elif std_ratio >= 0.3 or rsr >= 1.0:
                    print(f"  ⚠️ Stage 1 PARTIAL: Moving but weak")
                    print(f"     Std ratio = {std_ratio*100:.1f}%, RSR = {rsr:.2f}%, Alignment = {cos_sim:.3f}")
                else:
                    print(f"  🔴 Stage 1 FAILING: Identity collapse!")
                    print(f"     Std ratio = {std_ratio*100:.1f}%, RSR = {rsr:.2f}%, Alignment = {cos_sim:.3f}")
                    print(f"     Need: std ratio ≥ 50% AND RSR ≥ 5% AND alignment ≥ 0.3")
                
                # Check if std is decreasing (bad sign!)
                if epoch > 0 and len(training_history) > 0:
                    prev_std = training_history[-1]['train']['pred_residual_std']
                    if train_losses['pred_residual_std'] < prev_std * 0.95:  # 5% decrease
                        print(f"  🔴 WARNING: Std DECREASING! ({prev_std:.6f} → {train_losses['pred_residual_std']:.6f})")
                        print(f"     Model moving toward identity collapse!")
            
            # Check valid loss count
            if 'n_valid_losses' in train_losses:
                print(f"  Valid losses: {train_losses['n_valid_losses']}/6")  # Updated from 5 to 6
            
            # Save history
            training_history.append({
                'epoch': epoch,
                'train': train_losses,
                'val': val_losses if should_validate else {}
            })
            
            with open(output_dir / 'training_history.json', 'w') as f:
                json.dump(training_history, f, indent=2)
            
            # Save best model
            if should_validate and val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'normalizer_mean': normalizer.mean,
                    'normalizer_std': normalizer.std,
                    'config': {
                        'base_channels': 64,
                        'channel_multipliers': [1, 2, 4, 8],
                        'num_res_blocks': 2
                    }
                }
                
                torch.save(checkpoint, output_dir / 'best_progressive.pt')
                print(f"  ✓ Saved best model (val loss: {best_val_loss:.4f})")
            
            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                print(f"[DEBUG] Saving periodic checkpoint for epoch {epoch+1}")
                sys.stdout.flush()
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'normalizer_mean': normalizer.mean,
                    'normalizer_std': normalizer.std
                }
                torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
                print(f"[DEBUG] Periodic checkpoint saved")
                sys.stdout.flush()
        
        if is_main_process:
            print(f"[DEBUG] Epoch {epoch} logging complete, about to loop")
            sys.stdout.flush()
        
        # Clear cache and collect garbage to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        if is_main_process:
            print(f"[DEBUG] Epoch {epoch} COMPLETE - returning to loop start")
            sys.stdout.flush()
        
        # Explicit yield to allow other processes
        import time
        time.sleep(0.1)
    
    # Save final model
    if is_main_process:
        final_checkpoint = {
            'epoch': args.epochs - 1,
            'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'normalizer_mean': normalizer.mean,
            'normalizer_std': normalizer.std
        }
        torch.save(final_checkpoint, output_dir / 'final_progressive.pt')
        print("\n✓ Training complete!")
        print(f"  Best val loss: {best_val_loss:.4f}")
    
    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Progressive Structure Refinement Training")
    
    # Data
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints_progressive')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--val_frequency', type=int, default=5)
    
    # Progressive strategy
    parser.add_argument('--stage1_epochs', type=int, default=20,
                       help='Epochs for stage 1 (improvement disabled)')
    
    # Loss weights (initial)
    parser.add_argument('--lambda_residual', type=float, default=1.0,
                       help='Residual activation weight')
    parser.add_argument('--lambda_consistency', type=float, default=1.0,
                       help='Structure consistency weight')
    parser.add_argument('--lambda_improvement', type=float, default=0.1,
                       help='Structure improvement weight (stage 2)')
    parser.add_argument('--lambda_low_freq', type=float, default=0.5,
                       help='Low-frequency consistency weight')
    parser.add_argument('--lambda_smooth', type=float, default=0.05,
                       help='Smoothness TV weight')
    
    # Other parameters
    parser.add_argument('--insulation_window', type=int, default=5)
    parser.add_argument('--clip_factor', type=float, default=0.1)
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    
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
