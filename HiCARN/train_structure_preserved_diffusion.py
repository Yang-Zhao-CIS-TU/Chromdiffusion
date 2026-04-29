"""
Structure-Preserved Residual Diffusion Training Script

Key Changes from Original:
  1. Structure-oriented loss instead of pixel-level MSE
  2. Residual clipping to prevent loop destruction
  3. Never expose LR to diffusion (only residuals)
  4. Three-phase training with structure preservation

Training Data Flow:
  LR → HiCARN → HiCARN_pred
  HiCARN_pred + HR → residual (target)
  residual → diffusion → residual_pred
  
  ⚠️ Diffusion NEVER sees LR!

Loss Function:
  L_total = 0.1·L_residual + 1.0·L_insulation + 
            0.5·L_TAD_boundary + 0.5·L_low_freq

Usage:
    python train_structure_preserved_diffusion.py \
        --pred_path hicarn_predictions/predictions_norm.npy \
        --gt_path hicarn_predictions/ground_truth.npy \
        --output_dir checkpoints_structure_diffusion \
        --epochs 100 \
        --batch_size 16 \
        --gpu 2
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
from pathlib import Path

# Import structure-oriented losses
from structure_losses import (
    StructureLossCalculator,
    ResidualClipper
)

# Import diffusion model and scheduler
import sys
sys.path.insert(0, 'residual_diffusion')
from model import ResidualDiffusionUNet
from scheduler import DDPMScheduler
from data_loader import ResidualNormalizer


class ResidualDataset(Dataset):
    """
    Dataset for Structure-Preserved Residual Diffusion
    
    Returns:
        - hicarn_pred: HiCARN predictions (normalized)
        - residual: HR - HiCARN_pred (normalized)
        - hicarn_pred_raw: For computing clip values
    """
    def __init__(self, pred_path, gt_path):
        self.pred_norm = np.load(pred_path)
        self.gt_norm = np.load(gt_path)
        
        # Compute residuals
        self.residual = self.gt_norm - self.pred_norm
        
        print(f"Loaded data:")
        print(f"  HiCARN predictions: {self.pred_norm.shape}")
        print(f"  Ground truth: {self.gt_norm.shape}")
        print(f"  Residuals: {self.residual.shape}")
        print(f"  Residual range: [{self.residual.min():.4f}, {self.residual.max():.4f}]")
    
    def __len__(self):
        return len(self.pred_norm)
    
    def __getitem__(self, idx):
        hicarn_pred = self.pred_norm[idx]
        gt = self.gt_norm[idx]
        residual = self.residual[idx]
        
        # Ensure 3D: (C, H, W)
        if hicarn_pred.ndim == 2:
            hicarn_pred = hicarn_pred[None, :, :]
            gt = gt[None, :, :]
            residual = residual[None, :, :]
        
        return {
            'hicarn_pred': torch.from_numpy(hicarn_pred).float(),
            'gt': torch.from_numpy(gt).float(),
            'residual': torch.from_numpy(residual).float()
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
    phase='baseline'
):
    """
    Train for one epoch with structure-oriented losses
    
    Args:
        model: Diffusion U-Net
        dataloader: Training data
        optimizer: AdamW optimizer
        scheduler_diff: DDPM scheduler
        loss_calculator: Structure loss calculator
        residual_clipper: Residual magnitude clipper
        normalizer: Residual normalizer
        device: cuda/cpu
        epoch: Current epoch
        phase: Training phase (baseline/stabilized/advanced)
    
    Returns:
        avg_losses: Dictionary of average losses
    """
    model.train()
    
    epoch_losses = {
        'total': [],
        'residual': [],
        'insulation': [],
        'tad_boundary': [],
        'low_freq': []
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} ({phase})")
    
    for batch in pbar:
        hicarn_pred = batch['hicarn_pred'].to(device)  # (B, 1, H, W)
        gt = batch['gt'].to(device)
        target_residual = batch['residual'].to(device)
        
        batch_size = hicarn_pred.shape[0]
        
        # Normalize residual
        target_residual_norm = normalizer.transform(target_residual.cpu().numpy())
        target_residual_norm = torch.from_numpy(target_residual_norm).to(device)
        
        # Sample timestep
        t = torch.randint(0, scheduler_diff.num_train_timesteps, 
                         (batch_size,), device=device).long()
        
        # Add noise to target residual
        noise = torch.randn_like(target_residual_norm)
        noisy_residual = scheduler_diff.add_noise(target_residual_norm, noise, t)
        
        # Predict noise (condition on HiCARN prediction)
        noise_pred = model(noisy_residual, t, hicarn_pred)
        
        # Denoise to get predicted residual
        with torch.no_grad():
            # Simple one-step denoising for training (full denoising too expensive)
            pred_residual_norm = scheduler_diff.step(noise_pred, t, noisy_residual)[0]
            
            # Denormalize
            pred_residual = normalizer.inverse_transform(pred_residual_norm.cpu().numpy())
            pred_residual = torch.from_numpy(pred_residual).to(device)
            
            # CRITICAL: Clip residual to prevent large changes
            pred_residual = residual_clipper.clip_residual(pred_residual, hicarn_pred)
        
        # Construct predicted Hi-C
        pred_hic = hicarn_pred + pred_residual
        target_hic = gt
        
        # Compute structure-oriented loss
        total_loss, loss_dict = loss_calculator(
            pred_residual, target_residual,
            pred_hic, target_hic
        )
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Log losses
        for key in epoch_losses.keys():
            epoch_losses[key].append(loss_dict[key])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'ins': f"{loss_dict['insulation']:.4f}",
            'tad': f"{loss_dict['tad_boundary']:.4f}"
        })
    
    # Compute averages
    avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
    
    return avg_losses


def validate(
    model,
    dataloader,
    scheduler_diff,
    loss_calculator,
    residual_clipper,
    normalizer,
    device
):
    """Validation with structure-oriented metrics"""
    model.eval()
    
    val_losses = {
        'total': [],
        'residual': [],
        'insulation': [],
        'tad_boundary': [],
        'low_freq': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            hicarn_pred = batch['hicarn_pred'].to(device)
            gt = batch['gt'].to(device)
            target_residual = batch['residual'].to(device)
            
            batch_size = hicarn_pred.shape[0]
            
            # Full denoising (expensive but accurate)
            residual_t = torch.randn_like(target_residual)
            
            for t in reversed(range(scheduler_diff.num_train_timesteps)):
                t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
                noise_pred = model(residual_t, t_tensor, hicarn_pred)
                residual_t, _ = scheduler_diff.step(noise_pred, t, residual_t)
            
            pred_residual_norm = residual_t
            
            # Denormalize
            pred_residual = normalizer.inverse_transform(pred_residual_norm.cpu().numpy())
            pred_residual = torch.from_numpy(pred_residual).to(device)
            
            # Clip residual
            pred_residual = residual_clipper.clip_residual(pred_residual, hicarn_pred)
            
            # Construct Hi-C
            pred_hic = hicarn_pred + pred_residual
            target_hic = gt
            
            # Compute loss
            total_loss, loss_dict = loss_calculator(
                pred_residual, target_residual,
                pred_hic, target_hic
            )
            
            for key in val_losses.keys():
                val_losses[key].append(loss_dict[key])
    
    avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
    
    return avg_losses


def parse_args():
    parser = argparse.ArgumentParser(description='Structure-Preserved Residual Diffusion Training')
    
    # Data
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to HiCARN predictions (normalized)')
    parser.add_argument('--gt_path', type=str, required=True,
                       help='Path to ground truth (normalized)')
    parser.add_argument('--output_dir', type=str, default='checkpoints_structure_diffusion',
                       help='Output directory')
    
    # Model architecture
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    
    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear',
                       choices=['linear', 'scaled_linear', 'cosine'])
    
    # Loss weights (structure-oriented)
    parser.add_argument('--lambda_residual', type=float, default=0.1,
                       help='Weight for residual loss (WEAK)')
    parser.add_argument('--lambda_insulation', type=float, default=1.0,
                       help='Weight for insulation loss (STRONG)')
    parser.add_argument('--lambda_tad_boundary', type=float, default=0.5,
                       help='Weight for TAD boundary loss')
    parser.add_argument('--lambda_low_freq', type=float, default=0.5,
                       help='Weight for low-frequency consistency')
    
    # Residual clipping
    parser.add_argument('--clip_factor', type=float, default=0.1,
                       help='Residual clip factor (α = clip_factor × std)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # System
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("STRUCTURE-PRESERVED RESIDUAL DIFFUSION TRAINING")
    print("="*80)
    print(f"\nKey Features:")
    print(f"  ✓ Structure-oriented loss (preserves TADs)")
    print(f"  ✓ Residual clipping (prevents loop destruction)")
    print(f"  ✓ Never exposes LR to diffusion")
    print(f"\nLoss Weights:")
    print(f"  λ_residual:     {args.lambda_residual:.2f} (weak)")
    print(f"  λ_insulation:   {args.lambda_insulation:.2f} (strong)")
    print(f"  λ_tad_boundary: {args.lambda_tad_boundary:.2f}")
    print(f"  λ_low_freq:     {args.lambda_low_freq:.2f}")
    print(f"\nResidual Clipping:")
    print(f"  clip_factor: {args.clip_factor:.2f}")
    print(f"  α = {args.clip_factor:.2f} × std(HiCARN_pred)")
    print("="*80)
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    dataset = ResidualDataset(args.pred_path, args.gt_path)
    
    # Split train/val
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {n_val}")
    
    # Create model
    print("\n[2/6] Creating model...")
    model = ResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        num_res_blocks=args.num_res_blocks
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # Create diffusion scheduler
    print("\n[3/6] Creating diffusion scheduler...")
    scheduler_diff = DDPMScheduler(
        num_train_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule
    )
    print(f"  Timesteps: {args.num_timesteps}")
    print(f"  Beta schedule: {args.beta_schedule}")
    
    # Create loss calculator
    print("\n[4/6] Creating structure-oriented loss calculator...")
    loss_calculator = StructureLossCalculator(
        lambda_residual=args.lambda_residual,
        lambda_insulation=args.lambda_insulation,
        lambda_tad_boundary=args.lambda_tad_boundary,
        lambda_low_freq=args.lambda_low_freq
    ).to(device)
    
    # Create residual clipper
    residual_clipper = ResidualClipper(clip_factor=args.clip_factor)
    
    # Create residual normalizer
    print("\n[5/6] Fitting residual normalizer...")
    normalizer = ResidualNormalizer()
    all_residuals = []
    for batch in train_loader:
        all_residuals.append(batch['residual'].numpy())
    all_residuals = np.concatenate(all_residuals, axis=0)
    normalizer.fit(all_residuals)
    print(f"  Residual mean: {normalizer.mean:.6f}")
    print(f"  Residual std:  {normalizer.std:.6f}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\n[6/6] Starting training...")
    print("="*80)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(args.epochs):
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, scheduler_diff,
            loss_calculator, residual_clipper, normalizer,
            device, epoch, phase='structure_preserved'
        )
        
        # Validate
        val_losses = validate(
            model, val_loader, scheduler_diff,
            loss_calculator, residual_clipper, normalizer,
            device
        )
        
        # Update LR
        lr_scheduler.step()
        
        # Log
        print(f"\nEpoch {epoch}:")
        print(f"  Train - Total: {train_losses['total']:.4f}, "
              f"Ins: {train_losses['insulation']:.4f}, "
              f"TAD: {train_losses['tad_boundary']:.4f}, "
              f"LF: {train_losses['low_freq']:.4f}")
        print(f"  Val   - Total: {val_losses['total']:.4f}, "
              f"Ins: {val_losses['insulation']:.4f}, "
              f"TAD: {val_losses['tad_boundary']:.4f}, "
              f"LF: {val_losses['low_freq']:.4f}")
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train': train_losses,
            'val': val_losses,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': vars(args),
                'normalizer_mean': normalizer.mean,
                'normalizer_std': normalizer.std
            }
            
            torch.save(checkpoint, output_dir / 'best_model_structure_preserved.pt')
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(args),
                'normalizer_mean': normalizer.mean,
                'normalizer_std': normalizer.std
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model
    checkpoint = {
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
        'normalizer_mean': normalizer.mean,
        'normalizer_std': normalizer.std
    }
    torch.save(checkpoint, output_dir / 'final_model_structure_preserved.pt')
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}/")
    print(f"\nFiles:")
    print(f"  - best_model_structure_preserved.pt")
    print(f"  - final_model_structure_preserved.pt")
    print(f"  - training_history.json")
    print("="*80)


if __name__ == "__main__":
    main()
