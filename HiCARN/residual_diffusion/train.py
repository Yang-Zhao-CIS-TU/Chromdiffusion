"""
Training Script for Residual Diffusion

Main training loop for Hi-C refinement using residual diffusion.

Training phases:
1. Baseline: Diffusion loss only (no PH)
2. Stabilized: Diffusion + reconstruction loss
3. Advanced: Diffusion + reconstruction + PH loss (optional)

Critical: HiCARN is frozen - we only train the diffusion model
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from model import ResidualDiffusionUNet
from scheduler import DDPMScheduler, DDIMScheduler
from data_loader import load_hic_data, create_dataloaders
from losses import CombinedLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train Residual Diffusion for Hi-C Refinement')
    
    # Data
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to HiCARN predictions (predictions_norm.npy)')
    parser.add_argument('--gt_path', type=str, required=True,
                       help='Path to ground truth HR (ground_truth.npy)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_diffusion',
                       help='Directory to save checkpoints and logs')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels in U-Net')
    parser.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='Channel multipliers for each level')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                       help='Number of residual blocks per level')
    
    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--beta_schedule', type=str, default='linear',
                       choices=['linear', 'scaled_linear', 'cosine'],
                       help='Beta noise schedule')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.9,
                       help='Fraction of data for training')
    
    # Loss weights
    parser.add_argument('--lambda_recon', type=float, default=0.1,
                       help='Weight for reconstruction loss')
    parser.add_argument('--lambda_ph', type=float, default=0.01,
                       help='Weight for PH loss')
    parser.add_argument('--use_ph', action='store_true',
                       help='Use persistent homology loss (expensive)')
    
    # System
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Checkpointing
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    normalizer,
    device,
    epoch,
    writer=None
):
    """
    Train for one epoch
    
    Args:
        model: ResidualDiffusionUNet
        dataloader: training dataloader
        optimizer: optimizer
        scheduler: diffusion scheduler (DDPM)
        criterion: combined loss
        normalizer: residual normalizer
        device: device
        epoch: current epoch number
        writer: tensorboard writer
    
    Returns:
        avg_loss: average loss for the epoch
        loss_dict: dictionary of individual losses
    """
    model.train()
    
    total_losses = {
        'total': 0.0,
        'diffusion': 0.0,
        'reconstruction': 0.0,
        'ph': 0.0
    }
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        # Get data
        condition = batch['condition'].to(device)  # HiCARN prediction
        residual_norm = batch['residual_norm'].to(device)  # Normalized residual
        
        batch_size = condition.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, scheduler.num_train_timesteps,
            (batch_size,),
            device=device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(residual_norm)
        
        # Forward diffusion: add noise to residual
        # Δ_t = √α̅_t * Δ + √(1 - α̅_t) * ε
        noisy_residual = scheduler.add_noise(residual_norm, noise, timesteps)
        
        # Predict noise
        noise_pred = model(noisy_residual, timesteps, condition)
        
        # Compute reconstruction for optional losses
        if criterion.lambda_recon > 0 or criterion.use_ph:
            # Predict clean residual (simple prediction for training)
            with torch.no_grad():
                # Use predicted noise to estimate clean residual
                alpha_prod = scheduler.alphas_cumprod[timesteps].to(device)
                alpha_prod = alpha_prod.reshape(-1, 1, 1, 1)
                sqrt_alpha_prod = torch.sqrt(alpha_prod)
                sqrt_one_minus_alpha_prod = torch.sqrt(1 - alpha_prod)
                
                pred_residual_norm = (
                    noisy_residual - sqrt_one_minus_alpha_prod * noise_pred
                ) / sqrt_alpha_prod
                
                # Denormalize
                pred_residual = normalizer.inverse_transform(
                    pred_residual_norm.cpu().numpy()
                )
                pred_residual = torch.from_numpy(pred_residual).to(device)
                
                # Reconstruct: x̂ = x_pred + Δ̂
                condition_np = condition.cpu().numpy()
                gt_np = (condition_np + batch['residual_raw'].cpu().numpy())
                
                x_reconstructed = condition + pred_residual
                x_target = torch.from_numpy(gt_np).to(device)
        else:
            x_reconstructed = None
            x_target = None
        
        # Compute loss
        loss, loss_dict = criterion(
            noise_pred, noise,
            x_reconstructed, x_target
        )
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses.keys():
            if key in loss_dict:
                total_losses[key] += loss_dict[key]
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'diff': f"{loss_dict.get('diffusion', 0):.4f}"
        })
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    # Log to tensorboard
    if writer is not None:
        for key, value in avg_losses.items():
            writer.add_scalar(f'train/{key}_loss', value, epoch)
    
    return avg_losses['total'], avg_losses


@torch.no_grad()
def validate(
    model,
    dataloader,
    scheduler,
    criterion,
    normalizer,
    device,
    epoch,
    writer=None
):
    """
    Validate the model
    
    Args:
        model: ResidualDiffusionUNet
        dataloader: validation dataloader
        scheduler: diffusion scheduler
        criterion: combined loss
        normalizer: residual normalizer
        device: device
        epoch: current epoch number
        writer: tensorboard writer
    
    Returns:
        avg_loss: average validation loss
        loss_dict: dictionary of individual losses
    """
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'diffusion': 0.0,
        'reconstruction': 0.0,
        'ph': 0.0
    }
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Validation')
    
    for batch in pbar:
        condition = batch['condition'].to(device)
        residual_norm = batch['residual_norm'].to(device)
        
        batch_size = condition.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, scheduler.num_train_timesteps,
            (batch_size,),
            device=device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(residual_norm)
        
        # Forward diffusion
        noisy_residual = scheduler.add_noise(residual_norm, noise, timesteps)
        
        # Predict noise
        noise_pred = model(noisy_residual, timesteps, condition)
        
        # Compute reconstruction
        if criterion.lambda_recon > 0 or criterion.use_ph:
            alpha_prod = scheduler.alphas_cumprod[timesteps].to(device)
            alpha_prod = alpha_prod.reshape(-1, 1, 1, 1)
            sqrt_alpha_prod = torch.sqrt(alpha_prod)
            sqrt_one_minus_alpha_prod = torch.sqrt(1 - alpha_prod)
            
            pred_residual_norm = (
                noisy_residual - sqrt_one_minus_alpha_prod * noise_pred
            ) / sqrt_alpha_prod
            
            pred_residual = normalizer.inverse_transform(
                pred_residual_norm.cpu().numpy()
            )
            pred_residual = torch.from_numpy(pred_residual).to(device)
            
            condition_np = condition.cpu().numpy()
            gt_np = (condition_np + batch['residual_raw'].cpu().numpy())
            
            x_reconstructed = condition + pred_residual
            x_target = torch.from_numpy(gt_np).to(device)
        else:
            x_reconstructed = None
            x_target = None
        
        # Compute loss
        loss, loss_dict = criterion(
            noise_pred, noise,
            x_reconstructed, x_target
        )
        
        # Accumulate
        for key in total_losses.keys():
            if key in loss_dict:
                total_losses[key] += loss_dict[key]
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
    
    # Average
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    # Log
    if writer is not None:
        for key, value in avg_losses.items():
            writer.add_scalar(f'val/{key}_loss', value, epoch)
    
    return avg_losses['total'], avg_losses


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    normalizer,
    config,
    filepath
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'normalizer_mean': normalizer.mean,
        'normalizer_std': normalizer.std,
        'config': config
    }
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint: {filepath}")


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(args.output_dir, f'runs/{timestamp}'))
    
    # ====================================================================
    # LOAD DATA
    # ====================================================================
    
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    predictions, ground_truth = load_hic_data(args.pred_path, args.gt_path)
    
    train_loader, val_loader, normalizer = create_dataloaders(
        predictions,
        ground_truth,
        batch_size=args.batch_size,
        train_split=args.train_split,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Save normalizer
    normalizer_path = os.path.join(args.output_dir, 'residual_normalizer.json')
    normalizer.save(normalizer_path)
    
    # ====================================================================
    # CREATE MODEL
    # ====================================================================
    
    print("\n" + "="*80)
    print("STEP 2: CREATING MODEL")
    print("="*80)
    
    model = ResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        num_res_blocks=args.num_res_blocks
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # ====================================================================
    # CREATE SCHEDULER AND LOSS
    # ====================================================================
    
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule
    )
    
    criterion = CombinedLoss(
        lambda_recon=args.lambda_recon,
        lambda_ph=args.lambda_ph,
        use_ph=args.use_ph
    )
    
    # ====================================================================
    # OPTIMIZER
    # ====================================================================
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # ====================================================================
    # RESUME FROM CHECKPOINT
    # ====================================================================
    
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # ====================================================================
    # TRAINING LOOP
    # ====================================================================
    
    print("\n" + "="*80)
    print("STEP 3: TRAINING")
    print("="*80)
    print(f"Total epochs: {args.epochs}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss, train_loss_dict = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            normalizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_loss_dict = validate(
            model, val_loader, scheduler, criterion,
            normalizer, device, epoch, writer
        )
        
        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        
        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"    - Diffusion: {train_loss_dict['diffusion']:.6f}")
        if train_loss_dict['reconstruction'] > 0:
            print(f"    - Reconstruction: {train_loss_dict['reconstruction']:.6f}")
        if train_loss_dict['ph'] > 0:
            print(f"    - PH: {train_loss_dict['ph']:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                normalizer, vars(args), best_path
            )
            print(f"  ✓ New best model! Val loss: {val_loss:.6f}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                normalizer, vars(args), ckpt_path
            )
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    save_checkpoint(
        model, optimizer, args.epochs, val_loss,
        normalizer, vars(args), final_path
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {args.output_dir}")
    print("="*80)
    
    writer.close()


if __name__ == "__main__":
    main()
