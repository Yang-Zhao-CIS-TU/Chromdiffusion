"""
Improved Training Script for Residual Diffusion

Key improvements:
1. Fixed gradient flow (no torch.no_grad() for reconstruction)
2. EMA (Exponential Moving Average) for stable inference
3. v-parameterization support
4. Proper timestep sampling
5. Multi-objective training (diffusion + reconstruction + peak losses)
6. Better logging and monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import json
from copy import deepcopy

from model import ImprovedResidualDiffusionUNet
from scheduler import ImprovedDDPMScheduler
from data_loader import create_dataloaders, validate_data
from losses import CombinedResidualLoss, get_loss_weights


class EMA:
    """
    Exponential Moving Average for model weights
    
    Helps stabilize inference and often improves final performance
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA weights for inference"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights after inference"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train_one_epoch(
    model,
    ema,
    dataloader,
    scheduler,
    criterion,
    optimizer,
    device,
    epoch,
    writer,
    use_amp=True,
    timestep_weights=None
):
    """
    Train for one epoch
    
    Args:
        model: Diffusion model
        ema: EMA object
        dataloader: Training dataloader
        scheduler: Noise scheduler
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        writer: Tensorboard writer
        use_amp: Use automatic mixed precision
        timestep_weights: Timestep sampling weights
    """
    model.train()
    total_loss = 0
    total_diffusion_loss = 0
    total_recon_loss = 0
    
    scaler = GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Get data
        condition = batch['hicarn'].to(device)  # HiCARN prediction
        gt = batch['gt'].to(device)  # Ground truth
        residual_clean = batch['residual'].to(device)  # GT - HiCARN
        
        batch_size = condition.shape[0]
        
        # Sample timesteps (with optional weighting)
        if timestep_weights is not None:
            timesteps = torch.multinomial(
                timestep_weights,
                batch_size,
                replacement=True
            ).to(device)
        else:
            timesteps = torch.randint(
                0,
                scheduler.num_train_timesteps,
                (batch_size,),
                device=device
            ).long()
        
        # Sample noise
        noise = torch.randn_like(residual_clean)
        
        # Add noise to residual (forward diffusion)
        residual_noisy = scheduler.add_noise(residual_clean, noise, timesteps)
        
        with autocast(enabled=use_amp):
            # Predict noise or velocity
            model_output = model(residual_noisy, timesteps, condition)
            
            # ===== KEY FIX: Compute reconstruction WITHOUT torch.no_grad() =====
            # This allows gradients to flow through reconstruction loss
            
            # Get target based on parameterization
            if scheduler.parameterization == 'eps':
                target = noise
                # Predict clean residual for reconstruction
                pred_residual_clean = scheduler.predict_start_from_noise(
                    residual_noisy, timesteps, model_output
                )
            elif scheduler.parameterization == 'v':
                target = scheduler.get_v(residual_clean, noise, timesteps)
                # Predict clean residual for reconstruction
                pred_residual_clean = scheduler.predict_start_from_v(
                    residual_noisy, timesteps, model_output
                )
            else:
                raise ValueError(f"Unknown parameterization: {scheduler.parameterization}")
            
            # Diffusion loss (MSE on noise/velocity prediction)
            diffusion_loss = F.mse_loss(model_output, target)
            
            # ===== KEY FIX: Reconstruction loss with proper gradients =====
            # Stay in normalized space, no numpy conversion
            pred_gt = condition + pred_residual_clean  # Predicted GT
            
            # Multi-objective reconstruction loss
            # Uses peak localization losses (Laplacian, gradient, top-k)
            recon_loss, loss_components = criterion(
                pred_gt,
                gt,
                return_components=True
            )
            
            # Combined loss
            # FIXED: Reduced recon weight from 0.5 to 0.1 to fix plateau
            # This gives more importance to diffusion loss for better peak localization
            total_batch_loss = diffusion_loss + 0.3 * recon_loss
        
        # Backpropagation
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Logging
        total_loss += total_batch_loss.item()
        total_diffusion_loss += diffusion_loss.item()
        total_recon_loss += recon_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_batch_loss.item():.4f}',
            'diff': f'{diffusion_loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}'
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar('train/total_loss', total_batch_loss.item(), global_step)
            writer.add_scalar('train/diffusion_loss', diffusion_loss.item(), global_step)
            writer.add_scalar('train/recon_loss', recon_loss.item(), global_step)
            
            # Log individual loss components
            for name, value in loss_components.items():
                writer.add_scalar(f'train/loss_{name}', value, global_step)
    
    # Epoch summary
    avg_loss = total_loss / len(dataloader)
    avg_diff_loss = total_diffusion_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    
    return avg_loss, avg_diff_loss, avg_recon_loss


@torch.no_grad()
def validate(
    model,
    dataloader,
    scheduler,
    criterion,
    device,
    epoch,
    writer,
    num_samples_to_save=4
):
    """
    Validation loop
    
    Args:
        model: Diffusion model
        dataloader: Validation dataloader
        scheduler: Noise scheduler
        criterion: Loss function
        device: Device
        epoch: Current epoch
        writer: Tensorboard writer
        num_samples_to_save: Number of samples to save for visualization
    """
    model.eval()
    total_loss = 0
    total_diffusion_loss = 0
    total_recon_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
        condition = batch['hicarn'].to(device)
        gt = batch['gt'].to(device)
        residual_clean = batch['residual'].to(device)
        
        batch_size = condition.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (batch_size,),
            device=device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(residual_clean)
        
        # Add noise
        residual_noisy = scheduler.add_noise(residual_clean, noise, timesteps)
        
        # Predict
        model_output = model(residual_noisy, timesteps, condition)
        
        # Get target
        if scheduler.parameterization == 'eps':
            target = noise
            pred_residual_clean = scheduler.predict_start_from_noise(
                residual_noisy, timesteps, model_output
            )
        elif scheduler.parameterization == 'v':
            target = scheduler.get_v(residual_clean, noise, timesteps)
            pred_residual_clean = scheduler.predict_start_from_v(
                residual_noisy, timesteps, model_output
            )
        
        # Losses
        diffusion_loss = F.mse_loss(model_output, target)
        
        pred_gt = condition + pred_residual_clean
        recon_loss, _ = criterion(pred_gt, gt, return_components=True)
        
        total_batch_loss = diffusion_loss + 0.5 * recon_loss
        
        total_loss += total_batch_loss.item()
        total_diffusion_loss += diffusion_loss.item()
        total_recon_loss += recon_loss.item()
        
        # Save sample predictions for visualization
        if batch_idx == 0 and writer is not None:
            # Save first few samples
            num_save = min(num_samples_to_save, batch_size)
            for i in range(num_save):
                # Save as images (use first channel if multi-channel)
                writer.add_image(
                    f'val/sample_{i}/condition',
                    condition[i, 0:1].cpu(),
                    epoch
                )
                writer.add_image(
                    f'val/sample_{i}/gt',
                    gt[i, 0:1].cpu(),
                    epoch
                )
                writer.add_image(
                    f'val/sample_{i}/prediction',
                    pred_gt[i, 0:1].cpu(),
                    epoch
                )
                writer.add_image(
                    f'val/sample_{i}/residual',
                    pred_residual_clean[i, 0:1].cpu(),
                    epoch
                )
    
    avg_loss = total_loss / len(dataloader)
    avg_diff_loss = total_diffusion_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    
    if writer is not None:
        writer.add_scalar('val/total_loss', avg_loss, epoch)
        writer.add_scalar('val/diffusion_loss', avg_diff_loss, epoch)
        writer.add_scalar('val/recon_loss', avg_recon_loss, epoch)
    
    return avg_loss, avg_diff_loss, avg_recon_loss


def save_checkpoint(
    model,
    ema,
    optimizer,
    scheduler,
    epoch,
    best_loss,
    save_dir,
    is_best=False
):
    """Save checkpoint"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'scheduler_config': {
            'num_train_timesteps': scheduler.num_train_timesteps,
            'parameterization': scheduler.parameterization
        }
    }
    
    # Save EMA weights
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow
    
    # Save regular checkpoint
    torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, save_dir / 'checkpoint_best.pt')
    
    print(f"Checkpoint saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train residual diffusion model')
    
    # Data
    parser.add_argument('--train_hicarn', type=str, required=True)
    parser.add_argument('--train_gt', type=str, required=True)
    parser.add_argument('--val_hicarn', type=str, default=None)
    parser.add_argument('--val_gt', type=str, default=None)
    
    # Model
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--channel_mults', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attn_levels', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--parameterization', type=str, default='v', choices=['eps', 'v'])
    
    # Scheduler
    parser.add_argument('--num_train_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--timestep_sampling', type=str, default='uniform', 
                       choices=['uniform', 'snr', 'late'])
    
    # Loss
    parser.add_argument('--loss_strategy', type=str, default='peak_focused',
                       choices=['balanced', 'peak_focused', 'tad_focused'])
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--use_amp', action='store_true', default=True)
    
    # Paths
    parser.add_argument('--output_dir', type=str, default='checkpoints_v2')
    parser.add_argument('--log_dir', type=str, default='runs_v2')
    
    # Device
    parser.add_argument('--gpu', type=int, default=0)
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate data
    validate_data(args.train_hicarn, args.train_gt)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.train_hicarn,
        args.train_gt,
        args.val_hicarn,
        args.val_gt,
        batch_size=args.batch_size
    )
    
    # Create model
    model = ImprovedResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        cond_channels=1,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        attn_levels=tuple(args.attn_levels),
        parameterization=args.parameterization
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    
    # Create scheduler
    scheduler = ImprovedDDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        parameterization=args.parameterization
    )
    
    # Get timestep weights
    timestep_weights = scheduler.get_timestep_weights(
        strategy=args.timestep_sampling,
        device=device
    )
    
    # Create loss
    loss_weights = get_loss_weights(args.loss_strategy)
    criterion = CombinedResidualLoss(
        use_laplacian=True,
        use_gradient=True,
        use_topk=True,
        use_ranking=False,  # Expensive, use for fine-tuning
        use_multiscale=True,
        **loss_weights
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Scheduler (learning rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training setup
    start_epoch = 1
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  ✓ Loaded model weights")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("  ✓ Loaded optimizer state")
        
        # Load EMA if it exists
        if ema and 'ema_shadow' in checkpoint and checkpoint['ema_shadow']:
            ema.shadow = checkpoint['ema_shadow']
            ema.collected_params = checkpoint.get('ema_collected_params', None)
            print("  ✓ Loaded EMA weights")
        
        # Resume from next epoch
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"  → Resuming from epoch {start_epoch}")
        print(f"  → Best val loss so far: {best_val_loss:.4f}\n")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_diff, train_recon = train_one_epoch(
            model, ema, train_loader, scheduler, criterion,
            optimizer, device, epoch, writer, args.use_amp,
            timestep_weights
        )
        
        print(f"Train - Loss: {train_loss:.4f}, Diff: {train_diff:.4f}, Recon: {train_recon:.4f}")
        
        # Validate
        if val_loader is not None:
            # Use EMA weights for validation
            if ema is not None:
                ema.apply_shadow()
            
            val_loss, val_diff, val_recon = validate(
                model, val_loader, scheduler, criterion,
                device, epoch, writer
            )
            
            if ema is not None:
                ema.restore()
            
            print(f"Val - Loss: {val_loss:.4f}, Diff: {val_diff:.4f}, Recon: {val_recon:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, ema, optimizer, scheduler,
                    epoch, best_val_loss, args.output_dir, is_best=True
                )
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                model, ema, optimizer, scheduler,
                epoch, best_val_loss, args.output_dir
            )
        
        # Update learning rate
        lr_scheduler.step()
        
        if writer is not None:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    
    print("\nTraining complete!")
    writer.close()


if __name__ == '__main__':
    main()
