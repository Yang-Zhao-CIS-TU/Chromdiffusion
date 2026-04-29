"""
Enhanced Training Script for Residual Diffusion V2

NEW FEATURES (Based on Expert Recommendations):
1. Frequency-separated reconstruction (recon constrains low-freq/structure, diff handles high-freq/peaks)
2. Localization losses (Heatmap KL for peak positioning + Gradient consistency for sharp peaks)
3. Solves hit@1/IoU problem (peaks in correct positions with right shapes)
4. Resume functionality + Fixed recon_weight (0.1)

Key improvements over V1:
- 方案B: 分频训练 (blur/low-pass recon + diff for peaks)
- 定位损失: Soft-argmax KL + 梯度一致性
- Better TAD/structure without hurting loops
- Sharp, correctly-positioned peaks
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

from model import ImprovedResidualDiffusionUNet
from scheduler import ImprovedDDPMScheduler
from data_loader import create_dataloaders, validate_data

# Import enhanced losses
# Place losses_enhanced.py in the same directory as this script
try:
    from losses_enhanced import EnhancedCombinedLoss, get_enhanced_loss_config
    HAS_ENHANCED_LOSSES = True
except ImportError:
    HAS_ENHANCED_LOSSES = False
    print("⚠️  Warning: losses_enhanced.py not found, using fallback")
    from losses import CombinedResidualLoss, get_loss_weights


class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
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
    """Train for one epoch with enhanced losses"""
    model.train()
    total_loss = 0
    total_diffusion_loss = 0
    total_recon_loss = 0
    
    scaler = GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        condition = batch['hicarn'].to(device)
        gt = batch['gt'].to(device)
        residual_clean = batch['residual'].to(device)
        
        batch_size = condition.shape[0]
        
        # Sample timesteps
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
        
        noise = torch.randn_like(residual_clean)
        residual_noisy = scheduler.add_noise(residual_clean, noise, timesteps)
        
        with autocast(enabled=use_amp):
            model_output = model(residual_noisy, timesteps, condition)
            
            # Get target and predicted clean residual
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
            
            # Diffusion loss
            diffusion_loss = F.mse_loss(model_output, target)
            
            # Enhanced reconstruction loss
            # NEW: With frequency separation + localization
            pred_gt = condition + pred_residual_clean
            
            recon_loss, loss_components = criterion(
                pred_gt,
                gt,
                return_components=True
            )
            
            # Combined loss
            # FIXED: recon_weight = 0.1 (down from 0.5)
            # Gives diffusion more importance for peak learning
            total_batch_loss = diffusion_loss + 0.1 * recon_loss
        
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
        
        if ema is not None:
            ema.update()
        
        total_loss += total_batch_loss.item()
        total_diffusion_loss += diffusion_loss.item()
        total_recon_loss += recon_loss.item()
        
        pbar.set_postfix({
            'loss': f'{total_batch_loss.item():.4f}',
            'diff': f'{diffusion_loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}'
        })
        
        global_step = epoch * len(dataloader) + batch_idx
        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar('train/total_loss', total_batch_loss.item(), global_step)
            writer.add_scalar('train/diffusion_loss', diffusion_loss.item(), global_step)
            writer.add_scalar('train/recon_loss', recon_loss.item(), global_step)
            
            for name, value in loss_components.items():
                writer.add_scalar(f'train/loss_{name}', value, global_step)
    
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
    """Validation loop"""
    model.eval()
    total_loss = 0
    total_diffusion_loss = 0
    total_recon_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
        condition = batch['hicarn'].to(device)
        gt = batch['gt'].to(device)
        residual_clean = batch['residual'].to(device)
        
        batch_size = condition.shape[0]
        
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (batch_size,),
            device=device
        ).long()
        
        noise = torch.randn_like(residual_clean)
        residual_noisy = scheduler.add_noise(residual_clean, noise, timesteps)
        
        model_output = model(residual_noisy, timesteps, condition)
        
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
        
        diffusion_loss = F.mse_loss(model_output, target)
        
        pred_gt = condition + pred_residual_clean
        recon_loss = criterion(pred_gt, gt)
        
        total_batch_loss = diffusion_loss + 0.1 * recon_loss
        
        total_loss += total_batch_loss.item()
        total_diffusion_loss += diffusion_loss.item()
        total_recon_loss += recon_loss.item()
        
        # Save sample images
        if batch_idx == 0 and writer is not None:
            for i in range(min(num_samples_to_save, batch_size)):
                writer.add_image(f'val/sample_{i}/condition', condition[i, 0:1].cpu(), epoch)
                writer.add_image(f'val/sample_{i}/gt', gt[i, 0:1].cpu(), epoch)
                writer.add_image(f'val/sample_{i}/prediction', pred_gt[i, 0:1].cpu(), epoch)
                writer.add_image(f'val/sample_{i}/residual', pred_residual_clean[i, 0:1].cpu(), epoch)
    
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
    
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow
    
    torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pt')
    
    if is_best:
        torch.save(checkpoint, save_dir / 'checkpoint_best.pt')
    
    print(f"Checkpoint saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Residual Diffusion - Enhanced V2')
    
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
    
    # Loss - NEW: Enhanced loss strategies
    parser.add_argument('--loss_config', type=str, default='localization_focused',
                       choices=['localization_focused', 'balanced', 'structure_focused'],
                       help='Loss strategy: localization_focused (recommended), balanced, or structure_focused')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--use_amp', action='store_true', default=True)
    
    # Paths
    parser.add_argument('--output_dir', type=str, default='checkpoints_v2_enhanced')
    parser.add_argument('--log_dir', type=str, default='runs_v2_enhanced')
    
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
    
    # Create loss - NEW: Enhanced with freq separation + localization
    print(f"\n{'='*80}")
    print(f"ENHANCED LOSS CONFIGURATION: {args.loss_config}")
    print(f"{'='*80}")
    
    if HAS_ENHANCED_LOSSES:
        loss_config = get_enhanced_loss_config(args.loss_config)
        criterion = EnhancedCombinedLoss(**loss_config).to(device)
        
        print("\n📊 Loss Components:")
        print(f"  • Frequency-separated recon: λ={loss_config['lambda_freq_separated']:.1f}")
        print(f"    └─ Constrains LOW-FREQ/STRUCTURE only (blur σ={loss_config['blur_sigma']})")
        if loss_config['use_heatmap_kl']:
            print(f"  • Heatmap KL (peak positioning): λ={loss_config['lambda_heatmap_kl']:.1f}")
            print(f"    └─ Ensures peaks in CORRECT POSITIONS (τ={loss_config['kl_temperature']})")
        if loss_config['use_gradient_consistency']:
            print(f"  • Gradient consistency: λ={loss_config['lambda_gradient_consistency']:.1f}")
            print(f"    └─ Prevents peak SPREADING (sharp peaks)")
        print(f"  • Base MSE: λ={loss_config['lambda_base']:.1f}")
        
        print("\n✅ Expected Improvements:")
        print("  ✓ Better hit@1, IoU (peak positioning)")
        print("  ✓ Sharper peaks (gradient consistency)")
        print("  ✓ Better TAD/structure (low-freq recon)")
        print("  ✓ Doesn't hurt loops (diffusion handles high-freq)")
    else:
        print("\n⚠️  Fallback: Using standard losses")
        from losses import CombinedResidualLoss, get_loss_weights
        loss_weights = get_loss_weights('peak_focused')
        criterion = CombinedResidualLoss(**loss_weights).to(device)
    
    print(f"{'='*80}\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # LR scheduler
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
        print(f"\n{'='*80}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*80}")
        print(f"Checkpoint: {args.resume}\n")
        
        checkpoint = torch.load(args.resume, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  ✓ Loaded model weights")
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("  ✓ Loaded optimizer state")
        
        if ema and 'ema_shadow' in checkpoint and checkpoint['ema_shadow']:
            ema.shadow = checkpoint['ema_shadow']
            ema.collected_params = checkpoint.get('ema_collected_params', None)
            print("  ✓ Loaded EMA weights")
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"\n  → Resuming from epoch {start_epoch}")
        print(f"  → Best val loss: {best_val_loss:.4f}")
        print(f"{'='*80}\n")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_diff, train_recon = train_one_epoch(
            model, ema, train_loader, scheduler, criterion,
            optimizer, device, epoch, writer, args.use_amp,
            timestep_weights
        )
        
        print(f"Train - Loss: {train_loss:.4f}, Diff: {train_diff:.4f}, Recon: {train_recon:.4f}")
        
        if val_loader is not None:
            if ema is not None:
                ema.apply_shadow()
            
            val_loss, val_diff, val_recon = validate(
                model, val_loader, scheduler, criterion,
                device, epoch, writer
            )
            
            if ema is not None:
                ema.restore()
            
            print(f"Val - Loss: {val_loss:.4f}, Diff: {val_diff:.4f}, Recon: {val_recon:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, ema, optimizer, scheduler,
                    epoch, best_val_loss, args.output_dir, is_best=True
                )
        
        if epoch % 10 == 0:
            save_checkpoint(
                model, ema, optimizer, scheduler,
                epoch, best_val_loss, args.output_dir
            )
        
        lr_scheduler.step()
        
        if writer is not None:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    
    print("\n🎉 Training complete!")
    writer.close()


if __name__ == '__main__':
    main()
