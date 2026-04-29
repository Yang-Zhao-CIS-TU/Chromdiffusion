"""
Structure-Preserved Residual Diffusion Training Script - Multi-GPU Version

Supports both single-GPU and multi-GPU training using DistributedDataParallel.

Usage:
    # Single GPU
    python train_structure_preserved_diffusion_multigpu.py \
        --pred_path hicarn_predictions/predictions_norm.npy \
        --gt_path hicarn_predictions/ground_truth.npy \
        --output_dir checkpoints_structure \
        --epochs 100 \
        --batch_size 16 \
        --gpu 2
    
    # Multi-GPU (e.g., GPUs 2,3,4,5)
    python train_structure_preserved_diffusion_multigpu.py \
        --pred_path hicarn_predictions/predictions_norm.npy \
        --gt_path hicarn_predictions/ground_truth.npy \
        --output_dir checkpoints_structure \
        --epochs 100 \
        --batch_size 16 \
        --gpus 2 3 4 5
    
    # Multi-GPU with torchrun (recommended for multi-node)
    torchrun --nproc_per_node=4 train_structure_preserved_diffusion_multigpu.py \
        --pred_path hicarn_predictions/predictions_norm.npy \
        --gt_path hicarn_predictions/ground_truth.npy \
        --output_dir checkpoints_structure \
        --epochs 100 \
        --batch_size 16 \
        --distributed
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

# Import structure-oriented losses
from structure_losses_stable import (
    StableStructureLossCalculator,
    ResidualClipper
)

# Import sanity check visualization
try:
    from sanity_check_visualization import visualize_batch_sanity_check
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠ Warning: sanity_check_visualization.py not found, skipping visualizations")

# Import diffusion model and scheduler
import sys
sys.path.insert(0, 'residual_diffusion')
from model import ResidualDiffusionUNet
from scheduler import DDPMScheduler
from data_loader import ResidualNormalizer


class ResidualDataset(Dataset):
    """Dataset for Structure-Preserved Residual Diffusion"""
    def __init__(self, pred_path, gt_path):
        self.pred_norm = np.load(pred_path)
        self.gt_norm = np.load(gt_path)
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


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


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
    """
    Train for one epoch with STABLE structure-oriented losses
    
    KEY CHANGES:
    - Use counters for valid batches (not list append)
    - Track validity explicitly
    - Proper loss aggregation with n_valid tracking
    """
    model.train()
    
    # Use COUNTERS for loss aggregation (not lists!)
    epoch_losses = {
        'total_sum': 0.0,
        'diffusion_sum': 0.0,
        'residual_sum': 0.0,
        'insulation_sum': 0.0,
        'tad_boundary_sum': 0.0,
        'low_freq_sum': 0.0,
        'n_total': 0,
        'n_diffusion': 0,
        'n_residual': 0,
        'n_insulation': 0,
        'n_tad': 0,
        'n_lf': 0
    }
    
    # Only show progress bar on rank 0
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
        
        # Add noise to target residual
        noise = torch.randn_like(target_residual_norm)
        noisy_residual = scheduler_diff.add_noise(target_residual_norm, noise, t)
        
        # Predict noise (condition on HiCARN prediction)
        noise_pred = model(noisy_residual, t, hicarn_pred)
        
        # Compute diffusion loss on noise prediction
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        # Estimate predicted residual from noise prediction
        with torch.no_grad():
            # Get alpha values for current timesteps
            alpha_bar_t = scheduler_diff.alphas_cumprod[t].view(-1, 1, 1, 1)
            
            # Predict x_0 (residual in normalized space)
            pred_residual_norm = (noisy_residual - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            
            # Denormalize
            pred_residual = normalizer.inverse_transform(pred_residual_norm.cpu().numpy())
            pred_residual = torch.from_numpy(pred_residual).to(device)
            
            # CRITICAL: Clip residual to prevent large changes
            pred_residual = residual_clipper.clip_residual(pred_residual, hicarn_pred)
        
        # Construct predicted Hi-C
        pred_hic = hicarn_pred + pred_residual
        target_hic = gt
        
        # Compute structure-oriented loss components
        # Returns (loss, loss_dict) where loss_dict has validity flags
        structure_loss, loss_dict = loss_calculator(
            pred_residual, target_residual,
            pred_hic, target_hic
        )
        
        # Total loss = diffusion + structure
        total_loss = diffusion_loss + structure_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # PROPER LOSS AGGREGATION with validity tracking
        epoch_losses['total_sum'] += total_loss.item()
        epoch_losses['n_total'] += 1
        
        epoch_losses['diffusion_sum'] += diffusion_loss.item()
        epoch_losses['n_diffusion'] += 1
        
        # Track valid structure losses
        if loss_dict['residual'] is not None:
            epoch_losses['residual_sum'] += loss_dict['residual']
            epoch_losses['n_residual'] += 1
        
        if loss_dict['insulation'] is not None:
            epoch_losses['insulation_sum'] += loss_dict['insulation']
            epoch_losses['n_insulation'] += 1
        
        if loss_dict['tad_boundary'] is not None:
            epoch_losses['tad_boundary_sum'] += loss_dict['tad_boundary']
            epoch_losses['n_tad'] += 1
        
        if loss_dict['low_freq'] is not None:
            epoch_losses['low_freq_sum'] += loss_dict['low_freq']
            epoch_losses['n_lf'] += 1
        
        # Update progress bar (only on rank 0)
        if rank == 0:
            # Show current batch values
            pbar.set_postfix({
                'total': f"{total_loss.item():.4f}",
                'diff': f"{diffusion_loss.item():.4f}",
                'ins': f"{loss_dict['insulation']:.4f}" if loss_dict['insulation'] is not None else "N/A",
                'tad': f"{loss_dict['tad_boundary']:.4f}" if loss_dict['tad_boundary'] is not None else "N/A",
                'n_ins': epoch_losses['n_insulation'],
                'n_tad': epoch_losses['n_tad']
            })
    
    # Compute CORRECT epoch averages using counters
    avg_losses = {
        'total': epoch_losses['total_sum'] / max(epoch_losses['n_total'], 1),
        'diffusion': epoch_losses['diffusion_sum'] / max(epoch_losses['n_diffusion'], 1),
        'residual': epoch_losses['residual_sum'] / max(epoch_losses['n_residual'], 1),
        'insulation': epoch_losses['insulation_sum'] / max(epoch_losses['n_insulation'], 1),
        'tad_boundary': epoch_losses['tad_boundary_sum'] / max(epoch_losses['n_tad'], 1),
        'low_freq': epoch_losses['low_freq_sum'] / max(epoch_losses['n_lf'], 1),
        # Also track validity rates
        'valid_insulation_rate': epoch_losses['n_insulation'] / max(epoch_losses['n_total'], 1),
        'valid_tad_rate': epoch_losses['n_tad'] / max(epoch_losses['n_total'], 1),
        'valid_lf_rate': epoch_losses['n_lf'] / max(epoch_losses['n_total'], 1)
    }
    
    # Synchronize losses across all GPUs
    if world_size > 1:
        for key in avg_losses.keys():
            loss_tensor = torch.tensor(avg_losses[key], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_losses[key] = loss_tensor.item()
    
    return avg_losses


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
    """Validation with structure-oriented metrics"""
    model.eval()
    
    val_losses = {
        'total': [],
        'diffusion': [],
        'residual': [],
        'insulation': [],
        'tad_boundary': [],
        'low_freq': []
    }
    
    with torch.no_grad():
        # Only show progress bar on rank 0
        if rank == 0:
            pbar = tqdm(dataloader, desc="Validating")
        else:
            pbar = dataloader
        
        for batch in pbar:
            hicarn_pred = batch['hicarn_pred'].to(device)
            gt = batch['gt'].to(device)
            target_residual = batch['residual'].to(device)
            
            batch_size = hicarn_pred.shape[0]
            
            # Normalize target residual
            target_residual_norm = normalizer.transform(target_residual.cpu().numpy())
            target_residual_norm = torch.from_numpy(target_residual_norm).to(device)
            
            # Full denoising (expensive but accurate)
            residual_t = torch.randn_like(target_residual_norm)
            
            # Iteratively denoise
            for t_val in reversed(range(scheduler_diff.num_train_timesteps)):
                t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
                noise_pred = model(residual_t, t_tensor, hicarn_pred)
                
                # Denoise one step (process sample by sample)
                residual_t_new = torch.zeros_like(residual_t)
                for i in range(batch_size):
                    residual_t_new[i:i+1], _ = scheduler_diff.step(
                        noise_pred[i:i+1], 
                        t_val,
                        residual_t[i:i+1]
                    )
                residual_t = residual_t_new
            
            pred_residual_norm = residual_t
            
            # Denormalize
            pred_residual = normalizer.inverse_transform(pred_residual_norm.cpu().numpy())
            pred_residual = torch.from_numpy(pred_residual).to(device)
            
            # Clip residual
            pred_residual = residual_clipper.clip_residual(pred_residual, hicarn_pred)
            
            # Construct Hi-C
            pred_hic = hicarn_pred + pred_residual
            target_hic = gt
            
            # Compute losses
            total_loss, loss_dict = loss_calculator(
                pred_residual, target_residual,
                pred_hic, target_hic
            )
            
            # Add diffusion loss placeholder
            loss_dict['diffusion'] = 0.0
            
            for key in val_losses.keys():
                val_losses[key].append(loss_dict[key])
    
    avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
    
    # Synchronize losses across all GPUs
    if world_size > 1:
        for key in avg_losses.keys():
            loss_tensor = torch.tensor(avg_losses[key], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_losses[key] = loss_tensor.item()
    
    return avg_losses


def parse_args():
    parser = argparse.ArgumentParser(description='Structure-Preserved Residual Diffusion Training (Multi-GPU)')
    
    # Data
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints_structure_diffusion')
    
    # Model architecture
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    
    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    
    # Loss weights
    parser.add_argument('--lambda_residual', type=float, default=0.1)
    parser.add_argument('--lambda_insulation', type=float, default=1.0)
    parser.add_argument('--lambda_tad_boundary', type=float, default=0.5)
    parser.add_argument('--lambda_low_freq', type=float, default=0.5)
    
    # Structure loss parameters
    parser.add_argument('--insulation_window', type=int, default=5,
                       help='Window size for insulation score (default: 5 for 40x40 matrices)')
    
    # Backbone freezing (P2 fix)
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0,
                       help='Freeze UNet backbone for first N epochs (P2 stability fix, try 5)')
    
    # Residual clipping
    parser.add_argument('--clip_factor', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--val_frequency', type=int, default=1,
                       help='Validate every N epochs (default: 1, set to 5 for faster training)')
    parser.add_argument('--vis_frequency', type=int, default=5,
                       help='Save sanity check visualizations every N epochs (default: 5, 0=disable)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., checkpoints/checkpoint_epoch_20.pt)')
    parser.add_argument('--resume_from_best', action='store_true',
                       help='Resume from best_model_structure_preserved.pt in output_dir')
    parser.add_argument('--resume_from_last', action='store_true',
                       help='Resume from final_model_structure_preserved.pt in output_dir')
    
    # Multi-GPU
    parser.add_argument('--gpu', type=int, default=None,
                       help='Single GPU ID (for single-GPU training)')
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                       help='Multiple GPU IDs (e.g., --gpus 2 3 4 5)')
    parser.add_argument('--distributed', action='store_true',
                       help='Use with torchrun for distributed training')
    parser.add_argument('--num_workers', type=int, default=4)
    
    return parser.parse_args()


def main_worker(rank, world_size, args, gpu_ids=None):
    """Worker function for each process in distributed training"""
    
    # Setup distributed
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Set device
    if gpu_ids is not None:
        device = torch.device(f'cuda:{gpu_ids[rank]}')
        torch.cuda.set_device(gpu_ids[rank])
    else:
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
    
    # Only print from rank 0
    is_main_process = (rank == 0)
    
    if is_main_process:
        print(f"Process rank: {rank}/{world_size}")
        print(f"Device: {device}")
    
    # Create output directory (only on main process)
    if is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("STRUCTURE-PRESERVED RESIDUAL DIFFUSION TRAINING")
        if world_size > 1:
            print(f"MULTI-GPU MODE: {world_size} GPUs")
            if gpu_ids:
                print(f"GPU IDs: {gpu_ids}")
        print("="*80)
        print(f"\nLoss Weights:")
        print(f"  λ_residual:     {args.lambda_residual:.2f}")
        print(f"  λ_insulation:   {args.lambda_insulation:.2f}")
        print(f"  λ_tad_boundary: {args.lambda_tad_boundary:.2f}")
        print(f"  λ_low_freq:     {args.lambda_low_freq:.2f}")
        print("="*80)
    
    # Load dataset
    if is_main_process:
        print("\n[1/6] Loading dataset...")
    
    dataset = ResidualDataset(args.pred_path, args.gt_path)
    
    # Split train/val
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    # Use DistributedSampler for multi-GPU
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if is_main_process:
        print(f"  Train samples: {n_train}")
        print(f"  Val samples: {n_val}")
        print(f"  Effective batch size: {args.batch_size * world_size}")
    
    # Create model
    if is_main_process:
        print("\n[2/6] Creating model...")
    
    model = ResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        num_res_blocks=args.num_res_blocks
    ).to(device)
    
    # Wrap with DDP for multi-GPU
    if world_size > 1:
        # Use find_unused_parameters=True to handle backbone freezing
        model = DDP(
            model, 
            device_ids=[device.index], 
            output_device=device.index,
            find_unused_parameters=(args.freeze_backbone_epochs > 0)
        )
    
    if is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")
    
    # Create diffusion scheduler
    if is_main_process:
        print("\n[3/6] Creating diffusion scheduler...")
    
    scheduler_diff = DDPMScheduler(
        num_train_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule
    )
    
    # Create loss calculator
    if is_main_process:
        print("\n[4/6] Creating STABLE structure-oriented loss calculator...")
        print(f"  Insulation window size: {args.insulation_window}")
        print(f"  Using variance normalization for stability")
    
    loss_calculator = StableStructureLossCalculator(
        lambda_residual=args.lambda_residual,
        lambda_insulation=args.lambda_insulation,
        lambda_tad_boundary=args.lambda_tad_boundary,
        lambda_low_freq=args.lambda_low_freq,
        insulation_window=args.insulation_window
    ).to(device)
    
    # Create residual clipper
    residual_clipper = ResidualClipper(clip_factor=args.clip_factor)
    
    # Create residual normalizer
    if is_main_process:
        print("\n[5/6] Fitting residual normalizer...")
    
    normalizer = ResidualNormalizer()
    
    # Only fit on rank 0, then broadcast
    if rank == 0:
        all_residuals = []
        for batch in train_loader:
            all_residuals.append(batch['residual'].numpy())
        all_residuals = np.concatenate(all_residuals, axis=0)
        normalizer.fit(all_residuals)
        
        if is_main_process:
            print(f"  Residual mean: {normalizer.mean:.6f}")
            print(f"  Residual std:  {normalizer.std:.6f}")
    
    # Broadcast normalizer stats to all ranks
    if world_size > 1:
        stats = torch.tensor([normalizer.mean if rank == 0 else 0.0, 
                             normalizer.std if rank == 0 else 0.0], device=device)
        dist.broadcast(stats, src=0)
        normalizer.mean = stats[0].item()
        normalizer.std = stats[1].item()
        normalizer.fitted = True
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    training_history = []
    
    # Determine checkpoint path
    checkpoint_path = None
    if args.resume:
        checkpoint_path = args.resume
    elif args.resume_from_best:
        checkpoint_path = Path(args.output_dir) / 'best_model_structure_preserved.pt'
    elif args.resume_from_last:
        checkpoint_path = Path(args.output_dir) / 'final_model_structure_preserved.pt'
    
    if checkpoint_path is not None:
        if Path(checkpoint_path).exists():
            if is_main_process:
                print(f"\n🔄 Resuming from checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            if world_size > 1:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
            best_val_loss = checkpoint.get('loss', float('inf'))
            
            # Load normalizer
            if 'normalizer_mean' in checkpoint and 'normalizer_std' in checkpoint:
                normalizer.mean = checkpoint['normalizer_mean']
                normalizer.std = checkpoint['normalizer_std']
                normalizer.fitted = True
            
            # Try to load training history
            history_path = Path(args.output_dir) / 'training_history.json'
            if history_path.exists():
                with open(history_path, 'r') as f:
                    training_history = json.load(f)
            
            if is_main_process:
                print(f"  ✓ Resuming from epoch {start_epoch}")
                print(f"  ✓ Best validation loss so far: {best_val_loss:.4f}")
                print(f"  ✓ Loaded training history: {len(training_history)} epochs")
        else:
            if is_main_process:
                print(f"\n⚠ Warning: Checkpoint not found: {checkpoint_path}")
                print(f"  Starting training from scratch...")
    else:
        if is_main_process and (args.resume or args.resume_from_best or args.resume_from_last):
            print(f"\n⚠ Warning: No valid checkpoint found, starting from scratch")
    
    # Training loop
    if is_main_process:
        print("\n[6/6] Starting training...")
        if start_epoch > 0:
            print(f"  Resuming from epoch {start_epoch}")
        print("="*80)
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        # P2 FIX: Freeze/unfreeze backbone based on epoch
        if args.freeze_backbone_epochs > 0:
            # Get actual model (unwrap DDP if needed)
            actual_model = model.module if world_size > 1 else model
            
            if epoch < args.freeze_backbone_epochs:
                # Freeze backbone (early layers)
                if is_main_process and epoch == 0:
                    print(f"\n🔒 Freezing backbone for first {args.freeze_backbone_epochs} epochs")
                
                # Freeze all parameters first
                for param in actual_model.parameters():
                    param.requires_grad = False
                
                # Unfreeze only final layers
                # Unfreeze output conv
                if hasattr(actual_model, 'out_conv'):
                    for param in actual_model.out_conv.parameters():
                        param.requires_grad = True
                
                # Unfreeze last decoder block
                if hasattr(actual_model, 'up_blocks') and len(actual_model.up_blocks) > 0:
                    for param in actual_model.up_blocks[-1].parameters():
                        param.requires_grad = True
                
                # Unfreeze time embedding
                if hasattr(actual_model, 'time_mlp'):
                    for param in actual_model.time_mlp.parameters():
                        param.requires_grad = True
                
                # Unfreeze condition projection if exists
                if hasattr(actual_model, 'cond_proj'):
                    for param in actual_model.cond_proj.parameters():
                        param.requires_grad = True
                
            elif epoch == args.freeze_backbone_epochs:
                # Unfreeze all parameters
                if is_main_process:
                    print(f"\n🔓 Unfreezing all parameters at epoch {epoch}")
                
                for param in actual_model.parameters():
                    param.requires_grad = True
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, scheduler_diff,
            loss_calculator, residual_clipper, normalizer,
            device, epoch, rank, world_size
        )
        
        # Validate (conditionally based on frequency)
        should_validate = (epoch % args.val_frequency == 0) or (epoch == args.epochs - 1)
        
        if should_validate:
            val_losses = validate(
                model, val_loader, scheduler_diff,
                loss_calculator, residual_clipper, normalizer,
                device, rank, world_size
            )
        else:
            # Use previous validation losses
            if epoch > 0 and len(training_history) > 0:
                val_losses = training_history[-1]['val']
            else:
                val_losses = {k: 0.0 for k in train_losses.keys()}
        
        # Sanity check visualization (only on rank 0)
        should_visualize = (
            VISUALIZATION_AVAILABLE and 
            is_main_process and 
            args.vis_frequency > 0 and 
            (epoch % args.vis_frequency == 0 or epoch == args.epochs - 1)
        )
        
        if should_visualize:
            # Get a batch for visualization
            vis_batch = next(iter(val_loader))
            hicarn_pred_vis = vis_batch['hicarn_pred'].to(device)
            gt_vis = vis_batch['gt'].to(device)
            
            # Generate refined predictions
            with torch.no_grad():
                model.eval()
                
                # Start from noise
                residual_t = torch.randn_like(hicarn_pred_vis)
                
                # Denoise (use first 100 steps for speed)
                for t_val in reversed(range(0, min(100, scheduler_diff.num_train_timesteps), 10)):
                    t_tensor = torch.full((hicarn_pred_vis.shape[0],), t_val, device=device, dtype=torch.long)
                    noise_pred = model(residual_t, t_tensor, hicarn_pred_vis)
                    
                    for i in range(residual_t.shape[0]):
                        residual_t[i:i+1], _ = scheduler_diff.step(
                            noise_pred[i:i+1], t_val, residual_t[i:i+1]
                        )
                
                # Denormalize
                pred_residual = normalizer.inverse_transform(residual_t.cpu().numpy())
                pred_residual = torch.from_numpy(pred_residual).to(device)
                pred_residual = residual_clipper.clip_residual(pred_residual, hicarn_pred_vis)
                
                refined_vis = hicarn_pred_vis + pred_residual
                
                model.train()
            
            # Create visualizations
            vis_dir = Path(args.output_dir) / 'visualizations'
            visualize_batch_sanity_check(
                gt_batch=gt_vis.cpu(),
                hicarn_batch=hicarn_pred_vis.cpu(),
                refined_batch=refined_vis.cpu(),
                epoch=epoch,
                save_dir=vis_dir,
                n_samples=3
            )
        
        # Update LR
        lr_scheduler.step()
        
        # Log (only on rank 0)
        if is_main_process:
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Total: {train_losses['total']:.4f}, "
                  f"Diff: {train_losses['diffusion']:.4f}, "
                  f"Ins: {train_losses['insulation']:.4f}, "
                  f"TAD: {train_losses['tad_boundary']:.4f}, "
                  f"LF: {train_losses['low_freq']:.4f}")
            print(f"  Valid - Ins: {train_losses['valid_insulation_rate']*100:.1f}%, "
                  f"TAD: {train_losses['valid_tad_rate']*100:.1f}%, "
                  f"LF: {train_losses['valid_lf_rate']*100:.1f}%")
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
                
                # Get model state dict (unwrap DDP if needed)
                model_state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'config': vars(args),
                    'normalizer_mean': normalizer.mean,
                    'normalizer_std': normalizer.std
                }
                
                torch.save(checkpoint, Path(args.output_dir) / 'best_model_structure_preserved.pt')
                print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                model_state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': vars(args),
                    'normalizer_mean': normalizer.mean,
                    'normalizer_std': normalizer.std
                }
                torch.save(checkpoint, Path(args.output_dir) / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model (only on rank 0)
    if is_main_process:
        model_state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
        checkpoint = {
            'epoch': args.epochs - 1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(args),
            'normalizer_mean': normalizer.mean,
            'normalizer_std': normalizer.std
        }
        torch.save(checkpoint, Path(args.output_dir) / 'final_model_structure_preserved.pt')
        
        # Save training history
        with open(Path(args.output_dir) / 'training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        print(f"Models saved to: {args.output_dir}/")
        print("="*80)
    
    # Cleanup
    if world_size > 1:
        cleanup_distributed()


def main():
    args = parse_args()
    
    # Determine distributed setup
    if args.distributed:
        # Using torchrun
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        main_worker(rank, world_size, args)
        
    elif args.gpus is not None and len(args.gpus) > 1:
        # Manual multi-GPU setup using spawn
        world_size = len(args.gpus)
        gpu_ids = args.gpus
        
        print(f"Starting multi-GPU training on GPUs: {gpu_ids}")
        print(f"World size: {world_size}")
        print("="*80)
        
        # Use spawn to create processes
        mp.spawn(
            main_worker,
            args=(world_size, args, gpu_ids),
            nprocs=world_size,
            join=True
        )
        
    else:
        # Single GPU
        gpu_id = args.gpu if args.gpu is not None else 0
        print(f"Using single GPU: {gpu_id}")
        main_worker(0, 1, args, [gpu_id])


if __name__ == "__main__":
    main()
