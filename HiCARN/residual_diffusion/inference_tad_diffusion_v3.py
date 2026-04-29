"""
TAD-Only Residual Diffusion Inference Script
Architecture matches the training checkpoint

Output files:
  - refined_norm.npy: Refined predictions in normalized space
  - refined_raw.npy: Refined predictions in raw contact counts (denormalized)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
from pathlib import Path
import math


class DDPMScheduler:
    """Simple DDPM Scheduler (no external dependencies)"""
    def __init__(self, num_train_timesteps=1000, beta_schedule='linear', prediction_type='epsilon'):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        self.timesteps = None
    
    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_inference_steps) * step_ratio
        self.timesteps = torch.flip(self.timesteps, dims=[0])
    
    def step(self, model_output, timestep, sample):
        t = timestep
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")
        
        pred_original_sample_coeff = torch.sqrt(alpha_prod_t_prev) * self.betas[t] / beta_prod_t
        current_sample_coeff = torch.sqrt(self.alphas[t]) * (1 - alpha_prod_t_prev) / beta_prod_t
        
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        class StepOutput:
            def __init__(self, prev_sample):
                self.prev_sample = prev_sample
        
        return StepOutput(pred_prev_sample)


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResNetBlock(nn.Module):
    """ResNet block with time embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Only use Conv for residual when channels differ, else Identity
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        
        return h + self.residual_conv(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv.unbind(1)
        
        attn = torch.einsum('bhci,bhcj->bhij', q, k) / math.sqrt(c // self.num_heads)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhij,bhcj->bhci', attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        
        return out + x


class ResidualDiffusionUNet(nn.Module):
    """
    Full U-Net with attention for TAD residual diffusion
    Architecture matches the training checkpoint
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        cond_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_levels=(2, 3),
        time_emb_dim=256,
        dropout=0.1
    ):
        super().__init__()
        
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_levels = len(channel_mults)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Condition encoder (HiCARN prediction)
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        
        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        
        channels = []  # For skip connections
        in_ch = base_channels
        
        for level in range(self.num_levels):
            out_ch = base_channels * channel_mults[level]
            
            # ResNet blocks for this level
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResNetBlock(in_ch, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
            self.encoder_blocks.append(blocks)
            
            # Save output channels for skip connection
            channels.append(in_ch)
            
            # Attention for this level
            if level in attn_levels:
                attns = nn.ModuleList([SelfAttention(in_ch) for _ in range(num_res_blocks)])
            else:
                attns = nn.ModuleList([nn.Identity() for _ in range(num_res_blocks)])
            self.encoder_attns.append(attns)
            
            # Downsampling (except last level)
            if level < self.num_levels - 1:
                self.encoder_downsamples.append(
                    nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
                )
        
        # Middle
        self.mid_block1 = ResNetBlock(in_ch, in_ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(in_ch)
        self.mid_block2 = ResNetBlock(in_ch, in_ch, time_emb_dim, dropout)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        for level in reversed(range(self.num_levels)):
            out_ch = base_channels * channel_mults[level]
            
            # ResNet blocks for this level
            # Block 0: concatenates with skip connection
            # Block 1, 2: regular blocks without skip
            blocks = nn.ModuleList()
            
            # First block: concatenate with skip
            skip_ch = channels.pop()
            blocks.append(ResNetBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout))
            in_ch = out_ch
            
            # Remaining blocks: no skip connection
            for _ in range(num_res_blocks):
                blocks.append(ResNetBlock(in_ch, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
            
            self.decoder_blocks.append(blocks)
            
            # Attention for this level
            # Decoder iterates from deepest to shallowest: level 3->2->1->0
            # decoder_idx: 0->1->2->3
            # Checkpoint has attention at decoder_attns[0] and [1] (deepest levels)
            decoder_idx = self.num_levels - 1 - level
            if decoder_idx < 2:  # decoder_idx 0, 1 (deepest two levels)
                attns = nn.ModuleList([SelfAttention(in_ch) for _ in range(num_res_blocks + 1)])
            else:
                attns = nn.ModuleList([nn.Identity() for _ in range(num_res_blocks + 1)])
            self.decoder_attns.append(attns)
            
            # Upsampling (except last level)
            if level > 0:
                self.decoder_upsamples.append(
                    nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
                )
        
        # Output
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x, timesteps, condition):
        """
        Args:
            x: Noisy residual [B, 1, H, W]
            timesteps: Timesteps [B]
            condition: HiCARN prediction [B, 1, H, W]
        """
        # Time embedding
        t_emb = self.time_mlp(timesteps.float())
        
        # Encode condition
        cond = self.cond_encoder(condition)
        
        # Input
        h = self.input_conv(x)
        h = h + cond  # Add condition
        
        # Encoder
        skips = []
        for level in range(self.num_levels):
            for block, attn in zip(self.encoder_blocks[level], self.encoder_attns[level]):
                h = block(h, t_emb)
                h = attn(h)
            
            # Save skip connection after processing all blocks at this level
            skips.append(h)
            
            # Downsample (except last level)
            if level < self.num_levels - 1:
                h = self.encoder_downsamples[level](h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        for level in range(self.num_levels):
            blocks = self.decoder_blocks[level]
            attns = self.decoder_attns[level]
            
            # First block: concatenate with skip
            h = torch.cat([h, skips.pop()], dim=1)
            h = blocks[0](h, t_emb)
            h = attns[0](h)
            
            # Remaining blocks: no skip concatenation
            for i in range(1, len(blocks)):
                h = blocks[i](h, t_emb)
                h = attns[i](h)
            
            # Upsample after processing all blocks at this level
            if level < self.num_levels - 1:
                h = self.decoder_upsamples[level](h)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h


class RobustHiCPreprocessor:
    """HiC Preprocessor for denormalization"""
    def __init__(self):
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self.fitted = False
    
    def postprocess(self, Y_norm):
        """Denormalize predictions back to raw contact counts"""
        input_shape = Y_norm.shape
        if Y_norm.ndim == 3:
            Y_norm = Y_norm[:, np.newaxis, :, :]
        
        Y_norm_clipped = np.clip(Y_norm, -5, 5)
        Y_log = Y_norm_clipped * self.Y_std + self.Y_mean
        Y_raw = np.expm1(Y_log)
        Y_raw = np.maximum(Y_raw, 0.0)
        
        if len(input_shape) == 3:
            Y_raw = Y_raw.squeeze(1)
        
        return Y_raw


def load_model(checkpoint_path, device):
    """Load trained diffusion model"""
    print(f"Loading model from: {checkpoint_path}")
    
    model = ResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        cond_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_levels=(2, 3),
        time_emb_dim=256,
        dropout=0.1
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def load_preprocessor(preprocessor_path):
    """Load HiCARN preprocessor for denormalization"""
    if not Path(preprocessor_path).exists():
        print(f"⚠ Warning: Preprocessor not found at {preprocessor_path}")
        print("  Denormalization will not be accurate without preprocessor!")
        return None
    
    print(f"Loading preprocessor from: {preprocessor_path}")
    preprocessor = torch.load(preprocessor_path, map_location='cpu')
    print(f"✓ Loaded preprocessor")
    print(f"  Y_mean: {preprocessor.Y_mean:.4f}")
    print(f"  Y_std: {preprocessor.Y_std:.4f}")
    
    return preprocessor


def sample_tad_residual(model, scheduler, hicarn_pred, device, num_steps=50):
    """Sample TAD residual using DDPM"""
    residual_tad = torch.randn_like(hicarn_pred).to(device)
    scheduler.set_timesteps(num_steps)
    
    for t in scheduler.timesteps:
        t_batch = torch.full((hicarn_pred.shape[0],), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            noise_pred = model(residual_tad, t_batch, hicarn_pred)
        
        residual_tad = scheduler.step(noise_pred, t, residual_tad).prev_sample
    
    return residual_tad


def apply_loop_masking(residual_tad, hicarn_pred, loop_percentile=90):
    """Apply loop masking to preserve loops from HiCARN"""
    threshold = torch.quantile(hicarn_pred, loop_percentile / 100.0)
    loop_mask = (hicarn_pred > threshold).float()
    residual_masked = residual_tad * (1.0 - loop_mask)
    return residual_masked, loop_mask


def refine_predictions(hicarn_pred, model, scheduler, device, args):
    """Refine HiCARN predictions by adding TAD residuals"""
    if hicarn_pred.ndim == 3:
        hicarn_pred = hicarn_pred[:, np.newaxis, :, :]
    
    batch_size = args.batch_size
    num_samples = len(hicarn_pred)
    
    refined_pred_list = []
    residuals_list = []
    loop_masks_list = []
    
    print(f"\nRefining {num_samples} predictions...")
    print(f"  Denoising steps: {args.num_steps}")
    print(f"  Loop percentile: {args.loop_percentile}")
    print(f"  Apply masking: {args.apply_masking}")
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Sampling"):
            batch_end = min(i + batch_size, num_samples)
            batch_pred = torch.from_numpy(hicarn_pred[i:batch_end]).float().to(device)
            
            residual_tad = sample_tad_residual(
                model, scheduler, batch_pred, device, args.num_steps
            )
            
            if args.apply_masking:
                residual_tad, loop_mask = apply_loop_masking(
                    residual_tad, batch_pred, args.loop_percentile
                )
            else:
                loop_mask = torch.zeros_like(residual_tad)
            
            refined = batch_pred + residual_tad
            refined = torch.clamp(refined, min=0.0)
            
            refined_pred_list.append(refined.cpu().numpy())
            residuals_list.append(residual_tad.cpu().numpy())
            loop_masks_list.append(loop_mask.cpu().numpy())
    
    refined_pred = np.concatenate(refined_pred_list, axis=0)
    residuals = np.concatenate(residuals_list, axis=0)
    loop_masks = np.concatenate(loop_masks_list, axis=0)
    
    return refined_pred, residuals, loop_masks


def denormalize_predictions(pred_norm, preprocessor):
    """Denormalize predictions to raw contact counts"""
    if preprocessor is None:
        print("\n⚠ Warning: No preprocessor available")
        print("  Returning normalized predictions")
        return pred_norm
    
    print("\nDenormalizing to raw contact counts...")
    pred_raw = preprocessor.postprocess(pred_norm)
    
    print(f"  Normalized range: [{pred_norm.min():.2f}, {pred_norm.max():.2f}]")
    print(f"  Raw range: [{pred_raw.min():.2f}, {pred_raw.max():.2f}]")
    
    return pred_raw


def parse_args():
    parser = argparse.ArgumentParser(description='TAD-only residual diffusion inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to HiCARN predictions (normalized)')
    parser.add_argument('--preprocessor_path', type=str,
                        default='hicarn_predictions/hicarn_preprocessor.pt',
                        help='Path to HiCARN preprocessor for denormalization')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of denoising steps (default: 50)')
    parser.add_argument('--loop_percentile', type=float, default=90,
                        help='Percentile threshold for loop masking (default: 90)')
    parser.add_argument('--apply_masking', action='store_true', default=True,
                        help='Apply loop masking (default: True)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    
    parser.add_argument('--save_residuals', action='store_true',
                        help='Save TAD residuals')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save loop masks')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (default: 0)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    model = load_model(args.checkpoint, device)
    preprocessor = load_preprocessor(args.preprocessor_path)
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon'
    )
    
    print(f"\nLoading HiCARN predictions: {args.pred_path}")
    hicarn_pred = np.load(args.pred_path)
    print(f"  Shape: {hicarn_pred.shape}")
    print(f"  Range: [{hicarn_pred.min():.2f}, {hicarn_pred.max():.2f}]")
    
    refined_norm, residuals, loop_masks = refine_predictions(
        hicarn_pred, model, scheduler, device, args
    )
    
    print(f"\nRefined predictions shape: {refined_norm.shape}")
    
    refined_raw = denormalize_predictions(refined_norm, preprocessor)
    
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)
    
    if args.apply_masking:
        loop_fraction = (loop_masks > 0.5).mean() * 100
        print(f"Loop regions masked: {loop_fraction:.2f}%")
    
    residual_strength = np.abs(residuals).mean()
    print(f"Mean absolute residual: {residual_strength:.4f}")
    
    print(f"\nValue ranges:")
    print(f"  HiCARN (normalized): [{hicarn_pred.min():.2f}, {hicarn_pred.max():.2f}]")
    print(f"  Refined (normalized): [{refined_norm.min():.2f}, {refined_norm.max():.2f}]")
    print(f"  Refined (raw): [{refined_raw.min():.2f}, {refined_raw.max():.2f}]")
    
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    norm_path = output_dir / 'refined_norm.npy'
    np.save(norm_path, refined_norm)
    print(f"✓ Saved normalized predictions: {norm_path}")
    
    raw_path = output_dir / 'refined_raw.npy'
    np.save(raw_path, refined_raw)
    print(f"✓ Saved raw predictions: {raw_path}")
    
    if args.save_residuals:
        residual_path = output_dir / 'residuals_tad.npy'
        np.save(residual_path, residuals)
        print(f"✓ Saved TAD residuals: {residual_path}")
    
    if args.save_masks:
        mask_path = output_dir / 'loop_masks.npy'
        np.save(mask_path, loop_masks)
        print(f"✓ Saved loop masks: {mask_path}")
    
    metadata = {
        'num_samples': int(len(refined_norm)),
        'num_steps': args.num_steps,
        'loop_percentile': args.loop_percentile,
        'apply_masking': args.apply_masking,
        'loop_fraction': float(loop_fraction) if args.apply_masking else 0.0,
        'residual_strength': float(residual_strength),
        'normalized_range': [float(refined_norm.min()), float(refined_norm.max())],
        'raw_range': [float(refined_raw.min()), float(refined_raw.max())],
        'shape': list(refined_norm.shape)
    }
    
    metadata_path = output_dir / 'inference_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)
    print(f"\nOutput files in {output_dir}:")
    print(f"  1. refined_norm.npy  - Normalized predictions")
    print(f"  2. refined_raw.npy   - Raw contact counts")
    print(f"  3. inference_metadata.json - Inference metadata")
    if args.save_residuals:
        print(f"  4. residuals_tad.npy - TAD residuals")
    if args.save_masks:
        print(f"  5. loop_masks.npy    - Loop masks")


if __name__ == '__main__':
    main()
