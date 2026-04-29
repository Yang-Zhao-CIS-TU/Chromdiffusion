"""
Residual Diffusion Model for Hi-C Refinement

This module defines the U-Net based diffusion model that learns to predict
residuals (Δ = x_GT - x_HiCARN) conditioned on HiCARN predictions.

Key principles:
- HiCARN is frozen (no backprop)
- Diffusion learns error correction, not generation
- Conditional on HiCARN predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep embedding using sinusoidal encoding"""
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


class ResidualBlock(nn.Module):
    """Residual block with time and condition embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t_emb)
        h = h + time_emb[:, :, None, None]
        
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        
        return F.silu(h + self.residual_conv(x))


class Attention(nn.Module):
    """Self-attention block"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = k.reshape(B, C, H * W)  # (B, C, HW)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        
        # Attention
        scale = C ** -0.5
        attn = torch.softmax(torch.bmm(q, k) * scale, dim=-1)
        h = torch.bmm(attn, v)
        
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class ResidualDiffusionUNet(nn.Module):
    """
    U-Net for residual diffusion
    
    Input: noisy residual Δ_t at timestep t
    Condition: HiCARN prediction x̃
    Output: predicted noise ε
    
    Architecture:
    - Encoder: downsample residual
    - Bottleneck: attention + residual blocks
    - Decoder: upsample and combine with condition
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        use_attention=True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Condition encoder (HiCARN prediction)
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        
        # Input projection (noisy residual)
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        
        ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            ch_out = base_channels * mult
            
            # Residual blocks for this level
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(ch, ch_out, time_emb_dim))
                if use_attention and i >= len(channel_multipliers) - 2:
                    attns.append(Attention(ch_out))
                else:
                    attns.append(nn.Identity())
                ch = ch_out
            
            self.encoder_blocks.append(blocks)
            self.encoder_attns.append(attns)
            
            # Downsample
            if i < len(channel_multipliers) - 1:
                self.encoder_downsamples.append(
                    nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1)
                )
            else:
                self.encoder_downsamples.append(nn.Identity())
        
        # Bottleneck
        mid_channels = base_channels * channel_multipliers[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, time_emb_dim)
        self.mid_attn = Attention(mid_channels) if use_attention else nn.Identity()
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, time_emb_dim)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        ch = mid_channels
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            ch_out = base_channels * mult
            
            # Residual blocks for this level
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            
            # First block takes skip connection
            blocks.append(ResidualBlock(ch + ch_out, ch_out, time_emb_dim))
            if use_attention and i >= len(channel_multipliers) - 2:
                attns.append(Attention(ch_out))
            else:
                attns.append(nn.Identity())
            
            # Additional blocks
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(ch_out, ch_out, time_emb_dim))
                if use_attention and i >= len(channel_multipliers) - 2:
                    attns.append(Attention(ch_out))
                else:
                    attns.append(nn.Identity())
            
            self.decoder_blocks.append(blocks)
            self.decoder_attns.append(attns)
            
            # Upsample (for all levels except the first/shallowest)
            if i > 0:
                self.decoder_upsamples.append(
                    nn.ConvTranspose2d(ch_out, ch_out, 4, stride=2, padding=1)
                )
            else:
                self.decoder_upsamples.append(nn.Identity())
            
            ch = ch_out
        # Output
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x, t, cond):
        """
        Args:
            x: noisy residual (B, C, H, W)
            t: timestep (B,)
            cond: HiCARN prediction (B, C, H, W)
        
        Returns:
            predicted noise ε (B, C, H, W)
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encode condition
        cond_emb = self.cond_encoder(cond)
        
        # Input
        h = self.input_conv(x)
        h = h + cond_emb  # Add condition early
        
        # Encoder with skip connections
        skips = []
        for blocks, attns, downsample in zip(
            self.encoder_blocks, self.encoder_attns, self.encoder_downsamples
        ):
            for block, attn in zip(blocks, attns):
                h = block(h, t_emb)
                h = attn(h)
            # Save skip connection (only the last one from this level)
            skips.append(h)
            h = downsample(h)
        
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder with skip connections
        for blocks, attns, upsample in zip(
            self.decoder_blocks, self.decoder_attns, self.decoder_upsamples
        ):
            # Concatenate skip connection (same spatial size)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            
            # Process blocks
            for block, attn in zip(blocks, attns):
                h = block(h, t_emb)
                h = attn(h)
            
            # Upsample for next level
            h = upsample(h)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h


if __name__ == "__main__":
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2
    ).to(device)
    
    # Test forward pass
    batch_size = 4
    H, W = 40, 40
    
    x = torch.randn(batch_size, 1, H, W).to(device)  # Noisy residual
    t = torch.randint(0, 1000, (batch_size,)).to(device)  # Timestep
    cond = torch.randn(batch_size, 1, H, W).to(device)  # HiCARN prediction
    
    output = model(x, t, cond)
    
    print(f"Input shape: {x.shape}")
    print(f"Condition shape: {cond.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
