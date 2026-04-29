"""
Improved Residual Diffusion Model

Key improvements:
1. Stronger conditioning: concat + FiLM (Adaptive Group Norm)
2. Multi-scale conditioning injection
3. Support for v-parameterization
4. Better architecture for peak localization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive Group Norm for FiLM conditioning
    Modulates features based on condition
    """
    def __init__(self, num_groups, num_channels, cond_channels):
        super().__init__()
        self.num_groups = num_groups
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        
        # Generate scale and shift from condition
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_channels, num_channels * 2)
        )
    
    def forward(self, x, cond):
        """
        Args:
            x: [B, C, H, W]
            cond: [B, cond_channels]
        """
        # Normalize
        x_norm = self.norm(x)
        
        # Generate scale and shift
        params = self.mlp(cond)
        scale, shift = params.chunk(2, dim=1)
        
        # Apply FiLM
        out = x_norm * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return out


class ResNetBlock(nn.Module):
    """
    ResNet block with time embedding and adaptive conditioning
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_channels, dropout=0.1):
        super().__init__()
        
        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # First conv + adaptive norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(8, out_channels, cond_channels)
        
        # Second conv + adaptive norm
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = AdaptiveGroupNorm(8, out_channels, cond_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb, cond_emb):
        """
        Args:
            x: [B, C, H, W]
            time_emb: [B, time_emb_dim]
            cond_emb: [B, cond_channels]
        """
        h = self.conv1(x)
        h = self.norm1(h, cond_emb)  # FiLM conditioning
        h = F.silu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h, cond_emb)  # FiLM conditioning
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


class ImprovedResidualDiffusionUNet(nn.Module):
    """
    Improved U-Net for residual diffusion
    
    Key improvements:
    1. Condition concatenated as extra channel (stronger than add)
    2. FiLM conditioning at each level (Adaptive GroupNorm)
    3. Multi-scale condition injection
    4. Support for v-parameterization
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
        dropout=0.1,
        parameterization='eps'  # 'eps' or 'v'
    ):
        super().__init__()
        
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_levels = len(channel_mults)
        self.parameterization = parameterization
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Condition encoder (process HiCARN prediction)
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1)
        )
        
        # Flatten condition for FiLM
        self.cond_pool = nn.AdaptiveAvgPool2d(1)
        self.cond_mlp = nn.Sequential(
            nn.Linear(base_channels, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, base_channels * 2)
        )
        
        # Input convolution - concat noisy residual with condition
        # in_channels + cond_channels: both as input channels
        self.input_conv = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        
        in_ch = base_channels
        for level in range(self.num_levels):
            out_ch = base_channels * channel_mults[level]
            
            # ResNet blocks for this level
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResNetBlock(in_ch, out_ch, time_emb_dim, base_channels * 2, dropout))
                in_ch = out_ch
            self.encoder_blocks.append(blocks)
            
            # Attention for this level
            if level in attn_levels:
                attns = nn.ModuleList([SelfAttention(in_ch) for _ in range(num_res_blocks)])
            else:
                attns = nn.ModuleList([nn.Identity() for _ in range(num_res_blocks)])
            self.encoder_attns.append(attns)
            
            # Downsampling (except last level)
            if level < self.num_levels - 1:
                self.encoder_downsamples.append(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))
        
        # Middle
        self.mid_block1 = ResNetBlock(in_ch, in_ch, time_emb_dim, base_channels * 2, dropout)
        self.mid_attn = SelfAttention(in_ch)
        self.mid_block2 = ResNetBlock(in_ch, in_ch, time_emb_dim, base_channels * 2, dropout)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        for level in reversed(range(self.num_levels)):
            out_ch = base_channels * channel_mults[level]
            
            # ResNet blocks for this level
            blocks = nn.ModuleList()
            
            # First block: with skip connection
            skip_ch = base_channels * channel_mults[level]
            blocks.append(ResNetBlock(in_ch + skip_ch, out_ch, time_emb_dim, base_channels * 2, dropout))
            in_ch = out_ch
            
            # Remaining blocks
            for _ in range(num_res_blocks):
                blocks.append(ResNetBlock(in_ch, out_ch, time_emb_dim, base_channels * 2, dropout))
                in_ch = out_ch
            
            self.decoder_blocks.append(blocks)
            
            # Attention for this level
            decoder_idx = self.num_levels - 1 - level
            if decoder_idx < 2:  # Attention at deepest decoder levels
                attns = nn.ModuleList([SelfAttention(in_ch) for _ in range(num_res_blocks + 1)])
            else:
                attns = nn.ModuleList([nn.Identity() for _ in range(num_res_blocks + 1)])
            self.decoder_attns.append(attns)
            
            # Upsampling (except last level)
            if level > 0:
                self.decoder_upsamples.append(nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1))
        
        # Output
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x, timesteps, condition):
        """
        Args:
            x: Noisy residual [B, 1, H, W]
            timesteps: Timesteps [B]
            condition: HiCARN prediction [B, 1, H, W]
        
        Returns:
            model_output: Predicted noise (eps) or velocity (v)
        """
        # Time embedding
        t_emb = self.time_mlp(timesteps.float())
        
        # Encode condition (for FiLM)
        cond_feat = self.cond_encoder(condition)
        cond_pooled = self.cond_pool(cond_feat).squeeze(-1).squeeze(-1)
        cond_emb = self.cond_mlp(cond_pooled)  # [B, base_channels * 2]
        
        # Concatenate input with condition (stronger than add)
        h = torch.cat([x, condition], dim=1)
        h = self.input_conv(h)
        
        # Encoder
        skips = []
        for level in range(self.num_levels):
            for block, attn in zip(self.encoder_blocks[level], self.encoder_attns[level]):
                h = block(h, t_emb, cond_emb)
                h = attn(h)
            
            # Save skip connection
            skips.append(h)
            
            # Downsample
            if level < self.num_levels - 1:
                h = self.encoder_downsamples[level](h)
        
        # Middle
        h = self.mid_block1(h, t_emb, cond_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, cond_emb)
        
        # Decoder
        for level in range(self.num_levels):
            blocks = self.decoder_blocks[level]
            attns = self.decoder_attns[level]
            
            # First block: concatenate with skip
            h = torch.cat([h, skips.pop()], dim=1)
            h = blocks[0](h, t_emb, cond_emb)
            h = attns[0](h)
            
            # Remaining blocks
            for i in range(1, len(blocks)):
                h = blocks[i](h, t_emb, cond_emb)
                h = attns[i](h)
            
            # Upsample
            if level < self.num_levels - 1:
                h = self.decoder_upsamples[level](h)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h
