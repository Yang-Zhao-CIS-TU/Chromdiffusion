"""
Enhanced Loss Functions for Residual Diffusion

NEW ADDITIONS (Based on recommendations):
1. Frequency-separated reconstruction (blur/low-pass for recon, diff handles peaks)
2. Localization losses (Heatmap KL + Gradient consistency)
3. Better peak positioning (not just peak height)

Key improvements:
- Recon only constrains LOW-FREQUENCY/STRUCTURE
- Diffusion handles HIGH-FREQUENCY/PEAKS
- Heatmap KL ensures peaks are in correct positions
- Gradient consistency prevents peaks from spreading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianBlur(nn.Module):
    """Gaussian blur for low-pass filtering in reconstruction loss"""
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('kernel', kernel)
    
    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create 2D Gaussian kernel"""
        x = torch.arange(kernel_size) - kernel_size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        kernel_2d = gaussian_1d.unsqueeze(1) * gaussian_1d.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Shape: [1, 1, kernel_size, kernel_size]
        return kernel_2d.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        """Apply Gaussian blur"""
        # x: [B, C, H, W]
        padding = self.kernel_size // 2
        
        # Convert kernel to same dtype and device as input
        kernel = self.kernel.to(x.dtype).to(x.device)
        
        # Apply blur per channel
        blurred = F.conv2d(x, kernel, padding=padding, groups=1)
        
        return blurred


class HeatmapKLLoss(nn.Module):
    """
    Heatmap KL divergence loss for peak localization
    
    Treats contact maps as probability distributions and encourages
    peaks to be in the correct positions
    """
    def __init__(self, temperature=1.0, use_upper_triangle=True, min_diag=2):
        super().__init__()
        self.temperature = temperature
        self.use_upper_triangle = use_upper_triangle
        self.min_diag = min_diag
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] predicted contact map
            target: [B, 1, H, W] ground truth contact map
        """
        B, C, H, W = pred.shape
        
        # Create mask for upper triangle (excluding near-diagonal)
        if self.use_upper_triangle:
            mask = torch.ones(H, W, device=pred.device)
            for i in range(H):
                for j in range(W):
                    if abs(i - j) < self.min_diag:
                        mask[i, j] = 0
                    elif i > j:  # Lower triangle
                        mask[i, j] = 0
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        else:
            mask = torch.ones_like(pred)
        
        # Apply mask
        pred_masked = pred * mask
        target_masked = target * mask
        
        # Flatten spatial dimensions
        pred_flat = pred_masked.view(B, -1)  # [B, H*W]
        target_flat = target_masked.view(B, -1)  # [B, H*W]
        
        # Convert to probability distributions using softmax
        pred_prob = F.softmax(pred_flat / self.temperature, dim=1)
        target_prob = F.softmax(target_flat / self.temperature, dim=1)
        
        # KL divergence: KL(target || pred)
        # This encourages pred distribution to match target distribution
        kl_loss = F.kl_div(
            pred_prob.log(),
            target_prob,
            reduction='batchmean'
        )
        
        return kl_loss


class GradientConsistencyLoss(nn.Module):
    """
    Gradient consistency loss for sharp peak localization
    
    Encourages gradient patterns to match, preventing peaks from
    becoming blurry or spreading out
    """
    def __init__(self):
        super().__init__()
        
        # Sobel kernels for x and y gradients
        sobel_x = torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32)
        
        sobel_y = torch.tensor([
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        ], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0))
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] predicted contact map
            target: [B, 1, H, W] ground truth contact map
        """
        # Convert kernels to match input dtype and device
        sobel_x = self.sobel_x.to(pred.dtype).to(pred.device)
        sobel_y = self.sobel_y.to(pred.dtype).to(pred.device)
        
        # Compute gradients for prediction
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        
        # Compute gradients for target
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        # L1 loss on gradient differences
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return (loss_x + loss_y) / 2


class FrequencySeparatedReconLoss(nn.Module):
    """
    Frequency-separated reconstruction loss
    
    Recon loss only constrains LOW-FREQUENCY/STRUCTURE (via blur)
    Diffusion handles HIGH-FREQUENCY/PEAKS
    
    This improves TAD/structure without hurting loops
    """
    def __init__(self, blur_kernel_size=5, blur_sigma=1.5):
        super().__init__()
        self.blur = GaussianBlur(blur_kernel_size, blur_sigma)
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] predicted contact map
            target: [B, 1, H, W] ground truth contact map
        """
        # Blur both pred and target to get low-frequency components
        pred_low = self.blur(pred)
        target_low = self.blur(target)
        
        # MSE on low-frequency only
        loss = F.mse_loss(pred_low, target_low)
        
        return loss


class EnhancedCombinedLoss(nn.Module):
    """
    Enhanced combined loss with frequency separation and localization
    
    Components:
    1. Base MSE (low weight)
    2. Frequency-separated recon (blur/low-pass)
    3. Heatmap KL (peak positioning)
    4. Gradient consistency (sharp peaks)
    5. Optional: Laplacian, TopK for extra peak focus
    """
    def __init__(
        self,
        # Frequency separation
        use_freq_separated=True,
        lambda_freq_separated=1.0,
        blur_kernel_size=5,
        blur_sigma=1.5,
        # Localization losses
        use_heatmap_kl=True,
        lambda_heatmap_kl=2.0,
        kl_temperature=1.0,
        use_gradient_consistency=True,
        lambda_gradient_consistency=1.0,
        # Legacy peak losses (optional)
        use_laplacian=False,
        lambda_laplacian=0.5,
        use_topk=False,
        lambda_topk=1.0,
        topk_ratio=0.1,
        # Base loss
        lambda_base=0.1
    ):
        super().__init__()
        
        # Flags
        self.use_freq_separated = use_freq_separated
        self.use_heatmap_kl = use_heatmap_kl
        self.use_gradient_consistency = use_gradient_consistency
        self.use_laplacian = use_laplacian
        self.use_topk = use_topk
        
        # Weights
        self.lambda_freq_separated = lambda_freq_separated
        self.lambda_heatmap_kl = lambda_heatmap_kl
        self.lambda_gradient_consistency = lambda_gradient_consistency
        self.lambda_laplacian = lambda_laplacian
        self.lambda_topk = lambda_topk
        self.lambda_base = lambda_base
        
        self.topk_ratio = topk_ratio
        
        # Loss modules
        if use_freq_separated:
            self.freq_separated_loss = FrequencySeparatedReconLoss(
                blur_kernel_size, blur_sigma
            )
        
        if use_heatmap_kl:
            self.heatmap_kl_loss = HeatmapKLLoss(
                temperature=kl_temperature,
                use_upper_triangle=True,
                min_diag=2
            )
        
        if use_gradient_consistency:
            self.gradient_consistency_loss = GradientConsistencyLoss()
        
        if use_laplacian:
            # Laplacian kernel for edge detection
            laplacian = torch.tensor([
                [[0, -1, 0],
                 [-1, 4, -1],
                 [0, -1, 0]]
            ], dtype=torch.float32).unsqueeze(0)
            self.register_buffer('laplacian_kernel', laplacian)
    
    def forward(self, pred, target, return_components=False):
        """
        Args:
            pred: [B, 1, H, W] predicted contact map
            target: [B, 1, H, W] ground truth contact map
            return_components: If True, return dict of loss components
        """
        losses = {}
        
        # 1. Base MSE (low weight, just for stability)
        base_loss = F.mse_loss(pred, target)
        losses['base'] = base_loss.item()
        total_loss = self.lambda_base * base_loss
        
        # 2. Frequency-separated reconstruction (LOW-PASS only)
        if self.use_freq_separated:
            freq_loss = self.freq_separated_loss(pred, target)
            losses['freq_separated'] = freq_loss.item()
            total_loss = total_loss + self.lambda_freq_separated * freq_loss
        
        # 3. Heatmap KL (peak positioning)
        if self.use_heatmap_kl:
            kl_loss = self.heatmap_kl_loss(pred, target)
            losses['heatmap_kl'] = kl_loss.item()
            total_loss = total_loss + self.lambda_heatmap_kl * kl_loss
        
        # 4. Gradient consistency (sharp peaks)
        if self.use_gradient_consistency:
            grad_loss = self.gradient_consistency_loss(pred, target)
            losses['gradient_consistency'] = grad_loss.item()
            total_loss = total_loss + self.lambda_gradient_consistency * grad_loss
        
        # 5. Optional: Laplacian (edge emphasis)
        if self.use_laplacian:
            kernel = self.laplacian_kernel.to(pred.dtype).to(pred.device)
            pred_lap = F.conv2d(pred, kernel, padding=1)
            target_lap = F.conv2d(target, kernel, padding=1)
            lap_loss = F.mse_loss(pred_lap, target_lap)
            losses['laplacian'] = lap_loss.item()
            total_loss = total_loss + self.lambda_laplacian * lap_loss
        
        # 6. Optional: Top-K loss (peak values)
        if self.use_topk:
            B, C, H, W = pred.shape
            k = max(1, int(H * W * self.topk_ratio))
            
            pred_flat = pred.view(B, -1)
            target_flat = target.view(B, -1)
            
            pred_topk = torch.topk(pred_flat, k, dim=1)[0]
            target_topk = torch.topk(target_flat, k, dim=1)[0]
            
            topk_loss = F.mse_loss(pred_topk, target_topk)
            losses['topk'] = topk_loss.item()
            total_loss = total_loss + self.lambda_topk * topk_loss
        
        losses['total'] = total_loss.item()
        
        if return_components:
            return total_loss, losses
        return total_loss


def get_enhanced_loss_config(strategy='localization_focused'):
    """
    Get loss configuration for different strategies
    
    Strategies:
    - 'localization_focused': Focus on peak positioning (recommended)
    - 'balanced': Balance between structure and localization
    - 'structure_focused': More emphasis on overall structure
    """
    if strategy == 'localization_focused':
        # RECOMMENDED: Focus on getting peak positions right
        return {
            'use_freq_separated': True,
            'lambda_freq_separated': 1.0,  # Low-freq structure
            'blur_kernel_size': 5,
            'blur_sigma': 1.5,
            
            'use_heatmap_kl': True,
            'lambda_heatmap_kl': 3.0,  # HIGH: Peak positioning
            'kl_temperature': 1.0,
            
            'use_gradient_consistency': True,
            'lambda_gradient_consistency': 2.0,  # HIGH: Sharp peaks
            
            'use_laplacian': False,  # Not needed with gradient consistency
            'use_topk': False,  # Not needed with heatmap KL
            
            'lambda_base': 0.1  # Low base MSE
        }
    
    elif strategy == 'balanced':
        return {
            'use_freq_separated': True,
            'lambda_freq_separated': 1.5,
            'blur_kernel_size': 5,
            'blur_sigma': 1.5,
            
            'use_heatmap_kl': True,
            'lambda_heatmap_kl': 2.0,
            'kl_temperature': 1.0,
            
            'use_gradient_consistency': True,
            'lambda_gradient_consistency': 1.0,
            
            'use_laplacian': True,
            'lambda_laplacian': 0.5,
            
            'use_topk': False,
            
            'lambda_base': 0.2
        }
    
    elif strategy == 'structure_focused':
        return {
            'use_freq_separated': True,
            'lambda_freq_separated': 2.0,  # HIGH: Structure
            'blur_kernel_size': 7,
            'blur_sigma': 2.0,
            
            'use_heatmap_kl': True,
            'lambda_heatmap_kl': 1.0,  # LOWER
            'kl_temperature': 1.5,
            
            'use_gradient_consistency': True,
            'lambda_gradient_consistency': 0.5,
            
            'use_laplacian': True,
            'lambda_laplacian': 1.0,
            
            'use_topk': True,
            'lambda_topk': 0.5,
            
            'lambda_base': 0.3
        }
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == '__main__':
    # Test the enhanced loss
    B, C, H, W = 4, 1, 40, 40
    
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)
    
    # Create loss with localization focus
    config = get_enhanced_loss_config('localization_focused')
    criterion = EnhancedCombinedLoss(**config)
    
    loss, components = criterion(pred, target, return_components=True)
    
    print("Enhanced Loss Components:")
    for key, value in components.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nTotal loss: {loss.item():.4f}")
