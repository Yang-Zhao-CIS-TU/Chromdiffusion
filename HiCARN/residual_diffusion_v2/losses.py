"""
Improved Loss Functions for Residual Diffusion

Key improvements:
1. Peak localization losses (Laplacian, gradient, top-k focal)
2. Multi-scale structural losses
3. Proper loss weighting and combination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianLoss(nn.Module):
    """
    Laplacian loss for sharper peaks
    
    Encourages high-frequency consistency, making peaks more acute
    """
    def __init__(self):
        super().__init__()
        # Laplacian kernel
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel)
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
        """
        # Convert kernel to same dtype and device as input
        kernel = self.kernel.to(pred.dtype).to(pred.device)
        
        # Apply Laplacian
        pred_lap = F.conv2d(pred, kernel, padding=1)
        target_lap = F.conv2d(target, kernel, padding=1)
        
        # L1 loss on Laplacian
        loss = F.l1_loss(pred_lap, target_lap)
        return loss


class GradientLoss(nn.Module):
    """
    Gradient consistency loss (Sobel)
    
    Helps with edge sharpness and peak localization
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
        """
        # Convert kernels to same dtype and device as input
        sobel_x = self.sobel_x.to(pred.dtype).to(pred.device)
        sobel_y = self.sobel_y.to(pred.dtype).to(pred.device)
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        # L1 loss on gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        loss = (loss_x + loss_y) / 2
        return loss


class TopKFocalLoss(nn.Module):
    """
    Top-K Focal Loss for peak regions
    
    Focuses on high-value regions (loops/peaks), giving them higher weight
    """
    def __init__(self, top_k_ratio=0.1, gamma=2.0):
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
        """
        B, C, H, W = target.shape
        
        # Flatten spatial dimensions
        target_flat = target.view(B, C, -1)
        pred_flat = pred.view(B, C, -1)
        
        # Get top-k mask for each sample
        k = int(H * W * self.top_k_ratio)
        top_k_values, top_k_indices = torch.topk(target_flat, k, dim=2)
        
        # Create mask
        mask = torch.zeros_like(target_flat)
        mask.scatter_(2, top_k_indices, 1.0)
        mask = mask.view(B, C, H, W)
        
        # Compute error
        error = torch.abs(pred - target)
        
        # Focal weight (higher for larger errors)
        focal_weight = (error / (error.max() + 1e-8)) ** self.gamma
        
        # Apply mask and focal weight
        weighted_error = error * mask * (1 + focal_weight)
        
        # Average over top-k regions
        loss = weighted_error.sum() / (mask.sum() + 1e-8)
        
        return loss


class RankingLoss(nn.Module):
    """
    Ranking consistency loss
    
    Ensures that if target[i] > target[j], then pred[i] > pred[j]
    Helps with peak ordering and relative intensities
    """
    def __init__(self, margin=0.1, num_pairs=1000):
        super().__init__()
        self.margin = margin
        self.num_pairs = num_pairs
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
        """
        B, C, H, W = target.shape
        
        # Flatten
        target_flat = target.view(B, -1)
        pred_flat = pred.view(B, -1)
        
        total_loss = 0
        for b in range(B):
            # Sample random pairs
            num_elements = target_flat.shape[1]
            idx_i = torch.randint(0, num_elements, (self.num_pairs,), device=target.device)
            idx_j = torch.randint(0, num_elements, (self.num_pairs,), device=target.device)
            
            # Get values
            target_i = target_flat[b, idx_i]
            target_j = target_flat[b, idx_j]
            pred_i = pred_flat[b, idx_i]
            pred_j = pred_flat[b, idx_j]
            
            # Ranking loss: if target_i > target_j, then pred_i should > pred_j by margin
            ranking_diff = (target_i - target_j).sign()
            pred_diff = pred_i - pred_j
            
            loss = F.relu(self.margin - ranking_diff * pred_diff).mean()
            total_loss += loss
        
        return total_loss / B


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for capturing both global and local structure
    
    Helps with TAD structure (coarse) and loop peaks (fine)
    """
    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
        """
        total_loss = 0
        
        for scale in self.scales:
            if scale == 1:
                # Original resolution
                loss = F.l1_loss(pred, target)
            else:
                # Downsampled
                pred_down = F.avg_pool2d(pred, scale)
                target_down = F.avg_pool2d(target, scale)
                loss = F.l1_loss(pred_down, target_down)
            
            total_loss += loss
        
        return total_loss / len(self.scales)


class CombinedResidualLoss(nn.Module):
    """
    Combined loss for residual diffusion training
    
    Combines multiple loss terms with appropriate weighting
    """
    def __init__(
        self,
        use_laplacian=True,
        use_gradient=True,
        use_topk=True,
        use_ranking=False,
        use_multiscale=True,
        lambda_laplacian=0.5,
        lambda_gradient=0.3,
        lambda_topk=1.0,
        lambda_ranking=0.1,
        lambda_multiscale=0.5
    ):
        super().__init__()
        
        # Base loss (L1 or Huber)
        self.base_loss = nn.SmoothL1Loss()  # Huber loss (robust to outliers)
        
        # Peak localization losses
        self.use_laplacian = use_laplacian
        if use_laplacian:
            self.laplacian_loss = LaplacianLoss()
            self.lambda_laplacian = lambda_laplacian
        
        self.use_gradient = use_gradient
        if use_gradient:
            self.gradient_loss = GradientLoss()
            self.lambda_gradient = lambda_gradient
        
        self.use_topk = use_topk
        if use_topk:
            self.topk_loss = TopKFocalLoss(top_k_ratio=0.1, gamma=2.0)
            self.lambda_topk = lambda_topk
        
        self.use_ranking = use_ranking
        if use_ranking:
            self.ranking_loss = RankingLoss(margin=0.1, num_pairs=1000)
            self.lambda_ranking = lambda_ranking
        
        # Multi-scale for TAD structure
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.multiscale_loss = MultiScaleLoss(scales=[1, 2, 4])
            self.lambda_multiscale = lambda_multiscale
    
    def forward(self, pred, target, return_components=False):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
            return_components: If True, return dict with all loss components
        
        Returns:
            loss: Total loss
            loss_dict: (Optional) Dict with individual loss components
        """
        # Base reconstruction loss
        loss_base = self.base_loss(pred, target)
        total_loss = loss_base
        
        loss_dict = {'base': loss_base.item()}
        
        # Peak localization losses
        if self.use_laplacian:
            loss_lap = self.laplacian_loss(pred, target)
            total_loss = total_loss + self.lambda_laplacian * loss_lap
            loss_dict['laplacian'] = loss_lap.item()
        
        if self.use_gradient:
            loss_grad = self.gradient_loss(pred, target)
            total_loss = total_loss + self.lambda_gradient * loss_grad
            loss_dict['gradient'] = loss_grad.item()
        
        if self.use_topk:
            loss_tk = self.topk_loss(pred, target)
            total_loss = total_loss + self.lambda_topk * loss_tk
            loss_dict['topk'] = loss_tk.item()
        
        if self.use_ranking:
            loss_rank = self.ranking_loss(pred, target)
            total_loss = total_loss + self.lambda_ranking * loss_rank
            loss_dict['ranking'] = loss_rank.item()
        
        # Multi-scale for TAD structure
        if self.use_multiscale:
            loss_ms = self.multiscale_loss(pred, target)
            total_loss = total_loss + self.lambda_multiscale * loss_ms
            loss_dict['multiscale'] = loss_ms.item()
        
        loss_dict['total'] = total_loss.item()
        
        if return_components:
            return total_loss, loss_dict
        else:
            return total_loss


def get_loss_weights(strategy='balanced'):
    """
    Get loss weight configurations
    
    Returns:
        dict: Loss weight configuration
    """
    if strategy == 'balanced':
        return {
            'lambda_laplacian': 0.5,
            'lambda_gradient': 0.3,
            'lambda_topk': 1.0,
            'lambda_ranking': 0.1,
            'lambda_multiscale': 0.5
        }
    elif strategy == 'peak_focused':
        # Emphasize peak localization
        return {
            'lambda_laplacian': 1.0,
            'lambda_gradient': 0.5,
            'lambda_topk': 2.0,
            'lambda_ranking': 0.3,
            'lambda_multiscale': 0.3
        }
    elif strategy == 'tad_focused':
        # Emphasize large-scale structure
        return {
            'lambda_laplacian': 0.2,
            'lambda_gradient': 0.2,
            'lambda_topk': 0.5,
            'lambda_ranking': 0.1,
            'lambda_multiscale': 1.0
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
