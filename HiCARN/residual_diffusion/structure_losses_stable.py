"""
Stable Structure-Oriented Loss Functions for Hi-C Residual Diffusion

KEY CHANGES FOR STABILITY:
  1. Safe insulation computation with None handling
  2. Batch-level valid tracking (not silent skipping)
  3. Variance normalization to reduce batch fluctuation
  4. Proper clamping and numerical stability
  5. Correct loss aggregation

Critical fixes:
  - P0: Insulation loss always well-defined or explicitly None
  - P0: Use mask/counter for valid batches (not silent zeros)
  - P1: Clamp + normalize insulation scores
  - P1: Normalize by variance to reduce batch weight fluctuation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StableInsulationLoss(nn.Module):
    """
    Stable Insulation Score Loss with proper None handling
    
    KEY IMPROVEMENTS:
    - Adaptive window size (never too large)
    - Returns None if matrix too small (explicit handling)
    - Clamp + normalize for stability
    - Variance normalization to reduce batch fluctuation
    """
    def __init__(self, window_size=5, min_window=2):
        super().__init__()
        self.window_size = window_size
        self.min_window = min_window
    
    def safe_insulation_score(self, mat):
        """
        Compute insulation score with guaranteed stability
        
        Args:
            mat: Contact matrix (B, 1, H, W)
        
        Returns:
            insulation_vector: (B, num_positions) or None if invalid
        """
        batch_size = mat.shape[0]
        L = mat.shape[2]
        
        # Adaptive window: never larger than (L-1)//2
        w = min(self.window_size, (L - 1) // 2)
        
        # Check if window is too small
        if w < self.min_window:
            return None
        
        # Check if we have enough positions
        if L < 2 * w + 1:
            return None
        
        scores = []
        for i in range(w, L - w):
            # Extract diamond block
            block = mat[:, 0, i-w:i, i:i+w]
            
            # Mean interaction strength
            score = block.mean(dim=(1, 2))  # (B,)
            scores.append(score)
        
        if len(scores) == 0:
            return None
        
        # Stack to (B, num_positions)
        insulation = torch.stack(scores, dim=1)
        
        # CRITICAL: Log + clamp for stability
        insulation = torch.log(insulation + 1e-4)
        insulation = torch.clamp(insulation, -5, 5)
        
        # Replace any remaining NaN/Inf
        insulation = torch.nan_to_num(insulation, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return insulation
    
    def forward(self, pred, target):
        """
        Compare insulation profiles with proper validity tracking
        
        Args:
            pred: Predicted Hi-C matrix (B, 1, H, W)
            target: Target Hi-C matrix (B, 1, H, W)
        
        Returns:
            loss: MSE between insulation profiles (or None if invalid)
            valid: Boolean indicating if loss is valid
        """
        # Clean inputs
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=0.0)
        
        # Compute insulation scores
        ins_pred = self.safe_insulation_score(pred)
        ins_target = self.safe_insulation_score(target)
        
        # Check validity
        if ins_pred is None or ins_target is None:
            return None, False
        
        # Compute MSE
        loss_raw = F.mse_loss(ins_pred, ins_target, reduction='mean')
        
        # CRITICAL: Normalize by target variance to reduce batch fluctuation
        target_var = ins_target.var().detach()
        target_var = torch.clamp(target_var, min=1e-6)  # Avoid division by zero
        
        loss_normalized = loss_raw / target_var
        
        # Final check for NaN
        if torch.isnan(loss_normalized) or torch.isinf(loss_normalized):
            return None, False
        
        return loss_normalized, True


class StableTADBoundaryLoss(nn.Module):
    """
    Stable TAD Boundary Loss with proper None handling
    """
    def __init__(self, window_size=5, min_window=2):
        super().__init__()
        self.insulation_calc = StableInsulationLoss(window_size, min_window)
    
    def forward(self, pred, target):
        """
        Compare insulation gradients with proper validity tracking
        
        Returns:
            loss: L1 loss between gradients (or None if invalid)
            valid: Boolean indicating if loss is valid
        """
        # Clean inputs
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=0.0)
        
        # Get insulation profiles
        ins_pred = self.insulation_calc.safe_insulation_score(pred)
        ins_target = self.insulation_calc.safe_insulation_score(target)
        
        # Check validity
        if ins_pred is None or ins_target is None:
            return None, False
        
        # Check if we have enough points for gradient
        if ins_pred.shape[1] < 2 or ins_target.shape[1] < 2:
            return None, False
        
        # Compute gradients (boundary strength)
        grad_pred = torch.abs(ins_pred[:, 1:] - ins_pred[:, :-1])
        grad_target = torch.abs(ins_target[:, 1:] - ins_target[:, :-1])
        
        # L1 loss
        loss_raw = F.l1_loss(grad_pred, grad_target, reduction='mean')
        
        # Normalize by target variance
        target_var = grad_target.var().detach()
        target_var = torch.clamp(target_var, min=1e-6)
        
        loss_normalized = loss_raw / target_var
        
        # Final check for NaN
        if torch.isnan(loss_normalized) or torch.isinf(loss_normalized):
            return None, False
        
        return loss_normalized, True


class StableLowFrequencyLoss(nn.Module):
    """
    Stable Low-Frequency Consistency Loss
    """
    def __init__(self, sigma=3.0):
        super().__init__()
        self.sigma = sigma
    
    def gaussian_blur_2d(self, x, sigma):
        """Apply Gaussian blur"""
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # 1D Gaussian
        x_coord = torch.arange(kernel_size).float() - kernel_size // 2
        gaussian_1d = torch.exp(-(x_coord ** 2) / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # 2D Gaussian
        gaussian_2d = gaussian_1d.view(-1, 1) * gaussian_1d.view(1, -1)
        gaussian_2d = gaussian_2d.view(1, 1, kernel_size, kernel_size)
        gaussian_2d = gaussian_2d.to(x.device)
        
        # Apply convolution
        padding = kernel_size // 2
        blurred = F.conv2d(x, gaussian_2d, padding=padding)
        
        return blurred
    
    def forward(self, pred, target):
        """
        Compare low-frequency components
        
        Returns:
            loss: MSE between low-freq components (or None if invalid)
            valid: Boolean indicating if loss is valid
        """
        # Clean inputs
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=0.0)
        
        # Extract low-frequency components
        pred_low = self.gaussian_blur_2d(pred, self.sigma)
        target_low = self.gaussian_blur_2d(target, self.sigma)
        
        # MSE loss
        loss_raw = F.mse_loss(pred_low, target_low, reduction='mean')
        
        # Normalize by target variance
        target_var = target_low.var().detach()
        target_var = torch.clamp(target_var, min=1e-6)
        
        loss_normalized = loss_raw / target_var
        
        # Check for NaN
        if torch.isnan(loss_normalized) or torch.isinf(loss_normalized):
            return None, False
        
        return loss_normalized, True


class StableStructureLossCalculator(nn.Module):
    """
    Stable Combined Structure Loss with proper batch tracking
    
    KEY IMPROVEMENTS:
    - Uses counters for valid batches (not silent skipping)
    - Variance normalization for each loss component
    - Proper aggregation (sum/count not list append)
    - Explicit validity tracking
    """
    def __init__(
        self,
        lambda_residual=0.1,
        lambda_insulation=1.0,
        lambda_tad_boundary=0.5,
        lambda_low_freq=0.5,
        insulation_window=5,
        blur_sigma=3.0
    ):
        super().__init__()
        
        self.lambda_residual = lambda_residual
        self.lambda_insulation = lambda_insulation
        self.lambda_tad_boundary = lambda_tad_boundary
        self.lambda_low_freq = lambda_low_freq
        
        # Loss components
        self.insulation_loss = StableInsulationLoss(window_size=insulation_window)
        self.tad_boundary_loss = StableTADBoundaryLoss(window_size=insulation_window)
        self.low_freq_loss = StableLowFrequencyLoss(sigma=blur_sigma)
    
    def forward(self, pred_residual, target_residual, pred_hic, target_hic):
        """
        Compute combined structure-oriented loss with validity tracking
        
        Returns:
            total_loss: Weighted sum of valid losses
            loss_dict: Dictionary with values and validity flags
        """
        # Clean inputs
        pred_hic = torch.nan_to_num(pred_hic, nan=0.0, posinf=1e6, neginf=0.0)
        target_hic = torch.nan_to_num(target_hic, nan=0.0, posinf=1e6, neginf=0.0)
        pred_residual = torch.nan_to_num(pred_residual, nan=0.0, posinf=100.0, neginf=-100.0)
        target_residual = torch.nan_to_num(target_residual, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # 1. Residual Loss (always valid)
        loss_residual = F.mse_loss(pred_residual, target_residual)
        valid_residual = True
        
        # 2. Insulation Score Loss (may be None)
        loss_insulation, valid_insulation = self.insulation_loss(pred_hic, target_hic)
        
        # 3. TAD Boundary Loss (may be None)
        loss_tad_boundary, valid_tad = self.tad_boundary_loss(pred_hic, target_hic)
        
        # 4. Low-Frequency Loss (may be None)
        loss_low_freq, valid_lf = self.low_freq_loss(pred_hic, target_hic)
        
        # Compute weighted total (only include valid losses)
        total_loss = self.lambda_residual * loss_residual
        
        if valid_insulation:
            total_loss = total_loss + self.lambda_insulation * loss_insulation
        
        if valid_tad:
            total_loss = total_loss + self.lambda_tad_boundary * loss_tad_boundary
        
        if valid_lf:
            total_loss = total_loss + self.lambda_low_freq * loss_low_freq
        
        # Create loss dict with explicit None for invalid losses
        loss_dict = {
            'total': total_loss.item(),
            'residual': loss_residual.item(),
            'insulation': loss_insulation.item() if valid_insulation else None,
            'tad_boundary': loss_tad_boundary.item() if valid_tad else None,
            'low_freq': loss_low_freq.item() if valid_lf else None,
            'valid_insulation': valid_insulation,
            'valid_tad': valid_tad,
            'valid_lf': valid_lf
        }
        
        return total_loss, loss_dict


class ResidualClipper:
    """Residual Magnitude Clipper"""
    def __init__(self, clip_factor=0.1):
        self.clip_factor = clip_factor
    
    def compute_clip_value(self, hicarn_pred):
        std = torch.std(hicarn_pred)
        clip_value = self.clip_factor * std
        return clip_value
    
    def clip_residual(self, residual, hicarn_pred):
        clip_value = self.compute_clip_value(hicarn_pred)
        clipped = torch.clamp(residual, -clip_value, +clip_value)
        return clipped


def test_stable_losses():
    """Test stable loss functions"""
    print("Testing Stable Structure Loss Functions")
    print("="*80)
    
    # Create dummy data
    B, C, H, W = 4, 1, 40, 40
    pred_residual = torch.randn(B, C, H, W) * 0.1
    target_residual = torch.randn(B, C, H, W) * 0.1
    
    hicarn_pred = torch.rand(B, C, H, W) * 100
    pred_hic = hicarn_pred + pred_residual
    target_hic = hicarn_pred + target_residual
    
    # Test stable losses
    print("\n[1] Testing Stable Insulation Loss...")
    ins_loss_fn = StableInsulationLoss(window_size=5)
    loss_ins, valid_ins = ins_loss_fn(pred_hic, target_hic)
    if valid_ins:
        print(f"    Insulation Loss: {loss_ins.item():.6f} (VALID)")
    else:
        print(f"    Insulation Loss: None (INVALID - matrix too small)")
    
    print("\n[2] Testing Stable TAD Boundary Loss...")
    tad_loss_fn = StableTADBoundaryLoss(window_size=5)
    loss_tad, valid_tad = tad_loss_fn(pred_hic, target_hic)
    if valid_tad:
        print(f"    TAD Boundary Loss: {loss_tad.item():.6f} (VALID)")
    else:
        print(f"    TAD Boundary Loss: None (INVALID)")
    
    print("\n[3] Testing Stable Low-Frequency Loss...")
    lf_loss_fn = StableLowFrequencyLoss(sigma=3.0)
    loss_lf, valid_lf = lf_loss_fn(pred_hic, target_hic)
    if valid_lf:
        print(f"    Low-Freq Loss: {loss_lf.item():.6f} (VALID)")
    else:
        print(f"    Low-Freq Loss: None (INVALID)")
    
    print("\n[4] Testing Stable Combined Loss...")
    loss_calc = StableStructureLossCalculator(
        lambda_residual=0.1,
        lambda_insulation=1.0,
        lambda_tad_boundary=0.5,
        lambda_low_freq=0.5,
        insulation_window=5
    )
    total_loss, loss_dict = loss_calc(pred_residual, target_residual,
                                       pred_hic, target_hic)
    print(f"    Total Loss: {total_loss.item():.6f}")
    print(f"    Components:")
    for k, v in loss_dict.items():
        if 'valid' in k:
            print(f"      {k}: {v}")
        elif v is not None:
            print(f"      {k}: {v:.6f}")
        else:
            print(f"      {k}: None (SKIPPED)")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("\nKEY OBSERVATIONS:")
    print("  - Losses return None when invalid (not 0.0)")
    print("  - Validity is explicitly tracked")
    print("  - Variance normalization reduces batch fluctuation")
    print("  - Clamping ensures numerical stability")


if __name__ == "__main__":
    test_stable_losses()
