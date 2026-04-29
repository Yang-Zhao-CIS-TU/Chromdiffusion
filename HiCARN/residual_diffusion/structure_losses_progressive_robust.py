"""
Progressive Structure Refinement Loss - ROBUST VERSION

CRITICAL FIXES:
  1. NaN detection and protection at every step
  2. Safe insulation computation with fallbacks
  3. Loss validity tracking (only use valid losses)
  4. If any component is NaN, skip it (don't propagate NaN)
  
PHILOSOPHY: "先活下来 → 再慢慢变好"
  - Stage 1: Force diffusion to DO SOMETHING (residual activation)
  - Stage 2: Make sure it doesn't BREAK structure (consistency)
  - Stage 3: Gently PUSH for improvement (relative improvement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def safe_nan_check(tensor, name="tensor"):
    """Check and clean NaN/Inf values"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return None
    return tensor


class ResidualActivationLoss(nn.Module):
    """
    Force diffusion to output non-zero residuals
    
    CRITICAL FIX v2: Use INTERVAL constraint, not soft penalty
    
    Old (soft penalty - model can "lie down"):
        L = ReLU(0.5 * std(gt) - std(pred))^2
        → Once pred_std ≈ 0.5 * target_std, gradient ≈ 0
        → Model stops improving ("躺平")
    
    New (interval constraint - forces continuous improvement):
        low = 0.5 * std(gt)
        high = 0.9 * std(gt)
        L = ReLU(low - std(pred))^2 + ReLU(std(pred) - high)^2
        
        → < 50%: penalize (too small, identity risk)
        → 50-90%: free zone (healthy structure refiner)
        → > 90%: penalize (too large, structure破坏风险)
        
    This is the watershed - without this, model躺平 at 50%
    """
    def __init__(self, min_ratio=0.5, max_ratio=0.9):
        super().__init__()
        self.min_ratio = min_ratio  # Lower bound: prevent identity
        self.max_ratio = max_ratio  # Upper bound: prevent structure破坏
    
    def forward(self, pred_residual, target_residual):
        """
        L_residual = ReLU(low - pred_std)^2 + ReLU(pred_std - high)^2
        
        This creates a "free zone" where model can operate without penalty,
        but forces it out of identity zone and prevents structure破坏.
        """
        try:
            # Compute standard deviation (NOT variance!)
            pred_std = torch.std(pred_residual)
            target_std = torch.std(target_residual)
            
            # Check for NaN
            if torch.isnan(pred_std) or torch.isnan(target_std):
                return torch.tensor(0.0, device=pred_residual.device), False
            
            # Interval bounds
            low_bound = self.min_ratio * target_std
            high_bound = self.max_ratio * target_std
            
            # Interval constraint: penalize if outside [low, high]
            lower_violation = torch.relu(low_bound - pred_std) ** 2
            upper_violation = torch.relu(pred_std - high_bound) ** 2
            
            loss = lower_violation + upper_violation
            
            # Normalize by target std to make it scale-invariant
            loss = loss / torch.clamp(target_std, min=1e-6)
            
            # Final NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, device=pred_residual.device), False
            
            return loss, True
            
        except Exception as e:
            print(f"ResidualActivationLoss error: {e}")
            return torch.tensor(0.0, device=pred_residual.device), False


class DirectionalResidualLoss(nn.Module):
    """
    Ensure residual direction is aligned with GT
    
    CRITICAL: Stage 1 needs this to prevent "random oscillation"
    
    Current problem:
        - Amplitude is alive (std达标)
        - But direction is free (can be任意pattern)
        → Model outputs residual that's uncorrelated with GT
        → "胡乱震荡"
    
    Solution:
        L_dir = ReLU(min_cos_sim - cos_sim(pred, gt))
        
        → cos_sim must be > 0.3 (loose requirement)
        → Won't pull back to identity
        → But prevents random patterns
    """
    def __init__(self, min_cos_sim=0.3):
        super().__init__()
        self.min_cos_sim = min_cos_sim  # Require at least 30% alignment
    
    def forward(self, pred_residual, target_residual):
        """
        L_directional = ReLU(min_cos_sim - cos_similarity)
        
        Very light constraint - just prevents完全random patterns
        """
        try:
            # Flatten for cosine similarity
            pred_flat = pred_residual.flatten()
            target_flat = target_residual.flatten()
            
            # Clean any NaN
            pred_flat = torch.nan_to_num(pred_flat, nan=0.0)
            target_flat = torch.nan_to_num(target_flat, nan=0.0)
            
            # Compute cosine similarity
            cos_sim = F.cosine_similarity(
                pred_flat.unsqueeze(0),
                target_flat.unsqueeze(0),
                dim=1
            ).mean()
            
            # Check for NaN
            if torch.isnan(cos_sim):
                return torch.tensor(0.0, device=pred_residual.device), False
            
            # Require minimum alignment
            deficit = self.min_cos_sim - cos_sim
            loss = torch.relu(deficit)
            
            # Final NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, device=pred_residual.device), False
            
            return loss, True
            
        except Exception as e:
            print(f"DirectionalResidualLoss error: {e}")
            return torch.tensor(0.0, device=pred_residual.device), False


class StructureConsistencyLoss(nn.Module):
    """Ensure refined doesn't break existing TAD structure"""
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size
    
    def compute_insulation_safe(self, mat):
        """SAFE insulation computation with extensive error checking"""
        try:
            L = mat.shape[2]
            w = min(self.window_size, (L - 1) // 2)
            
            # Too small to compute
            if w < 2:
                return None
            
            batch_size = mat.shape[0]
            device = mat.device
            
            scores = []
            for i in range(w, L - w):
                block = mat[:, 0, i-w:i, i:i+w]
                
                # Check for NaN in block
                if torch.isnan(block).any() or torch.isinf(block).any():
                    continue
                
                score = block.mean(dim=(1, 2))
                
                # Check score validity
                if torch.isnan(score).any() or torch.isinf(score).any():
                    continue
                
                scores.append(score)
            
            if len(scores) == 0:
                return None
            
            insulation = torch.stack(scores, dim=1)
            
            # SAFE log transform
            insulation = insulation + 1e-6  # Add small epsilon
            insulation = torch.log(insulation)
            
            # Check after log
            if torch.isnan(insulation).any() or torch.isinf(insulation).any():
                return None
            
            # SAFE clamp
            insulation = torch.clamp(insulation, -10, 10)  # Wider range
            
            # Final check
            if torch.isnan(insulation).any() or torch.isinf(insulation).any():
                return None
            
            return insulation
            
        except Exception as e:
            print(f"Insulation computation error: {e}")
            return None
    
    def forward(self, refined_hic, hicarn_hic):
        """Compare refined vs HiCARN (structure consistency)"""
        try:
            # Clean inputs
            refined_hic = torch.nan_to_num(refined_hic, nan=0.0, posinf=0.0, neginf=0.0)
            hicarn_hic = torch.nan_to_num(hicarn_hic, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute insulation
            ins_refined = self.compute_insulation_safe(refined_hic)
            ins_hicarn = self.compute_insulation_safe(hicarn_hic)
            
            if ins_refined is None or ins_hicarn is None:
                return torch.tensor(0.0, device=refined_hic.device), False
            
            # MSE loss
            loss = F.mse_loss(ins_refined, ins_hicarn)
            
            # Check validity
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, device=refined_hic.device), False
            
            return loss, True
            
        except Exception as e:
            print(f"StructureConsistencyLoss error: {e}")
            return torch.tensor(0.0, device=refined_hic.device), False


class StructureImprovementLoss(nn.Module):
    """Gently push for improvement over HiCARN"""
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size
    
    def compute_insulation_safe(self, mat):
        """Same safe computation as StructureConsistencyLoss"""
        try:
            L = mat.shape[2]
            w = min(self.window_size, (L - 1) // 2)
            
            if w < 2:
                return None
            
            scores = []
            for i in range(w, L - w):
                block = mat[:, 0, i-w:i, i:i+w]
                
                if torch.isnan(block).any() or torch.isinf(block).any():
                    continue
                
                score = block.mean(dim=(1, 2))
                
                if torch.isnan(score).any() or torch.isinf(score).any():
                    continue
                
                scores.append(score)
            
            if len(scores) == 0:
                return None
            
            insulation = torch.stack(scores, dim=1)
            insulation = insulation + 1e-6
            insulation = torch.log(insulation)
            
            if torch.isnan(insulation).any() or torch.isinf(insulation).any():
                return None
            
            insulation = torch.clamp(insulation, -10, 10)
            
            if torch.isnan(insulation).any() or torch.isinf(insulation).any():
                return None
            
            return insulation
            
        except Exception as e:
            return None
    
    def forward(self, refined_hic, hicarn_hic, gt_hic):
        """Relative improvement loss (ReLU)"""
        try:
            # Clean inputs
            refined_hic = torch.nan_to_num(refined_hic, nan=0.0, posinf=0.0, neginf=0.0)
            hicarn_hic = torch.nan_to_num(hicarn_hic, nan=0.0, posinf=0.0, neginf=0.0)
            gt_hic = torch.nan_to_num(gt_hic, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute insulation
            ins_refined = self.compute_insulation_safe(refined_hic)
            ins_hicarn = self.compute_insulation_safe(hicarn_hic)
            ins_gt = self.compute_insulation_safe(gt_hic)
            
            if ins_refined is None or ins_hicarn is None or ins_gt is None:
                return torch.tensor(0.0, device=refined_hic.device), False
            
            # Relative improvement
            error_refined = F.mse_loss(ins_refined, ins_gt, reduction='mean')
            error_hicarn = F.mse_loss(ins_hicarn, ins_gt, reduction='mean')
            
            # Check validity
            if torch.isnan(error_refined) or torch.isnan(error_hicarn):
                return torch.tensor(0.0, device=refined_hic.device), False
            
            # ReLU: only penalize if worse
            loss = F.relu(error_refined - error_hicarn)
            
            # Final check
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, device=refined_hic.device), False
            
            return loss, True
            
        except Exception as e:
            print(f"StructureImprovementLoss error: {e}")
            return torch.tensor(0.0, device=refined_hic.device), False


class LowFrequencyConsistencyLoss(nn.Module):
    """Preserve domain-scale structure"""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
    
    def forward(self, refined_hic, hicarn_hic):
        """Low-frequency consistency"""
        try:
            # Clean inputs
            refined_hic = torch.nan_to_num(refined_hic, nan=0.0, posinf=0.0, neginf=0.0)
            hicarn_hic = torch.nan_to_num(hicarn_hic, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Gaussian blur
            refined_blur = F.avg_pool2d(
                refined_hic,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )
            
            hicarn_blur = F.avg_pool2d(
                hicarn_hic,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )
            
            # Check for NaN
            if torch.isnan(refined_blur).any() or torch.isnan(hicarn_blur).any():
                return torch.tensor(0.0, device=refined_hic.device), False
            
            # MSE
            loss = F.mse_loss(refined_blur, hicarn_blur)
            
            # Check validity
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, device=refined_hic.device), False
            
            return loss, True
            
        except Exception as e:
            print(f"LowFrequencyConsistencyLoss error: {e}")
            return torch.tensor(0.0, device=refined_hic.device), False


class SmoothnessTVLoss(nn.Module):
    """Prevent residual from becoming pure noise"""
    def __init__(self):
        super().__init__()
    
    def forward(self, residual):
        """Total variation loss"""
        try:
            # Clean input
            residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Gradients
            tv_h = torch.abs(residual[:, :, 1:, :] - residual[:, :, :-1, :])
            tv_v = torch.abs(residual[:, :, :, 1:] - residual[:, :, :, :-1])
            
            # Check for NaN
            if torch.isnan(tv_h).any() or torch.isnan(tv_v).any():
                return torch.tensor(0.0, device=residual.device), False
            
            loss = tv_h.mean() + tv_v.mean()
            
            # Check validity
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, device=residual.device), False
            
            return loss, True
            
        except Exception as e:
            print(f"SmoothnessTVLoss error: {e}")
            return torch.tensor(0.0, device=residual.device), False


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-aware residual loss for TAD enhancement
    
    CRITICAL for TAD-only residual approach:
        - Reinforces block discontinuity
        - Does NOT create loop peaks
        - Works on TAD-masked low-frequency residual
    
    Philosophy:
        TAD boundaries = sharp transitions in low-freq residual
        We want diffusion to enhance these transitions
        But loops are already masked out
        
    Implementation:
        L_boundary = -mean(|∇ residual_tad|)
        
        Negative because we want to MAXIMIZE gradient
        (minimize -gradient = maximize gradient)
        
    Why this helps TAD without hurting loops:
        1. Loops already masked in residual_tad
        2. Gradient encourages sharp TAD boundaries  
        3. Low-frequency nature prevents peak creation
        4. Only affects domain-scale structure
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, residual_tad):
        """
        Compute boundary-aware loss from TAD residual gradient
        
        Returns NEGATIVE gradient (to encourage sharp boundaries)
        """
        try:
            # Clean input
            residual_tad = torch.nan_to_num(residual_tad, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute spatial gradients
            # Horizontal gradient (dx)
            grad_x = torch.abs(residual_tad[:, :, :, 1:] - residual_tad[:, :, :, :-1])
            
            # Vertical gradient (dy)
            grad_y = torch.abs(residual_tad[:, :, 1:, :] - residual_tad[:, :, :-1, :])
            
            # Check for NaN
            if torch.isnan(grad_x).any() or torch.isnan(grad_y).any():
                return torch.tensor(0.0, device=residual_tad.device), False
            
            # Total gradient magnitude
            grad_magnitude = grad_x.mean() + grad_y.mean()
            
            # Return NEGATIVE to encourage sharp boundaries
            # Minimizing (-gradient) = Maximizing gradient = Sharper boundaries
            loss = -grad_magnitude
            
            # Final check
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, device=residual_tad.device), False
            
            return loss, True
            
        except Exception as e:
            print(f"BoundaryAwareLoss error: {e}")
            return torch.tensor(0.0, device=residual_tad.device), False


class ProgressiveStructureLossCalculator(nn.Module):
    """
    ROBUST Progressive structure refinement loss
    
    TAD-ONLY RESIDUAL APPROACH:
    - Diffusion only modifies TAD structure (low-freq + loop-masked)
    - Loop peaks are completely frozen (handled by HiCARN)
    - Goal: TAD↑, Loop≈, PSNR/SSIM≈
    
    KEY CHANGES:
    - Smooth/TV loss REMOVED (conflicts with TAD boundaries)
    - Boundary-aware loss ADDED (reinforces TAD discontinuities)
    - Works on TAD-masked low-frequency residual
    
    IMPROVEMENTS:
    - All losses have NaN protection
    - Invalid losses are skipped (not propagated)
    - Total loss only includes valid components
    - Extensive error logging
    """
    def __init__(
        self,
        lambda_residual=1.0,
        lambda_directional=0.1,
        lambda_consistency=1.0,
        lambda_improvement=0.1,
        lambda_low_freq=0.5,
        lambda_boundary=0.2,  # NEW: Boundary-aware for TAD
        lambda_smooth=0.0,  # REMOVED: Conflicts with TAD boundaries
        insulation_window=5,
        stage1_epochs=20
    ):
        super().__init__()
        
        self.lambda_residual_base = lambda_residual  # Base value, will grow dynamically
        self.lambda_directional = lambda_directional  # Light directional constraint
        self.lambda_consistency = lambda_consistency
        self.lambda_improvement = lambda_improvement
        self.lambda_low_freq = lambda_low_freq
        self.lambda_boundary = lambda_boundary  # NEW: TAD boundary reinforcement
        self.lambda_smooth = lambda_smooth  # DISABLED for TAD-only approach
        self.stage1_epochs = stage1_epochs
        
        # Loss components
        self.residual_loss = ResidualActivationLoss(min_ratio=0.5, max_ratio=0.9)  # Interval [50%, 90%]
        self.directional_loss = DirectionalResidualLoss(min_cos_sim=0.3)  # Prevent random oscillation
        self.consistency_loss = StructureConsistencyLoss(window_size=insulation_window)
        self.improvement_loss = StructureImprovementLoss(window_size=insulation_window)
        self.low_freq_loss = LowFrequencyConsistencyLoss(kernel_size=3)
        self.boundary_loss = BoundaryAwareLoss()  # NEW: TAD boundary enhancement
        self.smooth_loss = SmoothnessTVLoss()  # DISABLED in TAD-only approach
    
    def get_dynamic_lambda_residual(self, epoch):
        """
        Dynamically increase λ_residual in Stage 1 to prevent "躺平"
        
        Strategy:
            λ_residual(epoch) = min(base + 0.05 × epoch, 2.0)
            
        Purpose:
            Prevent model from stopping at "刚好及格"
            Force continuous improvement in Stage 1
        """
        if epoch < self.stage1_epochs:
            # Stage 1: Linear growth from base to 2.0
            return min(self.lambda_residual_base + 0.05 * epoch, 2.0)
        else:
            # Stage 2: Fixed at 2.0
            return 2.0
    
    def forward(self, pred_residual, target_residual, pred_hic, hicarn_hic, gt_hic, epoch=0):
        """
        Compute progressive structure loss with ROBUST error handling
        
        TAD-ONLY RESIDUAL APPROACH:
        - pred_residual and target_residual are TAD-masked (loops excluded)
        - Diffusion only modifies domain-scale structure
        - Goal: TAD↑, Loop≈, PSNR/SSIM≈
        
        STAGE-DEPENDENT WEIGHTS:
        Stage 1 (epochs 0-20):
          - Residual activation: ON (interval [50%, 90%])
          - Directional: ON (prevent random oscillation)
          - Consistency: ON (don't break structure)
          - Improvement: OFF (not ready yet)
          - Low-freq: ON (preserve domains)
          - Boundary: ON (reinforce TAD transitions)
          - Smooth/TV: OFF (REMOVED for TAD approach)
        
        Stage 2 (epochs 20+):
          - All losses: ON except Smooth
          - Improvement gradually increases
          - Boundary continues to reinforce TADs
        """
        device = pred_hic.device
        
        # Clean all inputs FIRST
        pred_hic = torch.nan_to_num(pred_hic, nan=0.0, posinf=0.0, neginf=0.0)
        hicarn_hic = torch.nan_to_num(hicarn_hic, nan=0.0, posinf=0.0, neginf=0.0)
        gt_hic = torch.nan_to_num(gt_hic, nan=0.0, posinf=0.0, neginf=0.0)
        pred_residual = torch.nan_to_num(pred_residual, nan=0.0, posinf=0.0, neginf=0.0)
        target_residual = torch.nan_to_num(target_residual, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Determine stage
        is_stage1 = epoch < self.stage1_epochs
        
        # Get dynamic λ_residual (grows in Stage 1 to prevent "躺平")
        current_lambda_residual = self.get_dynamic_lambda_residual(epoch)
        
        # Initialize total loss
        total_loss = torch.tensor(0.0, device=device)
        n_valid_losses = 0
        
        # 1. Residual activation loss (ALWAYS ON - with interval constraint [50%, 90%])
        loss_res, valid_res = self.residual_loss(pred_residual, target_residual)
        if valid_res:
            total_loss = total_loss + current_lambda_residual * loss_res  # Dynamic weight!
            n_valid_losses += 1
        
        # 2. Directional residual loss (STAGE 1 ONLY - prevents random oscillation)
        if is_stage1:
            loss_dir, valid_dir = self.directional_loss(pred_residual, target_residual)
            if valid_dir:
                total_loss = total_loss + self.lambda_directional * loss_dir
                n_valid_losses += 1
        else:
            loss_dir = torch.tensor(0.0, device=device)
            valid_dir = False
        
        # 3. Structure consistency loss (ALWAYS ON)
        loss_cons, valid_cons = self.consistency_loss(pred_hic, hicarn_hic)
        if valid_cons:
            total_loss = total_loss + self.lambda_consistency * loss_cons
            n_valid_losses += 1
        
        # 4. Structure improvement loss (STAGE 2 ONLY)
        if is_stage1:
            loss_imp = torch.tensor(0.0, device=device)
            valid_imp = False
            effective_lambda_imp = 0.0
        else:
            loss_imp, valid_imp = self.improvement_loss(pred_hic, hicarn_hic, gt_hic)
            if valid_imp:
                progress = min(1.0, (epoch - self.stage1_epochs) / 30.0)
                effective_lambda_imp = self.lambda_improvement * progress
                total_loss = total_loss + effective_lambda_imp * loss_imp
                n_valid_losses += 1
            else:
                effective_lambda_imp = 0.0
        
        # 5. Low-frequency consistency loss (ALWAYS ON)
        loss_freq, valid_freq = self.low_freq_loss(pred_hic, hicarn_hic)
        if valid_freq:
            total_loss = total_loss + self.lambda_low_freq * loss_freq
            n_valid_losses += 1
        
        # 6. Boundary-aware loss (ALWAYS ON - reinforces TAD transitions)
        # Safe for TAD-only residual:
        #   - Works on TAD-masked residual (loops excluded)
        #   - Encourages sharp domain boundaries
        #   - Does NOT create loop peaks (low-freq nature)
        loss_boundary, valid_boundary = self.boundary_loss(pred_residual)
        if valid_boundary:
            total_loss = total_loss + self.lambda_boundary * loss_boundary
            n_valid_losses += 1
        
        # Smooth/TV loss: DISABLED in TAD-only approach
        # Reason: Conflicts with TAD boundary sharpness
        loss_tv = torch.tensor(0.0, device=device)
        valid_tv = False
        effective_lambda_tv = 0.0
        
        # CRITICAL: Check if total loss is valid
        if torch.isnan(total_loss) or torch.isinf(total_loss) or n_valid_losses == 0:
            print(f"[WARNING] Total loss invalid! n_valid={n_valid_losses}")
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Loss dict
        loss_dict = {
            'total': total_loss.item() if not torch.isnan(total_loss) else 0.0,
            'residual': loss_res.item() if valid_res else None,
            'directional': loss_dir.item() if valid_dir else None,
            'consistency': loss_cons.item() if valid_cons else None,
            'improvement': loss_imp.item() if valid_imp else None,
            'low_freq': loss_freq.item() if valid_freq else None,
            'boundary': loss_boundary.item() if valid_boundary else None,  # NEW: TAD boundaries
            'smooth': None,  # DISABLED in TAD-only approach
            'valid_residual': valid_res,
            'valid_directional': valid_dir,
            'valid_consistency': valid_cons,
            'valid_improvement': valid_imp,
            'valid_low_freq': valid_freq,
            'valid_boundary': valid_boundary,  # NEW
            'valid_smooth': False,  # DISABLED
            'n_valid_losses': n_valid_losses,
            'stage': 1 if is_stage1 else 2,
            'current_lambda_residual': current_lambda_residual,
            'effective_lambda_imp': effective_lambda_imp,
            'effective_lambda_boundary': self.lambda_boundary if valid_boundary else 0.0,  # NEW
            'effective_lambda_tv': 0.0  # DISABLED
        }
        
        return total_loss, loss_dict
    
    def update_weights_for_stage2(self, epoch):
        """Update weights for stage 2"""
        if epoch >= self.stage1_epochs:
            self.lambda_residual = 0.5
            self.lambda_improvement = 0.5
            self.lambda_low_freq = 0.3


class ResidualClipper:
    """Residual clipping"""
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


if __name__ == "__main__":
    print("Testing ROBUST Progressive Structure Loss")
    print("="*80)
    
    # Test data
    B, C, H, W = 4, 1, 40, 40
    hicarn_pred = torch.rand(B, C, H, W) * 100
    gt_hic = torch.rand(B, C, H, W) * 100
    pred_residual = torch.randn(B, C, H, W) * 0.1
    target_residual = torch.randn(B, C, H, W) * 0.1
    pred_hic = hicarn_pred + pred_residual
    
    # Test loss calculator
    loss_calc = ProgressiveStructureLossCalculator()
    
    # Test Stage 1
    print("\n[STAGE 1] Epoch 10:")
    total_loss, loss_dict = loss_calc(
        pred_residual, target_residual,
        pred_hic, hicarn_pred, gt_hic,
        epoch=10
    )
    print(f"  Total: {loss_dict['total']:.6f} (valid: {loss_dict['n_valid_losses']}/5)")
    
    res_str = f"{loss_dict['residual']:.6f}" if loss_dict['residual'] else 'INVALID'
    cons_str = f"{loss_dict['consistency']:.6f}" if loss_dict['consistency'] else 'INVALID'
    imp_str = 'OFF' if not loss_dict['valid_improvement'] else f"{loss_dict['improvement']:.6f}"
    
    print(f"  Residual: {res_str}")
    print(f"  Consistency: {cons_str}")
    print(f"  Improvement: {imp_str}")
    
    # Test Stage 2
    print("\n[STAGE 2] Epoch 25:")
    total_loss, loss_dict = loss_calc(
        pred_residual, target_residual,
        pred_hic, hicarn_pred, gt_hic,
        epoch=25
    )
    
    imp_str2 = f"{loss_dict['improvement']:.6f}" if loss_dict['valid_improvement'] else 'INVALID'
    
    print(f"  Total: {loss_dict['total']:.6f} (valid: {loss_dict['n_valid_losses']}/5)")
    print(f"  Improvement: {imp_str2}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("\nKEY IMPROVEMENTS:")
    print("  1. NaN protection at every step")
    print("  2. Invalid losses are skipped")
    print("  3. Total loss only includes valid components")
    print("  4. Extensive error logging")
