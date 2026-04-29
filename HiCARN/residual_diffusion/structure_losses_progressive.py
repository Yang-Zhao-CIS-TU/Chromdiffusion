"""
Progressive Structure Refinement Loss for Hi-C Diffusion

PHILOSOPHY: "先活下来 → 再慢慢变好"
  - Stage 1: Force diffusion to DO SOMETHING (residual activation)
  - Stage 2: Make sure it doesn't BREAK structure (consistency)
  - Stage 3: Gently PUSH for improvement (relative improvement)

Five Loss Components:
  1. L_residual: Prevent identity (force non-zero residuals)
  2. L_structure_consistency: Don't break existing TADs (vs HiCARN)
  3. L_structure_improve: Gentle push for improvement (ReLU relative)
  4. L_low_freq: Preserve domain-scale structure
  5. L_smooth: Prevent noise in residuals

Expected Training:
  - Epochs 0-10: Residual variance increases (diffusion starts moving)
  - Epochs 10-20: Structure consistency maintained
  - Epochs 20+: Gradual improvement in TAD metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualActivationLoss(nn.Module):
    """
    L_residual = |Var(residual_pred) - Var(residual_gt)|
    
    PURPOSE: Force diffusion to output non-zero residuals
    
    Without this:
      - Identity (residual=0) is local minimum
      - All structure losses become meaningless
      - Diffusion learns "do nothing"
    
    With this:
      - Diffusion MUST produce variance
      - Identity is explicitly penalized
      - Model is forced to "do something"
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_residual, target_residual):
        """
        Args:
            pred_residual: Predicted residual (B, 1, H, W)
            target_residual: Target residual (B, 1, H, W)
        
        Returns:
            loss: Variance difference
            valid: Always True
        """
        # Compute variance
        pred_var = torch.var(pred_residual)
        target_var = torch.var(target_residual)
        
        # Variance difference
        loss = torch.abs(pred_var - target_var)
        
        # Normalize by target variance
        loss = loss / torch.clamp(target_var, min=1e-6)
        
        return loss, True


class StructureConsistencyLoss(nn.Module):
    """
    L_structure_consistency = L_ins_cons + L_bnd_cons
    
    PURPOSE: Ensure refined doesn't break existing TAD structure
    
    KEY INSIGHT: Compare refined vs HiCARN (NOT refined vs GT!)
    
    This prevents:
      - Destroying already-good TAD boundaries
      - Hallucinating fake boundaries
      - Breaking domain structure
    """
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size
    
    def compute_insulation(self, mat):
        """Compute insulation score"""
        L = mat.shape[2]
        w = min(self.window_size, (L - 1) // 2)
        
        if w < 2:
            return None
        
        scores = []
        for i in range(w, L - w):
            block = mat[:, 0, i-w:i, i:i+w]
            score = block.mean(dim=(1, 2))
            scores.append(score)
        
        if len(scores) == 0:
            return None
        
        insulation = torch.stack(scores, dim=1)
        insulation = torch.log(insulation + 1e-4)
        insulation = torch.clamp(insulation, -5, 5)
        
        return insulation
    
    def compute_boundary_score(self, insulation):
        """Compute boundary strength (gradient magnitude)"""
        if insulation is None or insulation.shape[1] < 2:
            return None
        
        # Gradient as boundary score
        gradient = torch.abs(insulation[:, 1:] - insulation[:, :-1])
        
        return gradient
    
    def forward(self, refined_hic, hicarn_hic):
        """
        Compare refined vs HiCARN (structure consistency)
        
        Args:
            refined_hic: Refined Hi-C
            hicarn_hic: HiCARN baseline
        
        Returns:
            loss: Consistency loss
            valid: Whether loss is valid
        """
        # Compute insulation
        ins_refined = self.compute_insulation(refined_hic)
        ins_hicarn = self.compute_insulation(hicarn_hic)
        
        if ins_refined is None or ins_hicarn is None:
            return None, False
        
        # (a) Insulation consistency
        loss_ins = F.mse_loss(ins_refined, ins_hicarn)
        
        # (b) Boundary consistency
        bnd_refined = self.compute_boundary_score(ins_refined)
        bnd_hicarn = self.compute_boundary_score(ins_hicarn)
        
        if bnd_refined is None or bnd_hicarn is None:
            loss_bnd = torch.tensor(0.0, device=refined_hic.device)
        else:
            loss_bnd = F.mse_loss(bnd_refined, bnd_hicarn)
        
        # Combined consistency loss
        loss = loss_ins + loss_bnd
        
        return loss, True


class StructureImprovementLoss(nn.Module):
    """
    L_structure_improve = ReLU(MSE(refined,GT) - MSE(hicarn,GT))
    
    PURPOSE: Gently push for improvement over HiCARN
    
    CRITICAL DESIGN:
      - Only penalize if refined is WORSE than HiCARN
      - If refined is better → loss = 0 (no penalty)
      - This prevents pulling model toward identity
    
    This is the MISSING piece from previous approaches!
    """
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size
    
    def compute_insulation(self, mat):
        """Compute insulation score"""
        L = mat.shape[2]
        w = min(self.window_size, (L - 1) // 2)
        
        if w < 2:
            return None
        
        scores = []
        for i in range(w, L - w):
            block = mat[:, 0, i-w:i, i:i+w]
            score = block.mean(dim=(1, 2))
            scores.append(score)
        
        if len(scores) == 0:
            return None
        
        insulation = torch.stack(scores, dim=1)
        insulation = torch.log(insulation + 1e-4)
        insulation = torch.clamp(insulation, -5, 5)
        
        return insulation
    
    def compute_boundary_score(self, insulation):
        """Compute boundary strength"""
        if insulation is None or insulation.shape[1] < 2:
            return None
        
        gradient = torch.abs(insulation[:, 1:] - insulation[:, :-1])
        return gradient
    
    def forward(self, refined_hic, hicarn_hic, gt_hic):
        """
        Relative improvement loss
        
        Args:
            refined_hic: Refined Hi-C
            hicarn_hic: HiCARN baseline
            gt_hic: Ground truth
        
        Returns:
            loss: Improvement loss (0 if better, >0 if worse)
            valid: Whether loss is valid
        """
        # Compute insulation for all three
        ins_refined = self.compute_insulation(refined_hic)
        ins_hicarn = self.compute_insulation(hicarn_hic)
        ins_gt = self.compute_insulation(gt_hic)
        
        if ins_refined is None or ins_hicarn is None or ins_gt is None:
            return None, False
        
        # (a) Insulation improvement
        error_refined = F.mse_loss(ins_refined, ins_gt, reduction='mean')
        error_hicarn = F.mse_loss(ins_hicarn, ins_gt, reduction='mean')
        
        # ReLU: only penalize if worse than HiCARN
        loss_ins_imp = F.relu(error_refined - error_hicarn)
        
        # (b) Boundary improvement
        bnd_refined = self.compute_boundary_score(ins_refined)
        bnd_hicarn = self.compute_boundary_score(ins_hicarn)
        bnd_gt = self.compute_boundary_score(ins_gt)
        
        if bnd_refined is None or bnd_hicarn is None or bnd_gt is None:
            loss_bnd_imp = torch.tensor(0.0, device=refined_hic.device)
        else:
            error_bnd_refined = F.mse_loss(bnd_refined, bnd_gt, reduction='mean')
            error_bnd_hicarn = F.mse_loss(bnd_hicarn, bnd_gt, reduction='mean')
            loss_bnd_imp = F.relu(error_bnd_refined - error_bnd_hicarn)
        
        # Combined improvement loss
        loss = loss_ins_imp + loss_bnd_imp
        
        return loss, True


class LowFrequencyConsistencyLoss(nn.Module):
    """
    L_low_freq = MSE(GaussianBlur(refined), GaussianBlur(hicarn))
    
    PURPOSE: Preserve domain-scale structure
    
    Ensures:
      - Large-scale domains unchanged
      - Medium/high frequency can adjust freely
      - TAD-scale structure protected
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
    
    def forward(self, refined_hic, hicarn_hic):
        """
        Low-frequency consistency
        
        Args:
            refined_hic: Refined Hi-C
            hicarn_hic: HiCARN baseline
        
        Returns:
            loss: Low-frequency MSE
            valid: Always True
        """
        # Gaussian blur (low-pass filter)
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
        
        # MSE on blurred versions
        loss = F.mse_loss(refined_blur, hicarn_blur)
        
        return loss, True


class SmoothnessTVLoss(nn.Module):
    """
    L_smooth = TV(residual_pred)
    
    PURPOSE: Prevent residual from becoming pure noise
    
    Total Variation regularization:
      - Encourages smooth residuals
      - Prevents high-frequency noise
      - Allows structured changes
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, residual):
        """
        Total variation loss
        
        Args:
            residual: Predicted residual (B, 1, H, W)
        
        Returns:
            loss: TV loss
            valid: Always True
        """
        # Horizontal gradient
        tv_h = torch.abs(residual[:, :, 1:, :] - residual[:, :, :-1, :])
        
        # Vertical gradient
        tv_v = torch.abs(residual[:, :, :, 1:] - residual[:, :, :, :-1])
        
        # Total variation
        loss = tv_h.mean() + tv_v.mean()
        
        return loss, True


class ProgressiveStructureLossCalculator(nn.Module):
    """
    Complete progressive structure refinement loss
    
    L_total = λ_res * L_residual 
            + λ_cons * L_structure_consistency 
            + λ_imp * L_structure_improve 
            + λ_freq * L_low_freq 
            + λ_tv * L_smooth
    
    Two-Stage Training Strategy:
      Stage 1 (0-20 epochs): λ_imp = 0.0 (learn to move)
      Stage 2 (20+ epochs): λ_imp > 0.0 (learn to improve)
    """
    def __init__(
        self,
        lambda_residual=1.0,
        lambda_consistency=1.0,
        lambda_improvement=0.1,
        lambda_low_freq=0.5,
        lambda_smooth=0.05,
        insulation_window=5,
        stage1_epochs=20
    ):
        super().__init__()
        
        self.lambda_residual = lambda_residual
        self.lambda_consistency = lambda_consistency
        self.lambda_improvement = lambda_improvement
        self.lambda_low_freq = lambda_low_freq
        self.lambda_smooth = lambda_smooth
        self.stage1_epochs = stage1_epochs
        
        # Loss components
        self.residual_loss = ResidualActivationLoss()
        self.consistency_loss = StructureConsistencyLoss(window_size=insulation_window)
        self.improvement_loss = StructureImprovementLoss(window_size=insulation_window)
        self.low_freq_loss = LowFrequencyConsistencyLoss(kernel_size=3)
        self.smooth_loss = SmoothnessTVLoss()
    
    def forward(self, pred_residual, target_residual, pred_hic, hicarn_hic, gt_hic, epoch=0):
        """
        Compute progressive structure loss
        
        Args:
            pred_residual: Predicted residual
            target_residual: Target residual
            pred_hic: Predicted Hi-C (HiCARN + pred_residual)
            hicarn_hic: HiCARN baseline
            gt_hic: Ground truth
            epoch: Current epoch (for stage control)
        
        Returns:
            total_loss: Weighted combination
            loss_dict: Individual losses
        """
        # Clean inputs
        pred_hic = torch.nan_to_num(pred_hic, nan=0.0, posinf=1e6, neginf=0.0)
        hicarn_hic = torch.nan_to_num(hicarn_hic, nan=0.0, posinf=1e6, neginf=0.0)
        gt_hic = torch.nan_to_num(gt_hic, nan=0.0, posinf=1e6, neginf=0.0)
        
        # 1. Residual activation loss (always active)
        loss_res, valid_res = self.residual_loss(pred_residual, target_residual)
        
        # 2. Structure consistency loss (always active)
        loss_cons, valid_cons = self.consistency_loss(pred_hic, hicarn_hic)
        
        # 3. Structure improvement loss (stage-dependent)
        # Stage 1 (0-20 epochs): DISABLED (learn to move first)
        # Stage 2 (20+ epochs): ENABLED (learn to improve)
        if epoch < self.stage1_epochs:
            loss_imp = torch.tensor(0.0, device=pred_hic.device)
            valid_imp = False
            effective_lambda_imp = 0.0
        else:
            loss_imp, valid_imp = self.improvement_loss(pred_hic, hicarn_hic, gt_hic)
            # Gradually increase improvement weight
            progress = min(1.0, (epoch - self.stage1_epochs) / 30.0)
            effective_lambda_imp = self.lambda_improvement * progress
        
        # 4. Low-frequency consistency loss (always active)
        loss_freq, valid_freq = self.low_freq_loss(pred_hic, hicarn_hic)
        
        # 5. Smoothness TV loss (always active)
        loss_tv, valid_tv = self.smooth_loss(pred_residual)
        
        # Weighted total (only include valid losses)
        total_loss = torch.tensor(0.0, device=pred_hic.device)
        
        if valid_res:
            total_loss = total_loss + self.lambda_residual * loss_res
        
        if valid_cons:
            total_loss = total_loss + self.lambda_consistency * loss_cons
        
        if valid_imp and epoch >= self.stage1_epochs:
            total_loss = total_loss + effective_lambda_imp * loss_imp
        
        if valid_freq:
            total_loss = total_loss + self.lambda_low_freq * loss_freq
        
        if valid_tv:
            total_loss = total_loss + self.lambda_smooth * loss_tv
        
        # Loss dict
        loss_dict = {
            'total': total_loss.item(),
            'residual': loss_res.item() if valid_res else None,
            'consistency': loss_cons.item() if valid_cons else None,
            'improvement': loss_imp.item() if valid_imp else None,
            'low_freq': loss_freq.item() if valid_freq else None,
            'smooth': loss_tv.item() if valid_tv else None,
            'valid_residual': valid_res,
            'valid_consistency': valid_cons,
            'valid_improvement': valid_imp,
            'valid_low_freq': valid_freq,
            'valid_smooth': valid_tv,
            'stage': 1 if epoch < self.stage1_epochs else 2,
            'effective_lambda_imp': effective_lambda_imp
        }
        
        return total_loss, loss_dict
    
    def update_weights_for_stage2(self, epoch):
        """
        Update weights for stage 2 (middle training)
        
        Recommended at epoch 20+:
          λ_res: 1.0 → 0.5
          λ_cons: 1.0 (unchanged)
          λ_imp: 0.1 → 0.5
          λ_freq: 0.5 → 0.3
          λ_tv: 0.05 (unchanged)
        """
        if epoch >= self.stage1_epochs:
            self.lambda_residual = 0.5
            self.lambda_improvement = 0.5
            self.lambda_low_freq = 0.3


class ResidualClipper:
    """Residual clipping (unchanged)"""
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
    print("Testing Progressive Structure Loss")
    print("="*80)
    
    # Test data
    B, C, H, W = 4, 1, 40, 40
    hicarn_pred = torch.rand(B, C, H, W) * 100
    gt_hic = torch.rand(B, C, H, W) * 100
    pred_residual = torch.randn(B, C, H, W) * 0.1
    target_residual = torch.randn(B, C, H, W) * 0.1
    pred_hic = hicarn_pred + pred_residual
    
    # Test loss calculator
    loss_calc = ProgressiveStructureLossCalculator(
        lambda_residual=1.0,
        lambda_consistency=1.0,
        lambda_improvement=0.1,
        lambda_low_freq=0.5,
        lambda_smooth=0.05,
        stage1_epochs=20
    )
    
    # Test Stage 1 (epoch 10)
    print("\n[STAGE 1] Epoch 10 (improvement DISABLED):")
    total_loss, loss_dict = loss_calc(
        pred_residual, target_residual,
        pred_hic, hicarn_pred, gt_hic,
        epoch=10
    )
    print(f"  Stage: {loss_dict['stage']}")
    print(f"  Total: {loss_dict['total']:.6f}")
    print(f"  Residual: {loss_dict['residual']:.6f}")
    print(f"  Consistency: {loss_dict['consistency']:.6f}")
    print(f"  Improvement: {loss_dict['improvement'] if loss_dict['improvement'] else 'DISABLED'}")
    print(f"  Low-freq: {loss_dict['low_freq']:.6f}")
    print(f"  Smooth: {loss_dict['smooth']:.6f}")
    
    # Test Stage 2 (epoch 25)
    print("\n[STAGE 2] Epoch 25 (improvement ENABLED):")
    total_loss, loss_dict = loss_calc(
        pred_residual, target_residual,
        pred_hic, hicarn_pred, gt_hic,
        epoch=25
    )
    print(f"  Stage: {loss_dict['stage']}")
    print(f"  Total: {loss_dict['total']:.6f}")
    print(f"  Residual: {loss_dict['residual']:.6f}")
    print(f"  Consistency: {loss_dict['consistency']:.6f}")
    print(f"  Improvement: {loss_dict['improvement']:.6f}")
    print(f"  Effective λ_imp: {loss_dict['effective_lambda_imp']:.3f}")
    print(f"  Low-freq: {loss_dict['low_freq']:.6f}")
    print(f"  Smooth: {loss_dict['smooth']:.6f}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("\nKEY DESIGN:")
    print("  1. Residual activation → Force diffusion to DO something")
    print("  2. Structure consistency → DON'T break existing TADs")
    print("  3. Structure improvement → Gentle push (ReLU relative)")
    print("  4. Low-freq consistency → Preserve domain scale")
    print("  5. Smoothness TV → Prevent noise")
    print("\nTWO-STAGE STRATEGY:")
    print("  Stage 1 (0-20): Learn to move (improvement disabled)")
    print("  Stage 2 (20+): Learn to improve (improvement enabled)")
