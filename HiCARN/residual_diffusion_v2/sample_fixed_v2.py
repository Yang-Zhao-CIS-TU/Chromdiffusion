#!/usr/bin/env python3
"""
修复版采样脚本

关键修复：
1. 确保使用 pred_original_sample (pred_x0) 而不是 x_{t-1}
2. 正确处理 v-parameterization
3. 正确计算最终输出: final = hicarn + residual_x0

问题诊断：
- 原脚本返回的 `residual` 是 x_{t-1}（仍含噪）
- 应该返回 `pred_original_sample` (pred_x0)，这才是干净的 residual
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import sys

# Import model and scheduler
from model import ImprovedResidualDiffusionUNet
from scheduler import ImprovedDDPMScheduler


# ================================================================
# For torch.load
# ================================================================
class RobustHiCPreprocessor:
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self._is_fitted = False

    def postprocess(self, Y_norm):
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted")
        Y_norm = np.clip(Y_norm, -5, 5)
        Y_log = Y_norm * self.Y_std + self.Y_mean
        Y_counts = np.expm1(Y_log)
        return np.maximum(Y_counts, 0.0).astype(np.float32)

sys.modules['__main__'].RobustHiCPreprocessor = RobustHiCPreprocessor


def load_preprocessor_stats(preprocessor_path):
    """Load Y_median and Y_iqr from preprocessor"""
    try:
        checkpoint = torch.load(preprocessor_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(preprocessor_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'preprocessor' in checkpoint:
        prep = checkpoint['preprocessor']
    else:
        prep = checkpoint
    
    if hasattr(prep, 'Y_mean') and hasattr(prep, 'Y_std'):
        return prep.Y_mean, prep.Y_std, prep
    else:
        return None, None, None


@torch.no_grad()
def sample_residual_diffusion_fixed(
    model,
    scheduler,
    condition,
    num_steps=50,
    use_ddim=True,
    ddim_eta=0.0,
    device='cuda',
    clip_output=True
):
    """
    修复版采样函数
    
    关键修复：
    1. 返回 pred_x0 而不是 x_{t-1}
    2. 正确处理 v-parameterization
    
    Args:
        model: Trained diffusion model
        scheduler: Noise scheduler (ImprovedDDPMScheduler)
        condition: HiCARN predictions [B, 1, H, W] (这是条件输入)
        num_steps: Denoising steps
        use_ddim: Use DDIM sampling
        ddim_eta: DDIM eta parameter
        device: Device
        clip_output: Whether to clip output to [-5, 5]
    
    Returns:
        final_pred: condition + residual_x0 [B, 1, H, W]
        residual_x0: The clean residual (pred_x0) [B, 1, H, W]
    """
    batch_size = condition.shape[0]
    
    # Start from pure noise (这是 residual 的 x_T)
    x_t = torch.randn_like(condition)
    
    # Set sampling timesteps
    scheduler.set_timesteps(num_steps, device=device, method='uniform')
    
    # 保存最终的 pred_x0
    pred_x0_final = None
    
    # Denoising loop
    for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Model predicts v (or eps, depending on parameterization)
        model_output = model(x_t, t_batch, condition)
        
        # Scheduler step - 关键是获取 pred_original_sample
        x_t_minus_1, pred_x0 = scheduler.step(
            model_output,
            t,
            x_t,
            eta=ddim_eta if use_ddim else 0.0,
            use_ddim=use_ddim
        )
        
        # Update x_t for next iteration
        x_t = x_t_minus_1
        
        # 保存 pred_x0（最后一步的 pred_x0 就是我们要的 clean residual）
        pred_x0_final = pred_x0
    
    # 最终的 clean residual 是 pred_x0，不是 x_t!
    # x_t 在最后一步可能还有残余噪声
    residual_x0 = pred_x0_final
    
    # Clip residual to reasonable range
    if clip_output:
        # Residual 应该在合理范围内，比如 [-2, 2]
        # 因为它是对 HiCARN 的小修正
        residual_x0 = torch.clamp(residual_x0, -3, 3)
    
    # 最终预测 = HiCARN + residual
    final_pred = condition + residual_x0
    
    # Clip final prediction to training range
    if clip_output:
        final_pred = torch.clamp(final_pred, -5, 5)
    
    return final_pred, residual_x0


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model and scheduler"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get configs
    scheduler_config = checkpoint.get('scheduler_config', {})
    model_config = checkpoint.get('config', {})
    
    num_train_timesteps = scheduler_config.get('num_train_timesteps', 1000)
    parameterization = scheduler_config.get('parameterization', 'v')
    
    print(f"  Scheduler: {num_train_timesteps} timesteps")
    print(f"  Parameterization: {parameterization}")
    
    # Create scheduler
    scheduler = ImprovedDDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        parameterization=parameterization
    )
    
    # Create model
    model = ImprovedResidualDiffusionUNet(
        in_channels=1,
        out_channels=1,
        cond_channels=1,
        base_channels=model_config.get('base_channels', 64),
        channel_mults=tuple(model_config.get('channel_multipliers', [1, 2, 4, 8])),
        num_res_blocks=model_config.get('num_res_blocks', 2),
        attn_levels=(2, 3),
        parameterization=parameterization
    ).to(device)
    
    # Load weights
    if 'ema_shadow' in checkpoint and checkpoint['ema_shadow']:
        print("  Using EMA weights")
        model.load_state_dict(checkpoint['ema_shadow'])
    else:
        print("  Using regular weights")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Loaded from epoch: {epoch}")
    
    return model, scheduler


def main():
    parser = argparse.ArgumentParser(description='Fixed sampling script')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--pred_path', type=str, required=True,
                       help='HiCARN predictions (normalized)')
    parser.add_argument('--preprocessor_path', type=str, default=None,
                       help='Preprocessor for denormalization')
    parser.add_argument('--output_dir', type=str, default='refined_fixed')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--use_ddim', action='store_true', default=True)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_clip', action='store_true',
                       help='Do not clip output')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ====================================================================
    # Load model
    # ====================================================================
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    model, scheduler = load_checkpoint(args.checkpoint, device)
    
    # ====================================================================
    # Load HiCARN predictions
    # ====================================================================
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    print(f"Loading HiCARN predictions: {args.pred_path}")
    hicarn_norm = np.load(args.pred_path)
    
    if hicarn_norm.ndim == 3:
        hicarn_norm = hicarn_norm[:, np.newaxis, :, :]
    
    num_samples = hicarn_norm.shape[0]
    print(f"  Shape: {hicarn_norm.shape}")
    print(f"  Range: [{hicarn_norm.min():.4f}, {hicarn_norm.max():.4f}]")
    
    # ====================================================================
    # Load preprocessor
    # ====================================================================
    hic_preprocessor = None
    if args.preprocessor_path:
        print(f"\nLoading preprocessor: {args.preprocessor_path}")
        Y_median, Y_iqr, prep = load_preprocessor_stats(args.preprocessor_path)
        if Y_median is not None:
            print(f"  Y_median: {Y_median:.4f}")
            print(f"  Y_iqr: {Y_iqr:.4f}")
            hic_preprocessor = prep
    
    # ====================================================================
    # Sampling
    # ====================================================================
    print("\n" + "="*80)
    print("SAMPLING (FIXED VERSION)")
    print("="*80)
    print(f"Method: {'DDIM' if args.use_ddim else 'DDPM'}")
    print(f"Steps: {args.num_steps}")
    print(f"Clip output: {not args.no_clip}")
    print()
    
    final_list = []
    residual_list = []
    
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Sampling"):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, num_samples)
        
        batch = hicarn_norm[start_idx:end_idx]
        batch_tensor = torch.from_numpy(batch).float().to(device)
        
        # 使用修复版采样函数
        final_batch, residual_batch = sample_residual_diffusion_fixed(
            model=model,
            scheduler=scheduler,
            condition=batch_tensor,
            num_steps=args.num_steps,
            use_ddim=args.use_ddim,
            ddim_eta=args.ddim_eta,
            device=device,
            clip_output=not args.no_clip
        )
        
        final_list.append(final_batch.cpu().numpy())
        residual_list.append(residual_batch.cpu().numpy())
    
    # Concatenate
    final_norm = np.concatenate(final_list, axis=0)
    residuals = np.concatenate(residual_list, axis=0)
    
    print(f"\n✓ Sampling complete!")
    print(f"  Final pred (norm): range [{final_norm.min():.4f}, {final_norm.max():.4f}]")
    print(f"  Residuals: range [{residuals.min():.4f}, {residuals.max():.4f}]")
    print(f"  Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}")
    
    # ====================================================================
    # 对比检查
    # ====================================================================
    print("\n" + "="*80)
    print("对比检查")
    print("="*80)
    
    # HiCARN的范围
    print(f"HiCARN range:     [{hicarn_norm.min():.4f}, {hicarn_norm.max():.4f}]")
    print(f"Final pred range: [{final_norm.min():.4f}, {final_norm.max():.4f}]")
    
    # 检查 final 是否和 hicarn 相近（应该相近，因为 residual 应该小）
    diff = final_norm - hicarn_norm
    print(f"Difference (final - hicarn):")
    print(f"  Range: [{diff.min():.4f}, {diff.max():.4f}]")
    print(f"  Mean: {diff.mean():.4f}, Std: {diff.std():.4f}")
    
    # ====================================================================
    # Save results
    # ====================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save final predictions (normalized)
    final_path = output_dir / 'refined_norm.npy'
    np.save(final_path, final_norm)
    print(f"✓ Saved: {final_path}")
    
    # Save residuals
    residual_path = output_dir / 'residuals.npy'
    np.save(residual_path, residuals)
    print(f"✓ Saved: {residual_path}")
    
    # Also save HiCARN copy for convenience
    hicarn_copy_path = output_dir / 'hicarn_norm.npy'
    np.save(hicarn_copy_path, hicarn_norm)
    print(f"✓ Saved: {hicarn_copy_path}")
    
    # Save stats
    stats = {
        'num_samples': int(num_samples),
        'hicarn_range': [float(hicarn_norm.min()), float(hicarn_norm.max())],
        'final_range': [float(final_norm.min()), float(final_norm.max())],
        'residual_range': [float(residuals.min()), float(residuals.max())],
        'residual_mean': float(residuals.mean()),
        'residual_std': float(residuals.std()),
        'method': 'DDIM' if args.use_ddim else 'DDPM',
        'num_steps': args.num_steps,
        'clipped': not args.no_clip
    }
    
    stats_path = output_dir / 'sampling_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: {stats_path}")
    
    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
关键修复：
1. 使用 pred_x0 (pred_original_sample) 而不是 x_t
2. Final = HiCARN + residual_x0
3. Residual 被 clip 到 [-3, 3]
4. Final 被 clip 到 [-5, 5]

输出文件:
  - {final_path} (最终预测，用于评估)
  - {residual_path} (残差)
  - {hicarn_copy_path} (HiCARN 副本)

下一步：用 evaluate_normalized.py 评估:
  python evaluate_normalized.py \\
      --pred_norm_path {final_path} \\
      --gt_path <GT_PATH> \\
      --preprocessor_path <PREPROCESSOR_PATH>
""")
    print("="*80)


if __name__ == '__main__':
    main()
