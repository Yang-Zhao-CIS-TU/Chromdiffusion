#!/usr/bin/env python3
"""
Sanity Check: 验证 Conditioning 是否真的生效

两个关键测试：
1. Check A: 同一个 hicarn，不同随机种子，residual 是否几乎相同？
   - 如果 conditioning 生效，不同种子应该给出不同的 residual（因为依赖噪声）
   - 但 residual 与 hicarn 的相关性应该较高

2. Check B: 打乱 hicarn 顺序（条件错位），性能是否明显下降？
   - 如果 conditioning 生效，错位条件会导致性能暴跌
   - 如果 conditioning 没生效，性能几乎不变

预期结果：
- 如果模型真的用了条件：Check B 性能暴跌
- 如果模型没用条件：Check B 性能几乎不变 ← 你现在很可能是这个情况
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy import stats
import sys

# Import model and scheduler
from model import ImprovedResidualDiffusionUNet
from scheduler import ImprovedDDPMScheduler


# For torch.load
class RobustHiCPreprocessor:
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self._is_fitted = False

sys.modules['__main__'].RobustHiCPreprocessor = RobustHiCPreprocessor


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model and scheduler"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    scheduler_config = checkpoint.get('scheduler_config', {})
    model_config = checkpoint.get('config', {})
    
    num_train_timesteps = scheduler_config.get('num_train_timesteps', 1000)
    parameterization = scheduler_config.get('parameterization', 'v')
    
    scheduler = ImprovedDDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        parameterization=parameterization
    )
    
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
    
    if 'ema_shadow' in checkpoint and checkpoint['ema_shadow']:
        model.load_state_dict(checkpoint['ema_shadow'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model, scheduler


@torch.no_grad()
def sample_with_condition(model, scheduler, condition, num_steps=50, device='cuda', seed=None):
    """Sample residual with given condition"""
    if seed is not None:
        torch.manual_seed(seed)
    
    batch_size = condition.shape[0]
    x_t = torch.randn_like(condition)
    
    scheduler.set_timesteps(num_steps, device=device, method='uniform')
    
    pred_x0 = None
    for t in scheduler.timesteps:
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        model_output = model(x_t, t_batch, condition)
        x_t, pred_x0 = scheduler.step(model_output, t, x_t, eta=0.0, use_ddim=True)
    
    return pred_x0  # Return the predicted clean residual


def normalize_gt(gt_raw, Y_median, Y_iqr):
    gt_log = np.log1p(gt_raw)
    gt_norm = (gt_log - Y_median) / Y_iqr
    return np.clip(gt_norm, -5, 5).astype(np.float32)


def load_preprocessor_stats(path):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    
    # Handle different formats
    if isinstance(checkpoint, dict):
        prep = checkpoint.get('preprocessor', checkpoint)
    else:
        prep = checkpoint
    
    # If prep is still a dict, try to get the actual preprocessor
    if isinstance(prep, dict) and 'preprocessor' in prep:
        prep = prep['preprocessor']
    
    return prep.Y_mean, prep.Y_std


def compute_metrics(pred, gt):
    """Compute MSE and PCC"""
    pred = pred.squeeze()
    gt = gt.squeeze()
    
    mse = float(np.mean((pred - gt) ** 2))
    pcc, _ = stats.pearsonr(pred.flatten(), gt.flatten())
    return mse, float(pcc) if not np.isnan(pcc) else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hicarn_path', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--preprocessor_path', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to test')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("CONDITIONING SANITY CHECK")
    print("="*80)
    
    # Load model
    model, scheduler = load_checkpoint(args.checkpoint, device)
    
    # Load data
    print("\nLoading data...")
    Y_median, Y_iqr = load_preprocessor_stats(args.preprocessor_path)
    
    hicarn = np.load(args.hicarn_path)
    gt_raw = np.load(args.gt_path)
    gt_norm = normalize_gt(gt_raw, Y_median, Y_iqr)
    
    # Ensure shapes
    if hicarn.ndim == 3:
        hicarn = hicarn[:, np.newaxis, :, :]
    if gt_norm.ndim == 4 and gt_norm.shape[-1] == 1:
        gt_norm = gt_norm.squeeze(-1)
    if gt_norm.ndim == 3:
        gt_norm = gt_norm[:, np.newaxis, :, :]
    
    # Limit samples
    n = min(args.num_samples, len(hicarn))
    hicarn = hicarn[:n]
    gt_norm = gt_norm[:n]
    
    print(f"  HiCARN: {hicarn.shape}, range [{hicarn.min():.3f}, {hicarn.max():.3f}]")
    print(f"  GT norm: {gt_norm.shape}, range [{gt_norm.min():.3f}, {gt_norm.max():.3f}]")
    
    hicarn_tensor = torch.from_numpy(hicarn).float().to(device)
    gt_tensor = torch.from_numpy(gt_norm).float().to(device)
    
    # ================================================================
    # Check A: 不同随机种子，residual 的一致性
    # ================================================================
    print("\n" + "="*80)
    print("CHECK A: 不同随机种子的 Residual 一致性")
    print("="*80)
    
    print("\n用相同条件，不同种子采样 3 次...")
    
    residuals_different_seeds = []
    for seed in [42, 123, 999]:
        residual = sample_with_condition(
            model, scheduler, hicarn_tensor,
            num_steps=args.num_steps, device=device, seed=seed
        )
        residuals_different_seeds.append(residual.cpu().numpy())
    
    r1, r2, r3 = residuals_different_seeds
    
    # 计算不同种子之间的相关性
    corr_12, _ = stats.pearsonr(r1.flatten(), r2.flatten())
    corr_13, _ = stats.pearsonr(r1.flatten(), r3.flatten())
    corr_23, _ = stats.pearsonr(r2.flatten(), r3.flatten())
    
    print(f"\n不同种子间 Residual 相关性:")
    print(f"  Corr(seed42, seed123): {corr_12:.4f}")
    print(f"  Corr(seed42, seed999): {corr_13:.4f}")
    print(f"  Corr(seed123, seed999): {corr_23:.4f}")
    
    avg_seed_corr = (corr_12 + corr_13 + corr_23) / 3
    print(f"  Average: {avg_seed_corr:.4f}")
    
    if avg_seed_corr > 0.95:
        print("\n⚠️  警告: 不同种子的 residual 几乎相同!")
        print("   这可能意味着模型输出不依赖随机噪声（或 conditioning 有问题）")
    else:
        print("\n✓ 不同种子产生不同 residual（正常）")
    
    # Residual 与 HiCARN 的相关性
    corr_res_hicarn, _ = stats.pearsonr(r1.flatten(), hicarn.flatten())
    print(f"\nResidual 与 HiCARN 的相关性: {corr_res_hicarn:.4f}")
    
    if abs(corr_res_hicarn) < 0.3:
        print("⚠️  Residual 与 HiCARN 相关性很低，可能没有对齐")
    
    # ================================================================
    # Check B: 打乱 HiCARN 顺序（条件错位）
    # ================================================================
    print("\n" + "="*80)
    print("CHECK B: 条件错位测试（打乱 HiCARN 顺序）")
    print("="*80)
    
    # 正确条件
    print("\n1. 使用正确条件采样...")
    torch.manual_seed(42)
    residual_correct = sample_with_condition(
        model, scheduler, hicarn_tensor,
        num_steps=args.num_steps, device=device, seed=42
    )
    final_correct = hicarn_tensor + residual_correct
    
    mse_correct, pcc_correct = compute_metrics(
        final_correct.cpu().numpy(), gt_norm
    )
    print(f"   MSE: {mse_correct:.6f}, PCC: {pcc_correct:.4f}")
    
    # 打乱条件
    print("\n2. 使用打乱条件采样...")
    perm = torch.randperm(n)
    hicarn_shuffled = hicarn_tensor[perm]
    
    torch.manual_seed(42)  # 相同种子
    residual_shuffled = sample_with_condition(
        model, scheduler, hicarn_shuffled,  # 打乱的条件
        num_steps=args.num_steps, device=device, seed=42
    )
    # 但最终还是加到原始 hicarn 上
    final_shuffled = hicarn_tensor + residual_shuffled
    
    mse_shuffled, pcc_shuffled = compute_metrics(
        final_shuffled.cpu().numpy(), gt_norm
    )
    print(f"   MSE: {mse_shuffled:.6f}, PCC: {pcc_shuffled:.4f}")
    
    # 全零条件
    print("\n3. 使用全零条件采样...")
    hicarn_zeros = torch.zeros_like(hicarn_tensor)
    
    torch.manual_seed(42)
    residual_zeros = sample_with_condition(
        model, scheduler, hicarn_zeros,
        num_steps=args.num_steps, device=device, seed=42
    )
    final_zeros = hicarn_tensor + residual_zeros
    
    mse_zeros, pcc_zeros = compute_metrics(
        final_zeros.cpu().numpy(), gt_norm
    )
    print(f"   MSE: {mse_zeros:.6f}, PCC: {pcc_zeros:.4f}")
    
    # ================================================================
    # 诊断结论
    # ================================================================
    print("\n" + "="*80)
    print("诊断结论")
    print("="*80)
    
    print(f"\n{'Condition':<20} {'MSE':<12} {'PCC':<12}")
    print("-"*44)
    print(f"{'Correct':<20} {mse_correct:<12.6f} {pcc_correct:<12.4f}")
    print(f"{'Shuffled':<20} {mse_shuffled:<12.6f} {pcc_shuffled:<12.4f}")
    print(f"{'Zeros':<20} {mse_zeros:<12.6f} {pcc_zeros:<12.4f}")
    
    # 判断
    mse_diff_shuffled = abs(mse_correct - mse_shuffled) / mse_correct * 100
    mse_diff_zeros = abs(mse_correct - mse_zeros) / mse_correct * 100
    
    print(f"\n性能变化:")
    print(f"  打乱条件 vs 正确条件: MSE 变化 {mse_diff_shuffled:.1f}%")
    print(f"  全零条件 vs 正确条件: MSE 变化 {mse_diff_zeros:.1f}%")
    
    if mse_diff_shuffled < 10 and mse_diff_zeros < 10:
        print("\n" + "❌"*20)
        print("❌ 结论: CONDITIONING 没有生效!")
        print("❌"*20)
        print("""
模型输出几乎不依赖条件输入，这意味着：
1. 训练时可能没有把 HiCARN 作为条件喂给模型
2. 或者模型架构没有正确使用条件

需要检查:
1. model.forward() 是否真的使用了 condition 参数
2. 训练时是否把 hicarn 作为条件传入
3. UNet 的 cond_channels 是否 > 0 并正确处理

修复方法:
1. 确保 UNet 输入是 cat([x_t, hicarn], dim=1)
2. 确保 in_channels = 2 (或使用 FiLM/cross-attention)
3. 重新训练模型
""")
    else:
        print("\n" + "✓"*20)
        print("✓ 结论: CONDITIONING 可能生效")
        print("✓"*20)
        print("""
条件变化会影响输出，但可能还需要进一步优化:
1. 检查是否 condition 强度足够
2. 考虑增加训练 epochs
3. 检查 loss 设计是否合理
""")
    
    # ================================================================
    # 额外检查：看看 residual 分布
    # ================================================================
    print("\n" + "="*80)
    print("额外检查: Residual 分布")
    print("="*80)
    
    residual_np = residual_correct.cpu().numpy()
    print(f"\nResidual 统计:")
    print(f"  Range: [{residual_np.min():.4f}, {residual_np.max():.4f}]")
    print(f"  Mean: {residual_np.mean():.4f}")
    print(f"  Std: {residual_np.std():.4f}")
    
    # 理想的 residual 应该是 gt - hicarn
    ideal_residual = gt_norm - hicarn
    print(f"\n理想 Residual (GT - HiCARN):")
    print(f"  Range: [{ideal_residual.min():.4f}, {ideal_residual.max():.4f}]")
    print(f"  Mean: {ideal_residual.mean():.4f}")
    print(f"  Std: {ideal_residual.std():.4f}")
    
    # 对比
    corr_actual_ideal, _ = stats.pearsonr(
        residual_np.flatten(), ideal_residual.flatten()
    )
    print(f"\n实际 Residual 与 理想 Residual 的相关性: {corr_actual_ideal:.4f}")
    
    if corr_actual_ideal < 0.3:
        print("⚠️  实际 residual 与理想 residual 相关性很低!")
        print("   模型没有学到正确的残差模式")
    elif corr_actual_ideal > 0.7:
        print("✓ 实际 residual 与理想 residual 有较好的相关性")
    
    print("\n" + "="*80)
    print("检查完成")
    print("="*80)


if __name__ == '__main__':
    main()
