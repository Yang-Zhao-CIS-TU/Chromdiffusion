#!/usr/bin/env python3
"""
诊断脚本：确定 refined_norm.npy 到底是什么

测试假设：
1. refined 是最终结果 (hicarn + residual) - 当前假设
2. refined 是 residual 本身 - 应该用 hicarn + refined
3. refined 是 负residual - 应该用 hicarn - refined
4. refined 是 x_t (噪声态) 而不是 x0

通过计算不同组合的 MSE 来判断哪个假设正确。
"""

import numpy as np
import torch
import sys
from pathlib import Path
from scipy import stats

# Setup for torch.load
class RobustHiCPreprocessor:
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self._is_fitted = False

sys.modules['__main__'].RobustHiCPreprocessor = RobustHiCPreprocessor


def ensure_nhw(arr):
    """Ensure array is (N, H, W) format."""
    arr = np.asarray(arr)
    if arr.ndim == 4:
        if arr.shape[1] == 1:
            return arr.squeeze(1)
        elif arr.shape[-1] == 1:
            return arr.squeeze(-1)
    return arr


def normalize_gt(gt_raw, Y_median, Y_iqr):
    """Normalize GT to match training space"""
    gt_log = np.log1p(gt_raw)
    gt_norm = (gt_log - Y_median) / Y_iqr
    gt_norm = np.clip(gt_norm, -5, 5)
    return gt_norm.astype(np.float32)


def load_preprocessor_stats(path):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'preprocessor' in checkpoint:
        prep = checkpoint['preprocessor']
    else:
        prep = checkpoint
    return prep.Y_mean, prep.Y_std


def compute_metrics(pred, gt):
    """Compute MSE and PCC"""
    pred = ensure_nhw(pred)
    gt = ensure_nhw(gt)
    
    mse = float(np.mean((pred - gt) ** 2))
    
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    pcc, _ = stats.pearsonr(pred_flat, gt_flat)
    pcc = float(pcc) if not np.isnan(pcc) else 0.0
    
    return mse, pcc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--chr', type=str, default='chr18')
    parser.add_argument('--hicarn_dir', type=str, 
                       default='/home/yangz/data/hic_data/HiCARN/hicarn_predictions')
    parser.add_argument('--gt_dir', type=str,
                       default='/data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40')
    parser.add_argument('--refined_dir', type=str, default='.')
    args = parser.parse_args()
    
    chr_name = args.chr
    
    print("="*80)
    print(f"诊断: {chr_name} - refined_norm.npy 到底是什么?")
    print("="*80)
    
    # Load preprocessor
    prep_path = f"{args.hicarn_dir}/{chr_name}/preprocessor.pt"
    Y_median, Y_iqr = load_preprocessor_stats(prep_path)
    print(f"\nPreprocessor stats:")
    print(f"  Y_median: {Y_median:.4f}")
    print(f"  Y_iqr:    {Y_iqr:.4f}")
    
    # Load data
    print("\n" + "-"*60)
    print("Loading data...")
    print("-"*60)
    
    # HiCARN predictions (normalized)
    hicarn_path = f"{args.hicarn_dir}/{chr_name}/predictions_norm.npy"
    hicarn = np.load(hicarn_path)
    hicarn = ensure_nhw(hicarn)
    print(f"\nHiCARN (A):")
    print(f"  Shape: {hicarn.shape}")
    print(f"  Range: [{hicarn.min():.4f}, {hicarn.max():.4f}]")
    
    # Refined (what we're diagnosing)
    refined_path = f"{args.refined_dir}/refined_{chr_name}/refined_norm.npy"
    refined = np.load(refined_path)
    refined = ensure_nhw(refined)
    print(f"\nRefined (R) - 需要诊断:")
    print(f"  Shape: {refined.shape}")
    print(f"  Range: [{refined.min():.4f}, {refined.max():.4f}]")
    
    # Ground truth (raw -> normalized)
    gt_path = f"{args.gt_dir}/hr_test_{chr_name}.npy"
    gt_raw = np.load(gt_path)
    gt = normalize_gt(gt_raw, Y_median, Y_iqr)
    gt = ensure_nhw(gt)
    print(f"\nGT (G) - normalized:")
    print(f"  Shape: {gt.shape}")
    print(f"  Range: [{gt.min():.4f}, {gt.max():.4f}]")
    
    # Also check if there's a residuals.npy
    residuals_path = f"{args.refined_dir}/refined_{chr_name}/residuals.npy"
    has_residuals = Path(residuals_path).exists()
    if has_residuals:
        residuals = np.load(residuals_path)
        residuals = ensure_nhw(residuals)
        print(f"\nResiduals file found:")
        print(f"  Shape: {residuals.shape}")
        print(f"  Range: [{residuals.min():.4f}, {residuals.max():.4f}]")
    
    # ================================================================
    # 诊断测试
    # ================================================================
    print("\n" + "="*80)
    print("诊断测试")
    print("="*80)
    
    results = {}
    
    # Test 1: HiCARN alone (baseline)
    mse_a, pcc_a = compute_metrics(hicarn, gt)
    results['HiCARN (A)'] = (mse_a, pcc_a)
    print(f"\n1. MSE(A, G) = {mse_a:.6f}, PCC = {pcc_a:.4f}")
    print(f"   HiCARN baseline - 这应该很好 (~0.06 MSE, ~0.95 PCC)")
    
    # Test 2: Refined alone (current assumption)
    mse_r, pcc_r = compute_metrics(refined, gt)
    results['Refined (R)'] = (mse_r, pcc_r)
    print(f"\n2. MSE(R, G) = {mse_r:.6f}, PCC = {pcc_r:.4f}")
    print(f"   当前结果 - 这是你看到的差结果")
    
    # Test 3: HiCARN + Refined
    combined_add = hicarn + refined
    mse_add, pcc_add = compute_metrics(combined_add, gt)
    results['A + R'] = (mse_add, pcc_add)
    print(f"\n3. MSE(A+R, G) = {mse_add:.6f}, PCC = {pcc_add:.4f}")
    print(f"   如果 R 是 residual，这应该更好")
    
    # Test 4: HiCARN - Refined
    combined_sub = hicarn - refined
    mse_sub, pcc_sub = compute_metrics(combined_sub, gt)
    results['A - R'] = (mse_sub, pcc_sub)
    print(f"\n4. MSE(A-R, G) = {mse_sub:.6f}, PCC = {pcc_sub:.4f}")
    print(f"   如果 R 是负residual，这应该更好")
    
    # Test 5: Just the difference (R - A)
    diff = refined - hicarn
    mse_diff, pcc_diff = compute_metrics(diff, gt - hicarn)  # Compare residuals
    print(f"\n5. R - A (refined减去hicarn):")
    print(f"   Range: [{diff.min():.4f}, {diff.max():.4f}]")
    print(f"   Mean: {diff.mean():.4f}, Std: {diff.std():.4f}")
    
    # Test 6: If residuals.npy exists, check hicarn + residuals
    if has_residuals:
        combined_res = hicarn + residuals
        mse_res, pcc_res = compute_metrics(combined_res, gt)
        results['A + residuals'] = (mse_res, pcc_res)
        print(f"\n6. MSE(A+residuals, G) = {mse_res:.6f}, PCC = {pcc_res:.4f}")
        print(f"   使用 residuals.npy 文件")
    
    # ================================================================
    # 诊断结论
    # ================================================================
    print("\n" + "="*80)
    print("诊断结论")
    print("="*80)
    
    # Find the best combination
    best_name = min(results, key=lambda x: results[x][0])
    best_mse, best_pcc = results[best_name]
    
    print(f"\n最佳组合: {best_name}")
    print(f"  MSE: {best_mse:.6f}")
    print(f"  PCC: {best_pcc:.4f}")
    
    # Diagnose
    print("\n" + "-"*60)
    
    if best_name == 'HiCARN (A)':
        print("❌ 问题严重: HiCARN alone 比任何 refined 组合都好!")
        print("   Diffusion 不仅没帮助，还在损害结果。")
        print("\n可能原因:")
        print("   1. 模型没有正确学习")
        print("   2. scheduler/parameterization 配置错误")
        print("   3. 推理代码有bug")
        
    elif best_name == 'A + R':
        print("✓ 发现问题: refined_norm 实际上是 RESIDUAL!")
        print("   你保存的是 residual，不是最终结果。")
        print("\n修复方法:")
        print("   final = hicarn + refined")
        print("   或者修改 sample 脚本，保存 hicarn + residual")
        
    elif best_name == 'A - R':
        print("✓ 发现问题: refined_norm 是 负RESIDUAL!")
        print("   残差的符号搞反了。")
        print("\n修复方法:")
        print("   final = hicarn - refined")
        print("   或者检查代码中 residual 的符号")
        
    elif best_name == 'Refined (R)':
        print("? refined 直接用是最好的，但结果仍然差")
        print("   说明模型本身有问题，不是合成逻辑错误。")
        
    elif best_name == 'A + residuals':
        print("✓ 发现问题: 应该用 residuals.npy 而不是 refined_norm.npy!")
        print("   refined_norm 可能是其他中间结果。")
    
    # 额外检查：refined 是否接近 hicarn
    corr_refined_hicarn, _ = stats.pearsonr(refined.flatten(), hicarn.flatten())
    print(f"\n额外信息:")
    print(f"  Corr(refined, hicarn): {corr_refined_hicarn:.4f}")
    
    if corr_refined_hicarn > 0.9:
        print("  refined 和 hicarn 高度相关，可能只是加了一点噪声")
    elif corr_refined_hicarn < 0.3:
        print("  refined 和 hicarn 相关性很低，说明 refined 可能是其他东西（如纯residual或噪声）")
    
    # ================================================================
    # 推荐修复
    # ================================================================
    print("\n" + "="*80)
    print("推荐修复")
    print("="*80)
    
    if best_name in ['A + R', 'A - R', 'A + residuals']:
        sign = '+' if best_name != 'A - R' else '-'
        residual_source = 'refined' if 'R' in best_name else 'residuals'
        
        print(f"""
在 sample_combined.py 中修改:

# 当前代码 (错误):
refined_norm = condition + residual_denorm  # 或者只保存了 residual

# 应该是:
# 如果 refined_norm 实际是 residual:
#   final = hicarn {sign} refined_norm

# 快速修复 - 创建正确的输出:
for chr in chr18 chr19 chr20 chr21 chr22; do
    python -c "
import numpy as np
hicarn = np.load('/home/yangz/data/hic_data/HiCARN/hicarn_predictions/${{chr}}/predictions_norm.npy')
refined = np.load('refined_${{chr}}/refined_norm.npy')

# 正确的最终结果
final = hicarn.squeeze() {sign} refined.squeeze()
np.save('refined_${{chr}}/final_correct.npy', final)
print(f'${{chr}}: saved final_correct.npy')
"
done
""")
    else:
        print("""
问题可能在更深层:
1. 检查 sample_combined.py 中的 scheduler 配置
2. 确认 parameterization='v' 在推理时正确处理
3. 检查模型输出是 x0 还是 v 还是 eps
""")
    
    print("\n" + "="*80)
    print("完成诊断")
    print("="*80)


if __name__ == '__main__':
    main()
