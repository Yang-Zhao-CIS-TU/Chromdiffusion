import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from Models.HiCARN_1 import Generator
from Models.HiCARN_1_Loss import GeneratorLoss
from Utils.SSIM import ssim
from math import log10
from Arg_Parser import root_dir

cs = np.column_stack

# ================================================================
# 🔹 PREPROCESSING - MATCHING DIFFUSION MODEL
# ================================================================

def ensure_nchw(x):
    """Ensure data is in NCHW format"""
    x = np.asarray(x)
    if x.ndim == 3:
        return x[:, None, :, :]
    elif x.ndim == 4 and x.shape[1] in [1,3]:
        return x
    elif x.ndim == 4 and x.shape[-1] in [1,3]:
        return np.transpose(x, (0,3,1,2))
    raise ValueError(f"Cannot convert to NCHW: shape={x.shape}")

class HiCPreprocessor:
    """
    Preprocessing matching diffusion model:
    - Log1p transformation
    - Robust normalization (median + IQR)
    - Clipping to [-5, 5]
    """
    def __init__(self, size=40):
        self.size = size
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        
    def fit(self, X, Y):
        """Fit preprocessor on training data"""
        X, Y = ensure_nchw(X), ensure_nchw(Y)
        
        # Log transform
        X_log = np.log1p(X)
        Y_log = np.log1p(Y)
        
        # Robust statistics (median + IQR)
        self.X_mean = np.median(X_log)
        self.X_std = np.percentile(X_log, 75) - np.percentile(X_log, 25) + 1e-8
        
        self.Y_mean = np.median(Y_log)
        self.Y_std = np.percentile(Y_log, 75) - np.percentile(Y_log, 25) + 1e-8
        
        print(f"\n{'='*80}")
        print(f"PREPROCESSING STATISTICS (Matching Diffusion Model)")
        print(f"{'='*80}")
        print(f"LR (Low Resolution):")
        print(f"  Log-median: {self.X_mean:.4f}")
        print(f"  Log-IQR: {self.X_std:.4f}")
        print(f"\nHR (High Resolution):")
        print(f"  Log-median: {self.Y_mean:.4f}")
        print(f"  Log-IQR: {self.Y_std:.4f}")
        print(f"{'='*80}\n")
        
    def preprocess(self, X, Y=None):
        """
        Preprocess data:
        1. Log1p transform
        2. Standardize using median and IQR
        3. Clip to [-5, 5]
        """
        X = ensure_nchw(X)
        
        # Log transform
        X_log = np.log1p(X)
        
        # Standardize
        Xn = (X_log - self.X_mean) / self.X_std
        
        # Clip
        Xn = np.clip(Xn, -5, 5).astype(np.float32)
        
        if Y is None:
            return Xn, None
        
        Y = ensure_nchw(Y)
        
        # Same for Y
        Y_log = np.log1p(Y)
        Yn = (Y_log - self.Y_mean) / self.Y_std
        Yn = np.clip(Yn, -5, 5).astype(np.float32)
        
        return Xn, Yn
    
    def postprocess(self, Yn):
        """
        Inverse preprocessing:
        1. Unclip (handled by clipping again)
        2. Unstandardize
        3. Inverse log1p (expm1)
        """
        # Clip back to safe range
        Yn = np.clip(Yn, -5, 5)
        
        # Unstandardize
        Ylog = Yn * self.Y_std + self.Y_mean
        
        # Inverse log1p
        Y = np.expm1(Ylog)
        
        # Ensure non-negative
        return np.maximum(Y, 0)

# ================================================================
# 🔹 CONFIGURATION
# ================================================================

def adjust_learning_rate(epoch):
    lr = 0.0003 * (0.1 ** (epoch // 30))
    return lr

# Data directory
data_dir = '/data/251021_HiC_Diffusion/NEW_mat_TK/GM12878/40x40/train_data_raw_ratio16.npz'

# Output directory
out_dir = os.path.join(root_dir, 'checkpoints_ty')
os.makedirs(out_dir, exist_ok=True)

datestr = time.strftime('%m_%d_%H_%M')
visdom_str = time.strftime('%m%d')

resos = '10kb40kb'
chunk = 40
stride = 40
bound = 201
pool = 'nonpool'
name = 'HiCARN_1'

num_epochs = 100
batch_size = 64

# Device
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print("CUDA available? ", torch.cuda.is_available())
print("Device being used: ", device)

# ================================================================
# 🔹 LOAD AND PREPROCESS DATA
# ================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

npz = np.load(data_dir)
data = npz['train_lr']      # LR (low-resolution)
target = npz['train_hr']  # HR (high-resolution)

print(f"Raw data loaded:")
print(f"  LR shape: {data.shape}")
print(f"  HR shape: {target.shape}")
print(f"  LR range: [{data.min():.2f}, {data.max():.2f}]")
print(f"  HR range: [{target.min():.2f}, {target.max():.2f}]")

# Ensure NCHW format
data = ensure_nchw(data)
target = ensure_nchw(target)

print(f"\nAfter NCHW conversion:")
print(f"  LR shape: {data.shape}")
print(f"  HR shape: {target.shape}")

# ================================================================
# 🔹 TRAIN/VALIDATION SPLIT
# ================================================================

num_samples = data.shape[0]
split_idx = int(num_samples * 0.8)

train_data_np = data[:split_idx]
train_target_np = target[:split_idx]

valid_data_np = data[split_idx:]
valid_target_np = target[split_idx:]

print(f"\nData split (80/20):")
print(f"  Training samples: {len(train_data_np)}")
print(f"  Validation samples: {len(valid_data_np)}")

# ================================================================
# 🔹 FIT PREPROCESSOR AND TRANSFORM
# ================================================================

# Initialize preprocessor
pre = HiCPreprocessor(size=40)

# Fit on training data
pre.fit(train_data_np, train_target_np)

# Preprocess training data
train_data_norm, train_target_norm = pre.preprocess(train_data_np, train_target_np)

# Preprocess validation data
valid_data_norm, valid_target_norm = pre.preprocess(valid_data_np, valid_target_np)

print(f"After preprocessing:")
print(f"  Train LR range: [{train_data_norm.min():.2f}, {train_data_norm.max():.2f}]")
print(f"  Train HR range: [{train_target_norm.min():.2f}, {train_target_norm.max():.2f}]")
print(f"  Valid LR range: [{valid_data_norm.min():.2f}, {valid_data_norm.max():.2f}]")
print(f"  Valid HR range: [{valid_target_norm.min():.2f}, {valid_target_norm.max():.2f}]")

# ================================================================
# 🔹 CONVERT TO TENSORS
# ================================================================

train_data = torch.tensor(train_data_norm, dtype=torch.float32)
train_target = torch.tensor(train_target_norm, dtype=torch.float32)

valid_data = torch.tensor(valid_data_norm, dtype=torch.float32)
valid_target = torch.tensor(valid_target_norm, dtype=torch.float32)

print(f"\nTensor shapes:")
print(f"  train_data: {train_data.shape}")
print(f"  train_target: {train_target.shape}")
print(f"  valid_data: {valid_data.shape}")
print(f"  valid_target: {valid_target.shape}")

# ================================================================
# 🔹 CREATE DATASETS AND DATALOADERS
# ================================================================

train_set = TensorDataset(train_data, train_target)
valid_set = TensorDataset(valid_data, valid_target)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Valid batches: {len(valid_loader)}")
print("="*80 + "\n")

# ================================================================
# 🔹 INITIALIZE MODEL
# ================================================================

print("="*80)
print("INITIALIZING MODEL")
print("="*80)

netG = Generator(num_channels=64).to(device)

n_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
print(f"Generator parameters: {n_params/1e6:.2f}M")

# Loss function
criterionG = GeneratorLoss().to(device)

# Optimizer
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)

print("="*80 + "\n")

# ================================================================
# 🔹 TRAINING LOOP
# ================================================================

ssim_scores = []
psnr_scores = []
mse_scores = []
mae_scores = []

best_ssim = 0

print("="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

for epoch in range(1, num_epochs + 1):
    run_result = {'nsamples': 0, 'g_loss': 0, 'g_score': 0}

    # Adjust learning rate
    alr = adjust_learning_rate(epoch)
    optimizerG = optim.Adam(netG.parameters(), lr=alr)

    # Clear gradients
    for p in netG.parameters():
        if p.grad is not None:
            del p.grad
    torch.cuda.empty_cache()

    # ================================================================
    # TRAINING
    # ================================================================
    
    netG.train()
    train_bar = tqdm(train_loader)
    for data_batch, target_batch in train_bar:
        batch_size_curr = data_batch.size(0)
        run_result['nsamples'] += batch_size_curr

        real_img = target_batch.to(device)
        z = data_batch.to(device)
        fake_img = netG(z)

        # Train generator
        netG.zero_grad()
        g_loss = criterionG(fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        run_result['g_loss'] += g_loss.item() * batch_size_curr

        train_bar.set_description(
            desc=f"[{epoch}/{num_epochs}] Loss_G: {run_result['g_loss'] / run_result['nsamples']:.4f}")
    
    train_gloss = run_result['g_loss'] / run_result['nsamples']
    train_gscore = run_result['g_score'] / run_result['nsamples']

    # ================================================================
    # VALIDATION
    # ================================================================
    
    valid_result = {'g_loss': 0, 'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    netG.eval()

    batch_ssims = []
    batch_mses = []
    batch_psnrs = []
    batch_maes = []

    valid_bar = tqdm(valid_loader)
    with torch.no_grad():
        for val_lr, val_hr in valid_bar:
            batch_size_curr = val_lr.size(0)
            valid_result['nsamples'] += batch_size_curr
            
            lr = val_lr.to(device)
            hr = val_hr.to(device)
            sr = netG(lr)

            # Loss
            g_loss = criterionG(sr, hr)
            valid_result['g_loss'] += g_loss.item() * batch_size_curr

            # Metrics (in normalized space)
            batch_mse = ((sr - hr) ** 2).mean()
            batch_mae = (abs(sr - hr)).mean()
            valid_result['mse'] += batch_mse * batch_size_curr
            
            batch_ssim = ssim(sr, hr)
            valid_result['ssims'] += batch_ssim * batch_size_curr
            
            valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
            valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
            
            valid_bar.set_description(
                desc=f"[Validation] PSNR: {valid_result['psnr']:.4f} dB SSIM: {valid_result['ssim']:.4f}")

            batch_ssims.append(valid_result['ssim'])
            batch_psnrs.append(valid_result['psnr'])
            batch_mses.append(batch_mse.item())
            batch_maes.append(batch_mae.item())

    # Record scores
    ssim_scores.append((sum(batch_ssims) / len(batch_ssims)))
    psnr_scores.append((sum(batch_psnrs) / len(batch_psnrs)))
    mse_scores.append((sum(batch_mses) / len(batch_mses)))
    mae_scores.append((sum(batch_maes) / len(batch_maes)))

    valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
    now_ssim = valid_result['ssim'].item()

    # Save best model
    if now_ssim > best_ssim:
        best_ssim = now_ssim
        print(f'\n✓ New best SSIM: {best_ssim:.6f}')
        best_ckpt_file = f'{datestr}_bestg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'
        
        # Save with preprocessor for inference
        save_dict = {
            'model_state_dict': netG.state_dict(),
            'preprocessor': pre,
            'epoch': epoch,
            'ssim': best_ssim,
            'config': {
                'resos': resos,
                'chunk': chunk,
                'stride': stride,
                'bound': bound,
                'pool': pool,
                'name': name
            }
        }
        torch.save(save_dict, os.path.join(out_dir, best_ckpt_file))
        print(f'  Saved: {best_ckpt_file}\n')

# ================================================================
# 🔹 SAVE FINAL MODEL
# ================================================================

final_ckpt_g = f'{datestr}_finalg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'

save_dict = {
    'model_state_dict': netG.state_dict(),
    'preprocessor': pre,
    'epoch': num_epochs,
    'ssim': now_ssim,
    'config': {
        'resos': resos,
        'chunk': chunk,
        'stride': stride,
        'bound': bound,
        'pool': pool,
        'name': name
    }
}
torch.save(save_dict, os.path.join(out_dir, final_ckpt_g))

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Best SSIM: {best_ssim:.6f}")
print(f"Final model saved: {final_ckpt_g}")
print(f"Checkpoints directory: {out_dir}")
print("="*80)

# ================================================================
# 🔹 OPTIONAL: SAVE METRICS
# ================================================================

# Uncomment to save training curves
# ssim_scores = np.array(ssim_scores)
# psnr_scores = np.array(psnr_scores)
# mse_scores = np.array(mse_scores)
# mae_scores = np.array(mae_scores)
#
# np.savetxt(f'valid_ssim_scores_{name}.txt', X=ssim_scores, delimiter=',')
# np.savetxt(f'valid_psnr_scores_{name}.txt', X=psnr_scores, delimiter=',')
# np.savetxt(f'valid_mse_scores_{name}.txt', X=mse_scores, delimiter=',')
# np.savetxt(f'valid_mae_scores_{name}.txt', X=mae_scores, delimiter=',')
