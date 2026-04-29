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


def adjust_learning_rate(epoch):
    lr = 0.0003 * (0.1 ** (epoch // 30))
    return lr


# data_dir: directory storing processed data
#data_dir = os.path.join(root_dir)
data_dir = '/home/yangz/data/hic_data/data_new/cr_train.npz'
# out_dir: directory storing checkpoint files
out_dir = os.path.join(root_dir, 'checkpoints_cr')
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

# whether using GPU for training
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print("CUDA available? ", torch.cuda.is_available())
print("Device being used: ", device)


npz = np.load(data_dir)
data = npz['data']      # 形状: (N, ...)
target = npz['target']  
data = np.clip(data, 0, 100) / 100.0

# target: >255 → 255, 然后 /255 → [0,1]
target = np.clip(target, 0, 255) / 255.0

# 5) 80/20 划分
num_samples = data.shape[0]
split_idx = int(num_samples * 0.8)

train_data_np = data[:split_idx]
train_target_np = target[:split_idx]

valid_data_np = data[split_idx:]
valid_target_np = target[split_idx:]
train_data = torch.tensor(train_data_np, dtype=torch.float32)
train_target = torch.tensor(train_target_np, dtype=torch.float32)

valid_data = torch.tensor(valid_data_np, dtype=torch.float32)
valid_target = torch.tensor(valid_target_np, dtype=torch.float32)

if train_data.ndim == 4:  # (N, H, W, C)
    train_data = train_data.permute(0, 3, 1, 2).contiguous()
if train_target.ndim == 4:
    train_target = train_target.permute(0, 3, 1, 2).contiguous()

if valid_data.ndim == 4:
    valid_data = valid_data.permute(0, 3, 1, 2).contiguous()
if valid_target.ndim == 4:
    valid_target = valid_target.permute(0, 3, 1, 2).contiguous()

print("train_data shape:", train_data.shape)     # 期望: (train_N, 1, 40, 40)
print("train_target shape:", train_target.shape) # 期望: (train_N, 1, 40, 40)

print("valid_data shape:", valid_data.shape)
print("valid_target shape:", valid_target.shape)
# 7) Dataset
train_set = TensorDataset(train_data, train_target)
valid_set = TensorDataset(valid_data, valid_target)

# 8) DataLoader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False)

print(f"Train size: {len(train_set)}, Valid size: {len(valid_set)}")
import torch
import numpy as np

def mmn(name, x):
    # works for numpy arrays or torch tensors
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.float()
    print(f"{name:>14s} | min={x.min().item():.6g}  max={x.max().item():.6g}  mean={x.mean().item():.6g}")

# before split (scaled arrays)
mmn("data (scaled)", data)
mmn("target (scaled)", target)

# after split/permutation (tensors)
mmn("train_data", train_data)
mmn("train_target", train_target)
mmn("valid_data", valid_data)
mmn("valid_target", valid_target)

# prepare training dataset
#train_file = os.path.join(data_dir, f'hicarn_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_train.npz')
#train_file = data_dir
#train = np.load(train_file)

#train_data = torch.tensor(train['data'], dtype=torch.float)
#train_target = torch.tensor(train['target'], dtype=torch.float)
#train_inds = torch.tensor(train['inds'], dtype=torch.long)

#train_set = TensorDataset(train_data, train_target)#, train_inds)

# prepare valid dataset
#valid_file = os.path.join(data_dir, f'hicarn_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_valid.npz')
#valid_file = data_dir
#valid = np.load(valid_file)

#valid_data = torch.tensor(valid['data'], dtype=torch.float)
#valid_target = torch.tensor(valid['target'], dtype=torch.float)
#valid_inds = torch.tensor(valid['inds'], dtype=torch.long)

#valid_set = TensorDataset(valid_data, valid_target, valid_inds)

# DataLoader for batched training
#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
#valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)

# load network
netG = Generator(num_channels=64).to(device)

# loss function
criterionG = GeneratorLoss().to(device)

# optimizer
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)

ssim_scores = []
psnr_scores = []
mse_scores = []
mae_scores = []

best_ssim = 0
for epoch in range(1, num_epochs + 1):
    run_result = {'nsamples': 0, 'g_loss': 0, 'g_score': 0}

    alr = adjust_learning_rate(epoch)
    optimizerG = optim.Adam(netG.parameters(), lr=alr)

    for p in netG.parameters():
        if p.grad is not None:
            del p.grad  # free some memory
    torch.cuda.empty_cache()

    netG.train()
    train_bar = tqdm(train_loader)
    for data, target in train_bar:
        batch_size = data.size(0)
        run_result['nsamples'] += batch_size

        real_img = target.to(device)
        z = data.to(device)
        fake_img = netG(z)

        ######### Train generator #########
        netG.zero_grad()
        g_loss = criterionG(fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        run_result['g_loss'] += g_loss.item() * batch_size

        train_bar.set_description(
            desc=f"[{epoch}/{num_epochs}] Loss_G: {run_result['g_loss'] / run_result['nsamples']:.4f}")
    train_gloss = run_result['g_loss'] / run_result['nsamples']
    train_gscore = run_result['g_score'] / run_result['nsamples']

    valid_result = {'g_loss': 0,
                    'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    netG.eval()

    batch_ssims = []
    batch_mses = []
    batch_psnrs = []
    batch_maes = []

    valid_bar = tqdm(valid_loader)
    with torch.no_grad():
        for val_lr, val_hr in valid_bar:
            batch_size = val_lr.size(0)
            valid_result['nsamples'] += batch_size
            lr = val_lr.to(device)
            hr = val_hr.to(device)
            sr = netG(lr)

            sr_out = sr
            hr_out = hr
            g_loss = criterionG(sr, hr)

            valid_result['g_loss'] += g_loss.item() * batch_size

            batch_mse = ((sr - hr) ** 2).mean()
            batch_mae = (abs(sr - hr)).mean()
            valid_result['mse'] += batch_mse * batch_size
            batch_ssim = ssim(sr, hr)
            valid_result['ssims'] += batch_ssim * batch_size
            valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
            valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
            valid_bar.set_description(
                desc=f"[Predicting in Test set] PSNR: {valid_result['psnr']:.4f} dB SSIM: {valid_result['ssim']:.4f}")

            batch_ssims.append(valid_result['ssim'])
            batch_psnrs.append(valid_result['psnr'])
            batch_mses.append(batch_mse)
            batch_maes.append(batch_mae)

    ssim_scores.append((sum(batch_ssims) / len(batch_ssims)))
    psnr_scores.append((sum(batch_psnrs) / len(batch_psnrs)))
    mse_scores.append((sum(batch_mses) / len(batch_mses)))
    mae_scores.append((sum(batch_maes) / len(batch_maes)))

    valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
    now_ssim = valid_result['ssim'].item()

    if now_ssim > best_ssim:
        best_ssim = now_ssim
        print(f'Now, Best ssim is {best_ssim:.6f}')
        best_ckpt_file = f'{datestr}_bestg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, best_ckpt_file))
final_ckpt_g = f'{datestr}_finalg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'


######### Uncomment to track scores across epochs #########
# ssim_scores = ssim_scores.cpu()
# psnr_scores = psnr_scores.cpu()
# mse_scores = mse_scores.cpu()
# mae_scores = mae_scores.cpu()

# ssim_scores = np.array(ssim_scores)
# psnr_scores = np.array(psnr_scores)
# mse_scores = np.array(mse_scores)
# mae_scores = np.array(mae_scores)
#
# np.savetxt(f'valid_ssim_scores_{name}', X=ssim_scores, delimiter=',')
# np.savetxt(f'valid_psnr_scores_{name}', X=psnr_scores, delimiter=',')
# np.savetxt(f'valid_mse_scores_{name}', X=mse_scores, delimiter=',')
# np.savetxt(f'valid_mae_scores_{name}', X=mae_scores, delimiter=',')

torch.save(netG.state_dict(), os.path.join(out_dir, final_ckpt_g))
