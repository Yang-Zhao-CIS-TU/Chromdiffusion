"""
Improved Diffusion Scheduler

Key improvements:
1. Support for v-parameterization (more stable than eps)
2. DDIM sampling for fewer steps and sharper peaks
3. Customizable timestep sampling strategies
4. Loss reweighting options
"""

import torch
import numpy as np
from typing import Optional, Union


class ImprovedDDPMScheduler:
    """
    Improved DDPM Scheduler with v-parameterization and DDIM support
    
    Parameterizations:
    - 'eps': Predict noise (standard DDPM)
    - 'v': Predict velocity (v = sqrt(alpha) * eps - sqrt(1-alpha) * x0)
           More stable, especially for residual refinement
    """
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_schedule='linear',
        beta_start=1e-4,
        beta_end=0.02,
        parameterization='eps',  # 'eps' or 'v'
        clip_sample=True,
        clip_sample_range=10.0
    ):
        self.num_train_timesteps = num_train_timesteps
        self.parameterization = parameterization
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        
        # Create beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # For v-parameterization
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For DDPM sampling
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # Timesteps for inference
        self.timesteps = None
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in Improved DDPM"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def set_timesteps(self, num_inference_steps, device='cpu', method='uniform'):
        """
        Set discrete timesteps for sampling
        
        Args:
            num_inference_steps: Number of denoising steps
            device: Device to use
            method: 'uniform' or 'quad' (quadratic spacing, better for DDIM)
        """
        if method == 'uniform':
            step_ratio = self.num_train_timesteps // num_inference_steps
            self.timesteps = torch.arange(0, num_inference_steps) * step_ratio
            self.timesteps = torch.flip(self.timesteps, dims=[0]).to(device)
        elif method == 'quad':
            # Quadratic spacing - more steps at high noise, fewer at low noise
            self.timesteps = (
                (torch.linspace(0, np.sqrt(self.num_train_timesteps * 0.8), num_inference_steps)) ** 2
            ).long()
            self.timesteps = torch.flip(self.timesteps, dims=[0]).to(device)
        else:
            raise ValueError(f"Unknown timestep method: {method}")
    
    def add_noise(self, x_start, noise, timesteps):
        """
        Add noise to clean samples (forward diffusion)
        
        Args:
            x_start: Clean samples [B, C, H, W]
            noise: Noise to add [B, C, H, W]
            timesteps: Timesteps [B]
        
        Returns:
            x_t: Noisy samples
        """
        # Move to same device as input
        sqrt_alpha_prod = self.sqrt_alphas_cumprod.to(x_start.device)[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[timesteps]
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(x_start.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(x_start.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        x_t = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        return x_t
    
    def get_v(self, x_start, noise, timesteps):
        """
        Compute velocity target for v-parameterization
        
        v = sqrt(alpha) * noise - sqrt(1-alpha) * x_start
        """
        # Move to same device as input
        sqrt_alpha_prod = self.sqrt_alphas_cumprod.to(x_start.device)[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[timesteps]
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(x_start.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(x_start.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        v = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * x_start
        return v
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise"""
        # Move to same device as input
        sqrt_recip = self.sqrt_recip_alphas_cumprod.to(x_t.device)[t]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod.to(x_t.device)[t]
        
        # Reshape for broadcasting
        sqrt_recip = sqrt_recip.flatten()
        while len(sqrt_recip.shape) < len(x_t.shape):
            sqrt_recip = sqrt_recip.unsqueeze(-1)
        
        sqrt_recipm1 = sqrt_recipm1.flatten()
        while len(sqrt_recipm1.shape) < len(x_t.shape):
            sqrt_recipm1 = sqrt_recipm1.unsqueeze(-1)
        
        pred_x0 = sqrt_recip * x_t - sqrt_recipm1 * noise
        
        if self.clip_sample:
            pred_x0 = torch.clamp(pred_x0, -self.clip_sample_range, self.clip_sample_range)
        
        return pred_x0
    
    def predict_start_from_v(self, x_t, t, v):
        """Predict x_0 from x_t and predicted velocity"""
        # Move to same device as input
        sqrt_alpha_prod = self.sqrt_alphas_cumprod.to(x_t.device)[t]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t]
        
        # Reshape
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(x_t.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(x_t.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        pred_x0 = sqrt_alpha_prod * x_t - sqrt_one_minus_alpha_prod * v
        
        if self.clip_sample:
            pred_x0 = torch.clamp(pred_x0, -self.clip_sample_range, self.clip_sample_range)
        
        return pred_x0
    
    def step(self, model_output, timestep, sample, eta=0.0, use_ddim=False):
        """
        Perform one denoising step
        
        Args:
            model_output: Output from model (noise or velocity)
            timestep: Current timestep
            sample: Current sample x_t
            eta: Amount of noise to add (0 = DDIM, 1 = DDPM)
            use_ddim: Use DDIM sampling (deterministic, fewer steps)
        
        Returns:
            prev_sample: Denoised sample x_{t-1}
            pred_x0: Predicted clean sample (for monitoring)
        """
        t = timestep
        
        # Predict x_0
        if self.parameterization == 'eps':
            pred_x0 = self.predict_start_from_noise(sample, t, model_output)
        elif self.parameterization == 'v':
            pred_x0 = self.predict_start_from_v(sample, t, model_output)
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
        
        if use_ddim:
            # DDIM sampling (deterministic)
            prev_sample = self._ddim_step(sample, t, pred_x0, eta)
        else:
            # DDPM sampling (stochastic)
            prev_sample = self._ddpm_step(sample, t, pred_x0)
        
        return prev_sample, pred_x0
    
    def _ddpm_step(self, x_t, t, pred_x0):
        """Standard DDPM sampling step"""
        # Get coefficients and move to device
        posterior_mean_coef1 = self.posterior_mean_coef1.to(x_t.device)[t]
        posterior_mean_coef2 = self.posterior_mean_coef2.to(x_t.device)[t]
        posterior_variance = self.posterior_variance.to(x_t.device)[t]
        
        # Reshape
        posterior_mean_coef1 = posterior_mean_coef1.flatten()
        while len(posterior_mean_coef1.shape) < len(x_t.shape):
            posterior_mean_coef1 = posterior_mean_coef1.unsqueeze(-1)
        
        posterior_mean_coef2 = posterior_mean_coef2.flatten()
        while len(posterior_mean_coef2.shape) < len(x_t.shape):
            posterior_mean_coef2 = posterior_mean_coef2.unsqueeze(-1)
        
        posterior_variance = posterior_variance.flatten()
        while len(posterior_variance.shape) < len(x_t.shape):
            posterior_variance = posterior_variance.unsqueeze(-1)
        
        # Compute posterior mean
        posterior_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x_t
        
        # Add noise
        if t > 0:
            noise = torch.randn_like(x_t)
            prev_sample = posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            prev_sample = posterior_mean
        
        return prev_sample
    
    def _ddim_step(self, x_t, t, pred_x0, eta=0.0):
        """
        DDIM sampling step (deterministic when eta=0)
        
        More stable and sharper for refinement tasks
        """
        # Move to device
        alpha_prod_t = self.alphas_cumprod.to(x_t.device)[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev.to(x_t.device)[t] if t > 0 else torch.tensor(1.0, device=x_t.device)
        
        # Reshape
        alpha_prod_t = alpha_prod_t.flatten()
        while len(alpha_prod_t.shape) < len(x_t.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        
        alpha_prod_t_prev = alpha_prod_t_prev.flatten() if isinstance(alpha_prod_t_prev, torch.Tensor) else alpha_prod_t_prev
        if isinstance(alpha_prod_t_prev, torch.Tensor):
            while len(alpha_prod_t_prev.shape) < len(x_t.shape):
                alpha_prod_t_prev = alpha_prod_t_prev.unsqueeze(-1)
        
        # Predict epsilon from x0
        pred_eps = (x_t - torch.sqrt(alpha_prod_t) * pred_x0) / torch.sqrt(1 - alpha_prod_t)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prod_t_prev - eta ** 2) * pred_eps
        
        # Random noise (eta controls amount)
        if t > 0 and eta > 0:
            noise = torch.randn_like(x_t)
            sigma_t = eta * torch.sqrt((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * torch.sqrt(1 - alpha_prod_t / alpha_prod_t_prev)
        else:
            noise = 0
            sigma_t = 0
        
        # Compute x_{t-1}
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        
        return prev_sample
    
    def get_timestep_weights(self, strategy='uniform', device='cpu'):
        """
        Get timestep sampling weights for training
        
        Args:
            strategy: 'uniform', 'snr' (emphasize mid-range), or 'late' (emphasize denoising)
        
        Returns:
            weights: Sampling weights for each timestep
        """
        if strategy == 'uniform':
            return torch.ones(self.num_train_timesteps, device=device)
        
        elif strategy == 'snr':
            # Emphasize middle timesteps (where SNR is moderate)
            # These are often most important for learning structure
            snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
            weights = 1.0 / (snr + 1)  # Lower weight for very low/high SNR
            weights = weights / weights.sum()
            return weights.to(device)
        
        elif strategy == 'late':
            # Emphasize late timesteps (low noise, important for refinement)
            weights = torch.linspace(0.5, 1.5, self.num_train_timesteps)
            weights = weights / weights.sum()
            return weights.to(device)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
