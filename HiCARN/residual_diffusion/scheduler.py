"""
Diffusion Scheduler for Residual Diffusion

Implements DDPM (Denoising Diffusion Probabilistic Models) scheduler
for the forward and reverse diffusion processes.

Forward process: Δ_t = √α_t * Δ + √(1 - α_t) * ε
Reverse process: Sample Δ from p(Δ|x_HiCARN) iteratively
"""

import torch
import torch.nn.functional as F
import numpy as np


class DDPMScheduler:
    """
    DDPM Scheduler for residual diffusion
    
    Implements the noise schedule and sampling algorithms from:
    "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    """
    
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True,
        clip_sample_range=5.0
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        
        # Beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # Used in Stable Diffusion
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps
            ) ** 2
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Alpha schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
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
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, original_samples, noise, timesteps):
        """
        Forward diffusion: q(Δ_t | Δ_0)
        
        Δ_t = √α̅_t * Δ_0 + √(1 - α̅_t) * ε
        
        Args:
            original_samples: clean residual Δ (B, C, H, W)
            noise: sampled noise ε ~ N(0, I) (B, C, H, W)
            timesteps: timestep for each sample (B,)
        
        Returns:
            noisy_samples: Δ_t (B, C, H, W)
        """
        # Move to same device
        self.alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(original_samples.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(original_samples.device)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting (B, 1, 1, 1)
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = (
            sqrt_alpha_prod * original_samples +
            sqrt_one_minus_alpha_prod * noise
        )
        
        return noisy_samples
    
    def step(self, model_output, timestep, sample):
        """
        Single reverse diffusion step: p(Δ_{t-1} | Δ_t, x̃)
        
        Args:
            model_output: predicted noise ε_θ (B, C, H, W)
            timestep: current timestep t (int or Tensor)
            sample: current noisy residual Δ_t (B, C, H, W)
        
        Returns:
            prev_sample: denoised Δ_{t-1} (B, C, H, W)
            pred_original_sample: predicted Δ_0 (B, C, H, W)
        """
        # Move to same device
        device = sample.device
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], device=device)
        
        t = timestep
        
        # Predict original sample (Δ_0) from noise
        pred_original_sample = (
            sample - self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1) * model_output
        ) / self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # Clip if needed
        if self.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample,
                -self.clip_sample_range,
                self.clip_sample_range
            )
        
        # Compute posterior mean
        pred_original_sample_coef = self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1)
        current_sample_coef = self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1)
        
        pred_prev_sample = (
            pred_original_sample_coef * pred_original_sample +
            current_sample_coef * sample
        )
        
        # Add noise (not for t=0)
        variance = 0
        if t > 0:
            noise = torch.randn_like(sample)
            variance = (self.posterior_variance[t].reshape(-1, 1, 1, 1) ** 0.5) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample, pred_original_sample
    
    def get_variance(self, timestep):
        """Get variance at timestep"""
        return self.posterior_variance[timestep]


class DDIMScheduler(DDPMScheduler):
    """
    DDIM Scheduler for faster sampling
    
    Implements deterministic sampling from:
    "Denoising Diffusion Implicit Models" (Song et al., 2021)
    
    Allows sampling with fewer steps (e.g., 50 instead of 1000)
    """
    
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50, **kwargs):
        super().__init__(num_train_timesteps=num_train_timesteps, **kwargs)
        
        self.num_inference_steps = num_inference_steps
        
        # Create subset of timesteps for inference
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_train_timesteps, step_ratio).flip(0)
    
    def step(self, model_output, timestep, sample, eta=0.0):
        """
        DDIM reverse step
        
        Args:
            model_output: predicted noise ε_θ
            timestep: current timestep
            sample: current noisy sample
            eta: controls stochasticity (0 = deterministic, 1 = DDPM)
        
        Returns:
            prev_sample: Δ_{t-1}
            pred_original_sample: predicted Δ_0
        """
        device = sample.device
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], device=device)
        
        # Get previous timestep
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        prev_timestep = torch.clamp(prev_timestep, min=0)
        
        # Current and previous alpha_cumprod
        alpha_prod_t = self.alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].reshape(-1, 1, 1, 1)
        
        # Predicted original sample
        pred_original_sample = (
            sample - torch.sqrt(1 - alpha_prod_t) * model_output
        ) / torch.sqrt(alpha_prod_t)
        
        if self.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample,
                -self.clip_sample_range,
                self.clip_sample_range
            )
        
        # Direction pointing to x_t
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev) * model_output
        
        # DDIM deterministic step
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        
        # Add stochastic noise if eta > 0
        if eta > 0:
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            sigma = eta * torch.sqrt(variance)
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + sigma.reshape(-1, 1, 1, 1) * noise
        
        return prev_sample, pred_original_sample


if __name__ == "__main__":
    # Test scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # DDPM
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    
    # Test forward diffusion
    residual = torch.randn(4, 1, 40, 40).to(device)
    noise = torch.randn_like(residual)
    timesteps = torch.randint(0, 1000, (4,)).to(device)
    
    noisy_residual = scheduler.add_noise(residual, noise, timesteps)
    
    print(f"Original residual shape: {residual.shape}")
    print(f"Noisy residual shape: {noisy_residual.shape}")
    print(f"Timesteps: {timesteps}")
    
    # DDIM
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
    print(f"\nDDIM inference timesteps: {ddim_scheduler.timesteps[:10]}")
    print(f"Number of inference steps: {len(ddim_scheduler.timesteps)}")
