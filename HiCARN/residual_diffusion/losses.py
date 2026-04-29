"""
Loss Functions for Residual Diffusion

Implements:
1. Diffusion loss (baseline): L_diff = ||ε - ε̂||²
2. Reconstruction loss (optional): L_recon = ||x̂ - x_GT||₁
3. Persistent Homology (PH) loss (advanced): Topology regularization

Training strategy:
- Phase 1 (baseline): L_diff only
- Phase 2 (stabilized): L_diff + λ₁ * L_recon
- Phase 3 (topology): L_diff + λ₁ * L_recon + λ₂ * L_PH
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiffusionLoss(nn.Module):
    """
    Baseline diffusion loss: MSE between predicted and true noise
    
    L_diff = ||ε - ε̂||²
    
    This is the standard DDPM loss.
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, noise_pred, noise_target):
        """
        Args:
            noise_pred: predicted noise from model (B, C, H, W)
            noise_target: ground truth noise (B, C, H, W)
        
        Returns:
            loss: scalar loss value
        """
        loss = F.mse_loss(noise_pred, noise_target, reduction=self.reduction)
        return loss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss: L1 distance in original space
    
    L_recon = ||x̂ - x_GT||₁
    
    where x̂ = x_pred + Δ̂
    
    Helps stabilize training by ensuring predicted residuals
    actually improve reconstruction.
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, x_reconstructed, x_target):
        """
        Args:
            x_reconstructed: pred + predicted_residual (B, C, H, W)
            x_target: ground truth (B, C, H, W)
        
        Returns:
            loss: scalar loss value
        """
        loss = F.l1_loss(x_reconstructed, x_target, reduction=self.reduction)
        return loss


class PersistenceImage:
    """
    Convert persistence diagrams to persistence images
    
    Persistence images are:
    - Differentiable
    - Stable
    - Easy to compute L2 loss on
    
    Reference: "Persistence Images: A Stable Vector Representation of 
    Persistent Homology" (Adams et al., 2017)
    """
    
    def __init__(
        self,
        resolution=(20, 20),
        bandwidth=1.0,
        birth_range=(0, 10),
        persistence_range=(0, 5)
    ):
        """
        Args:
            resolution: (height, width) of persistence image
            bandwidth: Gaussian kernel bandwidth
            birth_range: (min, max) range for birth time
            persistence_range: (min, max) range for persistence
        """
        self.resolution = resolution
        self.bandwidth = bandwidth
        self.birth_range = birth_range
        self.persistence_range = persistence_range
        
        # Create grid for persistence image
        self.x = np.linspace(birth_range[0], birth_range[1], resolution[1])
        self.y = np.linspace(persistence_range[0], persistence_range[1], resolution[0])
        self.X, self.Y = np.meshgrid(self.x, self.y)
    
    def transform(self, persistence_diagram):
        """
        Convert persistence diagram to persistence image
        
        Args:
            persistence_diagram: list of (birth, death) tuples
        
        Returns:
            image: (H, W) persistence image
        """
        image = np.zeros(self.resolution)
        
        for birth, death in persistence_diagram:
            if death == np.inf:
                continue
            
            persistence = death - birth
            
            # Weight by persistence
            weight = persistence
            
            # Add Gaussian centered at (birth, persistence)
            gaussian = weight * np.exp(
                -((self.X - birth)**2 + (self.Y - persistence)**2) / (2 * self.bandwidth**2)
            )
            image += gaussian
        
        return image


class TopologyExtractor:
    """
    Extract topological features from Hi-C matrices
    
    Converts Hi-C matrix → weighted graph → persistence diagram
    """
    
    def __init__(self, max_distance=10):
        self.max_distance = max_distance
    
    def hic_to_graph(self, hic_matrix):
        """
        Convert Hi-C matrix to weighted graph
        
        Args:
            hic_matrix: (H, W) contact matrix
        
        Returns:
            edges: list of (node1, node2, weight) tuples
        """
        # Threshold very small values
        hic_matrix = hic_matrix.copy()
        hic_matrix[hic_matrix < 0.01] = 0
        
        edges = []
        H, W = hic_matrix.shape
        
        for i in range(H):
            for j in range(i, W):
                if abs(i - j) <= self.max_distance and hic_matrix[i, j] > 0:
                    # Use contact strength as weight
                    # Or use 1/contact as distance
                    weight = hic_matrix[i, j]
                    edges.append((i, j, weight))
        
        return edges
    
    def compute_persistence(self, hic_matrix):
        """
        Compute persistence diagram from Hi-C matrix
        
        This is a placeholder - in practice, use libraries like:
        - ripser (fast, Python)
        - giotto-tda (scikit-learn style)
        - persim (for comparison)
        
        Returns:
            h0_diagram: H0 persistence diagram (connected components)
            h1_diagram: H1 persistence diagram (loops/cycles)
        """
        # PLACEHOLDER: This would use actual PH library
        # For now, return dummy diagrams
        
        # In real implementation:
        # from ripser import ripser
        # from gudhi import CubicalComplex
        # Or use filtration on weighted graph
        
        # Dummy example
        np.random.seed(42)
        n_points_h0 = 5
        n_points_h1 = 3
        
        h0_diagram = [(0, np.random.rand() * 5) for _ in range(n_points_h0)]
        h1_diagram = [(np.random.rand(), np.random.rand() * 3 + 1) for _ in range(n_points_h1)]
        
        return h0_diagram, h1_diagram


class PersistentHomologyLoss(nn.Module):
    """
    Persistent Homology Loss for topology regularization
    
    L_PH = ||PI_H0(x̂) - PI_H0(x_GT)||² + ||PI_H1(x̂) - PI_H1(x_GT)||²
    
    where PI is the persistence image representation.
    
    This encourages:
    - H0: Similar TAD-like domain structure (connected components)
    - H1: Similar loop structure (cycles)
    """
    
    def __init__(
        self,
        resolution=(20, 20),
        bandwidth=1.0,
        weight_h0=1.0,
        weight_h1=1.0
    ):
        super().__init__()
        
        self.pi_transformer = PersistenceImage(
            resolution=resolution,
            bandwidth=bandwidth
        )
        self.topology_extractor = TopologyExtractor()
        self.weight_h0 = weight_h0
        self.weight_h1 = weight_h1
    
    def forward(self, x_pred, x_target):
        """
        Args:
            x_pred: predicted Hi-C (B, C, H, W)
            x_target: ground truth Hi-C (B, C, H, W)
        
        Returns:
            loss: PH loss value
        """
        device = x_pred.device
        batch_size = x_pred.shape[0]
        
        total_loss = 0.0
        
        # Process each sample in batch
        for i in range(batch_size):
            # Extract 2D matrix
            pred_mat = x_pred[i, 0].detach().cpu().numpy()
            target_mat = x_target[i, 0].detach().cpu().numpy()
            
            # Compute persistence diagrams
            pred_h0, pred_h1 = self.topology_extractor.compute_persistence(pred_mat)
            target_h0, target_h1 = self.topology_extractor.compute_persistence(target_mat)
            
            # Convert to persistence images
            pred_pi_h0 = self.pi_transformer.transform(pred_h0)
            pred_pi_h1 = self.pi_transformer.transform(pred_h1)
            target_pi_h0 = self.pi_transformer.transform(target_h0)
            target_pi_h1 = self.pi_transformer.transform(target_h1)
            
            # Compute L2 loss
            loss_h0 = np.mean((pred_pi_h0 - target_pi_h0) ** 2)
            loss_h1 = np.mean((pred_pi_h1 - target_pi_h1) ** 2)
            
            total_loss += self.weight_h0 * loss_h0 + self.weight_h1 * loss_h1
        
        total_loss /= batch_size
        
        return torch.tensor(total_loss, device=device, requires_grad=False)


class CombinedLoss(nn.Module):
    """
    Combined loss for residual diffusion training
    
    L_total = L_diff + λ₁ * L_recon + λ₂ * L_PH
    
    Training phases:
    - Phase 1: λ₁ = 0, λ₂ = 0 (diffusion only)
    - Phase 2: λ₁ > 0, λ₂ = 0 (add reconstruction)
    - Phase 3: λ₁ > 0, λ₂ > 0 (add topology)
    """
    
    def __init__(
        self,
        lambda_recon=0.1,
        lambda_ph=0.01,
        use_ph=False
    ):
        """
        Args:
            lambda_recon: weight for reconstruction loss
            lambda_ph: weight for PH loss
            use_ph: whether to use PH loss (computationally expensive)
        """
        super().__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_ph = lambda_ph
        self.use_ph = use_ph
        
        self.diffusion_loss = DiffusionLoss()
        self.recon_loss = ReconstructionLoss()
        
        if use_ph:
            self.ph_loss = PersistentHomologyLoss()
        
        print(f"Combined loss initialized:")
        print(f"  λ_recon: {lambda_recon}")
        print(f"  λ_PH: {lambda_ph if use_ph else 'disabled'}")
    
    def forward(
        self,
        noise_pred,
        noise_target,
        x_reconstructed=None,
        x_target=None
    ):
        """
        Args:
            noise_pred: predicted noise
            noise_target: target noise
            x_reconstructed: reconstructed Hi-C (for recon & PH loss)
            x_target: target Hi-C (for recon & PH loss)
        
        Returns:
            total_loss: combined loss
            loss_dict: dictionary of individual losses
        """
        # Diffusion loss (always computed)
        loss_diff = self.diffusion_loss(noise_pred, noise_target)
        
        loss_dict = {'diffusion': loss_diff.item()}
        total_loss = loss_diff
        
        # Reconstruction loss (if weights provided)
        if self.lambda_recon > 0 and x_reconstructed is not None and x_target is not None:
            loss_recon = self.recon_loss(x_reconstructed, x_target)
            loss_dict['reconstruction'] = loss_recon.item()
            total_loss = total_loss + self.lambda_recon * loss_recon
        
        # PH loss (if enabled and weights provided)
        if self.use_ph and self.lambda_ph > 0 and x_reconstructed is not None and x_target is not None:
            loss_ph = self.ph_loss(x_reconstructed, x_target)
            loss_dict['ph'] = loss_ph.item()
            total_loss = total_loss + self.lambda_ph * loss_ph
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test losses
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test diffusion loss
    noise_pred = torch.randn(4, 1, 40, 40).to(device)
    noise_target = torch.randn(4, 1, 40, 40).to(device)
    
    diff_loss = DiffusionLoss()
    loss = diff_loss(noise_pred, noise_target)
    print(f"Diffusion loss: {loss.item():.6f}")
    
    # Test reconstruction loss
    x_recon = torch.randn(4, 1, 40, 40).to(device)
    x_target = torch.randn(4, 1, 40, 40).to(device)
    
    recon_loss = ReconstructionLoss()
    loss = recon_loss(x_recon, x_target)
    print(f"Reconstruction loss: {loss.item():.6f}")
    
    # Test combined loss
    combined_loss = CombinedLoss(lambda_recon=0.1, lambda_ph=0.01, use_ph=False)
    
    total_loss, loss_dict = combined_loss(
        noise_pred, noise_target,
        x_recon, x_target
    )
    
    print(f"\nCombined loss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
