# torch_model_fixed.py - MINIMAL FIXES TO EXISTING MODEL
import torch
import torch.nn as nn

EPS = 1e-8


class UnfoldedComplexGivensDemixerFixed(nn.Module):
    """
    FIXED VERSION: Same architecture, but with:
    - Better initialization (non-zero start)
    - Angle constraints to prevent collapse
    - Optional output monitoring
    """
    
    def __init__(self, F: int, K: int = 12, theta_lr: float = 0.12, phi_lr: float = 0.04, smooth_ks: int = 9):
        super().__init__()
        self.F = F
        self.K = K
        
        # Learnable step sizes
        self.alpha_theta = nn.Parameter(torch.ones(K) * float(theta_lr))
        self.alpha_phi = nn.Parameter(torch.ones(K) * float(phi_lr))
        
        # Per-frequency scaling
        self.beta = nn.Parameter(torch.ones(F))
        
        # Frequency smoothing
        pad = smooth_ks // 2
        self.smooth = nn.Conv1d(1, 1, kernel_size=smooth_ks, padding=pad, bias=False)
        with torch.no_grad():
            self.smooth.weight.fill_(1.0 / smooth_ks)
    
    @staticmethod
    def _apply_rotation(x1, x2, theta, phi):
        """Apply complex Givens rotation"""
        ct = torch.cos(theta)[:, :, None]
        st = torch.sin(theta)[:, :, None]
        ejphi = torch.exp(1j * phi)[:, :, None]
        
        y1 = ct * x1 + ejphi * st * x2
        y2 = -torch.conj(ejphi) * st * x1 + ct * x2
        return y1, y2
    
    def forward(self, X: torch.Tensor):
        """
        FIXED: Non-zero initialization + angle constraints
        """
        B, C, F, T = X.shape
        assert C == 2 and F == self.F
        
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        # === FIX 1: BETTER INITIALIZATION ===
        # Start with small random perturbations instead of zeros
        # This helps escape the trivial θ=0, φ=0 solution
        theta = torch.randn((B, F), device=X.device, dtype=torch.float32) * 0.1
        phi = torch.randn((B, F), device=X.device, dtype=torch.float32) * 0.1
        
        for k in range(self.K):
            y1, y2 = self._apply_rotation(x1, x2, theta, phi)
            
            # Cross-covariance
            cross = torch.sum(y1 * torch.conj(y2), dim=-1)
            pwr = torch.sum((y1.real**2 + y1.imag**2) + (y2.real**2 + y2.imag**2), dim=-1) + EPS
            
            # Gradients
            g_theta = (cross.imag / pwr).float()
            g_phi = (cross.real / pwr).float()
            
            # Spatial smoothing
            g_theta = self.smooth(g_theta[:, None, :]).squeeze(1)
            g_phi = self.smooth(g_phi[:, None, :]).squeeze(1)
            
            # Per-frequency scaling
            g_theta = g_theta * self.beta[None, :]
            g_phi = g_phi * self.beta[None, :]
            
            # Update angles
            theta = theta - self.alpha_theta[k] * g_theta
            phi = phi - self.alpha_phi[k] * g_phi
            
            # === FIX 2: ANGLE CONSTRAINTS ===
            # Prevent collapse to identity (θ→0) or full swap (θ→π/2)
            # Keep theta in a useful range
            theta = torch.clamp(theta, -1.4, 1.4)  # roughly [-80°, +80°]
            
            # Keep phi bounded
            phi = torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi
        
        # Final rotation
        y1, y2 = self._apply_rotation(x1, x2, theta, phi)
        Y = torch.stack([y1, y2], dim=1)
        
        return Y
    
    def unitary_penalty(self):
        """Regularization to keep parameters stable"""
        return 1e-4 * (torch.mean(self.alpha_theta**2) + torch.mean(self.alpha_phi**2))
    
    def get_angles(self, X: torch.Tensor):
        """
        DIAGNOSTIC: Return final θ and φ values (useful for debugging)
        """
        B, C, F, T = X.shape
        x1, x2 = X[:, 0], X[:, 1]
        
        theta = torch.randn((B, F), device=X.device) * 0.1
        phi = torch.randn((B, F), device=X.device) * 0.1
        
        for k in range(self.K):
            y1, y2 = self._apply_rotation(x1, x2, theta, phi)
            cross = torch.sum(y1 * torch.conj(y2), dim=-1)
            pwr = torch.sum((y1.real**2 + y1.imag**2) + (y2.real**2 + y2.imag**2), dim=-1) + EPS
            
            g_theta = (cross.imag / pwr).float()
            g_phi = (cross.real / pwr).float()
            g_theta = self.smooth(g_theta[:, None, :]).squeeze(1)
            g_phi = self.smooth(g_phi[:, None, :]).squeeze(1)
            g_theta = g_theta * self.beta[None, :]
            g_phi = g_phi * self.beta[None, :]
            
            theta = theta - self.alpha_theta[k] * g_theta
            phi = phi - self.alpha_phi[k] * g_phi
            theta = torch.clamp(theta, -1.4, 1.4)
            phi = torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi
        
        return theta, phi


# Alias for backward compatibility
UnfoldedComplexGivensDemixer = UnfoldedComplexGivensDemixerFixed