# torch_model.py
import torch
import torch.nn as nn

EPS = 1e-8


class UnfoldedComplexGivensDemixer(nn.Module):
    """
    2ch demixer via complex Givens (unitary) rotation per frequency:

      y1 =  cosθ * x1 + e^{jφ} sinθ * x2
      y2 = -e^{-jφ} sinθ * x1 + cosθ * x2

    θ(f) controls "direction" mixing
    φ(f) gives phase-aware rotation (KEY: fixes "no phase awareness")

    Update θ,φ iteratively (deep unfolding) using cross-covariance signal.
    Add explicit spatial constraint by smoothing update signals across frequency.
    """

    def __init__(self, F: int, K: int = 8, theta_lr: float = 0.15, phi_lr: float = 0.05, smooth_ks: int = 9):
        super().__init__()
        self.F = F
        self.K = K

        # Learnable step sizes per iteration
        self.alpha_theta = nn.Parameter(torch.ones(K) * float(theta_lr))
        self.alpha_phi = nn.Parameter(torch.ones(K) * float(phi_lr))

        # Learnable per-frequency gain on update
        self.beta = nn.Parameter(torch.ones(F))

        # Explicit spatial constraint: smooth update signals along frequency
        # (fixed low-pass initially; still learnable weights)
        pad = smooth_ks // 2
        self.smooth = nn.Conv1d(1, 1, kernel_size=smooth_ks, padding=pad, bias=False)
        with torch.no_grad():
            self.smooth.weight.fill_(1.0 / smooth_ks)

    @staticmethod
    def _apply_rotation(x1, x2, theta, phi):
        """
        x1,x2: (B,F,T) complex
        theta,phi: (B,F) float
        returns y1,y2: (B,F,T) complex
        """
        ct = torch.cos(theta)[:, :, None]  # (B,F,1)
        st = torch.sin(theta)[:, :, None]
        ejphi = torch.exp(1j * phi)[:, :, None]  # complex (B,F,1)

        y1 = ct * x1 + ejphi * st * x2
        y2 = -torch.conj(ejphi) * st * x1 + ct * x2
        return y1, y2

    def forward(self, X: torch.Tensor):
        """
        X: (B,2,F,T) complex64/complex32
        returns Y: (B,2,F,T) complex
        """
        B, C, F, T = X.shape
        assert C == 2 and F == self.F, f"Expected X (B,2,{self.F},T), got {tuple(X.shape)}"

        x1 = X[:, 0]  # (B,F,T)
        x2 = X[:, 1]

        theta = torch.zeros((B, F), device=X.device, dtype=torch.float32)
        phi = torch.zeros((B, F), device=X.device, dtype=torch.float32)

        for k in range(self.K):
            y1, y2 = self._apply_rotation(x1, x2, theta, phi)

            # Cross-covariance per frequency: sum_t y1 * conj(y2)
            cross = torch.sum(y1 * torch.conj(y2), dim=-1)  # (B,F) complex

            # Total power (stabilizer)
            pwr = torch.sum((y1.real**2 + y1.imag**2) + (y2.real**2 + y2.imag**2), dim=-1) + EPS  # (B,F)

            # Update signals:
            # - imag(cross) drives "direction" decorrelation (classic)
            # - real(cross) drives phase alignment (adds phase awareness)
            g_theta = (cross.imag / pwr).float()  # (B,F)
            g_phi = (cross.real / pwr).float()    # (B,F)

            # Explicit spatial smoothness across frequency
            g_theta = self.smooth(g_theta[:, None, :]).squeeze(1)  # (B,F)
            g_phi = self.smooth(g_phi[:, None, :]).squeeze(1)

            # Per-frequency learned scaling
            g_theta = g_theta * self.beta[None, :]
            g_phi = g_phi * self.beta[None, :]

            # Unfolded updates
            theta = theta - self.alpha_theta[k] * g_theta
            phi = phi - self.alpha_phi[k] * g_phi

            # Keep phi bounded (avoid wild phase)
            phi = torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi

        y1, y2 = self._apply_rotation(x1, x2, theta, phi)
        Y = torch.stack([y1, y2], dim=1)  # (B,2,F,T)
        return Y

    def unitary_penalty(self):
        """
        For this parameterization, the 2x2 matrix is unitary by construction,
        but numeric drift / training can still cause mild instability.
        Small regularizer helps keep updates stable.
        """
        # Penalize large step sizes (stability)
        return 1e-4 * (torch.mean(self.alpha_theta**2) + torch.mean(self.alpha_phi**2))
