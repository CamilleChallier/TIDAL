"""
Cosine DDIM noise schedule (``CosineDDIMScheduler``).
Precomputes α, σ, and SNR tables for T timesteps and exposes ``add_noise``,
``get_velocity``, and ``step`` (DDIM update) methods used by BaseDiffusionModel.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# 1.  Noise schedule
# ---------------------------------------------------------------------------

class NoiseSchedule(nn.Module):
    """
    Cosine or linear beta schedule (Ho et al. 2020).
    All buffers live on the same device as the model.
    Supports spatial latents of shape (B, C, D, H, W).
    """

    def __init__(
        self,
        T:          int   = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 0.02,
        beta_max:   float = 0.999,
        schedule:   str   = "cosine",
        s:          float = 0.008,
    ):
        super().__init__()
        self.T = T

        if schedule == "cosine":
            steps          = torch.arange(T + 1, dtype=torch.float64)
            f              = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar_full = (f / f[0]).float()
            alpha_bar      = alpha_bar_full[1:]
            alpha_bar_prev = alpha_bar_full[:-1]
            betas          = (1 - alpha_bar / alpha_bar_prev).clamp(0, beta_max)
        else:
            betas          = torch.linspace(beta_start, beta_end, T)
            alpha_bar      = torch.cumprod(1.0 - betas, dim=0)
            alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]], dim=0)

        alphas = 1.0 - betas

        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar",          alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar",(1.0 - alpha_bar).sqrt())
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar),
        )

    def _broadcast(self, coef: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return coef.view(-1, *([1] * (x.ndim - 1)))

    def q_sample(
        self,
        z0:  torch.Tensor,
        tau: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: z_τ = √ᾱ_τ · z₀ + √(1−ᾱ_τ) · ε"""
        if eps is None:
            eps = torch.randn_like(z0)
        sqrt_ab  = self._broadcast(self.sqrt_alpha_bar[tau],            z0)
        sqrt_1ab = self._broadcast(self.sqrt_one_minus_alpha_bar[tau],  z0)
        return sqrt_ab * z0 + sqrt_1ab * eps, eps

    def p_mean(
        self,
        z_tau:    torch.Tensor,
        tau:      torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM posterior mean μ_θ(z_τ, τ)."""
        beta_t   = self._broadcast(self.betas[tau],                     z_tau)
        alpha_t  = self._broadcast(self.alphas[tau],                    z_tau)
        sqrt_1ab = self._broadcast(self.sqrt_one_minus_alpha_bar[tau],  z_tau)
        return (z_tau - beta_t / sqrt_1ab * eps_pred) / alpha_t.sqrt()

    @torch.no_grad()
    def ddpm_step(self, z_tau, tau_idx, eps_pred):
        tau  = torch.full((z_tau.shape[0],), tau_idx - 1,
                          device=z_tau.device, dtype=torch.long)
        mean = self.p_mean(z_tau, tau, eps_pred)
        if tau_idx > 1:
            std = self._broadcast(self.posterior_variance[tau].sqrt(), z_tau)
            return mean + std * torch.randn_like(z_tau)
        return mean

    @torch.no_grad()
    def ddim_step(self, z_tau, tau_idx, tau_prev_idx, eps_pred, eta=0.0, z0_clamp: float = 10.0):
        t, t_prev = tau_idx - 1, tau_prev_idx - 1
        ab_t      = self.alpha_bar[t]
        ab_t_prev = self.alpha_bar[t_prev] if t_prev >= 0 else torch.ones(1, device=z_tau.device)
        z0_pred   = (z_tau - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        z0_pred   = z0_pred.clamp(-z0_clamp, z0_clamp)   # prevent numerical explosion
        dir_coef  = (1 - ab_t_prev - eta**2 * (1 - ab_t) / (1 - ab_t_prev)).clamp(min=0).sqrt()
        return ab_t_prev.sqrt() * z0_pred + dir_coef * eps_pred

    # ------------------------------------------------------------------
    # v-prediction helpers  (Salimans & Ho 2022)
    # ------------------------------------------------------------------

    def get_v_target(
        self, z0: torch.Tensor, eps: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """Ground-truth v = √ᾱ·ε − √(1−ᾱ)·z₀."""
        sqrt_ab  = self._broadcast(self.sqrt_alpha_bar[tau],            z0)
        sqrt_1ab = self._broadcast(self.sqrt_one_minus_alpha_bar[tau],  z0)
        return sqrt_ab * eps - sqrt_1ab * z0

    def predict_z0_from_v(
        self, z_tau: torch.Tensor, tau: torch.Tensor, v_pred: torch.Tensor
    ) -> torch.Tensor:
        """Recover z₀ from v-prediction: z₀ = √ᾱ·z_τ − √(1−ᾱ)·v."""
        sqrt_ab  = self._broadcast(self.sqrt_alpha_bar[tau],            z_tau)
        sqrt_1ab = self._broadcast(self.sqrt_one_minus_alpha_bar[tau],  z_tau)
        return sqrt_ab * z_tau - sqrt_1ab * v_pred

    def predict_eps_from_v(
        self, z_tau: torch.Tensor, tau: torch.Tensor, v_pred: torch.Tensor
    ) -> torch.Tensor:
        """Recover ε from v-prediction (needed for DDIM update): ε = √ᾱ·v + √(1−ᾱ)·z_τ."""
        sqrt_ab  = self._broadcast(self.sqrt_alpha_bar[tau],            z_tau)
        sqrt_1ab = self._broadcast(self.sqrt_one_minus_alpha_bar[tau],  z_tau)
        return sqrt_ab * v_pred + sqrt_1ab * z_tau


class CosineDDIMScheduler(NoiseSchedule):
    """
    Cosine DDIM scheduler with an explicit beta_max cap (default 0.2).

    Compared to the base NoiseSchedule (which clamps to 0.999), setting
    beta_max=0.2 prevents extremely noisy steps at the tail of the schedule,
    which stabilises training on compact medical latents.

    Usage: CosineDDIMScheduler(T=1000, beta_max=0.2)
    """

    def __init__(self, T: int = 1000, beta_max: float = 0.2, s: float = 0.008):
        super().__init__(T=T, beta_max=beta_max, schedule="cosine", s=s)
        print(f"[CosineDDIMScheduler] T={T}  beta_max={beta_max}  "
              f"β₀={self.betas[0].item():.5f}  β_T={self.betas[-1].item():.5f}  "
              f"ᾱ_T={self.alpha_bar[-1].item():.4f}")