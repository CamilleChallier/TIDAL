"""
Output dataclasses for diffusion model forward passes.
``DiffusionOutput`` holds the denoised prediction, per-component loss values,
and optional auxiliary tensors (latent samples, context embeddings).
"""
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DiffusionOutput:
    """Unified return type for both training and inference."""

    # Always present
    generated_dvf:   List[torch.Tensor]
    generated_vols:  List[torch.Tensor]

    # Training-only losses (all optional, all scalars)
    ddpm_loss:       Optional[torch.Tensor] = None   # denoising MSE in latent space
    dvf_recon_loss:  Optional[torch.Tensor] = None   # low-τ NCC in DVF space
    vmorph_vol_loss: Optional[torch.Tensor] = None   # NCC(voxelmorph warp, target vol)
    vol_recon_loss:  Optional[torch.Tensor] = None   # NCC(warped ref vol, target vol)
    nav_corr_loss:   Optional[torch.Tensor] = None   # (1 - Pearson(dvf_mag, phi)) * weight
    nav_signal_loss: Optional[torch.Tensor] = None   # MSE(navigator_amp_pred, navigator_amp_gt) * weight
    # Non-loss outputs
    warped_vols:     Optional[List[torch.Tensor]] = None
    cond_features:   Optional[object] = None

    @property
    def is_training(self) -> bool:
        return self.ddpm_loss is not None

    def total_loss(
        self,
        w_ddpm:       float = 1.0,
        w_dvf_recon:  float = 0.5,
        w_vmorph_vol: float = 0.0,
        w_vol_recon:  float = 0.5,
    ) -> torch.Tensor:
        assert self.is_training

        def _safe(t):
            return t is not None and not (torch.isnan(t) or torch.isinf(t))

        loss = w_ddpm * self.ddpm_loss
        if _safe(self.dvf_recon_loss):  loss = loss + w_dvf_recon       * self.dvf_recon_loss
        if _safe(self.vmorph_vol_loss): loss = loss + w_vmorph_vol      * self.vmorph_vol_loss
        if _safe(self.vol_recon_loss):  loss = loss + w_vol_recon       * self.vol_recon_loss
        return loss