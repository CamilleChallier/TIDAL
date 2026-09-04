"""
DVFVAE: spatial β-VAE for DVF volumes (Stage 1 of the TIDAL pipeline).
A residual encoder-decoder with GroupNorm compresses (B, 3, D, H, W) DVF sequences
into a compact latent space. The combined loss includes reconstruction (MSE/NCC),
KL divergence, gradient smoothness, bending energy, and optional focal-frequency terms.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_VAE import BaseVAE, Batch, StepOutput, ResBlock3d, ResBlockUpsample3d
from ...utils.losses import (
    gradient_loss, bending_energy, FocalFrequencyLoss3D, ncc_as_loss,
    band_split_spectral_loss, band_energy_ratio_loss,
)


class DVFVAE(BaseVAE):
    """
    Spatial β-VAE for DVF volumes — residual/GroupNorm variant of VAE_rayan.
    - Input/output: (B, 3, D, H, W)
    - Latent: (B, latent_channels, D/2^depth, H/2^depth, W/2^depth)

    Encoder: a full-res ResBlock3d stage (channel projection only, no
             downsampling — same convention as VAE_rayan's first stage),
             followed by (depth-1) ResBlock3d(downsample=True) stages →
             mu, logvar heads.
    Decoder: mirrors the encoder — (depth-1) ResBlockUpsample3d stages
             (trilinear x2 upsample + residual conv block), followed by a
             full-res ResBlock3d stage → out_conv.

    Loss composition, config kwargs and train/val step interface are
    identical to VAE_rayan — only the conv-block internals changed
    (residual + GroupNorm instead of plain Conv3d + BatchNorm3d, and a
    progressive upsample-then-resblock decoder instead of bulk-interpolate
    followed by a plain conv stack). Spatial downsampling is D/2^(depth-1),
    same as VAE_rayan, so the latent shape matches for the same `depth`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,
        latent_channels: int = 8,
        depth: int = 4,
        beta: float = 1e-3,
        lam_smooth: float = 0.0,
        lam_ffl:          float = 0.0,
        ffl_alpha:        float = 1.0,
        horizon:          int   = 1,
        use_bending:      bool  = False,
        recon_loss_type:  str   = "mse",
        band_low_thresh:  float = 0.3,
        band_high_thresh: float = 0.6,
        lam_band_mid:     float = 0.0,
        lam_band_high:    float = 0.0,
        band_loss_type:   str   = "pointwise",   # pointwise | energy_ratio
        device:           torch.device = None,
        **kwargs,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        self.beta_target     = beta
        self.beta            = 0.0
        self.lam_smooth      = lam_smooth
        self.horizon         = horizon
        self.depth           = depth
        self.use_bending     = use_bending
        self.recon_loss_type = recon_loss_type
        self.band_low_thresh  = band_low_thresh
        self.band_high_thresh = band_high_thresh
        self.lam_band_mid     = lam_band_mid
        self.lam_band_high    = lam_band_high
        if band_loss_type not in ("pointwise", "energy_ratio"):
            raise ValueError(f"Unknown band_loss_type: {band_loss_type}")
        self.band_loss_type   = band_loss_type
        self.ffl = FocalFrequencyLoss3D(loss_weight=lam_ffl, alpha=ffl_alpha) if lam_ffl > 0 else None

        enc_blocks = []
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            enc_blocks.append(ResBlock3d(in_ch, out_ch, downsample=(i != 0)))
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_blocks)

        self.to_mu     = nn.Conv3d(in_ch, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv3d(in_ch, latent_channels, kernel_size=1)

        self.from_latent = nn.Conv3d(latent_channels, in_ch, kernel_size=1)

        dec_blocks = []
        for i in reversed(range(depth)):
            out_ch = base_channels * (2 ** i)
            if i != 0:
                dec_blocks.append(ResBlockUpsample3d(in_ch, out_ch))
            else:
                dec_blocks.append(ResBlock3d(in_ch, out_ch, downsample=False))
            in_ch = out_ch
        self.decoder  = nn.Sequential(*dec_blocks)
        self.out_conv = nn.Conv3d(in_ch, in_channels, kernel_size=3, padding=1)

    def set_kl_weight(self, weight: float) -> None:
        self.beta = weight

    def encode(self, x: torch.Tensor, cond_feats=None):
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        z = eps.mul(torch.exp(0.5 * logvar)).add(mu)
        return z

    def decode(self, z: torch.Tensor, cond_feats=None) -> torch.Tensor:
        x = self.from_latent(z)
        x = self.decoder(x)
        return self.out_conv(x)

    def forward(self, dvf: torch.Tensor) -> dict:
        mu, logvar = self.encode(dvf)
        z          = self.reparametrize(mu, logvar)
        recon      = self.decode(z)

        if self.recon_loss_type == "ncc":
            recon_loss = ncc_as_loss(recon, dvf, device=dvf.device)
            if torch.isnan(recon_loss):
                recon_loss = F.mse_loss(recon, dvf)
        else:
            recon_loss = F.mse_loss(recon, dvf)
        kl_loss     = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        smooth_loss = bending_energy(recon) if self.use_bending else gradient_loss(recon)
        ffl_loss    = self.ffl(recon, dvf) if self.ffl is not None else torch.tensor(0.0, device=dvf.device)

        if self.lam_band_mid > 0 or self.lam_band_high > 0:
            band_fn = band_energy_ratio_loss if self.band_loss_type == "energy_ratio" else band_split_spectral_loss
            band_loss, band_metrics = band_fn(
                recon, dvf,
                low_thresh=self.band_low_thresh,
                high_thresh=self.band_high_thresh,
                lambda_mid=self.lam_band_mid,
                lambda_high=self.lam_band_high,
            )
        else:
            band_loss, band_metrics = torch.tensor(0.0, device=dvf.device), {}

        loss        = recon_loss + self.beta * kl_loss + self.lam_smooth * smooth_loss + ffl_loss + band_loss
        return {
            "recon":      recon,
            "loss":       loss,
            "recon_loss": recon_loss,
            "aux_loss":   kl_loss,
            "metrics": {
                "kl":     kl_loss.detach(),
                "beta":   torch.tensor(self.beta),
                "smooth": smooth_loss.detach(),
                "ffl":    ffl_loss.detach(),
                **band_metrics,
            },
        }

    def train_step(self, batch: Batch, device: torch.device) -> StepOutput:
        _, _, _, dvf_list = self._unpack_batch(batch, device)
        outputs = [self.forward(dvf_list[t]) for t in range(self.horizon)]
        return self._accumulate(outputs)

    @torch.no_grad()
    def val_step(self, batch: Batch, device: torch.device) -> StepOutput:
        _, _, _, dvf_list = self._unpack_batch(batch, device)
        outputs = [self.forward(dvf_list[t]) for t in range(self.horizon)]
        return self._accumulate(outputs)
