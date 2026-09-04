"""
Abstract base class for TIDAL VAE models.
Defines the ``BaseVAE`` interface (``encode``, ``decode``, ``forward``, ``loss_function``),
shared residual block modules (``ResBlock3d``, ``ResBlockUpsample3d``),
and the ``Batch`` / ``StepOutput`` dataclasses consumed by training scripts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import abc
from typing import Any, TypedDict
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _safe_groups(groups: int, channels: int) -> int:
    """Return the largest divisor of *channels* that is ≤ *groups*."""
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g

# ---------------------------------------------------------------------------
# Residual blocks  (shared by both models)
# ---------------------------------------------------------------------------
 
class ResBlock3d(nn.Module):
    """Standard 3-D ResNet block with optional stride-2 down-sampling."""
 
    def __init__(self, in_ch: int, out_ch: int,
                 downsample: bool = False, groups: int = 8):
        super().__init__()
        stride = 2 if downsample else 1
        self.norm1 = nn.GroupNorm(_safe_groups(groups, in_ch),  in_ch)
        self.conv1 = nn.Conv3d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_safe_groups(groups, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.skip = (
            nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False)
            if in_ch != out_ch or downsample else nn.Identity()
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)
 
 
class ResBlockUpsample3d(nn.Module):
    """Trilinear ×2 up-sample followed by a residual conv block."""
 
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(_safe_groups(groups, in_ch), in_ch)
        self.up    = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv1 = nn.Conv3d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_safe_groups(groups, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.skip  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.up(F.silu(self.norm1(x))))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)
 
 
class ResidualLayer3D(nn.Module):
    """Lightweight residual layer used inside VQ-VAE encoder / decoder."""

    def __init__(self, in_dim: int, h_dim: int, res_h_dim: int, groups: int = 8):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.GroupNorm(_safe_groups(groups, in_dim), in_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_dim, res_h_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_safe_groups(groups, res_h_dim), res_h_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(res_h_dim, h_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_block(x)


class ResidualStack3D(nn.Module):
    """Stack of *n_res_layers* :class:`ResidualLayer3D` blocks."""

    def __init__(self, in_dim: int, h_dim: int,
                 res_h_dim: int, n_res_layers: int, groups: int = 8):
        super().__init__()
        self.stack = nn.ModuleList(
            [ResidualLayer3D(in_dim, h_dim, res_h_dim, groups) for _ in range(n_res_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)

# ---------------------------------------------------------------------------
# Shared output contract
# ---------------------------------------------------------------------------

class StepOutput(TypedDict):
    """
    The dict every ``train_step`` / ``val_step`` must return.

    Fields
    ------
    loss        : scalar Tensor (or float for val) — total loss to minimise
    recon_loss  : reconstruction term (MSE / NCC)
    aux_loss    : regularisation term (KL / VQ embedding)
    metrics     : arbitrary extra scalars (kl, perplexity, kl_prior, …)
    """
    loss:       torch.Tensor
    recon_loss: torch.Tensor
    aux_loss:   torch.Tensor
    metrics:    dict[str, Any]


# (ref_volume, input_volume_list, current_volume_list, dvf_list, _meta)
Batch = tuple[
    torch.Tensor,        # ref_volume       (B, D, H, W)
    list[torch.Tensor],  # input_volume_list
    list[torch.Tensor],  # current_volume_list
    list[torch.Tensor],  # dvf_list          length = tp
    Any,                 # metadata (unused)
]

class BaseVAE(nn.Module, abc.ABC):
    """
    Common interface for MIA_VAE, VQVAE and CVAE.

    Attributes
    ----------
    horizon : int
        How many future timesteps this model handles per forward pass.
        Single-step models (MIA_VAE, VQVAE) set ``horizon = 1`` and expect
        the trainer to call ``train_step`` once per timestep.
        Multi-step models (CVAE) set ``horizon = tp`` and consume all
        timesteps in a single ``train_step`` call.
    """

    #: Sub-classes must declare this at class level *or* set it in __init__.
    horizon: int

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def train_step(self, batch: Batch, device: torch.device) -> StepOutput:
        """
        Perform a single training forward pass (no backward, no step).

        The trainer is responsible for:
            output = model.train_step(batch, device)
            output["loss"].backward()
            optimizer.step()

        Returns
        -------
        StepOutput  (always — including a Tensor ``loss`` for .backward())
        """

    @abc.abstractmethod
    def val_step(self, batch: Batch, device: torch.device) -> StepOutput:
        """
        Perform a forward pass without gradient tracking.

        The trainer wraps this in ``torch.no_grad()``.

        Returns
        -------
        StepOutput  (``loss`` may be a plain float here)
        """

    # ------------------------------------------------------------------
    # Convenience helpers available to all sub-classes
    # ------------------------------------------------------------------

    def set_kl_weight(self, weight: float) -> None:
        """
        Update the effective KL weight during annealing.
        Sub-classes override if they use a different attribute name.
        Default: sets ``self.beta`` (used by MIA_VAE).
        """
        self.beta = weight

    @staticmethod
    def _unpack_batch(
        batch: Batch, device: torch.device
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """
        Move all tensors to *device* and add channel dim to ref_volume.

        Returns
        -------
        ref_volume          : (B, 1, D, H, W)
        input_volume_list   : list of (B, D, H, W) — kept on CPU, move lazily
        current_volume_list : list of (B, D, H, W)
        dvf_list            : list of (B, 3, D, H, W)  on *device*
        """
        ref_volume, input_volume_list, current_volume_list, dvf_list, _ = batch
        ref_volume = ref_volume.unsqueeze(1).to(device)
        dvf_list   = [d.to(device) for d in dvf_list]
        current_volume_list = [v.to(device) for v in current_volume_list]
        input_volume_list = [v.to(device) for v in input_volume_list]
        return ref_volume, input_volume_list, current_volume_list, dvf_list

    @staticmethod
    def _accumulate(
        outputs: list[StepOutput],
    ) -> StepOutput:
        """Average a list of per-timestep StepOutputs into a single one."""
        n = len(outputs)
        loss       = sum(o["loss"]       for o in outputs) / n
        recon_loss = sum(o["recon_loss"] for o in outputs) / n
        aux_loss   = sum(o["aux_loss"]   for o in outputs) / n

        # Average any shared metric keys
        all_keys = set().union(*(o["metrics"] for o in outputs))
        metrics  = {
            k: sum(o["metrics"].get(k, 0.0) for o in outputs) / n
            for k in all_keys
        }
        return StepOutput(
            loss=loss, recon_loss=recon_loss,
            aux_loss=aux_loss, metrics=metrics,
        )