"""
Pixel-space temporal augmentations for navigator image sequences (B, C, T, H, W).
Applies random flip, intensity jitter, Gaussian noise, and frame drop during training.
``TemporalAugConfig`` and ``TemporalMaskConfig`` dataclasses control which
augmentations are active and their hyperparameters.

Config keys (all live under  augmentation.temporal  in your YAML):
─────────────────────────────────────────────────────────────────
  frame_drop:
    enabled:   true
    prob:      0.5        # probability of applying per batch item
    min_keep:  1          # minimum frames to keep

  spatial_mask:
    enabled:   true
    prob:      0.5
    mask_ratio: 0.30      # fraction of patches to zero out
    patch_size: 16        # square patch side (pixels)

  variable_frames:
    enabled:   true
    prob:      0.5
    min_frames: 1         # minimum number of real frames to keep
                          # missing frames are zero-padded on the LEFT
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FrameDropConfig:
    enabled:  bool  = False
    prob:     float = 0.5
    min_keep: int   = 1


@dataclass
class SpatialMaskConfig:
    enabled:    bool  = False
    prob:       float = 0.5
    mask_ratio: float = 0.30
    patch_size: int   = 16


@dataclass
class TemporalMaskConfig:
    enabled:    bool  = False
    prob:       float = 0.5
    mask_ratio: float = 0.33


@dataclass
class VariableFramesConfig:
    enabled:    bool  = False
    prob:       float = 0.5
    min_frames: int   = 1


@dataclass
class TemporalAugConfig:
    """Top-level container.  Instantiate from a dict with  .from_dict()."""
    frame_drop:      FrameDropConfig      = field(default_factory=FrameDropConfig)
    spatial_mask:    SpatialMaskConfig    = field(default_factory=SpatialMaskConfig)
    tube_mask:       SpatialMaskConfig    = field(default_factory=SpatialMaskConfig)
    variable_frames: VariableFramesConfig = field(default_factory=VariableFramesConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalAugConfig":
        """Build from a plain dict (e.g. loaded from YAML)."""
        if d is None:
            return cls()
        return cls(
            frame_drop      = FrameDropConfig     (**d.get("frame_drop",      {})),
            spatial_mask    = SpatialMaskConfig   (**d.get("spatial_mask",    {})),
            tube_mask       = SpatialMaskConfig   (**d.get("tube_mask",       {})),

            variable_frames = VariableFramesConfig(**d.get("variable_frames", {})),
        )

    def any_enabled(self) -> bool:
        return any([
            self.frame_drop.enabled,
            self.spatial_mask.enabled,
            self.tube_mask.enabled,
            self.variable_frames.enabled,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Random frame dropping  (pixel space, before backbone)
# ─────────────────────────────────────────────────────────────────────────────

def apply_frame_drop(
    Iseq:     torch.Tensor,
    cfg:      FrameDropConfig,
    training: bool = True,
) -> torch.Tensor:
    """
    Randomly zero-out entire 2-D frames in the sequence.

    Each batch item independently draws a number of frames to drop,
    always keeping at least `cfg.min_keep` frames.

    Args:
        Iseq:     (B, C, T, H, W)
        cfg:      FrameDropConfig
        training: no-op when False

    Returns:
        (B, C, T, H, W)  – dropped frames replaced with zeros
    """
    if not training or not cfg.enabled:
        return Iseq

    B, C, T, H, W = Iseq.shape
    # print(f"Applying frame_drop augmentation: prob={cfg.prob}, min_keep={cfg.min_keep}")
    out = Iseq.clone()

    for b in range(B):
        if torch.rand(1).item() > cfg.prob:
            continue

        # Count frames already entirely zero (e.g. from variable_frames)
        frame_norms = out[b].abs().sum(dim=(0, 2, 3))  # (T,)
        zero_mask   = frame_norms == 0
        n_active    = int((~zero_mask).sum().item())
        max_drop    = max(0, n_active - cfg.min_keep)

        if max_drop == 0:
            continue

        n_drop      = torch.randint(1, max_drop + 1, (1,)).item()
        active_idx  = (~zero_mask).nonzero(as_tuple=True)[0]
        drop_idx    = active_idx[torch.randperm(len(active_idx))[:n_drop]]
        out[b, :, drop_idx, :, :] = 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Spatial patch masking  (pixel space, before backbone)
# ─────────────────────────────────────────────────────────────────────────────

def apply_spatial_mask(
    Iseq:     torch.Tensor,
    cfg:      SpatialMaskConfig,
    training: bool = True,
) -> torch.Tensor:
    """
    Zero out random square patches inside each 2-D slice independently.
    Operates in pixel space, before the CNN backbone.

    Args:
        Iseq:     (B, C, T, H, W)
        cfg:      SpatialMaskConfig
        training: no-op when False

    Returns:
        (B, C, T, H, W)
    """
    if not training or not cfg.enabled:
        return Iseq

    B, C, T, H, W = Iseq.shape
    P = cfg.patch_size
    # print(f"Applying spatial_mask augmentation: prob={cfg.prob}, mask_ratio={cfg.mask_ratio}, patch_size={P}")
    out = Iseq.clone()

    # Number of non-overlapping patches along each spatial dimension
    n_ph = H // P
    n_pw = W // P
    n_patches = n_ph * n_pw
    n_mask_max = max(1, int(n_patches * cfg.mask_ratio))
    n_mask = torch.randint(1, n_mask_max + 1, (1,)).item()

    for b in range(B):
        if torch.rand(1).item() > cfg.prob:
            # print(f"Batch item {b}: skipping spatial_mask augmentation (random prob)")
            continue
        for t in range(T):
            idx = torch.randperm(n_patches)[:n_mask]
            # print(f"Batch item {b}, frame {t}: masking {n_mask} patches out of {n_patches}, indices {idx.tolist()}")
            for i in idx.tolist():
                ph = (i // n_pw) * P
                pw = (i  % n_pw) * P
                out[b, :, t, ph : ph + P, pw : pw + P] = 0.0

    return out

def apply_tube_mask(
    Iseq:     torch.Tensor,
    cfg:      SpatialMaskConfig,
    training: bool = True,
) -> torch.Tensor:
    """
    Tube masking: zero out the same random square patches across all T frames.
    The spatial mask is shared along the temporal dimension (VideoMAE-style).
    Operates in pixel space, before the CNN backbone.

    Args:
        Iseq:     (B, C, T, H, W)
        cfg:      SpatialMaskConfig
        training: no-op when False

    Returns:
        (B, C, T, H, W)  – same shape, same device, same dtype
    """
    if not training or not cfg.enabled:
        return Iseq
    
    # print(f"Applying tube_mask augmentation: prob={cfg.prob}, mask_ratio={cfg.mask_ratio}, patch_size={cfg.patch_size}")

    B, C, T, H, W = Iseq.shape
    P         = cfg.patch_size
    device    = Iseq.device
    n_ph      = H // P
    n_pw      = W // P
    n_patches = n_ph * n_pw

    # ── 1. Decide which batch items are active ─────────────────────────────
    # Shape: (B,)  –  tube is either on or off for the whole clip
    active = torch.rand(B, device=device) < cfg.prob
    if not active.any():
        return Iseq

    # ── 2. Sample a per-batch mask count in [1, n_mask_max] ───────────────
    n_mask_max = max(1, int(n_patches * cfg.mask_ratio))
    counts = torch.randint(1, n_mask_max + 1, (B,), device=device)  # (B,)
    counts = counts * active.long()

    # ── 3. Build a boolean patch mask  (B, n_patches) ─────────────────────
    scores     = torch.rand(B, n_patches, device=device)
    rank       = scores.argsort(dim=-1)                              # (B, n_patches)
    patch_mask = rank < counts.unsqueeze(-1)                         # (B, n_patches)

    # ── 4. Expand along T — same spatial mask for every frame ─────────────
    patch_mask = patch_mask.unsqueeze(1).expand(B, T, n_patches)     # (B, T, n_patches)

    # ── 5. Reshape to pixel space  (B, T, H, W) ───────────────────────────
    patch_mask_2d = patch_mask.reshape(B, T, n_ph, n_pw)
    pixel_mask    = patch_mask_2d.repeat_interleave(P, dim=-2) \
                                  .repeat_interleave(P, dim=-1)      # (B, T, H', W')
    pixel_mask    = pixel_mask[:, :, :H, :W]                         # (B, T, H, W)

    # ── 6. Apply mask ──────────────────────────────────────────────────────
    pixel_mask = pixel_mask.unsqueeze(1)                             # (B, 1, T, H, W)
    # print(f"Batch items with tube_mask applied: {active.nonzero(as_tuple=True)[0].tolist()}")
    return Iseq.masked_fill(pixel_mask, 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Variable number of past frames  (pixel space, zero-pad strategy)
# ─────────────────────────────────────────────────────────────────────────────

def apply_variable_frames(
    Iseq:       torch.Tensor,
    cfg:        VariableFramesConfig,
    training:   bool = True,
) -> torch.Tensor:
    """
    Randomly reduce the number of active past frames by zero-padding the
    OLDEST (left-most) frames.

    The model always receives T frames; "missing" frames are zeros so the
    architecture is unchanged (Option A from the discussion).

    E.g. with T=3, min_frames=1:
        3 frames kept → [f_{t-2}, f_{t-1}, f_t]      (no change)
        2 frames kept → [  zeros, f_{t-1}, f_t]
        1 frame  kept → [  zeros,   zeros, f_t]       (most recent only)

    The most recent frame (index T-1) is ALWAYS kept.

    Args:
        Iseq:     (B, C, T, H, W)
        cfg:      VariableFramesConfig
        training: no-op when False

    Returns:
        (B, C, T, H, W)
    """
    if not training or not cfg.enabled:
        return Iseq

    B, C, T, H, W = Iseq.shape
    # print(f"Applying variable_frames augmentation: prob={cfg.prob}, min_frames={cfg.min_frames}")
    # print(f"Input sequence shape: {Iseq.shape}")
    out = Iseq.clone()
    max_drop = max(0, T - cfg.min_frames)
    # print(f"Calculated max_drop={max_drop} (T={T}, min_frames={cfg.min_frames})")

    if max_drop == 0:
        return out

    for b in range(B):
        if torch.rand(1).item() > cfg.prob:
            # print(f"Batch item {b}: skipping augmentation (random prob)")
            continue
        # How many frames to keep for this sample (always includes the last one)
        n_keep = torch.randint(cfg.min_frames, T, (1,)).item()  # in [min_frames, T-1]
        n_zero = T - n_keep
        # print(f"Batch item {b}: keeping {n_keep} frames, zero-padding {n_zero} oldest frames")
        # Zero-pad the oldest n_zero frames
        out[b, :, :n_zero, :, :] = 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: apply all pixel-space augmentations in one call
# ─────────────────────────────────────────────────────────────────────────────

def apply_pixel_space_augmentations(
    Iseq:     torch.Tensor,
    cfg:      TemporalAugConfig,
    training: bool = True,
) -> torch.Tensor:
    """
    Sequentially applies all *pixel-space* augmentations (frame_drop,
    spatial_mask, variable_frames). 
    Call this on Iseq before passing it to the model.

    Args:
        Iseq:     (B, C, T, H, W)
        cfg:      TemporalAugConfig
        training: no-op when False

    Returns:
        (B, C, T, H, W)
    """
    # print(f"Applying pixel-space augmentations with config: {cfg}")
    Iseq = apply_variable_frames(Iseq, cfg.variable_frames, training)
    Iseq = apply_spatial_mask(Iseq,    cfg.spatial_mask,    training)
    Iseq = apply_tube_mask(Iseq,       cfg.tube_mask,       training)
    Iseq = apply_frame_drop(Iseq,      cfg.frame_drop,      training)
    return Iseq