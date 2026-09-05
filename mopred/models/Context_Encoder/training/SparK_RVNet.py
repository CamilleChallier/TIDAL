"""
SparK-style masked pretraining wrapper for RV-Net in 3D.
Randomly masks 3D patches, propagates the binary mask through the encoder to
suppress masked positions, densifies with learned mask tokens, and reconstructs
via a lightweight hierarchical ConvTranspose3d decoder. Loss is per-patch
normalised MSE on masked patches only. After pretraining only RVNet is kept.

Inspired by SparK (Tian et al., "Designing BERT for Convolutional Networks:
Sparse and Hierarchical Masked Modeling", ICLR 2023).
https://github.com/keyu-tian/SparK
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight hierarchical decoder (3-D, dense)
# ---------------------------------------------------------------------------

class _LightDecoder3D(nn.Module):
    """
    Hierarchical ConvTranspose3d decoder.

    Takes a list of densified feature maps ordered coarsest → finest and
    reconstructs the input volume through successive ×2 upsampling stages
    with skip-connection additions from the finer scales.

    Channel schedule mirrors SparK: each stage halves the channel count.

    Parameters
    ----------
    dec_width  : channels at the coarsest scale (= densify_proj[0] output)
    in_chans   : output channels (= encoder input channels)
    num_scales : number of stages = nb_convs of the wrapped encoder
    """

    def __init__(self, dec_width: int, in_chans: int, num_scales: int):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        ch = dec_width
        for _ in range(num_scales):
            out_ch = max(ch // 2, 8)
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose3d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.ReLU(inplace=True),
            ))
            ch = out_ch
        self.head = nn.Conv3d(ch, in_chans, kernel_size=1)

    def forward(self, to_dec: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        to_dec : coarsest-first list, one tensor per encoder scale.
                 Each is already projected to the expected decoder width.
        """
        x = to_dec[0]
        for i, block in enumerate(self.up_blocks):
            x = block(x)
            if i + 1 < len(to_dec):
                x = x + to_dec[i + 1]
        return self.head(x)


# ---------------------------------------------------------------------------
# SparKRVNet
# ---------------------------------------------------------------------------

class SparKRVNet(nn.Module):
    """SparK-style masked pretraining for RVNet (3-D). See module docstring."""

    def __init__(
        self,
        encoder,
        vol_size:   Tuple[int, ...] = (32, 64, 64),
        mask_ratio: float = 0.6,
        patch_size: int   = 4,
        dec_width:  int   = 64,
    ):
        super().__init__()
        self.encoder    = encoder
        self.vol_size   = vol_size
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        D, H, W = vol_size
        if D % patch_size or H % patch_size or W % patch_size:
            raise ValueError(
                f"vol_size {vol_size} must be divisible by patch_size {patch_size}."
            )
        self.fmap_size = (D // patch_size, H // patch_size, W // patch_size)

        # ---- Decompose encoder into per-stride-2-block module lists ----------
        # Stored as plain Python lists so parameters are NOT double-registered;
        # they are already owned by self.encoder which IS an nn.Module.
        self._enc_block_lists: List[List[nn.Module]] = self._split_encoder_blocks()
        self.enc_out_chs: List[int] = [
            next(m.out_channels for m in reversed(blk) if isinstance(m, nn.Conv3d))
            for blk in self._enc_block_lists
        ]
        self.num_scales = len(self._enc_block_lists)

        # ---- Densify components (coarsest → finest order) -------------------
        e_widths_coarse_first = list(reversed(self.enc_out_chs))
        self.mask_tokens   = nn.ParameterList()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()

        d_width = dec_width
        for e_width in e_widths_coarse_first:
            tok = nn.Parameter(torch.zeros(1, e_width, 1, 1, 1))
            nn.init.trunc_normal_(tok, std=0.02)
            self.mask_tokens.append(tok)

            self.densify_norms.append(nn.GroupNorm(min(8, e_width), e_width))

            # Use a 3×3 projection when channel dimensions differ, 1×1 otherwise.
            ksz = 1 if e_width == d_width else 3
            self.densify_projs.append(
                nn.Conv3d(e_width, d_width, kernel_size=ksz, padding=ksz // 2, bias=True)
            )
            d_width = max(d_width // 2, 8)

        in_chans = next(
            m.in_channels for m in self.encoder.encoder if isinstance(m, nn.Conv3d)
        )
        self.decoder = _LightDecoder3D(dec_width, in_chans, self.num_scales)

    # ------------------------------------------------------------------
    # Encoder introspection
    # ------------------------------------------------------------------

    def _split_encoder_blocks(self) -> List[List[nn.Module]]:
        """
        Split the flat encoder Sequential into per-stride-2-block lists.

        Each list begins with a stride-2 Conv3d and collects all subsequent
        modules (norm, ReLU, stride-1 refine conv, norm, ReLU, optional
        Dropout3d) until the next stride-2 Conv3d starts a new group.
        """
        groups: List[List[nn.Module]] = []
        current: List[nn.Module] = []
        for m in self.encoder.encoder:
            if isinstance(m, nn.Conv3d) and m.stride[0] == 2 and current:
                groups.append(current)
                current = []
            current.append(m)
        if current:
            groups.append(current)
        return groups

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def _make_active_mask(self, B: int, device: torch.device) -> torch.Tensor:
        """
        Random patch-level active mask.

        Returns
        -------
        active : (B, 1, gD, gH, gW)  —  True = active (visible), False = masked
        """
        gD, gH, gW = self.fmap_size
        L      = gD * gH * gW
        n_keep = max(1, round(L * (1 - self.mask_ratio)))

        noise       = torch.rand(B, L, device=device)
        ids         = noise.argsort(dim=1)
        active_flat = torch.zeros(B, L, dtype=torch.bool, device=device)
        active_flat.scatter_(1, ids[:, :n_keep], True)

        return active_flat.view(B, 1, gD, gH, gW)

    # ------------------------------------------------------------------
    # Sparse-simulated encoding
    # ------------------------------------------------------------------

    def _encode_with_mask(
        self,
        x:          torch.Tensor,
        mask_voxel: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Run the RVNet backbone module-by-module.

        After every complete stride-2 block the feature map is multiplied by
        the downsampled binary mask, forcing masked positions to zero before
        the next block's receptive field can mix information across the
        mask boundary (sparse-conv simulation).

        Parameters
        ----------
        x          : (B, C, D, H, W)  — input already zeroed at masked patches
        mask_voxel : (B, 1, D, H, W)  — 1 = active, 0 = masked

        Returns
        -------
        feats : [feat_block0, ..., feat_block_{n-1}]  finest → coarsest
        """
        feat  = x
        feats: List[torch.Tensor] = []
        scale = 1

        for block_mods in self._enc_block_lists:
            for m in block_mods:
                feat = m(feat)
            scale   *= 2
            mask_s   = F.max_pool3d(mask_voxel.float(), kernel_size=scale, stride=scale)
            feat     = feat * mask_s
            feats.append(feat)

        return feats   # finest → coarsest

    # ------------------------------------------------------------------
    # Patchify / unpatchify (3-D)
    # ------------------------------------------------------------------

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W)  →  (B, gD·gH·gW, C·p³)"""
        B, C, D, H, W  = x.shape
        p               = self.patch_size
        gD, gH, gW      = D // p, H // p, W // p
        x = x.reshape(B, C, gD, p, gH, p, gW, p)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()   # (B, gD, gH, gW, p, p, p, C)
        return x.reshape(B, gD * gH * gW, C * p ** 3)

    def unpatchify(self, x: torch.Tensor, C: int) -> torch.Tensor:
        """(B, gD·gH·gW, C·p³)  →  (B, C, D, H, W)"""
        B          = x.shape[0]
        p          = self.patch_size
        gD, gH, gW = self.fmap_size
        x = x.reshape(B, gD, gH, gW, p, p, p, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()   # (B, C, gD, p, gH, p, gW, p)
        return x.reshape(B, C, gD * p, gH * p, gW * p)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:            torch.Tensor,
        active_b1fff: torch.Tensor | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        x            : (B, C, D, H, W)
        active_b1fff : (B, 1, gD, gH, gW) pre-computed active mask, optional.
                       True = visible, False = masked.  Random if None.

        Returns
        -------
        dict:
            loss   : SparK per-patch normalised MSE on masked patches (scalar)
            active : (B, 1, gD, gH, gW) active mask used
            recon  : (B, C, D, H, W) reconstructed volume
        """
        B, C, D, H, W = x.shape
        device = x.device

        # 1. Mask -------------------------------------------------------
        if active_b1fff is None:
            active_b1fff = self._make_active_mask(B, device)  # (B,1,gD,gH,gW)

        p          = self.patch_size
        mask_voxel = (
            active_b1fff
            .repeat_interleave(p, 2)
            .repeat_interleave(p, 3)
            .repeat_interleave(p, 4)
        )   # (B, 1, D, H, W) — 1=active, 0=masked

        # 2. Sparse-simulated encode ------------------------------------
        feats     = self._encode_with_mask(x * mask_voxel, mask_voxel)
        feats_rev = list(reversed(feats))   # coarsest → finest

        # 3. Densify: replace masked positions with learned tokens -------
        # active_b1fff is at patch-grid resolution (gD, gH, gW).
        # The coarsest encoder feature may be at a lower resolution — downsample
        # the mask to match before expanding into the densify loop.
        coarsest_shape = feats_rev[0].shape[2:]
        if active_b1fff.shape[2:] != coarsest_shape:
            scale = active_b1fff.shape[2] // coarsest_shape[0]
            cur_active = F.max_pool3d(
                active_b1fff.float(), kernel_size=scale, stride=scale
            ).bool()
        else:
            cur_active = active_b1fff
        to_dec: List[torch.Tensor] = []
        for i, feat in enumerate(feats_rev):
            feat = self.densify_norms[i](feat)
            tok  = self.mask_tokens[i].expand_as(feat)
            feat = torch.where(cur_active.expand_as(feat), feat, tok)
            feat = self.densify_projs[i](feat)
            to_dec.append(feat)
            # Dilate mask for the next (finer) scale
            cur_active = (
                cur_active
                .repeat_interleave(2, 2)
                .repeat_interleave(2, 3)
                .repeat_interleave(2, 4)
            )

        # 4. Decode -----------------------------------------------------
        recon = self.decoder(to_dec)   # (B, C, spatial...)
        if recon.shape[2:] != (D, H, W):
            recon = F.interpolate(
                recon, size=(D, H, W), mode='trilinear', align_corners=False
            )

        # 5. SparK loss: per-patch normalised MSE on masked patches -----
        inp_pat  = self.patchify(x)      # (B, L, C·p³)
        rec_pat  = self.patchify(recon)  # (B, L, C·p³)

        mean     = inp_pat.mean(dim=-1, keepdim=True)
        var      = (inp_pat.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        inp_norm = (inp_pat - mean) / var

        l2         = (rec_pat - inp_norm).pow(2).mean(dim=-1)   # (B, L)
        non_active = active_b1fff.logical_not().int().view(B, -1)  # (B, L)
        loss       = l2.mul(non_active).sum() / (non_active.sum() + 1e-8)

        return {"loss": loss, "active": active_b1fff, "recon": recon}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full RVNet forward on an unmasked volume. Returns (B, output_dim)."""
        return self.encoder(x)
