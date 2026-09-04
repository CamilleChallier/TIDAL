"""
Stage-2 latent diffusion model for the TIDAL pipeline.
A 3D U-Net denoiser operates in the DVFVAE latent space, conditioned on TM-Net
temporal features (via FiLM or cross-attention) and the RV-Net reference embedding.
Supports DDPM/DDIM sampling, v-prediction, classifier-free guidance, and an optional
learned prior over the latent code.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ..Context_Encoder import TMNet_Tr_priormulti, TMNet_Tr_priormulti_mask, RVNet

from ...models import (
    SpatialTransformer,
    Voxelmorph,
)

from .base.base_model import BaseDiffusionModel
from .base.noise_schedule import CosineDDIMScheduler
from .base.outputs import DiffusionOutput

from ..Context_Encoder.temporal_augmentations import TemporalAugConfig, apply_pixel_space_augmentations, TemporalMaskConfig
from ...utils.navigator import Y_NAV as _Y_NAV
# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """τ (B,) → (B, emb_dim)"""

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        half = self.emb_dim // 2
        freq = torch.exp(
            -math.log(10_000) * torch.arange(half, device=tau.device) / (half - 1)
        )
        args = tau.float().unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Conditioning: FiLM and Cross-Attention
# ---------------------------------------------------------------------------

class FiLM3d(nn.Module):
    """
    Feature-wise Linear Modulation for 3-D feature maps.
    x ← x * (1 + scale) + shift, conditioned on a vector (B, cond_dim).
    Small normal init (not zero) so early training isn't dead.
    """

    def __init__(self, cond_dim: int, num_features: int, tanh_scale: bool = False):
        super().__init__()
        self.tanh_scale = tanh_scale
        self.proj = nn.Linear(cond_dim, num_features * 2)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        if self.tanh_scale:
            scale = torch.tanh(scale)
        scale = scale.view(scale.shape[0], -1, 1, 1, 1)
        shift = shift.view(shift.shape[0], -1, 1, 1, 1)
        if getattr(self, '_debug', False):
            print(f"  [FiLM ch={x.shape[1]:3d}] "
                  f"scale: mean={scale.abs().mean():.4f}  max={scale.abs().max():.4f} | "
                  f"shift: mean={shift.abs().mean():.4f}  max={shift.abs().max():.4f} | "
                  f"x_mod/x_orig ratio: {((x*(1+scale)+shift).std() / x.std().clamp(1e-6)):.4f}")
        return x * (1 + scale) + shift


class CrossAttention3d(nn.Module):
    """
    Spatial cross-attention: feature map (B,C,D,H,W) queries a single
    conditioning token (B, cond_dim).  Output projection is zero-init.
    """

    def __init__(self, num_features: int, cond_dim: int, n_heads: int = 4):
        super().__init__()
        assert num_features % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = num_features // n_heads
        self.scale    = self.head_dim ** -0.5

        self.norm   = nn.LayerNorm(num_features)
        self.to_q   = nn.Linear(num_features, num_features, bias=False)
        self.to_k   = nn.Linear(cond_dim,     num_features, bias=False)
        self.to_v   = nn.Linear(cond_dim,     num_features, bias=False)
        self.to_out = nn.Linear(num_features, num_features)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim) single token  — backward-compatible
        #    or (B, N, cond_dim) multi-token — used by ctx_mode="cross_attn_multi"
        B, C, D, H, W = x.shape
        N      = D * H * W
        x_flat = x.view(B, C, N).permute(0, 2, 1)       # (B, N, C)
        x_norm = self.norm(x_flat)

        tokens = cond if cond.dim() == 3 else cond.unsqueeze(1)   # (B, N_tok, cond_dim)
        q = self.to_q(x_norm)                            # (B, N, C)
        k = self.to_k(tokens)                            # (B, N_tok, C)
        v = self.to_v(tokens)

        def split_heads(t):
            b, s, _ = t.shape
            return t.view(b, s, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn    = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out     = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out     = self.to_out(out)
        return (x_flat + out).permute(0, 2, 1).view(B, C, D, H, W)


# ---------------------------------------------------------------------------
# Residual block with pluggable conditioning
# ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    """cond_type = 'film' | 'attn'"""

    def __init__(self, in_ch, out_ch, cond_dim, cond_type="film", groups=8, n_heads=4,
                 dropout=0.0, tanh_scale: bool = False):
        super().__init__()
        assert cond_type in ("film", "attn")
        self.cond_type = cond_type

        g_in  = self._safe_groups(groups, in_ch)
        g_out = self._safe_groups(groups, out_ch)

        self.norm1 = nn.GroupNorm(g_in,  in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.drop  = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)

        self.cond_layer = (FiLM3d(cond_dim, out_ch, tanh_scale=tanh_scale)
                           if cond_type == "film"
                           else CrossAttention3d(out_ch, cond_dim, n_heads=n_heads))
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    @staticmethod
    def _safe_groups(groups, channels):
        g = min(groups, channels)
        while channels % g != 0 and g > 1:
            g -= 1
        return g

    def forward(self, x, cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.cond_layer(h, cond)
        h = F.silu(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip(x)


def _select_swap_pairs(sim: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """
    Greedy non-overlapping matching of the `n_pairs` lowest-similarity token
    pairs from a (N, N) cosine-similarity matrix (diagonal pre-masked to a
    high value so tokens are never paired with themselves).

    Returns a permutation index tensor of shape (N,): identity everywhere
    except the selected pairs, whose indices are swapped.
    """
    N    = sim.shape[0]
    perm = torch.arange(N, device=sim.device)
    if n_pairs <= 0:
        return perm
    iu     = torch.triu_indices(N, N, offset=1, device=sim.device)
    order  = torch.argsort(sim[iu[0], iu[1]])           # ascending → most dissimilar first
    used   = torch.zeros(N, dtype=torch.bool, device=sim.device)
    n_done = 0
    for idx in order.tolist():
        i, j = iu[0, idx].item(), iu[1, idx].item()
        if not used[i] and not used[j]:
            perm[i], perm[j] = j, i
            used[i] = used[j] = True
            n_done += 1
            if n_done == n_pairs:
                break
    return perm


def _token_self_swap(x: torch.Tensor, swap_ratio: float, mode: str = "both") -> torch.Tensor:
    """
    Self-Swap perturbation (Zhang et al., "Guiding a Diffusion Model by
    Swapping Its Tokens", CVPR 2026): exchange the `swap_ratio` fraction of
    the most semantically-dissimilar token pairs — measured by cosine
    similarity between L2-normalised token vectors — in the spatial
    dimension (tokens = flattened D×H×W positions, each a C-dim vector),
    the channel dimension (tokens = channels, each a D·H·W-dim vector), or
    both jointly (paper reports best results combining the two).

    x: (B, C, D, H, W) feature map. Returns a perturbed copy of the same shape.
    """
    if swap_ratio <= 0.0:
        return x
    B, C, D, H, W = x.shape
    out = x

    if mode in ("spatial", "both"):
        flat    = out.reshape(B, C, -1).transpose(1, 2)        # (B, N, C) — tokens = positions
        unit    = F.normalize(flat, dim=-1)
        n_pairs = int(round(swap_ratio * flat.shape[1] / 2))
        rows = []
        for b in range(B):
            sim = unit[b] @ unit[b].transpose(0, 1)
            sim.fill_diagonal_(2.0)                              # never pair a token with itself
            rows.append(flat[b, _select_swap_pairs(sim, n_pairs)])
        out = torch.stack(rows, dim=0).transpose(1, 2).reshape(B, C, D, H, W)

    if mode in ("channel", "both"):
        flat    = out.reshape(B, C, -1)                          # (B, C, N) — tokens = channels
        unit    = F.normalize(flat, dim=-1)
        n_pairs = int(round(swap_ratio * C / 2))
        rows = []
        for b in range(B):
            sim = unit[b] @ unit[b].transpose(0, 1)
            sim.fill_diagonal_(2.0)
            rows.append(flat[b, _select_swap_pairs(sim, n_pairs)])
        out = torch.stack(rows, dim=0).reshape(B, C, D, H, W)

    return out


class SelfAttention3D(nn.Module):
    """Spatial self-attention over the full (D×H×W) token set with residual."""

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = channels // n_heads
        self.scale    = self.head_dim ** -0.5
        self.norm     = nn.GroupNorm(ResBlock3D._safe_groups(8, channels), channels)
        self.to_qkv   = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.to_out   = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # Self-Swap Guidance: toggled on/off around the network's two forward
        # passes during ssg_sample (see BaseDiffusionModel.ssg_sample). Swap is
        # applied to the block's *input*, before the residual is taken off, so
        # that both the attention path and the shortcut see perturbed tokens —
        # "at the beginning of each transformer block and before residual
        # shortcuts" (Zhang et al., CVPR 2026).
        self.ssg_enabled = False
        self.ssg_ratio   = 0.0
        self.ssg_mode    = "both"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ssg_enabled and self.ssg_ratio > 0.0:
            x = _token_self_swap(x, self.ssg_ratio, self.ssg_mode)
        B, C, D, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.to_qkv(h).chunk(3, dim=1)
        q = q.view(B, self.n_heads, self.head_dim, D * H * W)
        k = k.view(B, self.n_heads, self.head_dim, D * H * W)
        v = v.view(B, self.n_heads, self.head_dim, D * H * W)
        attn = torch.softmax(torch.einsum("bhcn,bhcm->bhnm", q, k) * self.scale, dim=-1)
        out  = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(B, C, D, H, W)
        return self.to_out(out) + x   # zero-init → neutral at init, learned residual


# ---------------------------------------------------------------------------
# Up / down-sampling
# ---------------------------------------------------------------------------

class Downsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Denoising U-Net
# ---------------------------------------------------------------------------

class DenoisingUNet3d(nn.Module):
    """ε_θ(z_τ, τ, c) → predicted noise, same shape as z_τ.

    Depth and channel widths are controlled by ch_mults, mirroring UNet3D_rayan.

    ch_mults=(1, 2)     → 1 downsampling level,  channels: base → base*2
    ch_mults=(1, 2, 4)  → 2 downsampling levels, channels: base → base*2 → base*4

    Migration from num_levels: num_levels=1 → ch_mults=(1, 2)
                                num_levels=2 → ch_mults=(1, 2, 4)

    Encoder skip connections are symmetric (same channel count as upsampled
    features from below), so each decoder cat doubles the channels before
    projecting back down to the next level width.
    """

    def __init__(
        self,
        latent_ch:      int   = 4,
        base_ch:        int   = 32,
        ch_mults:       tuple = (1, 2),
        cond_dim:       int   = 256,
        time_dim:       int   = 128,
        cond_type:      str   = "film",
        n_heads:        int   = 4,
        num_res_blocks: int   = 1,
        dropout:        float = 0.0,
        use_self_attn:  bool  = False,
        ctx_mode:       str   = "film",   # "film" | "concat" | "cross_attn" | "cross_attn_multi"
        use_bottleneck_attn: bool = False,
        self_attn_full_res: bool = False,
        use_phi_time_enc: bool = False,
    ):
        super().__init__()
        assert cond_type in ("film", "attn")
        assert ctx_mode  in ("film", "concat", "cross_attn", "cross_attn_multi")
        assert len(ch_mults) >= 2, "ch_mults needs at least 2 entries (e.g. (1, 2))"
        self.ctx_mode = ctx_mode
        self.use_phi_time_enc = use_phi_time_enc
        channels = [base_ch * m for m in ch_mults]
        n_downs  = len(channels) - 1

        def _blocks(first_in_ch, out_ch):
            return nn.ModuleList([
                ResBlock3D(
                    first_in_ch if i == 0 else out_ch, out_ch,
                    cond_dim, cond_type=cond_type, n_heads=n_heads, dropout=dropout,
                )
                for i in range(max(num_res_blocks, 1))
            ])

        def _attn(ch, at_full_res: bool = False):
            if use_self_attn and (not at_full_res or self_attn_full_res):
                print("Use self attention")
                return SelfAttention3D(ch, n_heads=n_heads)
            return nn.Identity()

        self.time_embed = SinusoidalEmbedding(time_dim)
        self.time_mlp   = nn.Sequential(
            nn.Linear(time_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # Phase φ injected into the time-embedding pathway so it modulates every
        # ResBlock via FiLM — prevents the "context-ignoring" local minimum.
        # phi_emb is pre-computed by UNet3D.predict_eps and passed in; no internal proj here.

        _use_cross = ctx_mode in ("cross_attn", "cross_attn_multi")

        if ctx_mode == "film":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(cond_dim * 2, cond_dim), nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.in_conv = nn.Conv3d(latent_ch, channels[0], 3, padding=1)
        elif ctx_mode == "concat":
            self.context_proj = nn.Linear(cond_dim, cond_dim)
            self.in_conv = nn.Conv3d(latent_ch + cond_dim, channels[0], 3, padding=1)
        else:  # cross_attn | cross_attn_multi — same modules, differ only in forward
            self.context_proj = nn.Linear(cond_dim, cond_dim)
            self.in_conv = nn.Conv3d(latent_ch, channels[0], 3, padding=1)

        # Encoder — one stage per level transition: channels[i] → channels[i+1]
        self.enc_blocks = nn.ModuleList()
        self.enc_attns  = nn.ModuleList()
        self.enc_cross_attns = nn.ModuleList() if _use_cross else None
        self.downs      = nn.ModuleList()
        for i in range(n_downs):
            self.enc_blocks.append(_blocks(channels[i], channels[i + 1]))
            self.enc_attns.append(_attn(channels[i + 1], at_full_res=(i == 0)))
            if _use_cross:
                self.enc_cross_attns.append(CrossAttention3d(channels[i + 1], cond_dim, n_heads=n_heads))
            self.downs.append(Downsample3D(channels[i + 1]))

        self.enc_alpf = nn.ModuleList([nn.Identity() for _ in range(n_downs)])

        # Bottleneck at channels[-1]
        self.mid1     = _blocks(channels[-1], channels[-1])
        self.mid_attn = (CrossAttention3d(channels[-1], cond_dim, n_heads=n_heads)
                        if use_bottleneck_attn else None)
        print(f"self.mid_attn: {self.mid_attn}")
        self.mid2     = _blocks(channels[-1], channels[-1])

        # Decoder — symmetric skips: cat(channels[i], channels[i]) → channels[i-1]
        rev = list(reversed(channels))
        self.ups         = nn.ModuleList()
        self.dec_blocks  = nn.ModuleList()
        self.dec_attns   = nn.ModuleList()
        self.dec_cross_attns = nn.ModuleList() if _use_cross else None
        self.skip_scales = nn.ParameterList()
        for i in range(n_downs):
            self.ups.append(Upsample3D(rev[i]))
            self.dec_blocks.append(_blocks(rev[i] * 2, rev[i + 1]))
            self.dec_attns.append(_attn(rev[i + 1], at_full_res=(i == n_downs - 1)))
            if _use_cross:
                self.dec_cross_attns.append(CrossAttention3d(rev[i + 1], cond_dim, n_heads=n_heads))
            self.skip_scales.append(nn.Parameter(torch.ones(1) * 0.5))

        self.skip_ahpf = nn.ModuleList([nn.Identity() for _ in range(n_downs)])

        self.out_conv = nn.Conv3d(channels[0], latent_ch, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.debug_shapes = False  # set via set_shape_debug(True)

    def set_shape_debug(self, enabled: bool = True) -> None:
        """Print every block's input/output tensor shape during forward()."""
        self.debug_shapes = enabled

    def _p(self, label: str, *tensors: torch.Tensor) -> None:
        if self.debug_shapes:
            shapes = ", ".join(str(tuple(t.shape)) for t in tensors)
            print(f"[UNet3D shape] {label:<28s} {shapes}")

    @staticmethod
    def _run(blocks, attn, x, c):
        for block in blocks:
            x = block(x, c)
        return attn(x)

    def set_self_swap(self, enabled: bool, ratio: float = 0.0, mode: str = "both") -> None:
        """Toggle Self-Swap Guidance perturbation on every SelfAttention3D block."""
        n = 0
        for m in self.modules():
            if isinstance(m, SelfAttention3D):
                m.ssg_enabled = enabled
                m.ssg_ratio   = ratio
                m.ssg_mode    = mode
                n += 1
        if enabled and n == 0:
            print(
                "[SSG] WARNING: set_self_swap(enabled=True) found 0 SelfAttention3D blocks. "
                "With ch_mults=(1,2) all attention slots are at full resolution and replaced "
                "with nn.Identity — SSG has no effect. Use ch_mults=(1,2,4) or add a "
                "bottleneck SelfAttention3D to enable SSG."
            )

    def forward(self, z_tau, tau, cond, phi_emb=None, phi_scalar=None):
        # cond: [B, cond_dim]    for film / concat / cross_attn
        #       [B, N, cond_dim] for cross_attn_multi  (N separate tokens)
        # phi_emb: (B, cond_dim) pre-computed by UNet3D.predict_eps from UNet3D.phi_proj.
        #   Injected additively into t_emb so phi modulates every ResBlock via FiLM.
        # phi_scalar: (B, 1) raw cardiac phase — when provided, applied as a broadcast
        #   multiplicative scale on the bottleneck (RMSim-style, use_phi_bottleneck_scale).
        self._p("input z_tau/cond", z_tau, cond)
        t_emb = self.time_mlp(self.time_embed(tau))
        if phi_emb is not None:
            t_emb = t_emb + phi_emb
        self._p("time_embed", t_emb)

        if self.ctx_mode == "film":
            c = self.fusion_mlp(torch.cat([cond, t_emb], dim=-1))
            x = self.in_conv(z_tau)
        elif self.ctx_mode == "concat":
            c = t_emb + self.context_proj(cond)
            D, H, W = z_tau.shape[2:]
            cond_spatial = cond[:, :, None, None, None].expand(-1, -1, D, H, W)
            x = self.in_conv(torch.cat([z_tau, cond_spatial], dim=1))
        else:  # cross_attn | cross_attn_multi
            # FiLM in ResBlocks uses the mean of tokens (or the single vector) + time.
            c_flat = cond.mean(dim=1) if cond.dim() == 3 else cond
            c = t_emb + self.context_proj(c_flat)
            x = self.in_conv(z_tau)
        self._p("fused context c", c)
        self._p("in_conv", x)

        _use_cross = self.ctx_mode in ("cross_attn", "cross_attn_multi")

        # Encoder — collect skips before each downsample
        skips = []
        enc_iter = zip(self.enc_blocks, self.enc_attns, self.downs, self.enc_alpf)
        for i, (enc, attn, down, alpf) in enumerate(enc_iter):
            x = self._run(enc, attn, x, c)
            self._p(f"enc[{i}] resblocks+attn", x)
            if self.enc_cross_attns is not None:
                x = self.enc_cross_attns[i](x, cond)
                self._p(f"enc[{i}] cross_attn", x)
            x = alpf(x)          # ← ALPF here, after ResBlocks, before downsample
            self._p(f"enc[{i}] alpf (skip)", x)
            skips.append(x)
            x = down(x)
            self._p(f"enc[{i}] downsample", x)

        # Bottleneck
        x = self._run(self.mid1, nn.Identity(), x, c)
        self._p("mid1", x)
        if self.mid_attn is not None:
            # print("use mid cross attn")
            x = self.mid_attn(x, cond if _use_cross else c)
        self._p("mid_attn", x)
        x = self._run(self.mid2, nn.Identity(), x, c)
        self._p("mid2", x)
        if phi_scalar is not None:
            x = x * (1.0 + phi_scalar.view(-1, 1, 1, 1, 1))

        # Decoder — symmetric cat then project down
        dec_iter = zip(self.dec_blocks, self.dec_attns, self.ups,
                       self.skip_scales, reversed(skips), self.skip_ahpf)
        for i, (dec, attn, up, scale, skip, ahpf) in enumerate(dec_iter):
            x = up(x)
            self._p(f"dec[{i}] upsample", x)
            sharp_skip = ahpf(skip * scale)    # ← AHPF on skip
            self._p(f"dec[{i}] skip (ahpf)", sharp_skip)
            x = self._run(dec, attn, torch.cat([x, sharp_skip], dim=1), c)
            self._p(f"dec[{i}] resblocks+attn", x)
            if self.dec_cross_attns is not None:
                x = self.dec_cross_attns[i](x, cond)
                self._p(f"dec[{i}] cross_attn", x)

        out = self.out_conv(x)
        self._p("out_conv", out)
        return out


# ---------------------------------------------------------------------------
# DVF encoder / decoder (deterministic — VAE lives outside this module)
# ---------------------------------------------------------------------------

class Encoder_DVF(nn.Module):
    """DVF (B,3,D,H,W) → spatial latent (B, latent_ch, D, H/2^n, W/2^n)."""

    def __init__(self, nb_convs, in_channels, out_channels, output_dim,
                 latent_ch=4, linear_input_dim=64, norm=nn.BatchNorm3d, dropout=False):
        super().__init__()
        assert nb_convs == len(out_channels)
        self.latent_ch = latent_ch

        layers = []
        for i in range(nb_convs):
            in_ch = in_channels if i == 0 else out_channels[i - 1]
            layers += [nn.Conv3d(in_ch, out_channels[i], 3, padding=1, stride=(1, 2, 2))]
            if norm: layers += [norm(out_channels[i], affine=True)]
            layers += [nn.ReLU(True),
                       nn.Conv3d(out_channels[i], out_channels[i], 3, padding=1)]
            if norm: layers += [norm(out_channels[i], affine=True)]
            layers += [nn.ReLU(True)]
            if dropout: layers += [nn.Dropout3d()]

        self.encoder = nn.Sequential(*layers)
        self.adap    = nn.Conv3d(out_channels[-1], latent_ch, 1, bias=False)
        nn.init.kaiming_normal_(self.adap.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, dvf):
        return self.adap(self.encoder(dvf))


class Decoder_DVF(nn.Module):
    """Spatial latent (B, latent_ch, ...) → DVF (B,3,D,H,W) with per-block FiLM."""

    def __init__(self, nb_convs, in_channels, out_channels, cond_dim=128,
                 norm=nn.BatchNorm3d, dropout=False, att=False):
        super().__init__()
        assert nb_convs == len(out_channels)

        self.film_layers = nn.ModuleList([
            FiLM3d(cond_dim, out_channels[i]) for i in range(nb_convs)
        ])
        self.blocks = nn.ModuleList()
        for i in range(nb_convs):
            in_ch = in_channels if i == 0 else out_channels[i - 1]
            block = [nn.ConvTranspose3d(in_ch, out_channels[i], (1,2,2), stride=(1,2,2))]
            if norm:    block += [norm(out_channels[i])]
            block += [nn.LeakyReLU(0.2),
                      nn.Conv3d(out_channels[i], out_channels[i], 3, padding=1)]
            if norm:    block += [norm(out_channels[i], affine=True)]
            block += [nn.LeakyReLU(0.2, inplace=True)]
            if dropout: block += [nn.Dropout3d()]
            self.blocks.append(nn.Sequential(*block))

        self.flow = nn.Conv3d(out_channels[-1], 3, 1)
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias   = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, z0, cond):
        x = z0
        for block, film in zip(self.blocks, self.film_layers):
            x = film(block(x), cond)
        return self.flow(x)


# ---------------------------------------------------------------------------
# Full latent diffusion model
# ---------------------------------------------------------------------------

class UNet3D(BaseDiffusionModel):

    # Set to True to print context norm balance and DVF stats each forward pass
    DEBUG = False

    def __init__(
        self,
        vae,
        num_frames:     int             = 3,
        horizon:        int             = 1,
        vol_size:       Tuple[int, ...] = (32, 64, 64),
        pre_latent_dim: int             = 16,
        Tr_n_heads:     int             = 8,
        Tr_enc_layers:  int             = 3,
        Tr_dec_layers:  int             = 3,
        condi_channels: List[int]       = (16, 32, 64),
        Tr_norm_before: bool            = True,
        condi_type:     str             = "2",
        prior_type:     str             = "learned",
        T:              int             = 1000,
        beta_max:       float           = 0.2,
        ddim_steps:     int             = 50,
        sampler:        str             = "ddim",
        unet_base_ch:   int             = 32,
        unet_time_dim:  int             = 128,
        unet_ch_mults:  tuple           = (1, 2),
        num_res_blocks: int             = 1,
        res_dropout:    float           = 0.0,
        use_self_attn:  bool            = False,
        cond_type:      str             = "film",
        cond_n_heads:   int             = 4,
        ctx_mode:       str             = "film",
        latent_ch:      int             = 4,
        cond_dim:       Optional[int]   = None,
        low_tau_frac:        float           = 0.25,
        snr_gamma:           float           = 5.0,
        use_spatial_weight:  bool            = False,
        spatial_weight_max:  float           = 5.0,
        cfg_scale:           float           = 1.0,
        ssg_scale:           float           = 1.0,
        ssg_swap_ratio:      float           = 0.1,
        ssg_mode:            str             = "both",
        predict_mode:        str             = "eps",
        temporal_aug: dict  | None = None,
        use_mask_cond:  bool            = False,
        dose_dropout_p: float           = 0.0,
        organ_bbox:     list | None     = None,
        organ_weight:   float           = 5.0,
        bg_weight:      float           = 0.1,
        use_h_t:        bool            = True,
        use_f_ref:      bool            = True,
        use_cond_proj:  bool            = True,
        image_mode:     bool            = False,
        use_bottleneck_attn: bool = False,
        self_attn_full_res: bool = False,
        recon_weight: float = 0.0,
        vmorph_weight: float = 0.0,
        zero_context: bool = False,
        use_phi_time_enc: bool = False,
        phi_prior_weight: float = 0.0,
        nav_corr_weight: float = 0.0,
        nav_corr_tau: int = 500,
        phi_reg_weight: float = 0.0,
        use_phi_bottleneck_scale: bool = False,
        inbatch_div_weight: float = 0.0,
        use_amplitude_scale: bool = False,
        nav_signal_weight: float = 0.0,
        ):

        super().__init__()
        self.vae                = vae
        self.horizon            = horizon
        self.horizon_idx        = None   # None = all; int = only that DDIM pass
        self.ddim_steps         = ddim_steps
        self.sampler            = sampler
        self.low_tau_frac       = low_tau_frac
        self.snr_gamma          = snr_gamma
        self.image_mode         = image_mode
        self.zero_context = zero_context

        self.use_spatial_weight = use_spatial_weight
        self.spatial_weight_max = spatial_weight_max
        self.cfg_scale          = cfg_scale
        self.ssg_scale          = ssg_scale          # Self-Swap Guidance scale ω (1.0 = disabled)
        self.ssg_swap_ratio     = ssg_swap_ratio     # fraction r of token pairs to swap
        self.ssg_mode           = ssg_mode           # "spatial" | "channel" | "both"
        self.predict_mode       = predict_mode
        self._latent_ch   = latent_ch
        self.dose_dropout_p = dose_dropout_p
        if temporal_aug is not None:
            # print(temporal_aug)
            self.temporal_aug_cfg = TemporalAugConfig.from_dict(temporal_aug)
        else :
            self.temporal_aug_cfg = None

        _cond_dim      = cond_dim if cond_dim is not None else max(128, 8 * pre_latent_dim)
        _need_cond_net = use_h_t
        _feat_streams  = (1 if use_f_ref else 0) + (1 if use_h_t else 0)
        if _feat_streams == 0:
            raise ValueError("At least one of use_f_ref, use_h_t must be enabled.")
        _proj_in          = _feat_streams * pre_latent_dim
        self.use_h_t          = use_h_t
        self.use_f_ref        = use_f_ref
        self._use_motion_in_c = False
        self.use_cond_proj    = use_cond_proj
        if use_cond_proj:
            self.cond_proj = nn.Sequential(
                nn.Linear(_proj_in, _cond_dim), nn.SiLU(),
                nn.Linear(_cond_dim, _cond_dim),
            )
            self.context_norm = nn.Identity()
        else:
            # No projection — c is the raw concatenation, FiLM sized to _proj_in.
            _cond_dim = _proj_in
            self.cond_proj    = nn.Identity()
            self.context_norm = nn.Identity()
        self._pre_latent_dim = pre_latent_dim
        self._num_frames     = num_frames

        # Respiratory state conditioning: encodes (amplitude, delta_amplitude) from the
        # last two past navigator frames — amplitude gives magnitude, delta gives direction
        # (positive = inhaling, negative = exhaling), together approximating breathing phase.
        # Zero-init on amplitude_proj so it starts as a no-op (checkpoint-compatible).
        if use_amplitude_scale:
            self.amplitude_enc = nn.Sequential(
                nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(32, 1), nn.Softplus(),
            )
            self.amplitude_proj = nn.Linear(2, _cond_dim)  # (amplitude, delta) → context
            nn.init.zeros_(self.amplitude_proj.weight)
            nn.init.zeros_(self.amplitude_proj.bias)
        else:
            self.amplitude_enc  = None
            self.amplitude_proj = None
        _need_ref_net  = use_f_ref

        import copy
        if _need_cond_net:
            if hasattr(vae, "cond_net"):
                self.cond_net = copy.deepcopy(vae.cond_net)
                print("Using VAE's existing cond_net for conditioning, masking:", isinstance(self.cond_net, TMNet_Tr_priormulti_mask))
            else:
                _cond_cls = TMNet_Tr_priormulti_mask if use_mask_cond else TMNet_Tr_priormulti
                self.cond_net = _cond_cls(
                    num_inputs=num_frames, horizon=horizon, in_channels=2,
                    out_channels=condi_channels, n_heads=Tr_n_heads,
                    enc_layers=Tr_enc_layers, dec_layers=Tr_dec_layers,
                    normalize_before=Tr_norm_before, output_dim=pre_latent_dim,
                    rnn="transformer", condi_type=condi_type, prior_type=prior_type, device=next(vae.parameters()).device,
                )
        else:
            self.cond_net = None

        if _need_ref_net:
            if hasattr(vae, "ref_net"):
                self.ref_net = copy.deepcopy(vae.ref_net)
            else:
                self.ref_net = RVNet(
                    nb_convs=len(condi_channels), in_channels=1,
                    out_channels=list(condi_channels), norm=nn.BatchNorm3d,
                    output_dim=pre_latent_dim, linear_input_dim=4 * 8 * 8,
                )
        else:
            self.ref_net = None
        self.motion_enc = None
        self.fourier_enc = None

        # For cross_attn_multi: separate linear projections per stream.
        # Each stream (f_ref, h_t, optionally f_motion) becomes its own token [B, 1, _cond_dim].
        # The denoising net then attends over all N tokens jointly.
        self._is_multi_token = (ctx_mode == "cross_attn_multi")
        if self._is_multi_token:
            self.ref_token_proj    = nn.Linear(pre_latent_dim, _cond_dim) if use_f_ref else None
            self.ht_token_proj     = nn.Linear(pre_latent_dim, _cond_dim) if use_h_t   else None
            self.motion_token_proj = None
            self.fourier_token_proj = None
            self.phi_token_proj     = None

        self.schedule      = CosineDDIMScheduler(T=T, beta_max=beta_max)
        self.denoising_net = DenoisingUNet3d(
            latent_ch=latent_ch, base_ch=unet_base_ch,
            cond_dim=_cond_dim, time_dim=unet_time_dim,
            cond_type=cond_type, n_heads=cond_n_heads,
            ch_mults=unet_ch_mults,
            num_res_blocks=num_res_blocks,
            dropout=res_dropout,
            use_self_attn=use_self_attn,
            ctx_mode=ctx_mode,
            use_bottleneck_attn=use_bottleneck_attn,
            self_attn_full_res=self_attn_full_res,
            use_phi_time_enc=use_phi_time_enc,
        )
        if not image_mode:
            self.spatial_transform = SpatialTransformer(vol_size)

        # Probe latent shape once at construction
        _z_in_ch = 1 if image_mode else 3
        vae.eval()
        with torch.no_grad():
            _z = vae.encode(torch.zeros(1, _z_in_ch, *vol_size).to(device=next(vae.parameters()).device))
            self._latent_shape = (_z[0] if isinstance(_z, tuple) else _z).shape[1:]

        self._init_latent_stats()   # registers latent_mean / latent_std buffers

        # Organ mask: up-weight the DDPM loss inside the bounding box.
        # Built at vol_size resolution then downsampled to latent resolution.
        if organ_bbox is not None:
            (d0, d1), (h0, h1), (w0, w1) = organ_bbox
            _mask = torch.full((1, 1, *vol_size), bg_weight)
            _mask[:, :, d0:d1, h0:h1, w0:w1] = organ_weight
            _mask_lat = F.interpolate(_mask, size=self._latent_shape[1:],
                                      mode="trilinear", align_corners=False)
            self.register_buffer("_organ_mask", _mask_lat)
        else:
            self._organ_mask = None
            
                
        self.recon_weight       = recon_weight
        self.nav_signal_weight  = nav_signal_weight
        self.inbatch_div_weight = inbatch_div_weight
        self.nav_corr_weight    = nav_corr_weight
        self.nav_corr_tau       = nav_corr_tau
        self.phi_reg_weight     = phi_reg_weight
        self.vmorph_weight      = vmorph_weight
        self.organ_weight       = organ_weight
        self.bg_weight          = bg_weight
        self.use_phi_time_enc        = use_phi_time_enc
        self.use_phi_bottleneck_scale = use_phi_bottleneck_scale
        self._phi_for_denoising      = None   # set per forward pass; consumed by predict_eps
        self._phi_scalar             = None   # raw (B,1) phase for bottleneck scale path

        # phi_proj is shared between the denoising net (time-emb path) and the prior head.
        # Training phi_prior_head with MSE(z0_pred, z0) forces phi_proj to learn
        # phase-discriminative embeddings — breaking the mean-predictor local minimum
        # that pure DDPM training falls into on cardiac data.
        if use_phi_time_enc:
            import math as _math
            self.phi_proj = nn.Sequential(
                nn.Linear(1, _cond_dim), nn.SiLU(),
                nn.Linear(_cond_dim, _cond_dim),
            )
            self.phi_prior_head = nn.Linear(_cond_dim, _math.prod(self._latent_shape))
            self.phi_prior_weight = phi_prior_weight
        else:
            self.phi_proj = None
            self.phi_prior_head = None
            self.phi_prior_weight = 0.0

        # phi_regressor: predicts scalar phase from global-avg-pooled z0_hat.
        # Applied inside the NavCorr pass (z_nc = randn → no phase in input).
        # Non-shortcuttable: a constant z0_hat gives constant phi_pred → MSE always large.
        if phi_reg_weight > 0.0:
            self.phi_regressor = nn.Sequential(
                nn.Linear(latent_ch, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
        else:
            self.phi_regressor = None

    # ------------------------------------------------------------------
    # BaseDiffusionModel interface
    # ------------------------------------------------------------------

    @property
    def latent_shape(self) -> Tuple[int, ...]:
        return self._latent_shape

    def encode_context(self, Vref, Iseq, is_training: bool):
        B = Vref.shape[0]
        
        if self.zero_context:
            z_ref = torch.zeros(B, self._pre_latent_dim, device=Vref.device)
            z_cond = [torch.zeros(B, self._pre_latent_dim, device=Vref.device)
                    for _ in range(self.horizon)]
            return z_ref, z_cond

        f_ref = (self.ref_net(Vref) if self.ref_net is not None
                 else torch.zeros(B, self._pre_latent_dim, device=Vref.device))

        if self.cond_net is None:
            cond_features = [torch.zeros(B, self._pre_latent_dim, device=Vref.device)
                             for _ in range(self.horizon)]
            return f_ref, cond_features

        if is_training:
            Ipast   = Iseq[:, :, :self.cond_net.num_frames]
            Ifuture = Iseq[:, :, self.cond_net.num_frames:]
            if Ifuture.shape[2] == 0:
                Ifuture = None
            if self.temporal_aug_cfg is not None:
                Ipast = apply_pixel_space_augmentations(Ipast, self.temporal_aug_cfg, training=True)
            cond_features, _ = self.cond_net(Ipast, Ifuture)
        else:
            cond_features, _ = self.cond_net(Iseq, None)

        if self.DEBUG:
            if self.ref_net is not None:
                print(f"[DEBUG UNet3D] ref_feat norm: {f_ref.norm(dim=-1).mean():.3f}")
            print(f"[DEBUG UNet3D] cond_feat norm: {cond_features[0].norm(dim=-1).mean():.3f}")

        return f_ref, cond_features

    def _build_c(self, f_ref, h_t, amp_scalar=None):
        # cross_attn_multi: project each stream separately → stack as token sequence [B, N, cond_dim]
        if self._is_multi_token:
            tokens = []
            if self.use_f_ref and self.ref_token_proj is not None:
                tokens.append(self.ref_token_proj(f_ref))
            if self.use_h_t and self.ht_token_proj is not None:
                tokens.append(self.ht_token_proj(h_t))
            c = torch.stack(tokens, dim=1)   # (B, N_tokens, cond_dim)
            if self.DEBUG:
                print(f"[_build_c] multi-token: {len(tokens)} tokens → c: {c.shape}")
            return c

        # Standard path: concatenate all parts → single vector
        parts = []
        if self.use_f_ref:
            parts.append(f_ref)
        if self.use_h_t:
            parts.append(h_t)
        vec = torch.cat(parts, dim=-1)
        c   = self.context_norm(self.cond_proj(vec)) if self.use_cond_proj else vec

        if self.amplitude_proj is not None and amp_scalar is not None:
            c = c + self.amplitude_proj(amp_scalar.float())

        if self.DEBUG:
            print(f"[_build_c] parts: {[p.shape[-1] for p in parts]} → vec: {vec.shape} → c: {c.shape}"
                  f"  (use_cond_proj={self.use_cond_proj})")

        return c

    def predict_eps(self, z_noisy, tau, context) -> torch.Tensor:
        phi_emb = None
        if self.phi_proj is not None and self._phi_for_denoising is not None:
            phi_emb = self.phi_proj(self._phi_for_denoising.float())
        return self.denoising_net(z_noisy, tau, context, phi_emb=phi_emb,
                                  phi_scalar=self._phi_scalar)

    def _set_self_swap(self, enabled: bool, ratio: float = 0.0, mode: str = "both") -> None:
        """Toggle Self-Swap Guidance perturbation for the next predict_eps call(s)."""
        self.denoising_net.set_self_swap(enabled, ratio, mode)

    def encode_dvf(self, dvf, cond_feats=None, phi: float = None):
        out = self.vae.encode(dvf, cond_feats)
        z = out[0] if isinstance(out, tuple) else out
        return self._norm(z, phi=phi)
        # print(f"z0 scale: mean={self._norm(z).abs().mean():.3f}, std={self._norm(z).std():.3f}")  
        # if isinstance(out, tuple): #Changed
        #     mu, logvar = out
        #     z = self.vae.reparametrize(mu, logvar)
        # else:
        #     z = out
        # return self._norm(z)                                                                                                                         

    def decode_latent(self, z0, cond_feats=None, phi: float = None):
        z_denorm = self._denorm(z0, phi=phi)

        # VQVAE: snap the continuous denoised estimate to the nearest codebook
        # entry before decoding.  The decoder was trained exclusively on
        # quantized z_q vectors; passing a raw continuous z0_hat (especially
        # the noisy Tweedie estimate early in training) produces out-of-
        # distribution inputs that degrade reconstruction quality.
        if hasattr(self.vae, "quantizer"):
            _, z_denorm, *_ = self.vae.quantizer(z_denorm)
            
        return self.vae.decode(z_denorm, cond_feats)

    def warp_volume(self, ref, dvf):
        if self.image_mode:
            return dvf  # decoded tensor is already the output volume
        return self.spatial_transform(ref, dvf)

    def set_film_debug(self, enabled: bool) -> None:
        """Enable/disable per-FiLM-layer scale/shift prints to diagnose context integration."""
        for m in self.denoising_net.modules():
            if isinstance(m, FiLM3d):
                m._debug = enabled

    def set_shape_debug(self, enabled: bool = True) -> None:
        """Enable/disable per-block input/output shape prints in the denoising U-Net."""
        self.denoising_net.set_shape_debug(enabled)

    def load_state_dict(self, state_dict, strict=True):
        # Remap checkpoints saved with the old architecture where phi_proj lived
        # inside DenoisingUNet3d as phi_time_proj.
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("denoising_net.phi_time_proj."):
                new_k = k.replace("denoising_net.phi_time_proj.", "phi_proj.")
                remapped[new_k] = v
                print(f"[ckpt remap] {k} → {new_k}")
            else:
                remapped[k] = v
        # phi_prior_head won't exist in old checkpoints — allow missing keys so the
        # head simply starts from random init (fine for inference / continued training).
        missing, unexpected = super().load_state_dict(remapped, strict=False)
        true_missing    = [k for k in missing    if "phi_prior_head" not in k and "phi_regressor" not in k
                           and "amplitude_enc" not in k and "amplitude_proj" not in k]
        true_unexpected = [k for k in unexpected if "denoising_net.phi_time_proj" not in k]
        if true_missing or true_unexpected:
            if strict:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for UNet3D:\n"
                    f"\tMissing key(s): {true_missing}\n"
                    f"\tUnexpected key(s): {true_unexpected}"
                )
            else:
                if true_missing:    print(f"[UNet3D] missing keys: {true_missing}")
                if true_unexpected: print(f"[UNet3D] unexpected keys: {true_unexpected}")

    def freeze_vae(self):
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        if self.ref_net is not None and not any(p.requires_grad for p in self.ref_net.parameters()):
            self.ref_net.eval()
        if self.cond_net is not None and not any(p.requires_grad for p in self.cond_net.backbone.parameters()):
            self.cond_net.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _extract_amplitude(self, Iseq: torch.Tensor) -> torch.Tensor | None:
        """Returns (B, 2): [amplitude_t, delta_amplitude] from the last two past frames.
        amplitude gives magnitude; delta gives breathing direction (+ inhale, − exhale)."""
        if self.amplitude_enc is None:
            return None
        last_idx = min(self._num_frames, Iseq.shape[2]) - 1
        amp_t    = self.amplitude_enc(Iseq[:, :, last_idx])          # (B, 1)
        prev_idx = max(last_idx - 1, 0)
        amp_prev = self.amplitude_enc(Iseq[:, :, prev_idx])          # (B, 1)
        return torch.cat([amp_t, amp_t - amp_prev], dim=-1)          # (B, 2)

    def forward(self, Vref, Vn, Iseq, dvf=None, criterion=None, phase=None, fixed_noise: bool = False,
                ddim_steps: int | None = None, sampler: str | None = None,
                seg_masks=None, seg_valid=None, **_) -> DiffusionOutput:
        is_training = dvf is not None
        f_ref, cond_features = self.encode_context(Vref, Iseq, is_training)
        amp_scalar = self._extract_amplitude(Iseq)
        if not is_training:
            return self._inference_forward(Vref, f_ref, cond_features, phase=phase,
                                           fixed_noise=fixed_noise, ddim_steps=ddim_steps,
                                           sampler=sampler, amp_scalar=amp_scalar)

        assert criterion is not None, "criterion required during training"
        return self._training_forward(Vref, dvf, Vn, f_ref, cond_features, criterion,
                                      phase=phase, seg_masks=seg_masks, seg_valid=seg_valid,
                                      amp_scalar=amp_scalar)

    def _training_forward(self, Vref, dvf, current_vols, f_ref, cond_features, criterion,
                          phase=None, seg_masks=None, seg_valid=None, amp_scalar=None):
        ddpm_total       = 0.0
        vol_recon_total  = torch.tensor(0.0, device=Vref.device)
        dvf_recon_total  = torch.tensor(0.0, device=Vref.device)
        dvf_recon_count  = 0
        vmorph_total     = torch.tensor(0.0, device=Vref.device)
        nav_corr_total   = torch.tensor(0.0, device=Vref.device)
        nav_corr_count   = 0
        nav_signal_total = torch.tensor(0.0, device=Vref.device)
        nav_signal_count = 0
        warped_vols  = []
        decoded_dvfs = []
        vol_recon_count = 0

        for t in range(self.horizon):
            h_t = cond_features[t]

            phi_scalar = None
            if phase is not None:
                phi_t      = phase[:, t] if phase.dim() == 2 else phase
                phi_scalar = phi_t.to(Vref.device).unsqueeze(-1)  # (B, 1)

            # Store for predict_eps → DenoisingUNet3d.forward.
            self._phi_for_denoising = phi_scalar if self.use_phi_time_enc else None
            self._phi_scalar        = phi_scalar if self.use_phi_bottleneck_scale else None

            c = self._build_c(f_ref, h_t, amp_scalar=amp_scalar)

            if self.DEBUG:
                print(f"[DEBUG UNet3D] t={t} | context shape: {c.shape}")

            cond_feats_t = torch.cat([cond_features[t], f_ref], dim=1)  # (B, pre_latent_dim*2)
            warped_t = self.warp_volume(Vref, dvf[t])
            warped_vols.append(warped_t)

            # Volume-space loss: voxelmorph warp vs target (skipped in image_mode)
            if not self.image_mode and self.vmorph_weight > 0:
                vmorph_loss_t = self._call_criterion(criterion, warped_t, current_vols[t], Vref.device)
                if not (torch.isnan(vmorph_loss_t) or torch.isinf(vmorph_loss_t)):
                    vmorph_total = vmorph_total + vmorph_loss_t

            if self.use_spatial_weight and not self.image_mode:
                with torch.no_grad():
                    dvf_mag = dvf[t].norm(dim=1, keepdim=True)           # (B,1,D,H,W)
                    w_sp    = F.interpolate(dvf_mag, size=self._latent_shape[1:],
                                            mode="trilinear", align_corners=False)
                    mean_w  = w_sp.mean(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
                    w_sp    = (1.0 + w_sp / mean_w).clamp(max=self.spatial_weight_max)
            else:
                w_sp = 1.0

            if seg_masks is not None:
                # print("SEG MASK")
                with torch.no_grad():
                    mask_t = F.interpolate(seg_masks[t], size=self._latent_shape[1:],
                                            mode="trilinear", align_corners=False)
                    w_organ_t = self.bg_weight + (self.organ_weight - self.bg_weight) * mask_t
                    if seg_valid is not None:
                        valid_b = seg_valid.view(-1, 1, 1, 1, 1).to(w_organ_t.dtype)
                        w_organ_t = valid_b * w_organ_t + (1 - valid_b) * torch.ones_like(w_organ_t)
                w_sp = w_sp * w_organ_t
            elif self._organ_mask is not None:
                w_sp = w_sp * self._organ_mask

            phi_val = None
            if phase is not None:
                phi_t   = phase[:, t] if phase.dim() == 2 else phase
                phi_val = phi_t.mean().item()
            ddpm_loss, dvf_recon_loss, dvf_hat = self._diffusion_step(
                dvf[t], context=c,
                snr_gamma=self.snr_gamma,
                low_tau_frac=self.low_tau_frac,
                criterion=criterion,
                cond_feats=cond_feats_t,
                spatial_weight=w_sp,
                predict_mode=self.predict_mode,
                dose_dropout_p=self.dose_dropout_p,
                phi=phi_val,
                recon_weight=self.recon_weight,
                diversity_weight=self.inbatch_div_weight,
                need_dvf_hat=(self.nav_signal_weight > 0),
            )

            decoded_dvfs.append(dvf_hat)

            if dvf_hat is not None and self.recon_weight > 0:
                vol_approx    = self.warp_volume(Vref, dvf_hat)
                vol_recon_t   = self._call_criterion(criterion, vol_approx, current_vols[t], Vref.device)
                if not (torch.isnan(vol_recon_t) or torch.isinf(vol_recon_t)):
                    vol_recon_total = vol_recon_total + vol_recon_t
                    vol_recon_count += 1

            if dvf_hat is not None and self.nav_signal_weight > 0:
                vol_nav  = self.warp_volume(Vref, dvf_hat)
                nav_pred = (vol_nav        - Vref).abs().mean(dim=[1, 2, 3, 4])  # (B,)
                nav_gt   = (current_vols[t] - Vref).abs().mean(dim=[1, 2, 3, 4])  # (B,)
                nav_sig_loss = F.mse_loss(nav_pred, nav_gt) * self.nav_signal_weight
                if not (torch.isnan(nav_sig_loss) or torch.isinf(nav_sig_loss)):
                    ddpm_total       = ddpm_total + nav_sig_loss
                    nav_signal_total = nav_signal_total + nav_sig_loss.detach()
                    nav_signal_count += 1

            ddpm_total += ddpm_loss

            # Direct z0 regression via phi_prior_head: provides a gradient path to
            # phi_proj that bypasses the FiLM entirely, breaking the mean-predictor
            # local minimum that DDPM training alone cannot escape on cardiac data.
            if (self.phi_prior_head is not None
                    and phi_scalar is not None
                    and self.phi_prior_weight > 0.0):
                with torch.no_grad():
                    z0_t = self.encode_dvf(dvf[t], cond_feats_t, phi=phi_val)
                phi_emb_prior = self.phi_proj(phi_scalar.float())
                B = phi_scalar.shape[0]
                z0_prior_pred = self.phi_prior_head(phi_emb_prior).view(B, *self._latent_shape)
                prior_loss = F.mse_loss(z0_prior_pred, z0_t.detach()) * self.phi_prior_weight
                ddpm_total = ddpm_total + prior_loss

            # Phi-gen loss: force phi-sensitive generation via MSE to target z0.
            # z_nc is a PRIOR sample (randn) — no phase info in the noisy input, so
            # the model cannot use the DDPM shortcut (recover z0 from z_nc).
            # MSE to the actual target z0 requires correct spatial content, blocking the
            # norm shortcut (Pearson-based losses were satisfied by phi_emb norm ∝ phi).
            # c is kept to avoid out-of-distribution denoising; c varies very little
            # across cardiac phases (cos_sim ≈ 0.99), so phi_emb carries most of the
            # residual phase signal needed to reach low MSE.
            B_nc = phi_scalar.shape[0] if phi_scalar is not None else 0
            _run_nc_pass = (phi_scalar is not None and B_nc >= 2
                            and (self.nav_corr_weight > 0.0 or self.phi_reg_weight > 0.0))
            if _run_nc_pass:
                # NavCorr forward pass: z_nc is pure noise so the model has no
                # phase info except φ_emb and c.  Both losses operate on z0_hat_nc.
                _tau_nc = max(self.nav_corr_tau, 1)  # guard against tau=0
                tau_nc_t = torch.full((B_nc,), _tau_nc,
                                      dtype=torch.long, device=Vref.device)
                z0_prior = torch.randn(B_nc, *self._latent_shape, device=Vref.device)
                z_nc, _ = self.schedule.q_sample(z0_prior, tau_nc_t)
                phi_emb_nc = (self.phi_proj(phi_scalar.float())
                              if self.phi_proj is not None else None)
                v_nc = self.denoising_net(z_nc, tau_nc_t, c, phi_emb=phi_emb_nc)
                z0_hat_nc = self.schedule.predict_z0_from_v(z_nc, tau_nc_t, v_nc)

                if self.nav_corr_weight > 0.0:
                    with torch.no_grad():
                        z0_target = self.encode_dvf(dvf[t], cond_feats_t, phi=phi_val)
                    nc_loss = F.mse_loss(z0_hat_nc, z0_target.detach()) * self.nav_corr_weight
                    ddpm_total     = ddpm_total + nc_loss
                    nav_corr_total = nav_corr_total + nc_loss.detach()
                    nav_corr_count += 1

                if self.phi_regressor is not None and self.phi_reg_weight > 0.0:
                    z0_pool  = z0_hat_nc.mean(dim=(2, 3, 4))           # (B, latent_ch)
                    phi_pred = self.phi_regressor(z0_pool).squeeze(-1)  # (B,)
                    reg_loss = F.mse_loss(phi_pred, phi_scalar.squeeze(-1)) * self.phi_reg_weight
                    ddpm_total     = ddpm_total + reg_loss
                    nav_corr_total = nav_corr_total + reg_loss.detach()
                    nav_corr_count += 1

            if dvf_recon_loss is not None:
                dvf_recon_total = dvf_recon_total + dvf_recon_loss
                dvf_recon_count += 1

        H = self.horizon

        return DiffusionOutput(
            generated_dvf    = [],
            generated_vols   = [],
            warped_vols      = warped_vols,
            cond_features    = cond_features,
            ddpm_loss        = ddpm_total / H,
            dvf_recon_loss   = dvf_recon_total / max(dvf_recon_count, 1),
            vmorph_vol_loss  = vmorph_total / H,
            vol_recon_loss   = vol_recon_total / max(vol_recon_count, 1),
            nav_corr_loss    = nav_corr_total   / max(nav_corr_count,   1),
            nav_signal_loss  = nav_signal_total / max(nav_signal_count, 1),
        )

    @torch.no_grad()
    def _inference_forward(self, Vref, f_ref, cond_features, phase=None,
                           fixed_noise: bool = False, ddim_steps: int | None = None,
                           sampler: str | None = None, amp_scalar=None):
        generated_dvf, generated_vols = [], []
        z_init = (torch.randn(Vref.shape[0], *self.latent_shape, device=Vref.device)
                  if fixed_noise else None)
        # TMNet always computed all horizon features (joint transformer).
        t_indices = [self.horizon_idx] if self.horizon_idx is not None else range(self.horizon)
        for t in t_indices:
            phi_scalar = None
            if phase is not None:
                phi_t      = phase[:, t] if phase.dim() == 2 else phase
                phi_scalar = phi_t.to(Vref.device).unsqueeze(-1)

            # Store for predict_eps → DenoisingUNet3d.forward.
            self._phi_for_denoising = phi_scalar if self.use_phi_time_enc else None
            self._phi_scalar        = phi_scalar if self.use_phi_bottleneck_scale else None

            c = self._build_c(f_ref, cond_features[t], amp_scalar=amp_scalar)
            if self.DEBUG:
                phi_str = (f"{phi_scalar.squeeze().tolist():.4f}"
                           if phi_scalar is not None else "None")
                print(f"[DEBUG UNet3D] t={t} | phi={phi_str}"
                      f" | c_norm={c.norm(dim=-1).mean():.4f}"
                      f" | c_std={c.std(dim=-1).mean():.4f}")
            cond_feats_t = torch.cat([cond_features[t], f_ref], dim=1)
            phi_val = None
            if phase is not None:
                phi_t   = phase[:, t] if phase.dim() == 2 else phase
                phi_val = phi_t.mean().item()
            _steps   = ddim_steps if ddim_steps is not None else self.ddim_steps
            _sampler = sampler if sampler is not None else self.sampler
            if self.ssg_scale > 1.0:
                z0_hat = self.ssg_sample(c, Vref.shape[0], Vref.device, _steps,
                                         ssg_scale=self.ssg_scale,
                                         swap_ratio=self.ssg_swap_ratio,
                                         swap_mode=self.ssg_mode,
                                         predict_mode=self.predict_mode)
            else:
                z0_hat = self._sample(_sampler, c, Vref.shape[0], Vref.device, _steps,
                                      cfg_scale=1.0,
                                      predict_mode=self.predict_mode,
                                      z_init=z_init)
            if self.DEBUG:
                print(f"[DEBUG UNet3D] t={t} | z0_hat mean={z0_hat.mean():.4f}"
                      f"  std={z0_hat.std():.4f}"
                      f"  min={z0_hat.min():.4f}"
                      f"  max={z0_hat.max():.4f}")

            dvf_t  = self.decode_latent(z0_hat, cond_feats_t, phi=phi_val)

            if self.DEBUG:
                print(f"[DEBUG UNet3D] t={t} | DVF mean={dvf_t.mean():.4f}"
                      f"  std={dvf_t.std():.4f}"
                      f"  abs_max={dvf_t.abs().max():.4f}")

                # Zero-latent: decoder unconditional prior
                dvf_zero = self.decode_latent(torch.zeros_like(z0_hat), torch.zeros_like(cond_feats_t))
                print(f"[DEBUG UNet3D] t={t} | Zero-latent DVF: mean={dvf_zero.mean():.4f}  peak_z={dvf_zero.norm(dim=1)[0].mean(dim=(1,2)).argmax().item()}")

                # Latent normalisation sanity
                print(f"[DEBUG UNet3D] t={t} | z0_hat in latent_stats range: "
                      f"latent_mean={self.latent_mean.item():.4f}  "
                      f"latent_std={self.latent_std.item():.4f}  "
                      f"z0_hat_normed_std={((z0_hat - self.latent_mean) / self.latent_std).std():.4f}")

                # ── Per-sample summary — compare rows where phi differs ──────────
                # If c/z0/dvf are identical across very different phi values → collapse.
                _phi_val = (phase[0, t].item() if phase is not None and phase.dim() == 2
                            else phase[0].item() if phase is not None else float("nan"))
                print(f"[DEBUG sample] phi={_phi_val:.3f} | "
                      f"c: norm={c.norm(dim=-1).mean():.4f}  mean={c.mean():.4f}  std={c.std():.4f} | "
                      f"z0: mean={z0_hat.mean():.4f}  std={z0_hat.std():.4f} | "
                      f"dvf: mean={dvf_t.mean():.4f}  abs_max={dvf_t.abs().max():.4f}  "
                      f"mag={dvf_t.norm(dim=1).mean():.4f}")

                # Context ablation
                c_zeros    = torch.zeros_like(c)
                z0_no_ctx  = self.ddim_sample(c_zeros, Vref.shape[0], Vref.device, self.ddim_steps)
                dvf_no_ctx = self.decode_latent(z0_no_ctx, torch.zeros_like(cond_feats_t))
                print(f"[DEBUG UNet3D] t={t} | DVF change when context zeroed: {(dvf_t - dvf_no_ctx).abs().mean():.4f}")

            if self.image_mode:
                generated_dvf.append(None)
                generated_vols.append(dvf_t)
            else:
                generated_dvf.append(dvf_t)
                vol = self.warp_volume(Vref, dvf_t)
                generated_vols.append(vol)

        return DiffusionOutput(generated_dvf=generated_dvf, generated_vols=generated_vols)