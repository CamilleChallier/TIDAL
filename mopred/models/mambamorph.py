"""
MambaMorph: Mamba-based deformable registration backbone for the TIDAL pipeline.
A hierarchical SSM (state-space model) encoder with a TransMorph-style convolutional
decoder predicts a dense DVF between a reference and a current volume.
Requires the ``mamba_ssm`` CUDA extension (install in the separate ``mamba`` conda env).
Ported and trimmed from the MambaMorph repository (https://github.com/mileswyn/MambaMorph).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal

try:
    from mamba_ssm import Mamba
except ImportError as e:
    raise ImportError(
        "MambaMorph requires the `mamba_ssm` package (CUDA extension), which "
        "is not installed in this environment. Activate the `mamba` conda env."
    ) from e


def _to_3tuple(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x, x)


# ---------------------------------------------------------------------------
# from mambamorph/torch/mamba.py
# ---------------------------------------------------------------------------
class PatchMerging(nn.Module):
    """Merges 2x2x2 neighboring patches and projects back down to `reduce_factor * dim`."""

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, T):
        """x: B, H*W*T, C"""
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

        x = x.view(B, H, W, T, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = x.view(B, -1, 8 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class MambaLayer(nn.Module):
    """One stage of the hierarchical encoder: LayerNorm -> Mamba (SSM) -> optional downsample."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, downsample=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        x_mamba = self.mamba(x_norm)
        x = x_mamba.type(x.dtype)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            # print(f"[MambaLayer] after downsample: x_down shape: {tuple(x_down.shape)}")
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T


# ---------------------------------------------------------------------------
# from mambamorph/torch/TransMorph.py (Mamba-specific classes only)
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """3D image-to-patch embedding via a single strided conv."""

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = _to_3tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        print(f"embed_dim{embed_dim}")

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        return x


class SinPositionalEncoding3D(nn.Module):
    """Fixed (non-learned) sinusoidal position encoding over a 3D feature grid."""

    def __init__(self, channels):
        super().__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z
        emb = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)


class GaussianBlur3D(nn.Module):
    """Fixed (non-learned) depthwise 3D Gaussian blur -- the low-pass operator
    that ALPF/AHPF below are built around."""

    def __init__(self, channels, kernel_size=3, sigma=1.0):
        super().__init__()
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        g1d = torch.exp(-(coords**2) / (2 * sigma**2))
        g1d = g1d / g1d.sum()
        kernel3d = torch.einsum("i,j,k->ijk", g1d, g1d, g1d)
        kernel3d = kernel3d / kernel3d.sum()
        weight = kernel3d.view(1, 1, kernel_size, kernel_size, kernel_size).repeat(channels, 1, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.channels = channels
        self.padding = kernel_size // 2

    def forward(self, x):
        return nnf.conv3d(x, self.weight, padding=self.padding, groups=self.channels)


class ALPF3D(nn.Module):
    """Adaptive Low-Pass Filter (Shi et al., "Integrating frequency-aware mamba
    with diffusion for 4D volumetric image synthesis"): restores low-frequency
    / global structure that stride-2 downsampling discards. A 1x1x1-receptive
    conv + softmax (over channels, at each spatial location) produces a
    spatially-varying weight map; this gates a Gaussian-blurred (low-pass)
    copy of the features, which is added back to the downsampled features
    scaled by a fixed residual strength `lambda_lp` (0.5 in the paper).
    """

    def __init__(self, channels, lambda_lp=0.5, kernel_size=3, sigma=1.0):
        super().__init__()
        self.gate = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.blur = GaussianBlur3D(channels, kernel_size=kernel_size, sigma=sigma)
        self.lambda_lp = lambda_lp

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=1)
        low_freq = self.blur(x)
        return x + self.lambda_lp * weights * low_freq


class AHPF3D(nn.Module):
    """Adaptive High-Pass Filter (Shi et al.), used in place of a plain skip
    connection. Recovers fine edge/boundary detail lost during encoding: one
    branch passes the skip features through unchanged, the other gates a
    high-pass component -- the skip features with their Gaussian-blurred
    (low-pass) copy subtracted out, i.e. the blur "reversed" -- with a learned
    spatially-varying weight map. The two branches are summed.
    """

    def __init__(self, channels, kernel_size=3, sigma=1.0):
        super().__init__()
        self.gate = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.blur = GaussianBlur3D(channels, kernel_size=kernel_size, sigma=sigma)

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=1)
        high_freq = x - self.blur(x)
        return x + weights * high_freq


class MambaBlock(nn.Module):
    """Hierarchical Mamba encoder: PatchEmbed -> (+ position encoding) -> stack of MambaLayer stages."""

    def __init__(
        self,
        patch_size=4,
        in_chans=2,
        embed_dim=96,
        depths=(2, 2, 4),
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        ape=False,
        spe=True,
        patch_norm=True,
        out_indices=(0, 1, 2),
        d_state=16,
        d_conv=4,
        expand=2,
        use_freq_aware=False,
        lambda_lp=0.5,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.use_freq_aware = use_freq_aware

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, 1, 1))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MambaLayer(
                dim=int(embed_dim * 2**i_layer),
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        for i_layer in out_indices:
            self.add_module(f"norm{i_layer}", norm_layer(self.num_features[i_layer]))

        if self.use_freq_aware:
            # One ALPF per downsampling step (i.e. every stage but the last) --
            # restores the low-frequency structure that PatchMerging's stride-2
            # pooling discards before it reaches the next stage.
            self.alpf_blocks = nn.ModuleList([
                ALPF3D(self.num_features[i_layer + 1], lambda_lp=lambda_lp)
                for i_layer in range(self.num_layers - 1)
            ])

    @staticmethod
    def _apply_alpf(alpf, x, H, W, T):
        B, L, C = x.shape
        x_sp = x.view(B, H, W, T, C).permute(0, 4, 1, 2, 3)
        x_sp = alpf(x_sp)
        return x_sp.permute(0, 2, 3, 4, 1).reshape(B, L, C)

    def forward(self, x):
        x = self.patch_embed(x)
        # print(f"[MambaBlock] patch_embed output shape: {tuple(x.shape)}")
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode="trilinear")
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
            # print(f"[MambaBlock] after absolute pos embd: {tuple(x.shape)}")
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
            # print(f"[MambaBlock] after sinusoidal pos embd: {tuple(x.shape)}")
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        # print(f"[MambaBlock] after pos_drop: {tuple(x.shape)}")

        outs = []
        for i in range(self.num_layers):
            x_out, H, W, T, x, Wh, Ww, Wt = self.layers[i](x, Wh, Ww, Wt)
            # print(f"[MambaBlock] after layer {i}: x_out shape: {tuple(x_out.shape)}, x_out: {(x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()).shape}, x shape: {tuple(x.shape)}")
            if self.use_freq_aware and i < self.num_layers - 1:
                x = self._apply_alpf(self.alpf_blocks[i], x, Wh, Ww, Wt)
                # print(f"[MambaBlock] after ALPF {i}: x shape: {tuple(x.shape)}, x: {(x.view(-1, Wh, Ww, Wt, self.num_features[i + 1]).permute(0, 4, 1, 2, 3).contiguous()).shape}")
            if i in self.out_indices:
                x_out = getattr(self, f"norm{i}")(x_out)
                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                # print(f"[MambaBlock] output from layer {i}: {tuple(out.shape)}")
                outs.append(out)
        return outs


class Conv3dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        relu = nn.LeakyReLU(inplace=True)
        nm = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.InstanceNorm3d(out_channels)
        super().__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True, upsample=True):
        super().__init__()
        self.conv1 = Conv3dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv3dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False) if upsample else nn.Identity()

    def forward(self, x, skip=None):
        x = self.up(x)
        # print(f"[DecoderBlock] after upsample: {tuple(x.shape)}")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            # print(f"[DecoderBlock] after concatenating skip: {tuple(x.shape)}")
        x = self.conv1(x)
        # print(f"[DecoderBlock] after conv1: {tuple(x.shape)}")
        x = self.conv2(x)
        # print(f"[DecoderBlock] after conv2: {tuple(x.shape)}")
        return x


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


class MambaMorph(nn.Module):
    """
    Mamba-based registration network -- predicts a dense flow field between
    a reference (moving) and current (fixed) volume. Pair with the existing
    `SpatialTransformer` (models/spatial_transform.py) the same way
    `Voxelmorph` is used in train_VoxelMorph_ACDC.py.
    """

    def __init__(
        self,
        vol_size,
        in_chans=2,
        embed_dim=96,
        depths=(2, 2, 4),
        reg_head_chan=16,
        d_state=16,
        d_conv=4,
        expand=2,
        patch_size=4,
        spe=True,
        ape=False,
        if_convskip=True,
        if_transskip=True,
        use_freq_aware=False,
        lambda_lp=0.5,
    ):
        super().__init__()
        self.if_convskip = if_convskip
        self.if_transskip = if_transskip
        self.use_freq_aware = use_freq_aware

        self.transformer = MambaBlock(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            ape=ape,
            spe=spe,
            out_indices=tuple(range(len(depths))),
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_freq_aware=use_freq_aware,
            lambda_lp=lambda_lp,
        )

        self.patch_size = patch_size
        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0, use_batchnorm=False)
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0, use_batchnorm=False)
        # patch_size=2: encoder bottleneck is (4,8,8) not (2,4,4), so 3 upsamplings already
        # reach vol_size — up4 must not upsample again, and avg_pool keeps full resolution.
        self.up4 = DecoderBlock(embed_dim // 2, reg_head_chan, skip_channels=reg_head_chan if if_convskip else 0,
                                use_batchnorm=False, upsample=(patch_size != 2))
        self.c1 = Conv3dReLU(in_chans, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(in_chans, reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(in_channels=reg_head_chan, out_channels=len(vol_size), kernel_size=3)
        self.avg_pool = nn.AvgPool3d(3, stride=1 if patch_size == 2 else 2, padding=1)

        if use_freq_aware:
            # AHPF replaces each plain skip connection (encoder-side trans/conv
            # skips), recovering high-frequency edge/boundary detail.
            self.ahpf1 = AHPF3D(embed_dim * 2) if if_transskip else None
            self.ahpf2 = AHPF3D(embed_dim) if if_transskip else None
            self.ahpf4 = AHPF3D(embed_dim // 2) if if_convskip else None
            self.ahpf5 = AHPF3D(reg_head_chan) if if_convskip else None

    def forward(self, v_ref, v_curr):

        # def # print_shape(name, x):
        #     if x is not None:
        #         # print(f"{name}: {tuple(x.shape)}")


        x = torch.cat([v_ref, v_curr], dim=1)

        # print_shape("Input concatenated", x)


        # ---------------------------
        # Convolutional skip branch
        # ---------------------------
        if self.if_convskip:

            x_s0 = x.clone()
            # print_shape("Conv skip x_s0", x_s0)

            x_s1 = self.avg_pool(x)
            # print_shape("Conv skip x_s1 pooled", x_s1)

            f4 = self.c1(x_s1)
            # print_shape("Conv skip f4", f4)

            f5 = self.c2(x_s0)
            # print_shape("Conv skip f5", f5)

        else:
            f4 = None
            f5 = None



        # ---------------------------
        # Mamba encoder
        # ---------------------------
        out_feats = self.transformer(x)

        # for i, feat in enumerate(out_feats):
            # print_shape(f"Mamba encoder level {i}", feat)



        # Transformer skips
        if self.if_transskip:
            f1, f2 = out_feats[-2], out_feats[-3]

            # print_shape("Transformer skip f1", f1)
            # print_shape("Transformer skip f2", f2)

        else:
            f1, f2 = None, None



        # Frequency blocks
        if self.use_freq_aware:

            if f1 is not None:
                f1 = self.ahpf1(f1)
                # print_shape("AHPF f1", f1)

            if f2 is not None:
                f2 = self.ahpf2(f2)
                # print_shape("AHPF f2", f2)

            if f4 is not None:
                f4 = self.ahpf4(f4)
                # print_shape("AHPF f4", f4)

            if f5 is not None:
                f5 = self.ahpf5(f5)
                # print_shape("AHPF f5", f5)



        # ---------------------------
        # Decoder
        # ---------------------------
        x = self.up1(out_feats[-1], f1)
        # print_shape("Decoder up1 output", x)

        x = self.up2(x, f2)
        # print_shape("Decoder up2 output", x)

        x = self.up3(x, f4)
        # print_shape("Decoder up3 output", x)

        x = self.up4(x, f5)
        # print_shape("Decoder up4 output", x)



        # Flow prediction
        flow = self.reg_head(x)

        # print_shape("Final flow", flow)

        return flow