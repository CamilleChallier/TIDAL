"""
Adapters that bridge pretrained TMNetEncoder / RVNet to the token-sequence
interface expected by CausalDiT (B, N, d_model).

Architecture
------------
TMNetDiTAdapter
    frozen TMNetEncoder  →  (B, T, hidden_dim)
    trainable Linear + LN   →  (B, T, d_model)
    Replaces CausalIseqEncoder in CausalDiT.

RVNetDiTAdapter
    frozen RVNet.encoder  →  (B, C, D', H', W')
    trainable Linear + LN + pos_emb  →  (B, N, d_model)
    Replaces RefVolumeEncoder in CausalDiT.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TMNetDiTAdapter(nn.Module):
    """
    Wraps a frozen TMNetEncoder and projects its temporal tokens to d_model.

    Parameters
    ----------
    encoder    : TMNetEncoder instance (with weights already loaded)
    hidden_dim : encoder's internal dimension (out_channels[-1], typically 64)
    d_model    : CausalDiT model dimension (e.g. 384)
    """

    def __init__(self, encoder, hidden_dim: int, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(hidden_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def forward(self, Iseq: torch.Tensor) -> torch.Tensor:
        """
        Iseq : (B, C, T, H, W)
        Returns (B, T_enc, d_model)  where T_enc = encoder.num_frames
        """
        T_enc = self.encoder.num_frames
        with torch.no_grad():
            tokens = self.encoder.encode_temporal(Iseq[:, :, :T_enc])  # (B, T_enc, hidden_dim)
        return self.norm(self.proj(tokens))


class RVNetDiTAdapter(nn.Module):
    """
    Wraps a frozen RVNet CNN backbone and projects its spatial tokens to d_model.

    Parameters
    ----------
    rvnet           : RVNet instance (with weights already loaded)
    backbone_out_ch : number of channels at the backbone output (enc_channels[-1])
    d_model         : CausalDiT model dimension (e.g. 384)
    vol_size        : input volume spatial size (D, H, W)
    nb_convs        : number of stride-2 conv blocks in the backbone
    """

    def __init__(
        self,
        rvnet,
        backbone_out_ch: int,
        d_model: int,
        vol_size: tuple = (32, 64, 64),
        nb_convs: int = 2,
    ):
        super().__init__()
        self.backbone = rvnet.encoder

        stride = 2 ** nb_convs
        d, h, w = vol_size[0] // stride, vol_size[1] // stride, vol_size[2] // stride
        n_tokens = d * h * w

        self.proj    = nn.Linear(backbone_out_ch, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, D, H, W)
        Returns (B, N, d_model)
        """
        with torch.no_grad():
            feat = self.backbone(x)          # (B, backbone_out_ch, D', H', W')
        B, C, d, h, w = feat.shape
        tokens = feat.permute(0, 2, 3, 4, 1).reshape(B, d * h * w, C)  # (B, N, C)
        return self.norm(self.proj(tokens)) + self.pos_emb
