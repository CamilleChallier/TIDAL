"""
RV-Net: reference volume encoder for the TIDAL pipeline.
A lightweight 3D CNN encodes a reference DVF volume into a fixed-size embedding
vector, which UNet3D uses to condition generation on the patient's reference
motion pattern.
"""
from __future__ import annotations

import torch.nn as nn

class RVNet(nn.Module):
    def __init__(
        self,
        nb_convs,
        in_channels,
        out_channels,
        output_dim,
        linear_input_dim=64,
        norm=nn.BatchNorm3d,
        dropout=False,
    ):

        super().__init__()

        assert nb_convs == len(out_channels)

        self.encoder = list()
        self.output_dim = output_dim
        self.linear_input_dim = linear_input_dim

        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]

            self.encoder += [
                nn.Conv3d(in_ch, out_channels[i], kernel_size=3, padding=1, stride=2)
            ]
            if norm is not None:
                self.encoder += [norm(out_channels[i])]
            self.encoder += [nn.ReLU(True)]

            self.encoder += [
                nn.Conv3d(
                    out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1
                )
            ]

            if norm is not None:
                self.encoder += [norm(out_channels[i])]

            self.encoder += [nn.ReLU(True)]
            if dropout:
                self.encoder += [
                    nn.Dropout3d()
                ]  #############################################3

        self.encoder = nn.Sequential(*self.encoder)

        self.adap = nn.Conv3d(out_channels[-1], 1, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(self.adap.weight, mode="fan_in", nonlinearity="relu")

        self.dvf_enc = nn.Linear(self.linear_input_dim, self.output_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the backbone spatial feature map (B, out_channels[-1], D', H', W'),
        bypassing the adap + linear bottleneck.  Used by SimMIM pretraining."""
        return self.encoder(x)

    def forward(self, dvf):
        B = dvf.shape[0]
        # print(f"dvf shape : {B}")
        encoding = self.encoder(dvf)
        # print(f"encoding: {encoding.shape}")
        encoding = self.adap(encoding).view(B, -1)
        # print(f"encoding: {encoding.shape}")
        encoding = self.dvf_enc(encoding)
        # print(f"encoding: {encoding.shape}")
        return encoding