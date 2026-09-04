"""
Standard multi-head self-attention module (``SelfAttention``) used in UNet3D
residual blocks and as a general-purpose attention primitive across the pipeline.
"""
# ============================================================
#
#   Models
#
#  author: Liset Vazquez Romaguera
#  email: lisetvr90@gmail.com
#  github id: lisetvr
#  MedICAL Lab
#  (Some code was taken from: https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/)
# ============================================================
from torch.distributions import Normal
import torch.nn.functional as nnf

# ----------------------
from positional_encodings.torch_encodings import (
    PositionalEncodingPermute3D,
    PositionalEncoding1D,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.f = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.g = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.h = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)

    def forward(self, x):
        m_batchsize, C, depth, width, height = x.size()
        f = self.f(x).view(
            m_batchsize, -1, depth * width * height
        )  # B * (C//8) * (D * W * H)
        g = self.g(x).view(
            m_batchsize, -1, depth * width * height
        )  # B * (C//8) * (D * W * H)
        h = self.h(x).view(
            m_batchsize, -1, depth * width * height
        )  # B * C * (D * W * H)
        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (D * W * H) * (D * W * H)
        attention = self.softmax(attention)
        self_attention = torch.bmm(h, attention)  # B * C * (D * W * H)
        self_attention = self_attention.view(
            m_batchsize, C, depth, width, height
        )  # B * C * d * W * H
        out = self.gamma * self_attention + x
        return out
