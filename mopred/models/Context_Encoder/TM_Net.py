"""
TM-Net: temporal context encoder for the TIDAL pipeline.
A 2D CNN backbone extracts per-frame navigator features; a Transformer encoder with a
learned Gaussian prior aggregates them into a temporal context embedding.
Main classes: ``TMNet_Tr_priormulti`` (multi-step stochastic prior) and
``TMNetEncoder`` (deterministic, used by UNet3D at inference time).
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
from ..spatial_transform import SpatialTransformer

# ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
class Backbone(nn.Module):
    def __init__(
        self, in_channels, out_channels, slice_orientation=2, norm=nn.BatchNorm2d
    ):
        super().__init__()
        nb_convs = len(out_channels)
        self.encoder = list()
        self.out_channels = out_channels
        if slice_orientation == 2:
            stride_list = [2, 2, (1, 2)]
        else:
            stride_list = [2, 2, 2]

        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]

            self.encoder += [
                nn.Conv2d(
                    in_ch,
                    out_channels[i],
                    kernel_size=3,
                    padding=1,
                    stride=stride_list[i],
                )
            ]
            if norm is not None:
                self.encoder += [norm(out_channels[i], affine=True)]
            self.encoder += [nn.ReLU(True)]

            self.encoder += [
                nn.Conv2d(
                    out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1
                )
            ]
            if norm is not None:
                self.encoder += [norm(out_channels[i], affine=True)]
            self.encoder += [nn.ReLU(True)]

        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, Iseq):
        bs, ch, l, h, w = Iseq.shape
        feats = []
        for i in range(l):
            feats.append(self.encoder(Iseq[:, :, i, :, :]))
        feats = torch.stack(feats, dim=2)
        return feats


class TMNet(nn.Module):

    def __init__(
        self,
        horizon,
        nb_convs,
        in_channels,
        out_channels,
        output_dim,
        sag_lin_input_dim,
        cor_lin_input_dim,
        rnn,
        norm=nn.BatchNorm2d,
        dropout=False,
    ):

        super().__init__()
        assert nb_convs == len(out_channels)
        self.rnn = rnn
        self.horizon = horizon
        self.encoder_sag = list()
        self.encoder_cor = list()
        self.dropout = dropout
        self.sag_lin_dim = sag_lin_input_dim
        self.cor_lin_dim = cor_lin_input_dim
        cor_custom_stride = [2, 2, (1, 2)]
        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]

            self.encoder_sag += [
                nn.Conv2d(in_ch, out_channels[i], kernel_size=3, padding=1, stride=2)
            ]
            self.encoder_cor += [
                nn.Conv2d(
                    in_ch,
                    out_channels[i],
                    kernel_size=3,
                    padding=1,
                    stride=cor_custom_stride[i],
                )
            ]

            if norm is not None:
                self.encoder_sag += [norm(out_channels[i], affine=True)]
                self.encoder_cor += [norm(out_channels[i], affine=True)]
            self.encoder_sag += [nn.ReLU(True)]
            self.encoder_cor += [nn.ReLU(True)]

            self.encoder_sag += [
                nn.Conv2d(
                    out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1
                )
            ]
            self.encoder_cor += [
                nn.Conv2d(
                    out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1
                )
            ]

            if norm is not None:
                self.encoder_sag += [norm(out_channels[i], affine=True)]
                self.encoder_cor += [norm(out_channels[i], affine=True)]
            self.encoder_sag += [nn.ReLU(True)]
            self.encoder_cor += [nn.ReLU(True)]
            if self.dropout:
                self.encoder_sag += [nn.Dropout2d()]  ####################
                self.encoder_cor += [nn.Dropout2d()]  ####################
        self.encoder_sag = nn.Sequential(*self.encoder_sag)
        self.encoder_cor = nn.Sequential(*self.encoder_cor)
        if rnn == "gru":
            self.sag_rnn_enc = ConvGRU(
                in_channels=out_channels[-1],
                hidden_channels=[out_channels[-1]],
                kernel_size=(3, 3),
                num_layers=1,
                batch_first=True,
            )
            self.sag_rnn_dec = ConvGRU(
                in_channels=out_channels[-1],
                hidden_channels=[out_channels[-1]],
                kernel_size=(3, 3),
                num_layers=1,
                batch_first=True,
            )
            self.cor_rnn_enc = ConvGRU(
                in_channels=out_channels[-1],
                hidden_channels=[out_channels[-1]],
                kernel_size=(3, 3),
                num_layers=1,
                batch_first=True,
            )
            self.cor_rnn_dec = ConvGRU(
                in_channels=out_channels[-1],
                hidden_channels=[out_channels[-1]],
                kernel_size=(3, 3),
                num_layers=1,
                batch_first=True,
            )
        elif rnn == "lstm":
            self.sag_rnn_enc = ConvLSTM(
                in_channels=out_channels[-1],
                hidden_channels=[out_channels[-1]],
                kernel_size=(3, 3),
                num_layers=1,
                batch_first=True,
            )
            self.sag_rnn_dec = ConvLSTM(
                in_channels=out_channels[-1],
                hidden_channels=[out_channels[-1]],
                kernel_size=(3, 3),
                num_layers=1,
                batch_first=True,
            )
            self.cor_rnn_enc = ConvLSTM(
                in_channels=out_channels[-1],
                hidden_channels=[out_channels[-1]],
                kernel_size=(3, 3),
                num_layers=1,
                batch_first=True,
            )
            self.cor_rnn_dec = ConvLSTM(
                in_channels=out_channels[-1],
                hidden_channels=[out_channels[-1]],
                kernel_size=(3, 3),
                num_layers=1,
                batch_first=True,
            )
        else:
            hidden_dim = 64
            self.num_queries = horizon
            self.query_learn = nn.Embedding(horizon, hidden_dim)
            self.pe3d = PositionalEncodingPermute3D(hidden_dim)
            self.pe_temp = PositionalEncoding1D(hidden_dim)
            self.backbone = Backbone(
                in_channels=2, out_channels=[64, 128, 256], slice_orientation=2
            )
            self.input_proj = nn.Conv3d(horizon, horizon, kernel_size=(1, 8, 8))
            self.transformer = Transformer(
                d_model=hidden_dim,
                nhead=8,
                num_encoder_layers=3,
                num_decoder_layers=3,
                normalize_before=True,
                return_intermediate_dec=False,
            )

        self.bn_sag = nn.BatchNorm3d(out_channels[-1])
        self.bn_cor = nn.BatchNorm3d(out_channels[-1])

        self.adap_sag = nn.Conv2d(
            out_channels[-1], 1, kernel_size=1, stride=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.adap_sag.weight, mode="fan_in", nonlinearity="relu"
        )
        self.adap_cor = nn.Conv2d(
            out_channels[-1], 1, kernel_size=1, stride=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.adap_cor.weight, mode="fan_in", nonlinearity="relu"
        )
        self.sag_enc = nn.Linear(self.sag_lin_dim, output_dim)
        self.cor_enc = nn.Linear(self.cor_lin_dim, output_dim)

    def forward(self, sag, cor):

        if sag is not None:  # case only sagittal
            encsag1 = self.encoder_sag(sag[:, :, 0, :, :])
            encsag2 = self.encoder_sag(sag[:, :, 1, :, :])
            encsag3 = self.encoder_sag(sag[:, :, 2, :, :])
            encsag = torch.stack([encsag1, encsag2, encsag3], dim=2).permute(
                0, 2, 1, 3, 4
            )
            # --------------------------------------------
            if self.rnn == "gru":
                decoder_input = (
                    self.sag_rnn_enc(encsag)[1][0]
                    .unsqueeze(1)
                    .repeat(1, self.horizon, 1, 1, 1)
                )  # Repeat last state
                states, last_state = self.sag_rnn_dec(decoder_input)
                x = states[0].permute(
                    0, 2, 1, 3, 4
                )  # (b, t, c, h, w) -> (b, c, t, h, w)
                encsag_multitime = self.bn_sag(x)
            elif self.rnn == "lstm":
                h_c = self.sag_rnn_enc(encsag)[1][0]
                h, c = h_c[0], h_c[1]
                decoder_input = h.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)
                states, last_state = self.sag_rnn_dec(decoder_input)
                x = states[0].permute(
                    0, 2, 1, 3, 4
                )  # (b, t, c, h, w) -> (b, c, t, h, w)
                encsag_multitime = self.bn_sag(x)
            else:
                src_proj = encsag  # 10, 3, 64, 8, 8
                # ----------------------------------------------------------------
                pos = self.pe3d(src_proj).to(src_proj.dtype)  # [bs, 3, 48, 8, 8]
                src_proj = src_proj.flatten(-3)
                pos = pos.flatten(-3)  # [bs, 3, 48, 64]
                # ----------------------------------------------------------------
                pos_dec_in = self.pe_temp(
                    self.query_learn.weight.unsqueeze(0).repeat(src_proj.shape[0], 1, 1)
                ).to(self.query_learn.weight.dtype)
                hs = self.transformer(
                    enc_in=src_proj,
                    enc_pos=pos,
                    dec_in=self.query_learn.weight.unsqueeze(0).repeat(
                        src_proj.shape[0], 1, 1
                    ),
                    dec_pos=pos_dec_in,
                )[-1]
                # [bs, num_outs, dim]
                x = hs.view(-1, self.num_queries, 2, 8, 8)
                x = x.permute(0, 2, 1, 3, 4)  # bs, ch, l, h, w
                encsag_multitime = self.bn_sag(x)
            # --------------------------------------------
            encsag_list = []
            for t in range(self.horizon):
                encsag = self.adap_sag(encsag_multitime[:, :, t, :, :]).view(
                    encsag_multitime.shape[0], self.sag_lin_dim
                )
                encsag = self.sag_enc(encsag)
                if self.dropout:
                    encsag = nn.Dropout()(encsag)  ####################
                encsag_list.append(encsag)
            return encsag_list

        else:  # case only coronal
            enccor1 = self.encoder_cor(cor[:, :, 0, :, :])
            enccor2 = self.encoder_cor(cor[:, :, 1, :, :])
            enccor3 = self.encoder_cor(cor[:, :, 2, :, :])
            enccor_stacked = torch.stack([enccor1, enccor2, enccor3], dim=2)
            enccor = enccor_stacked.permute(0, 2, 1, 3, 4)
            # --------------------------------------------
            # --------------------------------------------
            if self.rnn == "gru":
                decoder_input = (
                    self.sag_rnn_enc(enccor)[1][0]
                    .unsqueeze(1)
                    .repeat(1, self.horizon, 1, 1, 1)
                )  # Repeat last state
                states, last_state = self.cor_rnn_dec(decoder_input)
                x = states[0].permute(
                    0, 2, 1, 3, 4
                )  # (b, t, c, h, w) -> (b, c, t, h, w)
                enccor_multitime = self.bn_cor(x)
                enccor_list = []
                for t in range(self.horizon):
                    enccor = self.adap_cor(enccor_multitime[:, :, t, :, :]).view(
                        enccor_multitime.shape[0], self.sag_lin_dim
                    )
                    enccor = self.cor_enc(enccor)
                    if self.dropout:
                        enccor = nn.Dropout()(enccor)  ####################
                    enccor_list.append(enccor)

            elif self.rnn == "lstm":
                h_c = self.cor_rnn_enc(enccor)[1][0]
                h, c = h_c[0], h_c[1]
                decoder_input = h.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)
                states, last_state = self.cor_rnn_dec(decoder_input)
                x = states[0].permute(
                    0, 2, 1, 3, 4
                )  # (b, t, c, h, w) -> (b, c, t, h, w)
                enccor_multitime = self.bn_cor(x)
                enccor_list = []
                for t in range(self.horizon):
                    enccor = self.adap_cor(enccor_multitime[:, :, t, :, :]).view(
                        enccor_multitime.shape[0], self.sag_lin_dim
                    )
                    enccor = self.cor_enc(enccor)
                    if self.dropout:
                        enccor = nn.Dropout()(enccor)  ####################
                    enccor_list.append(enccor)
            else:
                src_proj = self.input_proj(enccor)
                # ----------------------------------------------------------------
                pos = self.pe3d(src_proj).to(src_proj.dtype)  # [bs, 3, 48, 8, 8]
                src_proj = src_proj.flatten(-3)
                pos = pos.flatten(-3)  # [bs, 3, 48, 64]
                # ----------------------------------------------------------------
                pos_dec_in = self.pe_temp(
                    self.query_learn.weight.unsqueeze(0).repeat(src_proj.shape[0], 1, 1)
                ).to(self.query_learn.weight.dtype)
                hs = self.transformer(
                    enc_in=src_proj,
                    enc_pos=pos,
                    dec_in=self.query_learn.weight.unsqueeze(0).repeat(
                        src_proj.shape[0], 1, 1
                    ),
                    dec_pos=pos_dec_in,
                )
                hs = hs.permute(1, 0, 2)

                enccor_list = []
                for t in range(self.horizon):
                    enccor = self.cor_enc(hs[:, t, :])
                    if self.dropout:
                        enccor = nn.Dropout()(enccor)  ####################
                    enccor_list.append(enccor)
            return enccor_list


class TMNet_Tr(nn.Module):  # --------------- Vanilla Transformer
    # Temporal predictor based on Transformer
    def __init__(
        self,
        num_inputs,
        horizon,
        in_channels,
        out_channels,
        n_heads,
        enc_layers,
        dec_layers,
        normalize_before,
        output_dim,
        rnn,
        condi_type,
    ):

        super().__init__()
        nb_convs = len(out_channels)
        self.rnn = rnn
        self.horizon = horizon
        self.backbone = list()
        norm = nn.BatchNorm2d
        cor_custom_stride = [2, 2, (1, 2)]

        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]

            if condi_type == "1":
                self.backbone += [
                    nn.Conv2d(
                        in_ch, out_channels[i], kernel_size=3, padding=1, stride=2
                    )
                ]
            else:
                self.backbone += [
                    nn.Conv2d(
                        in_ch,
                        out_channels[i],
                        kernel_size=3,
                        padding=1,
                        stride=cor_custom_stride[i],
                    )
                ]
            self.backbone += [norm(out_channels[i])]
            self.backbone += [nn.ReLU(True)]

            self.backbone += [
                nn.Conv2d(
                    out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1
                )
            ]
            self.backbone += [norm(out_channels[i])]
            self.backbone += [nn.ReLU(True)]
            self.backbone += [nn.Dropout(0.2)]
        self.backbone = nn.Sequential(*self.backbone)

        hidden_dim = out_channels[-1]
        self.num_frames = num_inputs
        self.num_queries = horizon
        self.query_learn = nn.Embedding(horizon, hidden_dim)
        self.pe3d = PositionalEncodingPermute3D(hidden_dim)
        self.pe_temp = PositionalEncoding1D(hidden_dim)

        self.input_proj = nn.Conv3d(
            self.num_frames, self.num_frames, kernel_size=(1, 8, 8)
        )
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=normalize_before,
            return_intermediate_dec=False,
        )

        self.bn = nn.BatchNorm3d(out_channels[-1])
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, Is):
        img_feats = []
        for i in range(self.num_frames):
            img_feats.append(self.backbone(Is[:, :, i, :, :]))
        img_feats = torch.stack(img_feats, dim=2).permute(0, 2, 1, 3, 4)

        src_proj = self.input_proj(img_feats)
        # ----------------------------------------------------------------
        pos = self.pe3d(src_proj).to(src_proj.dtype)  # [bs, 3, 48, 8, 8]
        src_proj = src_proj.flatten(-3)
        pos = pos.flatten(-3)  # [bs, 3, 48, 64]
        # ----------------------------------------------------------------
        pos_dec_in = self.pe_temp(
            self.query_learn.weight.unsqueeze(0).repeat(src_proj.shape[0], 1, 1)
        ).to(self.query_learn.weight.dtype)
        hs = self.transformer(
            enc_in=src_proj,
            enc_pos=pos,
            dec_in=self.query_learn.weight.unsqueeze(0).repeat(src_proj.shape[0], 1, 1),
            dec_pos=pos_dec_in,
        )
        hs = hs.permute(1, 0, 2)
        # ----------------------------------------------------------------
        out_feats = []
        for t in range(self.horizon):
            out_feats.append(self.linear(hs[:, t, :]))
        return out_feats


def kl_criterion(mu1, logvar1, mu2, logvar2, batch_size):
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = (
        torch.log(sigma2 / sigma1)
        + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2))
        - 1 / 2
    )
    return kld.sum() / batch_size


# ---------------------------------------------------------------------------------
# Temporal predictor - single output using Conv. prior
# ---------------------------------------------------------------------------------
class TMNet_Tr_priort1_conv(nn.Module):

    def __init__(
        self,
        num_inputs,
        horizon,
        in_channels,
        out_channels,
        n_heads,
        enc_layers,
        dec_layers,
        normalize_before,
        output_dim,
        rnn,
        condi_type,
    ):
        super().__init__()
        nb_convs = len(out_channels)
        self.rnn = rnn
        self.horizon = horizon
        self.backbone = list()
        norm = nn.BatchNorm2d
        cor_custom_stride = [2, 2, (1, 2)]

        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]

            if condi_type == "1":
                self.backbone += [
                    nn.Conv2d(
                        in_ch, out_channels[i], kernel_size=3, padding=1, stride=2
                    )
                ]
            else:
                self.backbone += [
                    nn.Conv2d(
                        in_ch,
                        out_channels[i],
                        kernel_size=3,
                        padding=1,
                        stride=cor_custom_stride[i],
                    )
                ]
            self.backbone += [norm(out_channels[i])]
            self.backbone += [nn.ReLU(True)]

            self.backbone += [
                nn.Conv2d(
                    out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1
                )
            ]
            self.backbone += [norm(out_channels[i])]
            self.backbone += [nn.ReLU(True)]

        self.backbone = nn.Sequential(*self.backbone)
        hidden_dim = out_channels[-1]
        self.hidden_dim = hidden_dim
        self.num_frames = num_inputs
        self.num_queries = horizon
        self.query_learn = nn.Embedding(horizon, hidden_dim)
        self.pe3d = PositionalEncodingPermute3D(hidden_dim)
        self.pe_temp = PositionalEncoding1D(hidden_dim)
        self.input_proj = nn.Conv3d(
            self.num_frames, self.num_frames, kernel_size=(1, 8, 8)
        )
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=normalize_before,
            return_intermediate_dec=False,
        )
        self.bn = nn.BatchNorm3d(out_channels[-1])
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.prior_nets = Gaussian_Enc(
            in_ch=3, nb_in=self.num_frames, d_model=hidden_dim, output_size=hidden_dim
        ).to(device)
        self.posterior_nets = Gaussian_Enc(
            in_ch=3 + 1,
            nb_in=self.num_frames,
            d_model=hidden_dim,
            output_size=hidden_dim,
        ).to(device)

    def forward(self, Ipast, Ifuture):

        if Ifuture is not None:  # -----> Train stage
            img_feats_past, img_feats_future = [], []
            # --------------- Backbone -------------------------------------
            for i in range(self.num_frames):
                img_feats_past.append(self.backbone(Ipast[:, :, i, :, :]))
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)
            src_proj = self.input_proj(img_feats_past)

            for j in range(self.num_queries):
                img_feats_future.append(self.backbone(Ifuture[:, :, j, :, :]))
            img_feats_future = torch.stack(img_feats_future, dim=2).permute(
                0, 2, 1, 3, 4
            )
            # ---------------- TR enc ----------------------------------------
            pos = self.pe3d(src_proj).to(src_proj.dtype)  # [bs, 3, 48, 8, 8]
            src_proj = src_proj.flatten(-3)
            pos = pos.flatten(-3)  # [bs, 3, 48, 64]
            # ----------------------------------------------------------------
            _, mu1, logvar1 = self.prior_nets(img_feats_past)
            who = torch.cat([img_feats_past, img_feats_future[:, :, :, :, :]], dim=1)
            z_t, mu2, logvar2 = self.posterior_nets(who)
            kl_loss = kl_criterion(mu1, logvar1, mu2, logvar2, Ipast.shape[0])

            the_queries = self.query_learn.weight.unsqueeze(0).repeat(
                src_proj.shape[0], 1, 1
            )
            dec_input = torch.cat(
                [the_queries, z_t.unsqueeze(1)], dim=1
            )  # the_queries + z_t.unsqueeze(1)  #torch.cat([the_queries[:, 0, :], z_t], dim=1) #
            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)

        else:  # -----> Test stage
            img_feats_past = []
            for i in range(self.num_frames):
                img_feats_past.append(self.backbone(Ipast[:, :, i, :, :]))
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)
            src_proj = self.input_proj(img_feats_past)
            # ----------------------------------------------------------------
            pos = self.pe3d(src_proj).to(src_proj.dtype)  # [bs, 3, 48, 8, 8]
            src_proj = src_proj.flatten(-3)
            pos = pos.flatten(-3)  # [bs, 3, 48, 64]
            # ----------------------------------------------------------------
            z_t, _, _ = self.prior_nets(img_feats_past)
            the_queries = self.query_learn.weight.unsqueeze(0).repeat(
                src_proj.shape[0], 1, 1
            )
            dec_input = torch.cat(
                [the_queries, z_t.unsqueeze(1)], dim=1
            )  #  #torch.cat([the_queries, zt], dim=1)
            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)
            kl_loss = None
        # ----------------------------------------------------------------
        hs = self.transformer(
            enc_in=src_proj, enc_pos=pos, dec_in=dec_input, dec_pos=pos_dec_in
        )
        hs = hs.permute(1, 0, 2)
        # ----------------------------------------------------------------
        return self.linear(hs[:, 0, :]), kl_loss


class Gaussian_Enc(nn.Module):
    def __init__(self, in_ch, nb_in, d_model, output_size):
        super(Gaussian_Enc, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, nb_in, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm3d(nb_in),
            nn.Conv3d(nb_in, nb_in, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm3d(nb_in),
        )
        self.linear = nn.Linear(nb_in * 64 * 8 * 8, d_model)
        self.mu_net = nn.Linear(d_model, output_size)
        self.logvar_net = nn.Linear(d_model, output_size)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = logvar.data.new(logvar.size()).normal_()
        return eps.mul(logvar).add_(mu)

    def forward(self, x_input):
        x = self.encoder(x_input)
        x = x.flatten(1)
        h_in = self.linear(x)  # 10 3 64 8 8
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


# ---------------------------------------------------------------------------------
# Temporal predictor - single output using Attention prior
# ---------------------------------------------------------------------------------
class TMNet_Tr_priort1_att(nn.Module):  # --------------- Transformer + prior
    # Temporal predictor based on Transformer with temporal prior
    def __init__(
        self,
        num_inputs,
        horizon,
        in_channels,
        out_channels,
        n_heads,
        enc_layers,
        dec_layers,
        normalize_before,
        output_dim,
        rnn,
        condi_type,
    ):

        super().__init__()
        nb_convs = len(out_channels)
        self.rnn = rnn
        self.horizon = horizon
        self.backbone = list()
        norm = nn.BatchNorm2d
        cor_custom_stride = [2, 2, (1, 2)]

        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]

            if condi_type == "1":
                self.backbone += [
                    nn.Conv2d(
                        in_ch, out_channels[i], kernel_size=3, padding=1, stride=2
                    )
                ]
            else:
                self.backbone += [
                    nn.Conv2d(
                        in_ch,
                        out_channels[i],
                        kernel_size=3,
                        padding=1,
                        stride=cor_custom_stride[i],
                    )
                ]
            self.backbone += [norm(out_channels[i])]
            self.backbone += [nn.ReLU(True)]

            self.backbone += [
                nn.Conv2d(
                    out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1
                )
            ]
            self.backbone += [norm(out_channels[i])]
            self.backbone += [nn.ReLU(True)]

        self.backbone = nn.Sequential(*self.backbone)
        hidden_dim = out_channels[-1]
        self.hidden_dim = hidden_dim
        self.num_frames = num_inputs
        self.num_queries = horizon
        self.query_learn = nn.Embedding(horizon, hidden_dim)
        self.pe3d = PositionalEncodingPermute3D(hidden_dim)
        self.pe_temp = PositionalEncoding1D(hidden_dim)
        self.input_proj = nn.Conv3d(
            self.num_frames, self.num_frames, kernel_size=(1, 8, 8)
        )
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=normalize_before,
            return_intermediate_dec=False,
        )
        self.bn = nn.BatchNorm3d(out_channels[-1])
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.prior_nets = Gaussian_TR(
            in_ch=3,
            d_model=hidden_dim,
            output_size=hidden_dim,
            n_layers=1,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=normalize_before,
        ).to(device)
        self.posterior_nets = Gaussian_TR(
            in_ch=3 + 1,
            d_model=hidden_dim,
            output_size=hidden_dim,
            n_layers=1,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=normalize_before,
        ).to(device)

    def forward(self, Ipast, Ifuture):

        if Ifuture is not None:  # -----> Train stage
            img_feats_past, img_feats_future = [], []
            # --------------- Backbone -------------------------------------
            for i in range(self.num_frames):
                img_feats_past.append(self.backbone(Ipast[:, :, i, :, :]))
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)
            src_proj = self.input_proj(img_feats_past)

            for j in range(self.num_queries):
                img_feats_future.append(self.backbone(Ifuture[:, :, j, :, :]))
            img_feats_future = torch.stack(img_feats_future, dim=2).permute(
                0, 2, 1, 3, 4
            )
            # ---------------- TR enc ----------------------------------------
            pos = self.pe3d(src_proj).to(src_proj.dtype)  # [bs, 3, 48, 8, 8]
            src_proj = src_proj.flatten(-3)
            pos = pos.flatten(-3)  # [bs, 3, 48, 64]
            # ----------------------------------------------------------------
            _, mu1, logvar1 = self.prior_nets(img_feats_past)
            who = torch.cat([img_feats_past, img_feats_future[:, :, :, :, :]], dim=1)
            z_t, mu2, logvar2 = self.posterior_nets(who)
            kl_loss = kl_criterion(mu1, logvar1, mu2, logvar2, Ipast.shape[0])
            the_queries = self.query_learn.weight.unsqueeze(0).repeat(
                src_proj.shape[0], 1, 1
            )
            dec_input = torch.cat([the_queries, z_t], dim=1)
            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)

        else:  # -----> Test stage
            img_feats_past = []
            for i in range(self.num_frames):
                img_feats_past.append(self.backbone(Ipast[:, :, i, :, :]))
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)
            src_proj = self.input_proj(img_feats_past)
            # ----------------------------------------------------------------
            pos = self.pe3d(src_proj).to(src_proj.dtype)  # [bs, 3, 48, 8, 8]
            src_proj = src_proj.flatten(-3)
            pos = pos.flatten(-3)  # [bs, 3, 48, 64]
            # ----------------------------------------------------------------
            z_t, _, _ = self.prior_nets(img_feats_past)
            the_queries = self.query_learn.weight.unsqueeze(0).repeat(
                src_proj.shape[0], 1, 1
            )
            dec_input = torch.cat([the_queries, z_t], dim=1)
            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)
            kl_loss = None
        # ----------------------------------------------------------------
        hs = self.transformer(
            enc_in=src_proj, enc_pos=pos, dec_in=dec_input, dec_pos=pos_dec_in
        )
        hs = hs.permute(1, 0, 2)
        # ----------------------------------------------------------------
        return self.linear(hs[:, 0, :]), kl_loss


class Gaussian_TR(nn.Module):
    def __init__(
        self,
        in_ch,
        d_model,
        output_size,
        n_layers,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        normalize_before,
    ):
        super(Gaussian_TR, self).__init__()
        self.input_proj = nn.Conv3d(in_ch, in_ch, kernel_size=(1, 8, 8))
        self.pe = PositionalEncodingPermute3D(d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        self.encoder = TransformerEncoder(
            encoder_layer, n_layers, nn.LayerNorm(d_model) if normalize_before else None
        )
        self.mu_net = nn.Linear(d_model, output_size)
        self.logvar_net = nn.Linear(d_model, output_size)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = logvar.data.new(logvar.size()).normal_()
        return eps.mul(logvar).add_(mu)

    def forward(self, x_input):
        src_proj = self.input_proj(x_input)
        # print(f"srcproj gaussian tr : {src_proj.shape}")
        enc_in = src_proj.flatten(-3).permute(1, 0, 2)
        enc_pos = self.pe(src_proj).to(src_proj.dtype).flatten(-3).permute(1, 0, 2)
        h_in = self.encoder(enc_in, pos=enc_pos).permute(1, 0, 2)
        h_in = h_in[:, -1:, :]  # -1
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


# ---------------------------------------------------------------------------------
# Temporal predictor - multiple outputs using Conv. prior
# ---------------------------------------------------------------------------------
class TMNet_Tr_priormulti(nn.Module):
    def __init__(
        self,
        num_inputs,
        horizon,
        in_channels,
        out_channels,
        n_heads,
        enc_layers,
        dec_layers,
        normalize_before,
        output_dim,
        rnn,
        condi_type,
        prior_type,
        device,
    ):

        super().__init__()
        nb_convs = len(out_channels)
        self.rnn = rnn
        self.horizon = horizon
        self.backbone = list()
        norm = nn.BatchNorm2d
        cor_custom_stride = [2, 2, (1, 2)]
        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]

            if condi_type == "1":
                self.backbone += [
                    nn.Conv2d(
                        in_ch, out_channels[i], kernel_size=3, padding=1, stride=2
                    )
                ]
            else:
                stride = cor_custom_stride[i] if i < len(cor_custom_stride) else 1
                self.backbone += [
                    nn.Conv2d(
                        in_ch,
                        out_channels[i],
                        kernel_size=3,
                        padding=1,
                        stride=stride,
                    )
                ]
            self.backbone += [norm(out_channels[i])]
            self.backbone += [nn.ReLU(True)]

            self.backbone += [
                nn.Conv2d(
                    out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1
                )
            ]
            self.backbone += [norm(out_channels[i])]
            self.backbone += [nn.ReLU(True)]
            self.backbone += [nn.Dropout(0.2)]
        self.backbone = nn.Sequential(*self.backbone)
        
                
        # def shape_hook(name):
        #     def hook(module, inp, out):
        #         print(f"{name}: {out.shape}")
        #     return hook

        # for i, layer in enumerate(self.backbone):
        #     if isinstance(layer, nn.Conv2d):
        #         layer.register_forward_hook(shape_hook(f"conv_{i}"))

        hidden_dim = out_channels[-1]
        self.hidden_dim = hidden_dim
        self.num_frames = num_inputs
        self.num_queries = horizon
        self.query_learn = nn.Embedding(horizon, hidden_dim)
        self.pe3d = PositionalEncodingPermute3D(hidden_dim)
        self.pe_temp = PositionalEncoding1D(
            hidden_dim
        )  # hidden_dim   hidden_dim*(self.num_frames+1)

        self.input_proj = nn.Conv3d(
            self.num_frames, self.num_frames, kernel_size=(1, 8, 8)
        )
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=normalize_before,
            return_intermediate_dec=False,
        )
        self.bn = nn.BatchNorm3d(out_channels[-1])
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.prior_type = prior_type

        # self.prior_nets, self.posterior_nets = [], [] # change
        self.prior_nets = nn.ModuleList()
        self.posterior_nets = nn.ModuleList()  # changes ----
        for i in range(self.num_queries):
            if prior_type == "learned_conv":
                self.prior_nets.append(
                    Gaussian_Enc(
                        in_ch=self.num_frames,
                        nb_in=self.num_frames,
                        d_model=hidden_dim,
                        output_size=hidden_dim,
                    ).to(device)
                )
                self.posterior_nets.append(
                    Gaussian_Enc(
                        in_ch=self.num_frames + i + 1,
                        nb_in=self.num_frames,
                        d_model=hidden_dim,
                        output_size=hidden_dim,
                    ).to(device)
                )
            else:
                self.prior_nets.append(
                    Gaussian_TR(
                        in_ch=self.num_frames,
                        d_model=hidden_dim,
                        output_size=hidden_dim,
                        n_layers=1,
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=0.1,
                        activation="relu",
                        normalize_before=normalize_before,
                    ).to(device)
                )
                self.posterior_nets.append(
                    Gaussian_TR(
                        in_ch=self.num_frames + i + 1,
                        d_model=hidden_dim,
                        output_size=hidden_dim,
                        n_layers=1,
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=0.1,
                        activation="relu",
                        normalize_before=normalize_before,
                    ).to(device)
                )

    def forward(self, Ipast, Ifuture, return_hidden=False):
        if Ifuture is not None:
            img_feats_past, img_feats_future = [], []
            # --------------- Backbone -------------------------------------
            # print(f"[TMNet][train] Ipast: {Ipast.shape}, Ifuture: {Ifuture.shape}")
            for i in range(self.num_frames):
                img_feats_past.append(self.backbone(Ipast[:, :, i, :, :]))
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)
            # print(f"[TMNet][train] img_feats_past (post backbone+stack): {img_feats_past.shape}")

            src_proj = self.input_proj(img_feats_past)
            # print(f"[TMNet][train] src_proj (post input_proj): {src_proj.shape}")
            for j in range(self.num_queries):
                img_feats_future.append(self.backbone(Ifuture[:, :, j, :, :]))
            img_feats_future = torch.stack(img_feats_future, dim=2).permute(
                0, 2, 1, 3, 4
            )
            # print(f"[TMNet][train] img_feats_future (post backbone+stack): {img_feats_future.shape}")
            # ---------------- TR enc ----------------------------------------
            pos = self.pe3d(src_proj).to(src_proj.dtype)  # [bs, 3, 48, 8, 8]
            src_proj = src_proj.flatten(-3)
            pos = pos.flatten(-3)  # [bs, 3, 48, 64]
            # print(f"[TMNet][train] enc_in (src_proj): {src_proj.shape}, enc_pos (pos): {pos.shape}")
            # ----------------------------------------------------------------
            kl_loss = 0
            zts = []
            for k in range(self.num_queries):
                _, mu1, logvar1 = self.prior_nets[k](img_feats_past)
                who = torch.cat(
                    [img_feats_past, img_feats_future[:, : k + 1, :, :, :]], dim=1
                )
                z_t, mu2, logvar2 = self.posterior_nets[k](who)
                kl_loss += kl_criterion(mu1, logvar1, mu2, logvar2, Ipast.shape[0])
                zts.append(z_t)
                # print(f"[TMNet][train] k={k}: prior mu1 {mu1.shape}, posterior z_t {z_t.shape}, "
                #       f"running kl_loss={float(kl_loss):.4f}")
            # for k in range(self.num_queries):
            #     if k==0:
            #         _, mu1, logvar1 = self.prior_nets[k](img_feats_past)
            #     else:
            #         who_new = self.feature_projector(torch.cat([img_feats_past, z_t], dim=1)) # map b,1,64 --> b,1,64,8,8            #
            #         _, mu1, logvar1 = self.prior_nets[k](who_new)
            #     who = torch.cat([img_feats_past, img_feats_future[:, :k+1, :, :, :]], dim=1)
            #     z_t, mu2, logvar2 = self.posterior_nets[k](who)
            #     kl_loss += kl_criterion(mu1, logvar1, mu2, logvar2, Ipast.shape[0])
            #     zts.append(z_t)

            the_queries = self.query_learn.weight.unsqueeze(0).repeat(
                src_proj.shape[0], 1, 1
            )
            zts = torch.stack(zts, dim=1)
            # print(f"[TMNet][train] the_queries: {the_queries.shape}, zts (stacked): {zts.shape}")
            if self.prior_type == "learned_conv":
                dec_input = torch.cat([the_queries, zts], dim=1)
            else:
                dec_input = torch.cat([the_queries, zts[:, :, 0, :]], dim=1)
            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)
            # print(f"[TMNet][train] dec_in (dec_input): {dec_input.shape}, dec_pos (pos_dec_in): {pos_dec_in.shape}")
            # ----------------------------------------------------------------

        else:
            img_feats_past = []
            # print(f"[TMNet][infer] Ipast: {Ipast.shape}")
            for i in range(self.num_frames):
                img_feats_past.append(self.backbone(Ipast[:, :, i, :, :]))
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)
            # print(f"[TMNet][infer] img_feats_past (post backbone+stack): {img_feats_past.shape}")

            src_proj = self.input_proj(img_feats_past)
            # print(f"[TMNet][infer] src_proj (post input_proj): {src_proj.shape}")
            # ----------------------------------------------------------------
            pos = self.pe3d(src_proj).to(src_proj.dtype)  # [bs, 3, 48, 8, 8]
            src_proj = src_proj.flatten(-3)
            pos = pos.flatten(-3)  # [bs, 3, 48, 64]
            # print(f"[TMNet][infer] enc_in (src_proj): {src_proj.shape}, enc_pos (pos): {pos.shape}")
            # ----------------------------------------------------------------
            zts = []
            for k in range(self.num_queries):
                z_t, mu1, logvar1 = self.prior_nets[k](img_feats_past)
                zts.append(z_t)
                # print(f"[TMNet][infer] k={k}: prior z_t {z_t.shape}")

            the_queries = self.query_learn.weight.unsqueeze(0).repeat(
                src_proj.shape[0], 1, 1
            )
            zts = torch.stack(zts, dim=1)
            # print(f"[TMNet][infer] the_queries: {the_queries.shape}, zts (stacked): {zts.shape}")
            if self.prior_type == "learned_conv":
                dec_input = torch.cat([the_queries, zts], dim=1)
            else:
                dec_input = torch.cat([the_queries, zts[:, :, 0, :]], dim=1)
            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)
            # print(f"[TMNet][infer] dec_in (dec_input): {dec_input.shape}, dec_pos (pos_dec_in): {pos_dec_in.shape}")
            # ----------------------------------------------------------------
            kl_loss = None

        # ----------------------------------------------------------------
        hs = self.transformer(
            enc_in=src_proj, enc_pos=pos, dec_in=dec_input, dec_pos=pos_dec_in
        )
        # print(f"[TMNet] hs (transformer output, pre-permute): {hs.shape}")
        hs = hs.permute(1, 0, 2)
        # ----------------------------------------------------------------
        out_feats = []
        for t in range(self.horizon):
            out_feats.append(self.linear(hs[:, t, :]))
        kl_str = "None" if kl_loss is None else f"{float(kl_loss):.4f}"
        # print(f"[TMNet] out_feats: {len(out_feats)} tensors, each {tuple(out_feats[0].shape)}, kl_loss={kl_str}")
        
        if return_hidden:
            hidden_feats = [hs[:, t, :] for t in range(self.horizon)]
            return out_feats, kl_loss, hidden_feats
        return out_feats, kl_loss

class TMNet_Tr_priormulti_mask(TMNet_Tr_priormulti):
    """
    Drop-in replacement for TMNet_Tr_priormulti that uses a
    src_key_padding_mask derived from zero-padded frames (produced by
    variable_frames augmentation).  Masked positions are ignored by both
    the encoder self-attention and the decoder cross-attention, so the
    model never attends to empty input frames.

    No changes to __init__ — identical hyper-parameters.
    """

    def forward(self, Ipast: torch.Tensor, Ifuture):
        # ---- Build padding mask from zero frames -------------------------
        # Ipast: (B, C, T, H, W) — frame t is masked when all values == 0
        B, C, T, H, W = Ipast.shape
        frame_energy         = Ipast.abs().sum(dim=(1, 3, 4))       # (B, T)
        src_key_padding_mask = (frame_energy == 0).to(Ipast.device) # (B, T) True=ignore

        if Ifuture is not None:
            img_feats_past, img_feats_future = [], []
            for i in range(self.num_frames):
                img_feats_past.append(self.backbone(Ipast[:, :, i, :, :]))
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)

            src_proj = self.input_proj(img_feats_past)
            for j in range(self.num_queries):
                img_feats_future.append(self.backbone(Ifuture[:, :, j, :, :]))
            img_feats_future = torch.stack(img_feats_future, dim=2).permute(0, 2, 1, 3, 4)

            pos      = self.pe3d(src_proj).to(src_proj.dtype)
            src_proj = src_proj.flatten(-3)
            pos      = pos.flatten(-3)

            kl_loss = 0
            zts = []
            for k in range(self.num_queries):
                _, mu1, logvar1 = self.prior_nets[k](img_feats_past)
                who = torch.cat([img_feats_past, img_feats_future[:, : k + 1, :, :, :]], dim=1)
                z_t, mu2, logvar2 = self.posterior_nets[k](who)
                kl_loss += kl_criterion(mu1, logvar1, mu2, logvar2, B)
                zts.append(z_t)

            the_queries = self.query_learn.weight.unsqueeze(0).repeat(src_proj.shape[0], 1, 1)
            zts = torch.stack(zts, dim=1)
            if self.prior_type == "learned_conv":
                dec_input = torch.cat([the_queries, zts], dim=1)
            else:
                dec_input = torch.cat([the_queries, zts[:, :, 0, :]], dim=1)
            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)

        else:
            img_feats_past = []
            for i in range(self.num_frames):
                img_feats_past.append(self.backbone(Ipast[:, :, i, :, :]))
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)

            src_proj = self.input_proj(img_feats_past)
            pos      = self.pe3d(src_proj).to(src_proj.dtype)
            src_proj = src_proj.flatten(-3)
            pos      = pos.flatten(-3)

            zts = []
            for k in range(self.num_queries):
                z_t, mu1, logvar1 = self.prior_nets[k](img_feats_past)
                zts.append(z_t)

            the_queries = self.query_learn.weight.unsqueeze(0).repeat(src_proj.shape[0], 1, 1)
            zts = torch.stack(zts, dim=1)
            if self.prior_type == "learned_conv":
                dec_input = torch.cat([the_queries, zts], dim=1)
            else:
                dec_input = torch.cat([the_queries, zts[:, :, 0, :]], dim=1)
            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)
            kl_loss = None

        # ---- Transformer with mask ---------------------------------------
        # Call encoder and decoder directly so we can inject the mask,
        # without modifying the shared Transformer.forward signature.
        enc_in  = src_proj.permute(1, 0, 2)   # (T, B, hidden)
        enc_pos = pos.permute(1, 0, 2)
        enc_out = self.transformer.encoder(
            enc_in,
            pos=enc_pos,
            src_key_padding_mask=src_key_padding_mask,
        )

        dec_in  = dec_input.permute(1, 0, 2)
        dec_pos = pos_dec_in.permute(1, 0, 2)
        hs = self.transformer.decoder(
            dec_in,
            enc_out,
            pos=enc_pos,
            query_pos=dec_pos,
            memory_key_padding_mask=src_key_padding_mask,
        )
        hs = hs.permute(1, 0, 2)

        out_feats = [self.linear(hs[:, t, :]) for t in range(self.horizon)]
        return out_feats, kl_loss


# =============================================================================
# Projector + TMNet_Tr_priormulti_image  (Strategy B)
# =============================================================================

# Backbone spatial output size — always 8×8 for both sagittal and coronal.
_FEAT_SIZE = 8


class Projector(nn.Module):
    """
    CNN decoder: (B, in_ch, T, 8, 8) → DVF_seq (B, 2, T, H, W) + I_seq (B, 1, T, H, W).

    Processes each timestep with shared 2-D convolutions, starting from the
    8×8 spatial structure produced by the backbone (hidden_dim = 8×8 = 64).
    Three upsampling stages with channel reduction [64, 32, 16] → 2 (DVF).
    DVF is fed to a SpatialTransformer to warp Iref → I_seq.

    Stage sizes (bilinear targets):
      sagittal 64×64: 8→16→32→64
      coronal  32×64: 8→(8×16)→(16×32)→(32×64)

    Parameters
    ----------
    in_ch              : hidden_dim // (8×8)  (= 1 when hidden_dim=64)
    out_channels       : channel progression, e.g. [64, 32, 16]
    img_h, img_w       : target slice dimensions
    spatial_transformer: SpatialTransformer instance (shared, no extra params)
    """

    def __init__(
        self,
        in_ch:              int,
        out_channels:       list,
        img_h:              int,
        img_w:              int,
        spatial_transformer,
    ):
        super().__init__()
        # Three bilinear target sizes going from 8×8 → H×W
        sz = [
            (max(_FEAT_SIZE, img_h // 4), max(_FEAT_SIZE, img_w // 4)),
            (img_h // 2, img_w // 2),
            (img_h, img_w),
        ]
        layers = []
        prev_ch = in_ch
        for ch, target in zip(out_channels, sz):
            layers += [
                nn.Upsample(size=target, mode='bilinear', align_corners=True),
                nn.Conv2d(prev_ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
            ]
            prev_ch = ch
        layers += [nn.Conv2d(prev_ch, 2, 3, padding=1)]   # → DVF (2 channels)
        self.decoder = nn.Sequential(*layers)
        self.st = spatial_transformer
        
        # def shape_hook(name):
        #     def hook(module, inp, out):
        #         print(f"{name}: {out.shape}")
        #     return hook

        # for i, layer in enumerate(self.decoder):
        #     if isinstance(layer, nn.Conv2d):
        #         layer.register_forward_hook(shape_hook(f"conv_decoder_{i}"))

    def forward(self, x: torch.Tensor, Iref: torch.Tensor):
        """
        x    : (B, in_ch, T, 8, 8)
        Iref : (B, 1, H, W)
        Returns DVF_seq (B, 2, T, H, W), I_seq (B, 1, T, H, W)
        """
        T = x.shape[2]
        dvfs, preds = [], []
        for t in range(T):
            dvf  = self.decoder(x[:, :, t])    # (B, 2, H, W)
            # print(f"decoder output : {dvf.shape}")
            pred = self.st(Iref, dvf)           # (B, 1, H, W)
            # print(f"pred : {pred.shape}")
            dvfs.append(dvf)
            preds.append(pred)
        return torch.stack(dvfs, dim=2), torch.stack(preds, dim=2)


class TMNet_Tr_priormulti_image(TMNet_Tr_priormulti):
    """
    TMNet_Tr_priormulti with Strategy-B image-supervision head.

    Key change vs the base class:
      - self.linear is removed.
      - The transformer decoder output hs is reshaped back to the 8×8 spatial
        structure it came from (hidden_dim = 8×8), then fed to a CNN Projector
        that upsamples to full resolution DVF + applies SpatialTransformer(Iref).

    Forward signature
    -----------------
    forward(Iref, Ipast, Ifuture=None) → (DVF_seq, I_seq, kl_loss, phi_pred, amp_pred, motion_map_pred, z_contrast)

      Iref    : (B, 1, H, W)     reference slice for warping
      Ipast   : (B, 2, T, H, W)  past frames with reference concatenated
      Ifuture : (B, 2, Q, H, W)  future frames with reference concatenated (train only)

    Output shapes
    -------------
      DVF_seq    : (B, 2, T', H, W)
      I_seq      : (B, 1, T', H, W)
      kl_loss    : scalar Tensor or None
      z_contrast : (B, horizon, 32) or None — per-horizon-step embedding for
                   PhaseContrastiveLoss; flatten to (B*horizon, 32) before use.
    """

    def __init__(
        self,
        num_inputs,
        horizon,
        in_channels,
        out_channels,
        n_heads,
        enc_layers,
        dec_layers,
        normalize_before,
        output_dim,          # kept for interface compat; not used by this head
        rnn,
        condi_type,
        prior_type,
        device,
        img_h:          int,
        img_w:          int,
        proj_channels:       list = None,
        use_phi_loss:        bool = False,
        phi_head_dim:        int  = 1,   # 1 = scalar phi (liver default); 2 = (cos,sin) direction-aware (ACDC)
        use_amp_loss:        bool = False,
        use_motion_loss:     bool = False,
        motion_pool_h:       int  = 4,
        motion_pool_w:       int  = 8,
        use_phase_contrastive: bool = False,
    ):
        super().__init__(
            num_inputs=num_inputs, horizon=horizon, in_channels=in_channels,
            out_channels=out_channels, n_heads=n_heads,
            enc_layers=enc_layers, dec_layers=dec_layers,
            normalize_before=normalize_before, output_dim=out_channels[-1],
            rnn=rnn, condi_type=condi_type, prior_type=prior_type, device=device,
        )
        # Remove the flat linear head — not used by this variant
        del self.linear

        hidden_dim  = out_channels[-1]
        in_ch       = hidden_dim // (_FEAT_SIZE * _FEAT_SIZE)   # = 1 when hidden_dim=64
        proj_ch     = proj_channels or [64, 32, 16]
        spatial_transformer = SpatialTransformer((img_h, img_w))
        self.projector = Projector(in_ch, proj_ch, img_h, img_w, spatial_transformer)

        # Optional auxiliary regression heads trained jointly with the image loss.
        self.phi_head = nn.Linear(hidden_dim, phi_head_dim) if use_phi_loss else None
        self.amp_head = nn.Linear(hidden_dim, 1) if use_amp_loss else None

        self.contrast_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 32)
        ) if use_phase_contrastive else None
                
        # Motion map reconstruction head: decodes h_pool back to a spatially-pooled
        # version of |c_last − c_ref|.  Forces h_t to encode the displacement map at
        # the same spatial resolution as motion_enc's bottleneck (default 4×8).
        # When this loss is used during pretraining, motion_enc in the downstream
        # diffusion model becomes redundant.
        self._motion_pool_h = motion_pool_h
        self._motion_pool_w = motion_pool_w
        self.motion_dec = (
            nn.Linear(hidden_dim, motion_pool_h * motion_pool_w)
            if use_motion_loss else None
        )
        
        print(f"Losses used: phi={use_phi_loss}, amp={use_amp_loss}, motion={use_motion_loss}, contrastive={use_phase_contrastive}")

    def forward(
        self,
        Iref:    torch.Tensor,
        Ipast:   torch.Tensor,
        Ifuture: torch.Tensor = None,
    ):
        B = Ipast.shape[0]
        
        # print(f"input shape : {Ipast.shape}")

        if Ifuture is not None:
            img_feats_past, img_feats_future = [], []
            for i in range(self.num_frames):
                img_feats_past.append(self.backbone(Ipast[:, :, i, :, :]))
            # print(f"img feats past : {img_feats_past[0].shape}")
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)
            # # print(f"img feats past stacked : {img_feats_past.shape}")

            src_proj = self.input_proj(img_feats_past)
            # print(f"src_proj : {src_proj.shape}")

            for j in range(self.num_queries):
                img_feats_future.append(self.backbone(Ifuture[:, :, j, :, :]))
            img_feats_future = torch.stack(img_feats_future, dim=2).permute(0, 2, 1, 3, 4)
            # print(f"img_feats_future : {img_feats_future[0].shape}")

            pos      = self.pe3d(src_proj).to(src_proj.dtype)
            # print(f"pos : {pos.shape}")
            src_proj = src_proj.flatten(-3)
            pos      = pos.flatten(-3)

            kl_loss = 0
            zts = []
            for k in range(self.num_queries):
                _, mu1, logvar1 = self.prior_nets[k](img_feats_past)
                who = torch.cat([img_feats_past, img_feats_future[:, :k + 1, :, :, :]], dim=1)
                z_t, mu2, logvar2 = self.posterior_nets[k](who)
                kl_loss += kl_criterion(mu1, logvar1, mu2, logvar2, B)
                zts.append(z_t)

            the_queries = self.query_learn.weight.unsqueeze(0).repeat(B, 1, 1)
            # print(f"the_queries : {the_queries.shape}")
            zts = torch.stack(zts, dim=1)
            # print(f"zts : {zts.shape}")
            if self.prior_type == "learned_conv":
                dec_input = torch.cat([the_queries, zts], dim=1)
            else:
                dec_input = torch.cat([the_queries, zts[:, :, 0, :]], dim=1)
            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)

        else:
            img_feats_past = []
            for i in range(self.num_frames):
                img_feats_past.append(self.backbone(Ipast[:, :, i, :, :]))
            
            # print(f"img feats past : {img_feats_past[0].shape}")
            
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)
            # print(f"img feats past stacked : {img_feats_past.shape}")

            src_proj = self.input_proj(img_feats_past)
            # print(f"src_proj : {src_proj.shape}")

            pos = self.pe3d(src_proj).to(src_proj.dtype)
            # print(f"pos : {pos.shape}")
            
            src_proj = src_proj.flatten(-3)
            pos = pos.flatten(-3)

            zts = []
            for k in range(self.num_queries):
                z_t, _, _ = self.prior_nets[k](img_feats_past)
                zts.append(z_t)

            the_queries = self.query_learn.weight.unsqueeze(0).repeat(B, 1, 1)
            # # print(f"the_queries : {the_queries.shape}")

            zts = torch.stack(zts, dim=1)
            # print(f"zts : {zts.shape}")

            if self.prior_type == "learned_conv":
                dec_input = torch.cat([the_queries, zts], dim=1)
            else:
                dec_input = torch.cat([the_queries, zts[:, :, 0, :]], dim=1)

            # print(f"dec_input : {dec_input.shape}")

            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)
            # print(f"pos_dec_in : {pos_dec_in.shape}")

            kl_loss = None

        hs = self.transformer(enc_in=src_proj, enc_pos=pos, dec_in=dec_input, dec_pos=pos_dec_in)
        # print(f"hs : {hs.shape}")

        # Strategy B: reshape hs back to the 8×8 spatial structure it came from.
        # hidden_dim = 64 = 8×8, so each token is naturally a (1, 8, 8) spatial map.
        x = hs.permute(1, 0, 2)                                   # (B, 2*horizon, hidden_dim)
        x = x[:, :self.horizon, :]                                 # (B, horizon,   hidden_dim)

        # Auxiliary predictions: per-horizon token → (B, horizon) each.
        # x is (B, horizon, hidden_dim); apply head independently per slot so
        # the gradient teaches each h_t[k] to encode the phase at t+k, not the
        # pooled average.  amp/motion still pool (no per-horizon GT available).
        h_pool = x.mean(dim=1)
        phi_pred = self.phi_head(x).squeeze(-1) if self.phi_head is not None else None
        amp_pred = self.amp_head(h_pool).squeeze(-1) if self.amp_head is not None else None
        motion_map_pred = (
            self.motion_dec(h_pool).view(B, 1, self._motion_pool_h, self._motion_pool_w)
            if self.motion_dec is not None else None
        )
        z_contrast = self.contrast_proj(x) if self.contrast_proj is not None else None


        x = x.view(B, self.horizon, -1, _FEAT_SIZE, _FEAT_SIZE)   # (B, horizon, C, 8, 8)
        x = x.permute(0, 2, 1, 3, 4)                              # (B, C, horizon, 8, 8)

        DVF_seq, I_seq = self.projector(x, Iref)
        # print(f"DVF_seq : {DVF_seq.shape}")
        # print(f"I_seq : {I_seq.shape}")
        return DVF_seq, I_seq, kl_loss, phi_pred, amp_pred, motion_map_pred, z_contrast

    @torch.no_grad()
    def encode(self, Ipast: torch.Tensor) -> list:
        """
        Run inference-time encoding (prior path, no Ifuture).
        Returns a list of `horizon` tensors (B, hidden_dim) — the transformer
        decoder output before the projector — for downstream conditioning stats.
        """
        img_feats_past = []
        for i in range(self.num_frames):
            img_feats_past.append(self.backbone(Ipast[:, :, i, :, :]))
        img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)
        src_proj = self.input_proj(img_feats_past)

        pos      = self.pe3d(src_proj).to(src_proj.dtype)
        src_proj = src_proj.flatten(-3)
        pos      = pos.flatten(-3)

        zts = []
        for k in range(self.num_queries):
            z_t, _, _ = self.prior_nets[k](img_feats_past)
            zts.append(z_t)

        the_queries = self.query_learn.weight.unsqueeze(0).repeat(src_proj.shape[0], 1, 1)
        zts = torch.stack(zts, dim=1)
        if self.prior_type == "learned_conv":
            dec_input = torch.cat([the_queries, zts], dim=1)
        else:
            dec_input = torch.cat([the_queries, zts[:, :, 0, :]], dim=1)
        pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)

        hs = self.transformer(enc_in=src_proj, enc_pos=pos, dec_in=dec_input, dec_pos=pos_dec_in)
        hs = hs.permute(1, 0, 2)                       # (B, 2*horizon, hidden_dim)
        return [hs[:, t, :] for t in range(self.horizon)]

class PhaseContrastiveLoss(nn.Module):
    """Soft supervised contrastive loss for a circular label (cardiac/respiratory phase).
    Uses a von Mises–style kernel on phase distance instead of a hard pos/neg split."""
    def __init__(self, temperature=0.1, kappa=8.0):
        super().__init__()
        self.temperature = temperature
        self.kappa = kappa  # concentration of the phase kernel; larger = sharper "same phase"

    def forward(self, z, cos_phi, sin_phi):
        z = F.normalize(z, dim=-1)
        sim = torch.matmul(z, z.T) / self.temperature

        cos_diff = cos_phi[:, None]*cos_phi[None, :] + sin_phi[:, None]*sin_phi[None, :]
        weight = torch.exp(self.kappa * (cos_diff - 1.0))  # von Mises kernel, in [0,1], peak at same phase

        N = z.shape[0]
        eye = torch.eye(N, device=z.device, dtype=torch.bool)
        weight = weight.masked_fill(eye, 0.0)

        exp_sim = torch.exp(sim).masked_fill(eye, 0.0)
        numerator   = (exp_sim * weight).sum(1)
        denominator = exp_sim.sum(1)

        has_signal = weight.sum(1) > 1e-4
        loss = -torch.log(numerator[has_signal] / (denominator[has_signal] + 1e-8) + 1e-8)
        return loss.mean() if has_signal.any() else torch.zeros((), device=z.device)

# =============================================================================
# TMNetEncoder — simplified deterministic encoder (no prior / posterior)
# =============================================================================

class TMNetEncoder(nn.Module):
    """
    Deterministic conditioning encoder: backbone + transformer encoder + linear.

    Replaces TMNet_Tr_priormulti when the encoder is used as a frozen feature
    extractor for the diffusion model.  Removes the prior, posterior, and
    transformer decoder entirely; all components are trained by MAE pretraining
    and frozen during CLDM training.

    Returns (out_feats, None) to keep the same interface as
    TMNet_Tr_priormulti so UNet3D needs no structural changes.
    """

    def __init__(
        self,
        num_inputs,
        horizon,
        in_channels,
        out_channels,
        n_heads,
        enc_layers,
        normalize_before,
        output_dim,
        condi_type,
        device,
    ):
        super().__init__()
        nb_convs        = len(out_channels)
        self.horizon    = horizon
        self.num_frames = num_inputs

        # ---- Backbone (identical to TMNet_Tr_priormulti) -----------------
        # GroupNorm instead of BatchNorm: BN running stats computed on
        # augmented (zero-padded) training frames diverge from clean val frames.
        norm = lambda ch: nn.GroupNorm(min(8, ch), ch)
        cor_custom_stride = [2, 2, (1, 2)]
        backbone = []
        for i in range(nb_convs):
            in_ch = in_channels if i == 0 else out_channels[i - 1]
            stride = 2 if condi_type == "1" else cor_custom_stride[i]
            backbone += [nn.Conv2d(in_ch, out_channels[i], 3, padding=1, stride=stride)]
            backbone += [norm(out_channels[i]), nn.ReLU(True)]
            backbone += [nn.Conv2d(out_channels[i], out_channels[i], 3, padding=1)]
            backbone += [norm(out_channels[i]), nn.ReLU(True)]
            backbone += [nn.Dropout(0.2)]
        self.backbone = nn.Sequential(*backbone)

        hidden_dim       = out_channels[-1]
        self.hidden_dim  = hidden_dim

        # ---- Input projection + positional encoding (same as priormulti) ----
        self.input_proj = nn.Conv3d(num_inputs, num_inputs, kernel_size=(1, 8, 8))
        self.pe3d       = PositionalEncodingPermute3D(hidden_dim)

        # ---- Transformer encoder only (reuses existing TransformerEncoder) --
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, n_heads, dim_feedforward=2048,
            dropout=0.2, activation="relu", normalize_before=normalize_before,
        )
        encoder_norm             = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.transformer_encoder = TransformerEncoder(encoder_layer, enc_layers, encoder_norm)

        # ---- Output projection ----------------------------------------------
        self.linear = nn.Linear(hidden_dim, output_dim)

    # ------------------------------------------------------------------

    def encode_temporal(self, frames, attn_mask=None) -> torch.Tensor:
        """
        Encode a sequence of frames through backbone + transformer encoder.

        Parameters
        ----------
        frames : (B, C, T, H, W)

        Returns
        -------
        (B, T, hidden_dim)
        """
        B, C, T, H, W = frames.shape

        # print(f"input shape: {frames[:, :, 0].shape}")
        feats   = [self.backbone(frames[:, :, t]) for t in range(T)]
        feats   = torch.stack(feats, dim=1)               # (B, T, ch, h, w)
        # print(f"feat : {feats.shape}")

        src     = self.input_proj(feats)                  # (B, T, ch, h', w')
        # print(f"src: {src.shape}")
        pos     = self.pe3d(src).to(src.dtype)
        # # print(f"pos: {pos.shape}")
        src_flat = src.flatten(-3)                        # (B, T, hidden_dim)
        pos_flat = pos.flatten(-3)
        # print(f"pos_flat: {pos_flat.shape}")

        frame_energy         = frames.abs().sum(dim=(1, 3, 4))  # (B, T)
        src_key_padding_mask = (frame_energy == 0).to(frames.device)
        if not src_key_padding_mask.any():
            src_key_padding_mask = None

        enc_in  = src_flat.permute(1, 0, 2)              # (T, B, hidden_dim)
        enc_pos = pos_flat.permute(1, 0, 2)
        enc_out = self.transformer_encoder(
            enc_in,
            mask=attn_mask,          # <-- causal mask here
            pos=enc_pos,
            src_key_padding_mask=src_key_padding_mask,
        )                                              # (T, B, hidden_dim)
        # print(f"enc_out : {enc_out.shape}")
        return enc_out.permute(1, 0, 2)                  # (B, T, hidden_dim)

    def forward(self, Ipast: torch.Tensor):
        """
        Parameters
        ----------
        Ipast : (B, C, T, H, W)

        Returns
        -------
        out_feats : list of `horizon` tensors, each (B, output_dim)
        None      : placeholder for KL loss (interface compatibility)
        """
        # hs     = self.encode_temporal(Ipast)              # (B, T, hidden_dim)
        # pooled = hs.mean(dim=1)                           # (B, hidden_dim)
        # out    = self.linear(pooled)                      # (B, output_dim)
        # return [out] * self.horizon, None
        
        hs = self.encode_temporal(Ipast)                  # (B, T, hidden_dim)
        frame_energy = Ipast.abs().sum(dim=(1, 3, 4))     # (B, T)
        valid        = (frame_energy > 0).long()           # (B, T)  1=valid, 0=padded
        last_idx     = valid.cumsum(dim=1).argmax(dim=1)   # (B,) — index of last valid frame
        pooled       = hs[torch.arange(hs.shape[0], device=hs.device), last_idx]  # (B, hidden_dim)

        out = self.linear(pooled)                          # (B, output_dim)
        return [out] * self.horizon, None


import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        # self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, enc_in, enc_pos, dec_in, dec_pos):
        bs, ch, xyz = enc_in.shape
        # dec_in, dec_pos: [bs, t, ch]
        enc_in = enc_in.permute(1, 0, 2)  # 201
        enc_pos = enc_pos.permute(1, 0, 2)  # [tgt_len, bsz, embed_dim]
        enc_out = self.encoder(enc_in, pos=enc_pos)  # [192 8 48]

        dec_in = dec_in.permute(1, 0, 2)
        dec_pos = dec_pos.permute(1, 0, 2)
        hs = self.decoder(dec_in, enc_out, pos=enc_pos, query_pos=dec_pos)
        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # print("src:", src.shape)
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
