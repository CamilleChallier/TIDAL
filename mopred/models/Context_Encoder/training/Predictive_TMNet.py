"""
Pretraining wrappers for TM-Net (temporal context encoder).
``DVFSupTMNet`` (final variant) supervises TMNetEncoder with DVF slices;
``PredictiveTMNet`` uses next-frame prediction; ``MopTRTMNet`` uses multi-step
future prediction. After pretraining only TMNetEncoder is kept — the decoder
scaffolds are discarded.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..temporal_augmentations import TemporalAugConfig, apply_pixel_space_augmentations


# ---------------------------------------------------------------------------
# Multi-step decoder  (MopTR-style training scaffold — discarded after pretraining)
# ---------------------------------------------------------------------------

class _MultiStepDecoder(nn.Module):
    """
    Maps last encoder hidden state (B, hidden_dim) → Q future frames (B, Q, H, W).

    Parameters
    ----------
    hidden_dim  : TMNetEncoder hidden dimension
    num_queries : number of future frames to predict (= cfg.tp)
    img_size    : (H, W) of a single output frame
    """

    def __init__(self, hidden_dim: int, num_queries: int, img_size: tuple):
        super().__init__()
        H, W = img_size
        self.num_queries = num_queries
        self.H = H
        self.W = W
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 4, num_queries * H * W),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, hidden_dim) → (B, Q, H, W)"""
        B = h.shape[0]
        return self.net(h).view(B, self.num_queries, self.H, self.W)


# ---------------------------------------------------------------------------
# Per-frame linear decoder  (training scaffold — discarded after pretraining)
# ---------------------------------------------------------------------------

class _FrameDecoder(nn.Module):
    """
    Maps a per-frame hidden state (B, hidden_dim) → (B, img_channels, H, W).

    Two-layer MLP → reshape.  Intentionally lightweight so the encoder must
    carry the predictive signal rather than the decoder.

    Parameters
    ----------
    hidden_dim   : transformer hidden dimension (= TMNetEncoder.hidden_dim)
    img_channels : number of output channels (typically 2)
    img_size     : (H, W) of a single frame
    """

    def __init__(
        self,
        hidden_dim:   int,
        img_channels: int,
        img_size:     tuple[int, int],
    ):
        super().__init__()
        H, W = img_size
        self.H = H
        self.W = W
        self.C = img_channels
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 4, img_channels * H * W),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, hidden_dim)  →  (B, C, H, W)"""
        return self.net(h).view(h.shape[0], self.C, self.H, self.W)


# ---------------------------------------------------------------------------
# PredictiveTMNet
# ---------------------------------------------------------------------------

class PredictiveTMNet(nn.Module):
    """
    Next-frame prediction pretraining wrapper for TMNetEncoder.

    Parameters
    ----------
    tm_net    : TMNetEncoder instance
    img_size     : (H, W) of a single input frame
    img_channels : C (channels per frame, typically 2)
    """

    def __init__(
        self,
        tm_net,
        img_size:     tuple[int, int],
        img_channels: int,
    ):
        super().__init__()
        self.condi   = tm_net
        self.decoder = _FrameDecoder(tm_net.hidden_dim, img_channels, img_size)

    # ------------------------------------------------------------------
    # Forward  (training / validation)
    # ------------------------------------------------------------------

    def forward(
        self,
        frames:   torch.Tensor,
        aug_cfg:  Optional[TemporalAugConfig] = None,
        training: bool = True,
    ) -> dict:
        """
        Parameters
        ----------
        frames   : (B, C, T, H, W)
        aug_cfg  : optional temporal augmentation config
        training : controls stochastic augmentations

        Returns
        -------
        dict with keys:
            loss  : scalar MSE averaged over all valid next-frame predictions
            preds : list of T-1 predicted frames, each (B, C, H, W) or None
                    when the target frame was fully zero-padded
            feat  : encoder hidden states (B, T, hidden_dim)
        """
        B, C, T, H, W = frames.shape
        device = frames.device

        if aug_cfg is not None:
            frames = apply_pixel_space_augmentations(frames, aug_cfg, training)

        # Encode all T frames at once
        T = frames.shape[2]
        causal_mask = torch.triu(
            torch.ones(T, T, device=frames.device), diagonal=1
        ).bool()  # (T, T) — True = blocked

        hs = self.condi.encode_temporal(frames, attn_mask=causal_mask)

        target_seq = frames.permute(0, 2, 1, 3, 4)    # (B, T, C, H, W)

        total_loss = torch.tensor(0.0, device=device)
        n_valid    = 0
        preds: list = []

        for t in range(T - 1):
            # Exclude zero-padded target frames from the loss
            energy = target_seq[:, t + 1].abs().sum(dim=(1, 2, 3))  # (B,)
            valid  = energy > 0                                       # (B,)
            if not valid.any():
                preds.append(None)
                continue

            pred = self.decoder(hs[:, t])              # (B, C, H, W)
            preds.append(pred)

            total_loss = total_loss + F.mse_loss(
                pred[valid], target_seq[:, t + 1][valid]
            )
            n_valid += 1

        loss = total_loss / max(n_valid, 1)

        return {"loss": loss, "preds": preds, "feat": hs}

    # ------------------------------------------------------------------
    # Inference — full TMNet forward (no decoder involved)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        sag: Optional[torch.Tensor] = None,
        cor: Optional[torch.Tensor] = None,
    ) -> list:
        """
        Run TMNetEncoder inference.

        Returns
        -------
        list of horizon tensors, each (B, output_dim)
        """
        frames = sag if sag is not None else cor
        if frames is None:
            raise ValueError("sag or cor must be provided")
        out_feats, _ = self.condi(frames)
        return out_feats


# ---------------------------------------------------------------------------
# DVFSupTMNet — DVF-slice supervision pretraining wrapper
# ---------------------------------------------------------------------------

class DVFSupTMNet(nn.Module):
    """
    DVF-supervised pretraining wrapper for TMNetEncoder.

    Trains the encoder to predict a 2-D slice of the full 3-D DVF at the
    navigator plane.  The supervision target is computed on-the-fly from a
    frozen VoxelMorph model: DVF = VM(ref_vol, target_vol), then sliced at
    the navigator plane:
      condi_type "2" (coronal):   dvf[:, :, :, 32, :]  → (B, 3, D, W)
      condi_type "1" (sagittal):  dvf[:, :, 16, :, :]  → (B, 3, H, W)

    No teacher forcing — the encoder sees only past frames, exactly as at
    test time.  No KL term — pure MSE supervision on the DVF slice.

    Architecture
    ------------
      frames (B, 2, T, H, W)
        → encode_temporal   → hs (B, T, hidden_dim)
        → mean over T       → h  (B, hidden_dim)
        → DVFDecoder (MLP)  → dvf_pred (B, 3, dvf_h, dvf_w)
      loss = MSE(dvf_pred, dvf_gt_slice)

    After pretraining only TMNetEncoder is kept; the decoder is discarded.
    """

    def __init__(
        self,
        tm_net,
        dvf_h:  int,
        dvf_w:  int,
        dvf_ch: int = 3,
    ):
        super().__init__()
        self.condi  = tm_net
        self.dvf_h  = dvf_h
        self.dvf_w  = dvf_w
        self.dvf_ch = dvf_ch

        hidden_dim = tm_net.hidden_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, dvf_ch * dvf_h * dvf_w),
        )

    def forward(
        self,
        frames:  torch.Tensor,
        dvf_gt:  Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Parameters
        ----------
        frames  : (B, 2, T, H, W) — past navigator slices
        dvf_gt  : (B, 3, dvf_h, dvf_w) — navigator-plane DVF slice, or None

        Returns
        -------
        dict: loss (None at inference), dvf_pred, h
        """
        hs = self.condi.encode_temporal(frames)    # (B, T, hidden_dim)
        h  = hs.mean(dim=1)                        # (B, hidden_dim)
        print(f"before decorder : {h.shape}")
        dvf_pred = self.decoder(h).view(
            h.shape[0], self.dvf_ch, self.dvf_h, self.dvf_w,
        )
        print(f"after dec : {dvf_pred.shape}")
        loss = F.mse_loss(dvf_pred, dvf_gt) if dvf_gt is not None else None
        return {"loss": loss, "dvf_pred": dvf_pred, "h": h}

    @torch.no_grad()
    def encode(
        self,
        sag: Optional[torch.Tensor] = None,
        cor: Optional[torch.Tensor] = None,
    ) -> list:
        frames = sag if sag is not None else cor
        if frames is None:
            raise ValueError("sag or cor must be provided")
        out_feats, _ = self.condi(frames)
        return out_feats


# ---------------------------------------------------------------------------
# MopTRTMNet — MopTR-style image prediction training wrapper
# ---------------------------------------------------------------------------

class MopTRTMNet(nn.Module):
    """
    MopTR-style training wrapper for TMNetEncoder.

    Trains the encoder to predict Q future image slices from T past slices,
    using ground-truth future frames as supervision (no MAE masking, no DVF cache).

    Architecture
    ------------
      Iseq  (B, 2, T, H, W)  — past slices concatenated with reference
        → encode_temporal      →  hs  (B, T, hidden_dim)
        → pool last valid frame →  h   (B, hidden_dim)
        → _MultiStepDecoder    →  preds  (B, Q, H, W)
      loss = MSE(preds, Igt.squeeze(1))   where Igt is (B, 1, Q, H, W)

    After pretraining only TMNetEncoder is kept; the decoder is discarded.

    Parameters
    ----------
    tm_net   : TMNetEncoder instance
    num_queries : number of future frames to predict (= cfg.tp)
    img_size    : (H, W) of a single output frame
                  sagittal → (64, 64), coronal → (32, 64) for standard vol_size
    """

    def __init__(
        self,
        tm_net,
        num_queries: int,
        img_size:    tuple,
    ):
        super().__init__()
        self.condi       = tm_net
        self.num_queries = num_queries
        self.decoder     = _MultiStepDecoder(tm_net.hidden_dim, num_queries, img_size)

    # ------------------------------------------------------------------
    # Forward  (training / validation)
    # ------------------------------------------------------------------

    def forward(
        self,
        Iseq: torch.Tensor,
        Igt:  Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Parameters
        ----------
        Iseq : (B, 2, T, H, W)   — past slices with reference concatenated
        Igt  : (B, 1, Q, H, W)   — ground-truth future slices (None at inference)

        Returns
        -------
        dict with keys:
            loss  : scalar MSE (None at inference)
            preds : (B, Q, H, W) predicted future frames
        """
        hs = self.condi.encode_temporal(Iseq)      # (B, T, hidden_dim)

        # Pool the last valid (non-zero-padded) frame's hidden state
        B = hs.shape[0]
        frame_energy = Iseq.abs().sum(dim=(1, 3, 4))      # (B, T)
        valid        = (frame_energy > 0).long()
        last_idx     = valid.cumsum(dim=1).argmax(dim=1)   # (B,)
        pooled       = hs[torch.arange(B, device=hs.device), last_idx]  # (B, hidden_dim)

        preds = self.decoder(pooled)               # (B, Q, H, W)

        loss = None
        if Igt is not None:
            loss = F.mse_loss(preds, Igt.squeeze(1))   # Igt: (B, 1, Q, H, W) → (B, Q, H, W)

        return {"loss": loss, "preds": preds}

    # ------------------------------------------------------------------
    # Inference — full TMNet forward (no decoder involved)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        sag: Optional[torch.Tensor] = None,
        cor: Optional[torch.Tensor] = None,
    ) -> list:
        """
        Run TMNetEncoder inference.

        Returns
        -------
        list of horizon tensors, each (B, output_dim)
        """
        frames = sag if sag is not None else cor
        if frames is None:
            raise ValueError("sag or cor must be provided")
        out_feats, _ = self.condi(frames)
        return out_feats
