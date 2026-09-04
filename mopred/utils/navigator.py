"""
Navigator-signal-based evaluation metrics for predicted 4D liver volumes.
``signal(t) = mean|cor_t − cor_exhale|`` tracks motion from the coronal slice;
``amplitude_ratio`` measures relative motion magnitude and ``phase_corr`` (Pearson R)
measures temporal phase tracking between predicted and ground-truth sequences.
"""

from __future__ import annotations

import numpy as np
import torch

Y_NAV = 32   # coronal slice index in downsampled space (H=64), matches data loader


def navigator_signal_3planes(volume: torch.Tensor, ref_volume: torch.Tensor) -> tuple[float, float, float]:
    """
    3 navigator signals from orthogonal planes (mean absolute intensity difference vs exhale):
      coronal  (D×W, fixed Y=H//2) — captures SI+LR motion
      sagittal (D×H, fixed X=W//2) — captures SI+AP motion
      axial    (H×W, fixed Z=D//2) — captures LR+AP motion
    """
    _, _, D, H, W = volume.shape
    v = volume[0, 0].float()
    r = ref_volume[0, 0].float()
    coronal  = (v[:, H // 2, :] - r[:, H // 2, :]).abs().mean().item()
    sagittal = (v[:, :, W // 2] - r[:, :, W // 2]).abs().mean().item()
    axial    = (v[D // 2, :, :] - r[D // 2, :, :]).abs().mean().item()
    return coronal, sagittal, axial


def navigator_signal(volume: torch.Tensor, ref_volume: torch.Tensor) -> float:
    """
    mean |cor_t − cor_exhale| on the coronal slice at Y_NAV.

    Parameters
    ----------
    volume     : (1, 1, D, H, W) — current frame (generated or actual)
    ref_volume : (1, 1, D, H, W) — exhale reference frame

    Returns
    -------
    float — scalar motion signal in voxel intensity units
    """
    y = min(Y_NAV, volume.shape[3] - 1)
    cor_t   = volume  [0, 0, :, y, :].float()
    cor_ref = ref_volume[0, 0, :, y, :].float()
    return (cor_t - cor_ref).abs().mean().item()


def navigator_metrics(
    signals_gt:   list[float],
    signals_pred: list[float],
) -> dict:
    """
    Aggregate per-frame navigator signals into scalar metrics.

    Parameters
    ----------
    signals_gt, signals_pred : lists of per-frame navigator signal values

    Returns
    -------
    dict with:
        amplitude_ratio : mean(pred) / mean(gt)   — motion amplitude retention
        phase_corr      : Pearson R(gt, pred)      — phase tracking correlation
    """
    gt   = np.array(signals_gt,   dtype=float)
    pred = np.array(signals_pred, dtype=float)

    valid = ~(np.isnan(gt) | np.isnan(pred))
    gt, pred = gt[valid], pred[valid]

    if len(gt) == 0:
        return {"amplitude_ratio": np.nan, "phase_corr": np.nan}

    mean_gt = gt.mean()
    amplitude_ratio = float(pred.mean() / mean_gt) if mean_gt > 1e-8 else np.nan

    if len(gt) >= 2 and gt.std() > 1e-4 and pred.std() > 1e-4:
        phase_corr = float(np.corrcoef(gt, pred)[0, 1])
    else:
        phase_corr = np.nan

    return {"amplitude_ratio": amplitude_ratio, "phase_corr": phase_corr}
