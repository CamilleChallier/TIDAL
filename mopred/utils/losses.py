"""
Loss functions for the TIDAL training pipeline.
Includes NCC, MSE, gradient/bending regularisation, ``FocalFrequencyLoss3D``,
band-split spectral losses, and a ``build_criterion`` factory.
Frequency-domain losses adapted from focal-frequency-loss
(https://github.com/EndlessSora/focal-frequency-loss).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


def ncc_loss(I, J, device, win=None, eps=1e-3):
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3]
    if win is None:
        win = [9] * ndims

    # Merge channels into batch so conv3d always sees 1-channel input
    B, C = I.shape[:2]
    I = I.reshape(B * C, 1, *I.shape[2:])
    J = J.reshape(B * C, 1, *J.shape[2:])

    sum_filt = torch.ones([1, 1, *win], device=device)
    pad_no   = math.floor(win[0] / 2)
    stride   = (1,) * ndims
    padding  = (pad_no,) * ndims

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    # Clamp variance to avoid division by near-zero
    denom = (I_var * J_var).clamp(min=eps)
    if torch.isnan(denom).any() or (denom < 0).any():
        print(f"[NCC warning] Negative or NaN variance — I_var min={I_var.min():.4f}, J_var min={J_var.min():.4f}")
    cc    = cross * cross / denom
    return -1 * torch.mean(cc)


def ncc_as_loss(*args, **kwargs):
    # print("Using NCC as a loss (1 + ncc_loss)")
    # print(f"ncc loss value: {ncc_loss(*args, **kwargs)}")
    # print(f"ncc_as_loss value: {1 + ncc_loss(*args, **kwargs)}")
    ncc = ncc_loss(*args, **kwargs)
    if torch.isnan(ncc) or torch.isinf(ncc):
        print(f"[Warning] NCC loss is NaN or Inf: {ncc}")
    return 1 + ncc

def bending_energy(dvf):
    d2x = dvf[:, :, 2:, :, :] - 2 * dvf[:, :, 1:-1, :, :] + dvf[:, :, :-2, :, :]
    d2y = dvf[:, :, :, 2:, :] - 2 * dvf[:, :, :, 1:-1, :] + dvf[:, :, :, :-2, :]
    d2z = dvf[:, :, :, :, 2:] - 2 * dvf[:, :, :, :, 1:-1] + dvf[:, :, :, :, :-2]
    return (d2x.pow(2).mean() + d2y.pow(2).mean() + d2z.pow(2).mean()) / 3


def gradient_loss(s, penalty="l2"):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if penalty == "l2":
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0

# def build_criterion(cfg):
#     """
#     Returns a callable loss function based on CLI args.
#     """
#     if cfg["training"]["loss"] == "mse":
#         return nn.MSELoss(reduction="mean")

#     if cfg["training"]["loss"] == "ncc":
#         return ncc_loss

#     raise ValueError(f"Unknown loss: {opt.loss}")


class FocalFrequencyLoss3D(nn.Module):
    """
    Focal Frequency Loss adapted to 3-D DVF volumes (B, C, D, H, W).

    Penalises frequency components that are poorly reconstructed, with an
    adaptive per-frequency weight proportional to the squared reconstruction
    error in the Fourier domain raised to the power `alpha`.

    Parameters
    ----------
    loss_weight  : overall scalar weight applied to the final loss value
    alpha        : exponent of the adaptive weight matrix (0 = uniform)
    patch_factor : optional spatial patching factor applied before FFT
                   (must divide D, H, and W; default 1 = no patching)
    log_matrix   : apply log(1 + weight) to smooth extreme weights
    batch_matrix : if True, use per-sample adaptive weights; otherwise
                   average the weight matrix across the batch
    """

    def __init__(
        self,
        loss_weight:  float = 1.0,
        alpha:        float = 1.0,
        patch_factor: int   = 1,
        log_matrix:   bool  = False,
        batch_matrix: bool  = True,
    ):
        super().__init__()
        self.loss_weight  = loss_weight
        self.alpha        = alpha
        self.patch_factor = patch_factor
        self.log_matrix   = log_matrix
        self.batch_matrix = batch_matrix

    def _to_freq(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) → stacked real/imag (..., 2) via rfftn."""
        p = self.patch_factor
        B, C, D, H, W = x.shape
        assert D % p == 0 and H % p == 0 and W % p == 0, \
            f"patch_factor={p} must divide D={D}, H={H}, W={W}"
        if p > 1:
            x = (x.reshape(B, C, p, D // p, p, H // p, p, W // p)
                  .permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
                  .reshape(B * p**3, C, D // p, H // p, W // p))
        freq = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        return torch.stack([freq.real, freq.imag], dim=-1)   # (..., 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f   = self._to_freq(pred)
        target_f = self._to_freq(target)

        if self.alpha != 0:
            diff = (pred_f.detach() - target_f.detach()).pow(2).sum(dim=-1)
            if not self.batch_matrix:
                diff = diff.mean(dim=0, keepdim=True).expand_as(diff)
            weight = diff.pow(self.alpha)
        else:
            weight = torch.ones_like(pred_f[..., 0])

        if self.log_matrix:
            weight = torch.log1p(weight)

        weight = weight / (weight.max() + 1e-8) + 1.0   # normalise to [1, 2]

        loss = (weight * (pred_f - target_f).pow(2).sum(dim=-1)).mean()
        return loss * self.loss_weight


def band_split_spectral_loss(
    pred,
    target,
    low_thresh=0.3,
    high_thresh=0.6,
    lambda_mid=0.1,
    lambda_high=1.0,
):
    """
    Decomposes the reconstruction error in Fourier space into low/mid/high
    frequency bands and returns an extra penalty for the mid/high bands so
    that under-reconstructed high-frequency content is explicitly penalised.

    pred, target: (B, C, D, H, W)

    Returns:
        loss: lambda_mid * L_mid + lambda_high * L_high
        metrics: dict with detached low/mid/high band losses
    """
    F_pred = torch.fft.rfftn(pred, dim=(-3, -2, -1), norm="ortho")
    F_gt   = torch.fft.rfftn(target, dim=(-3, -2, -1), norm="ortho")

    diff_sq = (F_pred - F_gt).abs().pow(2)

    D, H, W = pred.shape[-3:]

    fd = torch.fft.fftfreq(D, device=pred.device).abs()
    fh = torch.fft.fftfreq(H, device=pred.device).abs()
    fw = torch.fft.rfftfreq(W, device=pred.device)

    freq_mag = (
        fd[:, None, None] ** 2 +
        fh[None, :, None] ** 2 +
        fw[None, None, :] ** 2
    ).sqrt()

    low_mask  = (freq_mag < low_thresh)
    mid_mask  = (freq_mag >= low_thresh) & (freq_mag < high_thresh)
    high_mask = (freq_mag >= high_thresh)

    def masked_mean(x, mask):
        mask = mask.float()
        num   = (x * mask).sum(dim=(-3, -2, -1))
        denom = mask.sum() + 1e-8
        return (num / denom).mean()

    L_low  = masked_mean(diff_sq, low_mask)
    L_mid  = masked_mean(diff_sq, mid_mask)
    L_high = masked_mean(diff_sq, high_mask)

    loss = lambda_mid * L_mid + lambda_high * L_high

    metrics = {
        "band_low":  L_low.detach(),
        "band_mid":  L_mid.detach(),
        "band_high": L_high.detach(),
    }

    return loss, metrics


def band_energy_ratio_loss(
    pred,
    target,
    low_thresh=0.3,
    high_thresh=0.6,
    lambda_mid=0.1,
    lambda_high=1.0,
):
    """
    One-sided energy-deficit penalty for the mid/high frequency bands: per band,
    compares total spectral energy (Sum|F|^2, normalised by total power) between
    recon and target, and only penalises recon having *less* energy than target.

    Unlike `band_split_spectral_loss` (which matches each Fourier coefficient's
    magnitude and phase, so the L2-optimal solution for low-SNR/noise-like bands
    is to shrink them to ~0), this only pushes the model to restore the right
    *amount* of detail per band, without forcing it to match unpredictable
    phase/pattern bin-by-bin — avoiding the shrink-to-zero failure mode.

    pred, target: (B, C, D, H, W)

    Returns:
        loss: lambda_mid * deficit_mid + lambda_high * deficit_high
        metrics: dict with detached per-band energy ratios (recon / target, mean over batch & channels)
    """
    F_pred = torch.fft.rfftn(pred, dim=(-3, -2, -1), norm="ortho")
    F_gt   = torch.fft.rfftn(target, dim=(-3, -2, -1), norm="ortho")

    power_pred = F_pred.abs().pow(2)
    power_gt   = F_gt.abs().pow(2)

    D, H, W = pred.shape[-3:]

    fd = torch.fft.fftfreq(D, device=pred.device).abs()
    fh = torch.fft.fftfreq(H, device=pred.device).abs()
    fw = torch.fft.rfftfreq(W, device=pred.device)

    freq_mag = (
        fd[:, None, None] ** 2 +
        fh[None, :, None] ** 2 +
        fw[None, None, :] ** 2
    ).sqrt()

    mid_mask  = ((freq_mag >= low_thresh) & (freq_mag < high_thresh)).float()
    high_mask = (freq_mag >= high_thresh).float()

    def band_energy(power, mask):
        return (power * mask).sum(dim=(-3, -2, -1))   # (B, C)

    total_pred = power_pred.sum(dim=(-3, -2, -1))   # (B, C)
    total_gt   = power_gt.sum(dim=(-3, -2, -1))

    ratio_pred_mid  = band_energy(power_pred, mid_mask)  / (total_pred + 1e-8)
    ratio_gt_mid    = band_energy(power_gt,   mid_mask)  / (total_gt   + 1e-8)
    ratio_pred_high = band_energy(power_pred, high_mask) / (total_pred + 1e-8)
    ratio_gt_high   = band_energy(power_gt,   high_mask) / (total_gt   + 1e-8)

    deficit_mid  = F.relu(ratio_gt_mid  - ratio_pred_mid).mean()
    deficit_high = F.relu(ratio_gt_high - ratio_pred_high).mean()

    loss = lambda_mid * deficit_mid + lambda_high * deficit_high

    metrics = {
        "band_mid_energy_ratio":  ratio_pred_mid.mean().detach(),
        "band_high_energy_ratio": ratio_pred_high.mean().detach(),
        "band_mid_energy_ratio_gt":  ratio_gt_mid.mean().detach(),
        "band_high_energy_ratio_gt": ratio_gt_high.mean().detach(),
    }

    return loss, metrics


def laplacian_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference of Laplacian-filtered pred and target (3D).
    Penalises boundary/edge errors. Kernel: 6-connected discrete 3D Laplacian.
    """
    C = pred.shape[1]
    k = torch.zeros(1, 1, 3, 3, 3, device=pred.device, dtype=pred.dtype)
    k[0, 0, 1, 1, 1] = 6.0
    k[0, 0, 0, 1, 1] = -1.0
    k[0, 0, 2, 1, 1] = -1.0
    k[0, 0, 1, 0, 1] = -1.0
    k[0, 0, 1, 2, 1] = -1.0
    k[0, 0, 1, 1, 0] = -1.0
    k[0, 0, 1, 1, 2] = -1.0
    k = k.expand(C, 1, 3, 3, 3)
    lp = lambda x: torch.nn.functional.conv3d(x, k, padding=1, groups=C)
    return (lp(pred) - lp(target)).abs().mean()


def build_criterion(opt):
    """
    Returns a callable loss function based on CLI args.
    """
    if opt.loss == "mse":
        return nn.MSELoss(reduction="mean")

    if opt.loss == "ncc":
        return ncc_loss
    
    if opt.loss == "ncc_as_loss":
        return ncc_as_loss

    raise ValueError(f"Unknown loss: {opt.loss}")
