"""
utils/dvf_metrics.py
====================
DVF diversity and quality metrics shared across training scripts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


LIVER_SPACINGS = (3.5, 3.4, 3.4)  # mm/voxel: (SI, AP, LR)


def geo_error(
    dvf_gt: np.ndarray,
    dvf_pred: np.ndarray,
    spacings: tuple = LIVER_SPACINGS,
) -> np.ndarray:
    """Per-voxel geometric error between two DVFs in mm.

    dvf_gt, dvf_pred: (3, D, H, W) numpy arrays — no batch dimension.
    Returns a (D, H, W) array of per-voxel errors in mm.
    """
    return np.sqrt(
        ((dvf_gt[0] - dvf_pred[0]) * spacings[0]) ** 2
        + ((dvf_gt[1] - dvf_pred[1]) * spacings[1]) ** 2
        + ((dvf_gt[2] - dvf_pred[2]) * spacings[2]) ** 2
    )


def motion_amplitude(dvf: torch.Tensor) -> float:
    """Mean voxel-wise displacement magnitude. dvf: (1, 3, D, H, W)"""
    return dvf.norm(dim=1).mean().item()


def hf_energy_ratio(dvf: torch.Tensor) -> float:
    """Fraction of FFT power in the upper half of frequencies. dvf: (1, 3, D, H, W)"""
    ratios = []
    for c in range(3):
        f = torch.fft.fftn(dvf[0, c], norm="ortho")
        power = f.abs().pow(2)
        total = power.sum().item()
        if total < 1e-12:
            continue
        D, H, W = dvf.shape[2:]
        dz = torch.fft.fftfreq(D, device=dvf.device)
        dy = torch.fft.fftfreq(H, device=dvf.device)
        dx = torch.fft.fftfreq(W, device=dvf.device)
        gz, gy, gx = torch.meshgrid(dz, dy, dx, indexing="ij")
        hf_mask = (gz.abs() > 0.25) | (gy.abs() > 0.25) | (gx.abs() > 0.25)
        ratios.append(power[hf_mask].sum().item() / total)
    return float(np.mean(ratios)) if ratios else 0.0


def hf_energy_ratio_vol(vol: torch.Tensor) -> float:
    """Fraction of FFT power in the upper half of frequencies for a scalar volume.

    Same threshold and normalisation as hf_energy_ratio but for a single-channel
    intensity volume instead of a 3-component DVF. A ratio much lower than the GT
    ratio indicates the model is losing high-frequency detail (blurring).

    vol: any shape broadcastable to (D, H, W) — e.g. (1,1,D,H,W) or (D,H,W).
    """
    v = vol.squeeze().float()
    f = torch.fft.fftn(v, norm="ortho")
    power = f.abs().pow(2)
    total = power.sum().item()
    if total < 1e-12:
        return 0.0
    D, H, W = v.shape[-3], v.shape[-2], v.shape[-1]
    dz = torch.fft.fftfreq(D, device=vol.device)
    dy = torch.fft.fftfreq(H, device=vol.device)
    dx = torch.fft.fftfreq(W, device=vol.device)
    gz, gy, gx = torch.meshgrid(dz, dy, dx, indexing="ij")
    hf_mask = (gz.abs() > 0.25) | (gy.abs() > 0.25) | (gx.abs() > 0.25)
    return float(power[hf_mask].sum().item() / total)


def _jacobian_det(dvf3: torch.Tensor) -> torch.Tensor:
    """Jacobian determinant field. dvf3: (3, D, H, W) — no batch dim."""
    def _grad(v, dim):
        pad = [0, 0, 0, 0, 0, 0]
        pad[-(2 * dim + 1)] = 1
        pad[-(2 * dim + 2)] = 1
        v_pad = F.pad(v.unsqueeze(0), pad, mode="replicate").squeeze(0)
        return (v_pad.narrow(dim, 2, v.shape[dim]) - v_pad.narrow(dim, 0, v.shape[dim])) / 2.0

    dz0, dy0, dx0 = _grad(dvf3[0], 0), _grad(dvf3[0], 1), _grad(dvf3[0], 2)
    dz1, dy1, dx1 = _grad(dvf3[1], 0), _grad(dvf3[1], 1), _grad(dvf3[1], 2)
    dz2, dy2, dx2 = _grad(dvf3[2], 0), _grad(dvf3[2], 1), _grad(dvf3[2], 2)
    J00, J01, J02 = 1 + dz0, dy0, dx0
    J10, J11, J12 = dz1, 1 + dy1, dx1
    J20, J21, J22 = dz2, dy2, 1 + dx2
    return (
        J00 * (J11 * J22 - J12 * J21)
        - J01 * (J10 * J22 - J12 * J20)
        + J02 * (J10 * J21 - J11 * J20)
    )


def jacobian_det_volume(dvf: torch.Tensor) -> torch.Tensor:
    """Per-voxel Jacobian determinant field. dvf: (1, 3, D, H, W) -> (D, H, W)"""
    return _jacobian_det(dvf[0])


def jacobian_det_stats(dvf: torch.Tensor):
    """Returns (mean, std) of the Jacobian determinant. dvf: (1, 3, D, H, W)"""
    det = _jacobian_det(dvf[0])
    return det.mean().item(), det.std().item()


def jacobian_folding_ratio(dvf: torch.Tensor) -> float:
    """Fraction of voxels where det(J) ≤ 0 (physically impossible folding). dvf: (1, 3, D, H, W)"""
    det = _jacobian_det(dvf[0])
    return (det <= 0).float().mean().item()


def dvf_cosine_sim_stats(vecs: list, patient_ids: list):
    """Return (intra, inter) cosine similarity lists for all sample pairs."""
    Z = F.normalize(torch.cat(vecs, dim=0), dim=1)
    sim = (Z @ Z.T).cpu().numpy()
    intra, inter = [], []
    for a in range(len(patient_ids)):
        for b in range(a + 1, len(patient_ids)):
            (intra if patient_ids[a] == patient_ids[b] else inter).append(sim[a, b])
    return intra, inter


def dvf_diversity_diagnostic(
    patient_ids: list,
    gt_vecs:     list,
    gen_vecs:    list,
    amp_gt:      list, amp_gen:   list,
    hf_gt:       list, hf_gen:    list,
    jstd_gt:     list, jstd_gen:  list,
    jmean_gt:    list, jmean_gen: list,
    save_path:   str,
) -> None:
    lines: list = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    def fmt(vals_gt, vals_gen, label):
        gt = np.mean(vals_gt)
        gen = np.mean(vals_gen)
        ratio = gen / gt if gt > 1e-12 else float("nan")
        emit(f"  {label:<30s}  GT={gt:.5f}   Gen={gen:.5f}  ({ratio*100:.1f}% retained)")

    emit("\n" + "=" * 65)
    emit("DVF DIVERSITY DIAGNOSTIC  [generated vs GT]")
    emit("=" * 65)
    emit(f"  Samples evaluated : {len(patient_ids)}")
    emit(f"  Unique patients   : {len(set(patient_ids))}")
    fmt(amp_gt,  amp_gen,  "Motion amplitude")
    fmt(hf_gt,   hf_gen,   "HF energy ratio")
    fmt(jstd_gt, jstd_gen, "Jacobian det std")

    jmgt  = np.mean(jmean_gt)
    jmgen = np.mean(jmean_gen)
    ratio_jm = jmgen / jmgt if jmgt > 1e-12 else float("nan")
    emit(f"  {'Jacobian det mean':<30s}  GT={jmgt:.5f}   Gen={jmgen:.5f}  (ratio {ratio_jm:.3f})")

    emit()
    emit("Interpretation:")
    for vals_gt, vals_gen, label, thresh in [
        (amp_gt,  amp_gen,  "amplitude",            0.85),
        (hf_gt,   hf_gen,   "high-freq energy",     0.70),
        (jstd_gt, jstd_gen, "Jacobian det variety", 0.80),
    ]:
        ratio = np.mean(vals_gen) / max(np.mean(vals_gt), 1e-12)
        status = "WARNING  LOW" if ratio < thresh else "OK"
        emit(f"  {status}  {label}: {ratio*100:.1f}% retained (threshold {thresh*100:.0f}%)")

    emit()
    emit("Cosine similarity — intra-patient vs inter-patient:")
    for label, vecs in [("GT DVF", gt_vecs), ("Generated DVF", gen_vecs)]:
        intra, inter = dvf_cosine_sim_stats(vecs, patient_ids)
        emit(f"  {label}:")
        if intra:
            emit(f"    Intra-patient : mean={np.mean(intra):.4f}  std={np.std(intra):.4f}  n={len(intra)}")
        else:
            emit("    Intra-patient: not enough samples from the same patient.")
        if inter:
            emit(f"    Inter-patient : mean={np.mean(inter):.4f}  std={np.std(inter):.4f}  n={len(inter)}")
        else:
            emit("    Inter-patient: only one patient in the evaluated samples.")
        if intra and inter:
            gap = np.mean(intra) - np.mean(inter)
            status = "OK" if gap > 0.05 else "WARNING  LOW SEPARATION"
            emit(f"    {status}  intra−inter gap: {gap:+.4f} (>0.05 expected)")
        emit()

    with open(save_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"DVF diversity diagnostic saved to {save_path}")
