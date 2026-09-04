"""
utils/landmarks.py
==================
GT landmark loading and landmark tracking error computation.

Landmarks come from GT_landmarks/{patient_id}/*/main.txt (x y z t_idx).
Coordinates are in the original (pre-downsampled) voxel space; the dataset
applies a 0.5× factor to H and W, so effective spacings are:
    D  : 3.5   mm/voxel  (unchanged)
    H,W: 3.40  mm/voxel  (1.70454 * 2)
"""

from __future__ import annotations

import os
import numpy as np
import torch

from .one_resp_cycle import temp_crop

_SP_D  = 3.5       # mm/voxel — depth (D), unchanged after downsampling
_SP_HW = 1.70 * 2  # mm/voxel — height and width (0.5× DS)


def load_landmarks(data_dir: str, patient_id: str) -> dict:
    """
    Load GT landmark positions from
    ``{data_dir}/Complemetary_information/GT_landmarks/{patient_id}/*/main.txt``.

    When a landmark directory has no main.txt but has right.txt (L2-style
    annotation with flipped axes from NAVIGATOR_4D_Dataset_track), right.txt
    is loaded and its columns (lm_x, lm_y, lm_z) are converted to the
    synthetic (x_syn, y_syn, z_syn) convention expected by
    landmark_tracking_error so the DVF is sampled at the correct voxel:
        DVF-space: D = 32-lm_z, H = lm_x/2, W = 64-lm_y/2
        Stored as: x_syn=128-lm_y, y_syn=lm_x, z_syn=32-lm_z
        Recovered: W=x_syn/2=64-lm_y/2, H=y_syn/2=lm_x/2, D=z_syn=32-lm_z

    Returns
    -------
    dict  {lm_name: {t_idx: np.array([x, y, z])}}
    x, y, z in the convention described above.
    Empty dict when no data is found.
    """
    landmark_dir = os.path.join(
        data_dir, "Complemetary_information", "GT_landmarks", patient_id
    )
    if not os.path.isdir(landmark_dir):
        return {}
    result = {}
    for lm_name in sorted(os.listdir(landmark_dir)):
        main_txt  = os.path.join(landmark_dir, lm_name, "main.txt")
        right_txt = os.path.join(landmark_dir, lm_name, "right.txt")

        if os.path.isfile(main_txt):
            positions = {}
            with open(main_txt) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z, t = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        positions[t] = np.array([x, y, z], dtype=float)
            if positions:
                result[lm_name] = positions

        elif os.path.isfile(right_txt):
            # right.txt uses flipped axes; convert so landmark_tracking_error
            # samples the DVF at the correct D/H/W location.
            positions = {}
            with open(right_txt) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        lm_x, lm_y, lm_z, t = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        x_syn = 128 - lm_y   # -> W = x_syn/2 = 64 - lm_y/2
                        y_syn = lm_x         # -> H = y_syn/2 = lm_x/2
                        z_syn = 32 - lm_z    # -> D = z_syn   = 32 - lm_z
                        positions[t] = np.array([x_syn, y_syn, z_syn], dtype=float)
            if positions:
                result[lm_name + "_RT"] = positions

    return result


def landmark_dvf_error(
    dvf_pred:   torch.Tensor,
    dvf_gt:     torch.Tensor,
    landmarks:  dict,
    volume_idx: int,
    patient_id: str,
) -> float:
    """
    DVF discrepancy at landmark positions between pred and GT (VoxelMorph) DVF.

    Answers: "how well does the model replicate VoxelMorph's displacement at
    the annotated landmark?" — independent of VoxelMorph's own registration error.

    Parameters
    ----------
    dvf_pred, dvf_gt : (1, 3, D, H, W) — DVFs in voxels (downsampled space)
    """
    if not landmarks:
        return np.nan
    crop = temp_crop.get(patient_id)
    if crop is None:
        return np.nan
    t_idx = volume_idx - crop[0] + 1

    pred_np = dvf_pred[0].cpu().numpy()  # (3, D, H, W)
    gt_np   = dvf_gt[0].cpu().numpy()
    _, D, H, W = pred_np.shape

    errors = []
    for positions in landmarks.values():
        if 1 not in positions or t_idx not in positions:
            continue
        x_ref, y_ref, z_ref = positions[1]
        z_ds, y_ds, x_ds = z_ref, y_ref / 2.0, x_ref / 2.0

        d_pred = _sample_dvf_trilinear(pred_np, z_ds, y_ds, x_ds)   # ← changed
        d_gt   = _sample_dvf_trilinear(gt_np,   z_ds, y_ds, x_ds)   # ← changed

        diff_D, diff_H, diff_W = d_pred - d_gt

        err_mm = np.sqrt(
            (diff_D * _SP_D)  ** 2
            + (diff_H * _SP_HW) ** 2
            + (diff_W * _SP_HW) ** 2
        )
        errors.append(err_mm)

    return float(np.mean(errors)) if errors else np.nan


def landmark_tracking_error(
    dvf_pred:   torch.Tensor,
    landmarks:  dict,
    volume_idx: int,
    patient_id: str,
) -> float:
    """
    Mean 3-D landmark tracking error in mm for one predicted DVF frame.

    Parameters
    ----------
    dvf_pred   : (1, 3, D, H, W) — DVF in voxels (downsampled space)
    landmarks  : {lm_name: {t_idx: [x, y, z]}} — original (pre-DS) coords
    volume_idx : frame index extracted from the NIfTI filename
    patient_id : key into temp_crop for t_idx mapping

    Returns
    -------
    Mean error in mm across all available landmarks, or np.nan.
    """
    if not landmarks:
        return np.nan

    crop = temp_crop.get(patient_id)
    if crop is None:
        return np.nan
    t_idx = volume_idx - crop[0] + 1

    dvf_np = dvf_pred[0].cpu().numpy()   # (3, D, H, W)
    _, D, H, W = dvf_np.shape

    errors = []
    for positions in landmarks.values():
        if 1 not in positions or t_idx not in positions:
            continue

        # Reference position (t_idx=1 = exhale reference, where DVF ≈ 0)
        x_ref, y_ref, z_ref = positions[1]
        # Convert to downsampled space (0.5× in H and W, unchanged in D)
        x_ds = x_ref / 2.0
        y_ds = y_ref / 2.0
        z_ds = z_ref

        xi = int(np.clip(round(x_ds), 0, W - 1))
        yi = int(np.clip(round(y_ds), 0, H - 1))
        zi = int(np.clip(round(z_ds), 0, D - 1))

        # DVF displacement at reference landmark (voxels along D, H, W)
        d_D, d_H, d_W = _sample_dvf_trilinear(dvf_np, z_ds, y_ds, x_ds)

        # Predicted position in downsampled space
        pred_x = x_ds + d_W
        pred_y = y_ds + d_H
        pred_z = z_ds + d_D

        # GT position in downsampled space
        x_gt, y_gt, z_gt = positions[t_idx]
        gt_x = x_gt / 2.0
        gt_y = y_gt / 2.0
        gt_z = z_gt

        err_mm = np.sqrt(
            ((pred_x - gt_x) * _SP_HW) ** 2
            + ((pred_y - gt_y) * _SP_HW) ** 2
            + ((pred_z - gt_z) * _SP_D)  ** 2
        )
        errors.append(err_mm)

    return float(np.mean(errors)) if errors else np.nan

from scipy.ndimage import map_coordinates

def _sample_dvf_trilinear(dvf_np: np.ndarray, z: float, y: float, x: float) -> np.ndarray:
    """
    Trilinearly interpolate a (3, D, H, W) DVF at a continuous (z, y, x) coordinate.
    Returns [d_D, d_H, d_W].
    """
    _, D, H, W = dvf_np.shape
    z = np.clip(z, 0, D - 1)
    y = np.clip(y, 0, H - 1)
    x = np.clip(x, 0, W - 1)
    coords = np.array([[z], [y], [x]])  # shape (3, 1) for map_coordinates
    out = np.array([
        map_coordinates(dvf_np[c], coords, order=1, mode="nearest")[0]
        for c in range(3)
    ])
    return out  # [d_D, d_H, d_W]