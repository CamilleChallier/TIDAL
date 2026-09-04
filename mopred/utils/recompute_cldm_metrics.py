"""
utils/recompute_cldm_metrics.py
================================
Post-hoc recomputation of navigator and LM-DVF metrics from saved test volumes.

Use this when a test folder has saved tracking volumes and DVFs but is missing
nav_signals_gt.npy / nav_signals_pred.npy / landmark_dvf_error.npy
(e.g. the model was tested before those outputs were added to train_CLDM.py).

Assumptions
-----------
* Tracking volumes exist at:
    {fold_dir}/tracking/{patient}_t0/{patient}_t0-{vol_idx:5d}.nii    (predicted)
    {fold_dir}/tracking/{patient}_t0_GT/{patient}_t0_GT-{vol_idx:5d}.nii  (GT)
* DVFs exist at:
    {fold_dir}/volumes/DVF_t0/DVF_t0-{batch_idx:5d}.nii          (VoxelMorph GT)
    {fold_dir}/volumes/generated_DVF_t0/generated_DVF_t0-{batch_idx:5d}.nii
* The data loader iterated patients in the order listed in patients_test.txt,
  and volumes within each patient in ascending vol_idx order.
* cfg.tp == 1  (only t0 saved — standard for these runs).

Usage
-----
    python -m 4D_MoPred_liver.utils.recompute_cldm_metrics \
        --fold_dir /path/to/test_dir/0 \
        --data_dir /store/usagers/livazra/Nav_dataset

Or call directly from Python:
    from 4D_MoPred_liver.utils.recompute_cldm_metrics import recompute
    recompute("/path/to/test_dir/0", "/store/usagers/livazra/Nav_dataset")
"""

from __future__ import annotations

import argparse
import glob
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .landmarks import load_landmarks, landmark_dvf_error as _lm_dvf_error
from .navigator import navigator_signal, navigator_metrics
from .one_resp_cycle import temp_crop
from .training import summarize_test_metrics
from ..data.data_loaders.navigator_4d import _REF_ID


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_exhale(patient: str, data_dir: str) -> torch.Tensor:
    """Load exhale reference volume → (1,1,D,H,W), z-score normalised."""
    _, exh = _REF_ID[patient]
    vol_files = sorted(
        os.listdir(os.path.join(data_dir, patient)),
        key=lambda x: int(x[2:-7]),
    )
    ref_file = next(f for f in vol_files if f.endswith(f"t_{exh}.nii.gz"))
    vol = nib.load(os.path.join(data_dir, patient, ref_file)).get_fdata()
    t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, scale_factor=[1, 0.5, 0.5], mode="trilinear",
                      align_corners=False)
    t = (t - t.mean()) / t.std()
    return t


def _load_volume(path: str) -> torch.Tensor:
    """Load a (D,H,W) NIfTI volume → (1,1,D,H,W)."""
    arr = nib.load(path).get_fdata().astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def _load_dvf(path: str) -> torch.Tensor:
    """Load a (D,H,W,3) NIfTI DVF → (1,3,D,H,W)."""
    arr = nib.load(path).get_fdata().astype(np.float32)   # (D,H,W,3)
    t = torch.from_numpy(arr).permute(3, 0, 1, 2)         # (3,D,H,W)
    return t.unsqueeze(0)                                  # (1,3,D,H,W)


def _iter_tracking_files(track_dir: str, patient: str, tp: int = 0):
    """
    Yield (vol_idx, pred_path, gt_path) for one patient/timepoint,
    sorted by ascending vol_idx.
    """
    pattern = os.path.join(track_dir, f"{patient}_t{tp}", "*.nii")
    files = sorted(glob.glob(pattern),
                   key=lambda p: int(os.path.basename(p).split("-")[1].split(".")[0]))
    for pred_path in files:
        vol_idx = int(os.path.basename(pred_path).split("-")[1].split(".")[0])
        gt_path = pred_path.replace(
            f"/{patient}_t{tp}/",
            f"/{patient}_t{tp}_GT/",
        ).replace(
            f"{patient}_t{tp}-",
            f"{patient}_t{tp}_GT-",
        )
        if os.path.isfile(gt_path):
            yield vol_idx, pred_path, gt_path


def _dvf_path(vol_dir: str, kind: str, batch_idx: int, tp: int = 0) -> str:
    name = f"{kind}_t{tp}"
    return os.path.join(vol_dir, name, f"{name}-%5s.nii" % batch_idx)


# ── main recompute function ───────────────────────────────────────────────────

def recompute(fold_dir: str, data_dir: str, tp: int = 0) -> None:
    """
    Recompute and save nav_signals_gt/pred.npy and landmark_dvf_error.npy
    for one fold directory, then refresh summary_metrics.txt.

    Parameters
    ----------
    fold_dir : str
        Path to the fold directory (e.g. .../test_1/0).
    data_dir : str
        Root of the patient data (contains CoMoDo* subdirectories).
    tp : int
        Timepoint index to use (default 0 — only t0 is typically saved).
    """
    track_dir = os.path.join(fold_dir, "tracking")
    vol_dir   = os.path.join(fold_dir, "volumes")

    patients_file = os.path.join(fold_dir, "patients_test.txt")
    with open(patients_file) as f:
        patients = [l.strip() for l in f if l.strip()]

    landmarks_per_patient = {
        p: load_landmarks(data_dir, p) for p in patients
    }

    nav_signals_gt   = []
    nav_signals_pred = []
    lm_dvf_errors    = []

    # batch_idx increases globally across patients (same order as data loader)
    batch_idx = 0

    for patient in patients:
        ref_vol = _load_exhale(patient, data_dir)
        lm      = landmarks_per_patient[patient]

        nav_gt_seq   = []
        nav_pred_seq = []
        lm_dvf_seq   = []

        for vol_idx, pred_path, gt_path in _iter_tracking_files(
            track_dir, patient, tp
        ):
            # ── navigator ────────────────────────────────────────────────────
            pred_vol = _load_volume(pred_path)
            gt_vol   = _load_volume(gt_path)
            nav_gt_seq.append(navigator_signal(gt_vol,   ref_vol))
            nav_pred_seq.append(navigator_signal(pred_vol, ref_vol))

            # ── LM-DVF error ─────────────────────────────────────────────────
            dvf_gt_path   = _dvf_path(vol_dir, "DVF",           batch_idx, tp)
            dvf_pred_path = _dvf_path(vol_dir, "generated_DVF", batch_idx, tp)

            if os.path.isfile(dvf_gt_path) and os.path.isfile(dvf_pred_path):
                dvf_gt   = _load_dvf(dvf_gt_path)
                dvf_pred = _load_dvf(dvf_pred_path)
                err = _lm_dvf_error(dvf_pred, dvf_gt, lm, vol_idx, patient)
            else:
                err = np.nan

            lm_dvf_seq.append(err)
            batch_idx += 1

        nav_signals_gt.append(nav_gt_seq)
        nav_signals_pred.append(nav_pred_seq)
        lm_dvf_errors.append(lm_dvf_seq)

    # ── save npy ─────────────────────────────────────────────────────────────
    np.save(os.path.join(fold_dir, "nav_signals_gt.npy"),
            np.array(nav_signals_gt, dtype=object))
    np.save(os.path.join(fold_dir, "nav_signals_pred.npy"),
            np.array(nav_signals_pred, dtype=object))
    np.save(os.path.join(fold_dir, "landmark_dvf_error.npy"),
            np.array(lm_dvf_errors, dtype=object))

    # ── refresh summary ───────────────────────────────────────────────────────
    summarize_test_metrics(fold_dir)
    print(f"Done — updated {fold_dir}/summary_metrics.txt")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recompute navigator + LM-DVF metrics from saved test volumes."
    )
    parser.add_argument(
        "--fold_dir", required=True,
        help="Path to fold directory (e.g. .../test_1/0)",
    )
    parser.add_argument(
        "--data_dir", default="/store/usagers/livazra/Nav_dataset/",
        help="Root patient data directory.",
    )
    parser.add_argument(
        "--tp", type=int, default=0,
        help="Timepoint index (default 0).",
    )
    args = parser.parse_args()
    recompute(args.fold_dir, args.data_dir, args.tp)
