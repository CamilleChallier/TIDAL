"""
Dataset utility functions for the TIDAL pipeline.
Covers loading NIfTI volumes, building multi-timepoint image sequences (``build_Iseq``),
extracting patient IDs from directory structures, and saving run parameters to text.
"""
from __future__ import annotations

# =========================
# Standard library
# =========================
import argparse
import datetime
import os
import shutil
import warnings

warnings.filterwarnings("ignore")

# =========================
# Third-party
# =========================
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ss

from mopred.data.data_loaders.acdc_4d import get_reference_slice_index

def make_run_dirs(logging_dir, dir_name, fold):
    log_dir = os.path.join(logging_dir, "logs", dir_name, fold)
    run_dir = os.path.join(logging_dir, "runs", dir_name, fold)

    directories = [log_dir, run_dir]
    for d in directories:
        cond_mkdir(d)
    return log_dir, run_dir


def save_params_txt(opt, log_dir):
    with open(os.path.join(log_dir, "params.txt"), "w") as f:
        for k, v in vars(opt).items():
            f.write(f"{k}: {v}\n")
    config_path = getattr(opt, "_config_path", None)
    if config_path and os.path.isfile(config_path):
        dst = os.path.join(log_dir, os.path.basename(config_path))
        if not os.path.samefile(config_path, dst) if os.path.exists(dst) else True:
            shutil.copy(config_path, dst)


def build_Iseq(
    ref_volume, input_volume_list, current_volume_list, opt, device,
    include_future=True, vol_files=None, orientation=None,
):
    """
    Returns conditioning tensor Iseq with shape ~ (B, 2, T, H, W) depending on condi_type.
    Assumes volumes come as (B, 1, D, H, W) after unsqueeze(1).

    condi_type "1" (sagittal): per-sample D-index via get_reference_slice_index,
    with optional spatial crop to condi_img_size. vol_files is required for this path.
    orientation: {patient: entry} dict from load_orientation_info, or None to fall back
    to CANONICAL_SHAPE[0] // 2 for every patient.
    """
    condi_type = getattr(opt, "condi_type", "1")
    if condi_type == "1":
        if vol_files is None:
            raise ValueError(
                "build_Iseq: vol_files is required for condi_type '1' "
                "(needed to resolve each sample's per-patient reference slice D-index)."
            )
        img_h, img_w = getattr(opt, "condi_img_size", [128, 128])
        H, W = ref_volume.shape[3], ref_volume.shape[4]
        img_h, img_w = min(img_h, H), min(img_w, W)
        h0 = max((H - img_h) // 2, 0)
        w0 = max((W - img_w) // 2, 0)

        B = ref_volume.shape[0]
        patients = [p.split("/")[-2] for p in vol_files[0]]
        d_idx = torch.tensor(
            [get_reference_slice_index(p, opt.data_dir, orientation) for p in patients],
            device=device, dtype=torch.long,
        )
        b_idx = torch.arange(B, device=device)

        c1_ref = ref_volume[b_idx, :, d_idx, h0:h0+img_h, w0:w0+img_w].to(device)

        c1 = []
        for q in range(opt.nb_inputs):
            v = input_volume_list[q].unsqueeze(1).to(device)
            c_t = v[b_idx, :, d_idx, h0:h0+img_h, w0:w0+img_w]
            c1.append(torch.cat([c_t, c1_ref], dim=1))
        Iseq = torch.stack(c1, dim=2).to(device)

        if include_future and current_volume_list is not None:
            future = []
            for vol in current_volume_list:
                v = vol.unsqueeze(1).to(device)
                c_f = v[b_idx, :, d_idx, h0:h0+img_h, w0:w0+img_w]
                future.append(torch.cat([c_f, c1_ref], dim=1))
            Iseq = torch.cat([Iseq, torch.stack(future, dim=2).to(device)], dim=2)
    else:
        cor_pos = getattr(opt, "cor_pos", 32)
        c2_ref = ref_volume[:, :, :, cor_pos, :]
        c2 = []
        for q in range(opt.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1)[:, :, :, cor_pos, :]
            c2.append(torch.cat([c_t.to(device), c2_ref.to(device)], dim=1))
        Iseq = torch.stack(c2, dim=2).to(device)
        if include_future and current_volume_list is not None:
            future = []
            for vol in current_volume_list:
                cur = vol.unsqueeze(1).to(device)
                future.append(
                    torch.cat([cur[:, :, :, cor_pos, :], c2_ref.to(device)], dim=1)
                )
            Iseq = torch.cat([Iseq, torch.stack(future, dim=2).to(device)], dim=2)
    return Iseq

def extract_patient_ids(sequence_list):
    patients = set()
    for seq in sequence_list:
        # assuming sequence name encodes patient (adjust if needed)
        patient = seq.split("/")[0] if "/" in seq else seq[:8]
        patients.add(patient)
    return sorted(list(patients))