"""
Core training utilities for the TIDAL pipeline.
Provides ``build_reg_model`` (loads a frozen VoxelMorph or MambaMorph backbone)
and helper functions called by all train scripts (step loops, metric logging).
"""
import numpy as np
import os
import torch
import argparse
from ..utils.io import custom_load

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_VM_CHECKPOINT = os.path.join(_PROJECT_ROOT, "4D_MoPred_liver", "pretrained_models", "VM.pth")

def build_reg_model(cfg, vol_size: tuple, device: torch.device) -> torch.nn.Module:
    """Build and load the registration backbone (VoxelMorph or MambaMorph).

    Config keys read:
      reg_model              : "voxelmorph" (default) | "mambamorph"
      vm_checkpoint          : path to the pretrained checkpoint
      mambamorph_patch_size  : patch size for MambaMorph (default 4)
      mambamorph_freq_aware  : use freq-aware blocks (default False)
    """
    reg_model = getattr(cfg, "reg_model", "voxelmorph")
    print(reg_model)
    ckpt      = getattr(cfg, "vm_checkpoint", None) or _DEFAULT_VM_CHECKPOINT
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(_PROJECT_ROOT, ckpt)

    if reg_model == "mambamorph":
        from ..models.mambamorph import MambaMorph
        mm_kwargs = dict(
            embed_dim=int(getattr(cfg, "mambamorph_embed_dim", 96)),
            depths=(2, 2, 4), reg_head_chan=16,
            d_state=16, d_conv=4, expand=2,
            patch_size=int(getattr(cfg, "mambamorph_patch_size", 2)),
            use_freq_aware=getattr(cfg, "mambamorph_freq_aware", False),
        )

        model = MambaMorph(vol_size, **mm_kwargs).to(device)
    else:
        from ..models import Voxelmorph
        model = Voxelmorph(
            vol_size, [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16], full_size=True,
        ).to(device)

    custom_load(model, ckpt, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[reg_model] {reg_model} loaded from {ckpt}")
    return model


def summarize_test_metrics(log_dir: str):
    """
    Compute and save useful test metrics from .npy files for a specific fold.

    Parameters
    ----------
    log_dir : str
        Directory containing MSE_loss.npy, NCC_loss.npy, SSIM_loss.npy,
        geo_error.npy, and optionally landmark_error.npy.
    """
    fold_dir         = log_dir
    mse_file         = os.path.join(fold_dir, "MSE_loss.npy")
    mse_gt_file      = os.path.join(fold_dir, "MSE_loss_gt.npy")
    ncc_file         = os.path.join(fold_dir, "NCC_loss.npy")
    ssim_file        = os.path.join(fold_dir, "SSIM_loss.npy")
    geo_file         = os.path.join(fold_dir, "geo_error.npy")
    psnr_file        = os.path.join(fold_dir, "PSNR_loss.npy")
    folding_file     = os.path.join(fold_dir, "folding_ratio.npy")
    smooth_file      = os.path.join(fold_dir, "temporal_smoothness.npy")
    lm_file          = os.path.join(fold_dir, "landmark_error.npy")
    lm_gt_file       = os.path.join(fold_dir, "landmark_error_gt.npy")
    lm_dvf_file      = os.path.join(fold_dir, "landmark_dvf_error.npy")
    lm_main_file     = os.path.join(fold_dir, "landmark_error_main.npy")
    lm_dvf_main_file = os.path.join(fold_dir, "landmark_dvf_error_main.npy")
    lm_rt_file       = os.path.join(fold_dir, "landmark_error_rt.npy")
    lm_dvf_rt_file   = os.path.join(fold_dir, "landmark_dvf_error_rt.npy")
    nav_gt_file      = os.path.join(fold_dir, "nav_signals_gt.npy")
    nav_pred_file    = os.path.join(fold_dir, "nav_signals_pred.npy")
    nav_pred_std_file = os.path.join(fold_dir, "nav_pred_std.npy")
    patients_file    = os.path.join(fold_dir, "patients_test.txt")
    summary_file     = os.path.join(fold_dir, "summary_metrics.txt")

    mse  = np.load(mse_file)
    ncc  = np.load(ncc_file)
    ssim = np.load(ssim_file)
    geo  = np.load(geo_file) if os.path.exists(geo_file) else np.full_like(mse, np.nan, dtype=float)
    def _load_numeric_npy(path):
        if not os.path.exists(path):
            return None
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            arr = np.array(
                [[float(np.asarray(v).flat[0]) for v in row] for row in arr],
                dtype=float,
            )
        return arr

    psnr    = _load_numeric_npy(psnr_file)
    folding = _load_numeric_npy(folding_file)
    smooth  = _load_numeric_npy(smooth_file)

    # Landmark error: object array (ragged NaN-padded), flatten to 1-D
    def _load_lm_npy(path):
        if not os.path.exists(path):
            return None, np.nan, np.nan
        raw   = np.load(path, allow_pickle=True)
        flat  = np.array([
            float(v) for row in raw for v in (row if hasattr(row, "__iter__") else [row])
        ])
        valid = flat[~np.isnan(flat)]
        mean  = float(np.mean(valid)) if len(valid) else np.nan
        std   = float(np.std(valid) / np.sqrt(max(len(valid), 1)))
        return raw, mean, std

    has_lm = os.path.exists(lm_file)
    lm_raw,    lm_mean,    lm_std    = _load_lm_npy(lm_file)
    lm_gt_raw, lm_gt_mean, lm_gt_std = _load_lm_npy(lm_gt_file)
    lm_dvf_raw, lm_dvf_mean, lm_dvf_std = _load_lm_npy(lm_dvf_file)
    lm_main_raw,     lm_main_mean,     lm_main_std     = _load_lm_npy(lm_main_file)
    lm_dvf_main_raw, lm_dvf_main_mean, lm_dvf_main_std = _load_lm_npy(lm_dvf_main_file)
    lm_rt_raw,       lm_rt_mean,       lm_rt_std       = _load_lm_npy(lm_rt_file)
    lm_dvf_rt_raw,   lm_dvf_rt_mean,   lm_dvf_rt_std   = _load_lm_npy(lm_dvf_rt_file)

    # Navigator metrics (VM-free phase tracking)
    from ..utils.navigator import navigator_metrics as _nav_metrics
    _nav_gt_raw = _nav_pred_raw = None
    nav_ratio, nav_corr = np.nan, np.nan
    if os.path.exists(nav_gt_file) and os.path.exists(nav_pred_file):
        _nav_gt_raw  = np.load(nav_gt_file,   allow_pickle=True)
        _nav_pred_raw = np.load(nav_pred_file, allow_pickle=True)
        _nav_gt   = [float(v) for row in _nav_gt_raw   for v in (row if hasattr(row, "__iter__") else [row])]
        _nav_pred = [float(v) for row in _nav_pred_raw for v in (row if hasattr(row, "__iter__") else [row])]
        _nm = _nav_metrics(_nav_gt, _nav_pred)
        nav_ratio = _nm["amplitude_ratio"]
        nav_corr  = _nm["phase_corr"]
    nav_pred_std_val = float(np.load(nav_pred_std_file)) if os.path.exists(nav_pred_std_file) else np.nan
    nav_pred_arr_raw = np.load(nav_pred_file, allow_pickle=True) if os.path.exists(nav_pred_file) else None
    if nav_pred_arr_raw is not None:
        _flat = [float(v) for row in nav_pred_arr_raw for v in (row if hasattr(row, "__iter__") else [row])]
        _mean = float(np.mean(_flat)) if _flat else np.nan
        nav_pred_cv_val = (nav_pred_std_val / _mean) if (not np.isnan(nav_pred_std_val) and not np.isnan(_mean) and _mean > 1e-8) else np.nan
    else:
        nav_pred_cv_val = np.nan

    if os.path.exists(patients_file):
        with open(patients_file) as f:
            patients = [line.strip() for line in f]
    else:
        patients = [f"Sample_{i}" for i in range(mse.shape[0])]

    # If patient_ids.npy exists (one entry per DataLoader row), use it to
    # group rows by patient — needed when each row is one timepoint, not one patient.
    patient_ids_npy = os.path.join(fold_dir, "patient_ids.npy")
    row_pids = None
    if os.path.exists(patient_ids_npy):
        _raw_pids = np.load(patient_ids_npy, allow_pickle=True)
        _flat_pids = [str(x) for x in _raw_pids.ravel()]
        if len(_flat_pids) == mse.shape[0] and len(_flat_pids) > len(patients):
            row_pids = _flat_pids
            _seen: dict = {}
            for pid in row_pids:
                if pid not in _seen:
                    _seen[pid] = len(_seen)
            patients = sorted(_seen.keys(), key=lambda p: _seen[p])

    mse_mean,  mse_std  = np.nanmean(mse),  np.nanstd(mse)  / np.sqrt(max(np.sum(~np.isnan(mse)), 1))
    mse_gt     = _load_numeric_npy(mse_gt_file)
    mse_gt_mean, mse_gt_std = (np.nanmean(mse_gt), np.nanstd(mse_gt) / np.sqrt(max(np.sum(~np.isnan(mse_gt)), 1))) if mse_gt is not None else (np.nan, np.nan)
    ncc_mean,  ncc_std  = np.nanmean(ncc),  np.nanstd(ncc)  / np.sqrt(max(np.sum(~np.isnan(ncc)), 1))
    ssim_mean, ssim_std = np.nanmean(ssim), np.nanstd(ssim) / np.sqrt(max(np.sum(~np.isnan(ssim)), 1))
    geo_mean,  geo_std  = np.nanmean(geo),  np.nanstd(geo)  / np.sqrt(max(np.sum(~np.isnan(geo)), 1))
    psnr_mean,    psnr_std    = (np.nanmean(psnr),    np.nanstd(psnr)    / np.sqrt(max(np.sum(~np.isnan(psnr)),    1))) if psnr    is not None else (np.nan, np.nan)
    folding_mean, folding_std = (np.nanmean(folding), np.nanstd(folding) / np.sqrt(max(np.sum(~np.isnan(folding)), 1))) if folding is not None else (np.nan, np.nan)
    smooth_mean,  smooth_std  = (np.nanmean(smooth),  np.nanstd(smooth)  / np.sqrt(max(np.sum(~np.isnan(smooth)),  1))) if smooth  is not None else (np.nan, np.nan)

    # Per-timestep: mean over samples (axis=0), shape (n_tp,)
    mse_by_t  = np.nanmean(mse,  axis=0).ravel() if mse.ndim  > 1 else np.array([np.nanmean(mse)])
    ncc_by_t  = np.nanmean(ncc,  axis=0).ravel() if ncc.ndim  > 1 else np.array([np.nanmean(ncc)])
    ssim_by_t = np.nanmean(ssim, axis=0).ravel() if ssim.ndim > 1 else np.array([np.nanmean(ssim)])
    geo_by_t  = np.nanmean(geo,  axis=0).ravel() if geo.ndim  > 1 else np.array([np.nanmean(geo)])
    psnr_by_t    = np.nanmean(psnr,    axis=0).ravel() if psnr    is not None and psnr.ndim    > 1 else None
    folding_by_t = np.nanmean(folding, axis=0).ravel() if folding is not None and folding.ndim > 1 else None

    # Per-row means (over timepoints and trailing singleton dims)
    _mse_row  = np.nanmean(mse,  axis=tuple(range(1, mse.ndim)))
    _ncc_row  = np.nanmean(ncc,  axis=tuple(range(1, ncc.ndim)))
    _ssim_row = np.nanmean(ssim, axis=tuple(range(1, ssim.ndim)))
    _geo_row  = np.nanmean(geo,  axis=tuple(range(1, geo.ndim)))

    def _row_lm_means(raw, n_rows):
        if raw is None:
            return [np.nan] * n_rows
        result = []
        for row in raw:
            vals = [float(v) for v in (row if hasattr(row, "__iter__") else [row])]
            result.append(np.nanmean(vals) if vals else np.nan)
        return result

    _n = len(_mse_row)
    _lm_row     = _row_lm_means(lm_raw,     _n) if has_lm             else [np.nan] * _n
    _lm_dvf_row = _row_lm_means(lm_dvf_raw, _n) if lm_dvf_raw is not None else [np.nan] * _n

    if row_pids is not None:
        # Group rows by patient and compute per-patient means
        def _group(flat):
            grouped = {p: [] for p in patients}
            for i, pid in enumerate(row_pids):
                if pid in grouped:
                    v = float(flat[i])
                    if not np.isnan(v):
                        grouped[pid].append(v)
            return [np.nanmean(grouped[p]) if grouped[p] else np.nan for p in patients]

        mse_per_sample    = _group(_mse_row)
        ncc_per_sample    = _group(_ncc_row)
        ssim_per_sample   = _group(_ssim_row)
        geo_per_sample    = _group(_geo_row)
        lm_per_sample     = _group(_lm_row)
        lm_dvf_per_sample = _group(_lm_dvf_row)

        # Per-patient phase correlation from nav signals
        nav_corr_per_sample = []
        if _nav_gt_raw is not None and _nav_pred_raw is not None:
            for p in patients:
                gt_vals, pred_vals = [], []
                for i, pid in enumerate(row_pids):
                    if pid == p:
                        rgt   = _nav_gt_raw[i]   if i < len(_nav_gt_raw)   else []
                        rpred = _nav_pred_raw[i]  if i < len(_nav_pred_raw) else []
                        gt_vals.extend(  [float(v) for v in (rgt   if hasattr(rgt,   "__iter__") else [rgt])   if not np.isnan(float(v))])
                        pred_vals.extend([float(v) for v in (rpred  if hasattr(rpred, "__iter__") else [rpred]) if not np.isnan(float(v))])
                if len(gt_vals) > 1 and len(gt_vals) == len(pred_vals):
                    nav_corr_per_sample.append(_nav_metrics(gt_vals, pred_vals)["phase_corr"])
                else:
                    nav_corr_per_sample.append(np.nan)
        else:
            nav_corr_per_sample = [np.nan] * len(patients)
    else:
        # Legacy: one row = one patient
        mse_per_sample    = list(_mse_row)
        ncc_per_sample    = list(_ncc_row)
        ssim_per_sample   = list(_ssim_row)
        geo_per_sample    = list(_geo_row)
        lm_per_sample     = _lm_row
        lm_dvf_per_sample = _lm_dvf_row
        nav_corr_per_sample = [np.nan] * len(patients)

    with open(summary_file, "w") as f:
        f.write("=== Test Metrics Summary ===\n\n")
        f.write(f"Global MSE (vs VM pseudo-GT): {mse_mean:.6f} ± {mse_std:.6f}\n")
        if not np.isnan(mse_gt_mean):
            f.write(f"Global MSE (vs actual GT):    {mse_gt_mean:.6f} ± {mse_gt_std:.6f}  ← matches stacked barplot total\n")
        f.write(f"Global NCC: {ncc_mean:.6f} ± {ncc_std:.6f}\n")
        f.write(f"Global SSIM: {ssim_mean:.6f} ± {ssim_std:.6f}\n")
        f.write(f"Global Geo Error: {geo_mean:.6f} ± {geo_std:.6f}\n")
        if not np.isnan(psnr_mean):
            f.write(f"Global PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB\n")
        if not np.isnan(folding_mean):
            f.write(f"Global Folding Ratio: {folding_mean:.6f} ± {folding_std:.6f}  (fraction det(J)≤0)\n")
        if not np.isnan(smooth_mean):
            f.write(f"Global Temporal Smoothness: {smooth_mean:.6f} ± {smooth_std:.6f}  (mean frame-to-frame DVF L2 diff, lower=smoother)\n")
        if has_lm:
            f.write(f"Global Landmark Error (all):  {lm_mean:.4f} ± {lm_std:.4f} mm\n")
        if not np.isnan(lm_main_mean):
            f.write(f"Global Landmark Error (main): {lm_main_mean:.4f} ± {lm_main_std:.4f} mm\n")
        if not np.isnan(lm_rt_mean):
            f.write(f"Global Landmark Error (RT):   {lm_rt_mean:.4f} ± {lm_rt_std:.4f} mm\n")
        if not np.isnan(lm_gt_mean):
            f.write(f"Global Landmark Error GT: {lm_gt_mean:.4f} ± {lm_gt_std:.4f} mm\n")
        if not np.isnan(lm_dvf_main_mean):
            f.write(f"Global Landmark DVF Error (main): {lm_dvf_main_mean:.4f} ± {lm_dvf_main_std:.4f} mm\n")
        if not np.isnan(lm_dvf_rt_mean):
            f.write(f"Global Landmark DVF Error (RT):   {lm_dvf_rt_mean:.4f} ± {lm_dvf_rt_std:.4f} mm\n")
        if not np.isnan(lm_dvf_mean) and np.isnan(lm_dvf_main_mean) and np.isnan(lm_dvf_rt_mean):
            f.write(f"Global Landmark DVF Error (all):  {lm_dvf_mean:.4f} ± {lm_dvf_std:.4f} mm\n")
        if not np.isnan(nav_ratio):
            f.write(f"Global Navigator Amplitude Ratio: {nav_ratio:.4f} ± 0.0000\n")
        if not np.isnan(nav_corr):
            f.write(f"Global Navigator Phase Correlation: {nav_corr:.4f} ± 0.0000\n")
        if not np.isnan(nav_pred_std_val):
            f.write(f"Global NavPred Std: {nav_pred_std_val:.6f}  (0 = fully collapsed)\n")
        if not np.isnan(nav_pred_cv_val):
            f.write(f"Global NavPred CV: {nav_pred_cv_val:.4f}  (std/mean; <0.01 = collapsed)\n")
        f.write("\n")

        f.write("Per-sample metrics:\n")
        for i, p in enumerate(patients):
            lm_str     = f", LM={lm_per_sample[i]:.4f}mm"         if has_lm              and not np.isnan(lm_per_sample[i])     else ""
            lm_dvf_str = f", LM-DVF={lm_dvf_per_sample[i]:.4f}mm" if lm_dvf_raw is not None and not np.isnan(lm_dvf_per_sample[i]) else ""
            nav_str    = f", NavCorr={nav_corr_per_sample[i]:.4f}" if not np.isnan(nav_corr_per_sample[i])                        else ""
            f.write(
                f"{p}: MSE={mse_per_sample[i]:.6f}, NCC={ncc_per_sample[i]:.6f}, "
                f"SSIM={ssim_per_sample[i]:.6f}, Geo={geo_per_sample[i]:.6f}{lm_str}{lm_dvf_str}{nav_str}\n"
            )

        f.write("\nPer-timestep metrics (mean ± SE over samples):\n")
        n_t = len(mse_by_t)
        for t in range(n_t):
            mse_col  = mse[:, t]  if mse.ndim  > 1 else mse
            ncc_col  = ncc[:, t]  if ncc.ndim  > 1 else ncc
            ssim_col = ssim[:, t] if ssim.ndim > 1 else ssim
            geo_col  = geo[:, t]  if geo.ndim  > 1 else geo
            mse_se   = np.nanstd(mse_col)  / np.sqrt(max(np.sum(~np.isnan(mse_col)),  1))
            ncc_se   = np.nanstd(ncc_col)  / np.sqrt(max(np.sum(~np.isnan(ncc_col)),  1))
            ssim_se  = np.nanstd(ssim_col) / np.sqrt(max(np.sum(~np.isnan(ssim_col)), 1))
            geo_se   = np.nanstd(geo_col)  / np.sqrt(max(np.sum(~np.isnan(geo_col)),  1))
            line = (
                f"  t+{t+1}: MSE={mse_by_t[t]:.6f}±{mse_se:.6f}  "
                f"NCC={ncc_by_t[t]:.6f}±{ncc_se:.6f}  "
                f"SSIM={ssim_by_t[t]:.6f}±{ssim_se:.6f}  "
                f"Geo={geo_by_t[t]:.6f}±{geo_se:.6f}"
            )
            if psnr_by_t is not None:
                psnr_col = psnr[:, t] if psnr.ndim > 1 else psnr
                psnr_se  = np.nanstd(psnr_col) / np.sqrt(max(np.sum(~np.isnan(psnr_col)), 1))
                line += f"  PSNR={psnr_by_t[t]:.2f}±{psnr_se:.2f}dB"
            if folding_by_t is not None:
                fold_col = folding[:, t] if folding.ndim > 1 else folding
                fold_se  = np.nanstd(fold_col) / np.sqrt(max(np.sum(~np.isnan(fold_col)), 1))
                line += f"  Fold={folding_by_t[t]:.6f}±{fold_se:.6f}"
            f.write(line + "\n")

    # --- Tracker comparison block (only when tracker_*.npy files exist) ---
    tr_ncc_file  = os.path.join(fold_dir, "tracker_NCC_loss.npy")
    tr_mse_file  = os.path.join(fold_dir, "tracker_MSE_loss.npy")
    tr_geo_file  = os.path.join(fold_dir, "tracker_geo_error.npy")
    tr_ssim_file = os.path.join(fold_dir, "tracker_SSIM_loss.npy")
    tr_psnr_file = os.path.join(fold_dir, "tracker_PSNR_loss.npy")
    tr_lm_file          = os.path.join(fold_dir, "tracker_landmark_error.npy")
    tr_lm_dvf_file      = os.path.join(fold_dir, "tracker_landmark_dvf_error.npy")
    tr_lm_main_file     = os.path.join(fold_dir, "tracker_landmark_error_main.npy")
    tr_lm_dvf_main_file = os.path.join(fold_dir, "tracker_landmark_dvf_error_main.npy")
    tr_lm_rt_file       = os.path.join(fold_dir, "tracker_landmark_error_rt.npy")
    tr_lm_dvf_rt_file   = os.path.join(fold_dir, "tracker_landmark_dvf_error_rt.npy")
    if os.path.exists(tr_ncc_file) and os.path.exists(tr_mse_file):
        tr_ncc  = np.load(tr_ncc_file, allow_pickle=True).astype(float)
        tr_mse  = np.load(tr_mse_file, allow_pickle=True).astype(float)
        tr_geo  = _load_numeric_npy(tr_geo_file)
        tr_ssim = _load_numeric_npy(tr_ssim_file)
        tr_psnr = _load_numeric_npy(tr_psnr_file)
        # landmark error: object array like lm_file
        tr_lm_raw,         tr_lm_mean,         tr_lm_std         = _load_lm_npy(tr_lm_file)
        tr_lm_dvf_raw,     tr_lm_dvf_mean,     tr_lm_dvf_std     = _load_lm_npy(tr_lm_dvf_file)
        tr_lm_main_raw,    tr_lm_main_mean,    tr_lm_main_std    = _load_lm_npy(tr_lm_main_file)
        tr_lm_dvf_main_raw,tr_lm_dvf_main_mean,tr_lm_dvf_main_std= _load_lm_npy(tr_lm_dvf_main_file)
        tr_lm_rt_raw,      tr_lm_rt_mean,      tr_lm_rt_std      = _load_lm_npy(tr_lm_rt_file)
        tr_lm_dvf_rt_raw,  tr_lm_dvf_rt_mean,  tr_lm_dvf_rt_std  = _load_lm_npy(tr_lm_dvf_rt_file)

        def _stat(a):
            if a is None:
                return np.nan, np.nan
            a = np.asarray(a, dtype=float)
            m = np.nanmean(a)
            s = np.nanstd(a) / np.sqrt(max(np.sum(~np.isnan(a)), 1))
            return m, s

        tr_ncc_m,  tr_ncc_s  = _stat(tr_ncc)
        tr_mse_m,  tr_mse_s  = _stat(tr_mse)
        tr_geo_m,  tr_geo_s  = _stat(tr_geo)
        tr_ssim_m, tr_ssim_s = _stat(tr_ssim)
        tr_psnr_m, tr_psnr_s = _stat(tr_psnr)

        def _row(label, cldm_m, cldm_s, tr_m, tr_s):
            if np.isnan(tr_m):
                return ""
            return (f"  {label:<5}: {cldm_m:10.6f} ± {cldm_s:.6f}   "
                    f"{tr_m:10.6f} ± {tr_s:.6f}   {tr_m - cldm_m:+.6f}\n")

        with open(summary_file, "a") as f:
            f.write("\n=== Tracker vs CLDM comparison ===\n")
            f.write(f"{'':6}  {'CLDM':>28}   {'Tracker':>28}   Delta (Tracker-CLDM)\n")
            f.write(_row("MSE",  mse_mean,  mse_std,  tr_mse_m,  tr_mse_s))
            f.write(_row("NCC",  ncc_mean,  ncc_std,  tr_ncc_m,  tr_ncc_s))
            f.write(_row("SSIM", ssim_mean, ssim_std, tr_ssim_m, tr_ssim_s))
            f.write(_row("Geo",  geo_mean,  geo_std,  tr_geo_m,  tr_geo_s))
            if not np.isnan(psnr_mean) and not np.isnan(tr_psnr_m):
                f.write(_row("PSNR", psnr_mean, psnr_std, tr_psnr_m, tr_psnr_s))
            def _lm_row(label, cldm_m, cldm_s, tr_m, tr_s):
                if np.isnan(cldm_m) or np.isnan(tr_m):
                    return ""
                return (f"  {label:<12}: {cldm_m:8.4f} ± {cldm_s:.4f}   "
                        f"{tr_m:8.4f} ± {tr_s:.4f}   {tr_m - cldm_m:+.4f} mm\n")
            if has_lm and not np.isnan(tr_lm_mean):
                f.write(_lm_row("LM (all)",  lm_mean,      lm_std,      tr_lm_mean,      tr_lm_std))
                f.write(_lm_row("LM (main)", lm_main_mean, lm_main_std, tr_lm_main_mean, tr_lm_main_std))
                f.write(_lm_row("LM (RT)",   lm_rt_mean,   lm_rt_std,   tr_lm_rt_mean,   tr_lm_rt_std))
            if not np.isnan(lm_dvf_mean) and not np.isnan(tr_lm_dvf_mean):
                f.write(_lm_row("LM-DVF (all)",  lm_dvf_mean,      lm_dvf_std,      tr_lm_dvf_mean,      tr_lm_dvf_std))
                f.write(_lm_row("LM-DVF (main)", lm_dvf_main_mean, lm_dvf_main_std, tr_lm_dvf_main_mean, tr_lm_dvf_main_std))
                f.write(_lm_row("LM-DVF (RT)",   lm_dvf_rt_mean,   lm_dvf_rt_std,   tr_lm_dvf_rt_mean,   tr_lm_dvf_rt_std))

            f.write("\nPer-timestep tracker metrics (mean over samples):\n")
            tr_ncc_by_t  = np.nanmean(tr_ncc,  axis=0).ravel() if tr_ncc.ndim  > 1 else np.array([np.nanmean(tr_ncc)])
            tr_mse_by_t  = np.nanmean(tr_mse,  axis=0).ravel() if tr_mse.ndim  > 1 else np.array([np.nanmean(tr_mse)])
            tr_ssim_by_t = np.nanmean(tr_ssim, axis=0).ravel() if tr_ssim is not None and np.asarray(tr_ssim).ndim > 1 else (np.array([np.nanmean(tr_ssim)]) if tr_ssim is not None else None)
            tr_psnr_by_t = np.nanmean(tr_psnr, axis=0).ravel() if tr_psnr is not None and np.asarray(tr_psnr).ndim > 1 else (np.array([np.nanmean(tr_psnr)]) if tr_psnr is not None else None)
            for t in range(len(tr_ncc_by_t)):
                line = f"  t+{t+1}: MSE={tr_mse_by_t[t]:.6f}  NCC={tr_ncc_by_t[t]:.6f}"
                if tr_ssim_by_t is not None:
                    line += f"  SSIM={tr_ssim_by_t[t]:.6f}"
                if tr_psnr_by_t is not None:
                    line += f"  PSNR={tr_psnr_by_t[t]:.2f}dB"
                f.write(line + "\n")

    print(f"✅ Summary saved to {summary_file}")
    
def to_device_volumes(ref, current_list, device):
    ref = ref.unsqueeze(1).to(device)
    current = [v.unsqueeze(1).to(device) for v in current_list]
    return ref, current


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)

from .config import load_config, _apply_overrides  # noqa: F401

from ..models.VAE import DVFVAE
_VAE_REGISTRY: dict[str, type] = {
    "dvfvae": DVFVAE,
}

def build_vae(cfg: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Instantiate the VAE specified by *cfg.model* using its config sub-block."""
    model_key = cfg.vae_model.lower()
    if model_key not in _VAE_REGISTRY:
        raise ValueError(
            f"Unknown model '{cfg.vae_model}'.  "
            f"Available: {list(_VAE_REGISTRY.keys())}"
        )

    cls    = _VAE_REGISTRY[model_key]
    prefix = f"{model_key}_"

    # Collect only the kwargs that belong to this model and device
    kwargs = {
        k[len(prefix):]: v
        for k, v in vars(cfg).items()
        if k.startswith(prefix)
    }
    kwargs["device"] = device

    print(f"[factory] Building {cls.__name__} with kwargs: {kwargs}")
    return cls(**kwargs).to(device)

def build_optimizer(cfg: argparse.Namespace,
                    parameters) -> torch.optim.Optimizer:
    key = cfg.optimizer.lower()
    lr  = float(cfg.lr_vae)
    wd  = float(getattr(cfg, "weight_decay", 1e-4))

    if key == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    if key == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
    if key == "sgd":
        momentum = float(getattr(cfg, "momentum", 0.9))
        return torch.optim.SGD(parameters, lr=lr,
                               weight_decay=wd, momentum=momentum)
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'. "
                     "Choose from: adam, adamw, sgd")


def build_scheduler(cfg: argparse.Namespace,
                    optimizer: torch.optim.Optimizer,
                    train_loader_len: int):
    key         = getattr(cfg, "scheduler", "cosine").lower()
    vae_epochs  = cfg.vae_epochs

    if key == "cosine":
        total_steps = vae_epochs * train_loader_len
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = total_steps,
            eta_min = float(getattr(cfg, "cosine_eta_min", 1e-6)),
        )
    if key == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode      = "min",
            factor    = float(getattr(cfg, "plateau_factor",   0.5)),
            patience  = int(getattr(cfg,   "plateau_patience", 5)),
            min_lr    = float(getattr(cfg, "plateau_min_lr",   1e-10)),
            threshold = float(getattr(cfg, "plateau_threshold", 1e-4)),
        )
    if key == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = int(getattr(cfg,   "step_size",  10)),
            gamma     = float(getattr(cfg, "step_gamma", 0.5)),
        )
    if key == "none":
        return None
    raise ValueError(f"Unknown scheduler '{cfg.scheduler}'. "
                     "Choose from: cosine, plateau, step, none")

def vae_checkpoint_path(cfg: argparse.Namespace, dir_name: str, fold_idx: str) -> str:
    """Canonical path where train_vae() saves its best checkpoint."""
    return os.path.join(dir_name, f"fold_{fold_idx}", "vae", "model_best_vae.pth"
    )

def _kwargs_for(cfg: argparse.Namespace, prefix: str) -> dict:
    """Extract all ``{prefix}_*`` keys from *cfg* as a plain dict."""
    pfx = f"{prefix}_"
    return {k[len(pfx):]: v for k, v in vars(cfg).items() if k.startswith(pfx)}

from ..models.CLDM import UNet3D
_DIFFUSION_REGISTRY: dict[str, type] = {
    "unet3d": UNet3D,
}

import torch.nn as nn
def build_diffusion(
    cfg: argparse.Namespace,
    vae: nn.Module,
    stn: nn.Module | None,
    device: torch.device,
    fold_idx: str = "0",
) -> nn.Module:
    """Build the diffusion model, injecting the pre-built *vae* and *stn*."""
    key = cfg.diffusion_model.lower()
    if key not in _DIFFUSION_REGISTRY:
        raise ValueError(
            f"Unknown diffusion_model '{key}'. "
            f"Available: {list(_DIFFUSION_REGISTRY)}"
        )
    cls = _DIFFUSION_REGISTRY[key]
    # monai_unet3d and unet3d_ reuse the unet3d: block in the YAML config
    cfg_key = "unet3d" if key in ("monai_unet3d", "unet3d_", "unet3d_novae") else key
    kwargs  = _kwargs_for(cfg, cfg_key)
    # CFG warmup/rampup schedule is consumed directly by the training loop
    # (see train_CLDM.py), not by the model constructor.
    kwargs.pop("cfg_warmup_epochs", None)
    kwargs.pop("cfg_rampup_epochs", None)

    # CausalDiT needs the spatial transformer; UNet3D carries its own.
    if key == "causal_dit":
        kwargs["stn"] = stn

    # image_mode: VAE operates on images directly, no DVF/STN needed.
    if getattr(cfg, "image_mode", False) and key in ("unet3d", "unet3d_", "monai_unet3d", "unet3d_novae"):
        kwargs["image_mode"] = True

    # Inject shared DDPM schedule params from top-level config.
    kwargs.setdefault("T",          cfg.T)
    kwargs.setdefault("ddim_steps", cfg.ddim_steps)
    if hasattr(cfg, "sampler"):
        kwargs["sampler"] = cfg.sampler

    # Inject SSG inference params — overridable via --override ssg_scale=X etc.
    for _ssg_key, _ssg_default in (("ssg_scale", 1.0), ("ssg_swap_ratio", 0.1), ("ssg_mode", "both")):
        if hasattr(cfg, _ssg_key):
            kwargs[_ssg_key] = getattr(cfg, _ssg_key)
            
    if key in ("unet3d", "unet3d_novae"):
        print(getattr(cfg, "smooth_weight", 0.0))
        kwargs.setdefault("recon_weight",  getattr(cfg, "recon_weight",  0.0))
        kwargs.setdefault("vmorph_weight", getattr(cfg, "smooth_weight", 0.0))

    model = cls(vae=vae, **kwargs).to(device)
    
    _handle_conditional_encoders(cfg, model, device, fold_idx=fold_idx)

    print(f"[factory] Building diffusion model: {cls.__name__}  kwargs={kwargs}")
    return model


def _check_weights(module: torch.nn.Module, name: str, n: int = 3) -> None:
    """Print norm/mean of the first *n* parameter tensors to verify a load."""
    params = [(pname, p) for pname, p in module.named_parameters()][:n]
    for pname, p in params:
        print(f"  [{name}] {pname}: norm={p.data.norm():.4f}  mean={p.data.mean():.4f}")


def _handle_conditional_encoders(cfg, model, device, fold_idx="0"):
    """
    Handles loading/freezing or training of conditional encoders
    depending on VAE type.
    """

    vae_type = cfg.vae_model.lower()
    diff_type = cfg.diffusion_model.lower()  
    
    if diff_type in ("unet3d", "unet3d_", "unet3d_rayan", "monai_unet3d", "monai_diffusion", "unet3d_novae"):

        # ---------------------------------------------------------
        # CASE 1 — CVAE → load pretrained encoders + freeze
        # ---------------------------------------------------------
        if vae_type == "ir_cvae"or vae_type == "ir_cvae_new" or vae_type == "ir_cvae_nodec" or vae_type == "ir_cvae_compress" or vae_type=="cvae_old":
            print("[Diffusion] Using pretrained conditional encoders (CVAE)")
            
            if not hasattr(model, "cond_net") or not hasattr(model, "ref_net"):
                raise ValueError("Model must define cond_net and ref_net for CVAE.")

            # Load weights
            if getattr(cfg, "cond_encoder_checkpoint", None):
                print(f"[Diffusion] Loading conditional encoders from {cfg.cond_encoder_checkpoint}")
                ckpt = torch.load(cfg.cond_encoder_checkpoint, map_location=device, weights_only=False)
                model.cond_net.load_state_dict(ckpt["cond_net"])
                model.ref_net.load_state_dict(ckpt["ref_net"])
            elif hasattr(model.vae, "cond_net") and hasattr(model.vae, "ref_net"):
                print("[Diffusion] Copying cond_net / ref_net weights from pretrained CVAE.")
                model.cond_net.load_state_dict(model.vae.cond_net.state_dict())
                model.ref_net.load_state_dict(model.vae.ref_net.state_dict())
            else:
                print("[Warning] No conditional encoder checkpoint provided and VAE has no cond_net/ref_net.")

            # print("[Check] cond_net weights after load:")
            # _check_weights(model.cond_net, "cond_net")
            # print("[Check] ref_net weights after load:")
            # _check_weights(model.ref_net, "ref_net")

            # Freeze
            for enc in [model.cond_net, model.ref_net]:
                enc.eval()
                for p in enc.parameters():
                    p.requires_grad_(False)

                print(f"[Diffusion] Conditional encoders {enc.__class__.__name__} frozen.")

        # ---------------------------------------------------------
        # CASE 2 — VQ-VAE / MIA-VAE → train encoders jointly,
        #          or load a pretrained TMNet and partially freeze it
        #          if tmnet_path is provided in the config.
        # ---------------------------------------------------------
        elif vae_type in ["vqvae", "mia_vae"]:
            tmnet_path = getattr(cfg, "tmnet_path", None)

            if tmnet_path:
                print(f"[Diffusion] Loading pretrained cond_net from {tmnet_path}")
                model.cond_net = _load_tmnet(tmnet_path, device)
                print(f"[Diffusion] cond_net backbone+transformer frozen; linear trains.")
            else:
                print("[Diffusion] Training cond_net jointly.")
                if not hasattr(model, "cond_net"):
                    raise ValueError("Model must define cond_net.")
                model.cond_net.train()
                for p in model.cond_net.parameters():
                    p.requires_grad_(True)

            rvnet_path = getattr(cfg, "rvnet_path", None)

            if rvnet_path:
                print(f"[Diffusion] Loading pretrained ref_net from {rvnet_path}")
                model.ref_net = _load_rvnet(rvnet_path, device)
                model.ref_net.eval()
                for p in model.ref_net.parameters():
                    p.requires_grad_(False)
                print(f"[Diffusion] ref_net frozen.")
            else:
                print("[Diffusion] Training ref_net jointly.")
                if not hasattr(model, "ref_net"):
                    raise ValueError("Model must define ref_net.")
                model.ref_net.train()
                for p in model.ref_net.parameters():
                    p.requires_grad_(True)

        # ---------------------------------------------------------
        # CASE 3 — plain VAE → load pretrained tmnet + optional rvnet
        # ---------------------------------------------------------
        elif vae_type == "dvfvae":
            print("[Diffusion] dvfvae")
            tmnet_path = getattr(cfg, "tmnet_path", None)
            if tmnet_path:
                print(f"[Diffusion] Loading pretrained cond_net from {tmnet_path}")
                model.cond_net = _load_tmnet(tmnet_path, device)
            else:
                print("[Diffusion] Training cond_net jointly (no tmnet_path given).")

            rvnet_path = getattr(cfg, "rvnet_path", None)
            if rvnet_path:
                print(f"[Diffusion] Loading pretrained ref_net from {rvnet_path}")
                model.ref_net = _load_rvnet(rvnet_path, device)
            else:
                print("[Diffusion] Training ref_net jointly (no rvnet_path given).")

        else:
            raise ValueError(f"Unknown vae_type: {vae_type}")

    elif diff_type == "causal_dit":
        # ------------------------------------------------------------------
        # CausalDiT — optionally replace iseq_enc / ref_enc with frozen
        # pretrained encoders wrapped in a trainable linear projection.
        # ------------------------------------------------------------------
        tmnet_path   = getattr(cfg, "tmnet_path",   None)
        rvnet_path = getattr(cfg, "rvnet_path", None)

        if tmnet_path:
            print(f"[DiT] Loading pretrained Iseq encoder from {tmnet_path}")
            frozen_enc  = _load_tmnet_encoder_for_dit(tmnet_path, device)
            hidden_dim  = frozen_enc.hidden_dim
            d_model     = model.iseq_enc.frame_enc.net[-1].out_features  # d_model from existing enc
            model.iseq_enc = TMNetDiTAdapter(frozen_enc, hidden_dim, d_model).to(device)
            print(f"[DiT] iseq_enc replaced: TMNetDiTAdapter (hidden={hidden_dim} → d_model={d_model})")
            for name, p in model.iseq_enc.named_parameters():
                p.requires_grad_(not name.startswith("encoder."))
            print(f"[Diffusion] cond_net backbone+transformer frozen; proj/norm train.")
        else:
            print("[DiT] No tmnet_path — iseq_enc trains from scratch.")

        if rvnet_path:
            print(f"[DiT] Loading pretrained Ref encoder from {rvnet_path}")
            frozen_ref      = _load_rvnet(rvnet_path, device)
            cfg_path        = _Path(rvnet_path).parent / "RVNet.yaml"
            rcfg            = load_config(str(cfg_path))
            backbone_out_ch = rcfg.rvnet_enc_channels[-1]
            nb_convs        = len(rcfg.rvnet_enc_channels)
            vol_size        = tuple(rcfg.vol_size)
            d_model         = model.ref_enc.proj.out_features  # d_model from existing enc
            model.ref_enc = RVNetDiTAdapter(
                frozen_ref, backbone_out_ch, d_model, vol_size, nb_convs
            ).to(device)
            for name, p in model.ref_enc.named_parameters():
                p.requires_grad_(not name.startswith("backbone."))
            print(f"[Diffusion] ref_net backbone frozen; proj/norm/pos_emb train.")
            print(f"[DiT] ref_enc replaced: RVNetDiTAdapter (hidden={backbone_out_ch} → d_model={d_model})")
        else:
            print("[DiT] No rvnet_path — ref_enc trains from scratch.")

    else:
        print(f"[Diffusion] No special handling for diffusion model type '{diff_type}'.")
        
from ..models.Context_Encoder import (
    RVNet as _RVNet,
    TMNet_Tr_priormulti as _TMNet,
    TMNetEncoder as _TMNetEncoder,
    TMNetDiTAdapter,
    RVNetDiTAdapter,
)
from pathlib import Path as _Path

def _load_tmnet(path: str, device: torch.device) -> _TMNet:
    """Build TMNet_Tr_priormulti from its training config and load weights.

    The checkpoint is typically from TMNet_Tr_priormulti_image which deleted
    self.linear, so that layer stays randomly initialized.  Only the backbone
    and transformer (pretrained) are loaded; linear trains during diffusion.
    Returns the model with backbone+transformer frozen and linear trainable.
    """
    _ckpt_dir = _Path(path).parent
    for _candidate in ("TMNet.yaml", "TMNet_acdc.yaml", "TMNet_mm.yaml", "TMNet_horizon3.yaml", "TMNet_input5.yaml", "TMNet_horizon3.yaml"):
        if (_ckpt_dir / _candidate).exists():
            cfg_path = _ckpt_dir / _candidate
            break
    else:
        raise FileNotFoundError(
            f"[_load_tmnet] No TMNet*.yaml found in {_ckpt_dir}"
        )
    ccfg     = load_config(str(cfg_path))

    model = _TMNet(
        num_inputs       = ccfg.tmnet_num_frames,
        horizon          = ccfg.tmnet_horizon,
        in_channels      = 2,
        out_channels     = ccfg.tmnet_condi_channels,
        n_heads          = ccfg.tmnet_Tr_n_heads,
        enc_layers       = ccfg.tmnet_Tr_enc_layers,
        dec_layers       = ccfg.tmnet_Tr_dec_layers,
        normalize_before = ccfg.tmnet_Tr_norm_before,
        output_dim       = ccfg.tmnet_pre_latent_dim,
        rnn              = "transformer",
        condi_type       = ccfg.tmnet_condi_type,
        prior_type       = ccfg.tmnet_prior_type,
        device           = device,
    ).to(device)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd   = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[Diffusion] cond_net: {len(unexpected)} unexpected keys skipped (e.g. projector)")
    if missing:
        print(f"[Diffusion] cond_net: {len(missing)} missing keys will train (e.g. {missing[:2]})")

    # Freeze pretrained backbone + transformer; leave linear trainable.
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("linear"))

    return model


def _load_rvnet(path: str, device: torch.device) -> _RVNet:
    """Build a RVNet from its training config and load checkpoint weights.

    The checkpoint was saved from RefCondMAE where the inner RVNet is
    stored under the "encoder.*" prefix.  The correct architecture is read
    from RVNet.yaml sitting next to the checkpoint so that dimension
    mismatches (enc_channels, linear_input_dim) are impossible.
    """
    cfg_path = _Path(path).parent / "RVNet.yaml"
    rcfg     = load_config(str(cfg_path))

    enc_channels = rcfg.rvnet_enc_channels          # e.g. [32, 64]
    output_dim   = rcfg.rvnet_pre_latent_dim         # e.g. 16
    vol_size     = rcfg.vol_size                       # e.g. [32, 64, 64]
    nb_convs     = len(enc_channels)

    # Each stride-2 conv halves each spatial dim; adap reduces to 1 channel.
    linear_input_dim = 1
    for s in vol_size:
        linear_input_dim *= s // (2 ** nb_convs)

    norm_fn = lambda ch: nn.GroupNorm(min(8, ch), ch)
    rvnet = _RVNet(
        nb_convs         = nb_convs,
        in_channels      = 1,
        out_channels     = enc_channels,
        output_dim       = output_dim,
        linear_input_dim = linear_input_dim,
        norm             = norm_fn,
    ).to(device)

    ckpt  = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    # MoCo:           encoder_q.*       → strip "encoder_q." → RVNet keys
    # SparKRVNet: encoder.encoder.* → strip one "encoder." → RVNet keys
    # RefCondMAE:      encoder.*         → strip one "encoder." → RVNet keys
    if any(k.startswith("encoder_q.") for k in state):
        prefix = "encoder_q."
    else:
        prefix = "encoder."
    ref_state = {k[len(prefix):]: v
                 for k, v in state.items() if k.startswith(prefix)}
    rvnet.load_state_dict(ref_state)

    # Freeze backbone + adap; leave dvf_enc trainable.
    for name, p in rvnet.named_parameters():
        p.requires_grad_(name.startswith("dvf_enc"))
    print(f"[Diffusion] ref_net: backbone frozen, dvf_enc trains from checkpoint.")
    return rvnet

def _load_tmnet_encoder_for_dit(path: str, device: torch.device) -> _TMNetEncoder:
    """Load a TMNetEncoder from a TMNet_Tr_priormulti_image checkpoint.

    Key remapping handles the two structural differences between the checkpoint
    and TMNetEncoder:
      - 'transformer.encoder.*' → 'transformer_encoder.*'
      - BatchNorm running stats in the checkpoint are dropped (strict=False);
        the conv and GroupNorm affine weights load correctly by index.

    Returns a TMNetEncoder with all parameters frozen.
    """
    _ckpt_dir = _Path(path).parent
    for _candidate in ("TMNet.yaml", "TMNet_acdc.yaml", "TMNet_mm.yaml"):
        if (_ckpt_dir / _candidate).exists():
            cfg_path = _ckpt_dir / _candidate
            break
    else:
        raise FileNotFoundError(
            f"[_load_tmnet_encoder_for_dit] No TMNet*.yaml found in {_ckpt_dir}"
        )
    ccfg     = load_config(str(cfg_path))

    encoder = _TMNetEncoder(
        num_inputs       = ccfg.tmnet_num_frames,
        horizon          = ccfg.tmnet_horizon,
        in_channels      = 2,
        out_channels     = ccfg.tmnet_condi_channels,
        n_heads          = ccfg.tmnet_Tr_n_heads,
        enc_layers       = ccfg.tmnet_Tr_enc_layers,
        normalize_before = ccfg.tmnet_Tr_norm_before,
        output_dim       = ccfg.tmnet_pre_latent_dim,
        condi_type       = ccfg.tmnet_condi_type,
        device           = device,
    ).to(device)

    ckpt  = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)

    # Remap transformer.encoder.* → transformer_encoder.*
    remapped = {}
    for k, v in state.items():
        if k.startswith("transformer.encoder."):
            remapped["transformer_encoder." + k[len("transformer.encoder."):]] = v
        else:
            remapped[k] = v

    missing, unexpected = encoder.load_state_dict(remapped, strict=False)
    if unexpected:
        print(f"[DiT] condi encoder: {len(unexpected)} unexpected keys skipped")
    if missing:
        print(f"[DiT] condi encoder: {len(missing)} missing keys kept random "
              f"(e.g. {missing[:2]})")

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder


from ..data.loading import  extract_patient_ids
def save_patients(log_dir: str, folds: list[list[str]]):
    # extract_patient_ids uses seq[:8] which truncates ACDC "patientXXX" (10 chars)
    # to "patient0". Use it only when entries contain path separators (liver pipeline);
    # otherwise the entries are already plain patient IDs.
    def _ids(seq_list):
        if any("/" in s for s in seq_list):
            return extract_patient_ids(seq_list)
        return sorted(set(seq_list))

    train_patients = _ids(folds[0])
    val_patients = _ids(folds[1])

    with open(os.path.join(log_dir, "patients_split.txt"), "w") as f:
        f.write("=== TRAIN PATIENTS ===\n")
        for p in train_patients:
            f.write(p + "\n")

        f.write("\n=== VALIDATION PATIENTS ===\n")
        for p in val_patients:
            f.write(p + "\n")
     
from ..models.Context_Encoder import TMNetEncoder, TMNet_Tr_priormulti_image
import types as _types

def _build_tmnet_model(cfg, device: torch.device):
    """
    Instantiate TMNetEncoder + TMNetPriorMAE from config.

    Returns
    -------
    condi : TMNetEncoder instance
    mae   : TMNetPriorMAE wrapper (condi backbone + MAE decoder)
    """
    condi = TMNetEncoder(
        num_inputs       = cfg.tmnet_num_frames,
        horizon          = cfg.tmnet_horizon,
        in_channels      = 2,
        out_channels     = cfg.tmnet_condi_channels,
        n_heads          = cfg.tmnet_Tr_n_heads,
        enc_layers       = cfg.tmnet_Tr_enc_layers,
        normalize_before = cfg.tmnet_Tr_norm_before,
        output_dim       = cfg.tmnet_pre_latent_dim,
        condi_type       = cfg.tmnet_condi_type,
        device           = device,
    ).to(device)

    img_size    = tuple(cfg.tmnet_img_size)
    in_channels = getattr(cfg, "tmnet_in_channels", 2)

    mae = TMNetPriorMAE(
        tm_net    = condi,
        img_size     = img_size,
        img_channels = in_channels,
        mask_ratio   = cfg.tmnet_mask_ratio,
        patch_size   = cfg.tmnet_patch_size,
    ).to(device)

    return condi, mae


def _build_imgsup_model(cfg, device: torch.device) -> TMNet_Tr_priormulti_image:
    """
    Build TMNet_Tr_priormulti_image from config.

    After loading the checkpoint into the returned model, call
    ``_patch_imgsup_as_encoder`` to add the context-encoder interface.
    """
    img_h, img_w = tuple(cfg.tmnet_img_size)
    return TMNet_Tr_priormulti_image(
        num_inputs       = cfg.tmnet_num_frames,
        horizon          = cfg.tmnet_horizon,
        in_channels      = 2,
        out_channels     = cfg.tmnet_condi_channels,
        n_heads          = cfg.tmnet_Tr_n_heads,
        enc_layers       = cfg.tmnet_Tr_enc_layers,
        dec_layers       = cfg.tmnet_Tr_dec_layers,
        normalize_before = cfg.tmnet_Tr_norm_before,
        output_dim       = cfg.tmnet_pre_latent_dim,
        rnn              = "transformer",
        condi_type       = cfg.tmnet_condi_type,
        prior_type       = cfg.tmnet_prior_type,
        device           = device,
        img_h            = img_h,
        img_w            = img_w,
    ).to(device)


def _patch_imgsup_as_encoder(model: TMNet_Tr_priormulti_image) -> None:
    """
    Monkey-patch a loaded TMNet_Tr_priormulti_image so it exposes the same
    interface as TMNetEncoder (encode_temporal + forward returning features).

    The transformer encoder features are pooled at the last valid frame and
    used directly as context (dim = hidden_dim) without an extra linear head.
    """

    def encode_temporal(self, frames, attn_mask=None):
        B, C, T, H, W = frames.shape
        feats = torch.stack([self.backbone(frames[:, :, t]) for t in range(T)], dim=1)
        src      = self.input_proj(feats)
        pos      = self.pe3d(src).to(src.dtype)
        src_flat = src.flatten(-3).permute(1, 0, 2)    # (T, B, hidden_dim)
        pos_flat = pos.flatten(-3).permute(1, 0, 2)
        frame_energy         = frames.abs().sum(dim=(1, 3, 4))
        src_key_padding_mask = (frame_energy == 0).to(frames.device)
        if not src_key_padding_mask.any():
            src_key_padding_mask = None
        enc_out = self.transformer.encoder(
            src_flat, pos=pos_flat, src_key_padding_mask=src_key_padding_mask,
        )
        return enc_out.permute(1, 0, 2)               # (B, T, hidden_dim)

    def forward_ctx(self, Ipast):
        hs           = self.encode_temporal(Ipast)    # (B, T, hidden_dim)
        frame_energy = Ipast.abs().sum(dim=(1, 3, 4))
        valid        = (frame_energy > 0).long()
        last_idx     = valid.cumsum(dim=1).argmax(dim=1)
        pooled       = hs[torch.arange(hs.shape[0], device=hs.device), last_idx]
        return [pooled] * self.horizon, None

    model.encode_temporal = _types.MethodType(encode_temporal, model)
    model.forward         = _types.MethodType(forward_ctx,     model)
    model.output_dim      = model.hidden_dim            # used by _infer_context_dim
