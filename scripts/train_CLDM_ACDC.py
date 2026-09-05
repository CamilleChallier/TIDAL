"""
train_CLDM_ACDC.py
==================
ACDC cardiac adaptation of train_CLDM.py — Stage 2 diffusion training on ACDC.
Standalone copy: do not add liver switches here.

Requires a trained VAE (vae_dir_name) and TMNet checkpoint (tmnet_path)
in configs/CLDM/UNet3D_acdc.yaml before running.

Usage
-----
# Stage 2 — train
python -m scripts.train_CLDM_ACDC \
    --config configs/CLDM/UNet3D_acdc.yaml \
    --train_test train

# Stage 2 — test
python -m scripts.train_CLDM_ACDC \
    --config configs/CLDM/UNet3D_acdc.yaml \
    --train_test test \
    --checkpoint <path/to/logs/run_dir>
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ss
from torch.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, dynamic_ncols=True)

# --- Diffusion models ---
from mopred.models.CLDM import UNet3D

# --- Shared infrastructure ---
from mopred.models import Voxelmorph, SpatialTransformer, EMA

from mopred.data.splits                     import make_folds_acdc as make_train_val_test_folds
from mopred.data.data_loaders.acdc_4d import (
    ACDC_4D_Dataset, CANONICAL_SHAPE, LIVER_SHAPE, get_cardiac_phase,
    get_reference_slice_index, load_orientation_info
)
from mopred.data.loading      import save_params_txt, build_Iseq
# temp_crop not used for ACDC — no liver respiratory cycle gating
from mopred.utils.early_stopping  import EarlyStopping
from mopred.utils.assemble_volumes import assemble_volumes
from mopred.utils.io              import cond_mkdir, custom_load, custom_save, save_tensor_as_nifti
from mopred.utils.losses          import build_criterion, gradient_loss, ncc_loss
from mopred.utils.training        import load_config, _apply_overrides, vae_checkpoint_path, build_scheduler, build_optimizer, build_vae, save_patients, build_diffusion, summarize_test_metrics, _check_weights, build_reg_model
from mopred.utils.dvf_cache        import build_dvf_cache
from mopred.utils.dvf_metrics     import motion_amplitude, hf_energy_ratio, jacobian_det_stats, jacobian_folding_ratio, dvf_cosine_sim_stats, dvf_diversity_diagnostic, geo_error
from mopred.utils.navigator       import navigator_signal_3planes, navigator_metrics

_VAE_ALIGN_PREFIXES = ("context_encoder.", "align_loss_fn.")

# Cardiac phase phi ∈ [0, 0.5]: 0 at ED (reference), 0.5 at ES (peak contraction).
def _get_phi(patient: str, t_idx: int, data_dir: str) -> float:
    phase = get_cardiac_phase(patient, t_idx, data_dir)
    return phase #min(phase, 1.0 - phase)

def _acdc_dataset_kwargs(cfg) -> dict:
    """Shared ACDC_4D_Dataset kwargs derived from cfg — used by every
    instantiation site (train/valid/test) so they can never drift apart and
    silently end up preprocessed differently from one another."""
    return dict(
        downsample_to_liver=getattr(cfg, "downsample_to_liver", False),
        cache_dir="/volatile/data/ACDC/cache_preprocessed",
        reference_slices_path=getattr(cfg, "reference_slices_path", None),
        stride=getattr(cfg, "stride", 1),
    )

def compute_phase(vol_files, horizon: int, device) -> torch.Tensor | None:
    """
    Compute respiratory phase phi ∈ [0, 1] from output file paths.

    phi is derived from the diaphragm displacement curve:
      phi = 0   at exhale (reference)
      phi = 0.5 at inhale (peak displacement)
      phi → 1   returning to exhale

    The first call per patient loads all 31 frames to detect the diaphragm
    position; subsequent calls are served from an in-memory cache.

    Parameters
    ----------
    vol_files : list of (str or list-of-str), length = batch_size
    horizon   : number of predicted timepoints
    device    : torch.device

    Returns (B, horizon) float32 tensor, or None if paths cannot be parsed.
    """
    try:
        # DataLoader collates list-of-lists as (horizon × batch).
        # Transpose back to (batch × horizon) so each entry = one sample.
        if vol_files and isinstance(vol_files[0], (list, tuple)):
            vol_files = list(zip(*vol_files))

        phases = []
        for entry in vol_files:
            paths = entry if isinstance(entry, (list, tuple)) else [entry]
            row = []
            for path in paths[:horizon]:
                patient  = path.split("/")[-2]
                t_idx    = int(path.split("/")[-1][2:-7])
                data_dir = os.path.dirname(os.path.dirname(path))
                phi      = _get_phi(patient, t_idx, data_dir)
                row.append(phi)
            while len(row) < horizon:
                row.append(row[-1])
            phases.append(row)
        return torch.tensor(phases, dtype=torch.float32, device=device)
    except Exception:
        return None

def _load_vae_for_diffusion(vae, path, device):
    """Load a VAE checkpoint, silently dropping alignment-only keys.

    VAEs trained with context alignment store context_encoder.* and
    align_loss_fn.* in the checkpoint.  Those submodules are not present
    in the diffusion-side VAE (built without context_encoder_path), so they
    must be stripped before load_state_dict to avoid a strict-mode failure.
    """
    whole_dict = torch.load(path, map_location=device)
    sd = whole_dict.get("model", whole_dict)
    sd = {k: v for k, v in sd.items()
          if not any(k.startswith(p) for p in _VAE_ALIGN_PREFIXES)}
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[Stage 2] VAE load: {len(unexpected)} unexpected keys skipped")
    if missing:
        print(f"[Stage 2] VAE load: {len(missing)} missing keys — check checkpoint")

_ed_seg_cache: dict[str, torch.Tensor] = {}   # patient -> (D,H,W) canonical-space mask
_ed_seg_cache: dict[str, torch.Tensor | None] = {}   # patient -> mask or None (missing)

def _load_ed_seg(patient: str, cfg, device, orientation: dict | None) -> torch.Tensor | None:
    """Binary heart mask (RV+Myo+LV) at ED, in the SAME canonical space as the
    preprocessed images — same resample/rotate/crop/downsample chain as
    `_load_and_preprocess` in acdc_4d.py, but with nearest-neighbor interpolation
    for the label-map steps. Returns None if the ED segmentation file is missing
    for this patient (organ-mask weighting is then skipped for that patient,
    NOT set to bg_weight — see _training_forward's seg_valid handling)."""
    if patient in _ed_seg_cache:
        return _ed_seg_cache[patient]

    from mopred.data.data_loaders.acdc_4d import (
        _patient_info, _native_to_resampled_index, _native_to_resampled_point,
        _resample_to_spacing, _rotate_hw, _center_crop_or_pad, _downsample_to_liver_shape,
    )
    import nibabel as nib

    info = _patient_info(patient, cfg.data_dir)
    ed = info["ED"]
    seg_path = os.path.join(cfg.data_dir, patient, f"{patient}_frame{ed:02d}_gt.nii.gz")

    if not os.path.exists(seg_path):
        print(f"[Stage 2] Warning: no ED segmentation for {patient} "
              f"({seg_path}) — organ-mask weighting skipped for this patient.")
        _ed_seg_cache[patient] = None
        return None

    seg_native = np.asarray(nib.load(seg_path).dataobj, dtype=np.int32)   # (H, W, D)
    seg_native = np.transpose(seg_native, (2, 0, 1))                       # -> (D, H, W)

    spacing = info["spacing"]
    entry   = orientation.get(patient) if orientation is not None else None

    ref_slice       = entry.get("reference_slice") if entry else None
    centroid_native = entry.get("centroid") if entry else None
    rotation_angle  = entry.get("rotation_angle") if entry else None

    center_d = center_h = center_w = None
    if centroid_native is not None:
        center_d, center_h, center_w = _native_to_resampled_point(centroid_native, spacing)
    if ref_slice is not None:          # reference_slice wins over centroid's D estimate
        center_d = _native_to_resampled_index(ref_slice, spacing[0])

    seg = torch.from_numpy(np.ascontiguousarray(seg_native)).float()
    seg = _resample_to_spacing(seg, spacing, mode="nearest")       # labels: nearest
    if rotation_angle:
        pivot_h = center_h if center_h is not None else seg.shape[1] // 2
        pivot_w = center_w if center_w is not None else seg.shape[2] // 2
        seg = _rotate_hw(seg, rotation_angle, (pivot_h, pivot_w), mode="nearest")
    seg = _center_crop_or_pad(seg, center=(center_d, center_h, center_w))

    mask = (seg > 0).float()   # binarize BEFORE the soft downsample below
    if getattr(cfg, "downsample_to_liver", False):
        mask = _downsample_to_liver_shape(mask)   # trilinear -> soft edges, matches
                                                    # the model's own F.interpolate later

    _ed_seg_cache[patient] = mask.to(device)
    return _ed_seg_cache[patient]

# =============================================================================
# Diversity / contrastive loss
# =============================================================================
def phase_dispersion_loss(z0: torch.Tensor, phi: torch.Tensor, margin_scale: float = 0.5) -> torch.Tensor:
    """
    Hinge loss that pushes z0 latents apart in proportion to their phase
    distance within the batch. Directly counteracts collapse to a single
    phase-invariant "average" latent -- unlike generic variance maximization,
    the pushed-apart directions are tied to phi, not arbitrary.

    z0:  (B, C, D, H, W) latent (e.g. z0_hat from a single-step Tweedie/v estimate)
    phi: (B,) or (B, 1) scalar phase in [0, 1]
    margin_scale: desired latent L2 distance per unit of phase distance.
                  Log z_dist / phi_dist ratios during training and calibrate --
                  too large fights reconstruction, too small does nothing.
    """
    B = z0.shape[0]
    if B < 2:
        return z0.new_zeros(())
    z   = z0.flatten(1)
    phi = phi.view(-1).float()

    z_dist   = torch.cdist(z.unsqueeze(0), z.unsqueeze(0)).squeeze(0)   # (B, B)
    phi_dist = (phi.unsqueeze(0) - phi.unsqueeze(1)).abs()              # (B, B)

    target = margin_scale * phi_dist
    mask   = ~torch.eye(B, dtype=torch.bool, device=z.device)
    return F.relu(target - z_dist)[mask].mean()

@torch.no_grad()
def _one_step_real_z0_test(
    model,
    ref_volume,
    Iseq,
    dvf_gt,
    phase,
    device,
    tau_value=500,
):
    """
    Fast diffusion sanity test.

    Starts from a REAL z0 = encode_dvf(dvf_gt), adds known forward
    diffusion noise at tau, then performs ONE reverse prediction.

    If the diffusion model has learned the conditional mapping, z0_hat
    should be close to z0.
    """

    model.eval()

    # ---------------------------------------------------------
    # 1. Encode the real DVF -> z0
    # ---------------------------------------------------------
    z0 = model.encode_dvf(dvf_gt)

    # Make sure we are testing a single example
    if z0.shape[0] > 1:
        z0 = z0[:1]

    print("\n========== ONE-STEP REAL-z0 TEST ==========")
    print(f"z0 shape       : {tuple(z0.shape)}")
    print(f"||z0||         : {z0.flatten().norm().item():.6f}")
    print(f"z0 mean        : {z0.mean().item():.6f}")
    print(f"z0 std         : {z0.std().item():.6f}")

    # ---------------------------------------------------------
    # 2. Build the conditioning c for the same sample
    # ---------------------------------------------------------
    f_ref, cond_feats = model.encode_context(
        ref_volume[:1],
        Iseq[:1],
        False,
    )

    c = model._build_c(f_ref, cond_feats[0])

    # ---------------------------------------------------------
    # 3. Choose one intermediate diffusion timestep
    # ---------------------------------------------------------
    tau = torch.full(
        (1,),
        tau_value,
        dtype=torch.long,
        device=device,
    )

    # ---------------------------------------------------------
    # 4. Forward diffuse the REAL z0
    # ---------------------------------------------------------
    z_tau, noise = model.schedule.q_sample(z0, tau)

    print(f"\ntau            : {tau_value}")
    print(f"||z_tau||      : {z_tau.flatten().norm().item():.6f}")
    print(f"z_tau mean     : {z_tau.mean().item():.6f}")
    print(f"z_tau std      : {z_tau.std().item():.6f}")
    print(f"noise std      : {noise.std().item():.6f}")

    # ---------------------------------------------------------
    # 5. Ask model to predict from the noisy REAL latent
    # ---------------------------------------------------------
    v_pred = model.predict_eps(
        z_tau,
        tau,
        c,
    )

    # ---------------------------------------------------------
    # 6. Recover z0 in ONE reverse step
    # ---------------------------------------------------------
    if model.predict_mode == "v":
        z0_hat = model.schedule.predict_z0_from_v(
            z_tau,
            tau,
            v_pred,
        )
    else:
        z0_hat = model.schedule.predict_x0_from_eps(
            z_tau,
            tau,
            v_pred,
        )

    # ---------------------------------------------------------
    # 7. Compare predicted z0 with REAL z0
    # ---------------------------------------------------------
    diff = z0_hat - z0

    mse = diff.pow(2).mean().item()
    rmse = diff.pow(2).mean().sqrt().item()

    z0_norm = z0.flatten().norm().item()
    error_norm = diff.flatten().norm().item()

    relative_error = error_norm / (z0_norm + 1e-8)

    cosine = F.cosine_similarity(
        z0.flatten(1),
        z0_hat.flatten(1),
        dim=1,
    ).item()

    print("\n--- z0 recovery ---")
    print(f"||z0||              : {z0_norm:.6f}")
    print(f"||z0_hat||          : {z0_hat.flatten().norm().item():.6f}")
    print(f"error norm          : {error_norm:.6f}")
    print(f"relative error      : {relative_error:.6f}")
    print(f"MSE                 : {mse:.6f}")
    print(f"RMSE                : {rmse:.6f}")
    print(f"z0 std              : {z0.std().item():.6f}")
    print(f"z0_hat std          : {z0_hat.std().item():.6f}")
    print(f"cosine similarity   : {cosine:.6f}")

    print("============================================\n")

    return {
        "z0": z0,
        "z_tau": z_tau,
        "z0_hat": z0_hat,
        "cosine": cosine,
        "relative_error": relative_error,
        "mse": mse,
        "rmse": rmse,
    }

def amplitude_ratio_loss(dvf_pred: torch.Tensor, dvf_gt: torch.Tensor) -> torch.Tensor:
    """Penalises NavRatio ≠ 1 by matching predicted and GT DVF amplitude.

    Operates on mean voxel-wise L2 norm (per batch item), then MSE of the
    ratio to 1. Ignores spatial distribution — complementary to recon_weight.
    dvf_pred / dvf_gt: (B, 3, D, H, W)
    """
    pred_amp = dvf_pred.norm(dim=1).mean(dim=(1, 2, 3))   # (B,)
    gt_amp   = dvf_gt.norm(dim=1).mean(dim=(1, 2, 3))     # (B,)
    ratio    = pred_amp / gt_amp.clamp(min=1e-6)
    return (ratio - 1.0).pow(2).mean()


def _compute_diversity_loss(model, ref_volume, Iseq, phase, device, dvf_gt=None,
                             n_steps: int = 1, margin_scale: float = 0.5,
                             compute_amp: bool = False):
    """
    Anti-collapse regularizer on the model's own single-step z0_hat estimate
    (from pure noise at high tau).

    FIX: previously this always took the F.mse_loss(z0_hat, z0_gt) branch
    whenever dvf_gt was passed -- which it always was in this training loop --
    so the dispersion branch (-z0_hat.var(dim=0).mean()) never actually ran.
    "diversity_weight" was silently just a second reconstruction loss and
    never fought collapse. This version always applies the phase-targeted
    dispersion hinge; GT-MSE (if available) is kept as an additional anchor,
    not a substitute.

    Returns (div_loss, amp_loss). amp_loss is zero when compute_amp=False.
    """
    with torch.no_grad():
        f_ref, cond_feats = model.encode_context(ref_volume, Iseq, False)
        target_dvf = dvf_gt[0] if isinstance(dvf_gt, (list, tuple)) else dvf_gt
        z0_gt = model.encode_dvf(target_dvf).detach() if target_dvf is not None else None

    B = ref_volume.shape[0]
    c = model._build_c(f_ref, cond_feats[0])

    tau = torch.full((B,), model.schedule.T - 1, dtype=torch.long, device=device)
    z_T = torch.randn(B, *model.latent_shape, device=device)
    v_or_eps = model.predict_eps(z_T, tau, c)
    z0_hat = model.schedule.predict_z0_from_v(z_T, tau, v_or_eps) \
             if model.predict_mode == "v" else model.schedule.predict_x0_from_eps(z_T, tau, v_or_eps)

    loss = z0_hat.new_zeros(())
    if phase is not None:
        phi_for_disp = phase[:, 0] if phase.dim() == 2 else phase
        disp = phase_dispersion_loss(z0_hat, phi_for_disp.to(device), margin_scale=margin_scale)
        loss = loss + disp
    elif z0_gt is None:
        loss = loss - z0_hat.flatten(1).var(dim=0).mean()   # fallback if no phase at all

    if z0_gt is not None:
        loss = loss + F.mse_loss(z0_hat, z0_gt)

    amp_loss = z0_hat.new_zeros(())
    if compute_amp and target_dvf is not None:
        dvf_pred = model.decode_latent(z0_hat.detach(), cond_feats[0])
        amp_loss = amplitude_ratio_loss(dvf_pred, target_dvf)

    return loss, amp_loss

# =============================================================================
# Centroid cosine-similarity regularisation
# =============================================================================
def build_Iseq(ref_volume, input_volume_list, current_volume_list, opt, device,
                include_future=True, vol_files=None, orientation=None):
    """ACDC override: sagittal short-axis or coronal slice, with optional future frames.

    condi_type "1" (sagittal): per-sample D-index via `get_reference_slice_index`
    — wherever THIS patient's saved reference slice (or centroid) actually
    lands in the canonical crop, accounting for resampling + clamping —
    instead of a single fixed `sag_pos` shared across every patient.

    `vol_files`: the batch's virtual-path list (same structure as
    `ACDC_4D_Dataset.__getitem__`'s 4th return value), needed to recover each
    sample's patient id. Required when condi_type == "1"; ignored otherwise.
    `orientation`: {patient: entry} dict from `load_orientation_info`, or
    None to fall back to CANONICAL_SHAPE[0] // 2 for every patient (matches
    the old fixed-sag_pos behaviour).
    """
    condi_type = getattr(opt, "condi_type", "1")
    img_h, img_w = getattr(opt, "condi_img_size", [128, 128])

    H, W = ref_volume.shape[3], ref_volume.shape[4]
    h0 = (H - img_h) // 2
    w0 = (W - img_w) // 2

    if condi_type == "1":
        if vol_files is None:
            raise ValueError("build_Iseq: vol_files is required for condi_type '1' "
                              "(needed to resolve each sample's per-patient reference "
                              "slice D-index via get_reference_slice_index).")

        B = ref_volume.shape[0]
        # vol_files[0] = horizon-0 paths, one per batch element (same convention
        # as _build_priormulti_image_inputs).
        patients = [p.split("/")[-2] for p in vol_files[0]]
        d_idx = torch.tensor(
            [get_reference_slice_index(p, opt.data_dir, orientation) for p in patients],
            device=device, dtype=torch.long,
        )
        b_idx = torch.arange(B, device=device)

        # ref_volume[b_idx, :, d_idx, h0:h0+img_h, w0:w0+img_w] -> (B, 1, img_h, img_w)
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
        cor_pos = getattr(opt, "cor_pos", 64)
        c2_ref = ref_volume[:, :, :, cor_pos, :]
        c2 = []
        for q in range(opt.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1)[:, :, :, cor_pos, :]
            c2.append(torch.cat([c_t.to(device), c2_ref.to(device)], dim=1))
        Iseq = torch.stack(c2, dim=2).to(device)
    return Iseq

# =============================================================================
# Stage 2 — diffusion training
# =============================================================================

def train_diffusion(
    cfg:       argparse.Namespace,
    folds:     tuple,
    fold_idx:  str,
    dir_name:  str,
    vm:        nn.Module,
    stn:       nn.Module,
    criterion,
    device:    torch.device,
) -> None:
    """Train the diffusion model for one fold."""

    # ---- build VAE ---------------------------------------------------------
    vae = build_vae(cfg, device)

    if cfg.vae_dir_name:
        fold_vae_ckpt = vae_checkpoint_path(cfg, cfg.vae_dir_name, fold_idx)
        if os.path.exists(fold_vae_ckpt):
            print(f"[Stage 2] Loading VAE fold-{fold_idx} checkpoint: {fold_vae_ckpt}")
            _load_vae_for_diffusion(vae, fold_vae_ckpt, device)
        else:
            raise FileNotFoundError(
                f"[Stage 2] Expected VAE checkpoint not found: {fold_vae_ckpt}\n"
                f"          Run train_vae first, or check cfg.vae_dir_name."
            )
    elif getattr(cfg, "vae_checkpoint", None):
        print(f"[Stage 2] Loading static VAE checkpoint: {cfg.vae_checkpoint}")
        _load_vae_for_diffusion(vae, cfg.vae_checkpoint, device)
    else:
        print("[Stage 2] Warning: no VAE checkpoint specified — using random weights.")

    # ---- resolve per-fold context encoder paths ----------------------------
    # Store templates and restore after build so subsequent folds substitute correctly.
    _path_templates = {}
    for _attr in ("tmnet_path", "rvnet_path"):
        _val = getattr(cfg, _attr, None)
        if _val:
            _path_templates[_attr] = _val
            setattr(cfg, _attr, _val.replace("{fold_idx}", str(fold_idx)))

    # ---- build diffusion model ---------------------------------------------
    model = build_diffusion(cfg, vae, stn, device, fold_idx=str(fold_idx))

    for _attr, _tmpl in _path_templates.items():
        setattr(cfg, _attr, _tmpl)
    model.freeze_vae()

    # ---- load latent normalisation stats saved by train_VAE ------------------
    # Ensures latents fed to the diffusion process are ~ N(0,1), which is
    # required for the noise schedule to be well-calibrated.
    if hasattr(model, "set_latent_stats"):
        stats_path = vae_checkpoint_path(cfg, cfg.vae_dir_name, fold_idx)
        stats_path = os.path.join(os.path.dirname(stats_path), "latent_stats.pt")
        if os.path.exists(stats_path):
            _stats = torch.load(stats_path, map_location="cpu", weights_only=False)
            model.set_latent_stats(
                _stats["mean"], _stats["std"],
                bin_means = _stats.get("bin_means"),
                bin_stds  = _stats.get("bin_stds"),
                n_bins    = _stats.get("n_bins", 0),
            )
            print(f"[Stage 2] Loaded latent normalisation stats from: {stats_path}")
            
        else:
            print(
                f"[Stage 2] Warning: latent_stats.pt not found at {stats_path}. "
                "Run train_VAE first to generate it. Skipping normalisation."
            )
    
    # print("[Check] VAE encoder weights after load:")
    # _check_weights(model.vae, "vae")

    ema = EMA(model, decay=0.9999)

    if getattr(cfg, "checkpoint", None):
        print(f"[Stage 2] Resuming diffusion from: {cfg.checkpoint}")
        custom_load(model, cfg.checkpoint, device)
        ema_ckpt = cfg.checkpoint.replace("model_best.pth", "model_best_ema.pth")
        if os.path.exists(ema_ckpt):
            print(f"[Stage 2] Resuming EMA from: {ema_ckpt}")
            ema.load(ema_ckpt, device)
        else:
            print(f"[Stage 2] No EMA checkpoint found at {ema_ckpt} — EMA starts fresh.")

    # ---- unfreeze context encoders (fine-tuning phase) ---------------------
    if getattr(cfg, "unfreeze_context", False):
        for attr in ("cond_net", "ref_net"):
            module = getattr(model, attr, None)
            if module is not None:
                module.train()
                for p in module.parameters():
                    p.requires_grad_(True)
                print(f"[Stage 2] {attr} unfrozen for fine-tuning.")

    # ---- directories & logging ---------------------------------------------
    # When resuming from a checkpoint, reuse the original run's directories so
    # the best model is saved in-place and TensorBoard curves are continuous.
    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep,
                                  os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}")
        run_dir = os.path.join(cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}")
    val_vol_dir = os.path.join(log_dir, "validation_vols")
    for d in (log_dir, run_dir, val_vol_dir):
        cond_mkdir(d)

    _write_patient_split(folds, log_dir)
    save_params_txt(cfg, log_dir)

    for src in (
        os.path.abspath(__file__),
        os.path.join(os.path.dirname(__file__), "models", "CLDM", "UNet3D.py"),
    ):
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(log_dir, os.path.basename(src)))

    writer = SummaryWriter(run_dir)
    
    save_patients(log_dir, folds)

    # ---- data loaders ------------------------------------------------------
    train_loader, valid_loader = _make_loaders(cfg, folds)

    # ---- optimiser & schedule ----------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(trainable, lr=float(cfg.lr), weight_decay=1e-4)
    scheduler  = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=int(getattr(cfg, "scheduler_patience", 5)),
        min_lr=float(getattr(cfg, "scheduler_min_lr", 1e-7)),
    )
    early_stopper = EarlyStopping(
        patience=int(getattr(cfg, "early_stopping_patience", 12)),
        verbose=True,
        delta=float(getattr(cfg, "early_stopping_delta", 0.005)),
    )
    # scaler = GradScaler("cuda", init_scale=2**10)  # instead of 2**16

    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = np.inf

    print(f"\n[Stage 2] {cfg.diffusion_model.upper()} — fold {fold_idx}")
    
    for attr in ("cond_net", "iseq_enc"):
        obj = getattr(model, attr, None)
        if obj is not None:
            for name, p in obj.named_parameters():
                if not p.requires_grad:
                    print(f"[WARNING] {attr} param frozen: {name}")

    for attr in ("ref_net", "ref_enc"):
        obj = getattr(model, attr, None)
        if obj is not None:
            for name, p in obj.named_parameters():
                if not p.requires_grad:
                    print(f"[WARNING] {attr} param frozen: {name}")

    # ---- epoch loop --------------------------------------------------------
    recon_weight_end = float(getattr(cfg, "recon_weight_end", cfg.recon_weight))

    model.DEBUG = False

    orientation = load_orientation_info(cfg.reference_slices_path) if getattr(cfg, "reference_slices_path", None) else None

    for epoch in range(restart_epoch, cfg.diffusion_epochs):
        frac = epoch / max(cfg.diffusion_epochs - 1, 1)
        current_recon_weight = cfg.recon_weight + frac * (recon_weight_end - cfg.recon_weight)

        print(f"\n[Diffusion] Epoch {epoch}/{cfg.diffusion_epochs - 1}  recon_weight={current_recon_weight:.4f}")
        t0 = time.time()
        model.train()
        model.vae.eval()   # VAE stays frozen in eval mode throughout Stage 2
        
        #print element that are going to be trained and the ones that are frozen
        # print("Trainable parameters:")
        # for name, param in model.named_parameters():
        #     # if param.requires_grad:
        #     #     print(f"  {name}")
        #     if not param.requires_grad:
        #         print(f"  {name} (frozen)")
                
        # model.vae.to("cpu")
        torch.cuda.empty_cache()
        
        ep_loss = ep_steps = 0
        
        # batch_count = 0
        for ref_volume, input_volume_list, current_volume_list, vol_files in tqdm(train_loader):
            optimizer.zero_grad(set_to_none=True)

            ref_volume   = ref_volume.unsqueeze(1).to(device)
            current_vols = [v.unsqueeze(1).to(device) for v in current_volume_list]
            with torch.no_grad():
                dvf_gt = [vm(ref_volume, v) for v in current_vols]
                
            seg_masks = seg_valid = None
            if getattr(cfg, "use_seg_mask", False):
                patients = [p.split("/")[-2] for p in vol_files[0]]
                masks, valid = [], []
                for p in patients:
                    m = _load_ed_seg(p, cfg, device, orientation)
                    valid.append(m is not None)
                    masks.append(m if m is not None else torch.zeros_like(dvf_gt[0][0, 0]).unsqueeze(0))
                ed_mask_batch = torch.stack(masks, dim=0).to(device)          # (B, 1, D, H, W)
                seg_valid     = torch.tensor(valid, device=device, dtype=torch.bool)

                with torch.no_grad():
                    # print("ed_mask_batch:", ed_mask_batch.shape)
                    ed_mask_batch = ed_mask_batch.unsqueeze(1)   # (8, 32, 64, 64) → (8, 1, 32, 64, 64)
                    seg_masks = [stn(ed_mask_batch, dvf_gt[t]) for t in range(len(dvf_gt))]
                    # print("about to call stn with:", ed_mask_batch.shape)
                    # print(ed_mask_batch.unsqueeze(1).shape)   # expect (8, 1, 32, 64, 64)
                    # print(dvf_gt[0].shape)                     # expect (8, 3, 32, 64, 64)

            Iseq = build_Iseq(
                ref_volume, input_volume_list, current_volume_list,
                cfg, device, include_future=True, vol_files=vol_files, orientation=orientation,
            )

            phase = compute_phase(vol_files, model.horizon, device)

            if phase is not None and global_step % 100 == 0:
                phi_vals = phase[0].tolist()   # first sample in the batch
                phi_str  = "  ".join(f"t{t}→φ={v:.3f}" for t, v in enumerate(phi_vals))
                # print(f"[Step {global_step}] Phase  {phi_str}   (batch φ range [{phase.min():.3f}, {phase.max():.3f}])")
            elif phase is None and global_step % 100 == 0:
                print(f"[Step {global_step}] Phase is None!")

            if model.DEBUG:
                print(f"\n[Batch] ref_volume shape: {ref_volume.shape}, Iseq shape: {Iseq.shape}, dvf_gt[0] shape: {dvf_gt[0].shape if dvf_gt else 'N/A'}")

            # ---- unified forward (returns DiffusionOutput) -----------------
            # with autocast("cuda"):
            out = model(ref_volume, current_vols, Iseq, dvf=dvf_gt, criterion=criterion, phase=phase, seg_masks=seg_masks, seg_valid=seg_valid)

            _div_w = float(getattr(cfg, "diversity_weight", 0.0))
            _amp_w = float(getattr(cfg, "amplitude_weight", 0.0))
            diversity_loss = torch.tensor(0.0, device=device)
            amp_loss       = torch.tensor(0.0, device=device)
            if _div_w > 0.0 or _amp_w > 0.0:
                diversity_loss, amp_loss = _compute_diversity_loss(
                    model, ref_volume, Iseq, phase, device,
                    dvf_gt=dvf_gt,
                    n_steps=int(getattr(cfg, "diversity_n_steps", 1)),
                    margin_scale=float(getattr(cfg, "diversity_margin_scale", 0.5)),
                    compute_amp=(_amp_w > 0.0),
                )

            loss = out.total_loss(
                    w_ddpm       = cfg.ddpm_weight,
                    w_dvf_recon  = current_recon_weight,
                    w_vmorph_vol = float(getattr(cfg, "smooth_weight", 0.0)),
                    w_vol_recon  = current_recon_weight,
                ) + _div_w * diversity_loss + _amp_w * amp_loss

            for name, val in [
                ("ddpm", out.ddpm_loss),
                ("dvf_recon", out.dvf_recon_loss), ("vol_recon", out.vol_recon_loss)
            ]:
                if torch.is_tensor(val) and torch.isnan(val):
                    print(f"[NaN detected] {name} at step {global_step}")

            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # scaler.step(optimizer)
            # scaler.update()
            
            loss.backward()

            # Step 1: Check cond_net gradient flow
            if getattr(cfg, "debug", False) and global_step % 50 == 0 and model.cond_net is not None:
                for name, p in model.cond_net.named_parameters():
                    if p.grad is not None:
                        print(f"[cond_net grad] {name}: grad_norm={p.grad.norm():.4f}")
                    else:
                        print(f"[cond_net grad] {name}: NO GRADIENT")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ema.update()


            ep_loss  += loss.item()
            ep_steps += 1

            _log_diffusion_step(writer, global_step, cfg, out, optimizer, model, loss)
            writer.add_scalar("train/recon_weight", current_recon_weight, global_step)
            if _div_w > 0.0:
                writer.add_scalar("train/diversity_loss", diversity_loss.item(), global_step)
                # if global_step % 100 == 0:
                #     with torch.no_grad():
                #         print(f"[DivCheck] step {global_step}  diversity_loss={diversity_loss.item():.5f}")
            if _amp_w > 0.0:
                writer.add_scalar("train/amplitude_loss", amp_loss.item(), global_step)
                # if global_step % 100 == 0:
                #     with torch.no_grad():
                #         print(f"[AmpCheck] step {global_step}  amplitude_loss={amp_loss.item():.5f}")

            global_step += 1

        writer.add_scalar("train/epoch_loss", ep_loss / max(ep_steps, 1), epoch)

        # ---- validation ----------------------------------------------------
        torch.cuda.empty_cache()
        _compute_nav = (epoch % 2 == 0)
        ema.apply_shadow()
        val_loss_vol, val_loss_dvf, nav_dict = _validate_diffusion(model, valid_loader, cfg, criterion, device, vm, recon_weight=current_recon_weight, orientation=orientation, compute_nav=_compute_nav)
        ema.restore()
        val_loss = val_loss_vol + val_loss_dvf
        writer.add_scalar("val/loss_vol", val_loss_vol, epoch)
        writer.add_scalar("val/loss_dvf", val_loss_dvf, epoch)
        writer.add_scalar("val/loss",     val_loss,     epoch)
        if nav_dict is not None:
            writer.add_scalar("val/nav_ratio",    nav_dict["NavRatio"],   epoch)
            writer.add_scalar("val/nav_corr",     nav_dict["NavCorr"],    epoch)
            writer.add_scalar("val/nav_pred_std", nav_dict["NavPredStd"], epoch)
            writer.add_scalar("val/nav_pred_cv",  nav_dict["NavPredCV"],  epoch)

        if val_loss < best_val_loss:
            print(f"[Diffusion] ↓ {best_val_loss:.4f} → {val_loss:.4f} — saving.")
            best_val_loss = val_loss
            custom_save(model, os.path.join(log_dir, "model_best.pth"))
            ema.save(os.path.join(log_dir, "model_best_ema.pth"))
            if epoch >= 10:
                _save_val_vols(model, valid_loader, val_vol_dir, epoch, device, cfg, orientation=orientation)
        else:
            print(f"[Diffusion] No improvement from {best_val_loss:.4f}")

        scheduler.step(val_loss)
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("[Diffusion] Early stopping triggered.")
            break

        print(f"[Diffusion] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nStage 2 done.  Best checkpoint: {log_dir}/model_best.pth")
    writer.close()

def _validate_diffusion(
    model:        nn.Module,
    valid_loader: DataLoader,
    cfg:          argparse.Namespace,
    criterion,
    device:       torch.device,
    vm:           nn.Module = None,
    recon_weight: float = 1.0,
    orientation: dict | None = None,
    compute_nav:  bool = False,
) -> tuple:
    """
    Stage-2 validation — MSE(generated_vol, current_vol) + smooth_weight * gradient_loss(generated_dvf).
    Uses inference forward (dvf=None).
    Returns (val_loss_vol, val_loss_dvf, nav_dict | None).
    nav_dict is only computed when compute_nav=True and contains NavRatio, NavCorr,
    NavPredStd, NavPredCV.
    """
    model.eval()
    smooth_weight = float(getattr(cfg, "smooth_weight", 0.0))
    mse_fn        = nn.MSELoss(reduction="mean").to(device)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    val_loss_vol = val_loss_dvf = n = 0
    nav_gt_all, nav_pred_all = [], []

    with torch.no_grad():
        for ref_volume, input_volume_list, current_volume_list, vol_files in tqdm(valid_loader):
            ref_volume   = ref_volume.unsqueeze(1).to(device)
            current_vols = [v.unsqueeze(1).to(device) for v in current_volume_list]

            Iseq = build_Iseq(
                ref_volume, input_volume_list, None,
                cfg, device, include_future=False, vol_files=vol_files, orientation=orientation,
            )

            phase = compute_phase(vol_files, model.horizon, device)
            out = model(ref_volume, None, Iseq, dvf=None, ddim_steps=getattr(cfg, "ddim_steps", 50), phase=phase)

            n_preds = len(out.generated_vols)
            if n_preds == 0:
                continue

            batch_mse = batch_smooth = 0.0
            for tt in range(min(n_preds, cfg.tp)):
                vol_t = out.generated_vols[tt]
                if torch.isnan(vol_t).any() or torch.isinf(vol_t).any():
                    continue
                batch_mse += mse_fn(vol_t, current_vols[tt]).item()

                if smooth_weight > 0 and tt < len(out.generated_dvf):
                    dvf_t = out.generated_dvf[tt]
                    if not (torch.isnan(dvf_t).any() or torch.isinf(dvf_t).any()):
                        batch_smooth += gradient_loss(dvf_t).item()

                if compute_nav:
                    B = vol_t.shape[0]
                    for b in range(B):
                        _nc_gt,   _, _ = navigator_signal_3planes(current_vols[tt][b:b+1], ref_volume[b:b+1])
                        _nc_pred, _, _ = navigator_signal_3planes(vol_t[b:b+1],            ref_volume[b:b+1])
                        if not np.isnan(float(_nc_gt)) and not np.isnan(float(_nc_pred)):
                            nav_gt_all.append(float(_nc_gt))
                            nav_pred_all.append(float(_nc_pred))

            tp = max(min(n_preds, cfg.tp), 1)
            batch_loss_vol = batch_mse                      / tp
            batch_loss_dvf = smooth_weight * batch_smooth  / tp

            val_loss_vol += batch_loss_vol
            val_loss_dvf += batch_loss_dvf
            n            += 1

    nav_dict = None
    if compute_nav and len(nav_pred_all) > 1:
        nav_m        = navigator_metrics(nav_gt_all, nav_pred_all)
        nav_pred_arr = np.array(nav_pred_all, dtype=float)
        nav_pred_std = float(np.std(nav_pred_arr))
        nav_pred_cv  = (nav_pred_std / float(np.mean(nav_pred_arr))
                        if float(np.mean(nav_pred_arr)) > 1e-8 else float("nan"))
        nav_dict = {
            "NavRatio":   nav_m["amplitude_ratio"],
            "NavCorr":    nav_m["phase_corr"],
            "NavPredStd": nav_pred_std,
            "NavPredCV":  nav_pred_cv,
        }
        print(
            f"[Val Nav] NavRatio: {nav_dict['NavRatio']:.3f}  "
            f"NavCorr: {nav_dict['NavCorr']:.3f}  "
            f"NavPredStd: {nav_dict['NavPredStd']:.5f}  "
            f"NavPredCV: {nav_dict['NavPredCV']:.3f}"
        )

    return val_loss_vol / max(n, 1), val_loss_dvf / max(n, 1), nav_dict


def _save_val_vols(model, valid_loader, val_vol_dir, epoch, device, cfg, n_save=3, orientation=None):
    """Save a few generated / GT volume pairs, replacing previous saves."""
    import nibabel as nib

    model.eval()
    batch = next(iter(valid_loader))
    ref_volume, input_volume_list, current_volume_list, vol_files = batch

    ref_volume   = ref_volume.unsqueeze(1).to(device)
    current_vols = [v.unsqueeze(1).to(device) for v in current_volume_list]
    Iseq  = build_Iseq(ref_volume, input_volume_list, None, cfg, device, include_future=False, vol_files=vol_files, orientation=orientation)
    phase = compute_phase(vol_files, model.horizon, device)

    with torch.no_grad():
        out = model(ref_volume, None, Iseq, dvf=None, ddim_steps=getattr(cfg, "ddim_steps", 50), phase=phase)

    # Clear and recreate so old files are replaced
    if os.path.isdir(val_vol_dir):
        shutil.rmtree(val_vol_dir)
    os.makedirs(val_vol_dir)

    n = min(n_save, ref_volume.shape[0])
    for i in range(n):
        def _save(arr, name):
            nib.save(
                nib.Nifti1Image(arr.detach().cpu().numpy(), np.eye(4)),
                os.path.join(val_vol_dir, f"s{i:02d}_{name}_ep{epoch:03d}.nii.gz"),
            )

        _save(ref_volume[i, 0], "ref")
        if out.generated_vols:
            _save(out.generated_vols[0][i, 0], "gen")
        if current_vols:
            _save(current_vols[0][i, 0], "gt")

    print(f"[val_vols] Saved {n} samples to {val_vol_dir}  (epoch {epoch})")


# # =============================================================================
# # Test
# # =============================================================================

def _print_dvf_spatial(dvf: torch.Tensor, label: str, top_k: int = 3) -> None:
    """Print spatial distribution of DVF magnitude. dvf: [3, D, H, W]."""
    mag = dvf.norm(dim=0)          # [D, H, W]
    D, H, W = mag.shape

    # overall stats
    print(f"    [{label}] mean={mag.mean():.3f}  std={mag.std():.3f}  max={mag.max():.3f}")

    # peak voxel
    flat = mag.argmax().item()
    pz, py, px = flat // (H * W), (flat % (H * W)) // W, flat % W
    print(f"    [{label}] peak voxel  z={pz:3d}  y={py:3d}  x={px:3d}  mag={mag.max():.3f}")

    # top-k axial slices by mean magnitude
    slice_means = mag.mean(dim=(1, 2))           # [D]
    top = slice_means.topk(min(top_k, D))
    parts = [f"z={idx.item()}({val.item():.3f})"
             for val, idx in zip(top.values, top.indices)]
    print(f"    [{label}] top-{top_k} slices: {' | '.join(parts)}")


# ── DVF diversity diagnostic helpers — imported from utils/dvf_metrics.py ─────
# Private aliases kept for backward compat with call sites below.
_motion_amplitude       = motion_amplitude
_hf_energy_ratio        = hf_energy_ratio
_jacobian_det_stats     = jacobian_det_stats
_jacobian_folding_ratio = jacobian_folding_ratio
_dvf_cosine_sim_stats   = dvf_cosine_sim_stats
_dvf_diversity_diagnostic = dvf_diversity_diagnostic


# ── Voxel spacings (D, H, W) after 0.5× downsampling in H and W ─────────────
_SP_D  = 3.5       # mm / voxel — depth axis (unchanged)
_SP_HW = 1.70 * 2  # mm / voxel — height and width axes


def _load_landmarks(data_dir: str, patient_id: str) -> dict:
    """ACDC stub — no equivalent landmark file format yet."""
    return {}


def _landmark_tracking_error(
    dvf_pred:   torch.Tensor,
    landmarks:  dict,
    volume_idx: int,
    patient_id: str,
) -> float:
    """ACDC stub — no landmark format defined yet."""
    return np.nan


from mopred.utils.landmarks import landmark_dvf_error as _landmark_dvf_error


 
def test(
    cfg:      argparse.Namespace,
    fold:     list,
    fold_idx: str,
    dir_name: str,
    vm:       nn.Module,
    stn:      nn.Module,
    device:   torch.device,
) -> None:
    """Run inference and compute metrics for one test fold."""
    aff = [[10.0, 0, 0, 0], [0, 1.25, 0, 0], [0, 0, 1.25, 0], [0, 0, 0, 1]]  # ACDC target spacing
 
    # ---- build & load model ------------------------------------------------
    vae   = build_vae(cfg, device)
     
    _path_templates = {}
    for _attr in ("tmnet_path", "rvnet_path"):
        _val = getattr(cfg, _attr, None)
        if _val and "{fold_idx}" in _val:
            _path_templates[_attr] = _val
            setattr(cfg, _attr, _val.replace("{fold_idx}", str(fold_idx)))
 
    model = build_diffusion(cfg, vae, stn, device, fold_idx=str(fold_idx))
 
    for _attr, _tmpl in _path_templates.items():
        setattr(cfg, _attr, _tmpl)
 
    model.freeze_vae()
 
    ema_path = os.path.join(cfg.checkpoint, f"fold_{fold_idx}", "model_best_ema.pth")
    raw_path = os.path.join(cfg.checkpoint, f"fold_{fold_idx}", "model_best.pth")
 
    if os.path.exists(ema_path):
        print(f"[Test] Loading EMA weights: {ema_path}")
        ema = EMA(model)
        ema.load(ema_path, device)
    elif os.path.exists(raw_path):
        print(f"[Test] EMA not found — loading raw weights: {raw_path}")
        custom_load(model, raw_path, device)
    else:
        raise FileNotFoundError(
            f"[Test] No checkpoint found at:\n  {ema_path}\n  {raw_path}"
        )
 
    model.eval()
    vm.eval()
 
    model.DEBUG = False
    model.set_film_debug(False)
 
    # ---- data & output directories -----------------------------------------
    ds_kwargs = _acdc_dataset_kwargs(cfg)
    # ---- orientation info (per-patient reference slice / centroid / rotation) ---
    orientation = load_orientation_info(cfg.reference_slices_path) \
        if getattr(cfg, "reference_slices_path", None) else None
    print(orientation)
    
    test_set = ACDC_4D_Dataset(
        cfg.data_dir, sequence_list=fold, nb_pred=cfg.tp,
        nb_inputs=cfg.nb_inputs, test=True,
        **ds_kwargs
    )
    
    test_set.repeats = 1
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
    )
 
    test_subdir = f"test_{cfg.test_tag}" if getattr(cfg, "test_tag", None) else "test"
    save_dir  = os.path.join(dir_name, test_subdir, fold_idx)
    vol_dir   = os.path.join(save_dir, "volumes")
    fig_dir   = os.path.join(save_dir, "figures")
    track_dir = os.path.join(save_dir, "tracking")
    for d in (vol_dir, fig_dir, track_dir):
        cond_mkdir(d)
 
    test_patients = {
        seq.split("/")[0] if "/" in seq else seq for seq in fold
    }
    with open(os.path.join(save_dir, "patients_test.txt"), "w") as f:
        for p in sorted(test_patients):
            f.write(p + "\n")
 
    # ---- preload GT landmarks (one dict per patient) -----------------------
    all_landmarks: dict = {
        pid: _load_landmarks(cfg.data_dir, pid)
        for pid in test_patients
    }
    n_with_lm = sum(1 for v in all_landmarks.values() if v)
    print(f"[Landmark] GT landmarks found for {n_with_lm}/{len(test_patients)} test patients.")
 
    # ---- inference loop ----------------------------------------------------
    MSE_loss, NCC_loss, SSIM_loss, geo_errors, landmark_error, landmark_error_gt, landmark_dvf_error = [], [], [], [], [], [], []
    MSE_loss_gt       = []   # [n_samples, tp] — MSE vs true GT frame (not VoxelMorph pseudo-GT)
    nav_signals_gt, nav_signals_pred   = [], []
    nav_sagittal_gt, nav_sagittal_pred = [], []
    nav_axial_gt,    nav_axial_pred    = [], []
    cycle_phases      = []   # [n_samples, tp] — amplitude-based ∈ [0, 1]
    cycle_phases_time = []   # [n_samples, tp] — time-index-based ∈ [0, 1]
    PSNR_loss         = []   # [n_samples, tp]
    folding_ratio_gen = []   # [n_samples, tp] — fraction of voxels with det(J) ≤ 0
    temporal_smoothness = [] # [n_samples] — mean frame-to-frame DVF L2 diff
    vm_error, dvf_vae_error, unet3d_error, dvf_vae_mse = [], [], [], []
    mse_fn = nn.MSELoss(reduction="mean").to(device)
 
    # <-- added: per-sample, per-timepoint SI/AP amplitude accumulators
    # (shaped like geo_error / MSE_loss: one row per idx, `tp` values per row)
    # + one patient id per row, for amp_vs_phase_or_time_plot.py.
    si_amp_gt, si_amp_pred = [], []
    ap_amp_gt, ap_amp_pred = [], []
    lr_amp_gt, lr_amp_pred = [], []
    patient_ids_all: list = []
    jac_det_mean_gt,  jac_det_mean_pred  = [], []
    jac_det_std_gt,   jac_det_std_pred   = [], []
    vm_ssim_all,     dvf_vae_ssim_all = [], []
    vm_ncc_all,      dvf_vae_ncc_all  = [], []
 
    # Accumulation for DVF diversity diagnostic
    _diag_pids:     list = []
    _diag_gt_vecs:  list = []
    _diag_gen_vecs: list = []
    _diag_amp_gt,   _diag_amp_gen   = [], []
    _diag_hf_gt,    _diag_hf_gen    = [], []
    _diag_jstd_gt,  _diag_jstd_gen  = [], []
    _diag_jmean_gt, _diag_jmean_gen = [], []
 
    # Accumulators for z0/VAE phase diagnostic (model.DEBUG only).
    _dbg_phi, _dbg_z0_norm, _dbg_vae_mse, _dbg_dvf_mag = [], [], [], []

    # Accumulator for context-swap diagnostic (model.DEBUG only).
    # Each entry: (ref_vol_cpu, f_ref_cpu, h_t_cpu, phase_cpu, gt_vol_cpu)
    _swap_buf = []
 
    current_patient = None

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    with torch.no_grad():
        for idx, (ref_volume, input_volume_list, current_volume_list, vol_file) in enumerate(
            tqdm(test_loader)
        ):
            patient_no = vol_file[0][0].split("/")[-2]
            print(f"\nInference — patient: {patient_no}")
            
            _seed = getattr(cfg, "test_seed", 42)
            torch.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)
 
            ref_volume   = ref_volume.unsqueeze(1).to(device)
            vmorph_volume, dvf_list = [], []
 
            for vol in range(len(current_volume_list)):
                current_volume_list[vol] = current_volume_list[vol].unsqueeze(1).to(device)
                dvf_vm = vm(ref_volume, current_volume_list[vol])
                dvf_list.append(dvf_vm)
                vmorph_volume.append(stn(ref_volume, dvf_vm))
 
            Iseq = build_Iseq(
                ref_volume, input_volume_list, None,
                cfg, device, include_future=False, vol_files=vol_file, orientation=orientation,
            )
 
            phase = compute_phase(vol_file, model.horizon, device)
            
            with torch.no_grad():
                f_ref, cond_feats = model.encode_context(ref_volume, Iseq, False)
                h_t = cond_feats[0]
                phi_val = phase[0, 0].item() if phase is not None else float("nan")
                print(f"[CondCheck] idx={idx}  phi={phi_val:.3f}  "
                    f"h_t norm={h_t.norm(dim=-1).item():.4f}  h_t std={h_t.std().item():.4f}")
                if idx > 0:
                    print(f"[CondCheck] cos_sim(h_t[idx], h_t[idx-1]) = {F.cosine_similarity(h_t, _prev_h_t).item():.4f}")
                _prev_h_t = h_t

            # Accumulate for context-swap diagnostic (DEBUG only).
            if model.DEBUG:
                _swap_buf.append((
                    ref_volume.cpu(),
                    f_ref.cpu(),
                    h_t.cpu(),
                    phase.cpu() if phase is not None else None,
                    current_volume_list[0].cpu() if current_volume_list else None,
                ))

            if model.DEBUG:
                # ── z0 magnitude + VAE reconstruction diagnostic ──────────────
                # Two questions answered together:
                #   1. VAE culprit check: is vae_mse flat across phi?
                #      If yes → VAE is not the problem.
                #   2. Noise-schedule check: does ||z0|| grow with phi?
                #      If yes → high-motion latents are out-of-distribution for
                #      the noise schedule → DDIM errors at those phases.
                for tp in range(len(dvf_list)):
                    dvf_gt_tp = dvf_list[tp]
                    z0_true   = model.encode_dvf(dvf_gt_tp)
                    dvf_recon = model.decode_latent(z0_true)
                    vae_mse   = F.mse_loss(dvf_recon, dvf_gt_tp).item()
                    dvf_mag   = dvf_gt_tp.norm(dim=1).mean().item()
                    z0_norm   = z0_true.norm().item()
                    phi_val   = phase[0, tp].item() if (phase is not None and phase.dim() == 2) else (
                                phase[0].item() if phase is not None else float("nan"))
                    print(
                        f"[DEBUG z0] tp={tp}  phi={phi_val:.3f}"
                        f"  dvf_mag={dvf_mag:.4f}"
                        f"  ||z0||={z0_norm:.3f}"
                        f"  z0_std={z0_true.std().item():.4f}"
                        f"  vae_mse={vae_mse:.5f}"
                    )
                    _dbg_phi.append(phi_val)
                    _dbg_z0_norm.append(z0_norm)
                    _dbg_vae_mse.append(vae_mse)
                    _dbg_dvf_mag.append(dvf_mag)
                print("out_conv weight norm:", model.denoising_net.out_conv.weight.norm().item())
                print("out_conv bias norm:  ", model.denoising_net.out_conv.bias.norm().item())
                for name, p in model.denoising_net.named_parameters():
                    if "conv1.weight" in name or "conv2.weight" in name:
                        print(name, p.norm().item())
                        break  # just grab one or two
 
            # if model.DEBUG:
            #     for tp in range(len(dvf_list)):
            #         dvf_gt_tp = dvf_list[tp]
            #         with torch.no_grad():
            #             z_gt     = model.encode_dvf(dvf_gt_tp)
            #             dvf_recon = model.decode_latent(z_gt)
            #         gt_peak_z    = dvf_gt_tp.norm(dim=1)[0].mean(dim=(1, 2)).argmax().item()
            #         recon_peak_z = dvf_recon.norm(dim=1)[0].mean(dim=(1, 2)).argmax().item()
            #         recon_err    = (dvf_recon - dvf_gt_tp).abs().mean().item()
            #         print(f"[DEBUG VAE] tp={tp} | "
            #               f"GT peak_z={gt_peak_z}  recon peak_z={recon_peak_z}  "
            #               f"recon_err={recon_err:.4f}  "
            #               f"GT mag={dvf_gt_tp.norm(dim=1).mean():.4f}  "
            #               f"recon mag={dvf_recon.norm(dim=1).mean():.4f}")
 
            # out = model(ref_volume, None, Iseq, dvf=None, phase=phase)
            out = model(ref_volume, None, Iseq, dvf=None, ddim_steps=getattr(cfg, "ddim_steps", 50), phase=phase)
            generated_dvf  = out.generated_dvf
            generated_vols = out.generated_vols
 
            # ---- tracking volumes ------------------------------------------
            for tp in range(min(len(generated_vols), cfg.tp)):
                v_file     = vol_file[tp][0]
                volume_idx = int(v_file.split("/")[-1][2:-7])
                sequence   = v_file.split("/")[-2]
                if True:  # save all ACDC timepoints
                    save_tensor_as_nifti(
                        generated_vols[tp][0, 0],
                        sequence + f"_t{tp}", track_dir, iter=volume_idx,
                    )
                    save_tensor_as_nifti(
                        vmorph_volume[tp][0, 0],
                        sequence + f"_t{tp}_GT", track_dir, iter=volume_idx,
                    )
 
            # ---- per-timepoint metrics -------------------------------------
            this_mse, this_ncc, this_ssim, this_geo, this_lm, this_lm_gt, this_lm_dvf = [], [], [], [], [], [], []
            this_mse_gt = []   # MSE vs true GT frame
            this_nav_gt, this_nav_pred           = [], []
            this_nav_sag_gt, this_nav_sag_pred   = [], []
            this_nav_ax_gt,  this_nav_ax_pred    = [], []
            this_cycle = []
            this_psnr  = []
            this_fold  = []
            _dvf_seq_gen = []  # predicted DVFs per tp for temporal smoothness
            this_vm, this_dvfvae, this_unet3d, this_dvfvae_mse = [], [], [], []
            # <-- added: per-tp SI/AP amplitude for this sample
            this_si_amp_gt, this_si_amp_pred = [], []
            this_ap_amp_gt, this_ap_amp_pred = [], []
            this_lr_amp_gt, this_lr_amp_pred = [], []
            this_jac_mean_gt,  this_jac_mean_pred  = [], []
            this_jac_std_gt,   this_jac_std_pred   = [], []
            this_vm_ncc,    this_dvfvae_ncc  = [], []
            this_vm_ssim,   this_dvfvae_ssim = [], []
 
            for tp in range(cfg.tp):
                save_tensor_as_nifti(
                    vmorph_volume[tp][0, 0],
                    f"vm_volume_t{tp}", vol_dir, iter=idx,
                )
 
                if tp < len(generated_vols) and generated_vols[tp] is not None:
                    save_tensor_as_nifti(
                        generated_vols[tp][0, 0],
                        f"generated_volume_t{tp}", vol_dir, iter=idx, aff=aff,
                    )
                    _img_mse = mse_fn(generated_vols[tp], vmorph_volume[tp])
                    this_mse.append(np.ravel(_img_mse.item()))
                    this_mse_gt.append(np.ravel(mse_fn(generated_vols[tp], current_volume_list[tp]).item()))
                    this_ncc.append(np.ravel(
                        ncc_loss(generated_vols[tp], vmorph_volume[tp],
                                 device=device).item()
                    ))
                    gen = generated_vols[tp][0, 0].cpu().numpy()
                    gt  = vmorph_volume[tp][0, 0].cpu().numpy()
                    this_ssim.append(np.ravel(
                        ss(gen, gt, data_range=gt.max() - gt.min())
                    ))
                    this_psnr.append(np.ravel(
                        (10.0 * torch.log10(1.0 / (_img_mse + 1e-8))).item()
                    ))
                    # navigator signal: 3-plane intensity diff vs reference (ED)
                    _nc_gt,  _ns_gt,  _na_gt   = navigator_signal_3planes(current_volume_list[tp], ref_volume)
                    _nc_pred, _ns_pred, _na_pred = navigator_signal_3planes(generated_vols[tp],    ref_volume)
                    this_nav_gt.append(_nc_gt);   this_nav_pred.append(_nc_pred)
                    this_nav_sag_gt.append(_ns_gt);  this_nav_sag_pred.append(_ns_pred)
                    this_nav_ax_gt.append(_na_gt);   this_nav_ax_pred.append(_na_pred)
                else:
                    # Model did not generate this time-point (CausalDiT single-step).
                    this_mse.append(np.ravel(np.nan))
                    this_mse_gt.append(np.ravel(np.nan))
                    this_ncc.append(np.ravel(np.nan))
                    this_ssim.append(np.ravel(np.nan))
                    this_psnr.append(np.ravel(np.nan))
                    this_nav_gt.append(np.nan);      this_nav_pred.append(np.nan)
                    this_nav_sag_gt.append(np.nan);  this_nav_sag_pred.append(np.nan)
                    this_nav_ax_gt.append(np.nan);   this_nav_ax_pred.append(np.nan)
 
                save_tensor_as_nifti(
                    dvf_list[tp][0], f"DVF_t{tp}", vol_dir, iter=idx,
                )
 
                if tp < len(generated_dvf) and generated_dvf[tp] is not None:
                    save_tensor_as_nifti(
                        generated_dvf[tp][0],
                        f"generated_DVF_t{tp}", vol_dir, iter=idx,
                    )
                    dvf_gt_t   = dvf_list[tp][0]        # [3, D, H, W]
                    dvf_pred_t = generated_dvf[tp][0]   # [3, D, H, W]
 
                    if model.DEBUG:
                        print(f"  DVF spatial analysis — tp={tp}:")
                        _print_dvf_spatial(dvf_gt_t,   "GT  ")
                        _print_dvf_spatial(dvf_pred_t, "PRED")
 
                    # # At the GT peak voxel, compare GT vs predicted magnitude
                    # if model.DEBUG:
                    #     gt_mag   = dvf_gt_t.norm(dim=0)
                    #     pred_mag = dvf_pred_t.norm(dim=0)
                    #     flat     = gt_mag.argmax().item()
                    #     D, H, W  = gt_mag.shape
                    #     pz, py, px = flat // (H * W), (flat % (H * W)) // W, flat % W
                    #     print(f"    At GT peak (z={pz},y={py},x={px}): "
                    #         f"GT={gt_mag[pz,py,px]:.3f}  PRED={pred_mag[pz,py,px]:.3f}"
                    #         f"  ratio={pred_mag[pz,py,px]/(gt_mag[pz,py,px]+1e-6):.2f}")
 
                    np_gt   = dvf_gt_t.cpu().numpy()
                    np_pred = dvf_pred_t.cpu().numpy()
                    err = geo_error(np_gt, np_pred)
                    this_geo.append(np.ravel(err))

                    # <-- added: SI (axis 0 / D) and AP (axis 1 / H) mean abs
                    # displacement for this timepoint, GT vs predicted.
                    def _axis_mean_abs(dvf: "torch.Tensor", axis: int) -> float:
                        """Mean absolute displacement along one DVF channel (0=D/SI, 1=H/AP)."""
                        return dvf[0, axis].abs().mean().item()

                    this_si_amp_gt.append(_axis_mean_abs(dvf_list[tp],       axis=0))
                    this_si_amp_pred.append(_axis_mean_abs(generated_dvf[tp], axis=0))
                    this_ap_amp_gt.append(_axis_mean_abs(dvf_list[tp],       axis=1))
                    this_ap_amp_pred.append(_axis_mean_abs(generated_dvf[tp], axis=1))
                    this_lr_amp_gt.append(_axis_mean_abs(dvf_list[tp],       axis=2))
                    this_lr_amp_pred.append(_axis_mean_abs(generated_dvf[tp], axis=2))
                    
                    _ct_np = current_volume_list[tp][0, 0].cpu().numpy()
                    this_vm.append(mse_fn(vmorph_volume[tp], current_volume_list[tp]).item())
                    this_vm_ncc.append(ncc_loss(vmorph_volume[tp], current_volume_list[tp], device=device).item())
                    _vm_np = vmorph_volume[tp][0, 0].cpu().numpy()
                    this_vm_ssim.append(ss(_vm_np, _ct_np, data_range=_ct_np.max() - _ct_np.min()))

                    # ---- decomposed error sources (VoxelMorph / DVF-VAE / UNet3D) ----
                    this_vm.append(mse_fn(vmorph_volume[tp], current_volume_list[tp]).item())
                    if hasattr(model, "encode_dvf") and hasattr(model, "decode_latent"):
                        _dvf_vae_r = model.decode_latent(model.encode_dvf(dvf_list[tp]))
                        _np_vae    = _dvf_vae_r[0].cpu().numpy()
                        this_dvfvae.append(float(np.mean(geo_error(np_gt, _np_vae))))
                        this_unet3d.append(float(np.mean(geo_error(_np_vae, np_pred))))
                        _vmorph_vae = stn(ref_volume, _dvf_vae_r)
                        this_dvfvae_mse.append(mse_fn(_vmorph_vae, current_volume_list[tp]).item())
                        this_dvfvae_ncc.append(ncc_loss(_vmorph_vae, current_volume_list[tp], device=device).item())
                        _vae_np = _vmorph_vae[0, 0].cpu().numpy()
                        this_dvfvae_ssim.append(ss(_vae_np, _ct_np, data_range=_ct_np.max() - _ct_np.min()))
                    else:
                        this_dvfvae.append(np.nan)
                        this_unet3d.append(np.nan)
                        this_dvfvae_mse.append(np.nan)
                        this_dvfvae_ncc.append(np.nan)
                        this_dvfvae_ssim.append(np.nan)
 
                    # ---- landmark tracking error (pred + GT) -------------------
                    v_file_tp     = vol_file[tp][0]
                    volume_idx_tp = int(v_file_tp.split("/")[-1][2:-7])
                    lm_kw = dict(
                        landmarks  = all_landmarks.get(patient_no, {}),
                        volume_idx = volume_idx_tp,
                        patient_id = patient_no,
                    )
                    this_lm.append(_landmark_tracking_error(generated_dvf[tp], **lm_kw))
                    this_lm_gt.append(_landmark_tracking_error(dvf_list[tp],   **lm_kw))
                    this_lm_dvf.append(_landmark_dvf_error(generated_dvf[tp], dvf_list[tp], **lm_kw))
 
                    try:
                        this_cycle.append(get_cardiac_phase(v_file_tp.split("/")[-2], int(v_file_tp.split("/")[-1][2:-7]), cfg.data_dir))
                    except Exception:
                        this_cycle.append(float("nan"))
 
                    # Diversity diagnostic accumulation
                    _diag_pids.append(patient_no)
                    _diag_gt_vecs.append(dvf_list[tp].flatten(1).cpu())
                    _diag_gen_vecs.append(generated_dvf[tp].flatten(1).cpu())
                    _diag_amp_gt.append(_motion_amplitude(dvf_list[tp]))
                    _diag_amp_gen.append(_motion_amplitude(generated_dvf[tp]))
                    _diag_hf_gt.append(_hf_energy_ratio(dvf_list[tp]))
                    _diag_hf_gen.append(_hf_energy_ratio(generated_dvf[tp]))
                    _jmgt,  _jstgt  = _jacobian_det_stats(dvf_list[tp])
                    _jmgen, _jstgen = _jacobian_det_stats(generated_dvf[tp])
                    _diag_jmean_gt.append(_jmgt);   _diag_jstd_gt.append(_jstgt)
                    _diag_jmean_gen.append(_jmgen); _diag_jstd_gen.append(_jstgen)
                    this_jac_mean_gt.append(_jmgt);   this_jac_mean_pred.append(_jmgen)
                    this_jac_std_gt.append(_jstgt);   this_jac_std_pred.append(_jstgen)
                    this_fold.append(_jacobian_folding_ratio(generated_dvf[tp]))
                    _dvf_seq_gen.append(generated_dvf[tp].detach())
                else:
                    this_geo.append(np.ravel(np.nan))
                    this_vm.append(np.nan)
                    this_vm_ncc.append(np.nan);    this_vm_ssim.append(np.nan)
                    this_dvfvae.append(np.nan)
                    this_unet3d.append(np.nan)
                    this_dvfvae_mse.append(np.nan)
                    this_dvfvae_ncc.append(np.nan); this_dvfvae_ssim.append(np.nan)
                    this_lm.append(np.nan)
                    this_lm_gt.append(np.nan)
                    this_lm_dvf.append(np.nan)
                    this_fold.append(np.nan)
                    # <-- added: keep SI/AP amp rows aligned even when this
                    # timepoint has no generated DVF.
                    this_si_amp_gt.append(np.nan)
                    this_si_amp_pred.append(np.nan)
                    this_ap_amp_gt.append(np.nan)
                    this_ap_amp_pred.append(np.nan)
                    this_lr_amp_gt.append(np.nan)
                    this_lr_amp_pred.append(np.nan)
                    this_jac_mean_gt.append(np.nan);   this_jac_mean_pred.append(np.nan)
                    this_jac_std_gt.append(np.nan);    this_jac_std_pred.append(np.nan)
                    try:
                        _vf = vol_file[tp][0] if isinstance(vol_file[tp], (list, tuple)) else vol_file[tp]
                        this_cycle.append(get_cardiac_phase(_vf.split("/")[-2], int(_vf.split("/")[-1][2:-7]), cfg.data_dir))
                    except Exception:
                        this_cycle.append(float("nan"))
 
            # temporal smoothness: mean frame-to-frame DVF L2 diff (lower = smoother)
            if len(_dvf_seq_gen) >= 2:
                _diffs = [
                    (_dvf_seq_gen[i + 1] - _dvf_seq_gen[i]).norm(dim=1).mean().item()
                    for i in range(len(_dvf_seq_gen) - 1)
                ]
                _this_smooth = float(np.mean(_diffs))
            else:
                _this_smooth = np.nan
 
            MSE_loss.append(this_mse)
            MSE_loss_gt.append(this_mse_gt)
            NCC_loss.append(this_ncc)
            SSIM_loss.append(this_ssim)
            geo_errors.append(this_geo)
            landmark_error.append(this_lm)
            landmark_error_gt.append(this_lm_gt)
            landmark_dvf_error.append(this_lm_dvf)
            nav_signals_gt.append(this_nav_gt);      nav_signals_pred.append(this_nav_pred)
            nav_sagittal_gt.append(this_nav_sag_gt); nav_sagittal_pred.append(this_nav_sag_pred)
            nav_axial_gt.append(this_nav_ax_gt);     nav_axial_pred.append(this_nav_ax_pred)
            cycle_phases.append(this_cycle)
            PSNR_loss.append(this_psnr)
            folding_ratio_gen.append(this_fold)
            temporal_smoothness.append(_this_smooth)
            vm_error.append(this_vm)
            dvf_vae_error.append(this_dvfvae)
            unet3d_error.append(this_unet3d)
            dvf_vae_mse.append(this_dvfvae_mse)
            vm_ncc_all.append(this_vm_ncc);        dvf_vae_ncc_all.append(this_dvfvae_ncc)
            vm_ssim_all.append(this_vm_ssim);      dvf_vae_ssim_all.append(this_dvfvae_ssim)
 
            # <-- added: append this sample's SI/AP amplitude rows + patient id
            si_amp_gt.append(this_si_amp_gt)
            si_amp_pred.append(this_si_amp_pred)
            ap_amp_gt.append(this_ap_amp_gt)
            ap_amp_pred.append(this_ap_amp_pred)
            lr_amp_gt.append(this_lr_amp_gt)
            lr_amp_pred.append(this_lr_amp_pred)
            jac_det_mean_gt.append(this_jac_mean_gt);   jac_det_mean_pred.append(this_jac_mean_pred)
            jac_det_std_gt.append(this_jac_std_gt);     jac_det_std_pred.append(this_jac_std_pred)
            patient_ids_all.append(patient_no)

    # # ─── Context-swap diagnostic (DEBUG only) ─────────────────────────────────
    # # Re-runs inference with each sample's h_t replaced by the h_t from a sample
    # # shifted by N//2 positions (≈ half-cycle apart).
    # # NavCorr(swapped) << NavCorr(original)  →  model truly uses h_t content.
    # # NavCorr(swapped) ≈  NavCorr(original)  →  model ignores h_t (memorises position).
    # if model.DEBUG and len(_swap_buf) > 3:
    #     N     = len(_swap_buf)
    #     shift = N // 2
    #     print(f"\n{'='*64}")
    #     print(f"[ContextSwapTest] N={N}  shift={shift}  (h_t rotated by {shift} samples ≈ half cycle)")
    #     print(f"{'='*64}")

    #     swap_nav_pred, swap_nav_gt = [], []

    #     with torch.no_grad():
    #         for i in range(N):
    #             j = (i + shift) % N
    #             ref_d   = _swap_buf[i][0].to(device)
    #             f_ref_d = _swap_buf[i][1].to(device)
    #             h_t_j   = _swap_buf[j][2].to(device)
    #             phase_j = _swap_buf[j][3].to(device) if _swap_buf[j][3] is not None else None
    #             gt_d    = _swap_buf[i][4].to(device) if _swap_buf[i][4] is not None else None

    #             out_sw = model._inference_forward(
    #                 ref_d, f_ref_d, [h_t_j],
    #                 phase=phase_j, ddim_steps=50,
    #             )

    #             if not out_sw.generated_vols:
    #                 continue

    #             nc_sw, _, _ = navigator_signal_3planes(out_sw.generated_vols[0], ref_d)
    #             swap_nav_pred.append(float(nc_sw))

    #             if gt_d is not None:
    #                 nc_gt, _, _ = navigator_signal_3planes(gt_d, ref_d)
    #                 swap_nav_gt.append(float(nc_gt))

    #     orig_nav_pred = [float(v) for row in nav_signals_pred for v in row
    #                      if not np.isnan(float(v))]
    #     orig_nav_gt   = [float(v) for row in nav_signals_gt  for v in row
    #                      if not np.isnan(float(v))]

    #     n = min(len(swap_nav_gt), len(swap_nav_pred), len(orig_nav_pred), len(orig_nav_gt))
    #     if n > 3:
    #         from scipy.stats import pearsonr
    #         corr_orig, _ = pearsonr(orig_nav_gt[:n],  orig_nav_pred[:n])
    #         corr_swap, _ = pearsonr(swap_nav_gt[:n],  swap_nav_pred[:n])
    #         diff_mean    = float(np.mean(np.abs(
    #             np.array(orig_nav_pred[:n]) - np.array(swap_nav_pred[:n])
    #         )))
    #         print(f"  NavCorr original ctx : {corr_orig:+.3f}")
    #         print(f"  NavCorr swapped  ctx : {corr_swap:+.3f}")
    #         print(f"  Mean |nav_orig - nav_swap| : {diff_mean:.5f}")
    #         if diff_mean < 1e-4:
    #             interp = "Model IGNORES context (outputs are identical regardless of h_t)"
    #         elif abs(corr_orig - corr_swap) < 0.1:
    #             interp = "Model likely IGNORES h_t (swap has no effect on NavCorr)"
    #         elif corr_swap < 0 < corr_orig:
    #             interp = "Model STRONGLY uses context (NavCorr flips sign with swap)"
    #         elif corr_swap < corr_orig - 0.2:
    #             interp = "Model USES context (swapped h_t degrades NavCorr)"
    #         else:
    #             interp = "Ambiguous — inspect diff_mean and individual correlations"
    #         print(f"  -> {interp}")
    #     else:
    #         print(f"  Too few valid samples (n={n}) to compute correlations.")

    #     print('='*64)

    # ─── Noise-seed sensitivity test (DEBUG only) ─────────────────────────────
    # Re-runs inference with different RNG seeds → different starting noise z_T,
    # while keeping h_t and ref_volume identical.
    # NavCorr stays high  →  model learned a true conditional distribution (good).
    # NavCorr drops badly →  model memorised specific (z_T, h_t) → z_0 pairs.
    # if model.DEBUG and len(_swap_buf) > 3:
    #     print(f"\n{'='*64}")
    #     print(f"[NoiseSeedTest] Re-running with scrambled starting noise z_T (N={len(_swap_buf)})")
    #     print(f"{'='*64}")

    #     noise_nav_pred, noise_nav_gt = [], []

    #     with torch.no_grad():
    #         for i, entry in enumerate(_swap_buf):
    #             ref_d   = entry[0].to(device)
    #             f_ref_d = entry[1].to(device)
    #             h_t_d   = entry[2].to(device)
    #             phase_d = entry[3].to(device) if entry[3] is not None else None
    #             gt_d    = entry[4].to(device) if entry[4] is not None else None

    #             # Deliberately different seed → different z_T, same h_t
    #             torch.manual_seed(99999 + i * 7919)
    #             torch.cuda.manual_seed_all(99999 + i * 7919)

    #             out_ns = model._inference_forward(
    #                 ref_d, f_ref_d, [h_t_d],
    #                 phase=phase_d, ddim_steps=50,
    #             )

    #             if not out_ns.generated_vols:
    #                 continue

    #             nc_ns, _, _ = navigator_signal_3planes(out_ns.generated_vols[0], ref_d)
    #             noise_nav_pred.append(float(nc_ns))

    #             if gt_d is not None:
    #                 nc_gt, _, _ = navigator_signal_3planes(gt_d, ref_d)
    #                 noise_nav_gt.append(float(nc_gt))

    #     n_ns = min(len(noise_nav_gt), len(noise_nav_pred))
    #     if n_ns > 3:
    #         from scipy.stats import pearsonr
    #         corr_ns, _ = pearsonr(noise_nav_gt[:n_ns], noise_nav_pred[:n_ns])
    #         orig_nav_gt   = [float(v) for row in nav_signals_gt  for v in row if not np.isnan(float(v))]
    #         orig_nav_pred = [float(v) for row in nav_signals_pred for v in row if not np.isnan(float(v))]
    #         n_ref = min(len(orig_nav_gt), len(orig_nav_pred))
    #         corr_ref, _ = pearsonr(orig_nav_gt[:n_ref], orig_nav_pred[:n_ref])
    #         diff_ns = float(np.mean(np.abs(
    #             np.array(orig_nav_pred[:n_ns]) - np.array(noise_nav_pred[:n_ns])
    #         )))
    #         print(f"  NavCorr original seeds  : {corr_ref:+.3f}")
    #         print(f"  NavCorr scrambled seeds : {corr_ns:+.3f}")
    #         print(f"  Mean |nav_orig - nav_noise| : {diff_ns:.5f}")
    #         if diff_ns < 1e-4:
    #             interp_ns = "Outputs identical — model ignores starting noise (fully deterministic)"
    #         elif abs(corr_ref - corr_ns) < 0.15:
    #             interp_ns = "NavCorr robust to noise — model learned a true conditional distribution"
    #         elif corr_ns < corr_ref - 0.3:
    #             interp_ns = "NavCorr drops with different noise — starting noise may be memorised"
    #         else:
    #             interp_ns = "Moderate sensitivity — expected stochasticity in diffusion outputs"
    #         print(f"  -> {interp_ns}")
    #     else:
    #         print(f"  Too few valid samples (n={n_ns}).")

    #     print('='*64)

    np.save(os.path.join(save_dir, "NCC_loss.npy"),          np.asarray(NCC_loss))
    np.save(os.path.join(save_dir, "MSE_loss.npy"),          np.asarray(MSE_loss))
    np.save(os.path.join(save_dir, "MSE_loss_gt.npy"),       np.asarray(MSE_loss_gt))
    np.save(os.path.join(save_dir, "SSIM_loss.npy"),         np.asarray(SSIM_loss))
    np.save(os.path.join(save_dir, "geo_error.npy"),         np.asarray(geo_errors))
    np.save(os.path.join(save_dir, "PSNR_loss.npy"),         np.asarray(PSNR_loss))
    np.save(os.path.join(save_dir, "folding_ratio.npy"),     np.asarray(folding_ratio_gen))
    np.save(os.path.join(save_dir, "temporal_smoothness.npy"), np.asarray(temporal_smoothness))
    np.save(os.path.join(save_dir, "vm_error.npy"),      np.asarray(vm_error))
    np.save(os.path.join(save_dir, "dvf_vae_error.npy"), np.asarray(dvf_vae_error))
    np.save(os.path.join(save_dir, "unet3d_error.npy"),  np.asarray(unet3d_error))
    np.save(os.path.join(save_dir, "dvf_vae_mse.npy"),   np.asarray(dvf_vae_mse))
    np.save(os.path.join(save_dir, "cycle_phases.npy"),      np.asarray(cycle_phases))
    np.save(os.path.join(save_dir, "cycle_phases_time.npy"), np.asarray(cycle_phases_time))
    np.save(os.path.join(save_dir, "landmark_error.npy"),
            np.array(landmark_error, dtype=object))
    np.save(os.path.join(save_dir, "landmark_error_gt.npy"),
            np.array(landmark_error_gt, dtype=object))
    np.save(os.path.join(save_dir, "landmark_dvf_error.npy"),
            np.array(landmark_dvf_error, dtype=object))
    np.save(os.path.join(save_dir, "nav_signals_gt.npy"),
            np.array(nav_signals_gt, dtype=object))
    np.save(os.path.join(save_dir, "nav_signals_pred.npy"),
            np.array(nav_signals_pred, dtype=object))
    np.save(os.path.join(save_dir, "nav_sagittal_gt.npy"),
            np.array(nav_sagittal_gt, dtype=object))
    np.save(os.path.join(save_dir, "nav_sagittal_pred.npy"),
            np.array(nav_sagittal_pred, dtype=object))
    np.save(os.path.join(save_dir, "nav_axial_gt.npy"),
            np.array(nav_axial_gt, dtype=object))
    np.save(os.path.join(save_dir, "nav_axial_pred.npy"),
            np.array(nav_axial_pred, dtype=object))
    np.save(os.path.join(save_dir, "dvf_amp_gt.npy"),   np.asarray(_diag_amp_gt))
    np.save(os.path.join(save_dir, "dvf_amp_pred.npy"), np.asarray(_diag_amp_gen))
    np.save(os.path.join(save_dir, "vm_ncc.npy"),       np.asarray(vm_ncc_all))
    np.save(os.path.join(save_dir, "dvf_vae_ncc.npy"), np.asarray(dvf_vae_ncc_all))
    np.save(os.path.join(save_dir, "vm_ssim.npy"),      np.asarray(vm_ssim_all))
    np.save(os.path.join(save_dir, "dvf_vae_ssim.npy"),np.asarray(dvf_vae_ssim_all))
 
    # <-- added: SI/AP amplitude arrays + patient_ids.npy, one row per
    # sample (idx), for amp_vs_phase_or_time_plot.py.
    np.save(os.path.join(save_dir, "dvf_si_amp_gt.npy"),
            np.array(si_amp_gt, dtype=object))
    np.save(os.path.join(save_dir, "dvf_si_amp_pred.npy"),
            np.array(si_amp_pred, dtype=object))
    np.save(os.path.join(save_dir, "dvf_ap_amp_gt.npy"),
            np.array(ap_amp_gt, dtype=object))
    np.save(os.path.join(save_dir, "dvf_ap_amp_pred.npy"),
            np.array(ap_amp_pred, dtype=object))
    np.save(os.path.join(save_dir, "dvf_lr_amp_gt.npy"),
            np.array(lr_amp_gt, dtype=object))
    np.save(os.path.join(save_dir, "dvf_lr_amp_pred.npy"),
            np.array(lr_amp_pred, dtype=object))
    np.save(os.path.join(save_dir, "patient_ids.npy"),
            np.array(patient_ids_all, dtype=object))
    np.save(os.path.join(save_dir, "jac_det_mean_gt.npy"),
            np.array(jac_det_mean_gt,  dtype=object))
    np.save(os.path.join(save_dir, "jac_det_mean_pred.npy"),
            np.array(jac_det_mean_pred, dtype=object))
    np.save(os.path.join(save_dir, "jac_det_std_gt.npy"),
            np.array(jac_det_std_gt,   dtype=object))
    np.save(os.path.join(save_dir, "jac_det_std_pred.npy"),
            np.array(jac_det_std_pred,  dtype=object))
 
    lm_mean     = np.nanmean([v for row in landmark_error     for v in row])
    lm_gt_mean  = np.nanmean([v for row in landmark_error_gt  for v in row])
    lm_dvf_mean = np.nanmean([v for row in landmark_dvf_error for v in row])
    all_nav_gt   = [v for row in nav_signals_gt   for v in row if not np.isnan(float(v))]
    all_nav_pred = [v for row in nav_signals_pred for v in row if not np.isnan(float(v))]
    nav_m = navigator_metrics(all_nav_gt, all_nav_pred)
    nav_pred_arr = np.array(all_nav_pred, dtype=float)
    nav_pred_std = float(np.std(nav_pred_arr)) if len(nav_pred_arr) > 1 else float("nan")
    nav_pred_cv  = (nav_pred_std / float(np.mean(nav_pred_arr))
                    if len(nav_pred_arr) > 1 and float(np.mean(nav_pred_arr)) > 1e-8
                    else float("nan"))
    np.save(os.path.join(save_dir, "nav_pred_std.npy"), nav_pred_std)
    print(
        "\nTest avg  NCC: %.4f  MSE(vm): %.4f  MSE(gt): %.4f  SSIM: %.4f  PSNR: %.2f dB  "
        "Folding: %.4f  TempSmooth: %.4f  "
        "Landmark: %.4f mm  LM-GT: %.4f mm  LM-DVF: %.4f mm  NavRatio: %.3f  NavCorr: %.3f  "
        "NavPredStd: %.5f  NavPredCV: %.3f"
        % (
            np.nanmean(NCC_loss),
            np.nanmean(MSE_loss),
            np.nanmean(MSE_loss_gt),
            np.nanmean(SSIM_loss),
            np.nanmean(PSNR_loss),
            np.nanmean(folding_ratio_gen),
            np.nanmean(temporal_smoothness),
            lm_mean,
            lm_gt_mean,
            lm_dvf_mean,
            nav_m["amplitude_ratio"],
            nav_m["phase_corr"],
            nav_pred_std,
            nav_pred_cv,
        )
    )
 
    # ---- assemble tracking volumes into NIfTI sequences --------------------
    for base_dir in (track_dir, vol_dir):
        for case in next(os.walk(base_dir))[1]:
            if "DVF" in case:
                continue
            path      = os.path.join(base_dir, case, "")
            path_save = path[:-1] + ".nii.gz"
            assemble_volumes(path, path_save, target_imgs=False, downsampled=cfg.downsample_to_liver)
 
    summarize_test_metrics(save_dir)

    # from mopred.utils.acdc_seg_area_metric import compute_dice_for_test_dir
    # compute_dice_for_test_dir(
    #     save_dir    = save_dir,
    #     data_dir    = cfg.data_dir,
    #     methods     = {"MambaMorph": "DVF_t0", "MoPred-Diff": "generated_DVF_t0"},
    #     orientation = orientation,
    # )

    if model.DEBUG and _dbg_phi:
        # ── Phase-binned summary: VAE culprit check + noise-schedule check ───
        n_bins  = 10
        bin_w   = 0.5 / n_bins
        phi_arr = np.array(_dbg_phi)
        z0_arr  = np.array(_dbg_z0_norm)
        vae_arr = np.array(_dbg_vae_mse)
        dvf_arr = np.array(_dbg_dvf_mag)
 
        print("\n[DEBUG z0] Phase-binned summary")
        print(f"  {'phi_bin':>8}  {'n':>4}  {'dvf_mag':>8}  {'||z0||':>8}  {'vae_mse':>10}")
        print("  " + "-" * 48)
        for b in range(n_bins):
            mask = (phi_arr >= b * bin_w) & (phi_arr < (b + 1) * bin_w)
            if mask.sum() == 0:
                continue
            print(
                f"  phi≈{b*bin_w:.2f}-{(b+1)*bin_w:.2f}"
                f"  {mask.sum():>4}"
                f"  {dvf_arr[mask].mean():>8.4f}"
                f"  {z0_arr[mask].mean():>8.3f}"
                f"  {vae_arr[mask].mean():>10.6f}"
            )
        print(
            "\n  Interpretation:"
            "\n    vae_mse flat across bins → VAE is NOT the culprit (confirmed)"
            "\n    ||z0|| grows with phi    → noise schedule miscalibrated at high motion"
            "\n    ||z0|| flat              → look elsewhere (DDIM steps, conditioning)"
        )
 
    if _diag_pids:
        _dvf_diversity_diagnostic(
            patient_ids = _diag_pids,
            gt_vecs     = _diag_gt_vecs,
            gen_vecs    = _diag_gen_vecs,
            amp_gt      = _diag_amp_gt,    amp_gen   = _diag_amp_gen,
            hf_gt       = _diag_hf_gt,     hf_gen    = _diag_hf_gen,
            jstd_gt     = _diag_jstd_gt,   jstd_gen  = _diag_jstd_gen,
            jmean_gt    = _diag_jmean_gt,  jmean_gen = _diag_jmean_gen,
            save_path   = os.path.join(save_dir, "dvf_diversity_diagnostic.txt"),
        )


# =============================================================================
# Private helpers
# =============================================================================


def _make_loaders(
    cfg:   argparse.Namespace,
    folds: tuple,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for Stage 2."""
    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    train_set = ACDC_4D_Dataset(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp, valid=True,
        **ds_kwargs
    )
    
    train_set.repeats = 15
    
    print("="*50)
    print("ACDC training dataset")
    print("Number of samples:", len(train_set))
    print("Number of patients:", len(folds[0]))
    print("Repeat factor:", train_set.repeats)
    print("Base samples before repeats:", train_set._base_len)
    print("="*50)

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    # ---- orientation info (per-patient reference slice / centroid / rotation) ---
    valid_set = ACDC_4D_Dataset(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[1], nb_pred=cfg.tp, valid=True,
        **ds_kwargs
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        worker_init_fn=_seed_worker, generator=_g,
    )

    valid_loader = DataLoader(
        valid_set, batch_size=1,
        shuffle=False, num_workers=cfg.num_workers,
        worker_init_fn=_seed_worker,
    )
    return train_loader, valid_loader


def _write_patient_split(folds: tuple, log_dir: str) -> None:
    # ACDC fold entries are already plain "patientXXX" IDs (unlike the liver
    # pipeline's path/sequence strings), so no extraction is needed here —
    # extract_patient_ids' seq[:8] fallback would collapse e.g. "patient023"
    # and "patient099" into the same "patient0" bucket.
    train_patients = sorted(set(folds[0]))
    val_patients   = sorted(set(folds[1]))
    with open(os.path.join(log_dir, "patients_split.txt"), "w") as f:
        f.write("=== TRAIN PATIENTS ===\n")
        for p in train_patients:
            f.write(p + "\n")
        f.write("\n=== VALIDATION PATIENTS ===\n")
        for p in val_patients:
            f.write(p + "\n")


def _module_grad_norm(module) -> float:
    return sum(
        p.grad.data.norm(2).item() ** 2
        for p in module.parameters()
        if p.grad is not None
    ) ** 0.5


def _log_diffusion_step(writer, step, cfg, out, optimizer, model, loss):
    def _val(x):
        return x.item() if torch.is_tensor(x) else float(x or 0)

    writer.add_scalar("train/ddpm_loss",       _val(out.ddpm_loss),       step)
    writer.add_scalar("train/dvf_recon_loss",  _val(out.dvf_recon_loss),  step)
    writer.add_scalar("train/vmorph_vol_loss", _val(out.vmorph_vol_loss), step)
    writer.add_scalar("train/vol_recon_loss",  _val(out.vol_recon_loss),  step)
    writer.add_scalar("train/nav_corr_loss",   _val(out.nav_corr_loss),   step)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"],        step)
    writer.add_scalar("train/total_loss", loss.item(), step)

    total_norm = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5
    writer.add_scalar("train/grad_norm", total_norm, step)

    # Per-module gradient norms — zero means no gradient (frozen or unused)
    writer.add_scalar("grad/denoising_net", _module_grad_norm(model.denoising_net), step)
    writer.add_scalar("grad/cond_proj",     _module_grad_norm(model.cond_proj),     step)
    if model.cond_net is not None:
        writer.add_scalar("grad/cond_net", _module_grad_norm(model.cond_net), step)
    if model.ref_net is not None:
        writer.add_scalar("grad/ref_net",  _module_grad_norm(model.ref_net),  step)


# =============================================================================
# Multi-GPU per-fold worker
# =============================================================================

def _train_fold_worker(
    fi:          int,
    cfg:         argparse.Namespace,
    train_folds: list,
    valid_folds: list,
    dir_name:    str,
    gpu_idx:     int,
) -> None:
    """Entry point for a per-fold subprocess (one GPU per fold)."""
    import random

    device = torch.device(f"cuda:{gpu_idx}")
    torch.cuda.set_device(gpu_idx)
    cfg = argparse.Namespace(**vars(cfg))   # shallow copy so mutations stay local
    cfg.device = device

    random.seed(cfg.seed + fi)
    np.random.seed(cfg.seed + fi)
    torch.manual_seed(cfg.seed + fi)
    torch.cuda.manual_seed_all(cfg.seed + fi)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    current_path  = os.path.dirname(os.path.abspath(__file__))
    VM_CHECKPOINT = os.path.join(os.path.dirname(current_path), "pretrained_models", "VM_acdc.pth")
    VOL_SIZE      = LIVER_SHAPE if getattr(cfg, "downsample_to_liver", False) else CANONICAL_SHAPE

    vm = Voxelmorph(
        VOL_SIZE, [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16], full_size=True,
    ).to(device)
    custom_load(vm, VM_CHECKPOINT, device)
    vm.eval()
    for p in vm.parameters():
        p.requires_grad_(False)

    stn       = SpatialTransformer(VOL_SIZE).to(device)
    criterion = build_criterion(cfg)

    print(f"[Fold {fi}] starting on GPU {gpu_idx}")
    train_diffusion(
        cfg      = cfg,
        folds    = (train_folds[fi], valid_folds[fi]),
        fold_idx = str(fi),
        dir_name = dir_name,
        vm       = vm,
        stn      = stn,
        criterion= criterion,
        device   = device,
    )


# =============================================================================
# CLI & entry point
# =============================================================================

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Stage-2 diffusion training pipeline")
    p.add_argument("--config",     required=True,
                   help="Path to a YAML config file (see configs/).")
    p.add_argument("--train_test", required=True,
                   choices=["train", "test"],
                   help="'train' runs Stage-2 training; 'test' runs evaluation on the test fold.")
    p.add_argument("--fold_nb_training", type=int, default=0,
                   help="Number of folds to run (0 = all).")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override any config key, e.g. --override lr=5e-5 batch_size=8")
    p.add_argument("--test_tag", default=None,
                   help="Optional tag appended to the test output directory (e.g. 'ssg1.5'). "
                        "Results go to test_{tag}/ instead of test/, leaving existing results intact.")
    p.add_argument("--test_seed", type=int, default=42,
                   help="Random seed used during test-time sampling (default: 42).")
    p.add_argument("--metrics_only", action="store_true",
                   help="Skip saving NIfTI volumes and tracking sequences; only save metric .npy files and summary_metrics.txt.")
    return p.parse_args()


if __name__ == "__main__":
    import torch.multiprocessing as _mp

    args = _parse_cli()
    cfg  = load_config(args.config)
    cfg.fold_nb_training = args.fold_nb_training
    cfg.test_tag          = args.test_tag
    cfg.test_seed         = args.test_seed
    cfg.metrics_only      = args.metrics_only
    _apply_overrides(cfg, args.override)

    print("\n=== Config ===")
    print("\n".join(f"  {k}: {v}" for k, v in sorted(vars(cfg).items())))
    print("==============\n")

    # ---- parse GPU list -------------------------------------------------------
    # gpu_idx in config can be a single value ("0") or a comma-separated list
    # ("0,1,2").  With a list, each fold is launched in its own subprocess on
    # the corresponding GPU (cycling if there are more folds than GPUs).
    _raw_gpu = str(cfg.gpu_idx)
    gpu_list = [int(g.strip()) for g in _raw_gpu.split(",")]
    multi_gpu = len(gpu_list) > 1

    print("CUDA available:", torch.cuda.is_available())
    print("GPU(s):", gpu_list)

    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    patient_subset = getattr(cfg, "patient_subset", None)
    train_folds, valid_folds, test_folds = make_train_val_test_folds(
        data_dir=cfg.data_dir, patient_subset=patient_subset,
    )

    for fi, (tr, va, te) in enumerate(zip(train_folds, valid_folds, test_folds)):
        n = len(tr) + len(va) + len(te)
        print(f"valid patients names: {', '.join(va)}")
        print(f"[Fold {fi}] train={len(tr)} ({100*len(tr)/n:.0f}%)  "
              f"val={len(va)} ({100*len(va)/n:.0f}%)  "
              f"test={len(te)} ({100*len(te)/n:.0f}%)  total={n}")

    fold_nb = cfg.fold_nb_training or len(train_folds)
    print(f"Running on {fold_nb} fold(s).")

    if args.train_test == "train":
        dir_name = os.path.join(
            datetime.datetime.now().strftime("%m_%d"),
            datetime.datetime.now().strftime("%H.%M._") + cfg.name,
        )

        if multi_gpu:
            # ---- one subprocess per fold, each pinned to its GPU --------------
            _mp.set_start_method("spawn", force=True)
            processes = []
            for fi in range(fold_nb):
                gpu_idx = gpu_list[fi % len(gpu_list)]
                p = _mp.Process(
                    target=_train_fold_worker,
                    args=(fi, cfg, train_folds, valid_folds, dir_name, gpu_idx),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError(f"Fold subprocess exited with code {p.exitcode}")

        else:
            # ---- original sequential behaviour --------------------------------
            device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
            torch.cuda.set_device(gpu_list[0])
            cfg.device = device

            current_path  = os.path.dirname(os.path.abspath(__file__))
            VM_CHECKPOINT = os.path.join(os.path.dirname(current_path), "pretrained_models", "VM_acdc.pth")
            VOL_SIZE      = LIVER_SHAPE if getattr(cfg, "downsample_to_liver", False) else CANONICAL_SHAPE

            vm = Voxelmorph(
                VOL_SIZE, [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16], full_size=True,
            ).to(device)
            custom_load(vm, VM_CHECKPOINT, device)
            vm.eval()
            for p in vm.parameters():
                p.requires_grad_(False)

            stn       = SpatialTransformer(VOL_SIZE).to(device)
            criterion = build_criterion(cfg)

            for fi in range(fold_nb):
                random.seed(cfg.seed + fi)
                np.random.seed(cfg.seed + fi)
                torch.manual_seed(cfg.seed + fi)
                torch.cuda.manual_seed_all(cfg.seed + fi)
                if fi == 0 :
                    print(f"skipping [Fold {fi}]")
                    continue
                else : 
                    print(f"[Fold {fi}] starting on GPU {gpu_list[0]}")
                    train_diffusion(
                        cfg      = cfg,
                        folds    = (train_folds[fi], valid_folds[fi]),
                        fold_idx = str(fi),
                        dir_name = dir_name,
                        vm       = vm,
                        stn      = stn,
                        criterion= criterion,
                        device   = device,
                    )

    else:  # test
        device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(gpu_list[0])
        cfg.device = device

        VOL_SIZE = (32, 64, 64)
        vm = build_reg_model(cfg, VOL_SIZE, device)

        stn = SpatialTransformer(VOL_SIZE).to(device)

        dir_name = cfg.checkpoint

        folds_to_use = test_folds
        for fi in range(fold_nb):
            fold = folds_to_use[fi]
            test(
                cfg      = cfg,
                fold     = fold,
                fold_idx = str(fi),
                dir_name = dir_name,
                vm       = vm,
                stn      = stn,
                device   = device,
            )