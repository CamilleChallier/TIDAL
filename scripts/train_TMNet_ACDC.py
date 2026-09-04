"""
train_TMNet_ACDC.py
======================
Cardiac analog of train_TMNet.py — pretraining pipeline for TMNet variants
trained from scratch on the public ACDC cine-MRI dataset.

Standalone copy: hard-codes ACDC dataset/loaders/folds/VOL_SIZE/VM checkpoint.
Liver script left untouched.

Slice positions for conditioning (design decision #5 in plan):
  condi_type "2" (coronal):  mid-short-axis plane at H//2 = 64   (was Y_NAV=32 for liver)
  condi_type "1" (sagittal): per-patient D-index via get_reference_slice_index
                             (falls back to D//2 = 16 when no orientation info
                             is available — was a fixed 16 for liver)

Usage
-----
# DVF-supervised TMNet (dvfsup2) — primary ACDC training mode
python -m 4D_MoPred_liver.scripts.train_TMNet_ACDC  --config 4D_MoPred_liver/configs/CondNets/TMNet_acdc.yaml --train_test train_tmnet_priormulti_dvf
"""

from __future__ import annotations

import argparse
import datetime
import os
import time
import warnings

from torch.nn import functional as F

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, dynamic_ncols=True)

from mopred.models.Context_Encoder import (
    TMNetEncoder, TMNet_Tr_priormulti_image, PhaseContrastiveLoss,
    PredictiveTMNet, MopTRTMNet, DVFSupTMNet,
)
from mopred.models.Context_Encoder.temporal_augmentations import TemporalAugConfig

from mopred.data.splits       import make_folds_acdc as make_train_val_test_folds
from mopred.data.data_loaders.acdc_4d import (
    ACDC_4D_Dataset, CANONICAL_SHAPE, get_cardiac_phase, load_orientation_info,
    get_reference_slice_index,
)
from mopred.data.loading      import save_params_txt
from mopred.models            import Voxelmorph
from mopred.utils.early_stopping  import EarlyStopping
from mopred.utils.io              import cond_mkdir, custom_load, custom_save
from mopred.utils.training        import (
    load_config, _apply_overrides,
    build_scheduler, build_optimizer,
    save_patients, _build_tmnet_model, build_reg_model
)
from mopred.utils.losses    import ncc_loss

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


# =============================================================================
# Phase helpers — cardiac analog of get_phi / get_amplitude (navigator_4d.py)
# =============================================================================
# ACDC gives the exact cardiac phase directly from Info.cfg.
# _get_amplitude and _get_peak_amplitude have no cardiac equivalent but are
# used by a few optional training variants (aux-loss supervision); they return
# 0.0 so those variants compile and run without crashing (amplitude-based loss
# weights should be set to 0.0 in the config when training on ACDC).

def _get_phi(patient, t_idx, data_dir):
    """Distance-from-ED ∈ [0, 0.5] — cardiac analog of get_phi."""
    phase = get_cardiac_phase(patient, t_idx, data_dir)
    return min(phase, 1.0 - phase)


def _get_phi_vec(patient, t_idx, data_dir):
    """(cos, sin) of raw phase φ ∈ [0,1) — direction-aware alternative to _get_phi.

    cos(2πφ) alone reproduces _get_phi's symmetric "distance from ED" shape
    (continuous across the φ=0/1 wraparound — no discontinuity for an MSE loss),
    but min(φ,1-φ) throws away which side of ED/ES a frame is on. sin(2πφ) adds
    that back continuously (positive during systole/contracting, negative during
    diastole/relaxing) without a brittle threshold right at φ=0 or φ=0.5 the way
    a hard direction flag would have. Used by train_tmnet_priormulti_dvf only
    (TMNet_Tr_priormulti_image's phi_head with phi_head_dim=2) — see
    project_acdc_pipeline_status memory.
    """
    phase = get_cardiac_phase(patient, t_idx, data_dir)
    angle = 2 * np.pi * phase
    return np.cos(angle), np.sin(angle)


def _get_amplitude(patient, t_idx, data_dir):
    return 0.0


def _get_peak_amplitude(patient, data_dir):
    return 0.0


# Mid-ventricular plane constants replacing liver's Y_NAV=32.
# For ACDC (32,128,128): coronal mid-plane at H//2=64, sagittal mid at D//2=16.
# _D_CARDIAC is now only a FALLBACK for condi_type "1" — used when no
# orientation info is available at all (see get_reference_slice_index).
_Y_CARDIAC = CANONICAL_SHAPE[1] // 2   # 64  (coronal, condi_type "2")
_D_CARDIAC = CANONICAL_SHAPE[0] // 2   # 16  (sagittal, condi_type "1", fallback only)


# =============================================================================
# Input builders
# =============================================================================

def _unpack_condi_inputs(batch, device: torch.device, cfg) -> torch.Tensor:
    """
    Extract past mid-ventricular slices and build (B, 2, T, H, W) tensor.

    condi_type "1" → sagittal at cfg.sag_pos (default D//2 = 8)
    condi_type "2" → coronal  at cfg.cor_pos (default H//2 = 64)

    Mirrors liver's _unpack_condi_inputs but with ACDC-appropriate defaults
    (D//2 and H//2 of CANONICAL_SHAPE instead of hard-coded liver values 16/32).

    NOTE: unlike _build_priormulti_image_inputs, this helper still uses a
    fixed sag_pos shared across the whole batch — it backs the older
    train_tmnet / train_tmnet_predictive / train_tmnet_mopTR_style /
    train_tmnet_contrastive / train_tmnet_dvfsup variants, which have
    not been migrated to per-patient reference-slice indexing.
    """
    ref_raw, input_volume_list, *_ = batch
    ref_vol = ref_raw.unsqueeze(1).to(device)          # (B, 1, D, H, W)

    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    sag_pos    = getattr(cfg, "sag_pos", _D_CARDIAC)
    cor_pos    = getattr(cfg, "cor_pos", _Y_CARDIAC)

    frames_list = []

    if condi_type == "1":                              # sagittal
        c_ref = ref_vol[:, :, sag_pos, :, :]          # (B, 1, H, W)
        for q in range(cfg.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1).to(device)[:, :, sag_pos, :, :]
            frames_list.append(torch.cat([c_t, c_ref], dim=1))
    else:                                              # coronal
        c_ref = ref_vol[:, :, :, cor_pos, :]          # (B, 1, D, W)
        for q in range(cfg.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1).to(device)[:, :, :, cor_pos, :]
            frames_list.append(torch.cat([c_t, c_ref], dim=1))

    return torch.stack(frames_list, dim=2).float()     # (B, 2, T, H, W)


def _tmnet_checkpoint_path(cfg, dir_name: str, fold_idx: str) -> str:
    return os.path.join(
        cfg.logging_dir, "logs", dir_name,
        f"fold_{fold_idx}", "tmnet", "model_best_tmnet.pth",
    )


def _build_aug_cfg(cfg) -> TemporalAugConfig:
    aug_dict = {
        "frame_drop":      getattr(cfg, "temporal_aug_frame_drop",      {}),
        "spatial_mask":    getattr(cfg, "temporal_aug_spatial_mask",    {}),
        "tube_mask":       getattr(cfg, "temporal_aug_tube_mask",        {}),
        "variable_frames": getattr(cfg, "temporal_aug_variable_frames", {}),
    }
    return TemporalAugConfig.from_dict(aug_dict)

def _acdc_dataset_kwargs(cfg) -> dict:
    """Shared ACDC_4D_Dataset kwargs derived from cfg — used by every
    instantiation site (train/valid/test) so they can never drift apart and
    silently end up preprocessed differently from one another."""
    return dict(
        downsample_to_liver=getattr(cfg, "downsample_to_liver", False),
        # cache_dir=getattr(cfg, "cache_dir", None),
        reference_slices_path=getattr(cfg, "reference_slices_path", None),
        stride=getattr(cfg, "stride", 1),
    )


def _patients_from_vol_files(vol_files) -> list[str]:
    """Extract each batch element's patient id from vol_files[0] (horizon-0
    paths, one per batch element — same convention used throughout this
    file). Shared helper so every call site resolves patient ids the same
    way."""
    return [p.split("/")[-2] for p in vol_files[0]]


def _d_idx_from_batch(vol_files, cfg, orientation, device) -> torch.Tensor:
    """Per-sample D-index (B,) LongTensor: where each batch element's own
    patient's reference slice / centroid actually lands in the canonical
    crop, via get_reference_slice_index. Shared by
    _build_priormulti_image_inputs and _dvf_slice so the image conditioning
    and the DVF ground-truth target are always sliced at the SAME depth for
    a given patient — using a fixed scalar for one and a per-patient index
    for the other would silently misalign supervision."""
    patients = _patients_from_vol_files(vol_files)
    return torch.tensor(
        [get_reference_slice_index(p, cfg.data_dir, orientation) for p in patients],
        device=device, dtype=torch.long,
    )


# =============================================================================
# Stage 2b — TMNet MAE training
# =============================================================================

def train_tmnet(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    vm:       torch.nn.Module,
    device:   torch.device,
) -> None:

    condi, mae = _build_tmnet_model(cfg, device)
    print(
        f"\n[TMNet] {type(condi).__name__} + TMNetPriorMAE"
        f"  | output_dim={cfg.tmnet_pre_latent_dim}"
        f"  | mask_ratio={cfg.tmnet_mask_ratio}"
        f"  | patch_size={cfg.tmnet_patch_size}"
        f"  (device={next(condi.parameters()).device})"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[TMNet] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(condi, cfg.checkpoint, device)

    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet")
        run_dir = os.path.join(cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet")
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)
    
    ds_kwargs = _acdc_dataset_kwargs(cfg)
    train_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[0], nb_pred=cfg.tp, **ds_kwargs)
    valid_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[1], nb_pred=cfg.tp, valid=True, **ds_kwargs)

    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker, generator=_g)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker)

    optimizer     = build_optimizer(cfg, mae.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True,
                                  delta=cfg.early_stopping_delta)

    restart_epoch     = getattr(cfg, "restart_epoch", 0)
    warmup_epochs     = getattr(cfg, "tmnet_mask_warmup_epochs", 0)
    mask_ratio_target = cfg.tmnet_mask_ratio
    aug_cfg           = _build_aug_cfg(cfg)

    if aug_cfg.any_enabled():
        print(f"[TMNet] Temporal augmentations enabled.")
    else:
        print("[TMNet] No temporal augmentations enabled.")

    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    print(f"\nStage 2b — TMNet MAE training  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        condi.train()
        mae.train()

        if warmup_epochs > 0:
            annealed = min(1.0, (epoch + 1) / warmup_epochs)
            mae.mask_ratio = mask_ratio_target * annealed
            writer.add_scalar("tmnet_train/mask_ratio", mask_ratio_target * annealed, epoch)

        ep_loss = ep_steps = 0

        for batch in tqdm(train_loader):
            frames = _unpack_condi_inputs(batch, device, cfg)
            optimizer.zero_grad()
            out  = mae(frames, aug_cfg=aug_cfg, training=True)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            ep_loss  += loss.item()
            ep_steps += 1
            writer.add_scalar("tmnet_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("tmnet_train/epoch_loss", ep_avg, epoch)

        condi.eval()
        mae.eval()
        val_loss = val_n = 0

        train_mask_ratio = mae.mask_ratio
        mae.mask_ratio   = mask_ratio_target

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                frames   = _unpack_condi_inputs(batch, device, cfg)
                out      = mae(frames, aug_cfg=aug_cfg, training=False)
                val_loss += out["loss"].item()
                val_n    += 1

        mae.mask_ratio = train_mask_ratio

        val_avg = val_loss / max(val_n, 1)
        writer.add_scalar("tmnet_val/loss", val_avg, epoch)
        print(f"[TMNet] train={ep_avg:.6f}  val={val_avg:.6f}  mask={mae.mask_ratio:.3f}")

        if val_avg < best_val_loss:
            print(f"[TMNet] Improved {best_val_loss:.6f} → {val_avg:.6f} — saving.")
            best_val_loss = val_avg
            custom_save(condi, os.path.join(log_dir, "model_best_tmnet.pth"))
        else:
            print(f"[TMNet] No improvement from {best_val_loss:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            print("[TMNet] Early stopping triggered.")
            break

        print(f"[TMNet] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nStage 2b done.  Best checkpoint: {log_dir}/model_best_tmnet.pth")
    writer.close()

    print("[TMNet] Computing embedding statistics over full training set...")
    custom_load(condi, os.path.join(log_dir, "model_best_tmnet.pth"), device)
    condi.eval()

    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            frames   = _unpack_condi_inputs(batch, device, cfg)
            emb_list = mae.encode(sag=frames)
            for emb in emb_list:
                emb_sum    += emb.sum().item()
                emb_sq_sum += emb.pow(2).sum().item()
                emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)

    stats_path = os.path.join(log_dir, "tmnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(f"[TMNet] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  saved to {stats_path}")


# =============================================================================
# Stage 2b (variant) — predictive pretraining
# =============================================================================

def train_tmnet_predictive(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    vm:       torch.nn.Module,
    device:   torch.device,
) -> None:

    condi = TMNetEncoder(
        num_inputs=cfg.tmnet_num_frames, horizon=cfg.tmnet_horizon, in_channels=2,
        out_channels=cfg.tmnet_condi_channels, n_heads=cfg.tmnet_Tr_n_heads,
        enc_layers=cfg.tmnet_Tr_enc_layers, normalize_before=cfg.tmnet_Tr_norm_before,
        output_dim=cfg.tmnet_pre_latent_dim,
        condi_type=getattr(cfg, "tmnet_condi_type", "2"), device=device,
    ).to(device)

    img_size    = tuple(cfg.tmnet_img_size)
    in_channels = getattr(cfg, "tmnet_in_channels", 2)
    pred = PredictiveTMNet(tm_net=condi, img_size=img_size, img_channels=in_channels).to(device)

    if getattr(cfg, "checkpoint", None):
        custom_load(condi, cfg.checkpoint, device)

    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_pred")
        run_dir = os.path.join(cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_pred")
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    train_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[0], nb_pred=cfg.tp, downsample_to_liver=getattr(cfg, "downsample_to_liver", False), cache_dir=getattr(cfg, "cache_dir", None), stride=getattr(cfg, "stride", 1))
    valid_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[1], nb_pred=cfg.tp, valid=True, downsample_to_liver=getattr(cfg, "downsample_to_liver", False), cache_dir=getattr(cfg, "cache_dir", None), stride=getattr(cfg, "stride", 1))

    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker, generator=_g)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker)

    optimizer     = build_optimizer(cfg, pred.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True,
                                  delta=cfg.early_stopping_delta)
    aug_cfg       = _build_aug_cfg(cfg)
    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    print(f"\nStage 2b — TMNet predictive pretraining  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet-Pred] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        condi.train()
        pred.train()
        ep_loss = ep_steps = 0

        for batch in tqdm(train_loader):
            frames = _unpack_condi_inputs(batch, device, cfg)
            optimizer.zero_grad()
            out  = pred(frames, aug_cfg=aug_cfg, training=True)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            ep_loss  += loss.item()
            ep_steps += 1
            writer.add_scalar("tmnet_pred_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("tmnet_pred_train/epoch_loss", ep_avg, epoch)

        condi.eval()
        pred.eval()
        val_loss = val_n = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                frames   = _unpack_condi_inputs(batch, device, cfg)
                out      = pred(frames, aug_cfg=aug_cfg, training=False)
                val_loss += out["loss"].item()
                val_n    += 1

        val_avg = val_loss / max(val_n, 1)
        writer.add_scalar("tmnet_pred_val/loss", val_avg, epoch)
        print(f"[TMNet-Pred] train={ep_avg:.6f}  val={val_avg:.6f}")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            custom_save(condi, os.path.join(log_dir, "model_best_tmnet.pth"))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            break

        print(f"[TMNet-Pred] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    writer.close()

    custom_load(condi, os.path.join(log_dir, "model_best_tmnet.pth"), device)
    condi.eval()
    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            frames   = _unpack_condi_inputs(batch, device, cfg)
            emb_list = pred.encode(sag=frames)
            for emb in emb_list:
                emb_sum    += emb.sum().item()
                emb_sq_sum += emb.pow(2).sum().item()
                emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)
    stats_path = os.path.join(log_dir, "tmnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(f"[TMNet-Pred] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  saved to {stats_path}")


# =============================================================================
# Stage 2b — compute embedding statistics only
# =============================================================================

def compute_tmnet_stats(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    device:   torch.device,
) -> None:
    log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet")
    ckpt = os.path.join(log_dir, "model_best_tmnet.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[compute_stats] Checkpoint not found: {ckpt}")

    condi, mae = _build_tmnet_model(cfg, device)
    custom_load(condi, ckpt, device)
    condi.eval()
    print(f"[compute_stats] Loaded TMNet from {ckpt}")

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    train_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[0], nb_pred=cfg.tp, **ds_kwargs)
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    max_batches = getattr(cfg, "tmnet_stats_batches", 50)
    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            frames   = _unpack_condi_inputs(batch, device, cfg)
            emb_list = mae.encode(sag=frames)
            for emb in emb_list:
                emb_sum    += emb.sum().item()
                emb_sq_sum += emb.pow(2).sum().item()
                emb_count  += emb.numel()
            print(f"  [{i+1}/{max_batches}]", end="\r")

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)

    stats_path = os.path.join(log_dir, "tmnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(f"\n[compute_stats] mean={emb_mean:.4f}  std={emb_std:.4f}  → saved to {stats_path}")


# =============================================================================
# Stage 2b — test reconstruction quality
# =============================================================================

def test_tmnet(
    cfg,
    fold:     list,
    fold_idx: str,
    dir_name: str,
    device:   torch.device,
) -> None:
    condi, mae = _build_tmnet_model(cfg, device)
    log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet")
    ckpt = os.path.join(log_dir, "model_best_tmnet.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[test_tmnet] Checkpoint not found: {ckpt}")

    custom_load(condi, ckpt, device)
    condi.eval()
    mae.eval()
    mae.mask_ratio = cfg.tmnet_mask_ratio

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    test_set = ACDC_4D_Dataset(cfg.data_dir, sequence_list=fold, nb_pred=cfg.tp,
                               nb_inputs=cfg.nb_inputs, test=True,
                                **ds_kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    save_dir = os.path.join(cfg.logging_dir, dir_name, "test_tmnet", fold_idx)
    cond_mkdir(save_dir)

    test_patients = {seq.split("/")[0] if "/" in seq else seq[:10] for seq in fold}
    with open(os.path.join(save_dir, "patients_test.txt"), "w") as f:
        for p in sorted(test_patients):
            f.write(p + "\n")

    mse_all, mae_all, mape_all, psnr_all, ssim_all = [], [], [], [], []

    from pytorch_msssim import ssim as ssim_fn

    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader)):
            frames = _unpack_condi_inputs(batch, device, cfg)
            torch.manual_seed(42)
            out = mae(frames)

            mask   = out["mask_mae"]
            recon  = out["recon"]
            target = frames.permute(0, 2, 1, 3, 4)

            n_masked    = mask.sum()
            target_mean = (target * mask).sum() / n_masked
            var_target  = (((target - target_mean) ** 2) * mask).sum() / n_masked

            nmse    = (((recon - target) ** 2) * mask).sum() / n_masked / var_target
            mae_val = (torch.abs(recon - target) * mask).sum() / n_masked
            eps     = 1e-6
            mape_val = (torch.abs((recon - target) / (target.abs() + eps)) * mask).sum() / n_masked * 100.0
            raw_mse  = ((recon - target) ** 2 * mask).sum() / n_masked
            psnr_val = 10.0 * torch.log10(1.0 / (raw_mse + 1e-8))

            B, T, C, H, W = target.shape
            t_flat  = target.view(B * T, C, H, W).clamp(0, 1)
            r_flat  = recon.view(B * T, C, H, W).clamp(0, 1)
            ssim_val = ssim_fn(r_flat, t_flat, data_range=1.0, size_average=True)

            mse_all.append(nmse.item())
            mae_all.append(mae_val.item())
            mape_all.append(mape_val.item())
            psnr_all.append(psnr_val.item())
            ssim_all.append(ssim_val.item())

    mse_arr  = np.array(mse_all)
    rmse_arr = np.sqrt(mse_arr)
    mae_arr  = np.array(mae_all)
    mape_arr = np.array(mape_all)
    psnr_arr = np.array(psnr_all)
    ssim_arr = np.array(ssim_all)

    np.save(os.path.join(save_dir, "NMSE_masked.npy"),  mse_arr)
    np.save(os.path.join(save_dir, "MAE_masked.npy"),   mae_arr)
    np.save(os.path.join(save_dir, "MAPE_masked.npy"),  mape_arr)
    np.save(os.path.join(save_dir, "PSNR_frames.npy"),  psnr_arr)
    np.save(os.path.join(save_dir, "SSIM_frames.npy"),  ssim_arr)

    def _fmt(arr, decimals=6, unit=""):
        return (f"{np.nanmean(arr):.{decimals}f} ± {np.nanstd(arr):.{decimals}f}"
                f"  |  median {np.nanmedian(arr):.{decimals}f}{unit}")

    summary_path = os.path.join(save_dir, "summary_metrics.txt")
    with open(summary_path, "w") as f:
        f.write("TMNet MAE — test reconstruction metrics (ACDC)\n")
        f.write("=" * 56 + "\n")
        f.write(f"Fold:           {fold_idx}\n")
        f.write(f"Checkpoint:     {ckpt}\n")
        f.write(f"Model:          {type(condi).__name__}\n")
        f.write(f"Mask ratio:     {cfg.tmnet_mask_ratio}\n")
        f.write(f"N test samples: {len(mse_all)}\n\n")
        f.write(f"NMSE  (masked)   {_fmt(mse_arr)}\n")
        f.write(f"RMSE  (masked)   {_fmt(rmse_arr)}\n")
        f.write(f"MAE   (masked)   {_fmt(mae_arr)}\n")
        f.write(f"MAPE  (masked)   {_fmt(mape_arr, decimals=2, unit=' %')}\n")
        f.write(f"PSNR  (frames)   {_fmt(psnr_arr, decimals=2)}  dB\n")
        f.write(f"SSIM  (frames)   {_fmt(ssim_arr, decimals=4)}\n")

    print(
        f"[test_tmnet] "
        f"NMSE={np.nanmean(mse_arr):.4f}  RMSE={np.nanmean(rmse_arr):.4f}  "
        f"PSNR={np.nanmean(psnr_arr):.2f}dB  SSIM={np.nanmean(ssim_arr):.4f}"
        f"  — summary → {summary_path}"
    )


# =============================================================================
# Stage 2b — MopTR-style image prediction (primary ACDC variant)
# =============================================================================

def _build_mopTR_inputs(batch, device, cfg):
    """
    Build Iseq (B, 2, T, H, W) and Igt (B, 1, Q, H, W) for MopTR-style training.

    condi_type "1" → sagittal at cfg.sag_pos (default D//2 = 8)
    condi_type "2" → coronal  at cfg.cor_pos (default H//2 = 64)

    NOTE: still uses a fixed sag_pos shared across the batch — not yet
    migrated to per-patient reference-slice indexing (see _unpack_condi_inputs).
    """
    ref_raw, input_volume_list, current_volume_list, *_ = batch
    ref_vol = ref_raw.unsqueeze(1).to(device)

    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    sag_pos    = getattr(cfg, "sag_pos", _D_CARDIAC)
    cor_pos    = getattr(cfg, "cor_pos", _Y_CARDIAC)

    frames_list = []
    igt_list    = []

    if condi_type == "1":
        # Full short-axis slice: (B, 1, H, W) = (B, 1, 128, 128)
        # Center-crop to tmnet_img_size so backbone (stride [2,2,2]) yields (8,8)
        # features that fit the hardcoded Conv3d kernel (1, 8, 8).
        img_h, img_w = getattr(cfg, "tmnet_img_size", [64, 64])
        _H, _W = ref_vol.shape[3], ref_vol.shape[4]
        h0 = (_H - img_h) // 2;  w0 = (_W - img_w) // 2

        c_ref = ref_vol[:, :, sag_pos, h0:h0+img_h, w0:w0+img_w]
        for q in range(cfg.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1).to(device)[:, :, sag_pos, h0:h0+img_h, w0:w0+img_w]
            frames_list.append(torch.cat([c_t, c_ref], dim=1))
        Iseq = torch.stack(frames_list, dim=2).float()
        for i in range(cfg.tp):
            igt_list.append(current_volume_list[i].unsqueeze(1).to(device)[:, :, sag_pos, h0:h0+img_h, w0:w0+img_w])
    else:
        c_ref = ref_vol[:, :, :, cor_pos, :]
        for q in range(cfg.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1).to(device)[:, :, :, cor_pos, :]
            frames_list.append(torch.cat([c_t, c_ref], dim=1))
        Iseq = torch.stack(frames_list, dim=2).float()
        for i in range(cfg.tp):
            igt_list.append(current_volume_list[i].unsqueeze(1).to(device)[:, :, :, cor_pos, :])

    Igt = torch.stack(igt_list, dim=2).float()
    return Iseq, Igt


def train_tmnet_mopTR_style(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    device:   torch.device,
) -> None:

    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    # Use tmnet_img_size directly — for condi_type "1" the slice is center-cropped
    # in _build_mopTR_inputs, so img_size must match that crop, not vol_size.
    _img_size_cfg = getattr(cfg, "tmnet_img_size", None)
    if _img_size_cfg is not None:
        img_size = tuple(_img_size_cfg)
    else:
        vol_size = tuple(getattr(cfg, "vol_size", CANONICAL_SHAPE))
        img_size = (vol_size[1], vol_size[2]) if condi_type == "1" else (vol_size[0], vol_size[2])

    condi = TMNetEncoder(
        num_inputs=cfg.tmnet_num_frames, horizon=cfg.tmnet_horizon, in_channels=2,
        out_channels=cfg.tmnet_condi_channels, n_heads=cfg.tmnet_Tr_n_heads,
        enc_layers=cfg.tmnet_Tr_enc_layers, normalize_before=cfg.tmnet_Tr_norm_before,
        output_dim=cfg.tmnet_pre_latent_dim, condi_type=condi_type, device=device,
    ).to(device)

    model = MopTRTMNet(tm_net=condi, num_queries=cfg.tp, img_size=img_size).to(device)

    print(
        f"\n[TMNet-MopTR] TMNetEncoder + MopTRTMNet"
        f"  | img_size={img_size}  | num_queries={cfg.tp}"
        f"  | output_dim={cfg.tmnet_pre_latent_dim}"
        f"  (device={next(condi.parameters()).device})"
    )

    if getattr(cfg, "checkpoint", None):
        custom_load(condi, cfg.checkpoint, device)

    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_mopTR")
        run_dir = os.path.join(cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_mopTR")
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    train_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[0], nb_pred=cfg.tp, **ds_kwargs)
    valid_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[1], nb_pred=cfg.tp, valid=True, **ds_kwargs)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker, generator=_g)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker)

    optimizer     = build_optimizer(cfg, model.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True,
                                  delta=cfg.early_stopping_delta)
    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    print(f"\nStage 2b — TMNet MopTR-style pretraining  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet-MopTR] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        condi.train()
        model.train()
        ep_loss = ep_steps = 0

        for batch in tqdm(train_loader):
            Iseq, Igt = _build_mopTR_inputs(batch, device, cfg)
            optimizer.zero_grad()
            out  = model(Iseq, Igt)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            ep_loss  += loss.item()
            ep_steps += 1
            writer.add_scalar("tmnet_mopTR_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("tmnet_mopTR_train/epoch_loss", ep_avg, epoch)

        condi.eval()
        model.eval()
        val_loss = val_n = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                Iseq, Igt = _build_mopTR_inputs(batch, device, cfg)
                out       = model(Iseq, Igt)
                val_loss += out["loss"].item()
                val_n    += 1

        val_avg = val_loss / max(val_n, 1)
        writer.add_scalar("tmnet_mopTR_val/loss", val_avg, epoch)
        print(f"[TMNet-MopTR] train={ep_avg:.6f}  val={val_avg:.6f}")

        if val_avg < best_val_loss:
            print(f"[TMNet-MopTR] Improved {best_val_loss:.6f} → {val_avg:.6f} — saving.")
            best_val_loss = val_avg
            custom_save(condi, os.path.join(log_dir, "model_best_tmnet.pth"))
        else:
            print(f"[TMNet-MopTR] No improvement from {best_val_loss:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            print("[TMNet-MopTR] Early stopping triggered.")
            break

        print(f"[TMNet-MopTR] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nStage 2b done.  Best checkpoint: {log_dir}/model_best_tmnet.pth")
    writer.close()

    print("[TMNet-MopTR] Computing embedding statistics over full training set...")
    custom_load(condi, os.path.join(log_dir, "model_best_tmnet.pth"), device)
    condi.eval()

    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            Iseq, _ = _build_mopTR_inputs(batch, device, cfg)
            emb_list = model.encode(sag=Iseq)
            for emb in emb_list:
                emb_sum    += emb.sum().item()
                emb_sq_sum += emb.pow(2).sum().item()
                emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)
    stats_path = os.path.join(log_dir, "tmnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(f"[TMNet-MopTR] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  saved to {stats_path}")


# =============================================================================
# Aux-target helpers (for priormulti image / dvf variants)
# =============================================================================

def _phi_vec_gt_from_batch(batch, device) -> torch.Tensor:
    """Direction-aware phi target for train_tmnet_priormulti_dvf: (B, H, 2) of
    (cos, sin) per predicted future frame, via _get_phi_vec. See _get_phi_vec."""
    vol_files = batch[-1]   # list[H] of list[B]
    H = len(vol_files)

    phi_per_h = []
    for h in range(H):
        vecs = []
        for path in vol_files[h]:
            parts      = path.split("/")
            patient_id = parts[-2]
            t_idx      = int(parts[-1][2:-7])
            data_dir   = "/".join(parts[:-2])
            vecs.append(_get_phi_vec(patient_id, t_idx, data_dir))
        phi_per_h.append(torch.tensor(vecs, dtype=torch.float32, device=device))  # (B, 2)
    return torch.stack(phi_per_h, dim=1)   # (B, H, 2)


def _aux_targets_from_batch(batch, device) -> tuple:
    """Extract cardiac phi for every predicted future frame → phi_gt (B, H)."""
    vol_files = batch[-1]   # list[H] of list[B]
    H = len(vol_files)

    phi_per_h = []
    for h in range(H):
        phis = []
        for path in vol_files[h]:
            parts      = path.split("/")
            patient_id = parts[-2]
            t_idx      = int(parts[-1][2:-7])
            data_dir   = "/".join(parts[:-2])
            phis.append(_get_phi(patient_id, t_idx, data_dir))
        phi_per_h.append(torch.tensor(phis, dtype=torch.float32, device=device))
    phi_gt = torch.stack(phi_per_h, dim=1)   # (B, H)

    amps = []
    for path in vol_files[0]:
        parts      = path.split("/")
        patient_id = parts[-2]
        t_idx      = int(parts[-1][2:-7])
        data_dir   = "/".join(parts[:-2])
        amps.append(_get_amplitude(patient_id, t_idx, data_dir))
    amp_gt = torch.tensor(amps, dtype=torch.float32, device=device)

    return phi_gt, amp_gt


_priormulti_dindex_logged: set = set()   # module-level, patients already logged this run


def _build_priormulti_image_inputs(batch, device, cfg, orientation=None):
    """
    condi_type "1" (sagittal): per-sample D-index gathered via
    `get_reference_slice_index`, i.e. wherever THIS patient's saved
    reference slice (or centroid) actually lands in the canonical crop —
    accounts for resampling + clamping, not just a fixed CANONICAL_SHAPE[0]//2.

    condi_type "2" (coronal): unchanged, fixed `cor_pos` for every patient
    (heart-centered orientation only ever varies the D-axis reference slice
    per patient, not the H-axis coronal plane).

    `orientation`: {patient: entry} dict from `load_orientation_info` (or
    None to fall back to CANONICAL_SHAPE[0] // 2 for every patient, matching
    the old fixed-`_D_CARDIAC` behaviour). Once `orientation[p]` carries a
    precomputed `canonical_d_index` (written by
    `acdc_preprocessed_reference_grid.py`'s `update_orientation_csv`),
    `get_reference_slice_index` resolves each patient with a dict lookup —
    no per-call resampling — so this function stays cheap even at full
    batch/epoch scale.
    """
    ref_raw, input_volume_list, current_volume_list, vol_files = batch
    ref_vol = ref_raw.unsqueeze(1).to(device)          # (B, 1, D, H, W)
    B = ref_vol.shape[0]

    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    cor_pos    = getattr(cfg, "cor_pos", _Y_CARDIAC)

    if condi_type == "1":
        # Per-sample D-index: where THIS patient's reference slice actually
        # landed in the canonical crop (accounts for resampling + clamping).
        d_idx = _d_idx_from_batch(vol_files, cfg, orientation, device)
        b_idx = torch.arange(B, device=device)

        # Log each patient's mapped D-index once (first time seen), not every
        # batch — same info, but readable instead of spamming every step.
        patients = _patients_from_vol_files(vol_files)
        new_patients = [p for p in patients if p not in _priormulti_dindex_logged]
        if new_patients:
            for p in patients:
                if p in new_patients:
                    _priormulti_dindex_logged.add(p)
            # print(f"[priormulti_image] sagittal D-index — "
            #       f"{dict(zip(patients, d_idx.tolist()))}")

        img_h, img_w = getattr(cfg, "tmnet_img_size", [64, 64])
        _H, _W = ref_vol.shape[3], ref_vol.shape[4]
        h0 = (_H - img_h) // 2;  w0 = (_W - img_w) // 2

        # ref_vol[b_idx, :, d_idx, h0:h0+img_h, w0:w0+img_w] -> (B, 1, img_h, img_w)
        c_ref = ref_vol[b_idx, :, d_idx, h0:h0+img_h, w0:w0+img_w]

        past_list, future_list, igt_list = [], [], []
        for q in range(cfg.nb_inputs):
            v = input_volume_list[q].unsqueeze(1).to(device)
            c_t = v[b_idx, :, d_idx, h0:h0+img_h, w0:w0+img_w]
            past_list.append(torch.cat([c_t, c_ref], dim=1))
        for i in range(cfg.tp):
            v = current_volume_list[i].unsqueeze(1).to(device)
            c_f = v[b_idx, :, d_idx, h0:h0+img_h, w0:w0+img_w]
            future_list.append(torch.cat([c_f, c_ref], dim=1))
            igt_list.append(c_f)
    else:
        c_ref = ref_vol[:, :, :, cor_pos, :]
        past_list, future_list, igt_list = [], [], []
        for q in range(cfg.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1).to(device)[:, :, :, cor_pos, :]
            past_list.append(torch.cat([c_t, c_ref], dim=1))
        for i in range(cfg.tp):
            c_f = current_volume_list[i].unsqueeze(1).to(device)[:, :, :, cor_pos, :]
            future_list.append(torch.cat([c_f, c_ref], dim=1))
            igt_list.append(c_f)

    Ipast       = torch.stack(past_list,   dim=2).float()
    Ifuture_2ch = torch.stack(future_list, dim=2).float()
    Igt         = torch.stack(igt_list,    dim=2).float()
    Iref        = c_ref.float()
    return Iref, Ipast, Ifuture_2ch, Igt


# =============================================================================
# Stage 2b — DVF supervision + KL
# =============================================================================

def train_tmnet_priormulti_dvf(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    vm:       torch.nn.Module,
    device:   torch.device,
) -> None:
    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    _img_size_cfg = getattr(cfg, "tmnet_img_size", None)
    if _img_size_cfg is not None:
        img_h, img_w = _img_size_cfg[0], _img_size_cfg[1]
    else:
        vol_size = tuple(getattr(cfg, "vol_size", CANONICAL_SHAPE))
        img_h, img_w = (vol_size[1], vol_size[2]) if condi_type == "1" else (vol_size[0], vol_size[2])

    dec_layers         = getattr(cfg, "tmnet_Tr_dec_layers", 3)
    prior_type         = getattr(cfg, "tmnet_prior_type", "learned")
    kl_beta            = float(getattr(cfg, "tmnet_kl_beta", 1.0))
    kl_warmup_ep       = int(getattr(cfg, "tmnet_kl_warmup_epochs", 10))
    phi_loss_weight    = float(getattr(cfg, "tmnet_phi_loss_weight", 0.0))
    # Training always calls the model with Ifuture (posterior path, which sees the
    # ground-truth future frame), but inference/eval only ever uses the prior path
    # (Ifuture=None) -- so phi_head only ever learned to decode phase from features
    # with privileged future access, not from what's actually used at deployment.
    # Defaults to phi_loss_weight so it's on whenever the posterior-path phi loss is.
    phi_loss_weight_prior = float(getattr(cfg, "tmnet_phi_loss_weight_prior", phi_loss_weight))
    amp_loss_weight     = float(getattr(cfg, "tmnet_amp_loss_weight", 0.0))
    motion_loss_weight  = float(getattr(cfg, "tmnet_motion_loss_weight", 0.0))
    img_loss_weight     = float(getattr(cfg, "tmnet_img_loss_weight", 0.0))
    dvf_amp_loss_weight = float(getattr(cfg, "tmnet_dvf_amp_loss_weight", 0.0))
    lambda_phase_con    = float(getattr(cfg, "tmnet_lambda_phase_con", 0.0))
    phase_con_warmup_ep = int(getattr(cfg, "tmnet_phase_con_warmup_epochs", 10))
    phase_contrastive = None
    if lambda_phase_con > 0:
        phase_contrastive = PhaseContrastiveLoss(
            temperature=getattr(cfg, "tmnet_phase_con_temp", 0.1),
            kappa=getattr(cfg, "tmnet_phase_con_kappa", 8.0),
        )

    model = TMNet_Tr_priormulti_image(
        num_inputs=cfg.tmnet_num_frames, horizon=cfg.tmnet_horizon, in_channels=2,
        out_channels=cfg.tmnet_condi_channels, n_heads=cfg.tmnet_Tr_n_heads,
        enc_layers=cfg.tmnet_Tr_enc_layers, dec_layers=dec_layers,
        normalize_before=cfg.tmnet_Tr_norm_before, output_dim=cfg.tmnet_pre_latent_dim,
        rnn="transformer", condi_type=condi_type, prior_type=prior_type, device=device,
        img_h=img_h, img_w=img_w,
        use_phi_loss=phi_loss_weight > 0,
        phi_head_dim=2,   # (cos, sin) of raw phase — direction-aware, see _get_phi_vec
        use_amp_loss=amp_loss_weight > 0,
        use_motion_loss=motion_loss_weight > 0,
        use_phase_contrastive=lambda_phase_con > 0,
    ).to(device)

    if getattr(cfg, "checkpoint", None):
        custom_load(model, cfg.checkpoint, device)

    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_dvfsup2")
        run_dir = os.path.join(cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_dvfsup2")
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    train_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[0], nb_pred=cfg.tp, **ds_kwargs, cache_dir="/volatile/data/ACDC/cache_preprocessed")
    valid_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[1], nb_pred=cfg.tp, valid=True, **ds_kwargs, cache_dir="/volatile/data/ACDC/cache_preprocessed")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker, generator=_g)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker)

    cor_pos = getattr(cfg, "cor_pos", _Y_CARDIAC)
    sag_pos = getattr(cfg, "sag_pos", _D_CARDIAC)   # fallback only, see get_reference_slice_index

    # ---- orientation info (per-patient reference slice / centroid / rotation) ---
    # Loaded once here (not per-epoch) since reference_slices_path never
    # changes during training.
    orientation = load_orientation_info(cfg.reference_slices_path) if cfg.reference_slices_path else None

    optimizer     = build_optimizer(cfg, model.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True,
                                  delta=cfg.early_stopping_delta)

    def _dvf_slice(dvf: torch.Tensor, d_idx: torch.Tensor = None, cor_pos: int = None) -> torch.Tensor:
        """3-D DVF (B,3,D,H,W) → in-plane 2-D slice, center-cropped to (img_h, img_w).

        condi_type "1": `d_idx` is a per-sample (B,) LongTensor — each batch
        element's DVF is sliced at THAT patient's own mapped reference-slice
        D-index (computed via _d_idx_from_batch), matching the per-patient
        D-index _build_priormulti_image_inputs already uses for the image
        conditioning. Using a single shared scalar here (the old `sag_pos`
        behaviour) would silently misalign DVF supervision against image
        conditioning for every patient whose reference slice isn't at
        CANONICAL_SHAPE[0]//2.
        """
        if condi_type == "2":
            return dvf[:, [0, 2], :, cor_pos, :]
        else:
            B = dvf.shape[0]
            b_idx = torch.arange(B, device=dvf.device)
            # Gather each sample's own D-index first, then pick the H/W-motion
            # channels — two steps because mixing an integer-array dim (d_idx)
            # with a list-index dim ([1,2]) in one call reorders unpredictably.
            s = dvf[b_idx, :, d_idx][:, [1, 2]]          # (B, 2, H, W)
            _H, _W = s.shape[2], s.shape[3]
            h0 = (_H - img_h) // 2;  w0 = (_W - img_w) // 2
            return s[:, :, h0:h0+img_h, w0:w0+img_w]

    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    print(f"\nStage 2b — TMNet DVF-supervision + KL  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet-DVFSup2] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        model.train()

        beta = kl_beta * min(1.0, (epoch + 1) / max(kl_warmup_ep, 1))
        writer.add_scalar("tmnet_dvfsup2_train/kl_beta", beta, epoch)

        phase_con_weight = lambda_phase_con * min(1.0, (epoch + 1) / max(phase_con_warmup_ep, 1))
        writer.add_scalar("tmnet_dvfsup2_train/phase_con_weight", phase_con_weight, epoch)

        ep_dvf = ep_kl = ep_phi = ep_phi_prior = ep_amp = ep_motion = ep_img = ep_dvf_amp = 0
        ep_phase_con = ep_steps = 0

        for batch in tqdm(train_loader):
            Iref, Ipast, Ifuture_2ch, Igt = _build_priormulti_image_inputs(batch, device, cfg, orientation)

            ref_vol = batch[0].unsqueeze(1).to(device)
            # Per-sample D-index (same helper _build_priormulti_image_inputs
            # uses internally) so the DVF ground truth is sliced at the SAME
            # depth as the image conditioning, per patient.
            d_idx = _d_idx_from_batch(batch[3], cfg, orientation, device)
            with torch.no_grad():
                dvf_gt_list = [
                    _dvf_slice(vm(ref_vol, batch[2][t].unsqueeze(1).to(device)), d_idx=d_idx)
                    for t in range(cfg.tp)
                ]

            optimizer.zero_grad()
            DVF_seq, I_seq, kl_loss, phi_pred, amp_pred, motion_map_pred, z_contrast = model(Iref, Ipast, Ifuture_2ch)

            if epoch == restart_epoch and ep_steps == 0:
                print(f"[TMNet-DVFSup2][shapes] d_idx={d_idx.tolist()}  ref_vol={tuple(ref_vol.shape)}  "
                        f"Iref={tuple(Iref.shape)}  Ipast={tuple(Ipast.shape)}  "
                        f"Ifuture_2ch={tuple(Ifuture_2ch.shape) if Ifuture_2ch is not None else None}  "
                        f"dvf_gt={tuple(dvf_gt_list[0].shape)}  DVF_seq={tuple(DVF_seq.shape)}  "
                        f"z_contrast={tuple(z_contrast.shape) if z_contrast is not None else None}")
                if lambda_phase_con > 0:
                    print(f"[TMNet-DVFSup2] phase_contrastive enabled, lambda={lambda_phase_con}, "
                          f"warmup_epochs={phase_con_warmup_ep}")

            dvf_loss = torch.tensor(0.0, device=device)
            for t in range(cfg.tp):
                dvf_loss = dvf_loss + torch.nn.functional.mse_loss(DVF_seq[:, :, t], dvf_gt_list[t])
            dvf_loss = dvf_loss / cfg.tp

            loss = dvf_loss + (beta * kl_loss if kl_loss is not None else 0.0)

            # phi_gt (cos, sin target) is needed by both the phi-regression loss and
            # the phase-contrastive loss, so compute it once if either is active.
            need_phi_gt = (phi_pred is not None) or (amp_pred is not None) or \
                          (lambda_phase_con > 0 and z_contrast is not None)
            phi_gt = None
            if (phi_pred is not None) or (lambda_phase_con > 0 and z_contrast is not None):
                phi_gt = _phi_vec_gt_from_batch(batch, device)

            # Second ("prior path") forward pass, shared by phi_head and the
            # phase-contrastive loss — both need supervision on the representation
            # actually used at inference (no Ifuture), not just the posterior's.
            need_prior_pass = (phi_pred is not None) or (lambda_phase_con > 0 and z_contrast is not None)
            phi_pred_prior = z_contrast_prior = None
            if need_prior_pass:
                _, _, _, phi_pred_prior, _, _, z_contrast_prior = model(Iref, Ipast)

            phi_loss = phi_loss_prior = amp_loss = motion_loss = torch.tensor(0.0, device=device)
            if phi_pred is not None:
                phi_loss = phi_angular_loss(phi_pred, phi_gt)
                loss = loss + phi_loss_weight * phi_loss

                phi_loss_prior = phi_angular_loss(phi_pred_prior, phi_gt)
                loss = loss + phi_loss_weight_prior * phi_loss_prior
            if amp_pred is not None:
                _, amp_gt = _aux_targets_from_batch(batch, device)
                amp_loss = torch.nn.functional.mse_loss(amp_pred, amp_gt)
                loss = loss + amp_loss_weight * amp_loss
            if motion_map_pred is not None:
                motion_target = (Ipast[:, 0:1, -1] - Ipast[:, 1:2, -1]).abs()
                motion_target_pooled = torch.nn.functional.adaptive_avg_pool2d(
                    motion_target, (model._motion_pool_h, model._motion_pool_w)
                )
                motion_loss = torch.nn.functional.mse_loss(motion_map_pred, motion_target_pooled)
                loss = loss + motion_loss_weight * motion_loss

            img_loss = dvf_amp_loss = torch.tensor(0.0, device=device)
            if img_loss_weight > 0 and I_seq is not None and Igt is not None:
                for t in range(cfg.tp):
                    img_loss = img_loss + torch.nn.functional.mse_loss(I_seq[:, :, t], Igt[:, :, t])
                img_loss = img_loss / cfg.tp
                loss = loss + img_loss_weight * img_loss
            if dvf_amp_loss_weight > 0:
                for t in range(cfg.tp):
                    pred_a = DVF_seq[:, :, t].pow(2).sum(1).sqrt().mean(dim=(-2, -1))
                    gt_a   = dvf_gt_list[t].pow(2).sum(1).sqrt().mean(dim=(-2, -1))
                    dvf_amp_loss = dvf_amp_loss + torch.nn.functional.mse_loss(pred_a, gt_a)
                dvf_amp_loss = dvf_amp_loss / cfg.tp
                loss = loss + dvf_amp_loss_weight * dvf_amp_loss

            phase_cl_loss = torch.tensor(0.0, device=device)
            if lambda_phase_con > 0 and z_contrast_prior is not None:
                phi_gt_flat  = phi_gt.reshape(-1, 2)                                  # (B*horizon, 2)
                z_prior_flat = z_contrast_prior.reshape(-1, z_contrast_prior.shape[-1])  # (B*horizon, 32)
                phase_cl_loss = phase_contrastive(z_prior_flat, phi_gt_flat[:, 0], phi_gt_flat[:, 1])
                loss = loss + phase_con_weight * phase_cl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            ep_dvf       += dvf_loss.item()
            ep_kl        += kl_loss.item() if kl_loss is not None else 0.0
            ep_phi       += phi_loss.item()
            ep_phi_prior += phi_loss_prior.item()
            ep_amp       += amp_loss.item()
            ep_motion    += motion_loss.item()
            ep_img       += img_loss.item()
            ep_dvf_amp   += dvf_amp_loss.item()
            ep_phase_con += phase_cl_loss.item()
            ep_steps     += 1
            writer.add_scalar("tmnet_dvfsup2_train/dvf_loss", dvf_loss.item(), global_step)
            writer.add_scalar("tmnet_dvfsup2_train/kl_loss",  kl_loss.item() if kl_loss is not None else 0.0, global_step)
            writer.add_scalar("tmnet_dvfsup2_train/phase_con_loss", phase_cl_loss.item(), global_step)
            global_step += 1

        ep_dvf_avg       = ep_dvf       / max(ep_steps, 1)
        ep_kl_avg        = ep_kl        / max(ep_steps, 1)
        ep_phi_avg       = ep_phi       / max(ep_steps, 1)
        ep_phi_prior_avg = ep_phi_prior / max(ep_steps, 1)
        ep_amp_avg       = ep_amp       / max(ep_steps, 1)
        ep_motion_avg    = ep_motion    / max(ep_steps, 1)
        ep_img_avg       = ep_img       / max(ep_steps, 1)
        ep_dvf_amp_avg   = ep_dvf_amp   / max(ep_steps, 1)
        ep_phase_con_avg = ep_phase_con / max(ep_steps, 1)
        writer.add_scalar("tmnet_dvfsup2_train/epoch_dvf_loss",    ep_dvf_avg,     epoch)
        writer.add_scalar("tmnet_dvfsup2_train/epoch_kl_loss",     ep_kl_avg,      epoch)
        writer.add_scalar("tmnet_dvfsup2_train/epoch_phi_loss",    ep_phi_avg,     epoch)
        writer.add_scalar("tmnet_dvfsup2_train/epoch_phi_loss_prior", ep_phi_prior_avg, epoch)
        writer.add_scalar("tmnet_dvfsup2_train/epoch_amp_loss",    ep_amp_avg,     epoch)
        writer.add_scalar("tmnet_dvfsup2_train/epoch_motion_loss", ep_motion_avg,  epoch)
        writer.add_scalar("tmnet_dvfsup2_train/epoch_img_loss",    ep_img_avg,     epoch)
        writer.add_scalar("tmnet_dvfsup2_train/epoch_dvf_amp_loss", ep_dvf_amp_avg, epoch)
        writer.add_scalar("tmnet_dvfsup2_train/epoch_phase_con_loss", ep_phase_con_avg, epoch)

        model.eval()
        val_dvf = val_phi = val_n = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                Iref, Ipast, _, _ = _build_priormulti_image_inputs(batch, device, cfg, orientation)
                ref_vol = batch[0].unsqueeze(1).to(device)
                d_idx = _d_idx_from_batch(batch[3], cfg, orientation, device)
                dvf_gt_list = [
                    _dvf_slice(
                        vm(ref_vol, batch[2][t].unsqueeze(1).to(device)),
                        d_idx=d_idx,
                        cor_pos=cor_pos,
                    )
                    for t in range(cfg.tp)
                ]
                DVF_seq, _, _, phi_pred, _, _, _ = model(Iref, Ipast)
                r = torch.tensor(0.0, device=device)
                for t in range(cfg.tp):
                    r = r + torch.nn.functional.mse_loss(DVF_seq[:, :, t], dvf_gt_list[t])
                val_dvf += (r / cfg.tp).item()
                if phi_pred is not None:
                    phi_gt = _phi_vec_gt_from_batch(batch, device)
                    val_phi += phi_angular_loss(phi_pred, phi_gt).item()
                val_n   += 1

        val_avg     = val_dvf / max(val_n, 1)
        val_phi_avg = val_phi / max(val_n, 1)
        writer.add_scalar("tmnet_dvfsup2_val/dvf_loss", val_avg, epoch)
        writer.add_scalar("tmnet_dvfsup2_val/phi_loss", val_phi_avg, epoch)
        print(f"[TMNet-DVFSup2] train_dvf={ep_dvf_avg:.6f}  train_kl={ep_kl_avg:.6f}  "
              f"train_phi={ep_phi_avg:.6f}  train_phi_prior={ep_phi_prior_avg:.6f}  "
              f"train_phase_con={ep_phase_con_avg:.6f}  "
              f"train_img={ep_img_avg:.6f}  train_dvf_amp={ep_dvf_amp_avg:.6f}  "
              f"val_dvf={val_avg:.6f}  val_phi={val_phi_avg:.6f}")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            custom_save(model, os.path.join(log_dir, "model_best_tmnet.pth"))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            break

        print(f"[TMNet-DVFSup2] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    writer.close()

    custom_load(model, os.path.join(log_dir, "model_best_tmnet.pth"), device)
    model.eval()
    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            _, Ipast, _, _ = _build_priormulti_image_inputs(batch, device, cfg, orientation)
            emb_list = model.encode(Ipast)
            for emb in emb_list:
                emb_sum    += emb.sum().item()
                emb_sq_sum += emb.pow(2).sum().item()
                emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)
    stats_path = os.path.join(log_dir, "tmnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(f"[TMNet-DVFSup2] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  saved to {stats_path}")
    
# =============================================================================
# Stage 2b — DVF supervision + KL: test mode
# =============================================================================

def test_tmnet_priormulti_dvf(
    cfg,
    fold:     list,
    fold_idx: str,
    dir_name: str,
    vm:       torch.nn.Module,
    device:   torch.device,
) -> None:
    """
    Evaluate a trained tmnet_dvfsup2 (TMNet_Tr_priormulti_image) checkpoint
    on the test set:
      - DVF MSE against the VoxelMorph ground-truth deformation field.
      - Image-reconstruction quality (MSE/MAE/NCC/PSNR/SSIM) of the warped
        prediction I_seq against the ground-truth future frame.

    Results saved under <logging_dir>/<dir_name>/test_tmnet_dvfsup2/<fold_idx>/.
    """
    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    _img_size_cfg = getattr(cfg, "tmnet_img_size", None)
    if _img_size_cfg is not None:
        img_h, img_w = _img_size_cfg[0], _img_size_cfg[1]
    else:
        vol_size = tuple(getattr(cfg, "vol_size", CANONICAL_SHAPE))
        img_h, img_w = (vol_size[1], vol_size[2]) if condi_type == "1" else (vol_size[0], vol_size[2])

    dec_layers         = getattr(cfg, "tmnet_Tr_dec_layers", 3)
    prior_type         = getattr(cfg, "tmnet_prior_type", "learned")
    phi_loss_weight    = float(getattr(cfg, "tmnet_phi_loss_weight", 0.0))
    amp_loss_weight    = float(getattr(cfg, "tmnet_amp_loss_weight", 0.0))
    motion_loss_weight = float(getattr(cfg, "tmnet_motion_loss_weight", 0.0))
    lambda_phase_con    = float(getattr(cfg, "tmnet_lambda_phase_con", 0.0))

    model = TMNet_Tr_priormulti_image(
        num_inputs=cfg.tmnet_num_frames, horizon=cfg.tmnet_horizon, in_channels=2,
        out_channels=cfg.tmnet_condi_channels, n_heads=cfg.tmnet_Tr_n_heads,
        enc_layers=cfg.tmnet_Tr_enc_layers, dec_layers=dec_layers,
        normalize_before=cfg.tmnet_Tr_norm_before, output_dim=cfg.tmnet_pre_latent_dim,
        rnn="transformer", condi_type=condi_type, prior_type=prior_type, device=device,
        img_h=img_h, img_w=img_w,
        use_phi_loss=phi_loss_weight > 0,
        phi_head_dim=2,   # (cos, sin) of raw phase — direction-aware, see _get_phi_vec
        use_amp_loss=amp_loss_weight > 0,
        use_motion_loss=motion_loss_weight > 0,
        use_phase_contrastive=lambda_phase_con > 0,
    ).to(device)

    log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_dvfsup2")
    ckpt    = os.path.join(log_dir, "model_best_tmnet.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[test_tmnet_dvfsup2] Checkpoint not found: {ckpt}")
    custom_load(model, ckpt, device)
    model.eval()

    cor_pos = getattr(cfg, "cor_pos", _Y_CARDIAC)
    sag_pos = getattr(cfg, "sag_pos", _D_CARDIAC)   # fallback only, see get_reference_slice_index

    def _dvf_slice(dvf: torch.Tensor, d_idx: torch.Tensor = None) -> torch.Tensor:
        """3-D DVF (B,3,D,H,W) → in-plane 2-D slice, center-cropped to (img_h, img_w).

        Same per-sample gather as train_tmnet_priormulti_dvf's _dvf_slice —
        see that docstring for why a shared scalar sag_pos would misalign
        supervision against the per-patient image conditioning.
        """
        if condi_type == "2":
            return dvf[:, [0, 2], :, cor_pos, :]
        else:
            B = dvf.shape[0]
            b_idx = torch.arange(B, device=dvf.device)
            s = dvf[b_idx, :, d_idx][:, [1, 2]]        # (B, 2, H, W)
            _H, _W = s.shape[2], s.shape[3]
            h0 = (_H - img_h) // 2;  w0 = (_W - img_w) // 2
            return s[:, :, h0:h0+img_h, w0:w0+img_w]   # (B, 2, img_h, img_w)

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    test_set = ACDC_4D_Dataset(cfg.data_dir, sequence_list=fold, nb_pred=cfg.tp,
                               nb_inputs=cfg.nb_inputs, test=True,
                               **ds_kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    save_dir = os.path.join(cfg.logging_dir, dir_name, "test_tmnet_dvfsup2", fold_idx)
    cond_mkdir(save_dir)

    test_patients = {seq.split("/")[0] if "/" in seq else seq[:10] for seq in fold}
    with open(os.path.join(save_dir, "patients_test.txt"), "w") as f:
        for p in sorted(test_patients):
            f.write(p + "\n")

    dvf_mse_all, img_mse_all, img_mae_all, ncc_all, psnr_all, ssim_all = [], [], [], [], [], []

    from skimage.metrics import structural_similarity as ss

    orientation = load_orientation_info(cfg.reference_slices_path) if cfg.reference_slices_path else None

    with torch.no_grad():
        for batch in tqdm(test_loader):
            Iref, Ipast, _, Igt = _build_priormulti_image_inputs(batch, device, cfg, orientation)
            ref_vol = batch[0].unsqueeze(1).to(device)
            d_idx = _d_idx_from_batch(batch[3], cfg, orientation, device)
            dvf_gt_list = [
                _dvf_slice(vm(ref_vol, batch[2][t].unsqueeze(1).to(device)), d_idx=d_idx)
                for t in range(cfg.tp)
            ]

            DVF_seq, I_seq, _, _, _, _, _ = model(Iref, Ipast)

            for t in range(cfg.tp):
                dvf_mse = torch.nn.functional.mse_loss(DVF_seq[:, :, t], dvf_gt_list[t])
                dvf_mse_all.append(dvf_mse.item())

                img_pred = I_seq[:, :, t].clamp(0, 1)
                img_gt   = Igt[:, :, t].clamp(0, 1)

                img_mse  = torch.nn.functional.mse_loss(img_pred, img_gt)
                img_mae  = torch.nn.functional.l1_loss(img_pred, img_gt)
                ncc_val  = ncc_loss(img_pred.unsqueeze(2), img_gt.unsqueeze(2), device)
                psnr_val = 10.0 * torch.log10(1.0 / (img_mse + 1e-8))

                pred_np = img_pred.squeeze().cpu().numpy()
                gt_np   = img_gt.squeeze().cpu().numpy()
                ssim_val = ss(gt_np, pred_np, data_range=1.0)

                img_mse_all.append(img_mse.item())
                img_mae_all.append(img_mae.item())
                ncc_all.append(ncc_val.item())
                psnr_all.append(psnr_val.item())
                ssim_all.append(ssim_val)

    dvf_mse_arr  = np.array(dvf_mse_all)
    img_mse_arr  = np.array(img_mse_all)
    img_rmse_arr = np.sqrt(img_mse_arr)
    img_mae_arr  = np.array(img_mae_all)
    ncc_arr      = np.array(ncc_all)
    psnr_arr     = np.array(psnr_all)
    ssim_arr     = np.array(ssim_all)

    np.save(os.path.join(save_dir, "DVF_MSE.npy"), dvf_mse_arr)
    np.save(os.path.join(save_dir, "IMG_MSE.npy"), img_mse_arr)
    np.save(os.path.join(save_dir, "IMG_MAE.npy"), img_mae_arr)
    np.save(os.path.join(save_dir, "NCC.npy"),     ncc_arr)
    np.save(os.path.join(save_dir, "PSNR.npy"),    psnr_arr)
    np.save(os.path.join(save_dir, "SSIM.npy"),    ssim_arr)

    def _fmt(arr, decimals=6, unit=""):
        return (f"{np.nanmean(arr):.{decimals}f} ± {np.nanstd(arr):.{decimals}f}"
                f"  |  median {np.nanmedian(arr):.{decimals}f}{unit}")

    summary_path = os.path.join(save_dir, "summary_metrics.txt")
    with open(summary_path, "w") as f:
        f.write("TMNet DVF-supervision (priormulti_dvf) — test metrics (ACDC)\n")
        f.write("=" * 64 + "\n")
        f.write(f"Fold:           {fold_idx}\n")
        f.write(f"Checkpoint:     {ckpt}\n")
        f.write(f"N test samples: {len(dvf_mse_all)}\n\n")
        f.write(f"DVF MSE          {_fmt(dvf_mse_arr)}\n")
        f.write(f"Image MSE        {_fmt(img_mse_arr)}\n")
        f.write(f"Image RMSE       {_fmt(img_rmse_arr)}\n")
        f.write(f"Image MAE        {_fmt(img_mae_arr)}\n")
        f.write(f"NCC              {_fmt(ncc_arr, decimals=4)}\n")
        f.write(f"PSNR             {_fmt(psnr_arr, decimals=2)}  dB\n")
        f.write(f"SSIM             {_fmt(ssim_arr, decimals=4)}\n")

    print(
        f"[test_tmnet_dvfsup2] "
        f"DVF_MSE={np.nanmean(dvf_mse_arr):.6f}  IMG_MSE={np.nanmean(img_mse_arr):.6f}  "
        f"NCC={np.nanmean(ncc_arr):.4f}  PSNR={np.nanmean(psnr_arr):.2f}dB  SSIM={np.nanmean(ssim_arr):.4f}"
        f"  — summary → {summary_path}"
    )


# =============================================================================
# Stage 2b — image supervision (prior/posterior + KL)
# =============================================================================

def phi_angular_loss(phi_pred, phi_gt, eps=1e-7):
    """
    phi_pred, phi_gt: (..., 2) unnormalized (cos, sin) pairs.
    Returns 1 - cos(angle_pred - angle_gt), averaged. Bounded in [0, 2],
    invariant to prediction magnitude in a way plain MSE is not, and its
    gradient always pushes toward the correct *direction* rather than
    toward zero.
    """
    pred_n = F.normalize(phi_pred, dim=-1, eps=eps)
    gt_n   = F.normalize(phi_gt,   dim=-1, eps=eps)
    cos_err = (pred_n * gt_n).sum(-1)          # cos(Δφ), in [-1, 1]
    return (1.0 - cos_err).mean()

def train_tmnet_priormulti_image(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    device:   torch.device,
) -> None:
    vol_size   = tuple(getattr(cfg, "vol_size", CANONICAL_SHAPE))
    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    img_h, img_w = (
        (vol_size[1], vol_size[2]) if condi_type == "1"
        else (vol_size[0], vol_size[2])
    )

    dec_layers      = getattr(cfg, "tmnet_Tr_dec_layers", 3)
    prior_type      = getattr(cfg, "tmnet_prior_type", "learned")
    img_loss        = getattr(cfg, "tmnet_img_loss", "mse")
    kl_beta         = float(getattr(cfg, "tmnet_kl_beta", 1.0))
    kl_warmup_ep    = int(getattr(cfg, "tmnet_kl_warmup_epochs", 10))
    phi_loss_weight = float(getattr(cfg, "tmnet_phi_loss_weight", 0.0))
    amp_loss_weight = float(getattr(cfg, "tmnet_amp_loss_weight", 0.0))

    model = TMNet_Tr_priormulti_image(
        num_inputs=cfg.tmnet_num_frames, horizon=cfg.tmnet_horizon, in_channels=2,
        out_channels=cfg.tmnet_condi_channels, n_heads=cfg.tmnet_Tr_n_heads,
        enc_layers=cfg.tmnet_Tr_enc_layers, dec_layers=dec_layers,
        normalize_before=cfg.tmnet_Tr_norm_before, output_dim=cfg.tmnet_pre_latent_dim,
        rnn="transformer", condi_type=condi_type, prior_type=prior_type, device=device,
        img_h=img_h, img_w=img_w,
        use_phi_loss=phi_loss_weight > 0, use_amp_loss=amp_loss_weight > 0,
    ).to(device)

    if getattr(cfg, "checkpoint", None):
        custom_load(model, cfg.checkpoint, device)

    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_imgsup")
        run_dir = os.path.join(cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_imgsup")
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    train_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[0], nb_pred=cfg.tp, **ds_kwargs)
    valid_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[1], nb_pred=cfg.tp, valid=True, **ds_kwargs)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker, generator=_g)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker)

    # ---- orientation info (per-patient reference slice / centroid / rotation) ---
    # Loaded once here (not per-epoch) since reference_slices_path never
    # changes during training. Without this, _build_priormulti_image_inputs
    # silently falls back to CANONICAL_SHAPE[0]//2 for every patient.
    orientation = load_orientation_info(cfg.reference_slices_path) if cfg.reference_slices_path else None

    optimizer     = build_optimizer(cfg, model.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True,
                                  delta=cfg.early_stopping_delta)
    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")
    tf_prob       = float(getattr(cfg, "tmnet_teacher_forcing_prob", 1.0))

    print(f"\nStage 2b — TMNet image-supervision training  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet-ImgSup] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        model.train()

        beta = kl_beta * min(1.0, (epoch + 1) / max(kl_warmup_ep, 1))
        writer.add_scalar("tmnet_imgsup_train/kl_beta", beta, epoch)
        ep_recon = ep_kl = ep_steps = 0

        for batch in tqdm(train_loader):
            Iref, Ipast, Ifuture_2ch, Igt = _build_priormulti_image_inputs(batch, device, cfg, orientation)
            optimizer.zero_grad()

            use_tf = torch.rand(1).item() < tf_prob
            DVF_seq, I_pred, kl_loss, phi_pred, amp_pred = model(Iref, Ipast, Ifuture_2ch if use_tf else None)

            recon_loss = torch.tensor(0.0, device=device)
            for t in range(cfg.tp):
                pred_t = I_pred[:, :, t, :, :]
                gt_t   = Igt[:, :, t, :, :]
                if img_loss == "ncc":
                    recon_loss = recon_loss + (1 + ncc_loss(pred_t.unsqueeze(2), gt_t.unsqueeze(2), device))
                else:
                    recon_loss = recon_loss + torch.nn.functional.mse_loss(pred_t, gt_t)
            recon_loss = recon_loss / cfg.tp

            loss = recon_loss + (beta * kl_loss if kl_loss is not None else 0.0)

            phi_loss = amp_loss = torch.tensor(0.0, device=device)
            if phi_pred is not None or amp_pred is not None:
                phi_gt, amp_gt = _aux_targets_from_batch(batch, device)
                if phi_pred is not None:
                    phi_loss = phi_angular_loss(phi_pred, phi_gt)
                    loss = loss + phi_loss_weight * phi_loss
                if amp_pred is not None:
                    amp_loss = torch.nn.functional.mse_loss(amp_pred, amp_gt)
                    loss = loss + amp_loss_weight * amp_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            ep_recon += recon_loss.item()
            ep_kl    += kl_loss.item() if kl_loss is not None else 0.0
            ep_steps += 1
            writer.add_scalar("tmnet_imgsup_train/recon_loss", recon_loss.item(), global_step)
            global_step += 1

        ep_recon_avg = ep_recon / max(ep_steps, 1)
        ep_kl_avg    = ep_kl    / max(ep_steps, 1)
        writer.add_scalar("tmnet_imgsup_train/epoch_recon_loss", ep_recon_avg, epoch)
        writer.add_scalar("tmnet_imgsup_train/epoch_kl_loss",    ep_kl_avg,    epoch)

        model.eval()
        val_recon = val_n = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                Iref, Ipast, _, Igt = _build_priormulti_image_inputs(batch, device, cfg, orientation)
                DVF_seq, I_pred, _, _, _ = model(Iref, Ipast)
                r = torch.tensor(0.0, device=device)
                for t in range(cfg.tp):
                    pred_t = I_pred[:, :, t, :, :]
                    gt_t   = Igt[:, :, t, :, :]
                    if img_loss == "ncc":
                        r = r + (1 + ncc_loss(pred_t.unsqueeze(2), gt_t.unsqueeze(2), device))
                    else:
                        r = r + torch.nn.functional.mse_loss(pred_t, gt_t)
                val_recon += (r / cfg.tp).item()
                val_n     += 1

        val_recon_avg = val_recon / max(val_n, 1)
        writer.add_scalar("tmnet_imgsup_val/recon_loss", val_recon_avg, epoch)
        print(f"[TMNet-ImgSup] train_recon={ep_recon_avg:.6f}  train_kl={ep_kl_avg:.6f}  val_recon={val_recon_avg:.6f}")

        if val_recon_avg < best_val_loss:
            best_val_loss = val_recon_avg
            custom_save(model, os.path.join(log_dir, "model_best_tmnet.pth"))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_recon_avg)

        early_stopper(val_recon_avg)
        if early_stopper.early_stop:
            break

        print(f"[TMNet-ImgSup] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    writer.close()

    custom_load(model, os.path.join(log_dir, "model_best_tmnet.pth"), device)
    model.eval()
    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            _, Ipast, _, _ = _build_priormulti_image_inputs(batch, device, cfg, orientation)
            emb_list = model.encode(Ipast)
            for emb in emb_list:
                emb_sum    += emb.sum().item()
                emb_sq_sum += emb.pow(2).sum().item()
                emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)
    stats_path = os.path.join(log_dir, "tmnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(f"[TMNet-ImgSup] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  saved to {stats_path}")


# =============================================================================
# Stage 2b — contrastive pretraining with cardiac phase
# =============================================================================

def _phi_from_batch(batch, device: torch.device) -> torch.Tensor:
    """Extract cardiac phi ∈ [0, 0.5] for the last input frame of each batch element."""
    vol_files = batch[-1]
    paths     = vol_files[0]
    phis = []
    for path in paths:
        parts      = path.split("/")
        patient_id = parts[-2]
        t_idx      = int(parts[-1][2:-7]) - 1
        data_dir   = "/".join(parts[:-2])
        phis.append(_get_phi(patient_id, t_idx, data_dir))
    return torch.tensor(phis, dtype=torch.float32, device=device)


def train_tmnet_contrastive(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    device:   torch.device,
) -> None:

    condi = TMNetEncoder(
        num_inputs=cfg.tmnet_num_frames, horizon=cfg.tmnet_horizon, in_channels=2,
        out_channels=cfg.tmnet_condi_channels, n_heads=cfg.tmnet_Tr_n_heads,
        enc_layers=cfg.tmnet_Tr_enc_layers, normalize_before=cfg.tmnet_Tr_norm_before,
        output_dim=cfg.tmnet_pre_latent_dim,
        condi_type=getattr(cfg, "tmnet_condi_type", "2"), device=device,
    ).to(device)

    proj_dim    = int(getattr(cfg, "tmnet_contrast_proj_dim",    128))
    temperature = float(getattr(cfg, "tmnet_contrast_temperature", 0.07))
    tau_pos     = float(getattr(cfg, "tmnet_contrast_tau_pos",    0.05))
    tau_neg     = float(getattr(cfg, "tmnet_contrast_tau_neg",    0.15))

    model = PhiContrastiveTMNet(
        tm_net=condi, proj_dim=proj_dim, temperature=temperature,
        tau_pos=tau_pos, tau_neg=tau_neg,
    ).to(device)

    if getattr(cfg, "checkpoint", None):
        custom_load(condi, cfg.checkpoint, device)

    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_contrast")
        run_dir = os.path.join(cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_contrast")
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    train_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[0], nb_pred=cfg.tp, **ds_kwargs)
    valid_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[1], nb_pred=cfg.tp, valid=True, **ds_kwargs)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker, generator=_g)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker)

    optimizer     = build_optimizer(cfg, model.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True,
                                  delta=cfg.early_stopping_delta)
    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    print(f"\nStage 2b — TMNet contrastive pretraining  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet-Contrast] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        condi.train()
        model.train()
        ep_loss = ep_n_anchors = ep_frac_pos = ep_steps = 0

        for batch in tqdm(train_loader):
            Iseq = _build_mopTR_inputs(batch, device, cfg)[0]
            phi  = _phi_from_batch(batch, device)
            optimizer.zero_grad()
            out  = model(Iseq, phi)
            loss = out["loss"]

            if out["n_anchors"] == 0:
                ep_steps += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            ep_loss      += loss.item()
            ep_n_anchors += out["n_anchors"]
            ep_frac_pos  += out["frac_pos"]
            ep_steps     += 1
            writer.add_scalar("tmnet_contrast_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("tmnet_contrast_train/epoch_loss", ep_avg, epoch)

        condi.eval()
        model.eval()
        val_loss = val_n = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                Iseq = _build_mopTR_inputs(batch, device, cfg)[0]
                phi  = _phi_from_batch(batch, device)
                out  = model(Iseq, phi)
                if out["n_anchors"] > 0:
                    val_loss += out["loss"].item()
                    val_n    += 1

        val_avg = val_loss / max(val_n, 1)
        writer.add_scalar("tmnet_contrast_val/loss", val_avg, epoch)
        print(f"[TMNet-Contrast] train={ep_avg:.6f}  val={val_avg:.6f}")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            custom_save(condi, os.path.join(log_dir, "model_best_tmnet.pth"))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            break

        print(f"[TMNet-Contrast] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    writer.close()

    custom_load(condi, os.path.join(log_dir, "model_best_tmnet.pth"), device)
    condi.eval()
    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            Iseq, _ = _build_mopTR_inputs(batch, device, cfg)
            emb_list = model.encode(sag=Iseq)
            for emb in emb_list:
                emb_sum    += emb.sum().item()
                emb_sq_sum += emb.pow(2).sum().item()
                emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)
    stats_path = os.path.join(log_dir, "tmnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(f"[TMNet-Contrast] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  saved to {stats_path}")


# =============================================================================
# Stage 2b — DVF-slice supervision (no cache)
# =============================================================================

def train_tmnet_dvfsup(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    vm:       torch.nn.Module,
    device:   torch.device,
) -> None:
    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    dvf_h, dvf_w = tuple(cfg.tmnet_img_size)
    cor_pos = getattr(cfg, "cor_pos", _Y_CARDIAC)
    sag_pos = getattr(cfg, "sag_pos", _D_CARDIAC)

    condi = TMNetEncoder(
        num_inputs=cfg.tmnet_num_frames, horizon=cfg.tmnet_horizon, in_channels=2,
        out_channels=cfg.tmnet_condi_channels, n_heads=cfg.tmnet_Tr_n_heads,
        enc_layers=cfg.tmnet_Tr_enc_layers, normalize_before=cfg.tmnet_Tr_norm_before,
        output_dim=cfg.tmnet_pre_latent_dim, condi_type=condi_type, device=device,
    ).to(device)

    model = DVFSupTMNet(tm_net=condi, dvf_h=dvf_h, dvf_w=dvf_w).to(device)

    if getattr(cfg, "checkpoint", None):
        custom_load(condi, cfg.checkpoint, device)

    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_dvfsup")
        run_dir = os.path.join(cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_dvfsup")
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    train_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[0], nb_pred=cfg.tp, **ds_kwargs)
    valid_set = ACDC_4D_Dataset(cfg.data_dir, nb_inputs=cfg.nb_inputs, sequence_list=folds[1], nb_pred=cfg.tp, valid=True, **ds_kwargs)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker, generator=_g)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker)

    optimizer     = build_optimizer(cfg, model.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True,
                                  delta=cfg.early_stopping_delta)
    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    def _dvf_slice(dvf: torch.Tensor) -> torch.Tensor:
        """3-D DVF (B,3,D,H,W) → in-plane 2-D slice at the mid-cardiac plane."""
        if condi_type == "2":
            return dvf[:, :, :, cor_pos, :]
        else:
            return dvf[:, :, sag_pos, :, :]

    print(f"\nStage 2b — TMNet DVF-slice supervision  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet-DVFSup] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        model.train()
        ep_loss = ep_steps = 0

        for batch in tqdm(train_loader):
            frames = _unpack_condi_inputs(batch, device, cfg)
            ref_vol    = batch[0].unsqueeze(1).to(device)
            target_vol = batch[2][0].unsqueeze(1).to(device)
            with torch.no_grad():
                dvf_gt = _dvf_slice(vm(ref_vol, target_vol))

            optimizer.zero_grad()
            out  = model(frames, dvf_gt=dvf_gt)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            ep_loss  += loss.item()
            ep_steps += 1
            writer.add_scalar("tmnet_dvfsup_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("tmnet_dvfsup_train/epoch_loss", ep_avg, epoch)

        model.eval()
        val_loss = val_n = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                frames     = _unpack_condi_inputs(batch, device, cfg)
                ref_vol    = batch[0].unsqueeze(1).to(device)
                target_vol = batch[2][0].unsqueeze(1).to(device)
                dvf_gt     = _dvf_slice(vm(ref_vol, target_vol))
                out        = model(frames, dvf_gt=dvf_gt)
                val_loss  += out["loss"].item()
                val_n     += 1

        val_avg = val_loss / max(val_n, 1)
        writer.add_scalar("tmnet_dvfsup_val/loss", val_avg, epoch)
        print(f"[TMNet-DVFSup] train={ep_avg:.6f}  val={val_avg:.6f}")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            custom_save(condi, os.path.join(log_dir, "model_best_tmnet.pth"))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            break

        print(f"[TMNet-DVFSup] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nDone.  Best checkpoint: {log_dir}/model_best_tmnet.pth")
    writer.close()


# =============================================================================
# CLI
# =============================================================================

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TMNet pretrainer — ACDC")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--train_test", required=True,
        choices=[
            "train_tmnet", "train_tmnet_predictive",
            "train_tmnet_mopTR_style", "train_tmnet_priormulti_image",
            "train_tmnet_priormulti_dvf",
            "train_tmnet_contrastive", "train_tmnet_dvfsup",
            "compute_stats", "test_tmnet", "test_tmnet_priormulti_dvf",
        ],
    )
    p.add_argument("--fold_nb_training", type=int, default=3)
    p.add_argument("--checkpoint",       type=str, default=None)
    p.add_argument("--tmnet_dir_name", type=str, default=None)
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args = _parse_cli()

    cfg = load_config(args.config)
    cfg.fold_nb_training = args.fold_nb_training
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    if args.tmnet_dir_name is not None:
        cfg.tmnet_dir_name = args.tmnet_dir_name
    _apply_overrides(cfg, args.override)

    # Strip logging_dir/logs/ prefix from tmnet_dir_name if the saved config stored the
    # full path instead of just the timestamp portion that test functions expect.
    if hasattr(cfg, "tmnet_dir_name") and cfg.tmnet_dir_name:
        _dn_prefix = os.path.join(cfg.logging_dir, "logs") + os.sep
        if cfg.tmnet_dir_name.startswith(_dn_prefix):
            cfg.tmnet_dir_name = cfg.tmnet_dir_name[len(_dn_prefix):]

    print("\n=== Config ===")
    print("\n".join(f"  {k}: {v}" for k, v in sorted(vars(cfg).items())))
    print("==============\n")

    device     = torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    _needs_vm = args.train_test not in (
        "train_tmnet_mopTR_style", "train_tmnet_priormulti_image",
        "train_tmnet_contrastive", "test_tmnet", "compute_stats",
    ) or args.train_test in ("train_tmnet_priormulti_dvf", "test_tmnet_dvfsup2")
    if _needs_vm:
        VOL_SIZE = (32, 64, 64)
        vm = build_reg_model(cfg, VOL_SIZE, device)
    else:
        vm = None

    train_folds, valid_folds, test_folds = make_train_val_test_folds(data_dir=cfg.data_dir)
    fold_nb = cfg.fold_nb_training or len(train_folds)
    print(f"Training on {fold_nb} fold(s).")

    dir_name = os.path.join(
        datetime.datetime.now().strftime("%m_%d"),
        datetime.datetime.now().strftime("%H.%M._") + cfg.name,
    )

    if args.train_test == "train_tmnet_priormulti_dvf":
        for fold_idx in range(fold_nb):
            # if fold_idx < 1:
            #     print("pass fold")
            #     continue
            # else: 
            train_tmnet_priormulti_dvf(
                cfg=cfg, folds=(train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx=str(fold_idx), dir_name=dir_name, vm=vm, device=device,
            )

    elif args.train_test == "train_tmnet_dvfsup":
        for fold_idx in range(fold_nb):
            train_tmnet_dvfsup(
                cfg=cfg, folds=(train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx=str(fold_idx), dir_name=dir_name, vm=vm, device=device,
            )

    elif args.train_test == "train_tmnet_contrastive":
        for fold_idx in range(fold_nb):
            train_tmnet_contrastive(
                cfg=cfg, folds=(train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx=str(fold_idx), dir_name=dir_name, device=device,
            )

    elif args.train_test == "train_tmnet_priormulti_image":
        for fold_idx in range(fold_nb):
            train_tmnet_priormulti_image(
                cfg=cfg, folds=(train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx=str(fold_idx), dir_name=dir_name, device=device,
            )

    elif args.train_test == "train_tmnet_mopTR_style":
        for fold_idx in range(fold_nb):
            train_tmnet_mopTR_style(
                cfg=cfg, folds=(train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx=str(fold_idx), dir_name=dir_name, device=device,
            )

    elif args.train_test == "train_tmnet_predictive":
        for fold_idx in range(fold_nb):
            train_tmnet_predictive(
                cfg=cfg, folds=(train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx=str(fold_idx), dir_name=dir_name, vm=vm, device=device,
            )

    elif args.train_test == "train_tmnet":
        for fold_idx in range(fold_nb):
            train_tmnet(
                cfg=cfg, folds=(train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx=str(fold_idx), dir_name=dir_name, vm=vm, device=device,
            )

    elif args.train_test == "compute_stats":
        for fold_idx in range(fold_nb):
            compute_tmnet_stats(
                cfg=cfg, folds=(train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx=str(fold_idx), dir_name=cfg.tmnet_dir_name, device=device,
            )

    elif args.train_test == "test_tmnet":
        for fold_idx in range(fold_nb):
            test_tmnet(
                cfg=cfg, fold=test_folds[fold_idx],
                fold_idx=str(fold_idx), dir_name=cfg.tmnet_dir_name, device=device,
            )

    elif args.train_test == "test_tmnet_priormulti_dvf":
        for fold_idx in range(fold_nb):
            test_tmnet_priormulti_dvf(
                cfg=cfg, fold=test_folds[fold_idx],
                fold_idx=str(fold_idx), dir_name=cfg.tmnet_dir_name, vm=vm, device=device,
            )