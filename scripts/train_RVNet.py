"""
train_RVNet.py
===================
Pretraining pipeline for RVNet via Masked Autoencoding (MAE).

Mirrors the organisation of train_VAE.py:
  - fold-based training with the same data loaders and DVF cache
  - TensorBoard logging with the same scalar naming conventions
  - best checkpoint selected on val reconstruction loss
  - early stopping
  - embedding statistics saved to rvnet_stats.pt at the end
    (analogous to latent_stats.pt, consumed by train_CLDM)

Usage
-----
# Stage 2a — pretrain RVNet with MAE
python -m 4D_MoPred_liver.scripts.train_RVNet --config 4D_MoPred_liver/configs/CondNets/RVNet.yaml --train_test train_rvnet

# Stage 2a — compute embedding statistics from an existing checkpoint
python -m 4D_MoPred_liver.scripts.train_RVNet \\
    --config 4D_MoPred_liver/configs/RVNet.yaml \\
    --train_test compute_stats \\
    --rvnet_dir_name <run_dir>

# Stage 2a — test reconstruction quality on the test set
python -m 4D_MoPred_liver.scripts.train_RVNet --config outputs/RVNet/logs/04_24/13.17._RVNet_MAE_run/fold_0/rvnet/RVNet.yaml --train_test test_rvnet --rvnet_dir_name /home/challierc/outputs/RVNet/logs/04_24/13.17._RVNet_MAE_run

"""

from __future__ import annotations

import argparse
import datetime
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, dynamic_ncols=True)

from mopred.models.Context_Encoder import RVNet, SparKRVNet

from mopred.data.splits       import make_folds_3fold as make_train_val_test_folds
from mopred.data.data_loaders import NAVIGATOR_4D_Dataset_multitime, CachedDVF_Dataset, RefVolume_Dataset, RefVolume_Dataset_augment
from mopred.data.loading      import save_params_txt
from mopred.models            import Voxelmorph
from mopred.utils.early_stopping  import EarlyStopping
from mopred.utils.io              import cond_mkdir, custom_load, custom_save
from mopred.utils.training        import (
    load_config, _apply_overrides,
    build_scheduler, build_optimizer,
    save_patients, summarize_test_metrics,
)
from mopred.utils.dvf_cache import build_dvf_cache

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


# =============================================================================
# Helpers
# =============================================================================

def _rvnet_linear_dim(cfg) -> int:
    """Compute the flattened spatial size after nb_convs stride-2 convolutions."""
    nb_convs = len(cfg.rvnet_enc_channels)
    D, H, W  = tuple(cfg.vol_size)
    factor   = 2 ** nb_convs
    return (D // factor) * (H // factor) * (W // factor)


def _build_rvnet_model(cfg, device: torch.device) -> RefCondMAE:
    """Instantiate RVNet + RefCondMAE from config."""


    norm_map = {
        "batchnorm":    nn.BatchNorm3d,
        "instancenorm": nn.InstanceNorm3d,
        "none":         None,
    }
    # GroupNorm instead of BatchNorm: BN running stats diverge from val distribution.
    norm_fn = lambda ch: nn.GroupNorm(min(8, ch), ch)

    encoder = RVNet(
        nb_convs         = len(cfg.rvnet_enc_channels),
        in_channels      = 1,
        out_channels     = cfg.rvnet_enc_channels,
        output_dim       = cfg.rvnet_pre_latent_dim,
        linear_input_dim = _rvnet_linear_dim(cfg),
        norm             = norm_fn,
    )

    mae = RefCondMAE(
        encoder    = encoder,
        vol_size   = tuple(cfg.vol_size),
        mask_ratio = cfg.rvnet_mask_ratio,
        patch_size = cfg.rvnet_patch_size,
        norm       = norm_fn,
    ).to(device)

    return mae


def _build_simmim_model(cfg, device: torch.device) -> SimMIMRVNet:
    """Instantiate RVNet + SimMIMRVNet from config."""
    norm_fn = lambda ch: nn.GroupNorm(min(8, ch), ch)
    encoder = RVNet(
        nb_convs         = len(cfg.rvnet_enc_channels),
        in_channels      = 1,
        out_channels     = cfg.rvnet_enc_channels,
        output_dim       = cfg.rvnet_pre_latent_dim,
        linear_input_dim = _rvnet_linear_dim(cfg),
        norm             = norm_fn,
    )
    simmim = SimMIMRVNet(
        encoder    = encoder,
        vol_size   = tuple(cfg.vol_size),
        mask_ratio = cfg.rvnet_mask_ratio,
        patch_size = cfg.rvnet_patch_size,
    ).to(device)
    return simmim


def _build_spark_model(cfg, device: torch.device) -> SparKRVNet:
    """Instantiate RVNet + SparKRVNet from config."""
    norm_fn = lambda ch: nn.GroupNorm(min(8, ch), ch)
    encoder = RVNet(
        nb_convs         = len(cfg.rvnet_enc_channels),
        in_channels      = 1,
        out_channels     = cfg.rvnet_enc_channels,
        output_dim       = cfg.rvnet_pre_latent_dim,
        linear_input_dim = _rvnet_linear_dim(cfg),
        norm             = norm_fn,
    )
    spark = SparKRVNet(
        encoder    = encoder,
        vol_size   = tuple(cfg.vol_size),
        mask_ratio = cfg.rvnet_mask_ratio,
        patch_size = cfg.rvnet_patch_size,
        dec_width  = getattr(cfg, "rvnet_dec_width", 64),
    ).to(device)
    return spark

def _build_moco_model(cfg, device: torch.device) -> MoCoRVNet:
    """Instantiate RVNet + MoCoRVNet from config."""
    norm_fn = lambda ch: nn.GroupNorm(min(8, ch), ch)

    encoder = RVNet(
        nb_convs         = len(cfg.rvnet_enc_channels),
        in_channels      = 1,
        out_channels     = cfg.rvnet_enc_channels,
        output_dim       = cfg.rvnet_pre_latent_dim,
        linear_input_dim = _rvnet_linear_dim(cfg),
        norm             = norm_fn,
    )

    moco = MoCoRVNet(
        encoder      = encoder,
        proj_dim     = getattr(cfg, "rvnet_proj_dim", 128),
        queue_size   = getattr(cfg, "rvnet_queue_size", 4096),
        momentum     = getattr(cfg, "rvnet_momentum", 0.999),
        temperature  = getattr(cfg, "rvnet_temperature", 0.07),
    ).to(device)

    return moco


def _rvnet_checkpoint_path(cfg, dir_name: str, fold_idx: str) -> str:
    return os.path.join(
        cfg.logging_dir, "logs", dir_name,
        f"fold_{fold_idx}", "rvnet", "model_best_rvnet.pth",
    )


def _unpack_ref_volume(batch, device: torch.device) -> torch.Tensor:
    """
    Extract the reference volume tensor from a DataLoader batch.

    Handles both:
      - RefVolume_Dataset   → (ref_volume, sequence_name)
      - CachedDVF_Dataset   → (ref_volume, input_vols, current_vols, dvf_list, ...)

    ref_volume collated shape: (B, D, H, W) → returned as (B, 1, D, H, W).
    """
    ref_volume = batch[0]
    if ref_volume.dim() == 4:
        ref_volume = ref_volume.unsqueeze(1)
    return ref_volume.float().to(device)


# =============================================================================
# Stage 2a — RVNet MAE training
# =============================================================================

def train_rvnet(
    cfg,
    folds: tuple,
    fold_idx: str,
    dir_name: str,
    vm: torch.nn.Module,
    device: torch.device,
) -> None:
    """
    Pretrain RVNet with Masked Autoencoding for one fold.

    Loop structure mirrors train_vae():
        out = mae(ref_volume)
        out["loss"].backward()
        optimizer.step()

    The mask ratio is linearly warmed up from 0 → target over
    rvnet_mask_warmup_epochs epochs (analogous to KL annealing in the VAE).
    Checkpoint selection uses val reconstruction loss on masked voxels.
    """

    # ---- build model -------------------------------------------------------
    mae = _build_rvnet_model(cfg, device)
    print(
        f"\n[RefEnc] Model: RefCondMAE"
        f"  | output_dim={mae.encoder.output_dim}"
        f"  | mask_ratio={mae.mask_ratio}"
        f"  | patch_size={mae.patch_size}"
        f"  (device={next(mae.parameters()).device})"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[RefEnc] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(mae, cfg.checkpoint, device)

    # ---- directories & logging ---------------------------------------------
    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(
            os.sep + "logs" + os.sep,
            os.sep + "runs" + os.sep, 1,
        )
    else:
        log_dir = os.path.join(
            cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "rvnet",
        )
        run_dir = os.path.join(
            cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "rvnet",
        )
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    # ---- data loaders ------------------------------------------------------
    train_set = RefVolume_Dataset_augment(cfg.data_dir, sequence_list=folds[0], repeats=80, augment=True)
    valid_set = RefVolume_Dataset_augment(cfg.data_dir, sequence_list=folds[1], repeats=20)

    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

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

    # ---- optimiser & scheduler ---------------------------------------------
    optimizer    = build_optimizer(cfg, mae.parameters())
    scheduler    = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience = cfg.early_stopping_patience,
        verbose  = True,
        delta    = cfg.early_stopping_delta,
    )

    restart_epoch  = getattr(cfg, "restart_epoch", 0)
    warmup_epochs  = getattr(cfg, "rvnet_mask_warmup_epochs", 0)
    mask_ratio_target = cfg.rvnet_mask_ratio

    print(f"[RefEnc] Starting training from epoch {restart_epoch}...")
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    # ---- epoch loop --------------------------------------------------------
    print(f"\nStage 2a — RVNet MAE training  (fold {fold_idx})")
    
    mae.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *cfg.vol_size).to(device)
        _ = mae(dummy)
    for name, m in mae.named_modules():
        if isinstance(m, nn.BatchNorm3d):
            print(f"WARNING: BatchNorm3d found at {name} — expected GroupNorm!")
        

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[RefEnc] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        mae.train()

        # Mask-ratio warm-up: linearly ramp 0 → target over warmup_epochs.
        # A low mask ratio early in training lets the encoder learn basic
        # features before being asked to reconstruct heavily occluded volumes.
        if warmup_epochs > 0:
            annealed = min(1.0, (epoch + 1) / warmup_epochs)
            mae.mask_ratio = mask_ratio_target * annealed
            writer.add_scalar("rvnet_train/mask_ratio", mae.mask_ratio, epoch)

        ep_loss = ep_steps = 0

        for batch in tqdm(train_loader):
            ref_volume = _unpack_ref_volume(batch, device)  # (B, 1, D, H, W)

            optimizer.zero_grad()
            out  = mae(ref_volume)
            loss = out["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            ep_loss  += loss.item()
            ep_steps += 1

            writer.add_scalar("rvnet_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("rvnet_train/epoch_loss", ep_avg, epoch)

        # ---- validation ----------------------------------------------------
        mae.eval()
        val_loss = val_n = 0

        train_mask_ratio = mae.mask_ratio          # save annealed value
        mae.mask_ratio   = mask_ratio_target       # always eval at target ratio

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                ref_volume = _unpack_ref_volume(batch, device)
                out        = mae(ref_volume)
                val_loss  += out["loss"].item()
                val_n     += 1

        mae.mask_ratio = train_mask_ratio          # restore for next train epoch

        val_avg = val_loss / max(val_n, 1)
        writer.add_scalar("rvnet_val/loss", val_avg, epoch)
        print(
            f"[RefEnc] train_loss={ep_avg:.6f}  "
            f"val_loss={val_avg:.6f}  "
            f"mask_ratio={mae.mask_ratio:.3f}"
        )

        # ---- checkpoint ----------------------------------------------------
        if val_avg < best_val_loss:
            print(
                f"[RefEnc] Improved val_loss {best_val_loss:.6f} → {val_avg:.6f} — saving."
            )
            best_val_loss = val_avg
            custom_save(mae, os.path.join(log_dir, "model_best_rvnet.pth"))
        else:
            print(f"[RefEnc] No improvement from {best_val_loss:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            print("[RefEnc] Early stopping triggered.")
            break

        print(f"[RefEnc] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nStage 2a done.  Best checkpoint: {log_dir}/model_best_rvnet.pth")
    writer.close()

    # ---- compute and save embedding statistics over the full training set ---
    print("[RefEnc] Computing embedding statistics over full training set...")
    custom_load(mae, os.path.join(log_dir, "model_best_rvnet.pth"), device)
    mae.eval()

    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            ref_volume = _unpack_ref_volume(batch, device)
            emb        = mae.encode(ref_volume)      # (B, output_dim)
            emb_sum    += emb.sum().item()
            emb_sq_sum += emb.pow(2).sum().item()
            emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)

    stats_path = os.path.join(log_dir, "rvnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(
        f"[RefEnc] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  "
        f"saved to {stats_path}"
    )


# =============================================================================
# Stage 2a — compute embedding statistics only (no retraining)
# =============================================================================

def compute_rvnet_stats(
    cfg,
    folds: tuple,
    fold_idx: str,
    dir_name: str,
    device: torch.device,
) -> None:
    """
    Load an existing RVNet checkpoint and compute embedding statistics
    over the full training set. Saves rvnet_stats.pt next to the checkpoint.

    Usage
    -----
    python -m 4D_MoPred_liver.scripts.train_RVNet \\
        --config configs/RVNet.yaml \\
        --train_test compute_stats \\
        --rvnet_dir_name <run_dir>
    """
    log_dir = os.path.join(
        cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "rvnet",
    )
    ckpt = os.path.join(log_dir, "model_best_rvnet.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[compute_stats] Checkpoint not found: {ckpt}")

    mae = _build_rvnet_model(cfg, device)
    custom_load(mae, ckpt, device)
    mae.eval()
    print(f"[compute_stats] Loaded RefCondMAE from {ckpt}")

    cache_dir = os.path.join(os.path.dirname(cfg.data_dir), "dvf_cache")
    train_set = CachedDVF_Dataset(
        NAVIGATOR_4D_Dataset_multitime(
            cfg.data_dir, nb_inputs=cfg.nb_inputs,
            sequence_list=folds[0], nb_pred=cfg.tp,
        ),
        cache_dir,
    )
    loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
    )

    max_batches = getattr(cfg, "rvnet_stats_batches", 50)
    emb_sum = emb_sq_sum = emb_count = 0

    print(f"[compute_stats] Encoding up to {max_batches}/{len(loader)} batches...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ref_volume = _unpack_ref_volume(batch, device)
            emb        = mae.encode(ref_volume)
            emb_sum    += emb.sum().item()
            emb_sq_sum += emb.pow(2).sum().item()
            emb_count  += emb.numel()
            print(f"  [{i+1}/{max_batches}]", end="\r")

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)

    stats_path = os.path.join(log_dir, "rvnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(
        f"\n[compute_stats] mean={emb_mean:.4f}  std={emb_std:.4f}  "
        f"→ saved to {stats_path}"
    )


# =============================================================================
# Stage 2a — test reconstruction quality
# =============================================================================

def test_rvnet(
    cfg,
    fold: list,
    fold_idx: str,
    dir_name: str,
    device: torch.device,
) -> None:
    """
    Evaluate MAE reconstruction quality on the test fold.

    For each test sample the function:
      1. Applies a fixed mask to the reference volume.
      2. Encodes + decodes through the MAE.
      3. Measures MSE between reconstruction and original on masked voxels.
      4. Optionally saves reconstructed NIfTI volumes for visual inspection.

    Results are saved as .npy arrays and a human-readable summary_metrics.txt
    under <logging_dir>/test_rvnet/<dir_name>/<fold_idx>/.
    """
    from mopred.utils.io import save_tensor_as_nifti

    # ---- build & load checkpoint -------------------------------------------
    mae = _build_rvnet_model(cfg, device)
    log_dir = os.path.join(
        cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "rvnet",
    )
    ckpt = os.path.join(log_dir, "model_best_rvnet.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[test_rvnet] Checkpoint not found: {ckpt}")
    custom_load(mae, ckpt, device)
    mae.eval()
    # Fix mask ratio to its trained target for deterministic evaluation
    mae.mask_ratio = cfg.rvnet_mask_ratio
    print(f"[test_rvnet] Loaded RefCondMAE from {ckpt}")

    # ---- data & output dirs ------------------------------------------------
    test_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, sequence_list=fold, nb_pred=cfg.tp,
        nb_inputs=cfg.nb_inputs, test=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
    )

    save_dir = os.path.join(
        cfg.logging_dir, dir_name, "test_rvnet", fold_idx,
    )
    vol_dir = os.path.join(save_dir, "volumes")
    for d in (save_dir, vol_dir):
        cond_mkdir(d)

    test_patients = {
        seq.split("/")[0] if "/" in seq else seq[:8] for seq in fold
    }
    with open(os.path.join(save_dir, "patients_test.txt"), "w") as f:
        for p in sorted(test_patients):
            f.write(p + "\n")

        # ---- inference loop ----------------------------------------------------
    from pytorch_msssim import ssim as ssim_fn   # pip install pytorch-msssim

    # ---- inference loop ----------------------------------------------------
    mse_all, mae_all, mape_all, psnr_all, ssim_all = [], [], [], [], []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            ref_volume = _unpack_ref_volume(batch, device)  # (1, 1, D, H, W)

            torch.manual_seed(42)
            out = mae(ref_volume)

            mask  = out["mask"]   # (1, 1, D, H, W)
            recon = out["recon"]  # (1, 1, D, H, W)

            n_masked    = mask.sum()
            target_mean = (ref_volume * mask).sum() / n_masked
            var_target  = (((ref_volume - target_mean) ** 2) * mask).sum() / n_masked

            # ── NMSE ──────────────────────────────────────────────────────────
            nmse = (((recon - ref_volume) ** 2) * mask).sum() / n_masked / var_target
            mse_all.append(nmse.item())

            # ── MAE (masked voxels) ───────────────────────────────────────────
            mae_val = (torch.abs(recon - ref_volume) * mask).sum() / n_masked
            mae_all.append(mae_val.item())

            # ── MAPE (masked voxels) ──────────────────────────────────────────
            eps      = 1e-6
            mape_val = (torch.abs((recon - ref_volume) / (ref_volume.abs() + eps)) * mask).sum() / n_masked * 100.0
            mape_all.append(mape_val.item())

            # ── PSNR (masked voxels) ──────────────────────────────────────────
            # Change max_val to 2.0 if your volumes are in [-1, 1]
            max_val  = 1.0
            raw_mse  = ((recon - ref_volume) ** 2 * mask).sum() / n_masked
            psnr_val = 10.0 * torch.log10(max_val ** 2 / (raw_mse + 1e-8))
            psnr_all.append(psnr_val.item())

            # ── SSIM (slice-by-slice over D, full slices) ─────────────────────
            # ref_volume: (1, 1, D, H, W) → iterate over D slices as (1, 1, H, W)
            # SSIM needs spatial context so we use full slices (not masked)
            D = ref_volume.shape[2]
            slice_ssims = []
            for d in range(D):
                ref_slice   = ref_volume[:, :, d, :, :].clamp(0, 1)  # (1, 1, H, W)
                recon_slice = recon      [:, :, d, :, :].clamp(0, 1)
                slice_ssims.append(
                    ssim_fn(recon_slice, ref_slice, data_range=1.0, size_average=True).item()
                )
            ssim_all.append(np.mean(slice_ssims))

            # ── Save volumes ──────────────────────────────────────────────────
            save_tensor_as_nifti(ref_volume[0, 0], "ref_original",    vol_dir, iter=idx)
            save_tensor_as_nifti(recon      [0, 0], "ref_recon",       vol_dir, iter=idx)
            save_tensor_as_nifti(
                (ref_volume * (1 - mask))[0, 0], "ref_masked_input", vol_dir, iter=idx
            )

    # ── numpy arrays ──────────────────────────────────────────────────────────────
    nmse_arr = np.array(mse_all)
    rmse_arr = np.sqrt(nmse_arr)
    mae_arr  = np.array(mae_all)
    mape_arr = np.array(mape_all)
    psnr_arr = np.array(psnr_all)
    ssim_arr = np.array(ssim_all)

    # ── save ──────────────────────────────────────────────────────────────────────
    np.save(os.path.join(save_dir, "NMSE_masked.npy"), nmse_arr)
    np.save(os.path.join(save_dir, "MAE_masked.npy"),  mae_arr)
    np.save(os.path.join(save_dir, "MAPE_masked.npy"), mape_arr)
    np.save(os.path.join(save_dir, "PSNR_masked.npy"), psnr_arr)
    np.save(os.path.join(save_dir, "SSIM_frames.npy"), ssim_arr)

    # ── summary ───────────────────────────────────────────────────────────────────
    def _fmt(arr, decimals=6, unit=""):
        return (f"{np.nanmean(arr):.{decimals}f} ± {np.nanstd(arr):.{decimals}f}"
                f"  |  median {np.nanmedian(arr):.{decimals}f}{unit}")

    summary_path = os.path.join(save_dir, "summary_metrics.txt")
    with open(summary_path, "w") as f:
        f.write("RVNet MAE — test reconstruction metrics\n")
        f.write("=" * 56 + "\n")
        f.write(f"Fold:           {fold_idx}\n")
        f.write(f"Checkpoint:     {ckpt}\n")
        f.write(f"Mask ratio:     {cfg.rvnet_mask_ratio}\n")
        f.write(f"Patch size:     {cfg.rvnet_patch_size}\n")
        f.write(f"N test samples: {len(mse_all)}\n\n")
        f.write(f"NMSE (masked voxels)  mean ± std: {_fmt(nmse_arr)}   [0=perfect, 1=mean-pred baseline]\n")
        f.write(f"RMSE (masked voxels)  mean ± std: {_fmt(rmse_arr)}   [same units as signal σ]\n")
        f.write(f"MAE  (masked voxels)  mean ± std: {_fmt(mae_arr)}   [absolute voxel units]\n")
        f.write(f"MAPE (masked voxels)  mean ± std: {_fmt(mape_arr, decimals=2, unit=' %')}   [% error; sensitive to near-zero voxels]\n")
        f.write(f"PSNR (masked voxels)  mean ± std: {_fmt(psnr_arr, decimals=2)}   dB  [higher=better; >30 dB good]\n")
        f.write(f"SSIM (slices, full)   mean ± std: {_fmt(ssim_arr, decimals=4)}   [0–1, higher=better]\n")

    print(
        f"\n[test_rvnet] "
        f"NMSE={np.nanmean(nmse_arr):.4f}  "
        f"RMSE={np.nanmean(rmse_arr):.4f}  "
        f"MAE={np.nanmean(mae_arr):.4f}  "
        f"MAPE={np.nanmean(mape_arr):.2f}%  "
        f"PSNR={np.nanmean(psnr_arr):.2f}dB  "
        f"SSIM={np.nanmean(ssim_arr):.4f}"
        f"\n[test_rvnet] Summary saved to {summary_path}"
    )


# =============================================================================
# Stage 2a (variant) — RVNet SimMIM pretraining
# =============================================================================

def train_rvnet_simmim(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    vm:       torch.nn.Module,
    device:   torch.device,
) -> None:
    """
    Pretrain RVNet via SimMIM for one fold.

    The CNN decoder is replaced by a single 1×1 Conv3d head that predicts
    patch-mean voxel values directly from backbone spatial features.
    Drastically reduces decoder overfitting compared to MAE.

    Checkpoint format is identical to train_rvnet so downstream code
    (train_CLDM, compute_stats) can load it without changes.
    """

    # ---- build model -------------------------------------------------------
    simmim = _build_simmim_model(cfg, device)
    print(
        f"\n[RefEnc-SimMIM] SimMIMRVNet"
        f"  | output_dim={simmim.encoder.output_dim}"
        f"  | mask_ratio={simmim.mask_ratio}"
        f"  | patch_size={simmim.patch_size}"
        f"  (device={next(simmim.parameters()).device})"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[RefEnc-SimMIM] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(simmim, cfg.checkpoint, device)

    # ---- directories & logging ---------------------------------------------
    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(
            cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "rvnet_simmim",
        )
        run_dir = os.path.join(
            cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "rvnet_simmim",
        )
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    # ---- data loaders ------------------------------------------------------
    train_set = RefVolume_Dataset_augment(cfg.data_dir, sequence_list=folds[0], repeats=80, augment=True)
    valid_set = RefVolume_Dataset_augment(cfg.data_dir, sequence_list=folds[1], repeats=20)

    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

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

    # ---- optimiser & scheduler ---------------------------------------------
    optimizer     = build_optimizer(cfg, simmim.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        verbose=True,
        delta=cfg.early_stopping_delta,
    )

    restart_epoch     = getattr(cfg, "restart_epoch", 0)
    warmup_epochs     = getattr(cfg, "rvnet_mask_warmup_epochs", 0)
    mask_ratio_target = cfg.rvnet_mask_ratio

    print(f"[RefEnc-SimMIM] Starting training from epoch {restart_epoch}...")
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    print(f"\nStage 2a — RVNet SimMIM pretraining  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[RefEnc-SimMIM] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        simmim.train()

        if warmup_epochs > 0:
            annealed = min(1.0, (epoch + 1) / warmup_epochs)
            simmim.mask_ratio = mask_ratio_target * annealed
            writer.add_scalar("rvnet_simmim_train/mask_ratio", simmim.mask_ratio, epoch)

        ep_loss = ep_steps = 0

        for batch in tqdm(train_loader):
            ref_volume = _unpack_ref_volume(batch, device)
            optimizer.zero_grad()

            out  = simmim(ref_volume)
            loss = out["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(simmim.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            ep_loss  += loss.item()
            ep_steps += 1
            writer.add_scalar("rvnet_simmim_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("rvnet_simmim_train/epoch_loss", ep_avg, epoch)

        # ---- validation ----------------------------------------------------
        simmim.eval()
        val_loss = val_n = 0

        train_mask_ratio  = simmim.mask_ratio
        simmim.mask_ratio = mask_ratio_target     # always eval at target ratio

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                ref_volume = _unpack_ref_volume(batch, device)
                out        = simmim(ref_volume)
                val_loss  += out["loss"].item()
                val_n     += 1

        simmim.mask_ratio = train_mask_ratio      # restore for next train epoch

        val_avg = val_loss / max(val_n, 1)
        writer.add_scalar("rvnet_simmim_val/loss", val_avg, epoch)
        print(
            f"[RefEnc-SimMIM] train={ep_avg:.6f}  val={val_avg:.6f}  "
            f"mask={simmim.mask_ratio:.3f}"
        )

        # ---- checkpoint — save encoder only (head discarded after pretraining)
        if val_avg < best_val_loss:
            print(f"[RefEnc-SimMIM] Improved val_loss {best_val_loss:.6f} → {val_avg:.6f} — saving.")
            best_val_loss = val_avg
            custom_save(simmim, os.path.join(log_dir, "model_best_rvnet.pth"))
        else:
            print(f"[RefEnc-SimMIM] No improvement from {best_val_loss:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            print("[RefEnc-SimMIM] Early stopping triggered.")
            break

        print(f"[RefEnc-SimMIM] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nStage 2a done.  Best checkpoint: {log_dir}/model_best_rvnet.pth")
    writer.close()

    # ---- compute and save embedding statistics ----------------------------
    print("[RefEnc-SimMIM] Computing embedding statistics over full training set...")
    custom_load(simmim, os.path.join(log_dir, "model_best_rvnet.pth"), device)
    simmim.eval()

    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            ref_volume = _unpack_ref_volume(batch, device)
            emb        = simmim.encode(ref_volume)
            emb_sum    += emb.sum().item()
            emb_sq_sum += emb.pow(2).sum().item()
            emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)

    stats_path = os.path.join(log_dir, "rvnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(
        f"[RefEnc-SimMIM] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  "
        f"saved to {stats_path}"
    )


# =============================================================================
# Stage 2a (variant) — RVNet SparK pretraining
# =============================================================================

def train_rvnet_spark(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    vm:       torch.nn.Module,
    device:   torch.device,
) -> None:
    """
    Pretrain RVNet via SparK-style masked modelling for one fold.

    Mask leakage is eliminated by propagating the binary patch mask through
    each stride-2 encoder block (forcing masked feature positions back to zero
    after every block) and by filling those positions with learned mask tokens
    before decoding. Loss: per-patch normalised MSE on masked patches only.

    The checkpoint format is identical to train_rvnet / train_rvnet_simmim so
    downstream code (compute_stats, train_CLDM) can load it without changes.

    Config keys used (beyond the shared ones)
    ------------------------------------------
    rvnet_dec_width          : decoder starting width (default 64)
    rvnet_mask_warmup_epochs : epochs to ramp mask_ratio 0 → target (default 0)
    """

    # ---- build model -------------------------------------------------------
    spark = _build_spark_model(cfg, device)
    print(
        f"\n[RefEnc-SparK] SparKRVNet"
        f"  | output_dim={spark.encoder.output_dim}"
        f"  | mask_ratio={spark.mask_ratio}"
        f"  | patch_size={spark.patch_size}"
        f"  | dec_width={getattr(cfg, 'rvnet_dec_width', 64)}"
        f"  | num_scales={spark.num_scales}"
        f"  | enc_out_chs={spark.enc_out_chs}"
        f"  (device={next(spark.parameters()).device})"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[RefEnc-SparK] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(spark, cfg.checkpoint, device)

    # ---- directories & logging ---------------------------------------------
    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(
            cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "rvnet_spark",
        )
        run_dir = os.path.join(
            cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "rvnet_spark",
        )
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    # ---- data loaders ------------------------------------------------------
    train_set = RefVolume_Dataset_augment(cfg.data_dir, sequence_list=folds[0], repeats=80, augment=True)
    valid_set = RefVolume_Dataset_augment(cfg.data_dir, sequence_list=folds[1], repeats=20)

    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

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

    # ---- optimiser & scheduler ---------------------------------------------
    optimizer     = build_optimizer(cfg, spark.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        verbose=True,
        delta=cfg.early_stopping_delta,
    )

    restart_epoch     = getattr(cfg, "restart_epoch", 0)
    warmup_epochs     = getattr(cfg, "rvnet_mask_warmup_epochs", 0)
    mask_ratio_target = cfg.rvnet_mask_ratio

    print(f"[RefEnc-SparK] Starting training from epoch {restart_epoch}...")
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    print(f"\nStage 2a — RVNet SparK pretraining  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[RefEnc-SparK] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        spark.train()

        if warmup_epochs > 0:
            annealed = min(1.0, (epoch + 1) / warmup_epochs)
            spark.mask_ratio = mask_ratio_target * annealed
            writer.add_scalar("rvnet_spark_train/mask_ratio", spark.mask_ratio, epoch)

        ep_loss = ep_steps = 0

        for batch in tqdm(train_loader):
            ref_volume = _unpack_ref_volume(batch, device)
            optimizer.zero_grad()

            out  = spark(ref_volume)
            loss = out["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(spark.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            ep_loss  += loss.item()
            ep_steps += 1
            writer.add_scalar("rvnet_spark_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("rvnet_spark_train/epoch_loss", ep_avg, epoch)

        # ---- validation ----------------------------------------------------
        spark.eval()
        val_loss = val_n = 0

        train_mask_ratio = spark.mask_ratio
        spark.mask_ratio = mask_ratio_target   # always eval at target ratio

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                ref_volume = _unpack_ref_volume(batch, device)
                out        = spark(ref_volume)
                val_loss  += out["loss"].item()
                val_n     += 1

        spark.mask_ratio = train_mask_ratio    # restore for next train epoch

        val_avg = val_loss / max(val_n, 1)
        writer.add_scalar("rvnet_spark_val/loss", val_avg, epoch)
        print(
            f"[RefEnc-SparK] train={ep_avg:.6f}  val={val_avg:.6f}  "
            f"mask={spark.mask_ratio:.3f}"
        )

        # ---- checkpoint (full wrapper — encoder extractable via spark.encoder)
        if val_avg < best_val_loss:
            print(f"[RefEnc-SparK] Improved val_loss {best_val_loss:.6f} → {val_avg:.6f} — saving.")
            best_val_loss = val_avg
            custom_save(spark, os.path.join(log_dir, "model_best_rvnet.pth"))
        else:
            print(f"[RefEnc-SparK] No improvement from {best_val_loss:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            print("[RefEnc-SparK] Early stopping triggered.")
            break

        print(f"[RefEnc-SparK] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nStage 2a done.  Best checkpoint: {log_dir}/model_best_rvnet.pth")
    writer.close()

    # ---- compute and save embedding statistics ----------------------------
    print("[RefEnc-SparK] Computing embedding statistics over full training set...")
    custom_load(spark, os.path.join(log_dir, "model_best_rvnet.pth"), device)
    spark.eval()

    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            ref_volume = _unpack_ref_volume(batch, device)
            emb        = spark.encode(ref_volume)   # (B, output_dim) — encoder only
            emb_sum    += emb.sum().item()
            emb_sq_sum += emb.pow(2).sum().item()
            emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)

    stats_path = os.path.join(log_dir, "rvnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(
        f"[RefEnc-SparK] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  "
        f"saved to {stats_path}"
    )

def train_rvnet_moco(
    cfg,
    folds: tuple,
    fold_idx: str,
    dir_name: str,
    vm: torch.nn.Module,
    device: torch.device,
) -> None:
    """
    Pretrain RVNet via MoCo contrastive learning.
    """

    # ---- build model -------------------------------------------------------
    moco = _build_moco_model(cfg, device)
    print(
        f"\n[RefEnc-MoCo] MoCoRVNet"
        f" | output_dim={moco.encoder_q.output_dim}"
        f" | proj_dim={moco.proj_q.net[-1].out_features}"
        f" | queue={moco.queue_size}"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[RefEnc-MoCo] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(moco, cfg.checkpoint, device)

    # ---- dirs --------------------------------------------------------------
    log_dir = os.path.join(
        cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "rvnet_moco",
    )
    run_dir = os.path.join(
        cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "rvnet_moco",
    )
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    base_train = RefVolume_Dataset_augment(cfg.data_dir, sequence_list=folds[0], repeats=80, augment=True)
    base_valid = RefVolume_Dataset_augment(cfg.data_dir, sequence_list=folds[1], repeats=20)

    train_set = MoCoPairDataset(base_train)
    valid_set = MoCoPairDataset(base_valid)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
    )

    # ---- optimizer ---------------------------------------------------------
    optimizer     = build_optimizer(cfg, moco.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        verbose=True,
        delta=cfg.early_stopping_delta,
    )

    best_val_loss = float("inf")
    global_step   = 0

    print(f"\nStage 2a — RVNet MoCo pretraining (fold {fold_idx})")

    # ---- training loop -----------------------------------------------------
    for epoch in range(cfg.vae_epochs):
        print(f"\n[RefEnc-MoCo] Epoch {epoch}")
        t0 = time.time()

        moco.train()
        ep_loss = ep_steps = 0

        for q, k in tqdm(train_loader):
            q = q.to(device)
            k = k.to(device)

            optimizer.zero_grad()
            out  = moco(q, k, update=True)
            loss = out["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(moco.parameters(), 1.0)
            optimizer.step()

            if scheduler and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            ep_loss += loss.item()
            ep_steps += 1

            writer.add_scalar("rvnet_moco_train/loss", loss.item(), global_step)
            writer.add_scalar("rvnet_moco_train/acc1", out["acc1"], global_step)
            writer.add_scalar("rvnet_moco_train/acc5", out["acc5"], global_step)

            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("rvnet_moco_train/epoch_loss", ep_avg, epoch)

        # ---- validation (NO QUEUE UPDATE!) ---------------------------------
        moco.eval()
        val_loss = val_n = 0
        val_acc1 = 0

        with torch.no_grad():
            for q, k in tqdm(valid_loader):
                q = q.to(device)
                k = k.to(device)

                out = moco(q, k, update=False)

                val_loss += out["loss"].item()
                val_acc1 += out["acc1"]
                val_n    += 1

        val_avg  = val_loss / max(val_n, 1)
        val_acc1 = val_acc1 / max(val_n, 1)

        writer.add_scalar("rvnet_moco_val/loss", val_avg, epoch)
        writer.add_scalar("rvnet_moco_val/acc1", val_acc1, epoch)

        print(f"[RefEnc-MoCo] train={ep_avg:.6f} val={val_avg:.6f} acc1={val_acc1:.3f}")

        # ---- checkpoint ----------------------------------------------------
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            custom_save(moco, os.path.join(log_dir, "model_best_rvnet.pth"))
            print("[RefEnc-MoCo] Saved best model")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            print("[RefEnc-MoCo] Early stopping")
            break

        print(f"[RefEnc-MoCo] Epoch time {(time.time()-t0)/60:.2f} min")

    writer.close()

    # ---- compute embedding stats (encoder_q only) --------------------------
    print("[RefEnc-MoCo] Computing embedding stats...")
    custom_load(moco, os.path.join(log_dir, "model_best_rvnet.pth"), device)
    moco.eval()

    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for q, _ in tqdm(train_loader):
            q = q.to(device)
            emb = moco.encode(q)

            emb_sum    += emb.sum().item()
            emb_sq_sum += emb.pow(2).sum().item()
            emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)

    torch.save(
        {"mean": emb_mean, "std": emb_std},
        os.path.join(log_dir, "rvnet_stats.pt"),
    )

    print(f"[RefEnc-MoCo] mean={emb_mean:.4f}, std={emb_std:.4f}")

# =============================================================================
# CLI
# =============================================================================

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RVNet MAE pretrainer")
    p.add_argument(
        "--config", required=True,
        help="Path to a YAML config file (see configs/RVNet.yaml).",
    )
    p.add_argument(
        "--train_test", required=True,
        choices=[
            "train_rvnet",
            "train_rvnet_simmim",
            "train_rvnet_spark",
            "train_rvnet_moco", 
            "compute_stats",
            "test_rvnet",
        ],
        help=(
            "'train_rvnet'        — pretrain RVNet via MAE; "
            "'train_rvnet_simmim' — pretrain RVNet via SimMIM; "
            "'train_rvnet_spark'  — pretrain RVNet via SparK (no sparse-conv library); "
            "'compute_stats'       — compute embedding statistics from a checkpoint; "
            "'test_rvnet'         — evaluate reconstruction quality on the test set."
        ),
    )
    p.add_argument(
        "--fold_nb_training", type=int, default=3,
        help="How many folds to train on (0 = all).",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "[train_rvnet only] Path to a .pth checkpoint to resume from, e.g. "
            "'<logging_dir>/logs/04_10/14.30._run/fold_0/rvnet/model_best_rvnet.pth'."
        ),
    )
    p.add_argument(
        "--rvnet_dir_name", type=str, default=None,
        help="[compute_stats / test_rvnet] Run directory name to load from.",
    )
    p.add_argument(
        "--override", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override any config key, e.g. --override lr_rvnet=5e-5 batch_size=8",
    )
    return p.parse_args()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args = _parse_cli()

    # Load & patch config
    cfg = load_config(args.config)
    cfg.fold_nb_training = args.fold_nb_training
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    if args.rvnet_dir_name is not None:
        cfg.rvnet_dir_name = args.rvnet_dir_name
    _apply_overrides(cfg, args.override)

    print("\n=== Config ===")
    print("\n".join(f"  {k}: {v}" for k, v in sorted(vars(cfg).items())))
    print("==============\n")

    # Global setup
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

    # Frozen VoxelMorph — needed only to build the DVF cache (same as train_vae.py).
    # If the cache already exists from a prior VAE training run it is reused as-is.
    current_path  = os.path.dirname(os.path.abspath(__file__))
    VM_CHECKPOINT = os.path.join(os.path.dirname(current_path), "pretrained_models", "VM.pth")
    VOL_SIZE      = (32, 64, 64)

    vm = Voxelmorph(
        VOL_SIZE, [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16],
        full_size=True,
    ).to(device)
    custom_load(vm, VM_CHECKPOINT, device)
    vm.eval()
    for param in vm.parameters():
        param.requires_grad_(False)

    train_folds, valid_folds, test_folds = make_train_val_test_folds()

    fold_nb = cfg.fold_nb_training or len(train_folds)
    print(f"Training on {fold_nb} fold(s).")

    dir_name = os.path.join(
        datetime.datetime.now().strftime("%m_%d"),
        datetime.datetime.now().strftime("%H.%M._") + cfg.name,
    )

    # ------------------------------------------------------------------
    if args.train_test == "train_rvnet_spark":
        for fold_idx in range(fold_nb):
            print(f"\n=== Training fold {fold_idx} ===")
            train_rvnet_spark(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
                vm       = vm,
                device   = device,
            )

    elif args.train_test == "train_rvnet_simmim":
        for fold_idx in range(fold_nb):
            print(f"\n=== Training fold {fold_idx} ===")
            train_rvnet_simmim(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
                vm       = vm,
                device   = device,
            )
    elif args.train_test == "train_rvnet_moco":
        for fold_idx in range(fold_nb):
            train_rvnet_moco(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
                vm       = vm,
                device   = device,
            )

    elif args.train_test == "train_rvnet":
        for fold_idx in range(fold_nb):
            print(f"\n=== Training fold {fold_idx} ===")
            train_rvnet(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
                vm       = vm,
                device   = device,
            )

    elif args.train_test == "compute_stats":
        for fold_idx in range(fold_nb):
            compute_rvnet_stats(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = cfg.rvnet_dir_name,
                device   = device,
            )

    elif args.train_test == "test_rvnet":
        for fold_idx in range(fold_nb):
            test_rvnet(
                cfg      = cfg,
                fold     = test_folds[fold_idx],
                fold_idx = str(fold_idx),
                dir_name = cfg.rvnet_dir_name,
                device   = device,
            )