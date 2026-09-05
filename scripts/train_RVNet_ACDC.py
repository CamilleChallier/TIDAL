"""
train_RVNet_ACDC.py
========================
Cardiac analog of train_RVNet.py — RVNet pretraining via SparK-style
masked modelling, trained from scratch on the public ACDC cine-MRI dataset.

Standalone copy: hard-codes the ACDC dataset/loaders/folds/CANONICAL_SHAPE in
place of their liver counterparts. The liver script is left untouched. SparK
pretraining doesn't need a frozen VoxelMorph (unlike train_RVNet.py's
other variants), so no VM checkpoint is loaded here.

Usage
-----
# Pretrain RVNet with SparK
python -m scripts.train_RVNet_ACDC --config configs/CondNets/RVNet_acdc.yaml --train_test train_rvnet_spark

# Compute embedding statistics from an existing checkpoint
python -m scripts.train_RVNet_ACDC \\
    --config configs/CondNets/RVNet_acdc.yaml \\
    --train_test compute_stats \\
    --rvnet_dir_name <run_dir>
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, dynamic_ncols=True)

from mopred.models.Context_Encoder import RVNet, SparKRVNet

from mopred.data.splits       import make_folds_acdc as make_train_val_test_folds
from mopred.data.data_loaders.acdc_4d import RefVolume_Dataset_augment_ACDC, ACDC_4D_Dataset, CANONICAL_SHAPE
from mopred.data.loading      import save_params_txt
from mopred.utils.early_stopping  import EarlyStopping
from mopred.utils.io              import cond_mkdir, custom_load, custom_save
from mopred.utils.training        import (
    load_config, _apply_overrides,
    build_scheduler, build_optimizer,
    save_patients,
)

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

def _acdc_dataset_kwargs(cfg) -> dict:
    """Shared ACDC_4D_Dataset kwargs derived from cfg — used by every
    instantiation site (train/valid/test) so they can never drift apart and
    silently end up preprocessed differently from one another."""
    return dict(
        downsample_to_liver=getattr(cfg, "downsample_to_liver", False),
        cache_dir=getattr(cfg, "cache_dir", None),
        reference_slices_path=getattr(cfg, "reference_slices_path", None),
    )

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


def _unpack_ref_volume(batch, device: torch.device) -> torch.Tensor:
    """
    Extract the reference volume tensor from a DataLoader batch.

    Handles both:
      - RefVolume_Dataset_augment_ACDC → (ref_volume, patient)
      - ACDC_4D_Dataset                → (ref_volume, input_vols, current_vols, vol_files)

    ref_volume collated shape: (B, D, H, W) → returned as (B, 1, D, H, W).
    """
    ref_volume = batch[0]
    if ref_volume.dim() == 4:
        ref_volume = ref_volume.unsqueeze(1)
    return ref_volume.float().to(device)


def _seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)


# =============================================================================
# Stage 2a — RVNet SparK pretraining
# =============================================================================

def train_rvnet_spark(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    device:   torch.device,
) -> None:
    """
    Pretrain RVNet via SparK-style masked modelling for one fold.

    Mirrors train_RVNet.py::train_rvnet_spark, with the ACDC dataset in
    place of RefVolume_Dataset_augment.
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
    ds_kwargs = _acdc_dataset_kwargs(cfg)
    train_set = RefVolume_Dataset_augment_ACDC(cfg.data_dir, sequence_list=folds[0], repeats=80, augment=True, **ds_kwargs)
    valid_set = RefVolume_Dataset_augment_ACDC(cfg.data_dir, sequence_list=folds[1], repeats=20, augment=False, **ds_kwargs)

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

    print(f"\nStage 2a — RVNet SparK pretraining (ACDC)  (fold {fold_idx})")

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

            if epoch == restart_epoch and ep_steps == 0:
                print(f"[RefEnc-SparK][shapes] ref_volume={tuple(ref_volume.shape)}  "
                      f"loss={tuple(loss.shape) if hasattr(loss, 'shape') else 'scalar'}  "
                      f"out_keys={list(out.keys())}")

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
    """
    log_dir = os.path.join(
        cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "rvnet_spark",
    )
    ckpt = os.path.join(log_dir, "model_best_rvnet.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[compute_stats] Checkpoint not found: {ckpt}")

    spark = _build_spark_model(cfg, device)
    custom_load(spark, ckpt, device)
    spark.eval()
    print(f"[compute_stats] Loaded SparKRVNet from {ckpt}")

    ds_kwargs = _acdc_dataset_kwargs(cfg)
    train_set = ACDC_4D_Dataset(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp,
        **ds_kwargs
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
            emb        = spark.encode(ref_volume)
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
# Stage 2a — test mode: SparK reconstruction quality on the test set
# =============================================================================

def test_rvnet_spark(
    cfg,
    fold:     list,
    fold_idx: str,
    dir_name: str,
    device:   torch.device,
) -> None:
    """
    Evaluate a trained RVNet SparK checkpoint on the test set:
    reconstruction quality (NMSE/MAE/MAPE/PSNR/SSIM) on the masked patches of
    the reference volume, at the configured mask ratio.

    Results saved under <logging_dir>/<dir_name>/test_rvnet_spark/<fold_idx>/.
    """
    spark = _build_spark_model(cfg, device)
    log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "rvnet_spark")
    ckpt = os.path.join(log_dir, "model_best_rvnet.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[test_rvnet_spark] Checkpoint not found: {ckpt}")

    custom_load(spark, ckpt, device)
    spark.eval()
    spark.mask_ratio = cfg.rvnet_mask_ratio

    test_set = RefVolume_Dataset_augment_ACDC(cfg.data_dir, sequence_list=fold, repeats=1, augment=False,
                                               downsample_to_liver=getattr(cfg, "downsample_to_liver", False),
                                               cache_dir=getattr(cfg, "cache_dir", None))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    save_dir = os.path.join(cfg.logging_dir, dir_name, "test_rvnet_spark", fold_idx)
    cond_mkdir(save_dir)

    test_patients = sorted(set(fold))
    with open(os.path.join(save_dir, "patients_test.txt"), "w") as f:
        for p in test_patients:
            f.write(p + "\n")

    mse_all, mae_all, mape_all, psnr_all, ssim_all = [], [], [], [], []

    from skimage.metrics import structural_similarity as ss

    with torch.no_grad():
        for batch in tqdm(test_loader):
            ref_volume = _unpack_ref_volume(batch, device)
            torch.manual_seed(42)
            out = spark(ref_volume)

            target = ref_volume

            # `recon` is decoded in per-patch normalized units (zero mean,
            # unit variance per patch of `target`) — undo that normalization
            # so it's comparable to `target` in [0, 1] pixel space.
            C = target.shape[1]
            inp_pat  = spark.patchify(target)
            rec_pat  = spark.patchify(out["recon"])
            mean     = inp_pat.mean(dim=-1, keepdim=True)
            var      = (inp_pat.var(dim=-1, keepdim=True) + 1e-6).sqrt()
            recon_pat = rec_pat * var + mean
            recon = spark.unpatchify(recon_pat, C)

            active = out["active"].float()                       # (B,1,gD,gH,gW)
            mask   = 1.0 - F.interpolate(active, size=target.shape[2:], mode="nearest")

            n_masked    = mask.sum()
            target_mean = (target * mask).sum() / n_masked
            var_target  = (((target - target_mean) ** 2) * mask).sum() / n_masked

            nmse    = (((recon - target) ** 2) * mask).sum() / n_masked / var_target
            mae_val = (torch.abs(recon - target) * mask).sum() / n_masked
            eps     = 1e-6
            mape_val = (torch.abs((recon - target) / (target.abs() + eps)) * mask).sum() / n_masked * 100.0
            raw_mse  = ((recon - target) ** 2 * mask).sum() / n_masked
            psnr_val = 10.0 * torch.log10(1.0 / (raw_mse + 1e-8))

            t_np = target.clamp(0, 1).squeeze().cpu().numpy()   # (D, H, W)
            r_np = recon.clamp(0, 1).squeeze().cpu().numpy()
            ssim_val = float(np.mean([
                ss(t_np[d], r_np[d], data_range=1.0) for d in range(t_np.shape[0])
            ]))

            mse_all.append(nmse.item())
            mae_all.append(mae_val.item())
            mape_all.append(mape_val.item())
            psnr_all.append(psnr_val.item())
            ssim_all.append(ssim_val)

    mse_arr  = np.array(mse_all)
    rmse_arr = np.sqrt(mse_arr)
    mae_arr  = np.array(mae_all)
    mape_arr = np.array(mape_all)
    psnr_arr = np.array(psnr_all)
    ssim_arr = np.array(ssim_all)

    np.save(os.path.join(save_dir, "NMSE_masked.npy"), mse_arr)
    np.save(os.path.join(save_dir, "MAE_masked.npy"),  mae_arr)
    np.save(os.path.join(save_dir, "MAPE_masked.npy"), mape_arr)
    np.save(os.path.join(save_dir, "PSNR_volumes.npy"), psnr_arr)
    np.save(os.path.join(save_dir, "SSIM_volumes.npy"), ssim_arr)

    def _fmt(arr, decimals=6, unit=""):
        return (f"{np.nanmean(arr):.{decimals}f} ± {np.nanstd(arr):.{decimals}f}"
                f"  |  median {np.nanmedian(arr):.{decimals}f}{unit}")

    summary_path = os.path.join(save_dir, "summary_metrics.txt")
    with open(summary_path, "w") as f:
        f.write("RVNet SparK — test reconstruction metrics (ACDC)\n")
        f.write("=" * 56 + "\n")
        f.write(f"Fold:           {fold_idx}\n")
        f.write(f"Checkpoint:     {ckpt}\n")
        f.write(f"Mask ratio:     {cfg.rvnet_mask_ratio}\n")
        f.write(f"N test samples: {len(mse_all)}\n\n")
        f.write(f"NMSE  (masked)   {_fmt(mse_arr)}\n")
        f.write(f"RMSE  (masked)   {_fmt(rmse_arr)}\n")
        f.write(f"MAE   (masked)   {_fmt(mae_arr)}\n")
        f.write(f"MAPE  (masked)   {_fmt(mape_arr, decimals=2, unit=' %')}\n")
        f.write(f"PSNR  (volumes)  {_fmt(psnr_arr, decimals=2)}  dB\n")
        f.write(f"SSIM  (volumes)  {_fmt(ssim_arr, decimals=4)}\n")

    print(
        f"[test_rvnet_spark] "
        f"NMSE={np.nanmean(mse_arr):.4f}  RMSE={np.nanmean(rmse_arr):.4f}  "
        f"PSNR={np.nanmean(psnr_arr):.2f}dB  SSIM={np.nanmean(ssim_arr):.4f}"
        f"  — summary → {summary_path}"
    )


# =============================================================================
# CLI
# =============================================================================

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RVNet SparK pretrainer — ACDC")
    p.add_argument(
        "--config", required=True,
        help="Path to a YAML config file (see configs/CondNets/RVNet_acdc.yaml).",
    )
    p.add_argument(
        "--train_test", required=True,
        choices=["train_rvnet_spark", "compute_stats", "test_rvnet_spark"],
        help=(
            "'train_rvnet_spark' — pretrain RVNet via SparK; "
            "'compute_stats'      — compute embedding statistics from a checkpoint; "
            "'test_rvnet_spark'  — evaluate SparK reconstruction quality on the test set."
        ),
    )
    p.add_argument(
        "--fold_nb_training", type=int, default=3,
        help="How many folds to train on (0 = all).",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "[train_rvnet_spark only] Path to a .pth checkpoint to resume from, e.g. "
            "'<logging_dir>/logs/04_10/14.30._run/fold_0/rvnet_spark/model_best_rvnet.pth'."
        ),
    )
    p.add_argument(
        "--rvnet_dir_name", type=str, default=None,
        help="[compute_stats] Run directory name to load from.",
    )
    p.add_argument(
        "--override", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override any config key, e.g. --override lr_vae=5e-5 batch_size=8",
    )
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
    if args.rvnet_dir_name is not None:
        cfg.rvnet_dir_name = args.rvnet_dir_name
    _apply_overrides(cfg, args.override)

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

    train_folds, valid_folds, test_folds = make_train_val_test_folds(data_dir=cfg.data_dir)

    fold_nb = cfg.fold_nb_training or len(train_folds)
    print(f"Training on {fold_nb} fold(s).")

    dir_name = os.path.join(
        datetime.datetime.now().strftime("%m_%d"),
        datetime.datetime.now().strftime("%H.%M._") + cfg.name,
    )

    if args.train_test == "train_rvnet_spark":
        for fold_idx in range(fold_nb):
            # if fold_idx == 0 or fold_idx== 1 :
            #     continue
            # else : 
            print(f"\n=== Training fold {fold_idx} ===")
            train_rvnet_spark(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
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

    elif args.train_test == "test_rvnet_spark":
        for fold_idx in range(fold_nb):
            test_rvnet_spark(
                cfg      = cfg,
                fold     = test_folds[fold_idx],
                fold_idx = str(fold_idx),
                dir_name = cfg.rvnet_dir_name,
                device   = device,
            )
