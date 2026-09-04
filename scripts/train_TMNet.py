"""
train_TMNet.py
=================
Pretraining pipeline for TMNet_Tr_priormulti[_mask] via Masked Autoencoding (MAE).

Mirrors train_RVNet.py / train_VAE.py in organisation:
  - fold-based training with the same data loaders and DVF cache
  - TensorBoard logging with the same scalar naming conventions
  - best checkpoint on val reconstruction loss
  - early stopping
  - mask-ratio warm-up (analogous to KL annealing in VAE)
  - embedding statistics saved to tmnet_stats.pt at the end

The backbone (TMNet_Tr_priormulti or TMNet_Tr_priormulti_mask) is pretrained
The MAE decoder is a lightweight scaffold
that is discarded after pretraining.

Usage
-----
# Stage 2b — pretrain TMNet with MAE
python -m 4D_MoPred_liver.scripts.train_TMNet --config 4D_MoPred_liver/configs/CondNets/TMNet.yaml --train_test train_tmnet_priormulti_dvf

# Stage 2b — compute embedding statistics from an existing checkpoint
python -m 4D_MoPred_liver.scripts.train_TMNet \\
    --config 4D_MoPred_liver/configs/CondNets/TMNet.yaml \\
    --train_test compute_stats \\
    --tmnet_dir_name <run_dir>

# Stage 2b — test reconstruction quality on the test set
python -m 4D_MoPred_liver.scripts.train_TMNet --config outputs/TMNet/logs/04_24/13.17._TMNet_run/fold_0/tmnet/TMNet.yaml --train_test test_tmnet --tmnet_dir_name /home/challierc/outputs/TMNet/logs/04_24/13.17._TMNet_run
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, dynamic_ncols=True)

from mopred.models.Context_Encoder import (
    TMNetEncoder, TMNet_Tr_priormulti_image,
    PredictiveTMNet, MopTRTMNet, DVFSupTMNet,
)
from mopred.models.Context_Encoder.temporal_augmentations import TemporalAugConfig, apply_pixel_space_augmentations

from mopred.data.splits       import make_folds_3fold as make_train_val_test_folds
from mopred.data.data_loaders import NAVIGATOR_4D_Dataset_multitime
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
from mopred.data.data_loaders.navigator_4d import get_phi as _get_phi, get_amplitude as _get_amplitude, get_peak_amplitude as _get_peak_amplitude

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")



def _unpack_condi_inputs(batch, device: torch.device, cfg) -> torch.Tensor:
    """
    Extract past navigator frames from a dataset batch and build
    the (B, 2, T, H, W) tensor expected by TMNet / MAE.

    Mirrors build_Iseq: for each input volume, extract a 2D slice and
    concatenate with the matching reference slice (C=2).
    condi_type "1" → sagittal slice at d=16
    condi_type "2" → coronal   slice at h=32
    """
    ref_raw, input_volume_list, *_ = batch
    ref_vol = ref_raw.unsqueeze(1).to(device)          # (B, 1, D, H, W)

    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    frames_list = []

    if condi_type == "1":                              # sagittal
        c_ref = ref_vol[:, :, 16, :, :]               # (B, 1, H, W)
        for q in range(cfg.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1).to(device)[:, :, 16, :, :]
            frames_list.append(torch.cat([c_t, c_ref], dim=1))
    else:                                              # coronal
        c_ref = ref_vol[:, :, :, 32, :]               # (B, 1, D, W)
        for q in range(cfg.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1).to(device)[:, :, :, 32, :]
            frames_list.append(torch.cat([c_t, c_ref], dim=1))

    return torch.stack(frames_list, dim=2).float()     # (B, 2, T, H, W)


def _tmnet_checkpoint_path(cfg, dir_name: str, fold_idx: str) -> str:
    return os.path.join(
        cfg.logging_dir, "logs", dir_name,
        f"fold_{fold_idx}", "tmnet", "model_best_tmnet.pth",
    )


def _build_aug_cfg(cfg) -> TemporalAugConfig:
    """
    Reconstruct TemporalAugConfig from the flattened config namespace.

    load_config() flattens one level of nesting, so:
        temporal_aug: {frame_drop: {enabled: true, ...}}
    becomes cfg.temporal_aug_frame_drop = {"enabled": true, ...}.
    """
    aug_dict = {
        "frame_drop":      getattr(cfg, "temporal_aug_frame_drop",      {}),
        "spatial_mask":    getattr(cfg, "temporal_aug_spatial_mask",    {}),
        "tube_mask":       getattr(cfg, "temporal_aug_tube_mask",        {}),
        "variable_frames": getattr(cfg, "temporal_aug_variable_frames", {}),
    }
    return TemporalAugConfig.from_dict(aug_dict)


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
    """
    Pretrain TMNet with Masked Autoencoding for one fold.

    Loop structure:
        frames = _unpack_condi_inputs(batch, device, cfg)
        out    = mae(frames, aug_cfg=aug_cfg, training=True)
        loss   = out["loss"]
        loss.backward()
        optimizer.step()

    The mask ratio is linearly warmed up from 0 → target over
    tmnet_mask_warmup_epochs epochs, analogous to KL annealing in the VAE.
    Only the TMNet backbone is saved at checkpoints; the MAE decoder is
    discarded after pretraining.
    """

    # ---- build model -------------------------------------------------------
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

    # ---- directories & logging ---------------------------------------------
    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(
            os.sep + "logs" + os.sep,
            os.sep + "runs" + os.sep, 1,
        )
    else:
        log_dir = os.path.join(
            cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet",
        )
        run_dir = os.path.join(
            cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet",
        )
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    # ---- data loaders ------------------------------------------------------
    train_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp,
    )
    valid_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[1], nb_pred=cfg.tp, valid=True,
    )

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
    # mae.parameters() covers condi + MAE decoder + spatial_proj jointly.
    optimizer     = build_optimizer(cfg, mae.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience = cfg.early_stopping_patience,
        verbose  = True,
        delta    = cfg.early_stopping_delta,
    )

    restart_epoch     = getattr(cfg, "restart_epoch", 0)
    warmup_epochs     = getattr(cfg, "tmnet_mask_warmup_epochs", 0)
    mask_ratio_target = cfg.tmnet_mask_ratio

    # ---- augmentation config -----------------------------------------------
    aug_cfg = _build_aug_cfg(cfg)
    if aug_cfg.any_enabled():
        print(
            f"[TMNet] Temporal augmentations: "
            f"frame_drop={aug_cfg.frame_drop.enabled}, "
            f"spatial_mask={aug_cfg.spatial_mask.enabled}, "
            f"tube_mask={aug_cfg.tube_mask.enabled}, "
            f"variable_frames={aug_cfg.variable_frames.enabled}"
        )
    else:
        print("[TMNet] No temporal augmentations enabled.")

    print(f"[TMNet] Starting training from epoch {restart_epoch}...")
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    # ---- epoch loop --------------------------------------------------------
    print(f"\nStage 2b — TMNet MAE training  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        condi.train()
        mae.train()

        # Mask-ratio warm-up: linearly ramp 0 → target over warmup_epochs.
        if warmup_epochs > 0:
            annealed = min(1.0, (epoch + 1) / warmup_epochs)
            mae.mask_ratio = mask_ratio_target * annealed
            print(f"mask ratio : {mask_ratio_target * annealed}")
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

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            ep_loss  += loss.item()
            ep_steps += 1
            writer.add_scalar("tmnet_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("tmnet_train/epoch_loss", ep_avg, epoch)

        # ---- validation ----------------------------------------------------
        condi.eval()
        mae.eval()
        val_loss = val_n = 0

        train_mask_ratio = mae.mask_ratio          # save annealed value
        mae.mask_ratio   = mask_ratio_target       # always eval at target ratio

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                frames   = _unpack_condi_inputs(batch, device, cfg)
                out      = mae(frames, aug_cfg=aug_cfg, training=False)
                val_loss += out["loss"].item()
                val_n    += 1

        mae.mask_ratio = train_mask_ratio          # restore for next train epoch

        val_avg = val_loss / max(val_n, 1)
        writer.add_scalar("tmnet_val/loss", val_avg, epoch)
        print(
            f"[TMNet] train={ep_avg:.6f}  val={val_avg:.6f}  "
            f"mask={mae.mask_ratio:.3f}"
        )

        # ---- checkpoint — save TMNet only (decoder discarded after pretraining)
        if val_avg < best_val_loss:
            print(
                f"[TMNet] Improved val_loss {best_val_loss:.6f} → {val_avg:.6f} — saving."
            )
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

    # ---- compute and save embedding statistics over the full training set ---
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
    print(
        f"[TMNet] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  "
        f"saved to {stats_path}"
    )


# =============================================================================
# Stage 2b (variant) — TMNet predictive pretraining
# =============================================================================

def train_tmnet_predictive(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    vm:       torch.nn.Module,
    device:   torch.device,
) -> None:
    """
    Pretrain TMNet via next-frame prediction for one fold.

    The encoder is trained to predict frame t+1 from the hidden state at t,
    directly aligned with its downstream role as a temporal conditioning
    encoder.  No MAE decoder — simpler and less prone to overfitting.

    Checkpoint format is identical to train_tmnet so downstream code
    (train_CLDM, compute_stats) can load it without changes.
    """

    # ---- build model -------------------------------------------------------
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

    pred = PredictiveTMNet(
        tm_net    = condi,
        img_size     = img_size,
        img_channels = in_channels,
    ).to(device)

    print(
        f"\n[TMNet-Pred] PredictiveTMNet"
        f"  | output_dim={cfg.tmnet_pre_latent_dim}"
        f"  (device={next(condi.parameters()).device})"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[TMNet-Pred] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(condi, cfg.checkpoint, device)

    # ---- directories & logging ---------------------------------------------
    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(
            cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_pred",
        )
        run_dir = os.path.join(
            cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_pred",
        )
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    # ---- data loaders ------------------------------------------------------
    train_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp,
    )
    valid_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[1], nb_pred=cfg.tp, valid=True,
    )

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
    optimizer     = build_optimizer(cfg, pred.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        verbose=True,
        delta=cfg.early_stopping_delta,
    )

    aug_cfg       = _build_aug_cfg(cfg)
    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    print(f"[TMNet-Pred] Starting training from epoch {restart_epoch}...")
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

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            ep_loss  += loss.item()
            ep_steps += 1
            writer.add_scalar("tmnet_pred_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("tmnet_pred_train/epoch_loss", ep_avg, epoch)

        # ---- validation ----------------------------------------------------
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

        # ---- checkpoint — save TMNet only (decoder discarded) -----------
        if val_avg < best_val_loss:
            print(f"[TMNet-Pred] Improved val_loss {best_val_loss:.6f} → {val_avg:.6f} — saving.")
            best_val_loss = val_avg
            custom_save(condi, os.path.join(log_dir, "model_best_tmnet.pth"))
        else:
            print(f"[TMNet-Pred] No improvement from {best_val_loss:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            print("[TMNet-Pred] Early stopping triggered.")
            break

        print(f"[TMNet-Pred] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nStage 2b done.  Best checkpoint: {log_dir}/model_best_tmnet.pth")
    writer.close()

    # ---- compute and save embedding statistics ----------------------------
    print("[TMNet-Pred] Computing embedding statistics over full training set...")
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
    print(
        f"[TMNet-Pred] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  "
        f"saved to {stats_path}"
    )


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
    """
    Load an existing TMNet checkpoint and compute embedding statistics
    over the full training set.  Saves tmnet_stats.pt next to the checkpoint.

    NOTE: this function loads into TMNetEncoder + TMNetPriorMAE.
    It is NOT compatible with checkpoints from train_tmnet_priormulti_image
    (which saves TMNet_Tr_priormulti_image).  Those runs compute and save
    tmnet_stats.pt automatically at the end of training — no separate step needed.
    """
    log_dir = os.path.join(
        cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet",
    )
    ckpt = os.path.join(log_dir, "model_best_tmnet.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[compute_stats] Checkpoint not found: {ckpt}")

    condi, mae = _build_tmnet_model(cfg, device)
    custom_load(condi, ckpt, device)
    condi.eval()
    print(f"[compute_stats] Loaded TMNet from {ckpt}")

    train_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp,
    )
    loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
    )

    max_batches = getattr(cfg, "tmnet_stats_batches", 50)
    emb_sum = emb_sq_sum = emb_count = 0

    print(f"[compute_stats] Encoding up to {max_batches}/{len(loader)} batches...")
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
    print(
        f"\n[compute_stats] mean={emb_mean:.4f}  std={emb_std:.4f}  "
        f"→ saved to {stats_path}"
    )


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
    """
    Evaluate MAE reconstruction quality on the test fold.

    For each test sample:
      1. Apply a fixed mask to the navigator frame sequence.
      2. Encode + decode through TMNetPriorMAE.
      3. Measure MSE on masked pixels.
    Results saved under <logging_dir>/test_tmnet/<dir_name>/<fold_idx>/.
    """
    condi, mae = _build_tmnet_model(cfg, device)
    fold_root = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}")
    candidate_subdirs = ["tmnet", "tmnet_dvfsup2", "tmnet_dvfsup", "tmnet_mm"]
    log_dir = None
    for subdir in candidate_subdirs:
        candidate = os.path.join(fold_root, subdir)
        if os.path.exists(os.path.join(candidate, "model_best_tmnet.pth")):
            log_dir = candidate
            break
    if log_dir is None:
        raise FileNotFoundError(
            f"[test_tmnet] Checkpoint not found under {fold_root} "
            f"(tried: {candidate_subdirs})"
        )
    ckpt = os.path.join(log_dir, "model_best_tmnet.pth")

    custom_load(condi, ckpt, device)
    condi.eval()
    mae.eval()
    mae.mask_ratio = cfg.tmnet_mask_ratio    # fix to trained target
    print(f"[test_tmnet] Loaded TMNet from {ckpt}")

    test_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, sequence_list=fold, nb_pred=cfg.tp,
        nb_inputs=cfg.nb_inputs, test=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
    )

    save_dir = os.path.join(
        cfg.logging_dir, dir_name, "test_tmnet", fold_idx,
    )
    cond_mkdir(save_dir)

    test_patients = {
        seq.split("/")[0] if "/" in seq else seq[:8] for seq in fold
    }
    with open(os.path.join(save_dir, "patients_test.txt"), "w") as f:
        for p in sorted(test_patients):
            f.write(p + "\n")

    mse_all, mae_all, mape_all, psnr_all, ssim_all = [], [], [], [], []

    from pytorch_msssim import ssim as ssim_fn   # pip install pytorch-msssim


    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader)):
            frames = _unpack_condi_inputs(batch, device, cfg)

            torch.manual_seed(42)
            out = mae(frames)

            mask   = out["mask_mae"]    # (1, T, 1, H, W)
            recon  = out["recon"]       # (1, T, C, H, W)
            target = frames.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

            n_masked    = mask.sum()
            target_mean = (target * mask).sum() / n_masked
            var_target  = (((target - target_mean) ** 2) * mask).sum() / n_masked

            # ── existing ──────────────────────────────────────────────────────────
            nmse = (((recon - target) ** 2) * mask).sum() / n_masked / var_target
            mse_all.append(nmse.item())

            # ── MAE (mean absolute error, masked) ─────────────────────────────────
            mae_val = (torch.abs(recon - target) * mask).sum() / n_masked
            mae_all.append(mae_val.item())

            # ── MAPE (mean absolute percentage error, masked) ─────────────────────
            # Guard against near-zero targets to avoid division explosion
            eps = 1e-6
            pct_err = torch.abs((recon - target) / (target.abs() + eps)) * mask
            mape_val = pct_err.sum() / n_masked * 100.0          # in %
            mape_all.append(mape_val.item())

            # ── PSNR (masked MSE → dB) ────────────────────────────────────────────
            raw_mse  = ((recon - target) ** 2 * mask).sum() / n_masked
            # Assumes pixel range [0, 1]; change max_val if your data is in [-1,1] etc.
            max_val  = 1.0
            psnr_val = 10.0 * torch.log10(max_val ** 2 / (raw_mse + 1e-8))
            psnr_all.append(psnr_val.item())

            # ── SSIM (frame-averaged, full frames — SSIM needs spatial context) ───
            # Collapse T into batch dim: (B*T, C, H, W)
            B, T, C, H, W = target.shape
            t_flat = target.view(B * T, C, H, W).clamp(0, 1)
            r_flat = recon.view(B * T, C, H, W).clamp(0, 1)
            ssim_val = ssim_fn(r_flat, t_flat, data_range=1.0, size_average=True)
            ssim_all.append(ssim_val.item())

    # ── numpy arrays ──────────────────────────────────────────────────────────────
    mse_arr  = np.array(mse_all)
    rmse_arr = np.sqrt(mse_arr)
    mae_arr  = np.array(mae_all)
    mape_arr = np.array(mape_all)
    psnr_arr = np.array(psnr_all)
    ssim_arr = np.array(ssim_all)

    # ── save & summarise ──────────────────────────────────────────────────────────
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
        f.write("TMNet MAE — test reconstruction metrics\n")
        f.write("=" * 56 + "\n")
        f.write(f"Fold:           {fold_idx}\n")
        f.write(f"Checkpoint:     {ckpt}\n")
        f.write(f"Model:          {type(condi).__name__}\n")
        f.write(f"Mask ratio:     {cfg.tmnet_mask_ratio}\n")
        f.write(f"Patch size:     {cfg.tmnet_patch_size}\n")
        f.write(f"N test samples: {len(mse_all)}\n\n")
        f.write(f"NMSE  (masked)   {_fmt(mse_arr)}   [0=perfect, 1=mean-pred baseline]\n")
        f.write(f"RMSE  (masked)   {_fmt(rmse_arr)}   [same units as signal σ]\n")
        f.write(f"MAE   (masked)   {_fmt(mae_arr)}   [absolute pixel units]\n")
        f.write(f"MAPE  (masked)   {_fmt(mape_arr, decimals=2, unit=' %')}   [% error; sensitive to dark pixels]\n")
        f.write(f"PSNR  (frames)   {_fmt(psnr_arr, decimals=2)}   dB  [higher=better; >30 dB good]\n")
        f.write(f"SSIM  (frames)   {_fmt(ssim_arr, decimals=4)}   [0–1, higher=better]\n")

    print(
        f"[test_tmnet] "
        f"NMSE={np.nanmean(mse_arr):.4f}  "
        f"RMSE={np.nanmean(rmse_arr):.4f}  "
        f"MAE={np.nanmean(mae_arr):.4f}  "
        f"MAPE={np.nanmean(mape_arr):.2f}%  "
        f"PSNR={np.nanmean(psnr_arr):.2f}dB  "
        f"SSIM={np.nanmean(ssim_arr):.4f}"
        f"  — summary → {summary_path}"
    )


def test_tmnet_dvfsup2(
    cfg,
    fold:     list,
    fold_idx: str,
    dir_name: str,
    vm:       torch.nn.Module,
    device:   torch.device,
) -> None:
    """
    Evaluate TMNet_Tr_priormulti_image (tmnet_dvfsup2/) on the test fold
    using BOTH DVF-space and image-space metrics.

    Image-space metrics (NMSE, MAE, MAPE, PSNR, SSIM) are computed on the
    warped prediction I_pred vs. the ground-truth future frame Igt, mirroring
    test_tmnet's reconstruction metrics. This makes results comparable
    across registration backbones (VoxelMorph, MambaMorph, ...) since the
    comparison happens in image space rather than in each backbone's own
    DVF units/scale.

    DVF MSE is still reported for reference/diagnosis, but should not be used
    to compare across checkpoints trained against different registration
    backbones.
    """
    from mopred.data.data_loaders.navigator_4d import Y_NAV
    from pytorch_msssim import ssim as ssim_fn

    vol_size   = tuple(cfg.vol_size)
    condi_type = str(getattr(cfg, "tmnet_condi_type", "2"))
    img_h, img_w = (
        (vol_size[1], vol_size[2]) if condi_type == "1"
        else (vol_size[0], vol_size[2])
    )
    dec_layers         = getattr(cfg, "tmnet_Tr_dec_layers", 3)
    prior_type         = getattr(cfg, "tmnet_prior_type", "learned")
    phi_loss_weight    = float(getattr(cfg, "tmnet_phi_loss_weight", 0.0))
    amp_loss_weight    = float(getattr(cfg, "tmnet_amp_loss_weight", 0.0))
    motion_loss_weight = float(getattr(cfg, "tmnet_motion_loss_weight", 0.0))

    model = TMNet_Tr_priormulti_image(
        num_inputs       = cfg.tmnet_num_frames,
        horizon          = cfg.tmnet_horizon,
        in_channels      = 2,
        out_channels     = cfg.tmnet_condi_channels,
        n_heads          = cfg.tmnet_Tr_n_heads,
        enc_layers       = cfg.tmnet_Tr_enc_layers,
        dec_layers       = dec_layers,
        normalize_before = cfg.tmnet_Tr_norm_before,
        output_dim       = cfg.tmnet_pre_latent_dim,
        rnn              = "transformer",
        condi_type       = condi_type,
        prior_type       = prior_type,
        device           = device,
        img_h            = img_h,
        img_w            = img_w,
        use_phi_loss     = phi_loss_weight > 0,
        use_amp_loss     = amp_loss_weight > 0,
        use_motion_loss  = motion_loss_weight > 0,
    ).to(device)

    ckpt = os.path.join(
        cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}",
        "tmnet_dvfsup2", "model_best_tmnet.pth",
    )
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[test_tmnet_dvfsup2] Checkpoint not found: {ckpt}")
    custom_load(model, ckpt, device)
    model.eval()
    print(f"[test_tmnet_dvfsup2] Loaded model from {ckpt}")

    def _dvf_slice(dvf: torch.Tensor) -> torch.Tensor:
        if condi_type == "2":
            return dvf[:, [0, 2], :, Y_NAV, :]
        else:
            return dvf[:, [1, 2], 16, :, :]

    test_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, sequence_list=fold, nb_pred=cfg.tp,
        nb_inputs=cfg.nb_inputs, test=True,
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    save_dir = os.path.join(cfg.logging_dir, dir_name, "test_tmnet_dvfsup2", fold_idx)
    cond_mkdir(save_dir)

    test_patients = {seq.split("/")[0] if "/" in seq else seq[:8] for seq in fold}
    with open(os.path.join(save_dir, "patients_test.txt"), "w") as f:
        for p in sorted(test_patients):
            f.write(p + "\n")

    # ---- accumulators --------------------------------------------------
    dvf_mse_all = []
    img_nmse_all, img_mae_all, img_mape_all, img_psnr_all, img_ssim_all = [], [], [], [], []
    baseline_nmse_all = []   # zero-motion baseline (I_pred = Iref), in image space

    with torch.no_grad():
        for batch in tqdm(test_loader):
            Iref, Ipast, _, Igt = _build_priormulti_image_inputs(batch, device, cfg)

            # ---- DVF ground truth (for the diagnostic DVF-MSE only) ----
            ref_vol = batch[0].unsqueeze(1).to(device)
            dvf_gt_list = [
                _dvf_slice(vm(ref_vol, batch[2][t].unsqueeze(1).to(device)))
                for t in range(cfg.tp)
            ]
            DVF_seq, I_pred, _, _, _, _, _ = model(Iref, Ipast)  # prior path

            dvf_loss = sum(
                torch.nn.functional.mse_loss(DVF_seq[:, :, t], dvf_gt_list[t])
                for t in range(cfg.tp)
            ) / cfg.tp
            dvf_mse_all.append(dvf_loss.item())

            # ---- image-space metrics, per horizon step, averaged -------
            eps = 1e-6
            for t in range(cfg.tp):
                pred_t   = I_pred[:, :, t, :, :].clamp(0, 1)   # (B,1,H,W)
                target_t = Igt[:, :, t, :, :].clamp(0, 1)

                mse_t  = torch.nn.functional.mse_loss(pred_t, target_t)
                var_t  = target_t.var(unbiased=False).clamp_min(eps)
                nmse_t = mse_t / var_t
                mae_t  = torch.abs(pred_t - target_t).mean()
                mape_t = (torch.abs((pred_t - target_t) / (target_t.abs() + eps))).mean() * 100.0
                psnr_t = 10.0 * torch.log10(1.0 / (mse_t + 1e-8))
                ssim_t = ssim_fn(pred_t, target_t, data_range=1.0, size_average=True)

                img_nmse_all.append(nmse_t.item())
                img_mae_all.append(mae_t.item())
                img_mape_all.append(mape_t.item())
                img_psnr_all.append(psnr_t.item())
                img_ssim_all.append(ssim_t.item())

                # zero-motion baseline: predict Iref itself, no warping
                ref_t = Iref.clamp(0, 1)
                base_mse_t  = torch.nn.functional.mse_loss(ref_t, target_t)
                baseline_nmse_all.append((base_mse_t / var_t).item())

    # ---- numpy arrays ----------------------------------------------------
    dvf_arr        = np.array(dvf_mse_all)
    img_nmse_arr   = np.array(img_nmse_all)
    img_rmse_arr   = np.sqrt(img_nmse_arr * np.nan)  # placeholder, see note below
    img_mae_arr    = np.array(img_mae_all)
    img_mape_arr   = np.array(img_mape_all)
    img_psnr_arr   = np.array(img_psnr_all)
    img_ssim_arr   = np.array(img_ssim_all)
    baseline_arr   = np.array(baseline_nmse_all)

    np.save(os.path.join(save_dir, "DVF_MSE.npy"),        dvf_arr)
    np.save(os.path.join(save_dir, "IMG_NMSE.npy"),       img_nmse_arr)
    np.save(os.path.join(save_dir, "IMG_MAE.npy"),        img_mae_arr)
    np.save(os.path.join(save_dir, "IMG_MAPE.npy"),       img_mape_arr)
    np.save(os.path.join(save_dir, "IMG_PSNR.npy"),       img_psnr_arr)
    np.save(os.path.join(save_dir, "IMG_SSIM.npy"),       img_ssim_arr)
    np.save(os.path.join(save_dir, "IMG_NMSE_baseline.npy"), baseline_arr)

    def _fmt(arr, decimals=6, unit=""):
        return (f"{np.nanmean(arr):.{decimals}f} ± {np.nanstd(arr):.{decimals}f}"
                f"  |  median {np.nanmedian(arr):.{decimals}f}{unit}")

    skill_score = 1.0 - (np.nanmean(img_nmse_arr) / np.nanmean(baseline_arr))

    summary_path = os.path.join(save_dir, "summary_metrics.txt")
    with open(summary_path, "w") as f:
        f.write("TMNet DVFSup2 — test metrics (image-space primary, DVF diagnostic)\n")
        f.write("=" * 68 + "\n")
        f.write(f"Fold:           {fold_idx}\n")
        f.write(f"Checkpoint:     {ckpt}\n")
        f.write(f"N test samples: {len(dvf_mse_all)}\n\n")

        f.write("-- Image-space (predicted vs. ground-truth warped frame) --\n")
        f.write(f"NMSE (img)       {_fmt(img_nmse_arr)}   [0=perfect, ~1=zero-motion baseline]\n")
        f.write(f"MAE  (img)       {_fmt(img_mae_arr)}    [absolute pixel units]\n")
        f.write(f"MAPE (img)       {_fmt(img_mape_arr, decimals=2, unit=' %')}\n")
        f.write(f"PSNR (img)       {_fmt(img_psnr_arr, decimals=2)}   dB  [higher=better]\n")
        f.write(f"SSIM (img)       {_fmt(img_ssim_arr, decimals=4)}   [0-1, higher=better]\n")
        f.write(f"Zero-motion baseline NMSE   {_fmt(baseline_arr)}\n")
        f.write(f"Skill score (1 - NMSE/baseline_NMSE): {skill_score:.4f}"
                f"   [>0 = better than predicting no motion, <=0 = no better/worse]\n\n")

        f.write("-- DVF-space (diagnostic only — NOT comparable across registration backbones) --\n")
        f.write(f"DVF MSE (prior)  mean={np.mean(dvf_arr):.6f}  std={np.std(dvf_arr):.6f}"
                f"  median={np.median(dvf_arr):.6f}\n")

    print(
        f"[test_tmnet_dvfsup2] "
        f"IMG_NMSE={np.nanmean(img_nmse_arr):.4f}  "
        f"PSNR={np.nanmean(img_psnr_arr):.2f}dB  "
        f"SSIM={np.nanmean(img_ssim_arr):.4f}  "
        f"skill={skill_score:.4f}  "
        f"DVF_MSE={np.mean(dvf_arr):.6f}"
        f"  — summary → {summary_path}"
    )


# =============================================================================
# Stage 2b (variant) — MopTR-style image prediction pretraining
# =============================================================================

def _build_mopTR_inputs(batch, device, cfg):
    """
    Build Iseq (B, 2, T, H, W) and Igt (B, 1, Q, H, W) from a raw
    NAVIGATOR_4D_Dataset_multitime batch, mirroring MopTR's build_inputs.

    condi_type "1" → sagittal slice at d = sag_pos (default 16)
    condi_type "2" → coronal   slice at h = cor_pos (default 32)
    """
    ref_raw, input_volume_list, current_volume_list, *_ = batch
    ref_vol = ref_raw.unsqueeze(1).to(device)   # (B, 1, D, H, W)

    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    sag_pos    = getattr(cfg, "sag_pos", 16)
    cor_pos    = getattr(cfg, "cor_pos", 32)

    frames_list = []
    igt_list    = []

    if condi_type == "1":   # sagittal
        c_ref = ref_vol[:, :, sag_pos, :, :]       # (B, 1, H, W)
        for q in range(cfg.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1).to(device)[:, :, sag_pos, :, :]
            frames_list.append(torch.cat([c_t, c_ref], dim=1))
        Iseq = torch.stack(frames_list, dim=2).float()   # (B, 2, T, H, W)
        for i in range(cfg.tp):
            igt_list.append(
                current_volume_list[i].unsqueeze(1).to(device)[:, :, sag_pos, :, :]
            )
    else:                   # coronal
        c_ref = ref_vol[:, :, :, cor_pos, :]       # (B, 1, D, W)
        for q in range(cfg.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1).to(device)[:, :, :, cor_pos, :]
            frames_list.append(torch.cat([c_t, c_ref], dim=1))
        Iseq = torch.stack(frames_list, dim=2).float()   # (B, 2, T, D, W)
        for i in range(cfg.tp):
            igt_list.append(
                current_volume_list[i].unsqueeze(1).to(device)[:, :, :, cor_pos, :]
            )

    Igt = torch.stack(igt_list, dim=2).float()       # (B, 1, Q, H, W)
    return Iseq, Igt


def train_tmnet_mopTR_style(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    device:   torch.device,
) -> None:
    """
    Pretrain TMNet with MopTR-style image prediction for one fold.

    Unlike the MAE / predictive variants, this function:
      - Uses raw NAVIGATOR_4D_Dataset_multitime (no DVF cache needed).
      - Builds Iseq (past slices concatenated with reference) and
        Igt (ground-truth future slices) exactly as in the MopTR training loop.
      - Trains with MSE loss between the decoded predictions and Igt.

    Checkpoint format is identical to train_tmnet so downstream code
    (train_CLDM, compute_stats) can load it without changes.
    """

    # ---- determine spatial dimensions -----------------------------------
    vol_size   = tuple(cfg.vol_size)           # (D, H, W)
    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    if condi_type == "1":
        img_size = (vol_size[1], vol_size[2])  # sagittal: (H, W)
    else:
        img_size = (vol_size[0], vol_size[2])  # coronal:  (D, W)

    # ---- build model -------------------------------------------------------
    condi = TMNetEncoder(
        num_inputs       = cfg.tmnet_num_frames,
        horizon          = cfg.tmnet_horizon,
        in_channels      = 2,
        out_channels     = cfg.tmnet_condi_channels,
        n_heads          = cfg.tmnet_Tr_n_heads,
        enc_layers       = cfg.tmnet_Tr_enc_layers,
        normalize_before = cfg.tmnet_Tr_norm_before,
        output_dim       = cfg.tmnet_pre_latent_dim,
        condi_type       = condi_type,
        device           = device,
    ).to(device)

    model = MopTRTMNet(
        tm_net   = condi,
        num_queries = cfg.tp,
        img_size    = img_size,
    ).to(device)

    print(
        f"\n[TMNet-MopTR] TMNetEncoder + MopTRTMNet"
        f"  | img_size={img_size}  | num_queries={cfg.tp}"
        f"  | output_dim={cfg.tmnet_pre_latent_dim}"
        f"  (device={next(condi.parameters()).device})"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[TMNet-MopTR] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(condi, cfg.checkpoint, device)

    # ---- directories & logging ---------------------------------------------
    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(
            cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_mopTR",
        )
        run_dir = os.path.join(
            cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_mopTR",
        )
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    # ---- data loaders — raw volumes, no DVF cache --------------------------
    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    train_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp,
    )
    valid_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[1], nb_pred=cfg.tp, valid=True,
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

    # ---- optimiser & scheduler ---------------------------------------------
    optimizer     = build_optimizer(cfg, model.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience = cfg.early_stopping_patience,
        verbose  = True,
        delta    = cfg.early_stopping_delta,
    )

    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    print(f"[TMNet-MopTR] Starting training from epoch {restart_epoch}...")
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

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            ep_loss  += loss.item()
            ep_steps += 1
            writer.add_scalar("tmnet_mopTR_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("tmnet_mopTR_train/epoch_loss", ep_avg, epoch)

        # ---- validation ----------------------------------------------------
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
            print(f"[TMNet-MopTR] Improved val_loss {best_val_loss:.6f} → {val_avg:.6f} — saving.")
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

    # ---- compute and save embedding statistics ----------------------------
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
    print(
        f"[TMNet-MopTR] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  "
        f"saved to {stats_path}"
    )


# =============================================================================
# Stage 2b (variant) — TMNet_Tr_priormulti with image supervision
# =============================================================================

def _aux_targets_from_batch(batch, device) -> tuple:
    """
    Extract per-sample phi for every predicted future frame.

    batch[-1] is [nb_pred][batch_size] file paths.
    Returns phi_gt (B, H) and amp_gt (B,) on `device`.
    phi_gt[:, h] is the respiratory phase at the (h+1)-th future frame,
    matching the per-horizon phi_pred produced by TM_Net.phi_head.
    """
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


def _build_priormulti_image_inputs(batch, device, cfg):
    """
    Build Iref, Ipast, Ifuture_2ch, Igt from a raw dataset batch for
    TMNet_Tr_priormulti_image training.

    Returns
    -------
    Iref        : (B, 1, H, W)     reference slice  (for SpatialTransformer)
    Ipast       : (B, 2, T, H, W)  past slices with reference concatenated
    Ifuture_2ch : (B, 2, Q, H, W)  future slices with reference concatenated (posterior)
    Igt         : (B, 1, Q, H, W)  raw future slices (reconstruction target)
    """
    ref_raw, input_volume_list, current_volume_list, *_ = batch
    ref_vol = ref_raw.unsqueeze(1).to(device)    # (B, 1, D, H, W)

    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    sag_pos    = getattr(cfg, "sag_pos", 16)
    cor_pos    = getattr(cfg, "cor_pos", 32)

    if condi_type == "1":   # sagittal
        c_ref = ref_vol[:, :, sag_pos, :, :]       # (B, 1, H, W)
        past_list, future_list, igt_list = [], [], []
        for q in range(cfg.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1).to(device)[:, :, sag_pos, :, :]
            past_list.append(torch.cat([c_t, c_ref], dim=1))
        for i in range(cfg.tp):
            c_f = current_volume_list[i].unsqueeze(1).to(device)[:, :, sag_pos, :, :]
            future_list.append(torch.cat([c_f, c_ref], dim=1))
            igt_list.append(c_f)
    else:                   # coronal
        c_ref = ref_vol[:, :, :, cor_pos, :]       # (B, 1, D, W)
        past_list, future_list, igt_list = [], [], []
        for q in range(cfg.nb_inputs):
            c_t = input_volume_list[q].unsqueeze(1).to(device)[:, :, :, cor_pos, :]
            past_list.append(torch.cat([c_t, c_ref], dim=1))
        for i in range(cfg.tp):
            c_f = current_volume_list[i].unsqueeze(1).to(device)[:, :, :, cor_pos, :]
            future_list.append(torch.cat([c_f, c_ref], dim=1))
            igt_list.append(c_f)

    Ipast       = torch.stack(past_list,   dim=2).float()   # (B, 2, T, H, W)
    Ifuture_2ch = torch.stack(future_list, dim=2).float()   # (B, 2, Q, H, W)
    Igt         = torch.stack(igt_list,    dim=2).float()   # (B, 1, Q, H, W)
    Iref        = c_ref.float()                             # (B, 1, H, W)
    return Iref, Ipast, Ifuture_2ch, Igt


def train_tmnet_priormulti_dvf(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    vm:       torch.nn.Module,
    device:   torch.device,
) -> None:
    """
    Train TMNet_Tr_priormulti with DVF supervision + KL regularization.

    Same architecture as train_tmnet_priormulti_image (prior/posterior + KL),
    but supervises on the 2-D DVF slice at the navigator plane instead of the
    warped image.  Uses DVF-space features with a smooth KL-regularized latent,
    without a full CVAE training setup.

    Loss: MSE(DVF_seq_pred, dvf_slice_gt) + β * KL
      where β is linearly annealed from 0 → kl_beta over kl_warmup_epochs.

    DVF targets are computed on-the-fly from a frozen VoxelMorph (same as
    train_tmnet_dvfsup).  Teacher forcing is used during training (posterior
    path); validation uses the prior (mirrors test time).

    Checkpoint saved as model_best_tmnet.pth — compatible with _load_tmnet.
    """
    from mopred.data.data_loaders.navigator_4d import Y_NAV

    vol_size   = tuple(cfg.vol_size)
    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    img_h, img_w = (
        (vol_size[1], vol_size[2]) if condi_type == "1"
        else (vol_size[0], vol_size[2])
    )

    dec_layers      = getattr(cfg, "tmnet_Tr_dec_layers", 3)
    prior_type      = getattr(cfg, "tmnet_prior_type", "learned")
    kl_beta         = float(getattr(cfg, "tmnet_kl_beta", 1.0))
    kl_warmup_ep    = int(getattr(cfg, "tmnet_kl_warmup_epochs", 10))
    phi_loss_weight    = float(getattr(cfg, "tmnet_phi_loss_weight", 0.0))
    amp_loss_weight    = float(getattr(cfg, "tmnet_amp_loss_weight", 0.0))
    motion_loss_weight = float(getattr(cfg, "tmnet_motion_loss_weight", 0.0))
    
    aug_cfg = _build_aug_cfg(cfg)
    if aug_cfg.any_enabled():
        print(
            f"[TMNet-DVFSup2] Temporal augmentations: "
            f"frame_drop={aug_cfg.frame_drop.enabled}, "
            f"spatial_mask={aug_cfg.spatial_mask.enabled}, "
            f"tube_mask={aug_cfg.tube_mask.enabled}, "
            f"variable_frames={aug_cfg.variable_frames.enabled}"
        )
    else:
        print("[TMNet-DVFSup2] No temporal augmentations enabled.")

    model = TMNet_Tr_priormulti_image(
        num_inputs       = cfg.tmnet_num_frames,
        horizon          = cfg.tmnet_horizon,
        in_channels      = 2,
        out_channels     = cfg.tmnet_condi_channels,
        n_heads          = cfg.tmnet_Tr_n_heads,
        enc_layers       = cfg.tmnet_Tr_enc_layers,
        dec_layers       = dec_layers,
        normalize_before = cfg.tmnet_Tr_norm_before,
        output_dim       = cfg.tmnet_pre_latent_dim,
        rnn              = "transformer",
        condi_type       = condi_type,
        prior_type       = prior_type,
        device           = device,
        img_h            = img_h,
        img_w            = img_w,
        use_phi_loss     = phi_loss_weight > 0,
        use_amp_loss     = amp_loss_weight > 0,
        use_motion_loss  = motion_loss_weight > 0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n[TMNet-DVFSup2] TMNet_Tr_priormulti_image (DVF supervision)"
        f"  | img_size=({img_h},{img_w})  | prior={prior_type}"
        f"  | kl_beta={kl_beta}  | params={n_params:,}"
        f"  | phi_w={phi_loss_weight}  | amp_w={amp_loss_weight}"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[TMNet-DVFSup2] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(model, cfg.checkpoint, device)

    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(
            cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_dvfsup2",
        )
        run_dir = os.path.join(
            cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_dvfsup2",
        )
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

    train_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp,
    )
    valid_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[1], nb_pred=cfg.tp, valid=True,
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

    optimizer     = build_optimizer(cfg, model.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience = cfg.early_stopping_patience,
        verbose  = True,
        delta    = cfg.early_stopping_delta,
    )

    def _dvf_slice(dvf: torch.Tensor) -> torch.Tensor:
        """3-D DVF (B,3,D,H,W) → in-plane 2-D slice (B,2,dvf_h,dvf_w)."""
        if condi_type == "2":                      # coronal: D-W plane
            return dvf[:, [0, 2], :, Y_NAV, :]    # (dD, dW)
        else:                                      # sagittal: H-W plane
            return dvf[:, [1, 2], 16, :, :]        # (dH, dW)

    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    print(f"[TMNet-DVFSup2] Starting training from epoch {restart_epoch}...")
    print(f"\nStage 2b — TMNet DVF-supervision + KL  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet-DVFSup2] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        model.train()

        beta = kl_beta * min(1.0, (epoch + 1) / max(kl_warmup_ep, 1))
        writer.add_scalar("tmnet_dvfsup2_train/kl_beta", beta, epoch)

        ep_dvf = ep_kl = ep_phi = ep_amp = ep_motion = ep_steps = 0

        for batch in tqdm(train_loader):
            Iref, Ipast, Ifuture_2ch, _ = _build_priormulti_image_inputs(batch, device, cfg)
            Ipast = apply_pixel_space_augmentations(Ipast, aug_cfg, training=True)
            # is_zeroed = (Ipast[:, 0] == 0).all(dim=(-2, -1))   # (B, T) bool
            # n_zeroed  = is_zeroed.sum().item()
            # n_total   = is_zeroed.numel()
            # print(f"[TMNet-DVFSup2] zeroed frames: {n_zeroed}/{n_total} ({100*n_zeroed/n_total:.1f}%)")

            ref_vol = batch[0].unsqueeze(1).to(device)
            with torch.no_grad():
                dvf_gt_list = [
                    _dvf_slice(vm(ref_vol, batch[2][t].unsqueeze(1).to(device)))
                    for t in range(cfg.tp)
                ]  # each (B, 2, dvf_h, dvf_w) — DVF from ref to its own future frame

            optimizer.zero_grad()
            DVF_seq, _, kl_loss, phi_pred, amp_pred, motion_map_pred, _ = model(Iref, Ipast, Ifuture_2ch)  # posterior

            dvf_loss = torch.tensor(0.0, device=device)
            for t in range(cfg.tp):
                dvf_loss = dvf_loss + torch.nn.functional.mse_loss(DVF_seq[:, :, t], dvf_gt_list[t])
            dvf_loss = dvf_loss / cfg.tp

            loss = dvf_loss + (beta * kl_loss if kl_loss is not None else 0.0)

            phi_loss = amp_loss = motion_loss = torch.tensor(0.0, device=device)
            if phi_pred is not None or amp_pred is not None:
                phi_gt, amp_gt = _aux_targets_from_batch(batch, device)
                if phi_pred is not None:
                    phi_loss = torch.nn.functional.mse_loss(phi_pred, phi_gt)
                    loss = loss + phi_loss_weight * phi_loss
                if amp_pred is not None:
                    amp_loss = torch.nn.functional.mse_loss(amp_pred, amp_gt)
                    loss = loss + amp_loss_weight * amp_loss
            if motion_map_pred is not None:
                # Target: |c_last - c_ref| pooled to the same spatial resolution as motion_dec output
                motion_target = (Ipast[:, 0:1, -1] - Ipast[:, 1:2, -1]).abs()  # (B, 1, H, W)
                motion_target_pooled = torch.nn.functional.adaptive_avg_pool2d(
                    motion_target, (model._motion_pool_h, model._motion_pool_w)
                )
                motion_loss = torch.nn.functional.mse_loss(motion_map_pred, motion_target_pooled)
                loss = loss + motion_loss_weight * motion_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            ep_dvf    += dvf_loss.item()
            ep_kl     += kl_loss.item() if kl_loss is not None else 0.0
            ep_phi    += phi_loss.item()
            ep_amp    += amp_loss.item()
            ep_motion += motion_loss.item()
            ep_steps  += 1
            writer.add_scalar("tmnet_dvfsup2_train/loss",     loss.item(),     global_step)
            writer.add_scalar("tmnet_dvfsup2_train/dvf_loss", dvf_loss.item(), global_step)
            writer.add_scalar("tmnet_dvfsup2_train/kl_loss",  kl_loss.item() if kl_loss is not None else 0.0, global_step)
            if phi_pred is not None:
                writer.add_scalar("tmnet_dvfsup2_train/phi_loss",    phi_loss.item(),    global_step)
            if amp_pred is not None:
                writer.add_scalar("tmnet_dvfsup2_train/amp_loss",    amp_loss.item(),    global_step)
            if motion_map_pred is not None:
                writer.add_scalar("tmnet_dvfsup2_train/motion_loss", motion_loss.item(), global_step)
            global_step += 1

        ep_dvf_avg = ep_dvf / max(ep_steps, 1)
        ep_kl_avg  = ep_kl  / max(ep_steps, 1)
        ep_phi_avg = ep_phi / max(ep_steps, 1)
        ep_amp_avg = ep_amp / max(ep_steps, 1)
        writer.add_scalar("tmnet_dvfsup2_train/epoch_dvf_loss", ep_dvf_avg, epoch)
        writer.add_scalar("tmnet_dvfsup2_train/epoch_kl_loss",  ep_kl_avg,  epoch)
        ep_motion_avg = ep_motion / max(ep_steps, 1)
        if phi_loss_weight > 0:
            writer.add_scalar("tmnet_dvfsup2_train/epoch_phi_loss",    ep_phi_avg,    epoch)
        if amp_loss_weight > 0:
            writer.add_scalar("tmnet_dvfsup2_train/epoch_amp_loss",    ep_amp_avg,    epoch)
        if motion_loss_weight > 0:
            writer.add_scalar("tmnet_dvfsup2_train/epoch_motion_loss", ep_motion_avg, epoch)

        # ---- validation: prior path (no Ifuture → test-time behaviour) -----
        model.eval()
        val_dvf = val_phi = val_amp = val_motion = val_n = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                Iref, Ipast, _, _ = _build_priormulti_image_inputs(batch, device, cfg)
                ref_vol = batch[0].unsqueeze(1).to(device)
                dvf_gt_list = [
                    _dvf_slice(vm(ref_vol, batch[2][t].unsqueeze(1).to(device)))
                    for t in range(cfg.tp)
                ]

                DVF_seq, _, _, phi_pred, amp_pred, motion_map_pred, _ = model(Iref, Ipast)  # prior

                r = torch.tensor(0.0, device=device)
                for t in range(cfg.tp):
                    r = r + torch.nn.functional.mse_loss(DVF_seq[:, :, t], dvf_gt_list[t])
                val_dvf += (r / cfg.tp).item()

                if phi_pred is not None or amp_pred is not None:
                    phi_gt, amp_gt = _aux_targets_from_batch(batch, device)
                    if phi_pred is not None:
                        val_phi += torch.nn.functional.mse_loss(phi_pred, phi_gt).item()
                    if amp_pred is not None:
                        val_amp += torch.nn.functional.mse_loss(amp_pred, amp_gt).item()
                if motion_map_pred is not None:
                    motion_target = (Ipast[:, 0:1, -1] - Ipast[:, 1:2, -1]).abs()
                    motion_target_pooled = torch.nn.functional.adaptive_avg_pool2d(
                        motion_target, (model._motion_pool_h, model._motion_pool_w)
                    )
                    val_motion += torch.nn.functional.mse_loss(
                        motion_map_pred, motion_target_pooled
                    ).item()
                val_n += 1

        val_avg        = val_dvf    / max(val_n, 1)
        val_phi_avg    = val_phi    / max(val_n, 1)
        val_amp_avg    = val_amp    / max(val_n, 1)
        val_motion_avg = val_motion / max(val_n, 1)
        writer.add_scalar("tmnet_dvfsup2_val/dvf_loss", val_avg, epoch)
        if phi_loss_weight > 0:
            writer.add_scalar("tmnet_dvfsup2_val/phi_loss",    val_phi_avg,    epoch)
        if amp_loss_weight > 0:
            writer.add_scalar("tmnet_dvfsup2_val/amp_loss",    val_amp_avg,    epoch)
        if motion_loss_weight > 0:
            writer.add_scalar("tmnet_dvfsup2_val/motion_loss", val_motion_avg, epoch)

        aux_str = ""
        if phi_loss_weight > 0:
            aux_str += f"  phi_tr={ep_phi_avg:.4f}/val={val_phi_avg:.4f}(x{phi_loss_weight})"
        if amp_loss_weight > 0:
            aux_str += f"  amp_tr={ep_amp_avg:.4f}/val={val_amp_avg:.4f}(x{amp_loss_weight})"
        if motion_loss_weight > 0:
            aux_str += f"  motion_tr={ep_motion_avg:.4f}/val={val_motion_avg:.4f}(x{motion_loss_weight})"
        print(
            f"[TMNet-DVFSup2] train_dvf={ep_dvf_avg:.6f}  train_kl={ep_kl_avg:.6f}"
            f"{aux_str}  val_dvf={val_avg:.6f} (prior)  β={beta:.3f}"
        )

        if val_avg < best_val_loss:
            print(f"[TMNet-DVFSup2] Improved {best_val_loss:.6f} → {val_avg:.6f} — saving.")
            best_val_loss = val_avg
            custom_save(model, os.path.join(log_dir, "model_best_tmnet.pth"))
        else:
            print(f"[TMNet-DVFSup2] No improvement from {best_val_loss:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            print("[TMNet-DVFSup2] Early stopping triggered.")
            break

        print(f"[TMNet-DVFSup2] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nStage 2b done.  Best checkpoint: {log_dir}/model_best_tmnet.pth")
    writer.close()

    print("[TMNet-DVFSup2] Computing embedding statistics over full training set...")
    custom_load(model, os.path.join(log_dir, "model_best_tmnet.pth"), device)
    model.eval()

    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            _, Ipast, _, _ = _build_priormulti_image_inputs(batch, device, cfg)
            emb_list = model.encode(Ipast)
            for emb in emb_list:
                emb_sum    += emb.sum().item()
                emb_sq_sum += emb.pow(2).sum().item()
                emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)

    stats_path = os.path.join(log_dir, "tmnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(
        f"[TMNet-DVFSup2] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  "
        f"saved to {stats_path}"
    )


def train_tmnet_priormulti_image(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    device:   torch.device,
) -> None:
    """
    Train TMNet_Tr_priormulti with image supervision (DVF + warping head).

    Architecture: same Transformer encoder/decoder + prior/posterior as
    TMNet_Tr_priormulti, but the output head is replaced by:
        DVFProjector  →  SpatialTransformer(Iref)  →  I_pred

    Loss: recon(I_pred, I_gt) + β * KL
      where β is linearly annealed from 0 → kl_beta over kl_warmup_epochs.

    No DVF cache needed.  Checkpoint saved as model_best_tmnet.pth so that
    downstream train_CLDM can load it directly.
    """
    vol_size   = tuple(cfg.vol_size)
    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    img_h, img_w = (
        (vol_size[1], vol_size[2]) if condi_type == "1"   # sagittal
        else (vol_size[0], vol_size[2])                    # coronal
    )

    dec_layers      = getattr(cfg, "tmnet_Tr_dec_layers", 3)
    prior_type      = getattr(cfg, "tmnet_prior_type", "learned")
    img_loss        = getattr(cfg, "tmnet_img_loss", "mse")
    kl_beta         = float(getattr(cfg, "tmnet_kl_beta", 1.0))
    kl_warmup_ep    = int(getattr(cfg, "tmnet_kl_warmup_epochs", 10))
    phi_loss_weight = float(getattr(cfg, "tmnet_phi_loss_weight", 0.0))
    amp_loss_weight = float(getattr(cfg, "tmnet_amp_loss_weight", 0.0))

    # ---- build model -------------------------------------------------------
    model = TMNet_Tr_priormulti_image(
        num_inputs       = cfg.tmnet_num_frames,
        horizon          = cfg.tmnet_horizon,
        in_channels      = 2,
        out_channels     = cfg.tmnet_condi_channels,
        n_heads          = cfg.tmnet_Tr_n_heads,
        enc_layers       = cfg.tmnet_Tr_enc_layers,
        dec_layers       = dec_layers,
        normalize_before = cfg.tmnet_Tr_norm_before,
        output_dim       = cfg.tmnet_pre_latent_dim,
        rnn              = "transformer",
        condi_type       = condi_type,
        prior_type       = prior_type,
        device           = device,
        img_h            = img_h,
        img_w            = img_w,
        use_phi_loss     = phi_loss_weight > 0,
        use_amp_loss     = amp_loss_weight > 0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n[TMNet-ImgSup] TMNet_Tr_priormulti_image"
        f"  | img_size=({img_h},{img_w})  | prior={prior_type}"
        f"  | img_loss={img_loss}  | kl_beta={kl_beta}"
        f"  | params={n_params:,}"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[TMNet-ImgSup] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(model, cfg.checkpoint, device)

    # ---- directories & logging ---------------------------------------------
    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(
            cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_imgsup",
        )
        run_dir = os.path.join(
            cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_imgsup",
        )
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    # ---- data loaders — raw volumes, no DVF cache --------------------------
    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    train_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp,
    )
    valid_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[1], nb_pred=cfg.tp, valid=True,
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

    # ---- optimiser & scheduler ---------------------------------------------
    optimizer     = build_optimizer(cfg, model.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience = cfg.early_stopping_patience,
        verbose  = True,
        delta    = cfg.early_stopping_delta,
    )

    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")
    tf_prob       = float(getattr(cfg, "tmnet_teacher_forcing_prob", 1.0))

    print(f"[TMNet-ImgSup] Starting training from epoch {restart_epoch}...")
    print(f"[TMNet-ImgSup] Teacher-forcing prob: {tf_prob}")
    print(f"\nStage 2b — TMNet image-supervision training  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet-ImgSup] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        model.train()

        # KL β annealing: linearly ramp 0 → kl_beta over kl_warmup_ep epochs
        beta = kl_beta * min(1.0, (epoch + 1) / max(kl_warmup_ep, 1))
        writer.add_scalar("tmnet_imgsup_train/kl_beta", beta, epoch)

        ep_recon = ep_kl = ep_phi = ep_amp = ep_steps = 0

        for batch in tqdm(train_loader):
            Iref, Ipast, Ifuture_2ch, Igt = _build_priormulti_image_inputs(batch, device, cfg)
            optimizer.zero_grad()

            use_tf = torch.rand(1).item() < tf_prob
            DVF_seq, I_pred, kl_loss, phi_pred, amp_pred, _ , _= model(
                Iref, Ipast, Ifuture_2ch if use_tf else None
            )

            # Bug 2 fix: unsqueeze(2) to pass 5D tensors consistent with the
            # project-wide ncc_loss convention (same as train_CLDM.py volumes).
            recon_loss = torch.tensor(0.0, device=device)
            for t in range(cfg.tp):
                pred_t = I_pred[:, :, t, :, :]    # (B, 1, H, W)
                gt_t   = Igt[:, :, t, :, :]       # (B, 1, H, W)
                if img_loss == "ncc":
                    recon_loss = recon_loss + (
                        1 + ncc_loss(pred_t.unsqueeze(2), gt_t.unsqueeze(2), device)
                    )
                else:
                    recon_loss = recon_loss + torch.nn.functional.mse_loss(pred_t, gt_t)
            recon_loss = recon_loss / cfg.tp

            loss = recon_loss + (beta * kl_loss if kl_loss is not None else 0.0)

            # ── Auxiliary losses (optional) ──────────────────────────────────
            phi_loss = amp_loss = torch.tensor(0.0, device=device)
            if phi_pred is not None or amp_pred is not None:
                # print(f"[TMNet-ImgSup] Auxiliary losses enabled: {'phi' if phi_pred is not None else ''} {'amp' if amp_pred is not None else ''}")
                phi_gt, amp_gt = _aux_targets_from_batch(batch, device)
                if phi_pred is not None:
                    phi_loss = torch.nn.functional.mse_loss(phi_pred, phi_gt)
                    loss = loss + phi_loss_weight * phi_loss
                if amp_pred is not None:
                    amp_loss = torch.nn.functional.mse_loss(amp_pred, amp_gt)
                    loss = loss + amp_loss_weight * amp_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            ep_recon  += recon_loss.item()
            ep_kl     += kl_loss.item() if kl_loss is not None else 0.0
            ep_phi    += phi_loss.item()
            ep_amp    += amp_loss.item()
            ep_steps  += 1
            writer.add_scalar("tmnet_imgsup_train/loss",       loss.item(),       global_step)
            writer.add_scalar("tmnet_imgsup_train/recon_loss", recon_loss.item(), global_step)
            writer.add_scalar("tmnet_imgsup_train/kl_loss",    kl_loss.item() if kl_loss is not None else 0.0, global_step)
            if phi_pred is not None:
                writer.add_scalar("tmnet_imgsup_train/phi_loss", phi_loss.item(), global_step)
            if amp_pred is not None:
                writer.add_scalar("tmnet_imgsup_train/amp_loss", amp_loss.item(), global_step)
            global_step += 1

        ep_recon_avg = ep_recon / max(ep_steps, 1)
        ep_kl_avg    = ep_kl    / max(ep_steps, 1)
        ep_phi_avg   = ep_phi   / max(ep_steps, 1)
        ep_amp_avg   = ep_amp   / max(ep_steps, 1)
        writer.add_scalar("tmnet_imgsup_train/epoch_recon_loss", ep_recon_avg, epoch)
        writer.add_scalar("tmnet_imgsup_train/epoch_kl_loss",    ep_kl_avg,    epoch)
        if phi_loss_weight > 0:
            writer.add_scalar("tmnet_imgsup_train/epoch_phi_loss", ep_phi_avg, epoch)
        if amp_loss_weight > 0:
            writer.add_scalar("tmnet_imgsup_train/epoch_amp_loss", ep_amp_avg, epoch)

        # ---- validation — Bug 1 fix: use prior (Ifuture=None) ---------------
        # Passing Ifuture at val time gives posterior performance (optimistic).
        # Validation must mirror test time: only past frames available.
        model.eval()
        val_recon = val_phi = val_amp = val_n = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                Iref, Ipast, _, Igt = _build_priormulti_image_inputs(batch, device, cfg)
                # Ifuture=None → prior path, matches test-time behaviour
                DVF_seq, I_pred, _, phi_pred, amp_pred, _, _ = model(Iref, Ipast)

                r = torch.tensor(0.0, device=device)
                for t in range(cfg.tp):
                    pred_t = I_pred[:, :, t, :, :]
                    gt_t   = Igt[:, :, t, :, :]
                    if img_loss == "ncc":
                        r = r + (
                            1 + ncc_loss(pred_t.unsqueeze(2), gt_t.unsqueeze(2), device)
                        )
                    else:
                        r = r + torch.nn.functional.mse_loss(pred_t, gt_t)
                val_recon += (r / cfg.tp).item()

                if phi_pred is not None or amp_pred is not None:
                    phi_gt, amp_gt = _aux_targets_from_batch(batch, device)
                    if phi_pred is not None:
                        val_phi += torch.nn.functional.mse_loss(phi_pred, phi_gt).item()
                    if amp_pred is not None:
                        val_amp += torch.nn.functional.mse_loss(amp_pred, amp_gt).item()
                val_n += 1

        val_recon_avg = val_recon / max(val_n, 1)
        val_phi_avg   = val_phi   / max(val_n, 1)
        val_amp_avg   = val_amp   / max(val_n, 1)
        writer.add_scalar("tmnet_imgsup_val/recon_loss", val_recon_avg, epoch)
        if phi_loss_weight > 0:
            writer.add_scalar("tmnet_imgsup_val/phi_loss", val_phi_avg, epoch)
        if amp_loss_weight > 0:
            writer.add_scalar("tmnet_imgsup_val/amp_loss", val_amp_avg, epoch)

        aux_str = ""
        if phi_loss_weight > 0:
            aux_str += f"  phi_tr={ep_phi_avg:.4f}/val={val_phi_avg:.4f}(x{phi_loss_weight})"
        if amp_loss_weight > 0:
            aux_str += f"  amp_tr={ep_amp_avg:.4f}/val={val_amp_avg:.4f}(x{amp_loss_weight})"
        print(
            f"[TMNet-ImgSup] train_recon={ep_recon_avg:.6f}  train_kl={ep_kl_avg:.6f}"
            f"{aux_str}  val_recon={val_recon_avg:.6f} (prior)  β={beta:.3f}"
        )

        if val_recon_avg < best_val_loss:
            print(f"[TMNet-ImgSup] Improved recon {best_val_loss:.6f} → {val_recon_avg:.6f} — saving.")
            best_val_loss = val_recon_avg
            custom_save(model, os.path.join(log_dir, "model_best_tmnet.pth"))
        else:
            print(f"[TMNet-ImgSup] No improvement from {best_val_loss:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_recon_avg)

        early_stopper(val_recon_avg)
        if early_stopper.early_stop:
            print("[TMNet-ImgSup] Early stopping triggered.")
            break

        print(f"[TMNet-ImgSup] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nStage 2b done.  Best checkpoint: {log_dir}/model_best_tmnet.pth")
    writer.close()

    # ---- Bug 3 fix: compute and save embedding statistics ------------------
    # The downstream diffusion model needs tmnet_stats.pt for normalisation.
    # Use model.encode() (prior path) over the full training set.
    print("[TMNet-ImgSup] Computing embedding statistics over full training set...")
    custom_load(model, os.path.join(log_dir, "model_best_tmnet.pth"), device)
    model.eval()

    emb_sum = emb_sq_sum = emb_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            _, Ipast, _, _ = _build_priormulti_image_inputs(batch, device, cfg)
            emb_list = model.encode(Ipast)
            for emb in emb_list:
                emb_sum    += emb.sum().item()
                emb_sq_sum += emb.pow(2).sum().item()
                emb_count  += emb.numel()

    emb_mean = emb_sum / emb_count
    emb_std  = max((emb_sq_sum / emb_count - emb_mean ** 2) ** 0.5, 1e-6)

    stats_path = os.path.join(log_dir, "tmnet_stats.pt")
    torch.save({"mean": emb_mean, "std": emb_std}, stats_path)
    print(
        f"[TMNet-ImgSup] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  "
        f"saved to {stats_path}"
    )


# =============================================================================
# Stage 2b (variant) — TMNet contrastive pretraining with respiratory phase
# =============================================================================

def _phi_from_batch(batch, device: torch.device) -> torch.Tensor:
    """
    Extract phi ∈ [0, 0.5] for the last input frame of each batch element.

    The dataset returns files_output paths; the last input frame has t_idx one
    step before the first output frame (files_output[0]).
    """
    vol_files = batch[-1]     # list[nb_pred] of list[B] strings
    paths     = vol_files[0]  # first output step, list[B]
    phis = []
    for path in paths:
        parts      = path.split("/")
        patient_id = parts[-2]
        t_idx      = int(parts[-1][2:-7]) - 1   # last input = output[0] - 1
        data_dir   = "/".join(parts[:-2])
        phis.append(_get_phi(patient_id, t_idx, data_dir))
    return torch.tensor(phis, dtype=torch.float32, device=device)


def train_tmnet_dvfsup(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    vm:       torch.nn.Module,
    device:   torch.device,
) -> None:
    """
    Pretrain TMNet via navigator-plane DVF slice prediction for one fold.

    DVFs are computed on-the-fly from a frozen VoxelMorph model — no cache
    required.  Uses NAVIGATOR_4D_Dataset_multitime directly.

    The encoder sees only past frames (no teacher forcing), exactly as at
    test time.  Loss: MSE between predicted and VM-computed DVF slice at the
    navigator plane.

    Checkpoint saved as model_best_tmnet.pth (same key as other modes).
    """
    from mopred.data.data_loaders.navigator_4d import Y_NAV

    condi_type = getattr(cfg, "tmnet_condi_type", "2")
    dvf_h, dvf_w = tuple(cfg.tmnet_img_size)   # matches navigator slice dims

    # ---- build model -------------------------------------------------------
    condi = TMNetEncoder(
        num_inputs       = cfg.tmnet_num_frames,
        horizon          = cfg.tmnet_horizon,
        in_channels      = 2,
        out_channels     = cfg.tmnet_condi_channels,
        n_heads          = cfg.tmnet_Tr_n_heads,
        enc_layers       = cfg.tmnet_Tr_enc_layers,
        normalize_before = cfg.tmnet_Tr_norm_before,
        output_dim       = cfg.tmnet_pre_latent_dim,
        condi_type       = condi_type,
        device           = device,
    ).to(device)

    model = DVFSupTMNet(tm_net=condi, dvf_h=dvf_h, dvf_w=dvf_w).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n[TMNet-DVFSup] DVFSupTMNet"
        f"  | dvf_slice=({dvf_h},{dvf_w})  | condi_type={condi_type}"
        f"  | hidden_dim={condi.hidden_dim}  | params={n_params:,}"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[TMNet-DVFSup] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(condi, cfg.checkpoint, device)

    # ---- directories & logging ---------------------------------------------
    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(
            cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_dvfsup",
        )
        run_dir = os.path.join(
            cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_dvfsup",
        )
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    # ---- data loaders — raw volumes, no DVF cache --------------------------
    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    train_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp,
    )
    valid_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[1], nb_pred=cfg.tp, valid=True,
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        worker_init_fn=_seed_worker, generator=_g,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        worker_init_fn=_seed_worker,
    )

    # ---- optimiser & scheduler ---------------------------------------------
    optimizer     = build_optimizer(cfg, model.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience = cfg.early_stopping_patience,
        verbose  = True,
        delta    = cfg.early_stopping_delta,
    )

    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    def _dvf_slice(dvf: torch.Tensor) -> torch.Tensor:
        """Extract the navigator-plane DVF slice: (B,3,D,H,W) → (B,3,dvf_h,dvf_w)."""
        if condi_type == "2":          # coronal at H=Y_NAV
            return dvf[:, :, :, Y_NAV, :]
        else:                          # sagittal at D=16
            return dvf[:, :, 16, :, :]

    aug_cfg = _build_aug_cfg(cfg)
    if aug_cfg.any_enabled():
        print(
            f"[TMNet-DVFSup] Temporal augmentation enabled: "
            f"frame_drop={aug_cfg.frame_drop.enabled}, "
            f"spatial_mask={aug_cfg.spatial_mask.enabled}, "
            f"tube_mask={aug_cfg.tube_mask.enabled}, "
            f"variable_frames={aug_cfg.variable_frames.enabled}"
        )

    print(f"[TMNet-DVFSup] Starting training from epoch {restart_epoch}...")
    print(f"\nStage 2b — TMNet DVF-slice supervision  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet-DVFSup] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        model.train()
        ep_loss = ep_steps = 0

        for batch in tqdm(train_loader):
            frames = _unpack_condi_inputs(batch, device, cfg)   # (B,2,T,H,W)
            frames = apply_pixel_space_augmentations(frames, aug_cfg, training=True)
            indices = torch.nonzero((frames != 0).any(dim=(2, 3, 4)), as_tuple=False)
            print(indices)

            ref_vol    = batch[0].unsqueeze(1).to(device)       # (B,1,D,H,W)
            target_vol = batch[2][0].unsqueeze(1).to(device)    # (B,1,D,H,W)

            with torch.no_grad():
                dvf_gt = _dvf_slice(vm(ref_vol, target_vol))    # (B,3,dvf_h,dvf_w)

            optimizer.zero_grad()
            out  = model(frames, dvf_gt=dvf_gt)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            ep_loss  += loss.item()
            ep_steps += 1
            writer.add_scalar("tmnet_dvfsup_train/loss", loss.item(), global_step)
            global_step += 1

        ep_avg = ep_loss / max(ep_steps, 1)
        writer.add_scalar("tmnet_dvfsup_train/epoch_loss", ep_avg, epoch)

        # ---- validation ----------------------------------------------------
        model.eval()
        val_loss = val_n = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                frames = _unpack_condi_inputs(batch, device, cfg)

                ref_vol    = batch[0].unsqueeze(1).to(device)
                target_vol = batch[2][0].unsqueeze(1).to(device)
                dvf_gt     = _dvf_slice(vm(ref_vol, target_vol))

                out       = model(frames, dvf_gt=dvf_gt)
                val_loss += out["loss"].item()
                val_n    += 1

        val_avg = val_loss / max(val_n, 1)
        writer.add_scalar("tmnet_dvfsup_val/loss", val_avg, epoch)
        print(f"[TMNet-DVFSup] train={ep_avg:.6f}  val={val_avg:.6f}")

        if val_avg < best_val_loss:
            print(f"[TMNet-DVFSup] Improved {best_val_loss:.6f} → {val_avg:.6f} — saving.")
            best_val_loss = val_avg
            custom_save(condi, os.path.join(log_dir, "model_best_tmnet.pth"))
        else:
            print(f"[TMNet-DVFSup] No improvement from {best_val_loss:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            print("[TMNet-DVFSup] Early stopping triggered.")
            break

        print(f"[TMNet-DVFSup] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nDone.  Best checkpoint: {log_dir}/model_best_tmnet.pth")
    writer.close()


def train_tmnet_contrastive(
    cfg,
    folds:    tuple,
    fold_idx: str,
    dir_name: str,
    device:   torch.device,
) -> None:
    """
    Pretrain TMNetEncoder via supervised contrastive learning (SupCon) using
    respiratory phase phi as the label.

    Positive pairs:  sequences whose last input frame has |Δphi| < tau_pos.
    Negative pairs:  sequences with |Δphi| >= tau_neg.
    Neutral zone:    excluded from the SupCon denominator.

    No DVF cache needed — uses raw NAVIGATOR_4D_Dataset_multitime.
    Checkpoint format identical to other pretraining variants so that downstream
    train_CLDM can load it without changes.
    """

    # ---- build model -------------------------------------------------------
    condi = TMNetEncoder(
        num_inputs       = cfg.tmnet_num_frames,
        horizon          = cfg.tmnet_horizon,
        in_channels      = 2,
        out_channels     = cfg.tmnet_condi_channels,
        n_heads          = cfg.tmnet_Tr_n_heads,
        enc_layers       = cfg.tmnet_Tr_enc_layers,
        normalize_before = cfg.tmnet_Tr_norm_before,
        output_dim       = cfg.tmnet_pre_latent_dim,
        condi_type       = getattr(cfg, "tmnet_condi_type", "2"),
        device           = device,
    ).to(device)

    proj_dim    = int(getattr(cfg, "tmnet_contrast_proj_dim",    128))
    temperature = float(getattr(cfg, "tmnet_contrast_temperature", 0.07))
    tau_pos     = float(getattr(cfg, "tmnet_contrast_tau_pos",    0.05))
    tau_neg     = float(getattr(cfg, "tmnet_contrast_tau_neg",    0.15))

    model = PhiContrastiveTMNet(
        tm_net   = condi,
        proj_dim    = proj_dim,
        temperature = temperature,
        tau_pos     = tau_pos,
        tau_neg     = tau_neg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n[TMNet-Contrast] TMNetEncoder + PhiContrastiveTMNet"
        f"  | proj_dim={proj_dim}  | T={temperature}"
        f"  | tau_pos={tau_pos}  | tau_neg={tau_neg}"
        f"  | params={n_params:,}"
        f"  (device={next(condi.parameters()).device})"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[TMNet-Contrast] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(condi, cfg.checkpoint, device)

    # ---- directories & logging ---------------------------------------------
    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep, os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(
            cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "tmnet_contrast",
        )
        run_dir = os.path.join(
            cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "tmnet_contrast",
        )
    for d in (log_dir, run_dir):
        cond_mkdir(d)

    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)
    save_patients(log_dir, folds)

    # ---- data loaders — raw volumes, no DVF cache --------------------------
    import random as _random

    def _seed_worker(_worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        _random.seed(seed)

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)

    train_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp,
    )
    valid_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[1], nb_pred=cfg.tp, valid=True,
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        worker_init_fn=_seed_worker, generator=_g,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        worker_init_fn=_seed_worker,
    )

    # ---- optimiser & scheduler ---------------------------------------------
    optimizer     = build_optimizer(cfg, model.parameters())
    scheduler     = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience = cfg.early_stopping_patience,
        verbose  = True,
        delta    = cfg.early_stopping_delta,
    )

    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    print(f"[TMNet-Contrast] Starting training from epoch {restart_epoch}...")
    print(f"\nStage 2b — TMNet contrastive pretraining  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[TMNet-Contrast] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        condi.train()
        model.train()
        ep_loss = ep_n_anchors = ep_frac_pos = ep_steps = 0

        for batch in tqdm(train_loader):
            Iseq = _build_mopTR_inputs(batch, device, cfg)[0]  # (B, 2, T, H, W)
            phi  = _phi_from_batch(batch, device)              # (B,)
            optimizer.zero_grad()

            out  = model(Iseq, phi)
            loss = out["loss"]

            if out["n_anchors"] == 0:
                # No valid positive pairs in this batch — skip update
                ep_steps += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            ep_loss      += loss.item()
            ep_n_anchors += out["n_anchors"]
            ep_frac_pos  += out["frac_pos"]
            ep_steps     += 1
            writer.add_scalar("tmnet_contrast_train/loss",       loss.item(),         global_step)
            writer.add_scalar("tmnet_contrast_train/n_anchors",  out["n_anchors"],    global_step)
            writer.add_scalar("tmnet_contrast_train/frac_pos",   out["frac_pos"],     global_step)
            global_step += 1

        ep_avg     = ep_loss      / max(ep_steps, 1)
        ep_anc_avg = ep_n_anchors / max(ep_steps, 1)
        ep_pos_avg = ep_frac_pos  / max(ep_steps, 1)
        writer.add_scalar("tmnet_contrast_train/epoch_loss",      ep_avg,     epoch)
        writer.add_scalar("tmnet_contrast_train/epoch_n_anchors", ep_anc_avg, epoch)

        # ---- validation ----------------------------------------------------
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
        print(
            f"[TMNet-Contrast] train={ep_avg:.6f}  val={val_avg:.6f}"
            f"  anchors/step={ep_anc_avg:.1f}  frac_pos={ep_pos_avg:.3f}"
        )

        if val_avg < best_val_loss:
            print(f"[TMNet-Contrast] Improved {best_val_loss:.6f} → {val_avg:.6f} — saving.")
            best_val_loss = val_avg
            custom_save(condi, os.path.join(log_dir, "model_best_tmnet.pth"))
        else:
            print(f"[TMNet-Contrast] No improvement from {best_val_loss:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_avg)

        early_stopper(val_avg)
        if early_stopper.early_stop:
            print("[TMNet-Contrast] Early stopping triggered.")
            break

        print(f"[TMNet-Contrast] Epoch duration: {(time.time() - t0) / 60:.2f} min")

    print(f"\nStage 2b done.  Best checkpoint: {log_dir}/model_best_tmnet.pth")
    writer.close()

    # ---- compute and save embedding statistics ----------------------------
    print("[TMNet-Contrast] Computing embedding statistics over full training set...")
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
    print(
        f"[TMNet-Contrast] Embedding stats — mean={emb_mean:.4f}  std={emb_std:.4f}  "
        f"saved to {stats_path}"
    )


# =============================================================================
# CLI
# =============================================================================

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TMNet MAE pretrainer")
    p.add_argument(
        "--config", required=True,
        help="Path to a YAML config file (see configs/CondNets/TMNet.yaml).",
    )
    p.add_argument(
        "--train_test", required=True,
        choices=[
            "train_tmnet", "train_tmnet_predictive",
            "train_tmnet_mopTR_style", "train_tmnet_priormulti_image",
            "train_tmnet_priormulti_dvf",
            "train_tmnet_contrastive", "train_tmnet_dvfsup",
            "compute_stats", "test_tmnet", "test_tmnet_dvfsup2",
        ],
        help=(
            "'train_tmnet'                  — pretrain TMNet via MAE; "
            "'train_tmnet_predictive'       — pretrain TMNet via next-frame prediction; "
            "'train_tmnet_mopTR_style'      — pretrain TMNet via MopTR-style image prediction; "
            "'train_tmnet_priormulti_image' — train TMNet_Tr_priormulti with DVF+image supervision; "
            "'train_tmnet_contrastive'      — pretrain TMNet via SupCon loss with respiratory phase phi; "
            "'train_tmnet_dvfsup'           — pretrain TMNet via navigator-plane DVF slice prediction; "
            "'compute_stats'                   — compute embedding statistics from a checkpoint; "
            "'test_tmnet'                   — evaluate reconstruction quality on the test set."
        ),
    )
    p.add_argument(
        "--fold_nb_training", type=int, default=3,
        help="How many folds to train on (0 = all).",
    )
    p.add_argument(
        "--fold_start", type=int, default=0,
        help="First fold index to train (skip already-completed folds, e.g. --fold_start 2).",
    )
    p.add_argument(
        "--dir_name", type=str, default=None,
        help=(
            "Resume an existing run by specifying its directory relative to logging_dir/logs/, "
            "e.g. '07_02/11.25._TMNet_run'. If omitted a new timestamped directory is created."
        ),
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "[train_tmnet only] Path to a .pth checkpoint to resume from, e.g. "
            "'<logging_dir>/logs/04_10/14.30._run/fold_0/tmnet/model_best_tmnet.pth'."
        ),
    )
    p.add_argument(
        "--tmnet_dir_name", type=str, default=None,
        help="[compute_stats / test_tmnet] Run directory name to load from.",
    )
    p.add_argument(
        "--override", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override any config key, e.g. --override lr_vae=5e-5 batch_size=4",
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
    if args.tmnet_dir_name is not None:
        cfg.tmnet_dir_name = args.tmnet_dir_name
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

    # Frozen registration model — needed only to build the DVF cache (not for mopTR-style).
    _needs_vm = args.train_test not in (
        "train_tmnet_mopTR_style", "train_tmnet_priormulti_image",
        "train_tmnet_contrastive", "test_tmnet", "compute_stats",
    ) or args.train_test in ("train_tmnet_priormulti_dvf", "test_tmnet_dvfsup2")
    if _needs_vm:
        VOL_SIZE = (32, 64, 64)
        vm = build_reg_model(cfg, VOL_SIZE, device)
    else:
        vm = None

    train_folds, valid_folds, test_folds = make_train_val_test_folds()
    fold_nb = cfg.fold_nb_training or len(train_folds)
    print(f"Training on {fold_nb} fold(s).")

    if args.dir_name is not None:
        dir_name = args.dir_name
    else:
        dir_name = os.path.join(
            datetime.datetime.now().strftime("%m_%d"),
            datetime.datetime.now().strftime("%H.%M._") + cfg.name,
        )

    fold_start = args.fold_start
    if fold_start > 0:
        print(f"Skipping folds 0–{fold_start - 1}, starting from fold {fold_start}.")

    # ------------------------------------------------------------------
    if args.train_test == "train_tmnet_priormulti_dvf":
        for fold_idx in range(fold_start, fold_nb):
            # if fold_idx == 0 :
            #     continue
            # else : 
            print(f"\n=== Training fold {fold_idx} ===")
            train_tmnet_priormulti_dvf(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
                vm       = vm,
                device   = device,
            )

    elif args.train_test == "train_tmnet_dvfsup":
        for fold_idx in range(fold_start, fold_nb):
            print(f"\n=== Training fold {fold_idx} ===")
            train_tmnet_dvfsup(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
                vm       = vm,
                device   = device,
            )

    elif args.train_test == "train_tmnet_contrastive":
        for fold_idx in range(fold_start, fold_nb):
            print(f"\n=== Training fold {fold_idx} ===")
            train_tmnet_contrastive(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
                device   = device,
            )

    elif args.train_test == "train_tmnet_priormulti_image":
        for fold_idx in range(fold_start, fold_nb):
            print(f"\n=== Training fold {fold_idx} ===")
            train_tmnet_priormulti_image(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
                device   = device,
            )

    elif args.train_test == "train_tmnet_mopTR_style":
        for fold_idx in range(fold_start, fold_nb):
            print(f"\n=== Training fold {fold_idx} ===")
            train_tmnet_mopTR_style(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
                device   = device,
            )

    elif args.train_test == "train_tmnet_predictive":
        for fold_idx in range(fold_start, fold_nb):
            print(f"\n=== Training fold {fold_idx} ===")
            train_tmnet_predictive(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
                vm       = vm,
                device   = device,
            )

    elif args.train_test == "train_tmnet":
        for fold_idx in range(fold_start, fold_nb):
            print(f"\n=== Training fold {fold_idx} ===")
            train_tmnet(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = dir_name,
                vm       = vm,
                device   = device,
            )

    elif args.train_test == "compute_stats":
        for fold_idx in range(fold_start, fold_nb):
            compute_tmnet_stats(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = cfg.tmnet_dir_name,
                device   = device,
            )

    elif args.train_test == "test_tmnet_dvfsup2":
        for fold_idx in range(fold_start, fold_nb):
            test_tmnet_dvfsup2(
                cfg      = cfg,
                fold     = test_folds[fold_idx],
                fold_idx = str(fold_idx),
                dir_name = cfg.tmnet_dir_name,
                vm       = vm,
                device   = device,
            )

    elif args.train_test == "test_tmnet":
        for fold_idx in range(fold_start, fold_nb):
            test_tmnet(
                cfg      = cfg,
                fold     = test_folds[fold_idx],
                fold_idx = str(fold_idx),
                dir_name = cfg.tmnet_dir_name,
                device   = device,
            )
