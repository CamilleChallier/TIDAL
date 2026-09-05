"""
Stage 1 training pipeline for the DVFVAE spatial β-VAE (liver dataset).
Supports train_vae, compute_stats (latent statistics consumed by train_CLDM), and test_vae modes.

Usage
-----
python -m scripts.train_VAE --config configs/VAE/dvfvae_mm.yaml --train_test train_vae
python -m scripts.train_VAE --config configs/VAE/dvfvae_mm.yaml --train_test compute_stats
python -m scripts.train_VAE --config configs/VAE/dvfvae_mm.yaml --train_test test_vae
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
from skimage.metrics import structural_similarity as ss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, dynamic_ncols=True)

from mopred.models.VAE import DVFVAE
from mopred.models import SpatialTransformer

from mopred.data.splits       import make_folds_3fold as make_train_val_test_folds
from mopred.data.data_loaders import NAVIGATOR_4D_Dataset_multitime
from mopred.data.loading      import save_params_txt
from mopred.utils.early_stopping  import EarlyStopping
from mopred.utils.assemble_volumes import assemble_volumes
from mopred.utils.io              import cond_mkdir, custom_load, custom_save, save_tensor_as_nifti
from mopred.utils.losses          import build_criterion, ncc_loss
from mopred.utils.dvf_metrics     import geo_error
from mopred.data.data_loaders.navigator_4d import get_phi as _get_phi, get_cycle_phase as _get_cycle_phase
from mopred.utils.training        import load_config, _apply_overrides, vae_checkpoint_path, build_scheduler, build_optimizer, build_vae, save_patients, summarize_test_metrics, build_reg_model

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# =============================================================================
# Phase-balanced sampler
# =============================================================================

# =============================================================================
# Phase helpers
# =============================================================================


# =============================================================================
# Stage 1 — VAE training  (model-agnostic)
# =============================================================================
 
def train_vae(
    cfg: argparse.Namespace,
    folds: tuple,
    fold_idx: str,
    dir_name: str,
    vm: torch.nn.Module,
    device: torch.device,
) -> None:
    """
    Train any BaseVAE subclass for one fold.
 
    The loop is identical for MIA_VAE, VQVAE, and CVAE:
        out = model.train_step(batch, device)
        out["loss"].backward()
        optimizer.step()
 
    All horizon / tp / conditioning logic lives inside the model.
    """
 
    # ---- build model -------------------------------------------------------
    # Resolve per-fold context encoder path if a {fold_idx} placeholder is present.
    # Restore the template afterwards so subsequent folds substitute correctly.
    _ctx_template = getattr(cfg, "vae_context_encoder_path", None)
    if _ctx_template:
        cfg.vae_context_encoder_path = _ctx_template.replace("{fold_idx}", str(fold_idx))
    vae = build_vae(cfg, device)
    if _ctx_template:
        cfg.vae_context_encoder_path = _ctx_template
    print(
        f"\n[VAE] Model: {type(vae).__name__}  "
        f"| horizon={vae.horizon}  "
        f"(tp read from dvf_list length at runtime)"
        f"(device={next(vae.parameters()).device})"
    )

    if getattr(cfg, "checkpoint", None):
        print(f"[VAE] Resuming from checkpoint: {cfg.checkpoint}")
        custom_load(vae, cfg.checkpoint, device)
 
    # ---- directories & logging ---------------------------------------------
    # When resuming from a checkpoint, reuse the original run's directories so
    # the best model is saved in-place and TensorBoard curves are continuous.
    if getattr(cfg, "checkpoint", None):
        log_dir = os.path.dirname(cfg.checkpoint)
        run_dir = log_dir.replace(os.sep + "logs" + os.sep,
                                  os.sep + "runs" + os.sep, 1)
    else:
        log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "vae")
        run_dir = os.path.join(cfg.logging_dir, "runs", dir_name, f"fold_{fold_idx}", "vae")
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
    optimizer = build_optimizer(cfg, vae.parameters())
    scheduler = build_scheduler(cfg, optimizer, len(train_loader))
    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        verbose=True,
        delta=cfg.early_stopping_delta,
    )
 
    restart_epoch = getattr(cfg, "restart_epoch", 0)
    print(f"[VAE] Starting training from epoch {restart_epoch}...")
    global_step   = restart_epoch * len(train_loader)
    best_val_loss = float("inf")

    vae_debug = getattr(cfg, "debug", False)

    # ---- epoch loop --------------------------------------------------------
    print(f"\nStage 1 — {cfg.vae_model.upper()} training  (fold {fold_idx})")

    for epoch in range(restart_epoch, cfg.vae_epochs):
        print(f"\n[VAE] Epoch {epoch}/{cfg.vae_epochs - 1}")
        t0 = time.time()
        vae.train()

        # KL annealing: beta = min(beta_max, beta_max * epoch / beta_steps)
        beta_max   = getattr(cfg, "beta_max",   getattr(vae, "beta_target", 1e-3))
        beta_steps = getattr(cfg, "beta_steps", cfg.vae_epochs)
        beta = min(beta_max, beta_max * epoch / max(beta_steps, 1))
        vae.set_kl_weight(beta)
        writer.add_scalar("vae_train/kl_weight", beta, epoch)
 
        ep_loss = ep_recon = ep_aux = ep_steps = 0
 
        for batch in tqdm(train_loader):
            ref_vol, input_vols, current_vols, vol_files = batch
            ref_vol_d = ref_vol.unsqueeze(1).to(device)
            with torch.no_grad():
                dvf_list = [vm(ref_vol_d, cur.unsqueeze(1).to(device)) for cur in current_vols]
            batch = (ref_vol, input_vols, current_vols, dvf_list, vol_files)

            optimizer.zero_grad()

            # ---------------------------------------------------------------
            # Single call — model handles horizon internally
            # ---------------------------------------------------------------
            out = vae.train_step(batch, device)

            out["loss"].backward()
            # torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
 
            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()
 
            # Logging
            ep_loss  += out["loss"].item()
            ep_recon += out["recon_loss"].item()
            ep_aux   += out["aux_loss"].item()
            ep_steps += 1
 
            writer.add_scalar("vae_train/total_loss", out["loss"].item(),       global_step)
            writer.add_scalar("vae_train/recon_loss", out["recon_loss"].item(), global_step)
            writer.add_scalar("vae_train/aux_loss",   out["aux_loss"].item(),   global_step)
 
            for name, val in (out.get("metrics") or {}).items():
                writer.add_scalar(f"vae_train/{name}", val, global_step)
 
            global_step += 1
 
        writer.add_scalar("vae_train/epoch_loss",  ep_loss  / ep_steps, epoch)
        writer.add_scalar("vae_train/epoch_recon", ep_recon / ep_steps, epoch)
        writer.add_scalar("vae_train/epoch_aux",   ep_aux   / ep_steps, epoch)
 
        # ---- validation ----------------------------------------------------
        vae.eval()
        val_losses = val_recons = val_auxes = val_n = 0

        _dbg_phi, _dbg_z0_norm, _dbg_vae_mse, _dbg_dvf_mag = [], [], [], []

        for batch in tqdm(valid_loader):
            ref_vol, input_vols, current_vols, vol_files = batch
            ref_vol_d = ref_vol.unsqueeze(1).to(device)
            with torch.no_grad():
                dvf_list = [vm(ref_vol_d, cur.unsqueeze(1).to(device)) for cur in current_vols]

            if vae_debug:
                # ── VAE reconstruction diagnostic ────────────────────────────
                # Check whether vae_mse is flat across phi:
                #   flat  → VAE is not the culprit for phase-dependent errors
                #   grows → VAE struggles to reconstruct high-motion frames
                with torch.no_grad():
                    for tp in range(len(dvf_list)):
                        dvf_gt_tp = dvf_list[tp]
                        z = vae.encode(dvf_gt_tp)
                        if isinstance(z, tuple):
                            z = z[0]   # use mu, not logvar
                        dvf_recon = vae.forward(dvf_gt_tp)["recon"]
                        vae_mse   = F.mse_loss(dvf_recon, dvf_gt_tp).item()
                        dvf_mag   = dvf_gt_tp.norm(dim=1).mean().item()
                        z0_norm   = z.norm().item()
                        try:
                            tp_files = vol_files[tp] if isinstance(vol_files[tp], (list, tuple)) else [vol_files[tp]]
                            path     = tp_files[0]
                            pid      = path.split("/")[-2]
                            t_idx    = int(path.split("/")[-1][2:-7])
                            phi_val  = _get_phi(pid, t_idx, cfg.data_dir)
                        except Exception:
                            phi_val = float("nan")
                        print(
                            f"[DEBUG VAE] tp={tp}  phi={phi_val:.3f}"
                            f"  dvf_mag={dvf_mag:.4f}"
                            f"  ||z0||={z0_norm:.3f}"
                            f"  z0_std={z.std().item():.4f}"
                            f"  vae_mse={vae_mse:.5f}"
                        )
                        _dbg_phi.append(phi_val)
                        _dbg_z0_norm.append(z0_norm)
                        _dbg_vae_mse.append(vae_mse)
                        _dbg_dvf_mag.append(dvf_mag)

            batch = (ref_vol, input_vols, current_vols, dvf_list, vol_files)
            out = vae.val_step(batch, device)
            val_losses += out["loss"].item()
            val_recons += out["recon_loss"].item()
            val_auxes  += out["aux_loss"].item()
            val_n      += 1

        val_total = val_losses / max(val_n, 1)
        val_recon = val_recons / max(val_n, 1)
        val_aux   = val_auxes  / max(val_n, 1)

        if vae_debug and _dbg_phi:
            n_bins  = 10
            bin_w   = 0.5 / n_bins
            phi_arr = np.array(_dbg_phi)
            z0_arr  = np.array(_dbg_z0_norm)
            vae_arr = np.array(_dbg_vae_mse)
            dvf_arr = np.array(_dbg_dvf_mag)
            print(f"\n[DEBUG VAE] Epoch {epoch} — Phase-binned summary")
            print(f"  {'phi_bin':>14}  {'n':>4}  {'dvf_mag':>8}  {'||z0||':>8}  {'vae_mse':>10}")
            print("  " + "-" * 54)
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
                "\n    vae_mse grows with phi   → VAE struggles to reconstruct high-motion frames"
                "\n    ||z0|| grows with phi    → high-motion latents may be OOD for the noise schedule"
            )

        writer.add_scalar("vae_val/total_loss", val_total, epoch)
        writer.add_scalar("vae_val/recon_loss", val_recon, epoch)
        writer.add_scalar("vae_val/aux_loss",   val_aux,   epoch)
        print(
            f"[VAE] val_loss={val_total:.6f}  "
            f"recon={val_recon:.6f}  aux={val_aux:.6f}"
        )
 
        # ---- checkpoint ----------------------------------------------------
        ckpt_metric = val_recon  # use recon loss; total loss rises during KL annealing
        if ckpt_metric < best_val_loss:
            print(f"[VAE] Improved val_recon {best_val_loss:.6f} → {ckpt_metric:.6f} — saving.")
            best_val_loss = ckpt_metric
            custom_save(vae, os.path.join(log_dir, "model_best_vae.pth"))
        else:
            print(f"[VAE] No recon improvement from {best_val_loss:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_recon)

        early_stopper(val_recon)
        if early_stopper.early_stop:
            print("[VAE] Early stopping triggered.")
            break
 
        print(f"[VAE] Epoch duration: {(time.time() - t0) / 60:.2f} min")
 
    print(f"\nStage 1 done.  Best checkpoint: {log_dir}/model_best_vae.pth")
    writer.close()

    # ---- compute and save latent statistics over the full training set -------
    # Reload the best checkpoint so stats reflect the final best model.
    print("[VAE] Computing latent statistics over full training set...")
    custom_load(vae, os.path.join(log_dir, "model_best_vae.pth"))
    vae.eval()

    lat_sum = lat_sq_sum = lat_count = 0

    with torch.no_grad():
        for batch in tqdm(train_loader):
            ref_vol, input_vols, current_vols, vol_files = batch
            ref_vol_d = ref_vol.unsqueeze(1).to(device)
            dvf_list = [vm(ref_vol_d, cur.unsqueeze(1).to(device)) for cur in current_vols]
            batch = (ref_vol, input_vols, current_vols, dvf_list, vol_files)
            _, _, _, dvf_list = vae._unpack_batch(batch, device)
            for dvf in dvf_list:
                z = vae.encode(dvf)
                # encode() may return (mu, logvar) — use mu for statistics
                if isinstance(z, tuple):
                    z = z[0]
                lat_sum    += z.sum().item()
                lat_sq_sum += z.pow(2).sum().item()
                lat_count  += z.numel()

    lat_mean = lat_sum / lat_count
    lat_std  = max((lat_sq_sum / lat_count - lat_mean ** 2) ** 0.5, 1e-6)

    stats_path = os.path.join(log_dir, "latent_stats.pt")
    torch.save({"mean": lat_mean, "std": lat_std}, stats_path)
    print(
        f"[VAE] Latent stats — mean={lat_mean:.4f}  std={lat_std:.4f}  "
        f"saved to {stats_path}"
    )

# =============================================================================
# Entry point
# =============================================================================

def compute_latent_stats(
    cfg: argparse.Namespace,
    folds: tuple,
    fold_idx: str,
    dir_name: str,
    vm: nn.Module,
    device: torch.device,
) -> None:
    """
    Load an existing VAE checkpoint and compute latent statistics over the
    full training set.  Saves latent_stats.pt next to the checkpoint so
    train_CLDM can load it without retraining.

    Usage
    -----
    python -m 4D_MoPred_liver.scripts.train_VAE --config configs/MIA_VAE.yaml \
        --train_test compute_stats --vae_dir_name <run_dir>
    """
    log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "vae")
    ckpt    = os.path.join(log_dir, "model_best_vae.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[compute_stats] Checkpoint not found: {ckpt}")

    vae = build_vae(cfg, device)
    custom_load(vae, ckpt, device)
    vae.eval()
    print(f"[compute_stats] Loaded VAE from {ckpt}")

    train_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp,
    )
    loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
    )

    max_batches = getattr(cfg, "latent_stats_batches", 50)
    N_BINS = 10
    lat_sum = lat_sq_sum = lat_count = 0
    bin_sum    = np.zeros(N_BINS)
    bin_sq_sum = np.zeros(N_BINS)
    bin_count  = np.zeros(N_BINS, dtype=np.int64)

    print(f"[compute_stats] Encoding up to {max_batches}/{len(loader)} batches...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ref_volume, _, current_volume_list, vol_files = batch
            ref_volume = ref_volume.unsqueeze(1).to(device)
            dvf_list = [vm(ref_volume, v.unsqueeze(1).to(device)) for v in current_volume_list]
            for t, dvf in enumerate(dvf_list):
                z = vae.encode(dvf)
                if isinstance(z, tuple):
                    z = z[0]   # use mu, not logvar
                lat_sum    += z.sum().item()
                lat_sq_sum += z.pow(2).sum().item()
                lat_count  += z.numel()
                # per-phi bin accumulation
                try:
                    paths = vol_files[t] if isinstance(vol_files[t], (list, tuple)) else [vol_files[t]]
                    for b_idx, path in enumerate(paths):
                        pid   = path.split("/")[-2]
                        t_idx = int(path.split("/")[-1][2:-7])
                        phi   = _get_phi(pid, t_idx, cfg.data_dir)
                        bn    = min(int(phi / (0.5 / N_BINS)), N_BINS - 1)
                        z_b   = z[b_idx]
                        bin_sum[bn]    += z_b.sum().item()
                        bin_sq_sum[bn] += z_b.pow(2).sum().item()
                        bin_count[bn]  += z_b.numel()
                except Exception:
                    pass
            print(f"  [{i+1}/{max_batches}]", end="\r")

    lat_mean = lat_sum / lat_count
    lat_std  = max((lat_sq_sum / lat_count - lat_mean ** 2) ** 0.5, 1e-6)

    bin_means = []
    bin_stds  = []
    for bn in range(N_BINS):
        if bin_count[bn] > 0:
            m = bin_sum[bn] / bin_count[bn]
            s = max((bin_sq_sum[bn] / bin_count[bn] - m ** 2) ** 0.5, 1e-6)
        else:
            m, s = lat_mean, lat_std
        bin_means.append(m)
        bin_stds.append(s)

    stats_path = os.path.join(log_dir, "latent_stats.pt")
    torch.save({
        "mean": lat_mean, "std": lat_std,
        "bin_means": bin_means, "bin_stds": bin_stds, "n_bins": N_BINS,
    }, stats_path)
    print(
        f"[compute_stats] mean={lat_mean:.4f}  std={lat_std:.4f}  "
        f"→ saved to {stats_path}"
    )
    print("[compute_stats] Per-phi std:", "  ".join(
        f"bin{b}={bin_stds[b]:.3f}" for b in range(N_BINS)))


# =============================================================================
# Stage 1 — VAE test
# =============================================================================

def test_vae(
    cfg: argparse.Namespace,
    fold: list,
    fold_idx: str,
    dir_name: str,
    vm: torch.nn.Module,
    stn: torch.nn.Module,
    device: torch.device,
) -> None:
    """
    Run VAE reconstruction inference on the test fold and compute metrics.

    For each test sample the function:
      1. Computes ground-truth DVFs with the frozen VoxelMorph (``vm``).
      2. Encodes + decodes each DVF through the VAE to obtain a reconstruction.
      3. Warps the reference volume with both the GT and the reconstructed DVF.
      4. Measures MSE / NCC / SSIM on the warped volumes and geometric error
         on the DVF field itself.

    Results are saved as .npy arrays and a human-readable
    ``summary_metrics.txt`` under ``<logging_dir>/test/<dir_name>/<fold_idx>/``.

    Note: models whose ``forward()`` returns a dict with a ``"recon"`` key
    (MIA_VAE, VQVAE, MonaiVQVAE) are fully supported.  For CVAE, which uses
    a different forward signature, the reconstructed DVF falls back to the
    GT DVF so that the pipeline still runs (metrics will be perfect in that
    case — use train_cvae.py for proper CVAE testing).
    """

    # ---- build & load checkpoint -------------------------------------------
    vae = build_vae(cfg, device)
    log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, f"fold_{fold_idx}", "vae")
    ckpt = os.path.join(log_dir, "model_best_vae.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[test_vae] Checkpoint not found: {ckpt}")
    custom_load(vae, ckpt, device)
    vae.eval()
    print(f"[test_vae] Loaded {type(vae).__name__} from {ckpt}")

    # ---- data & output dirs ------------------------------------------------
    test_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, sequence_list=fold, nb_pred=cfg.tp,
        nb_inputs=cfg.nb_inputs, test=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
    )

    save_dir  = os.path.join(cfg.logging_dir, dir_name, "test", fold_idx)
    vol_dir   = os.path.join(save_dir, "volumes")
    track_dir = os.path.join(save_dir, "tracking")
    for d in (vol_dir, track_dir):
        cond_mkdir(d)

    test_patients = {
        seq.split("/")[0] if "/" in seq else seq[:8] for seq in fold
    }
    with open(os.path.join(save_dir, "patients_test.txt"), "w") as f:
        for p in sorted(test_patients):
            f.write(p + "\n")

    # ---- inference loop ----------------------------------------------------
    MSE_loss, NCC_loss, SSIM_loss, geo_errors = [], [], [], []
    MSE_loss_gt  = []   # [n_samples, tp] — MSE vs true GT frame (not VoxelMorph pseudo-GT)
    phi_values   = []   # [n_samples, tp] — distance from exhale ∈ [0, 0.5]
    cycle_phases = []   # [n_samples, tp] — full cycle, amplitude-based ∈ [0, 1]
    mse_fn = nn.MSELoss(reduction="mean").to(device)

    vae_debug = getattr(cfg, "debug", False)
    _dbg_phi, _dbg_z0_norm, _dbg_vae_mse, _dbg_dvf_mag = [], [], [], []

    with torch.no_grad():
        for idx, (ref_volume_raw, input_volume_list, current_volume_list_raw, vol_file) in enumerate(
            tqdm(test_loader)
        ):
            patient_no = vol_file[0][0].split("/")[-2]
            print(f"\nInference — patient: {patient_no}")

            ref_volume = ref_volume_raw.unsqueeze(1).to(device)
            vmorph_vols, dvf_gt_list, gt_vols = [], [], []

            for t in range(len(current_volume_list_raw)):
                cv = current_volume_list_raw[t].unsqueeze(1).to(device)
                gt_vols.append(cv)
                dvf_vm = vm(ref_volume, cv)
                dvf_gt_list.append(dvf_vm)
                vmorph_vols.append(stn(ref_volume, dvf_vm))

            # ---- get reconstructed DVFs ------------------------------------
            # DVFVAE: forward(dvf) → {"recon": ...}
            dvf_recon_list = [vae.forward(dvf_gt_list[t])["recon"]
                              for t in range(cfg.tp)]

            # ---- per-timepoint metrics -------------------------------------
            this_mse, this_ncc, this_ssim, this_geo, this_phi, this_cycle = [], [], [], [], [], []
            this_mse_gt = []   # MSE vs true GT frame

            for tp in range(cfg.tp):
                dvf_gt    = dvf_gt_list[tp]
                dvf_recon = dvf_recon_list[tp]
                vmorph_v  = vmorph_vols[tp]

                recon_vol = stn(ref_volume, dvf_recon)

                this_mse.append(np.ravel(mse_fn(recon_vol, vmorph_v).item()))
                this_mse_gt.append(np.ravel(mse_fn(recon_vol, gt_vols[tp]).item()))
                this_ncc.append(np.ravel(
                    ncc_loss(recon_vol, vmorph_v, device=device).item()
                ))
                gen = recon_vol[0, 0].cpu().numpy()
                gt  = vmorph_v[0, 0].cpu().numpy()
                this_ssim.append(np.ravel(ss(gen, gt, data_range=gt.max() - gt.min())))

                np_gt   = dvf_gt[0].cpu().numpy()
                np_pred = dvf_recon[0].cpu().numpy()
                err = geo_error(np_gt, np_pred)
                this_geo.append(np.ravel(err))

                # Always extract phi + cycle_phase so metrics_per_phase.py can bin correctly.
                try:
                    tp_files = vol_file[tp] if isinstance(vol_file[tp], (list, tuple)) else [vol_file[tp]]
                    _path    = tp_files[0]
                    _pid     = _path.split("/")[-2]
                    _t_idx   = int(_path.split("/")[-1][2:-7])
                    _phi   = _get_phi(_pid, _t_idx, cfg.data_dir)
                    _cycle = _get_cycle_phase(_pid, _t_idx, cfg.data_dir)
                except Exception:
                    _phi   = float("nan")
                    _cycle = float("nan")
                this_phi.append(_phi)
                this_cycle.append(_cycle)

                if vae_debug:
                    # ── z0 magnitude + VAE reconstruction diagnostic ─────────
                    # Check whether vae_mse is flat across phi:
                    #   flat  → VAE reconstruction quality is phase-independent
                    #   grows → VAE struggles with high-motion frames
                    z = vae.encode(dvf_gt)
                    if isinstance(z, tuple):
                        z = z[0]   # use mu, not logvar
                    vae_mse_dbg = F.mse_loss(dvf_recon, dvf_gt).item()
                    dvf_mag     = dvf_gt.norm(dim=1).mean().item()
                    z0_norm     = z.norm().item()
                    try:
                        tp_files = vol_file[tp] if isinstance(vol_file[tp], (list, tuple)) else [vol_file[tp]]
                        path     = tp_files[0]
                        pid      = path.split("/")[-2]
                        t_idx    = int(path.split("/")[-1][2:-7])
                        phi_val  = _get_phi(pid, t_idx, cfg.data_dir)
                    except Exception:
                        phi_val = float("nan")
                    print(
                        f"[DEBUG VAE] tp={tp}  phi={phi_val:.3f}"
                        f"  dvf_mag={dvf_mag:.4f}"
                        f"  ||z0||={z0_norm:.3f}"
                        f"  z0_std={z.std().item():.4f}"
                        f"  vae_mse={vae_mse_dbg:.5f}"
                    )
                    _dbg_phi.append(phi_val)
                    _dbg_z0_norm.append(z0_norm)
                    _dbg_vae_mse.append(vae_mse_dbg)
                    _dbg_dvf_mag.append(dvf_mag)

                save_tensor_as_nifti(vmorph_v[0, 0], f"vm_volume_t{tp}", vol_dir, iter=idx)
                save_tensor_as_nifti(recon_vol[0, 0], f"recon_volume_t{tp}", vol_dir, iter=idx)

            MSE_loss.append(this_mse)
            MSE_loss_gt.append(this_mse_gt)
            NCC_loss.append(this_ncc)
            SSIM_loss.append(this_ssim)
            geo_errors.append(this_geo)
            phi_values.append(this_phi)
            cycle_phases.append(this_cycle)

    np.save(os.path.join(save_dir, "NCC_loss.npy"),   np.asarray(NCC_loss))
    np.save(os.path.join(save_dir, "MSE_loss.npy"),   np.asarray(MSE_loss))
    np.save(os.path.join(save_dir, "MSE_loss_gt.npy"), np.asarray(MSE_loss_gt))
    np.save(os.path.join(save_dir, "SSIM_loss.npy"),  np.asarray(SSIM_loss))
    np.save(os.path.join(save_dir, "geo_error.npy"),  np.asarray(geo_errors))
    np.save(os.path.join(save_dir, "phi_values.npy"),   np.asarray(phi_values))
    np.save(os.path.join(save_dir, "cycle_phases.npy"), np.asarray(cycle_phases))

    print(
        "\nTest avg  NCC: %.4f  MSE(vm): %.4f  MSE(gt): %.4f  SSIM: %.4f"
        % (np.nanmean(NCC_loss), np.nanmean(MSE_loss), np.nanmean(MSE_loss_gt), np.nanmean(SSIM_loss))
    )

    if vae_debug and _dbg_phi:
        n_bins  = 10
        bin_w   = 0.5 / n_bins
        phi_arr = np.array(_dbg_phi)
        z0_arr  = np.array(_dbg_z0_norm)
        vae_arr = np.array(_dbg_vae_mse)
        dvf_arr = np.array(_dbg_dvf_mag)
        print("\n[DEBUG VAE] Phase-binned summary")
        print(f"  {'phi_bin':>14}  {'n':>4}  {'dvf_mag':>8}  {'||z0||':>8}  {'vae_mse':>10}")
        print("  " + "-" * 54)
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
            "\n    vae_mse grows with phi   → VAE struggles to reconstruct high-motion frames"
            "\n    ||z0|| grows with phi    → high-motion latents may be OOD for the noise schedule"
        )

    # Assemble tracking volumes into NIfTI sequences
    for base_dir in (track_dir, vol_dir):
        for case in next(os.walk(base_dir))[1]:
            if "DVF" in case:
                continue
            path      = os.path.join(base_dir, case, "")
            path_save = path[:-1] + ".nii.gz"
            assemble_volumes(path, path_save, target_imgs=False, downsampled=True)

    summarize_test_metrics(save_dir)


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified VAE trainer")
    p.add_argument("--config",     required=True,
                   help="Path to a YAML config file (see configs/).")
    p.add_argument("--train_test", required=True,
                   choices=["train_vae", "compute_stats", "test_vae"],
                   help="'train_vae' trains the model; 'compute_stats' computes "
                        "latent statistics from an existing checkpoint; "
                        "'test_vae' evaluates reconstruction quality on the test set.")
    p.add_argument("--fold_nb_training", type=int, default=3,
                   help="How many folds to train on (0 = all).")
    p.add_argument("--fold_start", type=int, default=0,
                   help="First fold index to train (default 0). Use to skip already-completed folds.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="[train_vae only] Path to a .pth checkpoint to resume "
                        "training from, e.g. "
                        "'<logging_dir>/logs/04_10/14.30._run/fold_0/vae/model_best_vae.pth'.")
    # Allow ad-hoc overrides: --override key=value
    p.add_argument("--override", nargs="*", default=[],
                   metavar="KEY=VALUE",
                   help="Override any config key, e.g. --override lr_vae=5e-5 batch_size=8")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_cli()

    # Load & patch config
    cfg = load_config(args.config)
    cfg.fold_nb_training = args.fold_nb_training
    cfg.fold_start       = args.fold_start
    if args.checkpoint is not None:          # CLI wins; otherwise keep YAML value
        cfg.checkpoint = args.checkpoint
    _apply_overrides(cfg, args.override)

    print("\n=== Config ===")
    print("\n".join(f"  {k}: {v}" for k, v in sorted(vars(cfg).items())))
    print("==============\n")

    # Global setup
    device = torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() else "cpu")
    cfg.device = device
    
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    VOL_SIZE = (32, 64, 64)
    vm = build_reg_model(cfg, VOL_SIZE, device)

    stn = SpatialTransformer(VOL_SIZE).to(device)

    train_folds, valid_folds, test_folds = make_train_val_test_folds()
        
    fold_nb    = cfg.fold_nb_training or len(train_folds)
    fold_start = getattr(cfg, "fold_start", 0)
    print(f"Training fold(s) {fold_start}..{fold_nb - 1}.")

    dir_name = os.path.join(
        datetime.datetime.now().strftime("%m_%d"),
        datetime.datetime.now().strftime("%H.%M._") + cfg.name,
    )

    if args.train_test == "train_vae":
        for fold_idx in range(fold_start, fold_nb):
            if fold_idx == 0 or fold_idx == 1:
                continue
            else:
                train_vae(
                    cfg      = cfg,
                    folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                    fold_idx = str(fold_idx),
                    dir_name = dir_name,
                    vm       = vm,
                    device   = device,
                )

    elif args.train_test == "compute_stats":
        for fold_idx in range(fold_start, fold_nb):
            compute_latent_stats(
                cfg      = cfg,
                folds    = (train_folds[fold_idx], valid_folds[fold_idx]),
                fold_idx = str(fold_idx),
                dir_name = cfg.vae_dir_name,
                vm       = vm,
                device   = device,
            )

    elif args.train_test == "test_vae":
        test_dir_name = cfg.vae_dir_name

        for fold_idx in range(fold_start, fold_nb):
            test_vae(
                cfg      = cfg,
                fold     = test_folds[fold_idx],
                fold_idx = str(fold_idx),
                dir_name = test_dir_name,
                vm       = vm,
                stn      = stn,
                device   = device,
            )