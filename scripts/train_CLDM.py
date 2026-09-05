"""
Stage 2 training and evaluation pipeline for the TIDAL diffusion model (liver).
Trains UNet3D on a frozen DVFVAE latent space conditioned on TM-Net and RV-Net embeddings.

Usage
-----
# Stage 2 — train
python -m scripts.train_CLDM --config configs/CLDM/UNet3D.yaml --train_test train

# Stage 3 — test
python -m scripts.train_CLDM --config configs/CLDM/UNet3D.yaml --train_test test --checkpoint <path/to/run_dir>
"""

from __future__ import annotations

import argparse
import datetime
import json
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
from mopred.models import SpatialTransformer, EMA

from mopred.data.splits       import make_folds_3fold as make_train_val_test_folds
from mopred.data.data_loaders import NAVIGATOR_4D_Dataset_multitime, NAVIGATOR_4D_Dataset_multitime_continuous
from mopred.data.data_loaders.navigator_4d import _REF_ID as _PATIENT_REF_ID, get_phi as _get_phi, get_cycle_phase as _get_cycle_phase
from mopred.data.loading      import save_params_txt, build_Iseq, extract_patient_ids
from mopred.utils.one_resp_cycle  import temp_crop
from mopred.utils.early_stopping  import EarlyStopping
from mopred.utils.assemble_volumes import assemble_volumes
from mopred.utils.io              import cond_mkdir, custom_load, custom_save, save_tensor_as_nifti
from mopred.utils.losses          import build_criterion, gradient_loss, ncc_loss
from mopred.utils.training        import load_config, _apply_overrides, vae_checkpoint_path, build_scheduler, build_optimizer, build_vae, save_patients, build_diffusion, summarize_test_metrics, _check_weights, build_reg_model
from mopred.utils.dvf_metrics     import motion_amplitude, hf_energy_ratio, hf_energy_ratio_vol, jacobian_det_stats, jacobian_folding_ratio, dvf_cosine_sim_stats, dvf_diversity_diagnostic, geo_error
from mopred.utils.navigator       import navigator_signal, navigator_signal_3planes, navigator_metrics

_VAE_ALIGN_PREFIXES = ("context_encoder.", "align_loss_fn.")


def _cycle_phase_from_path(path: str) -> float:
    """Cycle phase ∈ [0, 1] from file path alone — no image I/O."""
    patient = path.split("/")[-2]
    t_idx   = int(path.split("/")[-1][2:-7])
    inh_idx = _PATIENT_REF_ID[patient][0]
    if t_idx <= inh_idx:
        return 0.5 * t_idx / inh_idx if inh_idx > 0 else 0.0
    else:
        remaining = _CYCLE_LEN - 1 - inh_idx
        return 0.5 + 0.5 * (t_idx - inh_idx) / remaining if remaining > 0 else 0.5


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
    except Exception as e:
        print(f"[compute_phase] FAILED: {type(e).__name__}: {e}")
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

# =============================================================================
# Diversity / contrastive loss
# =============================================================================
def phase_dispersion_loss(z0: torch.Tensor, phi: torch.Tensor, margin_scale: float = 0.5) -> torch.Tensor:
    """
    Hinge loss that pushes z0 latents apart in proportion to their phase
    distance within the batch.
    z0:  (B, C, D, H, W) latent
    phi: (B,) or (B, 1) scalar phase in [0, 1]
    """
    B = z0.shape[0]
    if B < 2:
        return z0.new_zeros(())
    z   = z0.flatten(1)
    phi = phi.view(-1).float()
    z_dist   = torch.cdist(z.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
    phi_dist = (phi.unsqueeze(0) - phi.unsqueeze(1)).abs()
    target = margin_scale * phi_dist
    mask   = ~torch.eye(B, dtype=torch.bool, device=z.device)
    return F.relu(target - z_dist)[mask].mean()


def amplitude_ratio_loss(dvf_pred: torch.Tensor, dvf_gt: torch.Tensor) -> torch.Tensor:
    """Penalises NavRatio ≠ 1 by matching predicted and GT DVF amplitude."""
    pred_amp = dvf_pred.norm(dim=1).mean(dim=(1, 2, 3))
    gt_amp   = dvf_gt.norm(dim=1).mean(dim=(1, 2, 3))
    ratio    = pred_amp / gt_amp.clamp(min=1e-6)
    return (ratio - 1.0).pow(2).mean()


def _compute_diversity_loss(model, ref_volume, Iseq, phase, device, dvf_gt=None,
                             n_steps: int = 1, margin_scale: float = 0.5,
                             compute_amp: bool = False):
    """Anti-collapse regularizer on the model's own single-step z0_hat estimate."""
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
        loss = loss - z0_hat.flatten(1).var(dim=0).mean()

    if z0_gt is not None:
        loss = loss + F.mse_loss(z0_hat, z0_gt)

    amp_loss = z0_hat.new_zeros(())
    if compute_amp and target_dvf is not None:
        dvf_pred = model.decode_latent(z0_hat.detach(), cond_feats[0])
        amp_loss = amplitude_ratio_loss(dvf_pred, target_dvf)

    return loss, amp_loss


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
    if hasattr(model, "set_latent_stats") and cfg.vae_dir_name:
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
    recon_weight_end  = float(getattr(cfg, "recon_weight_end",  cfg.recon_weight))
    smooth_weight_end = float(getattr(cfg, "smooth_weight_end", getattr(cfg, "smooth_weight", 0.0)))
    recon_rampup_epochs  = int(getattr(cfg, "recon_rampup_epochs",  cfg.diffusion_epochs - 1))
    smooth_rampup_epochs = int(getattr(cfg, "smooth_rampup_epochs", cfg.diffusion_epochs - 1))

    model.DEBUG = False
    _shapes_printed = False  # print UNet3D block shapes once, on the first training batch

    for epoch in range(restart_epoch, cfg.diffusion_epochs):
        recon_frac  = min(epoch / max(recon_rampup_epochs, 1),  1.0)
        smooth_frac = min(epoch / max(smooth_rampup_epochs, 1), 1.0)
        current_recon_weight  = cfg.recon_weight + recon_frac * (recon_weight_end - cfg.recon_weight)
        current_smooth_weight = float(getattr(cfg, "smooth_weight", 0.0)) + smooth_frac * (smooth_weight_end - float(getattr(cfg, "smooth_weight", 0.0)))

        print(f"\n[Diffusion] Epoch {epoch}/{cfg.diffusion_epochs - 1}  recon_weight={current_recon_weight:.4f}")
        t0 = time.time()
        model.train()
        if model.vae is not None:
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
            if getattr(cfg, "image_mode", False):
                dvf_gt = current_vols
            else:
                with torch.no_grad():
                    dvf_gt = [vm(ref_volume, v) for v in current_vols]

            Iseq = build_Iseq(
                ref_volume, input_volume_list, current_volume_list,
                cfg, device, include_future=True, vol_files=vol_files,
            )

            phase = compute_phase(vol_files, model.horizon, device)

            if phase is not None and global_step % 100 == 0:
                phi_vals = phase[0].tolist()   # first sample in the batch
                phi_str  = "  ".join(f"t{t}→φ={v:.3f}" for t, v in enumerate(phi_vals))
                # print(f"[Step {global_step}] Phase  {phi_str}   (batch φ range [{phase.min():.3f}, {phase.max():.3f}])")
                phi_vals = phase[0].tolist()
                phi_str  = "  ".join(f"t{t}→φ={v:.3f}" for t, v in enumerate(phi_vals))
                # print(f"[Step {global_step}] Phase  {phi_str}  (batch range [{phase.min():.3f}, {phase.max():.3f}])")
            elif phase is None and global_step % 100 == 0:
                print(f"[Step {global_step}] Phase is None!")

            if model.DEBUG:
                print(f"\n[Batch] ref_volume shape: {ref_volume.shape}, Iseq shape: {Iseq.shape}, dvf_gt[0] shape: {dvf_gt[0].shape if dvf_gt else 'N/A'}")

            if getattr(cfg, "debug_shapes", False) and not _shapes_printed and hasattr(model, "set_shape_debug"):
                model.set_shape_debug(True)

            # ---- unified forward (returns DiffusionOutput) -----------------
            # with autocast("cuda"):
            out = model(ref_volume, current_vols, Iseq, dvf=dvf_gt, criterion=criterion, phase=phase)

            if getattr(cfg, "debug_shapes", False) and not _shapes_printed and hasattr(model, "set_shape_debug"):
                model.set_shape_debug(False)
                _shapes_printed = True

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
                    w_vmorph_vol = current_smooth_weight,
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
            writer.add_scalar("train/recon_weight",  current_recon_weight,  global_step)
            writer.add_scalar("train/smooth_weight", current_smooth_weight, global_step)
            if _div_w > 0.0:
                writer.add_scalar("train/diversity_loss", diversity_loss.item(), global_step)
            if _amp_w > 0.0:
                writer.add_scalar("train/amplitude_loss", amp_loss.item(), global_step)

            global_step += 1

        writer.add_scalar("train/epoch_loss", ep_loss / max(ep_steps, 1), epoch)

        # ---- validation ----------------------------------------------------
        ema.apply_shadow()
        _compute_nav = (epoch % 2 == 0)
        val_loss_vol, val_loss_dvf, nav_dict = _validate_diffusion(
            model, valid_loader, cfg, criterion, device, vm,
            recon_weight=current_recon_weight,
            smooth_weight=current_smooth_weight,
            compute_nav=_compute_nav,
        )
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
                _save_val_vols(model, valid_loader, val_vol_dir, epoch, device, cfg)
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
    recon_weight:  float = 1.0,
    smooth_weight: float = None,
    compute_nav:   bool = False,
) -> tuple:
    """
    Stage-2 validation — MSE(generated_vol, current_vol) + smooth_weight * gradient_loss(generated_dvf).
    Uses inference forward (dvf=None).
    Returns (val_loss_vol, val_loss_dvf, nav_dict | None).
    nav_dict is only computed when compute_nav=True and contains NavRatio, NavCorr,
    NavPredStd, NavPredCV.
    """
    model.eval()
    if smooth_weight is None:
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
                cfg, device, include_future=False, vol_files=vol_files,
            )

            phase = compute_phase(vol_files, model.horizon, device)
            out = model(ref_volume, None, Iseq, dvf=None, ddim_steps=50, phase=phase)

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


def _save_val_vols(model, valid_loader, val_vol_dir, epoch, device, cfg, n_save=3):
    """Save a few generated / GT volume pairs, replacing previous saves."""
    import nibabel as nib

    model.eval()
    batch = next(iter(valid_loader))
    ref_volume, input_volume_list, current_volume_list, vol_files = batch

    ref_volume   = ref_volume.unsqueeze(1).to(device)
    current_vols = [v.unsqueeze(1).to(device) for v in current_volume_list]
    Iseq  = build_Iseq(ref_volume, input_volume_list, None, cfg, device, include_future=False, vol_files=vol_files)
    phase = compute_phase(vol_files, model.horizon, device)

    with torch.no_grad():
        out = model(ref_volume, None, Iseq, dvf=None, ddim_steps=50, phase=phase)

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
_hf_energy_ratio_vol    = hf_energy_ratio_vol
_jacobian_det_stats     = jacobian_det_stats
_jacobian_folding_ratio = jacobian_folding_ratio
_dvf_cosine_sim_stats   = dvf_cosine_sim_stats
_dvf_diversity_diagnostic = dvf_diversity_diagnostic


# ── Voxel spacings (D, H, W) after 0.5× downsampling in H and W ─────────────
_SP_D  = 3.5       # mm / voxel — depth axis (unchanged)
_SP_HW = 1.70 * 2  # mm / voxel — height and width axes
_N_BANDS = 11      # number of anatomical sub-bands per axis for spatial NCC analysis


def _load_landmarks(data_dir: str, patient_id: str) -> dict:
    """
    Load GT landmark positions from GT_landmarks/{patient_id}/*/main.txt.

    Falls back to right.txt (L2-style, flipped axes) when main.txt is absent,
    converting coordinates so landmark_tracking_error samples the DVF at the
    correct voxel (same location as the tracker ROI from _load_patch_centers):
        right.txt: lm_x, lm_y, lm_z  ->  x_syn=128-lm_y, y_syn=lm_x, z_syn=32-lm_z
        recovered: W=x_syn/2=64-lm_y/2, H=y_syn/2=lm_x/2, D=z_syn=32-lm_z

    Returns empty dict when no data is found.
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


from mopred.utils.landmarks import (
      landmark_dvf_error as _landmark_dvf_error,
      landmark_tracking_error as _landmark_tracking_error,
  )


def _dvf_axis_displacement_mm(dvf: "torch.Tensor | None", x_ref: float, y_ref: float, z_ref: float) -> tuple:
    """Sample a DVF at (x_ref, y_ref, z_ref) [original-voxel coords] and
    return (SI_mm, AP_mm) displacement. Returns (nan, nan) if dvf is None."""
    if dvf is None:
        return np.nan, np.nan
    dvf_np = dvf[0].cpu().numpy()   # (3, D, H, W)
    _, D, H, W = dvf_np.shape
    xi = int(np.clip(round(x_ref / 2.0), 0, W - 1))
    yi = int(np.clip(round(y_ref / 2.0), 0, H - 1))
    zi = int(np.clip(round(z_ref),       0, D - 1))
    SI_mm = dvf_np[0, zi, yi, xi] * _SP_D
    AP_mm = dvf_np[1, zi, yi, xi] * _SP_HW
    return SI_mm, AP_mm


def _axis_mean_abs(dvf: "torch.Tensor", axis: int) -> float:
    """Mean absolute displacement along one DVF channel (0=D/SI, 1=H/AP)."""
    return dvf[0, axis].abs().mean().item()


def _ncc_bands(pred: torch.Tensor, gt: torch.Tensor, device,
               axis: int, n_bands: int = _N_BANDS) -> list:
    """NCC per equal sub-band along one spatial axis.

    pred, gt : (B, 1, D, H, W)
    axis     : 0 = D (cranial-caudal / SI), 1 = H (AP), 2 = W (LR)
    Returns  : list of n_bands floats, same sign convention as ncc_loss (∈ [-1, 0]).
    """
    size  = pred.shape[2 + axis]
    edges = np.round(np.linspace(0, size, n_bands + 1)).astype(int)
    out   = []
    for i in range(n_bands):
        s, e = int(edges[i]), int(edges[i + 1])
        if e - s < 2:
            out.append(float("nan"))
            continue
        if axis == 0:
            p, g = pred[:, :, s:e, :, :], gt[:, :, s:e, :, :]
        elif axis == 1:
            p, g = pred[:, :, :, s:e, :], gt[:, :, :, s:e, :]
        else:
            p, g = pred[:, :, :, :, s:e], gt[:, :, :, :, s:e]
        out.append(ncc_loss(p, g, device=device).item())
    return out


def _load_patch_centers(data_dir: str, roi: list, landmark: str = "L2") -> dict:
    """Load per-subject ROI bounds from landmark annotations (32×64×64 voxel space).

    landmark: "L1" uses main.txt (standard axes), "L2" uses right.txt (flipped axes).
    """
    annot_dir = os.path.join(data_dir, "Complemetary_information", "GT_landmarks")
    dx, dy, dz = roi
    bounds = {}
    for subject in sorted(os.listdir(annot_dir)):
        lm_dir    = os.path.join(annot_dir, subject, landmark)
        main_txt  = os.path.join(lm_dir, "main.txt")
        right_txt = os.path.join(lm_dir, "right.txt")
        if os.path.isfile(main_txt):
            data = np.loadtxt(main_txt, dtype=float)
            if data.ndim == 1: data = data[None]
            lm_x, lm_y, lm_z = int(data[0, 0]), int(data[0, 1]), int(data[0, 2])
            x1 = lm_z      - dx // 2
            y1 = lm_y // 2 - dy // 2
            z1 = lm_x // 2 - dz // 2
        elif os.path.isfile(right_txt):
            data = np.loadtxt(right_txt, dtype=float)
            if data.ndim == 1: data = data[None]
            lm_x, lm_y, lm_z = int(data[0, 0]), int(data[0, 1]), int(data[0, 2])
            x1 = (32 - lm_z) - dx // 2
            y1 = lm_x // 2   - dy // 2
            z1 = 64 - lm_y // 2 - dz // 2
        else:
            continue
        x1 = max(0, min(x1, 32 - dx)); x2 = x1 + dx
        y1 = max(0, min(y1, 64 - dy)); y2 = y1 + dy
        z1 = max(0, min(z1, 64 - dz)); z2 = z1 + dz
        bounds[subject] = (x1, x2, y1, y2, z1, z2)
    return bounds


def _build_3d_mask_tracker(vol_files, bounds_dict: dict) -> torch.Tensor:
    """Build (B, 1, 32, 64, 64) float mask with 1s inside each sample's tracker ROI."""
    B = len(vol_files[0]) if isinstance(vol_files[0], (list, tuple)) else len(vol_files)
    first_files = [
        vol_files[0][b] if isinstance(vol_files[0], (list, tuple)) else vol_files[b]
        for b in range(B)
    ]
    masks = []
    for f in first_files:
        subj = os.path.basename(os.path.dirname(f))
        mask = torch.zeros(1, 32, 64, 64, dtype=torch.float32)
        if subj in bounds_dict:
            x1, x2, y1, y2, z1, z2 = bounds_dict[subj]
            mask[0, x1:x2, y1:y2, z1:z2] = 1.0
        masks.append(mask)
    return torch.stack(masks, dim=0)  # (B, 1, D, H, W)


def test(
    cfg:      argparse.Namespace,
    fold:     list,
    fold_idx: str,
    dir_name: str,
    vm:       nn.Module,
    stn:      nn.Module,
    device:   torch.device,
    tracker:  nn.Module | None = None,
    patch_centers: dict | None = None,
) -> None:
    """Run inference and compute metrics for one test fold."""
    aff = [[3.5, 0, 0, 0], [0, 1.70 * 2, 0, 0], [0, 0, 1.70 * 2, 0], [0, 0, 0, 1]]

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
    if hasattr(model, "set_film_debug"):
        model.set_film_debug(False)

    if tracker is not None:
        tr_ckpt = os.path.join(cfg.tracker_checkpoint, f"fold_{fold_idx}", "model_best.pth")
        if not os.path.isfile(tr_ckpt):
            raise FileNotFoundError(f"[Tracker] Checkpoint not found: {tr_ckpt}")
        custom_load(tracker, tr_ckpt, device)
        tracker.eval()
        print(f"[Tracker] Loaded fold-{fold_idx}: {tr_ckpt}")

    # ---- data & output directories -----------------------------------------
    test_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, sequence_list=fold, nb_pred=cfg.tp,
        nb_inputs=cfg.nb_inputs, test=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
    )

    test_subdir = f"test_{cfg.test_tag}" if getattr(cfg, "test_tag", None) else "test"
    save_dir  = os.path.join(dir_name, test_subdir, fold_idx)
    vol_dir   = os.path.join(save_dir, "volumes")
    fig_dir   = os.path.join(save_dir, "figures")
    track_dir = os.path.join(save_dir, "tracking")
    _metrics_only = getattr(cfg, "metrics_only", False)
    cond_mkdir(save_dir)
    if not _metrics_only:
        for d in (vol_dir, fig_dir, track_dir):
            cond_mkdir(d)

    test_patients = {
        seq.split("/")[0] if "/" in seq else seq[:8] for seq in fold
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
    MSE_loss, NCC_loss, SSIM_loss, geo_errors, landmark_error, landmark_error_gt, landmark_dvf_error, landmark_error_vae = [], [], [], [], [], [], [], []
    MSE_loss_gt       = []   # [n_samples, tp] — MSE vs true GT frame (not VoxelMorph pseudo-GT)
    NCC_loss_bands_d  = []   # [n_samples][tp][n_bands] — per-band NCC along D (cranial-caudal)
    NCC_loss_bands_h  = []   # [n_samples][tp][n_bands] — per-band NCC along H (AP)
    NCC_loss_bands_w  = []   # [n_samples][tp][n_bands] — per-band NCC along W (LR)
    nav_signals_gt,    nav_signals_pred    = [], []   # coronal plane (backward compat)
    nav_sagittal_gt,   nav_sagittal_pred   = [], []
    nav_axial_gt,      nav_axial_pred      = [], []
    cycle_phases      = []   # [n_samples, tp] — amplitude-based ∈ [0, 1]
    cycle_phases_time = []   # [n_samples, tp] — time-index-based ∈ [0, 1]
    PSNR_loss         = []
    folding_ratio_gen = []
    temporal_smoothness = []
    vm_error, dvf_vae_error, unet3d_error, dvf_vae_mse = [], [], [], []
    tr_NCC_loss, tr_MSE_loss, tr_geo_errors = [], [], []  # tracker full-vol metrics
    tr_SSIM_loss, tr_PSNR_loss, tr_landmark_error, tr_landmark_dvf_error = [], [], [], []
    landmark_error_main, landmark_dvf_error_main = [], []   # main.txt (e.g. L1) landmarks
    landmark_error_rt,   landmark_dvf_error_rt   = [], []   # right.txt (e.g. L2) landmarks
    tr_landmark_error_main, tr_landmark_dvf_error_main = [], []
    tr_landmark_error_rt,   tr_landmark_dvf_error_rt   = [], []
    mse_fn = nn.MSELoss(reduction="mean").to(device)

    # Accumulation for DVF diversity diagnostic
    _diag_pids:     list = []
    _diag_gt_vecs:  list = []
    _diag_gen_vecs: list = []
    _diag_amp_gt,   _diag_amp_gen   = [], []
    _diag_hf_gt,    _diag_hf_gen    = [], []
    _diag_hf_vol_gt, _diag_hf_vol_gen = [], []   # image-space HF ratio
    _diag_jstd_gt,  _diag_jstd_gen  = [], []
    _diag_jmean_gt, _diag_jmean_gen = [], []

    trajectory_by_patient: dict = {}
    patient_ids_all: list = []
    vm_ncc_all,      dvf_vae_ncc_all  = [], []
    vm_ssim_all,     dvf_vae_ssim_all = [], []
    dvf_amp_gt_all,    dvf_amp_pred_all    = [], []
    dvf_si_amp_gt_all, dvf_si_amp_pred_all = [], []
    dvf_ap_amp_gt_all, dvf_ap_amp_pred_all = [], []
    dvf_lr_amp_gt_all, dvf_lr_amp_pred_all = [], []

    # # Step 2: Check if context varies across patients (mode-collapse detection)
    # if model.DEBUG:
    #     with torch.no_grad():
    #         _b1 = _b2 = None
    #         _pid1 = None
    #         for _batch in test_loader:
    #             _pid = _batch[3][0][0].split("/")[-2]
    #             if _b1 is None:
    #                 _b1, _pid1 = _batch, _pid
    #             elif _pid != _pid1:
    #                 _b2 = _batch
    #                 break
    #         if _b1 is not None and _b2 is not None:
    #             _pid2 = _b2[3][0][0].split("/")[-2]
    #             _Vref1 = _b1[0].unsqueeze(1).to(device)
    #             _Iseq1 = build_Iseq(_Vref1, _b1[1], None, cfg, device, include_future=False)
    #             _Vref2 = _b2[0].unsqueeze(1).to(device)
    #             _Iseq2 = build_Iseq(_Vref2, _b2[1], None, cfg, device, include_future=False)
    #             _f1, _c1, _ = model.encode_context(_Vref1, _Iseq1, False)
    #             _f2, _c2, _ = model.encode_context(_Vref2, _Iseq2, False)
    #             print(f"[Step2] comparing patient '{_pid1}' vs '{_pid2}'")
    #             print(f"[Step2] f_ref cosine sim: {F.cosine_similarity(_f1, _f2).mean():.3f}")
    #             print(f"[Step2] cond  cosine sim: {F.cosine_similarity(_c1[0], _c2[0]).mean():.3f}")
    #             if F.cosine_similarity(_f1, _f2).mean() > 0.95:
    #                 print("[Step2] WARNING: f_ref may have collapsed to a constant representation")
    #             if F.cosine_similarity(_c1[0], _c2[0]).mean() > 0.95:
    #                 print("[Step2] WARNING: cond may have collapsed to a constant representation")
    #         else:
    #             print("[Step2] Could not find two batches from different patients")

    # Accumulators for z0/VAE phase diagnostic (model.DEBUG only).
    _dbg_phi, _dbg_z0_norm, _dbg_vae_mse, _dbg_dvf_mag = [], [], [], []

    with torch.no_grad():
        for idx, (ref_volume, input_volume_list, current_volume_list, vol_file) in enumerate(
            tqdm(test_loader)
        ):
            patient_no = vol_file[0][0].split("/")[-2]
            print(f"\nInference — patient: {patient_no}")

            ref_volume   = ref_volume.unsqueeze(1).to(device)
            vmorph_volume, dvf_list = [], []

            for vol in range(len(current_volume_list)):
                current_volume_list[vol] = current_volume_list[vol].unsqueeze(1).to(device)
                dvf_vm = vm(ref_volume, current_volume_list[vol])
                dvf_list.append(dvf_vm)
                vmorph_volume.append(stn(ref_volume, dvf_vm))

            Iseq = build_Iseq(
                ref_volume, input_volume_list, None,
                cfg, device, include_future=False, vol_files=vol_file,
            )

            if tracker is not None:
                _, _cond_feat = model.encode_context(ref_volume, Iseq, is_training=False)
                if isinstance(_cond_feat, (list, tuple)):
                    _zts = list(_cond_feat)
                    if len(_zts) != cfg.tp:
                        _zts = [_zts[0]] * cfg.tp
                else:
                    _zts = [_cond_feat] * cfg.tp

            phase = compute_phase(vol_file, model.horizon, device)

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

            _seed = getattr(cfg, "test_seed", None)
            if _seed is not None:
                torch.manual_seed(_seed)
                torch.cuda.manual_seed_all(_seed)

            out = model(ref_volume, None, Iseq, dvf=None, phase=phase,
                        fixed_noise=getattr(cfg, "fixed_noise", False))
            if idx == 0 and hasattr(model, "set_film_debug"):
                model.set_film_debug(False)
            generated_dvf  = out.generated_dvf
            generated_vols = out.generated_vols

            if tracker is not None:
                # Three separate clones:
                #  _mm_dvf_fv    — full-vol, used to build combined DVF after tracker
                #  _mm_dvf_clone — disposable tracker input (tracker overwrites list elements
                #                  in-place: att crops to ROI, unet zeros outside mask)
                #  _gt_dvf_clone — protects dvf_list from the same in-place overwrite
                # generated_dvf and dvf_list are therefore never touched → MM metrics unchanged
                _mm_dvf_fv    = [d.clone() for d in generated_dvf]
                _mm_dvf_clone = [d.clone() for d in generated_dvf]
                _gt_dvf_clone = [d.clone() for d in dvf_list]
                _tr_mask = _build_3d_mask_tracker(vol_file, patch_centers).to(device)
                _, _, _, _tr_dvf_list, _, _ = tracker(
                    _zts, _gt_dvf_clone, _mm_dvf_clone, ref_volume, _tr_mask)
                _seq0    = vol_file[0][0].split("/")[-2]
                _bounds0 = patch_centers.get(_seq0)
                _tr_combined_dvf = []
                for _tp in range(cfg.tp):
                    _combined = _mm_dvf_fv[_tp].clone()
                    if _bounds0 is not None and _tr_dvf_list[_tp].shape != _mm_dvf_fv[_tp].shape:
                        bx1, bx2, by1, by2, bz1, bz2 = _bounds0
                        _combined[:, :, bx1:bx2, by1:by2, bz1:bz2] = _tr_dvf_list[_tp]
                    elif _tr_dvf_list[_tp].shape == _mm_dvf_fv[_tp].shape:
                        _combined = _tr_dvf_list[_tp] + _mm_dvf_fv[_tp] * (1 - _tr_mask)
                    _tr_combined_dvf.append(_combined)

            # ---- tracking volumes ------------------------------------------
            if not _metrics_only:
              for tp in range(min(len(generated_vols), cfg.tp)):
                v_file     = vol_file[tp][0]
                volume_idx = int(v_file.split("/")[-1][2:-7])
                sequence   = v_file.split("/")[-2]
                if volume_idx in range(
                    temp_crop[sequence][0], temp_crop[sequence][1] + 1
                ):
                    save_tensor_as_nifti(
                        generated_vols[tp][0, 0],
                        sequence + f"_t{tp}", track_dir, iter=volume_idx,
                    )
                    save_tensor_as_nifti(
                        vmorph_volume[tp][0, 0],
                        sequence + f"_t{tp}_GT", track_dir, iter=volume_idx,
                    )

            # ---- per-timepoint metrics -------------------------------------
            this_mse, this_ncc, this_ssim, this_geo, this_lm, this_lm_gt, this_lm_dvf, this_lm_vae = [], [], [], [], [], [], [], []
            this_ncc_bands_d = []  # [tp][n_bands] — per-band NCC along D (cranial-caudal)
            this_ncc_bands_h = []  # [tp][n_bands] — per-band NCC along H (AP)
            this_ncc_bands_w = []  # [tp][n_bands] — per-band NCC along W (LR)
            this_mse_gt = []   # MSE vs true GT frame
            this_nav_gt,    this_nav_pred    = [], []
            this_nav_sag_gt, this_nav_sag_pred = [], []
            this_nav_ax_gt,  this_nav_ax_pred  = [], []
            this_cycle = []
            this_psnr  = []
            this_fold  = []
            _dvf_seq_gen = []
            this_vm, this_dvfvae, this_unet3d, this_dvfvae_mse = [], [], [], []
            this_vm_ncc,    this_dvfvae_ncc  = [], []
            this_vm_ssim,   this_dvfvae_ssim = [], []
            this_amp_gt,    this_amp_pred    = [], []
            this_si_amp_gt, this_si_amp_pred = [], []
            this_ap_amp_gt, this_ap_amp_pred = [], []
            this_lr_amp_gt, this_lr_amp_pred = [], []
            this_tr_ncc, this_tr_mse, this_tr_geo = [], [], []
            this_tr_ssim, this_tr_psnr, this_tr_lm, this_tr_lm_dvf = [], [], [], []
            this_lm_main,    this_lm_dvf_main    = [], []
            this_lm_rt,      this_lm_dvf_rt      = [], []
            this_tr_lm_main, this_tr_lm_dvf_main = [], []
            this_tr_lm_rt,   this_tr_lm_dvf_rt   = [], []

            for tp in range(cfg.tp):
                if not _metrics_only:
                    save_tensor_as_nifti(
                        vmorph_volume[tp][0, 0],
                        f"vm_volume_t{tp}", vol_dir, iter=idx,
                    )

                if tp < len(generated_vols) and generated_vols[tp] is not None:
                    if not _metrics_only:
                        save_tensor_as_nifti(
                            generated_vols[tp][0, 0],
                            f"generated_volume_t{tp}", vol_dir, iter=idx, aff=aff,
                        )
                    # image_mode: the model is trained to predict the raw current
                    # volume directly (see train_diffusion / _validate_diffusion),
                    # not the VoxelMorph-warped reconstruction — use that as GT here too.
                    if getattr(cfg, "image_mode", False):
                        gt_vol = current_volume_list[tp]
                    else:
                        gt_vol = vmorph_volume[tp]

                    _img_mse = mse_fn(generated_vols[tp], gt_vol)
                    this_mse.append(np.ravel(_img_mse.item()))
                    this_mse_gt.append(np.ravel(mse_fn(generated_vols[tp], current_volume_list[tp]).item()))
                    this_psnr.append(np.ravel(
                        (10.0 * torch.log10(1.0 / (_img_mse + 1e-8))).item()
                    ))
                    this_ncc.append(np.ravel(
                        ncc_loss(generated_vols[tp], gt_vol,
                                 device=device).item()
                    ))
                    this_ncc_bands_d.append(_ncc_bands(generated_vols[tp], gt_vol, device, axis=0))
                    this_ncc_bands_h.append(_ncc_bands(generated_vols[tp], gt_vol, device, axis=1))
                    this_ncc_bands_w.append(_ncc_bands(generated_vols[tp], gt_vol, device, axis=2))
                    gen = generated_vols[tp][0, 0].cpu().numpy()
                    gt  = gt_vol[0, 0].cpu().numpy()
                    this_ssim.append(np.ravel(
                        ss(gen, gt, data_range=gt.max() - gt.min())
                    ))
                    # navigator signal: 3 orthogonal planes (coronal / sagittal / axial)
                    _nc_gt,  _ns_gt,  _na_gt  = navigator_signal_3planes(current_volume_list[tp], ref_volume)
                    _nc_pred, _ns_pred, _na_pred = navigator_signal_3planes(generated_vols[tp],    ref_volume)
                    this_nav_gt.append(_nc_gt);      this_nav_pred.append(_nc_pred)
                    this_nav_sag_gt.append(_ns_gt);  this_nav_sag_pred.append(_ns_pred)
                    this_nav_ax_gt.append(_na_gt);   this_nav_ax_pred.append(_na_pred)

                    # image-space HF ratio: gen < gt  →  model losing high-freq detail
                    _diag_hf_vol_gt.append(_hf_energy_ratio_vol(gt_vol))
                    _diag_hf_vol_gen.append(_hf_energy_ratio_vol(generated_vols[tp]))

                    if tracker is not None:
                        _cb_vol = stn(ref_volume, _tr_combined_dvf[tp])
                        _tr_mse_val = mse_fn(_cb_vol, gt_vol)
                        this_tr_ncc.append(ncc_loss(_cb_vol, gt_vol, device=device).item())
                        this_tr_mse.append(_tr_mse_val.item())
                        this_tr_psnr.append((10.0 * torch.log10(1.0 / (_tr_mse_val + 1e-8))).item())
                        _tr_gen = _cb_vol[0, 0].cpu().numpy()
                        _tr_gt  = gt_vol[0, 0].cpu().numpy()
                        this_tr_ssim.append(ss(_tr_gen, _tr_gt, data_range=_tr_gt.max() - _tr_gt.min()))
                else:
                    # Model did not generate this time-point (CausalDiT single-step).
                    if tracker is not None:
                        this_tr_ncc.append(np.nan)
                        this_tr_mse.append(np.nan)
                        this_tr_psnr.append(np.nan)
                        this_tr_ssim.append(np.nan)
                    this_mse.append(np.ravel(np.nan))
                    this_mse_gt.append(np.ravel(np.nan))
                    this_psnr.append(np.ravel(np.nan))
                    this_ncc.append(np.ravel(np.nan))
                    this_ncc_bands_d.append([float("nan")] * _N_BANDS)
                    this_ncc_bands_h.append([float("nan")] * _N_BANDS)
                    this_ncc_bands_w.append([float("nan")] * _N_BANDS)
                    this_ssim.append(np.ravel(np.nan))
                    this_nav_gt.append(np.nan);     this_nav_pred.append(np.nan)
                    this_nav_sag_gt.append(np.nan); this_nav_sag_pred.append(np.nan)
                    this_nav_ax_gt.append(np.nan);  this_nav_ax_pred.append(np.nan)

                if not _metrics_only:
                    save_tensor_as_nifti(
                        dvf_list[tp][0], f"DVF_t{tp}", vol_dir, iter=idx,
                    )

                if tp < len(generated_dvf) and generated_dvf[tp] is not None:
                    if not _metrics_only:
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

                    if tracker is not None:
                        _tr_np = _tr_combined_dvf[tp][0].cpu().numpy()
                        this_tr_geo.append(np.ravel(geo_error(np_gt, _tr_np)))

                    # ---- decomposed error sources (VoxelMorph / DVF-VAE / UNet3D) ----
                    _ct_np = current_volume_list[tp][0, 0].cpu().numpy()
                    this_vm.append(mse_fn(vmorph_volume[tp], current_volume_list[tp]).item())
                    this_vm_ncc.append(ncc_loss(vmorph_volume[tp], current_volume_list[tp], device=device).item())
                    _vm_np = vmorph_volume[tp][0, 0].cpu().numpy()
                    this_vm_ssim.append(ss(_vm_np, _ct_np, data_range=_ct_np.max() - _ct_np.min()))
                    _dvf_vae_r = None
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
                    _lm_all  = all_landmarks.get(patient_no, {})
                    _lm_main = {k: v for k, v in _lm_all.items() if not k.endswith("_RT")}
                    _lm_rt   = {k: v for k, v in _lm_all.items() if k.endswith("_RT")}
                    lm_kw = dict(
                        landmarks  = _lm_all,
                        volume_idx = volume_idx_tp,
                        patient_id = patient_no,
                    )
                    lm_kw_main = dict(landmarks=_lm_main, volume_idx=volume_idx_tp, patient_id=patient_no)
                    lm_kw_rt   = dict(landmarks=_lm_rt,   volume_idx=volume_idx_tp, patient_id=patient_no)
                    if tracker is not None:
                        this_tr_lm.append(_landmark_tracking_error(_tr_combined_dvf[tp], **lm_kw))
                        this_tr_lm_dvf.append(_landmark_dvf_error(_tr_combined_dvf[tp], dvf_list[tp], **lm_kw))
                        this_tr_lm_main.append(_landmark_tracking_error(_tr_combined_dvf[tp], **lm_kw_main))
                        this_tr_lm_dvf_main.append(_landmark_dvf_error(_tr_combined_dvf[tp], dvf_list[tp], **lm_kw_main))
                        this_tr_lm_rt.append(_landmark_tracking_error(_tr_combined_dvf[tp], **lm_kw_rt))
                        this_tr_lm_dvf_rt.append(_landmark_dvf_error(_tr_combined_dvf[tp], dvf_list[tp], **lm_kw_rt))
                    this_lm.append(_landmark_tracking_error(generated_dvf[tp], **lm_kw))
                    this_lm_gt.append(_landmark_tracking_error(dvf_list[tp],   **lm_kw))
                    this_lm_dvf.append(_landmark_dvf_error(generated_dvf[tp], dvf_list[tp], **lm_kw))
                    this_lm_main.append(_landmark_tracking_error(generated_dvf[tp], **lm_kw_main))
                    this_lm_dvf_main.append(_landmark_dvf_error(generated_dvf[tp], dvf_list[tp], **lm_kw_main))
                    this_lm_rt.append(_landmark_tracking_error(generated_dvf[tp], **lm_kw_rt))
                    this_lm_dvf_rt.append(_landmark_dvf_error(generated_dvf[tp], dvf_list[tp], **lm_kw_rt))
                    this_lm_vae.append(
                        _landmark_tracking_error(_dvf_vae_r, **lm_kw) if _dvf_vae_r is not None else np.nan
                    )

                    try:
                        this_cycle.append(_get_cycle_phase(v_file_tp.split("/")[-2], volume_idx_tp, cfg.data_dir))
                    except Exception:
                        this_cycle.append(float("nan"))

                    # Diversity diagnostic accumulation
                    _diag_pids.append(patient_no)
                    _diag_gt_vecs.append(dvf_list[tp].flatten(1).cpu())
                    _diag_gen_vecs.append(generated_dvf[tp].flatten(1).cpu())
                    _diag_amp_gt.append(_motion_amplitude(dvf_list[tp]))
                    _diag_amp_gen.append(_motion_amplitude(generated_dvf[tp]))

                    if tp == 0:
                        volume_idx_tp0 = int(vol_file[0][0].split("/")[-1][2:-7])
                        patient_landmarks = all_landmarks.get(patient_no, {})
                        if patient_landmarks:
                            lm_name = sorted(patient_landmarks.keys())[0]
                            positions = patient_landmarks[lm_name]
                            if 1 in positions:
                                x_ref, y_ref, z_ref = positions[1]
                                SI_gt_dvf,   AP_gt_dvf   = _dvf_axis_displacement_mm(dvf_list[0],       x_ref, y_ref, z_ref)
                                SI_pred_dvf, AP_pred_dvf = _dvf_axis_displacement_mm(generated_dvf[0],  x_ref, y_ref, z_ref)

                                trajectory_by_patient.setdefault(patient_no, []).append(dict(
                                    volume_idx=volume_idx_tp0,
                                    SI_gt_dvf=float(SI_gt_dvf),     AP_gt_dvf=float(AP_gt_dvf),
                                    SI_pred_dvf=float(SI_pred_dvf), AP_pred_dvf=float(AP_pred_dvf),
                                ))

                    this_amp_gt.append(_motion_amplitude(dvf_list[tp]))
                    this_amp_pred.append(_motion_amplitude(generated_dvf[tp]))
                    this_si_amp_gt.append(_axis_mean_abs(dvf_list[tp],        axis=0))
                    this_si_amp_pred.append(_axis_mean_abs(generated_dvf[tp], axis=0))
                    this_ap_amp_gt.append(_axis_mean_abs(dvf_list[tp],        axis=1))
                    this_ap_amp_pred.append(_axis_mean_abs(generated_dvf[tp], axis=1))
                    this_lr_amp_gt.append(_axis_mean_abs(dvf_list[tp],        axis=2))
                    this_lr_amp_pred.append(_axis_mean_abs(generated_dvf[tp], axis=2))
                    _diag_hf_gt.append(_hf_energy_ratio(dvf_list[tp]))
                    _diag_hf_gen.append(_hf_energy_ratio(generated_dvf[tp]))
                    _jmgt,  _jstgt  = _jacobian_det_stats(dvf_list[tp])
                    _jmgen, _jstgen = _jacobian_det_stats(generated_dvf[tp])
                    _diag_jmean_gt.append(_jmgt);   _diag_jstd_gt.append(_jstgt)
                    _diag_jmean_gen.append(_jmgen); _diag_jstd_gen.append(_jstgen)

                    this_fold.append(_jacobian_folding_ratio(generated_dvf[tp]))
                    _dvf_seq_gen.append(generated_dvf[tp].detach())
                else:
                    this_geo.append(np.ravel(np.nan))
                    if tracker is not None:
                        this_tr_geo.append(np.ravel(np.nan))
                        this_tr_lm.append(np.nan)
                        this_tr_lm_dvf.append(np.nan)
                        this_tr_lm_main.append(np.nan);  this_tr_lm_dvf_main.append(np.nan)
                        this_tr_lm_rt.append(np.nan);    this_tr_lm_dvf_rt.append(np.nan)
                    this_lm.append(np.nan)
                    this_lm_gt.append(np.nan)
                    this_lm_dvf.append(np.nan)
                    this_lm_main.append(np.nan);   this_lm_dvf_main.append(np.nan)
                    this_lm_rt.append(np.nan);     this_lm_dvf_rt.append(np.nan)
                    this_lm_vae.append(np.nan)
                    this_fold.append(np.nan)
                    this_vm.append(np.nan)
                    this_vm_ncc.append(np.nan);    this_vm_ssim.append(np.nan)
                    this_dvfvae.append(np.nan)
                    this_unet3d.append(np.nan)
                    this_dvfvae_mse.append(np.nan)
                    this_dvfvae_ncc.append(np.nan); this_dvfvae_ssim.append(np.nan)
                    this_amp_gt.append(np.nan);    this_amp_pred.append(np.nan)
                    this_si_amp_gt.append(np.nan); this_si_amp_pred.append(np.nan)
                    this_ap_amp_gt.append(np.nan); this_ap_amp_pred.append(np.nan)
                    this_lr_amp_gt.append(np.nan); this_lr_amp_pred.append(np.nan)
                    try:
                        _vf = vol_file[tp][0] if isinstance(vol_file[tp], (list, tuple)) else vol_file[tp]
                        this_cycle.append(_get_cycle_phase(_vf.split("/")[-2], int(_vf.split("/")[-1][2:-7]), cfg.data_dir))
                    except Exception:
                        this_cycle.append(float("nan"))

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
            NCC_loss_bands_d.append(this_ncc_bands_d)
            NCC_loss_bands_h.append(this_ncc_bands_h)
            NCC_loss_bands_w.append(this_ncc_bands_w)
            SSIM_loss.append(this_ssim)
            geo_errors.append(this_geo)
            landmark_error.append(this_lm)
            landmark_error_gt.append(this_lm_gt)
            landmark_dvf_error.append(this_lm_dvf)
            landmark_error_vae.append(this_lm_vae)
            landmark_error_main.append(this_lm_main);   landmark_dvf_error_main.append(this_lm_dvf_main)
            landmark_error_rt.append(this_lm_rt);       landmark_dvf_error_rt.append(this_lm_dvf_rt)
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
            patient_ids_all.append(patient_no)
            dvf_amp_gt_all.append(this_amp_gt);       dvf_amp_pred_all.append(this_amp_pred)
            dvf_si_amp_gt_all.append(this_si_amp_gt); dvf_si_amp_pred_all.append(this_si_amp_pred)
            dvf_ap_amp_gt_all.append(this_ap_amp_gt); dvf_ap_amp_pred_all.append(this_ap_amp_pred)
            dvf_lr_amp_gt_all.append(this_lr_amp_gt); dvf_lr_amp_pred_all.append(this_lr_amp_pred)
            if tracker is not None:
                tr_NCC_loss.append(this_tr_ncc)
                tr_MSE_loss.append(this_tr_mse)
                tr_geo_errors.append(this_tr_geo)
                tr_SSIM_loss.append(this_tr_ssim)
                tr_PSNR_loss.append(this_tr_psnr)
                tr_landmark_error.append(this_tr_lm)
                tr_landmark_dvf_error.append(this_tr_lm_dvf)
                tr_landmark_error_main.append(this_tr_lm_main);   tr_landmark_dvf_error_main.append(this_tr_lm_dvf_main)
                tr_landmark_error_rt.append(this_tr_lm_rt);       tr_landmark_dvf_error_rt.append(this_tr_lm_dvf_rt)

    np.save(os.path.join(save_dir, "vm_ncc.npy"),       np.asarray(vm_ncc_all))
    np.save(os.path.join(save_dir, "dvf_vae_ncc.npy"), np.asarray(dvf_vae_ncc_all))
    np.save(os.path.join(save_dir, "vm_ssim.npy"),      np.asarray(vm_ssim_all))
    np.save(os.path.join(save_dir, "dvf_vae_ssim.npy"),np.asarray(dvf_vae_ssim_all))
    np.save(os.path.join(save_dir, "NCC_loss.npy"),          np.asarray(NCC_loss))
    np.save(os.path.join(save_dir, "NCC_loss_bands_d.npy"), np.array(NCC_loss_bands_d, dtype=object))
    np.save(os.path.join(save_dir, "NCC_loss_bands_h.npy"), np.array(NCC_loss_bands_h, dtype=object))
    np.save(os.path.join(save_dir, "NCC_loss_bands_w.npy"), np.array(NCC_loss_bands_w, dtype=object))
    np.save(os.path.join(save_dir, "MSE_loss.npy"),          np.asarray(MSE_loss))
    np.save(os.path.join(save_dir, "MSE_loss_gt.npy"),       np.asarray(MSE_loss_gt))
    np.save(os.path.join(save_dir, "SSIM_loss.npy"),         np.asarray(SSIM_loss))
    np.save(os.path.join(save_dir, "geo_error.npy"),         np.asarray(geo_errors))
    np.save(os.path.join(save_dir, "cycle_phases.npy"),      np.asarray(cycle_phases))
    np.save(os.path.join(save_dir, "cycle_phases_time.npy"), np.asarray(cycle_phases_time))
    np.save(os.path.join(save_dir, "landmark_error.npy"),
            np.array(landmark_error, dtype=object))
    np.save(os.path.join(save_dir, "landmark_error_gt.npy"),
            np.array(landmark_error_gt, dtype=object))
    np.save(os.path.join(save_dir, "landmark_dvf_error.npy"),
            np.array(landmark_dvf_error, dtype=object))
    np.save(os.path.join(save_dir, "landmark_error_vae.npy"),
            np.array(landmark_error_vae, dtype=object))
    np.save(os.path.join(save_dir, "landmark_error_main.npy"),
            np.array(landmark_error_main, dtype=object))
    np.save(os.path.join(save_dir, "landmark_dvf_error_main.npy"),
            np.array(landmark_dvf_error_main, dtype=object))
    np.save(os.path.join(save_dir, "landmark_error_rt.npy"),
            np.array(landmark_error_rt, dtype=object))
    np.save(os.path.join(save_dir, "landmark_dvf_error_rt.npy"),
            np.array(landmark_dvf_error_rt, dtype=object))
    np.save(os.path.join(save_dir, "nav_signals_gt.npy"),
            np.array(nav_signals_gt,  dtype=object))
    np.save(os.path.join(save_dir, "nav_signals_pred.npy"),
            np.array(nav_signals_pred, dtype=object))
    np.save(os.path.join(save_dir, "nav_sagittal_gt.npy"),
            np.array(nav_sagittal_gt,  dtype=object))
    np.save(os.path.join(save_dir, "nav_sagittal_pred.npy"),
            np.array(nav_sagittal_pred, dtype=object))
    np.save(os.path.join(save_dir, "nav_axial_gt.npy"),
            np.array(nav_axial_gt,  dtype=object))
    np.save(os.path.join(save_dir, "nav_axial_pred.npy"),
            np.array(nav_axial_pred, dtype=object))
    np.save(os.path.join(save_dir, "PSNR_loss.npy"),
            np.array(PSNR_loss, dtype=object))
    np.save(os.path.join(save_dir, "folding_ratio.npy"),
            np.array(folding_ratio_gen, dtype=object))
    np.save(os.path.join(save_dir, "temporal_smoothness.npy"),
            np.asarray(temporal_smoothness))
    np.save(os.path.join(save_dir, "vm_error.npy"),      np.asarray(vm_error))
    np.save(os.path.join(save_dir, "dvf_vae_error.npy"), np.asarray(dvf_vae_error))
    np.save(os.path.join(save_dir, "unet3d_error.npy"),  np.asarray(unet3d_error))
    np.save(os.path.join(save_dir, "dvf_vae_mse.npy"),   np.asarray(dvf_vae_mse))
    if tracker is not None:
        np.save(os.path.join(save_dir, "tracker_NCC_loss.npy"),     np.asarray(tr_NCC_loss))
        np.save(os.path.join(save_dir, "tracker_MSE_loss.npy"),     np.asarray(tr_MSE_loss))
        np.save(os.path.join(save_dir, "tracker_geo_error.npy"),    np.array(tr_geo_errors,    dtype=object))
        np.save(os.path.join(save_dir, "tracker_SSIM_loss.npy"),    np.asarray(tr_SSIM_loss))
        np.save(os.path.join(save_dir, "tracker_PSNR_loss.npy"),    np.array(tr_PSNR_loss,     dtype=object))
        np.save(os.path.join(save_dir, "tracker_landmark_error.npy"),     np.array(tr_landmark_error,     dtype=object))
        np.save(os.path.join(save_dir, "tracker_landmark_dvf_error.npy"), np.array(tr_landmark_dvf_error, dtype=object))
        np.save(os.path.join(save_dir, "tracker_landmark_error_main.npy"),     np.array(tr_landmark_error_main,     dtype=object))
        np.save(os.path.join(save_dir, "tracker_landmark_dvf_error_main.npy"), np.array(tr_landmark_dvf_error_main, dtype=object))
        np.save(os.path.join(save_dir, "tracker_landmark_error_rt.npy"),     np.array(tr_landmark_error_rt,     dtype=object))
        np.save(os.path.join(save_dir, "tracker_landmark_dvf_error_rt.npy"), np.array(tr_landmark_dvf_error_rt, dtype=object))

    with open(os.path.join(save_dir, "landmark_trajectory.json"), "w") as _f:
        json.dump(trajectory_by_patient, _f, indent=2)
    np.save(os.path.join(save_dir, "patient_ids.npy"),
            np.asarray(patient_ids_all, dtype=object))
    np.save(os.path.join(save_dir, "dvf_amp_gt.npy"),      np.asarray(dvf_amp_gt_all))
    np.save(os.path.join(save_dir, "dvf_amp_pred.npy"),    np.asarray(dvf_amp_pred_all))
    np.save(os.path.join(save_dir, "dvf_si_amp_gt.npy"),   np.asarray(dvf_si_amp_gt_all))
    np.save(os.path.join(save_dir, "dvf_si_amp_pred.npy"), np.asarray(dvf_si_amp_pred_all))
    np.save(os.path.join(save_dir, "dvf_ap_amp_gt.npy"),   np.asarray(dvf_ap_amp_gt_all))
    np.save(os.path.join(save_dir, "dvf_ap_amp_pred.npy"), np.asarray(dvf_ap_amp_pred_all))
    np.save(os.path.join(save_dir, "dvf_lr_amp_gt.npy"),   np.asarray(dvf_lr_amp_gt_all))
    np.save(os.path.join(save_dir, "dvf_lr_amp_pred.npy"), np.asarray(dvf_lr_amp_pred_all))

    lm_mean     = np.nanmean([v for row in landmark_error     for v in row])
    lm_gt_mean  = np.nanmean([v for row in landmark_error_gt  for v in row])
    lm_dvf_mean = np.nanmean([v for row in landmark_dvf_error for v in row])
    all_nav_gt   = [v for row in nav_signals_gt   for v in row if not np.isnan(float(v))]
    all_nav_pred = [v for row in nav_signals_pred for v in row if not np.isnan(float(v))]
    nav_m = navigator_metrics(all_nav_gt, all_nav_pred)
    print(
        "\nTest avg  NCC: %.4f  MSE(vm): %.4f  MSE(gt): %.4f  SSIM: %.4f  PSNR: %.2f dB  Landmark: %.4f mm  LM-GT: %.4f mm  LM-DVF: %.4f mm  NavRatio: %.3f  NavCorr: %.3f  Folding: %.6f  TempSmooth: %.6f"
        % (
            np.nanmean(NCC_loss),
            np.nanmean(MSE_loss),
            np.nanmean(MSE_loss_gt),
            np.nanmean(SSIM_loss),
            np.nanmean(PSNR_loss),
            lm_mean,
            lm_gt_mean,
            lm_dvf_mean,
            nav_m["amplitude_ratio"],
            nav_m["phase_corr"],
            np.nanmean(folding_ratio_gen),
            np.nanmean(temporal_smoothness),
        )
    )
    if tracker is not None:
        print(
            "Tracker    NCC: %.4f  MSE(vm): %.4f  Geo: %.4f mm  (MM: NCC=%.4f  MSE=%.4f  Geo=%.4f mm)"
            % (
                np.nanmean(tr_NCC_loss),
                np.nanmean(tr_MSE_loss),
                np.nanmean([v for row in tr_geo_errors for item in row for v in np.asarray(item).ravel()]),
                np.nanmean(NCC_loss),
                np.nanmean(MSE_loss),
                np.nanmean([v for row in geo_errors for item in row for v in np.asarray(item).ravel()]),
            )
        )

    # ---- assemble tracking volumes into NIfTI sequences --------------------
    if not _metrics_only:
        for base_dir in (track_dir, vol_dir):
            for case in next(os.walk(base_dir))[1]:
                if "DVF" in case:
                    continue
                path      = os.path.join(base_dir, case, "")
                path_save = path[:-1] + ".nii.gz"
                assemble_volumes(path, path_save, target_imgs=False, downsampled=True)

    summarize_test_metrics(save_dir)

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

    if _diag_hf_vol_gt:
        hf_vol_gt  = np.mean(_diag_hf_vol_gt)
        hf_vol_gen = np.mean(_diag_hf_vol_gen)
        ratio      = hf_vol_gen / (hf_vol_gt + 1e-12)
        print(
            f"\n[HF-image] GT={hf_vol_gt:.4f}  Gen={hf_vol_gen:.4f}  "
            f"ratio={ratio:.3f}  ({'OK' if ratio > 0.85 else 'LOW — model losing HF detail'})"
        )
        np.save(os.path.join(save_dir, "hf_vol_gt.npy"),  np.asarray(_diag_hf_vol_gt))
        np.save(os.path.join(save_dir, "hf_vol_gen.npy"), np.asarray(_diag_hf_vol_gen))


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

    train_set = NAVIGATOR_4D_Dataset_multitime(
        cfg.data_dir, nb_inputs=cfg.nb_inputs,
        sequence_list=folds[0], nb_pred=cfg.tp,
    )
    
    print("="*50)
    print("Liver training dataset")
    print("Number of samples:", len(train_set))
    print("Number of patients:", len(folds[0]))
    print("="*50)
    
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
    return train_loader, valid_loader


def _write_patient_split(folds: tuple, log_dir: str) -> None:
    train_patients = extract_patient_ids(folds[0])
    val_patients   = extract_patient_ids(folds[1])
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

    writer.add_scalar("train/ddpm_loss",        _val(out.ddpm_loss),        step)
    writer.add_scalar("train/dvf_recon_loss",   _val(out.dvf_recon_loss),   step)
    writer.add_scalar("train/vmorph_vol_loss",  _val(out.vmorph_vol_loss),  step)
    writer.add_scalar("train/vol_recon_loss",   _val(out.vol_recon_loss),   step)
    writer.add_scalar("train/nav_corr_loss",    _val(out.nav_corr_loss),    step)
    writer.add_scalar("train/nav_signal_loss",  _val(out.nav_signal_loss),  step)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"],          step)
    writer.add_scalar("train/total_loss", loss.item(),                      step)

    total_norm = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5
    writer.add_scalar("train/grad_norm", total_norm, step)

    # Per-module gradient norms — zero means no gradient (frozen or unused)
    for name in ("denoising_net", "cond_proj", "cond_net", "ref_net", "blocks", "iseq_enc", "ref_enc"):
        module = getattr(model, name, None)
        if module is not None:
            writer.add_scalar(f"grad/{name}", _module_grad_norm(module), step)


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

    VOL_SIZE = (32, 64, 64)
    vm = build_reg_model(cfg, VOL_SIZE, device)

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
    p.add_argument("--debug_shapes", action="store_true",
                   help="Print every UNet3D block's input/output tensor shape on the "
                        "first training batch, then stop printing.")
    p.add_argument("--test_tag", default=None,
                   help="Optional tag appended to the test output directory (e.g. 'ssg1.5'). "
                        "Results go to test_{tag}/ instead of test/, leaving existing results intact.")
    p.add_argument("--test_seed", type=int, default=None,
                   help="Random seed for test-time DDIM sampling. Omit to use natural RNG flow (matches pre-tracker baseline).")
    p.add_argument("--metrics_only", action="store_true",
                   help="Skip saving NIfTI volumes and tracking sequences; only save metric .npy files and summary_metrics.txt.")
    p.add_argument("--tracker_checkpoint", default=None,
                   help="Dir containing fold_N/model_best.pth. Enables tracker post-processing on top of the MM DVF.")
    p.add_argument("--tracker_model", default="att", choices=["att", "unet", "single", "lesionlocator"],
                   help="Tracker architecture variant (default: att).")
    p.add_argument("--landmark_target", type=str, default="L2", choices=["L1", "L2"],
                   help="Landmark used as tracker ROI center. L1=main.txt, L2=right.txt.")
    p.add_argument("--roi", nargs=3, type=int, default=[4, 8, 8], metavar=("D", "H", "W"),
                   help="Tracker ROI size in voxels (default: 4 8 8).")
    return p.parse_args()


if __name__ == "__main__":
    import torch.multiprocessing as _mp

    args = _parse_cli()
    cfg  = load_config(args.config)
    cfg.fold_nb_training    = args.fold_nb_training
    cfg.debug_shapes        = args.debug_shapes
    cfg.test_tag            = args.test_tag
    cfg.test_seed           = args.test_seed
    cfg.metrics_only        = args.metrics_only
    cfg.tracker_checkpoint  = args.tracker_checkpoint
    cfg.tracker_model       = args.tracker_model
    cfg.roi                 = args.roi
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

    train_folds, valid_folds, test_folds = make_train_val_test_folds()

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

            VOL_SIZE = (32, 64, 64)
            vm = build_reg_model(cfg, VOL_SIZE, device)

            stn       = SpatialTransformer(VOL_SIZE).to(device)
            criterion = build_criterion(cfg)

            for fi in range(fold_nb):
                random.seed(cfg.seed + fi)
                np.random.seed(cfg.seed + fi)
                torch.manual_seed(cfg.seed + fi)
                torch.cuda.manual_seed_all(cfg.seed + fi)
                # if fi == 0 or  fi==1:
                #     continue
                # else : 
                print(fi)
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

        # Optionally build tracker module (checkpoint loaded per-fold inside test())
        tracker_module = None
        patch_centers  = None
        if args.tracker_checkpoint:
            _pre_latent_dim = getattr(cfg, f"{cfg.diffusion_model}_pre_latent_dim",
                                      getattr(cfg, "pre_latent_dim", 16))
            _roi = args.roi
            if args.tracker_model == "att":
                tracker_module = Tracker_module_attentionRefiner(
                    horizon=cfg.tp, F_g=1, F_l=1, F_int=32,
                    gate_dim=_pre_latent_dim, roi=_roi,
                ).to(device)
            elif args.tracker_model == "unet":
                tracker_module = Tracker_module_unetRefiner(horizon=cfg.tp, roi=_roi).to(device)
            elif args.tracker_model == "lesionlocator":
                tracker_module = Tracker_module_LesionLocator(horizon=cfg.tp).to(device)
            else:
                tracker_module = Tracker_module_singleRefiner(horizon=cfg.tp, roi=_roi).to(device)
            patch_centers = _load_patch_centers(cfg.data_dir, _roi, args.landmark_target)
            print(f"[Tracker] Module built: {args.tracker_model}  ROI={_roi}  "
                  f"pre_latent_dim={_pre_latent_dim}")

        dir_name = cfg.checkpoint
        folds_to_use = test_folds
        for fi in range(fold_nb):
            if fi == 0 or fi ==1:
                continue
            else :
                fold = [p for p in folds_to_use[fi] if p != "CoMoDo26"]
                test(
                    cfg           = cfg,
                    fold          = fold,
                    fold_idx      = str(fi),
                    dir_name      = dir_name,
                    vm            = vm,
                    stn           = stn,
                    device        = device,
                    tracker       = tracker_module,
                    patch_centers = patch_centers,
                )