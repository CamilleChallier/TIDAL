"""
MambaMorph registration training on the liver respiratory navigator dataset.
Requires the ``mamba`` conda environment (mamba_ssm is a CUDA extension).
Saves the trained checkpoint; does NOT overwrite pretrained_models/VM.pth.

This does NOT touch pretrained_models/VM.pth (the canonical frozen VoxelMorph
checkpoint other liver stages load) — the trained checkpoint is saved to
pretrained_models/MM_liver.pth instead, so downstream VAE/TMNet/CLDM
training is unaffected until you explicitly choose to swap it in.
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, dynamic_ncols=True)

from mopred.models.mambamorph import MambaMorph
from mopred.models import SpatialTransformer
from mopred.data.splits import make_folds_3fold
from mopred.data.data_loaders.navigator_4d import NAVIGATOR_4D_Dataset_multitime
from mopred.data.loading import save_params_txt
from mopred.utils.early_stopping import EarlyStopping
from mopred.utils.io import cond_mkdir, custom_load, custom_save
from mopred.utils.dvf_metrics import motion_amplitude, hf_energy_ratio, hf_energy_ratio_vol, dvf_cosine_sim_stats, jacobian_det_volume
from mopred.utils.losses import ncc_as_loss, gradient_loss, laplacian_edge_loss
from mopred.utils.landmarks import load_landmarks, landmark_tracking_error
# utils.config has no heavy deps (no monai/VAE zoo) — safe to import in the `mamba` env
from mopred.utils.config import load_config, _apply_overrides

VOL_SIZE  = (32, 64, 64)   # liver pipeline volume size (NAVIGATOR_4D_Dataset_multitime downsamples to this internally)
VIS_EVERY = 250            # log images + grad-flow every N global steps

_DEFAULT_TEST_CFG = {
    "logging_dir":  "outputs/MambaMorph",
    "data_dir":     "/store/usagers/livazra/Nav_dataset/",
    "name":         "mambamorph_test_liver",
    "gpu_idx":      "0",
    "num_workers":  12,
    "seed":         123,
    "mambamorph_freq_aware": False,
    "mambamorph_patch_size": 2,
}

# Matched to train_VoxelMorph_ACDC.py's hardcoded train_list/valid_list, so MM and VM
# are trained/evaluated on identical patient splits. NOTE: VM's script has no explicit
# test_list — its test() function reads from a "Test" folder on disk. Confirm which
# patients are physically in that folder before finalizing TEST_LIST below.
VM_TRAIN_LIST = ["CoMoDo01b", "CoMoDo02", "CoMoDo03", "CoMoDo04", "CoMoDo06", "CoMoDo09", "CoMoDo11", "CoMoDo12",
                 "CoMoDo13", "CoMoDo15", "CoMoDo16", "CoMoDo17", "CoMoDo18", "CoMoDo19", "CoMoDo20", "CoMoDo21",
                 "CoMoDo22", "CoMoDo24", "CoMoDo25", "CoMoDo26", "CoMoDo27"]
VM_VALID_LIST = ["CoMoDo05", "CoMoDo08b", "CoMoDo10", "CoMoDo28"]
VM_TEST_LIST  = [...]  


# =============================================================================
# Visualisation helpers
# =============================================================================

def _norm_slice(t: torch.Tensor) -> torch.Tensor:
    """Normalise a 2-D tensor to [0, 1]."""
    mn, mx = t.min(), t.max()
    return (t - mn) / (mx - mn + 1e-8)


def _log_images(writer: SummaryWriter, tag: str,
                v_ref: torch.Tensor, warped: torch.Tensor,
                v_curr: torch.Tensor, step: int) -> None:
    """Log a 3-panel (reference | warped | ground-truth) mid-axial slice."""
    mid_d = VOL_SIZE[0] // 2
    ref_s  = _norm_slice(v_ref [0, 0, mid_d].detach().cpu())
    warp_s = _norm_slice(warped[0, 0, mid_d].detach().cpu())
    curr_s = _norm_slice(v_curr[0, 0, mid_d].detach().cpu())
    grid = torch.stack([ref_s, warp_s, curr_s], dim=0)
    writer.add_image(f"{tag}/ref_warped_gt", grid, step)


def plot_grad_flow(named_parameters) -> plt.Figure:
    """Bar chart of mean |gradient| per layer (Agg backend — no display needed)."""
    layers, ave_grads = [], []
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())

    fig, ax = plt.subplots(figsize=(max(6, len(layers) // 2), 4))
    ax.bar(range(len(ave_grads)), ave_grads, alpha=0.7)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=5)
    ax.set_xlabel("Layers")
    ax.set_ylabel("|mean gradient|")
    ax.set_title("Gradient flow")
    plt.tight_layout()
    return fig


# =============================================================================
# CLI
# =============================================================================

def _parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="Path to a YAML config. Required for --train_test train. "
                        "Optional for --train_test test (falls back to _DEFAULT_TEST_CFG, "
                        "overridable via --override).")
    p.add_argument("--train_test", default="train", choices=["train", "test"],
                   help="'train' trains from scratch (optionally resuming from "
                        "--checkpoint); 'test' evaluates --checkpoint on the held-out "
                        "test fold.")
    p.add_argument("--checkpoint", default=None,
                   help="[train] Resume from this MM checkpoint. [test] Checkpoint to evaluate (required).")
    p.add_argument("--n_qualitative", type=int, default=8,
                   help="[test] Number of test samples to save qualitative plots for")
    p.add_argument("--override", nargs="*", default=[], help="KEY=VALUE config overrides")
    return p.parse_args()


def _seed_worker(_worker_id):
    import random as _random
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    _random.seed(seed)

def _load_landmarks(data_dir: str, patient_id: str) -> dict:
    """
    Load GT landmark positions from GT_landmarks/{patient_id}/*/main.txt.

    Returns
    -------
    dict  {lm_name: {t_idx: np.array([x, y, z])}}
    x, y, z are in original (pre-downsampled) voxel coordinates.
    Returns empty dict when no data is found.
    """
    landmark_dir = os.path.join(
        data_dir, "Complemetary_information", "GT_landmarks", patient_id
    )
    if not os.path.isdir(landmark_dir):
        return {}
    result = {}
    for lm_name in sorted(os.listdir(landmark_dir)):
        txt = os.path.join(landmark_dir, lm_name, "main.txt")
        if not os.path.isfile(txt):
            continue
        positions = {}
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z, t = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    positions[t] = np.array([x, y, z], dtype=float)
        if positions:
            result[lm_name] = positions
    return result


from mopred.utils.landmarks import (
      landmark_dvf_error as _landmark_dvf_error,
      landmark_tracking_error as _landmark_tracking_error,
  )

# =============================================================================
# Training loop
# =============================================================================

def train_mambamorph(cfg, folds, dir_name, mm, stn, device):
    log_dir = os.path.join(cfg.logging_dir, "logs", dir_name, "mambamorph")
    run_dir = os.path.join(cfg.logging_dir, "runs", dir_name, "mambamorph")
    for d in (log_dir, run_dir):
        cond_mkdir(d)
    save_params_txt(cfg, log_dir)
    writer = SummaryWriter(run_dir)

    # nb_inputs=1, nb_pred=1: each item is (exhale reference, single other frame) —
    # current_volume_list[0] cycles through every frame of the sequence.
    train_set = NAVIGATOR_4D_Dataset_multitime(cfg.data_dir, nb_inputs=1, nb_pred=1, sequence_list=folds[0],
                                               augment=True)
    valid_set = NAVIGATOR_4D_Dataset_multitime(cfg.data_dir, nb_inputs=1, nb_pred=1, sequence_list=folds[1], valid=True)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker, generator=g)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, worker_init_fn=_seed_worker)

    optimizer = torch.optim.Adam(mm.parameters(), lr=float(cfg.lr),
                                 weight_decay=float(getattr(cfg, "weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.scheduler_patience, min_lr=1e-7,
    )
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True,
                                  delta=cfg.early_stopping_delta)

    best_val  = float("inf")
    ckpt_path = os.path.join(log_dir, "model_best_mm.pth")

    restart_epoch = getattr(cfg, "restart_epoch", 0)
    global_step   = restart_epoch * len(train_loader)

    for epoch in range(restart_epoch, cfg.epochs):
        print(f"\n[MM] Epoch {epoch}/{cfg.epochs - 1}")
        t0 = time.time()
        mm.train()

        lam_edge = float(getattr(cfg, "lam_edge", 0.1))

        ep_loss = ep_sim = ep_smooth = ep_edge = ep_steps = 0
        for batch in tqdm(train_loader):
            ref_vol, _input_vols, current_vols, _vol_files = batch
            v_ref  = ref_vol.unsqueeze(1).to(device)
            v_curr = current_vols[0].unsqueeze(1).to(device)

            optimizer.zero_grad()
            dvf    = mm(v_ref, v_curr)
            warped = stn(v_ref, dvf)

            if epoch == restart_epoch and ep_steps == 0:
                print(f"[MM][shapes] v_ref={tuple(v_ref.shape)}  v_curr={tuple(v_curr.shape)}  "
                      f"dvf={tuple(dvf.shape)}  warped={tuple(warped.shape)}")

            sim    = ncc_as_loss(warped, v_curr, device)
            smooth = gradient_loss(dvf)
            edge   = laplacian_edge_loss(warped, v_curr)
            loss   = sim + cfg.lam_smooth * smooth + lam_edge * edge

            loss.backward()
            optimizer.step()

            ep_loss   += loss.item()
            ep_sim    += sim.item()
            ep_smooth += smooth.item()
            ep_edge   += edge.item()
            ep_steps  += 1

            writer.add_scalar("mm_train/loss",   loss.item(),   global_step)
            writer.add_scalar("mm_train/sim",    sim.item(),    global_step)
            writer.add_scalar("mm_train/smooth", smooth.item(), global_step)
            writer.add_scalar("mm_train/edge",   edge.item(),   global_step)

            if global_step % VIS_EVERY == 0:
                _log_images(writer, "mm_train", v_ref, warped, v_curr, global_step)
                fig = plot_grad_flow(mm.named_parameters())
                writer.add_figure("mm_train/grad_flow", fig, global_step)
                plt.close(fig)

            global_step += 1

        writer.add_scalar("mm_train/epoch_loss",   ep_loss   / ep_steps, epoch)
        writer.add_scalar("mm_train/epoch_sim",    ep_sim    / ep_steps, epoch)
        writer.add_scalar("mm_train/epoch_smooth", ep_smooth / ep_steps, epoch)
        writer.add_scalar("mm_train/epoch_edge",   ep_edge   / ep_steps, epoch)

        # ---- validation ------------------------------------------------------
        mm.eval()
        val_loss = val_sim = val_smooth = val_edge = val_n = 0
        _val_vis = None
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                ref_vol, _input_vols, current_vols, _vol_files = batch
                v_ref  = ref_vol.unsqueeze(1).to(device)
                v_curr = current_vols[0].unsqueeze(1).to(device)

                dvf    = mm(v_ref, v_curr)
                warped = stn(v_ref, dvf)

                sim    = ncc_as_loss(warped, v_curr, device)
                smooth = gradient_loss(dvf)
                edge   = laplacian_edge_loss(warped, v_curr)
                loss   = sim + cfg.lam_smooth * smooth + lam_edge * edge

                val_loss   += loss.item()
                val_sim    += sim.item()
                val_smooth += smooth.item()
                val_edge   += edge.item()
                if _val_vis is None:
                    _val_vis = (v_ref, warped, v_curr)
                val_n      += 1

        val_loss   /= max(val_n, 1)
        val_sim    /= max(val_n, 1)
        val_smooth /= max(val_n, 1)
        val_edge   /= max(val_n, 1)

        writer.add_scalar("mm_valid/loss",   val_loss,   epoch)
        writer.add_scalar("mm_valid/sim",    val_sim,    epoch)
        writer.add_scalar("mm_valid/smooth", val_smooth, epoch)
        writer.add_scalar("mm_valid/edge",   val_edge,   epoch)
        if _val_vis is not None:
            _log_images(writer, "mm_valid", *_val_vis, epoch)

        print(f"[MM] Epoch {epoch}  train_loss={ep_loss / ep_steps:.4f}  "
              f"val_loss={val_loss:.4f} (sim={val_sim:.4f}, smooth={val_smooth:.4f}, edge={val_edge:.4f})  "
              f"time={time.time() - t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            custom_save(mm, ckpt_path)
            print(f"[MM] New best (val_loss={best_val:.4f}) -> saved to {ckpt_path}")

        scheduler.step(val_loss)
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"[MM] Early stopping at epoch {epoch}")
            break

    writer.close()
    return ckpt_path


# =============================================================================
# Test / evaluation
# =============================================================================

def _save_qualitative(v_ref, warped, v_curr, out_path, title):
    mid_d = VOL_SIZE[0] // 2
    ref_s  = _norm_slice(v_ref [0, 0, mid_d].detach().cpu()).numpy()
    warp_s = _norm_slice(warped[0, 0, mid_d].detach().cpu()).numpy()
    curr_s = _norm_slice(v_curr[0, 0, mid_d].detach().cpu()).numpy()
    diff_s = warp_s - curr_s

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        (ref_s,  "Reference (exhale)", "gray", 0, 1),
        (warp_s, "Warped",             "gray", 0, 1),
        (curr_s, "Target (GT)",        "gray", 0, 1),
        (diff_s, "Warped - GT",        "bwr",  -1, 1),
    ]
    for ax, (img, name, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def test_mambamorph(cfg, test_seqs, mm, stn, device, out_dir, n_qualitative):
    """Evaluate `mm` on the held-out test fold and save metrics + qualitative panels."""
    test_set = NAVIGATOR_4D_Dataset_multitime(cfg.data_dir, nb_inputs=1, nb_pred=1,
                                              sequence_list=test_seqs, valid=True, test=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    mm.eval()

    sims, smooths, amps = [], [], []
    jac_means, jac_stds, neg_jac_fracs = [], [], []
    hf_dvf, hf_vol, mses = [], [], []
    dvf_vecs, patient_ids = [], []

    # ---- preload GT landmarks (one dict per patient) -----------------------
    test_patients = {
        seq.split("/")[0] if "/" in seq else seq[:8] for seq in test_seqs
    }
    all_landmarks: dict = {
        pid: _load_landmarks(cfg.data_dir, pid)
        for pid in test_patients
    }
    n_with_lm = sum(1 for v in all_landmarks.values() if v)
    print(f"[Landmark] GT landmarks found for {n_with_lm}/{len(test_patients)} test patients.")
    landmark_error_dvf = []

    qual_dir = os.path.join(out_dir, "qualitative")
    cond_mkdir(qual_dir)

    for i, batch in enumerate(tqdm(test_loader)):
        ref_vol, _input_vols, current_vols, vol_files = batch
        v_ref  = ref_vol.unsqueeze(1).to(device)
        v_curr = current_vols[0].unsqueeze(1).to(device)

        dvf    = mm(v_ref, v_curr)
        warped = stn(v_ref, dvf)

        sim    = ncc_as_loss(warped, v_curr, device)
        smooth = gradient_loss(dvf)
        amp    = motion_amplitude(dvf)

        det = jacobian_det_volume(dvf)
        jac_means.append(det.mean().item())
        jac_stds.append(det.std().item())
        neg_jac_fracs.append((det <= 0).float().mean().item())

        sims.append(sim.item())
        smooths.append(smooth.item())
        amps.append(amp)

        hf_dvf.append(hf_energy_ratio(dvf))
        hf_vol.append(hf_energy_ratio_vol(warped))
        mses.append(torch.nn.functional.mse_loss(warped, v_curr).item())

        patient_id = str(vol_files[0][0]).split("/")[-2]
        dvf_vecs.append(dvf.flatten().cpu().unsqueeze(0))
        patient_ids.append(patient_id)

        # ---- landmark tracking error for this sample's DVF ------------------
        v_file_i     = str(vol_files[0][0])
        volume_idx_i = int(v_file_i.split("/")[-1][2:-7])
        lm_kw = dict(
            landmarks  = all_landmarks.get(patient_id, {}),
            volume_idx = volume_idx_i,
            patient_id = patient_id,
        )
        landmark_error_dvf.append(_landmark_tracking_error(dvf, **lm_kw))

        if i < n_qualitative:
            tag = os.path.splitext(str(vol_files[0]))[0].replace("/", "_")
            _save_qualitative(v_ref, warped, v_curr,
                               os.path.join(qual_dir, f"{tag}.png"), tag)

    cosine_stats = dvf_cosine_sim_stats(dvf_vecs, patient_ids)

    lm_arr = np.asarray(landmark_error_dvf, dtype=float)
    np.save(os.path.join(out_dir, "landmark_error_dvf.npy"), lm_arr)

    summary = {
        "ncc_sim (lower=better)":    (np.mean(sims),    np.std(sims)),
        "mse (warped vs fixed)":     (np.mean(mses),    np.std(mses)),
        "smoothness":                (np.mean(smooths), np.std(smooths)),
        "hf_energy_ratio (dvf)":     (np.mean(hf_dvf),  np.std(hf_dvf)),
        "hf_energy_ratio (vol)":     (np.mean(hf_vol),  np.std(hf_vol)),
        "motion_amplitude":          (np.mean(amps),    np.std(amps)),
        "jacobian_det_mean":         (np.mean(jac_means), np.std(jac_means)),
        "jacobian_det_std":          (np.mean(jac_stds),  np.std(jac_stds)),
        "neg_jacobian_frac (folds)": (np.mean(neg_jac_fracs), np.std(neg_jac_fracs)),
        "landmark_error_dvf (mm)":   (np.nanmean(lm_arr), np.nanstd(lm_arr)),
    }
    
    lines = [f"MambaMorph liver test — {len(test_set)} samples\n"]
    for name, (m, s) in summary.items():
        lines.append(f"  {name:28s} = {m:.6f} +/- {s:.6f}")
    intra, inter = cosine_stats
    lines.append("")
    lines.append("DVF cosine similarity — intra vs inter-patient:")
    if intra:
        lines.append(f"  intra-patient: mean={np.mean(intra):.4f}  std={np.std(intra):.4f}  n={len(intra)}")
    if inter:
        lines.append(f"  inter-patient: mean={np.mean(inter):.4f}  std={np.std(inter):.4f}  n={len(inter)}")
    if intra and inter:
        lines.append(f"  intra−inter gap: {np.mean(intra) - np.mean(inter):+.4f}")
    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(out_dir, "test_metrics.txt"), "w") as f:
        f.write(report + "\n")

    return summary


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    args = _parse_cli()

    if args.config is not None:
        cfg = load_config(args.config)
    else:
        if args.train_test != "test":
            raise ValueError("--config is required for --train_test train")
        print("[MM] No --config given — using _DEFAULT_TEST_CFG (override with --override)")
        cfg = argparse.Namespace(**_DEFAULT_TEST_CFG)
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    _apply_overrides(cfg, args.override)

    print("\n=== Config ===")
    print("\n".join(f"  {k}: {v}" for k, v in sorted(vars(cfg).items())))
    print("==============\n")

    device     =  torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # MambaMorph architecture — hardcoded here, mirrors how Voxelmorph's channel
    # lists are hardcoded in train_VoxelMorph_ACDC.py rather than config-driven.
    # use_freq_aware adds the ALPF/AHPF blocks from Shi et al. (frequency-aware
    # mamba + diffusion paper); config-gated since it changes the architecture
    # and won't load older checkpoints trained without it.
    embed_dim = int(getattr(cfg,"mambamorph_embed_dim", 96))
    mm_kwargs = dict(embed_dim=embed_dim, depths=(2, 2, 4), reg_head_chan=16,
                     d_state=16, d_conv=4, expand=2,
                     patch_size=int(getattr(cfg, "mambamorph_patch_size", 4)),
                     use_freq_aware=getattr(cfg, "mambamorph_freq_aware", False))

    if args.train_test == "test":
        if not getattr(cfg, "checkpoint", None):
            raise ValueError("--checkpoint is required for --train_test test")

        mm  = MambaMorph(VOL_SIZE, **mm_kwargs).to(device)
        stn = SpatialTransformer(VOL_SIZE).to(device)

        # Train/valid matched to VoxelMorph's split; test fold unchanged (still make_folds_3fold).
        _, _, test_folds = make_folds_3fold(n_splits=1)   # kept only to confirm/report test fold if needed
        train_folds = [VM_TRAIN_LIST]
        valid_folds = [VM_VALID_LIST]

        print(f"[MM] Loading checkpoint: {cfg.checkpoint}")
        custom_load(mm, cfg.checkpoint, device)

        out_dir = os.path.join(os.path.dirname(cfg.checkpoint), "test")
        cond_mkdir(out_dir)
        test_mambamorph(cfg, test_folds[0], mm, stn, device, out_dir, args.n_qualitative)

    else:
        mm  = MambaMorph(VOL_SIZE, **mm_kwargs).to(device)
        stn = SpatialTransformer(VOL_SIZE).to(device)

        # A single fold suffices: like VoxelMorph, this is a generic registration
        # network trained once and then frozen for downstream stages.
        _, _, test_folds = make_folds_3fold(n_splits=1)   # kept only to confirm/report test fold if needed
        train_folds = [VM_TRAIN_LIST]
        valid_folds = [VM_VALID_LIST]

        if getattr(cfg, "checkpoint", None):
            print(f"[MM] Resuming from checkpoint: {cfg.checkpoint}")
            custom_load(mm, cfg.checkpoint, device)

        dir_name = os.path.join(
            datetime.datetime.now().strftime("%m_%d"),
            datetime.datetime.now().strftime("%H.%M._") + cfg.name,
        )

        ckpt_path = train_mambamorph(
            cfg=cfg, folds=(train_folds[0], valid_folds[0]),
            dir_name=dir_name, mm=mm, stn=stn, device=device,
        )

        # Separate from pretrained_models/VM.pth on purpose — downstream
        # VAE/TMNet/CLDM stages keep loading the VoxelMorph checkpoint
        # until you explicitly decide to swap MambaMorph in.
        out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pretrained_models", "MM_liver.pth")
        cond_mkdir(os.path.dirname(out_path))
        shutil.copy2(ckpt_path, out_path)
        print(f"[MM] Best checkpoint copied to {out_path}")
