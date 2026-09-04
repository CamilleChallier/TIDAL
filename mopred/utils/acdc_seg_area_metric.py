"""
Area-based segmentation evaluation for the ACDC test set.
Warps ED GT segmentations with predicted DVFs (optionally refining RV/LV with MobileSAM),
then reports Mean Absolute Difference and relative area error (%) for RV and LV labels.

Usage
-----
    python -m mopred.utils.acdc_seg_area_metric \
        [--data_dir /volatile/data/cam/ACDC/database] \
        [--out path/to/results.csv] \
        [--img_dir path/to/seg_images/] \
        [--n_img_patients 5]
"""

import argparse
import glob
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import (
    binary_closing, binary_fill_holes, generate_binary_structure, zoom,
)

from mopred.data.data_loaders.acdc_4d import _patient_info, load_orientation_info

# acdc_propagate_seg lives in visualisations/, outside the mopred editable package
import importlib.util as _ilu, pathlib as _pl
_PROP_SEG = str(_pl.Path(__file__).parents[2]
                / "visualisations/scripts/ACDC/acdc_propagate_seg.py")
_spec = _ilu.spec_from_file_location("acdc_propagate_seg", _PROP_SEG)
_mod  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
preprocess_seg    = _mod.preprocess_seg
warp_seg_with_dvf = _mod.warp_seg_with_dvf

# ── Config ────────────────────────────────────────────────────────────────────
BASE     = "/usagers4/p123507/MoPred/outputs/CLDM_ACDC/unet3d/logs"
TEST_DIR = f"{BASE}/08_16/11.25._overfit_acdc_diversityloss_128pfull/test/0"
DATA_DIR = "/volatile/data/cam/ACDC/database"
ORIENTATION_CSV = "/store/usagers/p123507/MoPred/reference_slices.csv"

MOBILE_SAM_PKG  = "/volatile/p123507/python_packages"
MOBILE_SAM_CKPT = "/volatile/p123507/checkpoints/mobile_sam.pt"

METHODS = {
    "MambaMorph":  {"test_dir": TEST_DIR, "dvf_name": "DVF_t0"},
    "MoPred-Diff": {"test_dir": TEST_DIR, "dvf_name": "generated_DVF_t0"},
}

EVAL_LABELS = {1: "RV", 3: "LV"}

# Outlier thresholds — predictions that exceed EITHER are excluded.
# 1. Relative: pred > MAX_AREA_RATIO × GT (catches inflated masks when GT > 0).
# 2. Absolute: pred > MAX_ABS_FRAC of the whole slice (catches full-image SAM
#    failures regardless of GT, e.g. when GT area is 0 or very small).
MAX_AREA_RATIO = 3.0
MAX_ABS_FRAC   = 0.20   # >20 % of slice area is certainly wrong for a cardiac cavity

SEG_COLORS = {
    1: np.array([0.31, 0.67, 0.82]),   # RV  — steel blue
    2: np.array([0.96, 0.64, 0.38]),   # Myo — warm orange
    3: np.array([0.90, 0.22, 0.27]),   # LV  — red
}
SEG_ALPHA = 0.55

# ── Seg helpers (mirrors dvf_magnitude_comparison.py) ─────────────────────────

def _resize_seg_nn(seg2d, target_hw):
    if tuple(seg2d.shape) == tuple(target_hw):
        return seg2d
    zy, zx = target_hw[0] / seg2d.shape[0], target_hw[1] / seg2d.shape[1]
    return zoom(seg2d, (zy, zx), order=0)


def _resize_seg_bilinear(seg2d, target_hw):
    if tuple(seg2d.shape) == tuple(target_hw):
        return seg2d
    labels = np.unique(seg2d[seg2d > 0])
    result = np.zeros(target_hw, dtype=seg2d.dtype)
    best   = np.zeros(target_hw, dtype=np.float32)
    zy, zx = target_hw[0] / seg2d.shape[0], target_hw[1] / seg2d.shape[1]
    for lbl in labels:
        resized = zoom((seg2d == lbl).astype(np.float32), (zy, zx), order=1)
        mask = resized > best
        result[mask] = lbl
        best[mask]   = resized[mask]
    return result


def _cleanup_seg(seg2d, closing_iter=1):
    struct = generate_binary_structure(2, 2)
    result = np.zeros_like(seg2d)
    for lbl in np.unique(seg2d[seg2d > 0]):
        m = binary_closing(seg2d == lbl, structure=struct, iterations=closing_iter)
        m = binary_fill_holes(m)
        result[m] = lbl
    return result


_sam_predictor = None


def _get_sam_predictor(device="cuda"):
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    sys.path.insert(0, MOBILE_SAM_PKG)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    from mobile_sam import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_t"](checkpoint=None)
    state = torch.load(MOBILE_SAM_CKPT, map_location="cpu", weights_only=False)
    sam.load_state_dict(state)
    for dev in ([torch.device("cuda")] if torch.cuda.is_available() else []) + [torch.device("cpu")]:
        try:
            sam.to(dev).eval()
            break
        except RuntimeError:
            print(f"[SAM] failed to load on {dev}, retrying on cpu")
    _sam_predictor = SamPredictor(sam)
    print(f"[SAM] MobileSAM loaded on {dev}")
    return _sam_predictor


def _refine_seg_sam(anat, seg_for_prompts, seg_myo_nn):
    if anat is None or not np.any(seg_for_prompts > 0):
        return seg_for_prompts
    try:
        predictor = _get_sam_predictor()
    except Exception as e:
        print(f"[SAM] load failed ({e}), falling back")
        return seg_for_prompts

    # Resize anatomy to match seg resolution so SAM masks align with result
    if anat.shape != seg_for_prompts.shape:
        zy = seg_for_prompts.shape[0] / anat.shape[0]
        zx = seg_for_prompts.shape[1] / anat.shape[1]
        anat = zoom(anat, (zy, zx), order=1)

    lo, hi  = np.percentile(anat, 1), np.percentile(anat, 99)
    gray    = np.clip((anat - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    img_rgb = (np.stack([gray] * 3, axis=-1) * 255).astype(np.uint8)
    predictor.set_image(img_rgb)

    centroids = {}
    for lbl in [1, 2, 3]:
        region = seg_for_prompts == lbl
        if region.any():
            ys, xs = np.where(region)
            centroids[lbl] = (int(xs.mean()), int(ys.mean()))

    raw_masks = {}
    for lbl in [1, 3]:
        if lbl not in centroids:
            continue
        fg_pt      = np.array([centroids[lbl]])
        bg_pts     = np.array([c for k, c in centroids.items() if k != lbl])
        points     = np.concatenate([fg_pt, bg_pts]) if len(bg_pts) else fg_pt
        point_lbls = np.array([1] + [0] * len(bg_pts))
        masks, scores, _ = predictor.predict(
            point_coords=points, point_labels=point_lbls, multimask_output=True)
        raw_masks[lbl] = masks[scores.argmax()]

    empty    = np.zeros(seg_for_prompts.shape, dtype=bool)
    rv_mask  = raw_masks.get(1, empty)
    lv_mask  = raw_masks.get(3, empty)
    myo_mask = (seg_myo_nn == 2) & ~lv_mask

    result = np.zeros_like(seg_for_prompts)
    result[rv_mask]  = 1
    result[myo_mask] = 2
    result[lv_mask]  = 3
    return result.astype(np.int32)


def _refine_volume(pred_seg_vol, anat_vol, slices=None):
    """Refine with SAM only on the requested slices (default: all)."""
    refined = pred_seg_vol.copy()
    indices = slices if slices is not None else range(pred_seg_vol.shape[0])
    for sl in indices:
        seg_slice  = pred_seg_vol[sl]
        anat_slice = anat_vol[sl] if anat_vol is not None else None
        seg_nn     = _resize_seg_nn(seg_slice, seg_slice.shape)
        seg_clean  = _cleanup_seg(_resize_seg_bilinear(seg_slice, seg_slice.shape))
        refined[sl] = _refine_seg_sam(anat_slice, seg_clean, seg_nn)
    return refined


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _seg_overlay(anat_slice, seg_slice):
    """Return H×W×3 float32: grayscale anatomy with coloured seg overlay."""
    if anat_slice.shape != seg_slice.shape:
        zy = seg_slice.shape[0] / anat_slice.shape[0]
        zx = seg_slice.shape[1] / anat_slice.shape[1]
        anat_slice = zoom(anat_slice, (zy, zx), order=1)
    lo, hi = np.percentile(anat_slice, 1), np.percentile(anat_slice, 99)
    gray   = np.clip((anat_slice - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    rgb    = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for lbl, color in SEG_COLORS.items():
        mask = seg_slice == lbl
        if mask.any():
            rgb[mask] = (1 - SEG_ALPHA) * rgb[mask] + SEG_ALPHA * color
    return rgb


def _pick_slices(seg_vol, n=5):
    """Return up to n slice indices that contain at least one labelled voxel,
    evenly spaced across the labelled range."""
    labelled = [sl for sl in range(seg_vol.shape[0])
                if np.any(seg_vol[sl] > 0)]
    if not labelled:
        return []
    if len(labelled) <= n:
        return labelled
    idx = np.round(np.linspace(0, len(labelled) - 1, n)).astype(int)
    return [labelled[i] for i in idx]


def _save_seg_figure(patient, anat_vol, es_seg_gt, pred_vols, img_dir,
                     es_phase, slices=None):
    """
    Save a grid figure:
      rows = selected slices
      cols = GT | MambaMorph | MoPred-Diff  (anatomy + seg overlay)
    """
    if slices is None:
        slices = _pick_slices(es_seg_gt, n=5)
    if not slices:
        print(f"  [fig] no labelled slices for {patient}, skipping")
        return

    method_names = list(pred_vols.keys())
    n_cols = 1 + len(method_names)   # GT + one per method
    n_rows = len(slices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2),
                             gridspec_kw=dict(hspace=0.04, wspace=0.04),
                             facecolor="white")
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    col_titles = ["GT ES"] + method_names
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9, color="#222222", pad=3)

    for r, sl in enumerate(slices):
        anat = anat_vol[sl] if anat_vol is not None else np.zeros((128, 128))

        # GT column
        axes[r, 0].imshow(_seg_overlay(anat, es_seg_gt[sl]), interpolation="bilinear")
        axes[r, 0].set_ylabel(f"sl {sl}", fontsize=7, color="#555555", labelpad=2)

        # Method columns
        for c, mname in enumerate(method_names, start=1):
            pred_vol = pred_vols[mname]
            axes[r, c].imshow(_seg_overlay(anat, pred_vol[sl]), interpolation="bilinear")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=SEG_COLORS[l],
                              label={1: "RV", 2: "Myo", 3: "LV"}[l])
               for l in sorted(SEG_COLORS)]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"{patient}  |  ES φ={es_phase:.3f}", fontsize=10,
                 color="#111111", y=1.01)

    os.makedirs(img_dir, exist_ok=True)
    out = os.path.join(img_dir, f"{patient}_es_seg.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [fig] saved → {out}")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_seg_vol(data_dir, patient, frame, orientation=None):
    path   = os.path.join(data_dir, patient, f"{patient}_frame{frame:02d}_gt.nii.gz")
    native = np.asarray(nib.load(path).dataobj, dtype=np.int32).transpose(2, 0, 1)
    info   = _patient_info(patient, data_dir)
    return preprocess_seg(native, info["spacing"],
                          orientation=orientation, patient=patient).numpy()


def _load_dvf(vol_dir, dvf_name, it, target_shape=None):
    d      = os.path.join(vol_dir, dvf_name)
    suffix = f"{it:5d}.nii"
    path   = next((f for f in glob.glob(os.path.join(d, "*.nii"))
                   if f.endswith(suffix)), None)
    if path is None:
        return None
    arr = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    dvf = torch.from_numpy(arr.transpose(3, 0, 1, 2))
    if target_shape and tuple(dvf.shape[1:]) != tuple(target_shape):
        scale = [t / s for t, s in zip(target_shape, dvf.shape[1:])]
        dvf = F.interpolate(dvf.unsqueeze(0), size=target_shape,
                            mode="trilinear", align_corners=False).squeeze(0)
        for c, s in enumerate(scale):
            dvf[c] *= s
    return dvf


def _load_anat_vol(save_dir, it):
    d      = os.path.join(save_dir, "volumes", "vm_volume_t0")
    suffix = f"{it:5d}.nii"
    path   = next((f for f in glob.glob(os.path.join(d, "*.nii"))
                   if f.endswith(suffix)), None)
    if path is None:
        return None
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


# ── Core evaluation (reusable from train scripts) ─────────────────────────────

SLICE_IDX = 8   # always evaluate on slice 8

def compute_for_test_dir(
    save_dir:    str,
    data_dir:    str,
    methods:     dict,           # {display_name: dvf_subfolder_name}
    orientation=None,
    img_dir:     str | None = None,
    n_img_patients: int = 0,
) -> list[dict]:
    """
    Compute RV/LV area metric (slice 8) for a completed test run.

    Parameters
    ----------
    save_dir    : directory that contains patient_ids.npy + cycle_phases.npy
    data_dir    : ACDC database root
    methods     : {display_name: dvf_subfolder} e.g.
                  {"MambaMorph": "DVF_t0", "MoPred-Diff": "generated_DVF_t0"}
    orientation : output of load_orientation_info(), or None
    img_dir     : if set, save per-patient seg figures here
    n_img_patients : how many patients to visualise (0 = none)

    Returns list of per-patient-method row dicts; also writes
    <save_dir>/seg_area_metric_sl8.csv.
    """
    vol_dir = os.path.join(save_dir, "volumes")

    all_pids   = np.load(os.path.join(save_dir, "patient_ids.npy"),
                         allow_pickle=True).flatten()
    all_phases = np.load(os.path.join(save_dir, "cycle_phases.npy")).flatten()
    all_idx    = np.arange(len(all_pids))

    test_patients = sorted(set(str(p) for p in all_pids))
    rows = []

    for i_pat, patient in enumerate(test_patients):
        print(f"\n── {patient} ──")
        info  = _patient_info(patient, data_dir)
        ed_fr, es_fr = info["ED"], info["ES"]

        ed_seg = _load_seg_vol(data_dir, patient, ed_fr, orientation)
        es_seg = _load_seg_vol(data_dir, patient, es_fr, orientation)

        gt_areas = {lbl: int(np.sum(es_seg[[SLICE_IDX]] == lbl)) for lbl in EVAL_LABELS}
        print(f"  GT ES areas (sl{SLICE_IDX}): "
              f"{ {EVAL_LABELS[k]: v for k, v in gt_areas.items()} }")

        pat_mask   = np.array([str(p) == patient for p in all_pids])
        pat_idx    = all_idx[pat_mask]
        pat_phases = all_phases[pat_mask]
        es_local   = int(np.argmin(np.abs(pat_phases - 0.5)))
        es_iter    = int(pat_idx[es_local])
        es_phase   = float(pat_phases[es_local])
        print(f"  ES phase: closest=φ{es_phase:.3f} (iter {es_iter})")

        anat_vol  = _load_anat_vol(save_dir, es_iter)
        pred_vols = {}

        for method_name, dvf_name in methods.items():
            dvf = _load_dvf(vol_dir, dvf_name, es_iter,
                            target_shape=tuple(ed_seg.shape))
            if dvf is None:
                print(f"  [{method_name}] DVF '{dvf_name}' not found — skipping")
                continue

            pred_vol = warp_seg_with_dvf(ed_seg, dvf, torch.device("cpu"))
            pred_vol = pred_vol.numpy() if hasattr(pred_vol, "numpy") else np.asarray(pred_vol)
            # Only run SAM on slice 8 — avoids processing every slice in the volume
            pred_vol = _refine_volume(pred_vol, anat_vol, slices=[SLICE_IDX])
            pred_vols[method_name] = pred_vol

            pred_vol_sl  = pred_vol[[SLICE_IDX]]
            slice_area   = pred_vol_sl.shape[1] * pred_vol_sl.shape[2]
            abs_cap      = int(MAX_ABS_FRAC * slice_area)

            row = {"patient": patient, "method": method_name,
                   "es_phase": es_phase, "slice": SLICE_IDX}
            for lbl, name in EVAL_LABELS.items():
                gt_a    = gt_areas[lbl]
                pred_a  = int(np.sum(pred_vol_sl == lbl))
                diff    = abs(pred_a - gt_a)
                rel     = diff / max(gt_a, 1)
                outlier_rel  = (gt_a > 0) and (pred_a > MAX_AREA_RATIO * gt_a)
                outlier_abs  = pred_a > abs_cap
                outlier_zero = (gt_a == 0)   # rel undefined; exclude from summary
                outlier      = outlier_rel or outlier_abs or outlier_zero
                if outlier:
                    if outlier_zero:
                        reason = f"GT=0 (rel undefined)"
                    elif outlier_rel:
                        reason = f">{MAX_AREA_RATIO}×GT({gt_a})"
                    else:
                        reason = f">{MAX_ABS_FRAC:.0%} of slice({slice_area}px)"
                    print(f"  [{method_name}] {name}: pred={pred_a} {reason} — outlier")
                row.update({f"{name}_gt": gt_a, f"{name}_pred": pred_a,
                             f"{name}_diff": diff, f"{name}_rel": rel,
                             f"{name}_outlier": outlier})
            rows.append(row)
            print(f"  [{method_name}] pred sl{SLICE_IDX}: "
                  f"{ {EVAL_LABELS[k]: int(np.sum(pred_vol_sl==k)) for k in EVAL_LABELS} }")

        if img_dir and i_pat < n_img_patients and pred_vols:
            _save_seg_figure(patient, anat_vol, es_seg, pred_vols,
                             img_dir, es_phase, slices=[SLICE_IDX])

    if not rows:
        print("[seg_area_metric] No data collected.")
        return rows

    # ── Summary ───────────────────────────────────────────────────────────────
    method_names = list(dict.fromkeys(r["method"] for r in rows))
    col_w = 26
    print("\n" + "=" * (16 + len(EVAL_LABELS) * col_w))
    print(f"{'Method':<16}", end="")
    for name in EVAL_LABELS.values():
        print(f"  {(name + ' MAD(vox)'):<12}  {(name + ' Rel(%)'):<10}", end="")
    print()
    print("-" * (16 + len(EVAL_LABELS) * col_w))
    for method_name in method_names:
        mrows = [r for r in rows if r["method"] == method_name]
        print(f"{method_name:<16}", end="")
        for name in EVAL_LABELS.values():
            valid  = [r for r in mrows if not r.get(f"{name}_outlier", False)]
            n_out  = len(mrows) - len(valid)
            diffs  = [r[f"{name}_diff"] for r in valid]
            rels   = [r[f"{name}_rel"]  for r in valid]
            suffix = f" (excl.{n_out})" if n_out else ""
            if diffs:
                print(f"  {np.mean(diffs):6.0f} ±{np.std(diffs):<5.0f}"
                      f"  {100*np.mean(rels):5.1f}±{100*np.std(rels):.1f}%{suffix}", end="")
            else:
                print(f"  {'all outliers':<22}", end="")
        print()

    # RelDiff summary: model vs MambaMorph
    mm_rows_d    = {r["patient"]: r for r in rows if r["method"] == "MambaMorph"}
    other_methods = [m for m in method_names if m != "MambaMorph"]
    if mm_rows_d and other_methods:
        print()
        print(f"  RelDiff vs MambaMorph ({other_methods[0]}):")
        for name in EVAL_LABELS.values():
            diffs = []
            for patient, mm_r in mm_rows_d.items():
                other_r = next((r for r in rows
                                if r["method"] == other_methods[0]
                                and r["patient"] == patient), None)
                if other_r is None:
                    continue
                if mm_r.get(f"{name}_outlier") or other_r.get(f"{name}_outlier"):
                    continue
                mm_pred = int(mm_r[f"{name}_pred"])
                if mm_pred == 0:
                    continue
                diffs.append(abs(int(other_r[f"{name}_pred"]) - mm_pred) / mm_pred)
            if diffs:
                print(f"    {name}: {100*np.mean(diffs):.1f} ± {100*np.std(diffs):.1f}%  (n={len(diffs)})")

    # ── CSV ───────────────────────────────────────────────────────────────────
    import csv
    out_path = os.path.join(save_dir, "seg_area_metric_sl8.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[seg_area_metric] Saved → {out_path}")

    # ── Append to summary_metrics.txt so aggregate_fold_metrics picks them up ─
    # Metric: |pred_model - pred_MambaMorph| / pred_MambaMorph  (SAM noise cancels)
    summary_path = os.path.join(save_dir, "summary_metrics.txt")
    mm_rows  = {r["patient"]: r for r in rows if r["method"] == "MambaMorph"}
    other_methods = [m for m in method_names if m != "MambaMorph"]
    if mm_rows and other_methods:
        other_name = other_methods[0]
        other_rows = {r["patient"]: r for r in rows if r["method"] == other_name}
        with open(summary_path, "a") as f:
            for name in EVAL_LABELS.values():
                diffs = []
                for patient, mm_r in mm_rows.items():
                    if patient not in other_rows:
                        continue
                    if mm_r.get(f"{name}_outlier") or other_rows[patient].get(f"{name}_outlier"):
                        continue
                    mm_pred    = int(mm_r[f"{name}_pred"])
                    other_pred = int(other_rows[patient][f"{name}_pred"])
                    if mm_pred == 0:   # reference undefined
                        continue
                    diffs.append(abs(other_pred - mm_pred) / mm_pred)
                if not diffs:
                    continue
                mean_pct = 100 * float(np.mean(diffs))
                std_pct  = 100 * float(np.std(diffs))
                f.write(f"Global Seg {name} RelDiff%: {mean_pct:.4f} ± {std_pct:.4f}\n")
        print(f"[seg_area_metric] Appended RV/LV RelDiff% vs MambaMorph to {summary_path}")

    return rows


# ── 3-D Dice metric (no SAM, full volume) ────────────────────────────────────

DICE_LABELS = {1: "RV", 2: "Myo", 3: "LV"}


def _dice(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    p = pred == label
    g = gt   == label
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0   # both empty → perfect agreement
    return float(2 * (p & g).sum() / denom)


def compute_dice_for_test_dir(
    save_dir:    str,
    data_dir:    str,
    methods:     dict,
    orientation=None,
) -> list[dict]:
    """
    3-D Dice score (RV / Myo / LV) for a completed ACDC test run.

    Warps the ED GT seg with each method's DVF at ES phase (no SAM),
    then computes Dice vs the GT ES seg over the full 3-D volume.

    Results are written to <save_dir>/seg_dice_metric.csv and appended
    to <save_dir>/summary_metrics.txt as:
        Global Seg <label> Dice <method>: <mean> ± <std>
    """
    all_pids   = np.load(os.path.join(save_dir, "patient_ids.npy"),
                         allow_pickle=True).flatten()
    all_phases = np.load(os.path.join(save_dir, "cycle_phases.npy")).flatten()
    all_idx    = np.arange(len(all_pids))
    vol_dir    = os.path.join(save_dir, "volumes")

    test_patients = sorted(set(str(p) for p in all_pids))
    rows = []

    for patient in test_patients:
        print(f"\n── {patient} ──")
        info         = _patient_info(patient, data_dir)
        ed_seg       = _load_seg_vol(data_dir, patient, info["ED"], orientation)
        es_seg       = _load_seg_vol(data_dir, patient, info["ES"], orientation)

        pat_mask  = np.array([str(p) == patient for p in all_pids])
        pat_idx   = all_idx[pat_mask]
        pat_phases = all_phases[pat_mask]
        es_local  = int(np.argmin(np.abs(pat_phases - 0.5)))
        es_iter   = int(pat_idx[es_local])
        es_phase  = float(pat_phases[es_local])

        for method_name, dvf_name in methods.items():
            dvf = _load_dvf(vol_dir, dvf_name, es_iter,
                            target_shape=tuple(ed_seg.shape))
            if dvf is None:
                print(f"  [{method_name}] DVF not found — skipping")
                continue

            pred_seg = warp_seg_with_dvf(ed_seg, dvf, torch.device("cpu"))
            pred_seg = pred_seg.numpy() if hasattr(pred_seg, "numpy") else np.asarray(pred_seg)

            row = {"patient": patient, "method": method_name, "es_phase": es_phase}
            for lbl, name in DICE_LABELS.items():
                d = _dice(pred_seg, es_seg, lbl)
                row[f"{name}_dice"] = d
            rows.append(row)
            scores = {DICE_LABELS[l]: f"{row[f'{DICE_LABELS[l]}_dice']:.3f}"
                      for l in DICE_LABELS}
            print(f"  [{method_name}] Dice: {scores}")

    if not rows:
        print("[seg_dice] No data collected.")
        return rows

    # ── Console summary ───────────────────────────────────────────────────────
    method_names = list(dict.fromkeys(r["method"] for r in rows))
    print(f"\n{'Method':<16}", end="")
    for name in DICE_LABELS.values():
        print(f"  {name} Dice    ", end="")
    print()
    print("-" * (16 + len(DICE_LABELS) * 14))
    for method_name in method_names:
        mrows = [r for r in rows if r["method"] == method_name]
        print(f"{method_name:<16}", end="")
        for name in DICE_LABELS.values():
            vals = [r[f"{name}_dice"] for r in mrows]
            print(f"  {np.mean(vals):.3f} ±{np.std(vals):.3f}  ", end="")
        print()

    # ── CSV ───────────────────────────────────────────────────────────────────
    import csv
    out_path = os.path.join(save_dir, "seg_dice_metric.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[seg_dice] Saved → {out_path}")

    # ── Append to summary_metrics.txt ─────────────────────────────────────────
    summary_path = os.path.join(save_dir, "summary_metrics.txt")
    with open(summary_path, "a") as f:
        for method_name in method_names:
            mrows = [r for r in rows if r["method"] == method_name]
            for name in DICE_LABELS.values():
                vals = [r[f"{name}_dice"] for r in mrows]
                f.write(f"Global Seg {name} Dice {method_name}: "
                        f"{np.mean(vals):.4f} ± {np.std(vals):.4f}\n")
    print(f"[seg_dice] Appended Dice scores to {summary_path}")

    return rows


# ── 3-D Dice vs MambaMorph (no GT ES needed) ──────────────────────────────────

def compute_dice_vs_mambamorph(
    save_dir:    str,
    data_dir:    str,
    methods:     dict,      # must contain "MambaMorph": dvf_subfolder
    orientation=None,
) -> list[dict]:
    """
    3-D Dice between each model's warped ED seg and MambaMorph's warped ED seg.

    Instead of comparing against GT ES, this uses MambaMorph's prediction as
    the reference.  A model that perfectly replicates MambaMorph scores 1.0;
    the zero-motion baseline (identity DVF vs MambaMorph) is ~0.60 for RV —
    the same useful range without needing GT labels at ES.

    Results appended to <save_dir>/summary_metrics.txt as:
        Global Seg <label> DiceVsMM <method>: <mean> ± <std>
    """
    if "MambaMorph" not in methods:
        print("[seg_dice_vs_mm] 'MambaMorph' not in methods — skipping")
        return []

    all_pids   = np.load(os.path.join(save_dir, "patient_ids.npy"),
                         allow_pickle=True).flatten()
    all_phases = np.load(os.path.join(save_dir, "cycle_phases.npy")).flatten()
    all_idx    = np.arange(len(all_pids))
    vol_dir    = os.path.join(save_dir, "volumes")

    test_patients  = sorted(set(str(p) for p in all_pids))
    mm_dvf_name    = methods["MambaMorph"]
    other_methods  = {k: v for k, v in methods.items() if k != "MambaMorph"}
    rows = []

    for patient in test_patients:
        print(f"\n── {patient} ──")
        info   = _patient_info(patient, data_dir)
        ed_seg = _load_seg_vol(data_dir, patient, info["ED"], orientation)

        pat_mask   = np.array([str(p) == patient for p in all_pids])
        pat_idx    = all_idx[pat_mask]
        pat_phases = all_phases[pat_mask]
        es_local   = int(np.argmin(np.abs(pat_phases - 0.5)))
        es_iter    = int(pat_idx[es_local])
        es_phase   = float(pat_phases[es_local])

        mm_dvf = _load_dvf(vol_dir, mm_dvf_name, es_iter,
                           target_shape=tuple(ed_seg.shape))
        if mm_dvf is None:
            print(f"  [MambaMorph] DVF not found — skipping {patient}")
            continue
        mm_seg = warp_seg_with_dvf(ed_seg, mm_dvf, torch.device("cpu"))
        mm_seg = mm_seg.numpy() if hasattr(mm_seg, "numpy") else np.asarray(mm_seg)

        for method_name, dvf_name in other_methods.items():
            dvf = _load_dvf(vol_dir, dvf_name, es_iter,
                            target_shape=tuple(ed_seg.shape))
            if dvf is None:
                print(f"  [{method_name}] DVF not found — skipping")
                continue

            pred_seg = warp_seg_with_dvf(ed_seg, dvf, torch.device("cpu"))
            pred_seg = pred_seg.numpy() if hasattr(pred_seg, "numpy") else np.asarray(pred_seg)

            row = {"patient": patient, "method": method_name, "es_phase": es_phase}
            for lbl, name in DICE_LABELS.items():
                row[f"{name}_dice_vs_mm"] = _dice(pred_seg, mm_seg, lbl)
            rows.append(row)
            scores = {DICE_LABELS[l]: f"{row[f'{DICE_LABELS[l]}_dice_vs_mm']:.3f}"
                      for l in DICE_LABELS}
            print(f"  [{method_name}] Dice vs MM: {scores}")

    if not rows:
        print("[seg_dice_vs_mm] No data collected.")
        return rows

    method_names = list(dict.fromkeys(r["method"] for r in rows))
    print(f"\n{'Method':<16}", end="")
    for name in DICE_LABELS.values():
        print(f"  {name} DiceVsMM ", end="")
    print()
    print("-" * (16 + len(DICE_LABELS) * 14))
    for method_name in method_names:
        mrows = [r for r in rows if r["method"] == method_name]
        print(f"{method_name:<16}", end="")
        for name in DICE_LABELS.values():
            vals = [r[f"{name}_dice_vs_mm"] for r in mrows]
            print(f"  {np.mean(vals):.3f} ±{np.std(vals):.3f}  ", end="")
        print()

    summary_path = os.path.join(save_dir, "summary_metrics.txt")
    with open(summary_path, "a") as f:
        for method_name in method_names:
            mrows = [r for r in rows if r["method"] == method_name]
            for name in DICE_LABELS.values():
                vals = [r[f"{name}_dice_vs_mm"] for r in mrows]
                f.write(f"Global Seg {name} DiceVsMM {method_name}: "
                        f"{np.mean(vals):.4f} ± {np.std(vals):.4f}\n")
    print(f"[seg_dice_vs_mm] Appended DiceVsMM scores to {summary_path}")

    return rows


# ── CLI entry point ───────────────────────────────────────────────────────────

def evaluate(data_dir=DATA_DIR, out=None, img_dir=None, n_img_patients=5):
    orientation = load_orientation_info(ORIENTATION_CSV) \
        if os.path.exists(ORIENTATION_CSV) else None
    rows = compute_for_test_dir(
        save_dir   = TEST_DIR,
        data_dir   = data_dir,
        methods    = {name: cfg["dvf_name"] for name, cfg in METHODS.items()},
        orientation = orientation,
        img_dir    = img_dir,
        n_img_patients = n_img_patients,
    )
    if out and rows:
        import csv
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Saved → {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--out", default=None, help="Path for per-patient CSV")
    p.add_argument("--img_dir", default=None,
                   help="Directory to save segmentation comparison images")
    p.add_argument("--n_img_patients", type=int, default=5,
                   help="Number of patients to visualise (default: 5)")
    args = p.parse_args()
    evaluate(data_dir=args.data_dir, out=args.out,
             img_dir=args.img_dir, n_img_patients=args.n_img_patients)


if __name__ == "__main__":
    main()
