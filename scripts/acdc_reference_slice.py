"""
acdc_reference_slice.py
========================
Selects, for each patient, the single short-axis slice that best represents
the ED->ES phase cycle, using only ED and ES segmentations. Also estimates,
from the same segmentations, a per-patient 3D heart centroid and an in-plane
(H-W) rotation-correction angle, so downstream preprocessing (`acdc_4d.py`)
can crop around the heart itself — instead of the volume's geometric center —
and bring every patient's heart to a consistent orientation.

Selection criteria for the reference slice (per candidate slice):
  1. Completeness — RV, Myocardium and LV cavity all present (>0 px) at
     BOTH ED and ES. Slices failing this are discarded (base drop-out /
     apex vanishing cavity).
  2. Informativeness — larger total cross-sectional area (RV+Myo+LV) at ED
     is preferred (mid-ventricular slices are the largest & cleanest).
  3. Typical contraction — the LV area-change fraction
     (lv_ed - lv_es) / lv_ed is preferred to be close to the patient's own
     median across candidate slices (avoids picking an atypical /
     valve-plane-affected slice).
  4. (optional) Position prior — an explicit bias toward the anatomical
     mid-ventricle (rel_pos=0.5), controlled by --w_pos (0 = disabled).

score(s) = z(total_area_ed)  -  |z(ef_frac)|  -  w_pos * |rel_pos - 0.5|

The slice with the highest score per patient is the reference slice.

Heart centroid:
  3D centroid (d, h, w), in NATIVE voxel space, of the union of RV+Myo+LV
  labels over the FULL ED segmentation stack (not just the reference slice) —
  a full-stack centroid is more robust than a single-slice one, since it
  doesn't depend on the reference slice choice being anatomically "typical"
  in every dimension.

Rotation-correction angle:
  Estimated ONLY at the reference slice (where the completeness filter
  guarantees RV/Myo/LV are all well-formed): the angle of the LV->RV
  centroid vector in the (H, W) plane. Rotating a frame by `-rotation_angle`
  (see `acdc_4d._rotate_hw`) brings this vector onto a fixed canonical
  direction (+W) for every patient, correcting the scanner/patient-dependent
  in-plane rotation that ACDC volumes are otherwise inconsistent about.

Usage:
    python scripts/acdc_reference_slice.py \\
        --data_dir /path/to/ACDC/database \\
        --out_csv mopred/data/reference_slices.csv
"""

from __future__ import annotations
import argparse
import os
import csv
import math

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {1: "RV cavity", 2: "Myocardium", 3: "LV cavity"}
LABEL_COLORS_RGB = {
    1: (0.31, 0.62, 0.77),   # RV — blue
    2: (0.96, 0.64, 0.38),   # Myo — orange
    3: (0.90, 0.22, 0.27),   # LV — red
}


def _read_info(patient_dir: str) -> dict:
    info = {}
    with open(os.path.join(patient_dir, "Info.cfg")) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return info


def _load_seg(patient_dir: str, patient: str, t_idx: int) -> np.ndarray:
    path = os.path.join(patient_dir, f"{patient}_frame{t_idx:02d}_gt.nii.gz")
    arr  = np.asarray(nib.load(path).dataobj, dtype=np.int32)
    return arr.transpose(2, 0, 1)   # (D, H, W)


def _load_image(patient_dir: str, patient: str, t_idx: int) -> np.ndarray:
    path = os.path.join(patient_dir, f"{patient}_frame{t_idx:02d}.nii.gz")
    arr  = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return arr.transpose(2, 0, 1)   # (D, H, W)


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def compute_heart_centroid(seg: np.ndarray, labels=(1, 2, 3)) -> tuple[float, float, float] | None:
    """3D centroid (d, h, w), in NATIVE voxel space, of the union of `labels`
    over the full `seg` stack (D, H, W). Returns None if no voxel of any
    label is present (shouldn't happen for a patient that has a valid
    reference slice, but guarded defensively)."""
    mask = np.isin(seg, labels)
    if not mask.any():
        return None
    d, h, w = np.nonzero(mask)
    return float(d.mean()), float(h.mean()), float(w.mean())


def compute_rotation_angle(seg_slice: np.ndarray) -> float | None:
    """Angle (radians) of the LV->RV centroid vector within a single 2D
    segmentation slice (H, W). Returns None if RV or LV is absent in this
    slice (shouldn't happen at a slice that passed the completeness filter).

    This is a simple, standard proxy for cardiac in-plane orientation: the
    vector from the LV center to the RV center points in a fairly consistent
    anatomical direction (RV is always alongside/anterior to the LV), so its
    angle in image coordinates captures how much this particular acquisition
    is rotated relative to any other patient's."""
    lv_h, lv_w = np.nonzero(seg_slice == 3)
    rv_h, rv_w = np.nonzero(seg_slice == 1)
    if len(lv_h) == 0 or len(rv_h) == 0:
        return None
    dh = rv_h.mean() - lv_h.mean()
    dw = rv_w.mean() - lv_w.mean()
    return float(np.arctan2(dh, dw))


def select_reference_slices(data_dir: str, w_pos: float = 0.0) -> list[dict]:
    """
    w_pos : weight of the anatomical mid-slice position prior.
            Adds  -w_pos * |rel_pos - 0.5|  to the score, penalizing slices
            far from the anatomical mid-ventricle (rel_pos=0.5), regardless
            of the area/typicality proxies. w_pos=0 (default) disables it,
            reproducing the original area-only behaviour.
    """
    patients = sorted(d for d in os.listdir(data_dir) if d.startswith("patient"))
    results = []

    for patient in patients:
        patient_dir = os.path.join(data_dir, patient)
        info = _read_info(patient_dir)
        ed, es = int(info["ED"]), int(info["ES"])

        ed_seg = _load_seg(patient_dir, patient, ed)
        es_seg = _load_seg(patient_dir, patient, es)
        n_d = ed_seg.shape[0]

        candidates = []  # list of dicts, one per valid slice
        for s in range(n_d):
            areas_ed = {l: int((ed_seg[s] == l).sum()) for l in LABELS}
            areas_es = {l: int((es_seg[s] == l).sum()) for l in LABELS}

            # completeness: all labels present at both ED and ES
            if any(areas_ed[l] == 0 or areas_es[l] == 0 for l in LABELS):
                # print(f"  {patient}: slice {s} skipped (incomplete labels)")
                continue

            total_area_ed = sum(areas_ed.values())
            lv_ed, lv_es = areas_ed[3], areas_es[3]
            ef_frac = (lv_ed - lv_es) / lv_ed

            candidates.append({
                "slice": s,
                "rel_pos": round(s / max(n_d - 1, 1) * 10) / 10,
                "total_area_ed": total_area_ed,
                "ef_frac": ef_frac,
                "lv_ed": lv_ed,
                "lv_es": lv_es,
            })

        if not candidates:
            print(f"  {patient}: no valid slice found (all-label completeness failed)")
            continue

        total_areas = np.array([c["total_area_ed"] for c in candidates], dtype=float)
        ef_fracs    = np.array([c["ef_frac"] for c in candidates], dtype=float)
        rel_poses   = np.array([c["rel_pos"] for c in candidates], dtype=float)

        z_area = _zscore(total_areas)
        z_ef   = _zscore(ef_fracs)
        pos_penalty = np.abs(rel_poses - 0.5)   # 0 at mid-ventricle, grows toward base/apex

        scores = z_area - np.abs(z_ef) - w_pos * pos_penalty

        best_idx = int(np.argmax(scores))
        best = candidates[best_idx]
        best_slice = best["slice"]

        # Heart centroid: 3D, over the full ED stack (native voxel space).
        centroid = compute_heart_centroid(ed_seg)

        # Rotation-correction angle: 2D, at the chosen reference slice only
        # (guaranteed complete RV/Myo/LV there by the filter above).
        rotation_angle = compute_rotation_angle(ed_seg[best_slice])

        results.append({
            "patient": patient,
            "n_slices": n_d,
            "reference_slice": best_slice,
            "rel_position": best["rel_pos"],
            "score": float(scores[best_idx]),
            "total_area_ed": best["total_area_ed"],
            "ef_frac": round(best["ef_frac"], 4),
            "lv_ed_area": best["lv_ed"],
            "lv_es_area": best["lv_es"],
            "n_candidates": len(candidates),
            "pos_penalty": round(float(pos_penalty[best_idx]), 4),
            "centroid_d": round(centroid[0], 2) if centroid is not None else "",
            "centroid_h": round(centroid[1], 2) if centroid is not None else "",
            "centroid_w": round(centroid[2], 2) if centroid is not None else "",
            "rotation_angle": round(rotation_angle, 5) if rotation_angle is not None else "",
        })
        rot_str = f"{rotation_angle:+.3f} rad" if rotation_angle is not None else "N/A"
        print(f"  {patient}: reference slice = {best_slice} "
              f"(rel_pos={best['rel_pos']:.1f}, ef_frac={best['ef_frac']:.2f}, "
              f"{len(candidates)}/{n_d} candidates, centroid={centroid}, rotation={rot_str})")

    return results


def write_txt(results: list[dict], out_txt: str) -> None:
    with open(out_txt, "w") as f:
        for r in results:
            rot_str = f"{r['rotation_angle']:+.3f}" if r["rotation_angle"] != "" else "N/A"
            f.write(f"{r['patient']}: slice {r['reference_slice']} "
                     f"(rel_pos={r['rel_position']:.1f}, "
                     f"ef_frac={r['ef_frac']:.2f}, rotation_angle={rot_str})\n")
    print(f"Saved reference slices (txt) → {out_txt}")


def _normalize_img(img_slice: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img_slice, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img_slice - lo) / (hi - lo), 0, 1)
    return out


def plot_reference_slices(data_dir: str, results: list[dict], out_png: str,
                           alpha: float = 0.4) -> None:
    n = len(results)
    if n == 0:
        print("Nothing to plot.")
        return

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.5 * rows),
                              facecolor="white")
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, results):
        patient = r["patient"]
        s = r["reference_slice"]
        patient_dir = os.path.join(data_dir, patient)
        info = _read_info(patient_dir)
        ed = int(info["ED"])

        img = _load_image(patient_dir, patient, ed)[s]
        seg = _load_seg(patient_dir, patient, ed)[s]

        img_norm = _normalize_img(img)
        rgb = np.stack([img_norm] * 3, axis=-1)

        overlay = rgb.copy()
        for label, color in LABEL_COLORS_RGB.items():
            mask = seg == label
            for c in range(3):
                overlay[..., c] = np.where(mask, (1 - alpha) * rgb[..., c] + alpha * color[c],
                                            overlay[..., c])

        ax.imshow(overlay)
        ax.set_title(f"{patient}\nslice {s} (rel={r['rel_position']:.1f})",
                     fontsize=7)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    # shared legend
    handles = [plt.Line2D([0], [0], marker="s", color="w",
               markerfacecolor=color, markersize=10, label=LABELS[label])
               for label, color in LABEL_COLORS_RGB.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved reference slice grid plot → {out_png}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Select reference slice per ACDC patient.")
    p.add_argument("--data_dir", default="/volatile/data/cam/ACDC/database")
    p.add_argument("--out_csv", default="reference_slices.csv")
    p.add_argument("--out_txt", default="reference_slices.txt")
    p.add_argument("--out_png", default="reference_slices_grid.png")
    p.add_argument("--w_pos", type=float, default=0.0,
                   help="Weight of the anatomical mid-slice position prior "
                        "(-w_pos * |rel_pos - 0.5|). 0 = disabled (default). "
                        "Try e.g. 1.0-3.0 to bias explicitly toward mid-ventricle.")
    args = p.parse_args()

    print("Selecting reference slices …")
    results = select_reference_slices(args.data_dir, w_pos=args.w_pos)

    fieldnames = ["patient", "n_slices", "reference_slice", "rel_position", "score",
                  "total_area_ed", "ef_frac", "lv_ed_area", "lv_es_area",
                  "n_candidates", "pos_penalty",
                  "centroid_d", "centroid_h", "centroid_w", "rotation_angle"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    rel_positions = [r["rel_position"] for r in results]
    n_with_rotation = sum(1 for r in results if r["rotation_angle"] != "")
    print(f"\nSaved {len(results)} reference slices → {args.out_csv}")
    print(f"Reference slice rel_position: mean={np.mean(rel_positions):.2f}, "
          f"median={np.median(rel_positions):.2f} "
          f"(expect ~0.4-0.6, i.e. mid-ventricle)")
    print(f"Rotation angle estimated for {n_with_rotation}/{len(results)} patients")

    write_txt(results, args.out_txt)
    plot_reference_slices(args.data_dir, results, args.out_png)


if __name__ == "__main__":
    main()