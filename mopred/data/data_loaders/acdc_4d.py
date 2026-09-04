"""
PyTorch Dataset for the ACDC cine-MRI cardiac dataset.
Loads 4D NIfTI volumes lazily, resamples to a fixed voxel spacing, and
center-crops to a canonical (32, 128, 128) shape with min-max normalisation.
Cardiac phase is computed from ED/ES frame indices in Info.cfg; orientation
info (centroid, rotation) is read from a pre-computed CSV.
"""

from torch.utils.data.dataset import Dataset
import os
import math
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F


# ── Canonical preprocessing target ───────────────────────────────────────────
# (D, H, W) order, matching the rest of the pipeline's volume convention.
_TARGET_SPACING = (3.12, 1.5, 1.5)     # mm / voxel — slice thickness x in-plane x in-plane
CANONICAL_SHAPE = (32, 128, 128)       # new VOL_SIZE for the ACDC pipeline (replaces liver's (32,64,64))

# liver's NAVIGATOR_4D_Dataset_multitime halves H/W in-plane (F.interpolate,
# scale_factor=[1, 0.5, 0.5], trilinear) on top of its own native resolution —
# matching that exact op here gives ACDC the same (32, 64, 64) final shape.
LIVER_SHAPE = (32, 64, 64)


def _cache_path(cache_dir: str, patient: str, t_idx: int, downsample: bool,
                 orientation_entry: dict | None = None) -> str:
    """Return the .npy path for a pre-processed frame inside cache_dir.

    The suffix depends on which geometry was applied, so caches built under
    different orientation settings never collide with (or get silently
    served from) each other:
      - no orientation info at all                       -> no suffix
      - reference_slice only (D-axis centering only)      -> "_refslice"
      - centroid and/or rotation_angle present            -> "_oriented"
        (full heart-centered crop, optionally rotated)
    """
    suffix = "_liver" if downsample else ""
    if orientation_entry is not None:
        has_rot = bool(orientation_entry.get("rotation_angle"))
        has_centroid = orientation_entry.get("centroid") is not None
        has_slice = orientation_entry.get("reference_slice") is not None
        if has_rot or has_centroid:
            suffix += "_oriented"
        elif has_slice:
            suffix += "_refslice"
    return os.path.join(cache_dir, patient, f"t_{t_idx:02d}{suffix}.npy")


def _downsample_to_liver_shape(vol: torch.Tensor) -> torch.Tensor:
    """vol: (D, H, W) at CANONICAL_SHAPE. Halve H/W to match LIVER_SHAPE."""
    vol = vol.unsqueeze(0).unsqueeze(0)
    vol = F.interpolate(vol, scale_factor=[1, 0.5, 0.5], mode="trilinear", align_corners=False)
    return vol.squeeze(0).squeeze(0)


# ── Info.cfg / phase machinery ────────────────────────────────────────────────

_INFO_CACHE: dict = {}   # (patient, data_dir) -> {"ED", "ES", "NbFrame", "spacing"}


def _read_info_cfg(patient_dir: str) -> dict:
    """Parse `Info.cfg` (simple `key: value` lines) into a dict of strings."""
    info = {}
    with open(os.path.join(patient_dir, "Info.cfg")) as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            info[key.strip()] = val.strip()
    return info


def _patient_info(patient: str, data_dir: str) -> dict:
    """Cached per-patient {ED, ES, NbFrame, spacing=(D,H,W) mm/voxel}."""
    key = (patient, data_dir)
    if key not in _INFO_CACHE:
        patient_dir = os.path.join(data_dir, patient)
        cfg = _read_info_cfg(patient_dir)
        img = nib.load(os.path.join(patient_dir, f"{patient}_4d.nii.gz"))
        h, w, d, _ = img.header.get_zooms()
        _INFO_CACHE[key] = {
            "ED":      int(cfg["ED"]),
            "ES":      int(cfg["ES"]),
            "NbFrame": int(cfg["NbFrame"]),
            "spacing": (float(d), float(h), float(w)),   # reordered to (D, H, W)
        }
    return _INFO_CACHE[key]


def get_cardiac_phase(patient: str, t_idx: int, data_dir: str) -> float:
    """
    Exact cardiac phase in [0, 1], computed directly from Info.cfg's ED/ES/NbFrame
    (cardiac analog of `get_cycle_phase`, navigator_4d.py:79-96 — but exact rather
    than estimated from a navigator-intensity curve):

        0.0 / 1.0 -> End-Diastole  (relaxed/maximally-filled, the reference pose)
        0.5       -> End-Systole   (maximally contracted)

    Linearly interpolates within each half of the cycle:
      ED -> ES (systole, contracting):  phase 0.0 -> 0.5
      ES -> ED (diastole, relaxing):    phase 0.5 -> 1.0
    """
    info = _patient_info(patient, data_dir)
    ed, es, nb = info["ED"], info["ES"], info["NbFrame"]

    rel    = (t_idx - ed) % nb        # frames-since-ED, cyclic
    es_rel = (es - ed) % nb           # ES position relative to ED

    if rel <= es_rel:
        return 0.5 * rel / es_rel
    else:
        return 0.5 + 0.5 * (rel - es_rel) / (nb - es_rel)


def get_acdc_ref_id(patient: str, data_dir: str) -> tuple[int, int]:
    """(ES_idx, ED_idx) — cardiac analog of `_REF_ID`'s (inhale, exhale) convention:
    ES = peak-displacement pose (like inhale), ED = relaxed reference pose (like exhale)."""
    info = _patient_info(patient, data_dir)
    return info["ES"], info["ED"]


# ── Orientation info: reference slice + heart centroid + rotation angle ──────
def load_orientation_info(path: str) -> dict:
    """
    Load the per-patient orientation info produced by `acdc_reference_slice.py`
    (either the combined `.csv` it writes, or the legacy `.txt` summary).

    Returns {patient: entry} where entry is a dict with keys:
      "reference_slice"  : int | None  — index into the ORIGINAL, native-resolution
                                          D axis (as indexed directly into
                                          `ed_seg`/`es_seg`, *before*
                                          `_resample_to_spacing` changes the slice
                                          count). Convert with
                                          `_native_to_resampled_index` before using
                                          it to crop a resampled volume.
      "centroid"          : (d, h, w) | None — 3D centroid of the heart (RV+Myo+LV)
                                          at ED, in the SAME native D/H/W grid as
                                          `reference_slice`. Convert with
                                          `_native_to_resampled_point` before use.
      "rotation_angle"    : float | None — in-plane (H-W) correction angle in
                                          radians, estimated at the reference slice
                                          from the LV->RV centroid vector. Rotating
                                          the H-W plane by `-rotation_angle` brings
                                          every patient's heart to the same
                                          orientation (see `_rotate_hw`).
      "canonical_d_index"  : int | None — D-axis index this patient's reference
                                          slice/centroid actually lands at in the
                                          FINAL CANONICAL_SHAPE-cropped volume, as
                                          precomputed by
                                          `acdc_preprocessed_reference_grid.py`'s
                                          `update_orientation_csv`. When present,
                                          `get_reference_slice_index` uses this
                                          directly instead of recomputing it via a
                                          dummy resample — only absent for CSVs
                                          that predate that script's caching.

    Any field not present in the source file is set to None, so a legacy
    `.txt`/`.csv` that only has `reference_slice` degrades gracefully to the
    old "D-axis-only" centering behaviour.
    """
    ext = os.path.splitext(path)[1].lower()
    orientation: dict = {}

    if ext == ".csv":
        import csv
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                patient = row["patient"]

                ref_slice = None
                if row.get("reference_slice", "") not in (None, ""):
                    ref_slice = int(row["reference_slice"])

                centroid = None
                if all(row.get(k, "") not in (None, "") for k in
                       ("centroid_d", "centroid_h", "centroid_w")):
                    centroid = (float(row["centroid_d"]), float(row["centroid_h"]),
                                float(row["centroid_w"]))

                rotation_angle = None
                if row.get("rotation_angle", "") not in (None, ""):
                    rotation_angle = float(row["rotation_angle"])

                canonical_d_index = None
                if row.get("canonical_d_index", "") not in (None, ""):
                    canonical_d_index = int(row["canonical_d_index"])

                orientation[patient] = {
                    "reference_slice": ref_slice,
                    "centroid": centroid,
                    "rotation_angle": rotation_angle,
                    "canonical_d_index": canonical_d_index,
                }
    else:
        # Legacy .txt lines look like: "patient119: slice 5 (rel_pos=0.3, ef_frac=0.58)"
        # — reference_slice only, no centroid/rotation/canonical-index info available.
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                patient, rest = line.split(":", 1)
                tok = rest.strip().split()
                slice_idx = int(tok[1]) if tok and tok[0] == "slice" else None
                if slice_idx is None:
                    continue
                orientation[patient.strip()] = {
                    "reference_slice": slice_idx,
                    "centroid": None,
                    "rotation_angle": None,
                    "canonical_d_index": None,
                }

    return orientation


def _native_to_resampled_index(native_idx: int, native_spacing_d: float,
                                target_spacing_d: float = _TARGET_SPACING[0]) -> int:
    """
    Convert a slice index in the ORIGINAL (native) D-axis grid to the
    corresponding index after `_resample_to_spacing`.

    `_resample_to_spacing` calls F.interpolate(..., scale_factor=native/target, ...),
    which stretches the D axis by exactly that same factor (new_size ≈
    old_size * native_spacing_d / target_spacing_d). A given anatomical
    position at native index `s` sits at physical depth `s * native_spacing_d`
    mm from the first slice; in the resampled grid that same physical depth
    corresponds to index `s * native_spacing_d / target_spacing_d`.
    """
    scale = native_spacing_d / target_spacing_d
    return int(round(native_idx * scale))


def _native_to_resampled_point(native_dhw: tuple, spacing: tuple,
                                target_spacing: tuple = _TARGET_SPACING) -> tuple:
    """Same idea as `_native_to_resampled_index`, generalised to all 3 axes.
    native_dhw / spacing are both (d, h, w) tuples; returns (d, h, w) indices
    in resampled space, rounded to the nearest voxel."""
    return tuple(int(round(n * s / t)) for n, s, t in zip(native_dhw, spacing, target_spacing))


# ── Resampling-by-spacing, rotation, canonical crop/pad ──────────────────────

def _resample_to_spacing(vol: torch.Tensor, spacing: tuple, target_spacing: tuple = _TARGET_SPACING,
                          mode: str = "trilinear") -> torch.Tensor:
    """vol: (D, H, W). Resample so every voxel covers `target_spacing` mm of tissue —
    keeps anatomy at a consistent physical scale across patients with differing
    native voxel spacing (design decision #3).

    mode: "trilinear" for images (default), "nearest" for segmentation/label maps
    (avoids inventing fractional label values between classes)."""
    scale = [s / t for s, t in zip(spacing, target_spacing)]
    vol = vol.unsqueeze(0).unsqueeze(0)
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    vol = F.interpolate(vol, scale_factor=scale, mode=mode, **kwargs)
    return vol.squeeze(0).squeeze(0)


def _rotate_hw(vol: torch.Tensor, angle: float, center_hw: tuple,
               mode: str = "bilinear") -> torch.Tensor:
    """Rotate the H-W plane of `vol` (D, H, W) by `-angle` radians, pivoting on
    `center_hw` = (center_h, center_w) (indices in the SAME space as `vol`,
    i.e. already resampled — see `_native_to_resampled_point`). Applied
    identically to every D slice; the D axis itself is left untouched (only
    the in-plane rotation around the D axis varies across ACDC patients, not
    the short-axis slice ordering).

    `angle` is the LV->RV centroid-vector angle estimated by
    `acdc_reference_slice.py`; rotating by `-angle` brings that vector onto a
    fixed canonical direction (+W) for every patient. Sign convention here is
    the standard one for this construction, but since it can't be verified
    without visual inspection, always sanity-check with
    `acdc_preprocessed_reference_grid.py` and flip the sign of `angle` at the
    source (in `acdc_reference_slice.py`) if RV/LV end up mirrored.

    mode: "bilinear" for images (default), "nearest" for segmentation/label maps.
    """
    if not angle:
        return vol

    D, H, W = vol.shape
    center_h, center_w = center_hw
    # Normalize pivot to [-1, 1], matching affine_grid/grid_sample conventions.
    cy = (center_h / max(H - 1, 1)) * 2 - 1
    cx = (center_w / max(W - 1, 1)) * 2 - 1
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    v = vol.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    # torch's 5D affine_grid expects theta (N, 3, 4) acting on (x, y, z) <-> (W, H, D).
    # Row 0 -> sampled W-coord, row 1 -> sampled H-coord, row 2 -> sampled D-coord
    # (identity: D is never rotated/shifted).
    theta = torch.tensor([[
        [cos_a, -sin_a, 0.0, cx * (1 - cos_a) + sin_a * cy],
        [sin_a,  cos_a, 0.0, cy * (1 - cos_a) - sin_a * cx],
        [0.0,    0.0,   1.0, 0.0],
    ]], dtype=v.dtype)
    grid = F.affine_grid(theta, v.shape, align_corners=False)
    padding_mode = "zeros"
    v = F.grid_sample(v, grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    return v.squeeze(0).squeeze(0)


def _center_crop_or_pad(vol: torch.Tensor, shape: tuple = CANONICAL_SHAPE,
                         center: tuple = (None, None, None)) -> torch.Tensor:
    """Crop or zero-pad each of the 3 spatial dims of `vol` (D, H, W) to `shape`.

    `center` is a (center_d, center_h, center_w) tuple, each entry either an
    index (in the SAME space as `vol`, i.e. already resampled) to crop/pad
    around for that axis, or None to fall back to that axis's own geometric
    center (the original, patient-agnostic behaviour).
    """
    for dim, target in enumerate(shape):
        cur = vol.shape[dim]
        c = center[dim]
        if c is None:
            c = cur // 2

        if cur >= target:
            start = c - target // 2
            start = max(0, min(start, cur - target))
            vol = vol.narrow(dim, start, target)
        else:
            # Volume shorter than target: pad, keeping `c` as close to centered
            # in the padded result as the available margin allows.
            before = min(c, target - cur)
            before = max(0, before)
            after = target - cur - before
            pad = [0, 0, 0, 0, 0, 0]              # F.pad order: (W_b, W_a, H_b, H_a, D_b, D_a)
            pad[2 * (2 - dim)]     = before
            pad[2 * (2 - dim) + 1] = after
            vol = F.pad(vol, pad)
    return vol


def _load_and_preprocess(root_dir: str, patient: str, t_idx: int, img_cache: dict,
                          downsample: bool = False,
                          cache_dir: str | None = None,
                          orientation: dict | None = None) -> torch.Tensor:
    """Load + preprocess one frame (resample to target spacing, optional rotation
    correction, canonical crop/pad, min-max normalise to [0, 1]). Shared by
    ACDC_4D_Dataset._load_frame and RefVolume_Dataset_augment_ACDC.

    downsample: if True, halve H/W after the canonical crop/pad so the final
    shape matches the liver pipeline's LIVER_SHAPE=(32,64,64) instead of
    CANONICAL_SHAPE=(32,128,128).
    cache_dir: if set and the pre-computed .npy exists, load it directly
    (skips gzip decompress + trilinear resample — ~300x faster per frame).
    orientation: optional {patient: entry} dict from `load_orientation_info`.
      - entry["reference_slice"] (if set) centers the D-axis crop on that
        slice instead of the volume's geometric center.
      - entry["centroid"] (if set) centers the H/W crop on the heart's own
        centroid instead of the volume's geometric center. If both centroid
        and reference_slice are given, reference_slice wins for the D axis
        (it was chosen via the segmentation-completeness pipeline, more
        robust per-slice than the centroid's own D estimate).
      - entry["rotation_angle"] (if set) rotates the H-W plane before
        cropping, pivoting on the centroid (or the volume center if no
        centroid was given) — see `_rotate_hw`.
    """
    entry = orientation.get(patient) if orientation is not None else None

    if cache_dir is not None:
        cp = _cache_path(cache_dir, patient, t_idx, downsample, entry)
        if os.path.exists(cp):
            # print(f"[acdc_4d] loading preprocessed frame from cache: {cp}")
            return torch.from_numpy(np.load(cp))

    if patient not in img_cache:
        patient_dir = os.path.join(root_dir, patient)
        img_cache[patient] = nib.load(os.path.join(patient_dir, f"{patient}_4d.nii.gz"))
    img = img_cache[patient]

    frame = np.asarray(img.dataobj[..., t_idx], dtype=np.float32)   # (H, W, D)
    frame = np.transpose(frame, (2, 0, 1))                          # -> (D, H, W)

    spacing = _patient_info(patient, root_dir)["spacing"]           # (D, H, W) mm/voxel

    ref_slice      = entry.get("reference_slice") if entry else None
    centroid_native = entry.get("centroid") if entry else None
    rotation_angle  = entry.get("rotation_angle") if entry else None

    center_d = center_h = center_w = None
    if centroid_native is not None:
        cd, ch, cw = _native_to_resampled_point(centroid_native, spacing)
        center_h, center_w = ch, cw
        center_d = cd
        # print(f"Patient {patient}: native centroid {centroid_native} -> resampled (D,H,W) = ({cd},{ch},{cw})")
    if ref_slice is not None:
        center_d = _native_to_resampled_index(ref_slice, spacing[0])
        # print(f"Patient {patient}: native reference_slice {ref_slice} -> resampled D index = {center_d}")

    vol = torch.from_numpy(np.ascontiguousarray(frame))
    vol = _resample_to_spacing(vol, spacing)
    if rotation_angle:
        pivot_h = center_h if center_h is not None else vol.shape[1] // 2
        pivot_w = center_w if center_w is not None else vol.shape[2] // 2
        vol = _rotate_hw(vol, rotation_angle, (pivot_h, pivot_w))
        # print(f"Patient {patient}: rotated H-W plane by {-rotation_angle:.3f} rad around (H,W)=({pivot_h},{pivot_w})")
    vol = _center_crop_or_pad(vol, center=(center_d, center_h, center_w))
    if downsample:
        vol = _downsample_to_liver_shape(vol)
    vol = (vol - vol.mean()) / vol.std()
    
    if cache_dir is not None:
        print(f"[acdc_4d] saving preprocessed frame to cache: {cp}")
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        np.save(cp, vol.cpu().numpy())
    
    return vol


def _mapped_index(c: int | None, cur: int, target: int) -> int:
    """Where does native-space index `c` (already converted to resampled
    space) end up along a single axis after `_center_crop_or_pad` crops/pads
    `cur` -> `target`? Mirrors the exact clamping logic in
    `_center_crop_or_pad`, so the returned index is the TRUE post-crop
    location of `c` — not just an assumed `target // 2` — which matters
    whenever the reference slice sits close enough to a volume boundary that
    the crop window gets clamped."""
    if c is None:
        return target // 2
    if cur >= target:
        start = max(0, min(c - target // 2, cur - target))
        return c - start
    else:
        before = max(0, min(c, target - cur))
        return c + before
_REF_INDEX_CACHE: dict = {}   # (patient, data_dir) -> mapped canonical index


def get_reference_slice_index(patient: str, data_dir: str, orientation: dict | None) -> int:
    """
    D-axis index into the CANONICAL_SHAPE-cropped volume where THIS
    patient's saved reference slice (or centroid) actually lands, after the
    same resample -> crop chain `_load_and_preprocess` applies.

    Fast path: if `orientation[patient]["canonical_d_index"]` is already set
    (written by `acdc_preprocessed_reference_grid.py`'s
    `update_orientation_csv`), use it directly — no resampling needed.

    Slow path (legacy CSVs without that column): resample a tiny
    (native_D, 4, 4) dummy tensor through the real `_resample_to_spacing`
    call to get the true post-resample D-axis length, then map through
    `_mapped_index`. H/W=4 (not 1) because `F.interpolate`'s
    scale_factor path floors each axis's output size independently, and an
    H/W=1 dummy floors to 0 whenever the in-plane scale factor is < 1 (i.e.
    whenever native in-plane spacing < the 1.5mm target — common in ACDC),
    which F.interpolate then rejects outright even though only the D-axis
    output is used. This only ever runs once per (patient, data_dir), then
    is cached in `_REF_INDEX_CACHE`.
    """
    key = (patient, data_dir)
    if key in _REF_INDEX_CACHE:
        return _REF_INDEX_CACHE[key]

    entry = orientation.get(patient) if orientation else None
    if entry is None:
        idx = CANONICAL_SHAPE[0] // 2
        _REF_INDEX_CACHE[key] = idx
        return idx

    cached = entry.get("canonical_d_index")
    if cached is not None:
        _REF_INDEX_CACHE[key] = cached
        return cached

    spacing = _patient_info(patient, data_dir)["spacing"]

    patient_dir = os.path.join(data_dir, patient)
    native_d = nib.load(os.path.join(patient_dir, f"{patient}_4d.nii.gz")).shape[2]

    ref_slice = entry.get("reference_slice")
    centroid  = entry.get("centroid")

    center_d = None
    if centroid is not None:
        center_d, _, _ = _native_to_resampled_point(centroid, spacing)
    if ref_slice is not None:   # reference_slice wins over centroid's own D estimate
        center_d = _native_to_resampled_index(ref_slice, spacing[0])

    dummy = torch.zeros(native_d, 4, 4)
    resampled_d = _resample_to_spacing(dummy, spacing, mode="nearest").shape[0]

    idx = _mapped_index(center_d, resampled_d, CANONICAL_SHAPE[0])
    idx = max(0, min(idx, CANONICAL_SHAPE[0] - 1))

    _REF_INDEX_CACHE[key] = idx
    return idx

# ── Dataset ───────────────────────────────────────────────────────────────────

class ACDC_4D_Dataset(Dataset):
    """
    Cardiac analog of `NAVIGATOR_4D_Dataset_multitime_continuous` (navigator_4d.py:446-520):
    builds (input, output) frame-index windows cyclically over each patient's single
    cardiac cycle, using End-Diastole as the fixed reference pose.

    Unlike the liver dataset, frames are not pre-materialized as per-timepoint files —
    they are loaded lazily, on the fly, from each patient's `patientXXX_4d.nii.gz`
    via nibabel's memory-mapped `dataobj` (no full-volume load, no writes to disk).

    Returns (per item): ref_volume, input_volume_list, current_volume_list, vol_files
    — exactly the tuple shape `NAVIGATOR_4D_Dataset_multitime` returns, so the rest
    of the pipeline (compute_phase, build_Iseq, training loops, …) is reusable.
    """

    def __init__(self, root_dir, nb_inputs=3, nb_pred=1, sequence_list=(), valid=False, test=False,
                 downsample_to_liver=False, repeats=None, cache_dir=None, stride=1,
                 reference_slices_path=None):
        self.root_dir  = root_dir
        self.nb_inputs = nb_inputs
        self.nb_pred   = nb_pred
        self.test      = test
        self.valid     = valid
        self.downsample_to_liver = downsample_to_liver
        self.cache_dir = cache_dir
        self.stride    = stride
        self.orientation = load_orientation_info(reference_slices_path) \
            if reference_slices_path is not None else None
        print(f"[ACDC_4D_Dataset] downsample_to_liver={downsample_to_liver}  cache_dir={cache_dir}  "
              f"stride={stride}  orientation={'None' if self.orientation is None else f'{len(self.orientation)} patients'}")
        # Mirror liver's temp_navs defaults (80 train / 20 valid / 1 test)
        if repeats is None:
            if test:
                repeats = 1
            elif valid:
                repeats = 1
            else:
                repeats = 15
        self.repeats = repeats

        self.files_input   = []
        self.files_output  = []
        self.ref_vol_files = {}
        self._img_cache    = {}   # patient -> lazy nibabel image proxy

        patients = list(sequence_list) if len(sequence_list) else sorted(
            d for d in os.listdir(root_dir) if d.startswith("patient")
        )

        m, n = self.nb_inputs, self.nb_pred
        _MIN_NB_FRAME = 17  # patients with fewer frames have too sparse phase coverage
        for patient in patients:
            if patient == ".directory":
                continue
            patient_dir = os.path.join(root_dir, patient)
            info        = _patient_info(patient, root_dir)
            nb_frame    = info["NbFrame"]
            if nb_frame < _MIN_NB_FRAME:
                continue
            ed          = info["ED"]

            paths = [os.path.join(patient_dir, f"t_{t}.nii.gz") for t in range(nb_frame)]

            for t in range(nb_frame):
                img   = [paths[(t + i * stride) % nb_frame] for i in range(m)]
                label = [paths[(t + (m + j) * stride) % nb_frame] for j in range(n)]
                self.files_input.append(img)
                self.files_output.append(label)

            ref_vol_file = paths[ed]
            if not (self.test or self.valid):
                # Don't let the model be asked to move the reference volume during training.
                self.files_input = [
                    [ele for ele in sub if ele != ref_vol_file]
                    for sub in self.files_input
                ]
                indices = [i for i, v in enumerate(self.files_input) if len(v) != m]
                self.files_input  = [v for i, v in enumerate(self.files_input)  if i not in indices]
                self.files_output = [v for i, v in enumerate(self.files_output) if i not in indices]

            self.ref_vol_files[patient] = ref_vol_file

        self._base_len = len(self.files_input)

    def __len__(self):
        return self._base_len * self.repeats

    def _load_frame(self, virtual_path: str) -> torch.Tensor:
        """Load + preprocess one frame from its virtual `.../patientXXX/t_<idx>.nii.gz` path."""
        patient = virtual_path.split("/")[-2]
        t_idx   = int(os.path.basename(virtual_path)[2:-7])
        return _load_and_preprocess(self.root_dir, patient, t_idx, self._img_cache,
                                     downsample=self.downsample_to_liver,
                                     cache_dir=self.cache_dir,
                                     orientation=self.orientation)

    def _augment_sample(self, ref, inputs, outputs):
        """Intensity-only augmentation for ACDC cardiac data.

        Spatial transforms (flips, zoom) are disabled: flips invalidate the
        fixed conditioning slice index (cor_pos), causing the navigator encoder
        to see an inconsistent anatomical location and breaking phase conditioning.
        """
        def _intensity(v):
            v = v + torch.randn_like(v) * 0.05
            v = v + torch.empty(1).uniform_(-0.2, 0.2).item()
            v = v * torch.empty(1).uniform_(0.85, 1.15).item()
            return v

        ref     = _intensity(ref)
        inputs  = [_intensity(v) for v in inputs]
        outputs = [_intensity(v) for v in outputs]

        return ref, inputs, outputs

    def __getitem__(self, idx):
        idx = idx % self._base_len
        input_volume_list   = [self._load_frame(p) for p in self.files_input[idx]]
        current_volume_list = [self._load_frame(p) for p in self.files_output[idx]]

        patient    = self.files_input[idx][0].split("/")[-2]
        ref_volume = self._load_frame(self.ref_vol_files[patient])

        if not (self.test or self.valid):
            ref_volume, input_volume_list, current_volume_list = \
                self._augment_sample(ref_volume, input_volume_list, current_volume_list)

        return ref_volume, input_volume_list, current_volume_list, self.files_output[idx]


# ── Reference-volume pretraining dataset (RVNet MAE / SimMIM / SparK) ────

class RefVolume_Dataset_augment_ACDC(Dataset):
    """
    ACDC analog of `RefVolume_Dataset_augment` (navigator_4d.py): one item per
    patient, the End-Diastole (ED) frame as reference volume, with optional
    augmentation for RVNet pretraining.

    Returns (per item)
    -------------------
    ref_volume : (D, H, W) = (32, 128, 128) float32 tensor, min-max normalised to [0, 1]
    patient    : patient id (str)
    """

    def __init__(self, root_dir: str, sequence_list=(), repeats: int = 1, augment: bool = False,
                 downsample_to_liver: bool = False, cache_dir: str | None = None,
                 reference_slices_path: str | None = None):
        self.root_dir          = root_dir
        self.repeats           = repeats
        self.augment           = augment
        self.downsample_to_liver = downsample_to_liver
        self.cache_dir         = cache_dir
        self._img_cache: dict  = {}
        self.orientation       = load_orientation_info(reference_slices_path) \
            if reference_slices_path is not None else None

        self.items = [p for p in sequence_list if p != ".directory"]

    def __len__(self) -> int:
        return len(self.items) * self.repeats

    def __getitem__(self, idx: int):
        patient = self.items[idx % len(self.items)]
        ed      = _patient_info(patient, self.root_dir)["ED"]

        vol = _load_and_preprocess(self.root_dir, patient, ed, self._img_cache,
                                   downsample=self.downsample_to_liver,
                                   cache_dir=self.cache_dir,
                                   orientation=self.orientation)

        if self.augment:
            vol = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W) for grid_sample compat
            vol = self._augment(vol)
            vol = vol.squeeze(0).squeeze(0)      # back to (D, H, W)

        return vol, patient

    @torch.no_grad()
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, 1, D, H, W). Mirrors RefVolume_Dataset_augment._augment (navigator_4d.py)."""
        # --- Random flips ------------------------------------------------
        for dim in (2, 3, 4):
            if torch.rand(1).item() < 0.5:
                x = x.flip(dim)

        # --- Additive Gaussian noise --------------------------------------
        x = x + torch.randn_like(x) * 0.05

        # --- Random brightness shift ----------------------------------------
        x = x + torch.empty(1).uniform_(-0.2, 0.2).item()

        # --- Random contrast scaling ------------------------------------------
        x = x * torch.empty(1).uniform_(0.85, 1.15).item()

        # --- Random zoom ±10% --------------------------------------------------
        scale = torch.empty(1).uniform_(0.9, 1.1).item()
        theta = torch.eye(3, 4, device=x.device).unsqueeze(0) * scale
        grid  = F.affine_grid(theta, x.shape, align_corners=False)
        x     = F.grid_sample(x, grid, mode="bilinear",
                               padding_mode="border", align_corners=False)

        return x


# ── Cache builder ─────────────────────────────────────────────────────────────

def build_acdc_cache(root_dir: str, cache_dir: str, downsample: bool = False,
                      reference_slices_path: str | None = None) -> None:
    """Pre-compute every patient frame and save as float32 .npy arrays.

    Run once before training; then pass cache_dir to ACDC_4D_Dataset /
    RefVolume_Dataset_augment_ACDC (or set it in the YAML config) to skip
    the per-sample gzip decompress + trilinear resample (~300x faster).

    reference_slices_path: optional path to the combined .csv (or legacy .txt)
    from `acdc_reference_slice.py`. When given, cached volumes are cropped
    around each patient's reference slice / heart centroid, and rotated,
    instead of using the geometric center — pass the SAME path to
    ACDC_4D_Dataset/RefVolume_Dataset_augment_ACDC's reference_slices_path so
    the cache actually gets used (the cache key differs between the crop
    modes, see `_cache_path`).

    Usage
    -----
    python -m 4D_MoPred_liver.data.data_loaders.acdc_4d \\
        --data_dir /volatile/data/cam/ACDC/database \\
        --cache_dir /volatile/data/cam/ACDC/cache_liver \\
        --downsample \\
        --reference_slices /path/to/reference_slices.csv
    """
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **_):
            return it

    orientation = load_orientation_info(reference_slices_path) \
        if reference_slices_path is not None else None

    patients = sorted(d for d in os.listdir(root_dir) if d.startswith("patient"))
    img_cache: dict = {}
    n_written = 0

    for patient in _tqdm(patients, desc="build_acdc_cache"):
        info = _patient_info(patient, root_dir)
        patient_cache = os.path.join(cache_dir, patient)
        os.makedirs(patient_cache, exist_ok=True)
        entry = orientation.get(patient) if orientation is not None else None
        for t_idx in range(info["NbFrame"]):
            cp = _cache_path(cache_dir, patient, t_idx, downsample, entry)
            if os.path.exists(cp):
                continue
            vol = _load_and_preprocess(root_dir, patient, t_idx, img_cache, downsample=downsample,
                                        orientation=orientation)
            np.save(cp, vol.numpy())
            n_written += 1

    shape_str = f"LIVER_SHAPE{LIVER_SHAPE}" if downsample else f"CANONICAL_SHAPE{CANONICAL_SHAPE}"
    print(f"build_acdc_cache: wrote {n_written} frames at {shape_str} → {cache_dir}")

if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Pre-compute ACDC frame cache (run once before training).")
    _p.add_argument("--data_dir",  required=True, help="ACDC database root")
    _p.add_argument("--cache_dir", required=True, help="Directory to write .npy files into")
    _p.add_argument("--downsample", action="store_true",
                    help="Cache at LIVER_SHAPE (32,64,64); omit for CANONICAL_SHAPE (32,128,128)")
    _p.add_argument("--reference_slices", default=None,
                    help="Path to reference_slices.csv/.txt from acdc_reference_slice.py — "
                         "when set, crops are centered on each patient's reference slice / "
                         "heart centroid, and rotated to a canonical orientation.")
    _a = _p.parse_args()
    build_acdc_cache(_a.data_dir, _a.cache_dir, downsample=_a.downsample,
                      reference_slices_path=_a.reference_slices)