"""
PyTorch Dataset for the liver 4D navigator dataset.
Loads multi-timepoint DVF sequences and 2D navigator slices, with support for
reference-patient filtering, temporal sub-sampling, and on-disk DVF caching.
"""
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

import hashlib

def _vol_key(filepath: str) -> str:
    """Stable cache key: SHA1 of the absolute path."""
    return hashlib.sha1(os.path.abspath(filepath).encode()).hexdigest()


# ── Respiratory phase computation ────────────────────────────────────────────

CYCLE_LEN = 31
Y_NAV     = 32   # coronal slice index used by the navigator (condi_type "2")

_PHI_CACHE: dict = {}    # (patient, data_dir) → phi array  (CYCLE_LEN,)
_SIG_CACHE: dict = {}    # (patient, data_dir) → raw signal array (CYCLE_LEN,)


def _compute_raw_signals(patient: str, data_dir: str) -> np.ndarray:
    """
    Compute mean|cor_t − cor_exhale| for all CYCLE_LEN frames.
    Un-normalised navigator signal shared by get_phi and get_amplitude.
    """
    key = (patient, data_dir)
    if key in _SIG_CACHE:
        return _SIG_CACHE[key]

    _, exh  = _REF_ID[patient]
    pat_dir = os.path.join(data_dir, patient)
    ref_path = os.path.join(pat_dir, f"t_{exh}.nii.gz")
    if not os.path.isfile(ref_path):
        _SIG_CACHE[key] = np.zeros(CYCLE_LEN)
        return _SIG_CACHE[key]

    ref_vol = nib.load(ref_path).get_fdata()[:, ::2, ::2]
    y       = min(Y_NAV, ref_vol.shape[1] - 1)
    ref_cor = ref_vol[:, y, :].astype(np.float32)

    signals = np.zeros(CYCLE_LEN)
    for t in range(CYCLE_LEN):
        path = os.path.join(pat_dir, f"t_{t}.nii.gz")
        if not os.path.isfile(path):
            continue
        vol        = nib.load(path).get_fdata()[:, ::2, ::2]
        signals[t] = np.abs(vol[:, y, :].astype(np.float32) - ref_cor).mean()

    _SIG_CACHE[key] = signals
    return signals


def _compute_patient_phis(patient: str, data_dir: str) -> np.ndarray:
    """
    Compute phi ∈ [0, 0.5] for all CYCLE_LEN frames.

    phi = 0 at exhale, phi = 0.5 at inhale.
    Normalised by each patient's peak so it only encodes phase, not amplitude.
    """
    inh, _     = _REF_ID[patient]
    signals    = _compute_raw_signals(patient, data_dir)
    sig_at_inh = signals[inh]
    if sig_at_inh < 1e-6:
        return np.zeros(CYCLE_LEN)
    return np.clip(signals / sig_at_inh, 0.0, 1.0) * 0.5


def get_phi(patient: str, t_idx: int, data_dir: str) -> float:
    """Return phi ∈ [0, 0.5] for (patient, frame t_idx), cached per patient."""
    key = (patient, data_dir)
    if key not in _PHI_CACHE:
        _PHI_CACHE[key] = _compute_patient_phis(patient, data_dir)
    return float(_PHI_CACHE[key][t_idx % CYCLE_LEN])


# def get_cycle_phase(patient: str, t_idx: int, data_dir: str) -> float:
#     inh_idx, exh_idx = _REF_ID[patient]
#     phi_val = get_phi(patient, t_idx, data_dir)

#     t        = t_idx % CYCLE_LEN
#     rel_t    = (t - exh_idx) % CYCLE_LEN          # distance forward from exhale
#     rel_inh  = (inh_idx - exh_idx) % CYCLE_LEN    # distance from exhale to inhale

#     if rel_t <= rel_inh:
#         return phi_val          # rising: exhale → inhale
#     else:
#         return 1.0 - phi_val    # falling: inhale → exhale
    
def get_cycle_phase(patient: str, t_idx: int, data_dir: str) -> float:
    inh_idx = _REF_ID[patient][0]
    phi_val  = get_phi(patient, t_idx, data_dir)
    if t_idx % CYCLE_LEN <= inh_idx:
        return phi_val
    else:
        return 1.0 - phi_val

def get_amplitude(patient: str, t_idx: int, data_dir: str) -> float:
    """
    Return raw diaphragm displacement (mean |cor_t − cor_exhale|) for frame t_idx.

    Unlike phi, this is NOT normalised by the patient's peak, so values are
    comparable across patients and encode both phase and breathing depth.
    """
    return float(_compute_raw_signals(patient, data_dir)[t_idx % CYCLE_LEN])


def get_peak_amplitude(patient: str, data_dir: str) -> float:
    """Return the peak diaphragm displacement (signal at the inhale frame)."""
    inh, _ = _REF_ID[patient]
    return float(_compute_raw_signals(patient, data_dir)[inh])


# Inhale / exhale reference phase indices per patient (shared by both dataset classes).
_REF_ID = {
    "CoMoDo01b": (4,  9),
    "CoMoDo02":  (19, 26),
    "CoMoDo03":  (12, 7),
    "CoMoDo04":  (24, 2),
    "CoMoDo05":  (7,  1),
    "CoMoDo06":  (22, 9),
    "CoMoDo08b": (10, 19),
    "CoMoDo09":  (6,  18),
    "CoMoDo10":  (27, 10),
    "CoMoDo11":  (15, 0),
    "CoMoDo12":  (4,  23),
    "CoMoDo13":  (7,  14),
    "CoMoDo15":  (8,  23),
    "CoMoDo16":  (4,  20),
    "CoMoDo17":  (18, 3),
    "CoMoDo18":  (9,  18),
    "CoMoDo19":  (11, 22),
    "CoMoDo20":  (17, 13),
    "CoMoDo21":  (10, 5),
    "CoMoDo22":  (8,  18),
    "CoMoDo24":  (4,  12),
    "CoMoDo25":  (14, 19),
    "CoMoDo26":  (12, 6),
    "CoMoDo27":  (9,  5),
    "CoMoDo28":  (6,  19),
}


class RefVolume_Dataset(Dataset):
    """
    Minimal dataset for reference-volume pretraining (RVNet MAE / SimMIM / SparK).

    One logical item per patient/sequence.  The ``repeats`` parameter inflates
    the dataset length so that each epoch contains ``repeats`` passes over each
    volume with independent random masks — matching the training dynamics of the
    full NAVIGATOR_4D_Dataset_multitime without loading any navigators or DVFs.

    Uses the same exhale-phase reference volume as NAVIGATOR_4D_Dataset_multitime
    and the same (D, H, W) preprocessing (trilinear ×0.5 in H and W, z-score norm).

    Parameters
    ----------
    root_dir      : root data directory (same as NAVIGATOR_4D_Dataset_multitime)
    sequence_list : patient case names, typically one split from make_folds_3fold()
    repeats       : how many times each volume appears per epoch (default 1).
                    Recommended: 80 for train, 20 for val (mirrors temp_navs).

    Returns (per item)
    ------------------
    ref_volume : (D, H, W) float32 tensor, z-score normalised
    sequence   : patient name (str)
    """

    def __init__(self, root_dir: str, sequence_list=(), repeats: int = 1):
        self.root_dir = root_dir
        self.repeats  = repeats
        self.items: list[tuple[str, str]] = []   # (ref_vol_path, sequence_name)

        for sequence in sequence_list:
            if sequence == ".directory":
                continue
            data_dir = os.path.join(root_dir, sequence)
            _, exh   = _REF_ID[sequence]          # always use exhale as reference
            # Resolve the filename — sort by numeric index to be safe
            vol_files = sorted(
                os.listdir(data_dir),
                key=lambda x: int(x[2:-7]),
            )
            ref_file = next(f for f in vol_files if f.endswith(f"t_{exh}.nii.gz"))
            self.items.append((os.path.join(data_dir, ref_file), sequence))

    def __len__(self) -> int:
        return len(self.items) * self.repeats

    def __getitem__(self, idx: int):
        path, seq = self.items[idx % len(self.items)]
        vol = nib.load(path).get_fdata()
        vol = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
        vol = F.interpolate(vol, scale_factor=[1, 0.5, 0.5], mode="trilinear").squeeze()
        vol = (vol - vol.mean()) / vol.std()
        return vol, seq

class RefVolume_Dataset_augment(Dataset):
    """
    Minimal dataset for reference-volume pretraining (RVNet MAE / SimMIM / SparK).
    ...
    Parameters
    ----------
    augment : whether to apply random augmentations (True for train, False for val/test)
    """

    def __init__(self, root_dir: str, sequence_list=(), repeats: int = 1, augment: bool = False):
        self.root_dir = root_dir
        self.repeats  = repeats
        self.augment  = augment
        self.items: list[tuple[str, str]] = []

        for sequence in sequence_list:
            if sequence == ".directory":
                continue
            data_dir = os.path.join(root_dir, sequence)
            _, exh   = _REF_ID[sequence]
            vol_files = sorted(
                os.listdir(data_dir),
                key=lambda x: int(x[2:-7]),
            )
            ref_file = next(f for f in vol_files if f.endswith(f"t_{exh}.nii.gz"))
            self.items.append((os.path.join(data_dir, ref_file), sequence))

    def __len__(self) -> int:
        return len(self.items) * self.repeats

    def __getitem__(self, idx: int):
        path, seq = self.items[idx % len(self.items)]

        # --- load & preprocess (unchanged) -----------------------------------
        vol = nib.load(path).get_fdata()
        vol = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        vol = F.interpolate(vol, scale_factor=[1, 0.5, 0.5], mode="trilinear").squeeze()
        vol = (vol - vol.mean()) / vol.std()  # (D, H, W)

        # --- augmentation (train only) ---------------------------------------
        if self.augment:
            vol = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W) for grid_sample compat
            vol = self._augment(vol)
            vol = vol.squeeze(0).squeeze(0)      # back to (D, H, W)

        return vol, seq

    @torch.no_grad()
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, 1, D, H, W) — all ops in-place safe, no gradients needed.

        NOTE: z-score normalisation has already been applied, so the volume
        lives roughly in [-3, 3].  Brightness/gamma are adapted accordingly.
        """
        D, H, W = x.shape[2:]

        # --- Random flips ----------------------------------------------------
        # H and W flips are anatomically safe for liver MRI.
        # D (axial) flip is also safe — liver position is learned from context.
        for dim in (2, 3, 4):
            if torch.rand(1).item() < 0.5:
                x = x.flip(dim)

        # --- Additive Gaussian noise -----------------------------------------
        # std=0.05 is ~5% of a typical z-scored signal range
        x = x + torch.randn_like(x) * 0.05

        # --- Random brightness shift (after z-score, so shift in std units) --
        x = x + torch.empty(1).uniform_(-0.2, 0.2).item()

        # --- Random contrast scaling (multiplicative, around 0) --------------
        # Equivalent to gamma on z-scored data: stretches/compresses distribution
        x = x * torch.empty(1).uniform_(0.85, 1.15).item()

        # --- Random zoom ±10% (affine grid, no extra deps) -------------------
        scale = torch.empty(1).uniform_(0.9, 1.1).item()
        theta = torch.eye(3, 4, device=x.device).unsqueeze(0) * scale
        grid  = F.affine_grid(theta, x.shape, align_corners=False)
        x     = F.grid_sample(x, grid, mode="bilinear",
                               padding_mode="border", align_corners=False)

        return x

@torch.no_grad()
def _sample_spatial_aug_params():
    """Shared flip/zoom params for one registration pair (sampled once, applied
    identically to every volume in the pair so the correspondence between them
    -- the thing a registration network is meant to learn -- is preserved)."""
    flips = [dim for dim in (2, 3, 4) if torch.rand(1).item() < 0.5]
    scale = torch.empty(1).uniform_(0.9, 1.1).item()
    return flips, scale


@torch.no_grad()
def _apply_spatial_aug(x: torch.Tensor, flips: list, scale: float) -> torch.Tensor:
    """x: (1, 1, D, H, W). Applies the given (shared) flip + zoom."""
    for dim in flips:
        x = x.flip(dim)
    theta = torch.eye(3, 4, device=x.device).unsqueeze(0) * scale
    grid  = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


@torch.no_grad()
def _apply_intensity_aug(x: torch.Tensor) -> torch.Tensor:
    """x: (1, 1, D, H, W), already z-scored. Independent per-volume noise/
    brightness/contrast jitter -- safe to differ between volumes in a
    registration pair since it doesn't move anatomy."""
    x = x + torch.randn_like(x) * 0.05
    x = x + torch.empty(1).uniform_(-0.2, 0.2).item()
    x = x * torch.empty(1).uniform_(0.85, 1.15).item()
    return x


class NAVIGATOR_4D_Dataset_multitime(Dataset):

    def __init__(
        self,
        root_dir,
        nb_inputs=3,
        nb_pred=1,
        sequence_list=(),
        valid=False,
        test=False,
        augment=False,
    ):
        self.root_dir = root_dir
        self.nb_inputs = nb_inputs
        self.nb_pred = nb_pred
        self.augment = augment
        if len(sequence_list) != 0:
            sequences_dir = sequence_list
        else:
            sequences_dir = os.listdir(self.root_dir)
        self.files_input = list()
        self.files_output = list()
        self.ref_vol_files = {}
        self.exhale_as_reference = True
        self.test = test
        self.valid = valid
        self.ref_id = {
            "CoMoDo01b": (4, 9),  # B (inhale, exhale)
            "CoMoDo02": (19, 26),  # B
            "CoMoDo03": (12, 7),  # B
            "CoMoDo04": (24, 2),  # B
            "CoMoDo05": (7, 1),  # B
            "CoMoDo06": (22, 9),  # B
            "CoMoDo08b": (10, 19),  # B-
            "CoMoDo09": (6, 18),  # -----------
            "CoMoDo10": (27, 10),  # R # -----------
            "CoMoDo11": (15, 0),  # B
            "CoMoDo12": (4, 23),  # B-
            "CoMoDo13": (7, 14),  # B
            "CoMoDo15": (8, 23),  # B
            "CoMoDo16": (4, 20),  # B
            "CoMoDo17": (18, 3),  # B
            "CoMoDo18": (9, 18),  # B
            "CoMoDo19": (11, 22),  # B
            "CoMoDo20": (17, 13),  # B
            "CoMoDo21": (10, 5),  # B
            "CoMoDo22": (8, 18),  # B
            "CoMoDo24": (4, 12),  # B
            "CoMoDo25": (14, 19),  # R # -----------
            "CoMoDo26": (12, 6),  # B
            "CoMoDo27": (9, 5),  # B
            "CoMoDo28": (6, 19),
        }  # B

        if self.test:
            temp_navs = 2  # 20 for vessel traj
        elif self.valid:
            temp_navs = 20
        else:
            temp_navs = 80

        for sequence in sequences_dir:
            if sequence == ".directory":
                continue
            data_dir = os.path.join(self.root_dir, sequence)

            for z in range(temp_navs):
                for t in range(31 - (self.nb_inputs + self.nb_pred)):  # 20
                    img = list()
                    label = list()
                    for i in range(self.nb_inputs):
                        img.append(data_dir + "/t_" + str((31 * z + t) + i) + ".nii.gz")
                    for p in range(1, nb_pred + 1):
                        label.append(
                            data_dir
                            + "/t_"
                            + str((31 * z + t) + (self.nb_inputs - 1) + p)
                            + ".nii.gz"
                        )
                    self.files_input.append(img)
                    self.files_output.append(label)

            volume_files = os.listdir(data_dir)
            volume_files.sort(key=lambda x: int(x[2:-7]))
            inh, exh = self.ref_id[sequence]

            if self.exhale_as_reference:
                ref_phase = exh
            else:
                ref_phase = inh

            ref_vol_file = [
                file
                for file in volume_files
                if file.endswith("t_" + str(ref_phase) + ".nii.gz")
            ][0]

            ref_vol_path = os.path.join(data_dir, ref_vol_file)
            if (
                not self.test
            ):  # When testing we want to make sure that the model does not move the reference volume
                self.files_input = [
                    [ele for ele in sub if ele != ref_vol_path] for sub in self.files_input
                ]
                indices = []
                for ind, value in enumerate(self.files_input):
                    if len(value) != self.nb_inputs:
                        indices.append(ind)
                self.files_input = [
                    i for i in self.files_input if len(i) == self.nb_inputs
                ]
                self.files_output = [
                    j for (i, j) in enumerate(self.files_output) if i not in indices
                ]

            self.ref_vol_files[sequence] = ref_vol_path
            # self.files_input=self.files_input[:20]
            # self.files_output=self.files_output[:20]

    def __len__(self):
        return len(self.files_input)

    def __getitem__(self, idx):
        # Load input volumes
        input_volume_list = list()
        for vol_file in self.files_input[idx]:
            input_volume = nib.load(vol_file).get_fdata()
            input_volume = (
                (torch.from_numpy(input_volume)).float().unsqueeze(0).unsqueeze(0)
            )
            input_volume = F.interpolate(
                input_volume, scale_factor=[1, 0.5, 0.5], mode="trilinear"
            ).squeeze()
            input_volume = (input_volume - torch.mean(input_volume)) / torch.std(
                input_volume
            )
            input_volume_list.append(input_volume)

        # Load output volume
        output_volume_list = list()
        for vol_file in self.files_output[idx]:
            output_volume = nib.load(vol_file).get_fdata()
            output_volume = (
                (torch.from_numpy(output_volume)).float().unsqueeze(0).unsqueeze(0)
            )
            output_volume = F.interpolate(
                output_volume, scale_factor=[1, 0.5, 0.5], mode="trilinear"
            ).squeeze()
            output_volume = (output_volume - torch.mean(output_volume)) / torch.std(
                output_volume
            )
            output_volume_list.append(output_volume)

        # Load reference volume
        ref_vol_file = self.ref_vol_files[self.files_input[idx][0].split("/")[-2]]
        ref_volume = nib.load(ref_vol_file).get_fdata()
        ref_volume = (torch.from_numpy(ref_volume)).float().unsqueeze(0).unsqueeze(0)
        ref_volume = F.interpolate(
            ref_volume, scale_factor=[1, 0.5, 0.5], mode="trilinear"
        ).squeeze()
        ref_volume = (ref_volume - torch.mean(ref_volume)) / torch.std(ref_volume)

        if self.augment:
            flips, scale = _sample_spatial_aug_params()

            def _aug_one(v: torch.Tensor) -> torch.Tensor:
                v = v.unsqueeze(0).unsqueeze(0)
                v = _apply_spatial_aug(v, flips, scale)
                v = _apply_intensity_aug(v)
                return v.squeeze(0).squeeze(0)

            ref_volume         = _aug_one(ref_volume)
            input_volume_list  = [_aug_one(v) for v in input_volume_list]
            output_volume_list = [_aug_one(v) for v in output_volume_list]

        return ref_volume, input_volume_list, output_volume_list, self.files_output[idx]


class NAVIGATOR_4D_Dataset_multitime_continuous(Dataset):

    def __init__(
        self,
        root_dir,
        nb_inputs=3,
        nb_pred=3,
        sequence_list=(),
        valid=False,
        test=False,
    ):
        self.root_dir = root_dir
        self.nb_inputs = nb_inputs
        self.nb_pred = nb_pred
        self.test = test
        self.valid = valid
        self.files_input = []
        self.files_output = []
        self.ref_vol_files = {}

        patients = sequence_list if len(sequence_list) != 0 else os.listdir(self.root_dir)

        for patient in patients:
            if patient == ".directory":
                continue
            data_dir = os.path.join(self.root_dir, patient)

            # All t_X.nii.gz frames sorted by index
            all_files = sorted(
                [f for f in os.listdir(data_dir)
                 if f.startswith("t_") and f.endswith(".nii.gz")],
                key=lambda x: int(x[2:-7]),
            )
            n_frames = len(all_files)
            paths = [os.path.join(data_dir, f) for f in all_files]

            m, n = self.nb_inputs, self.nb_pred

            for t in range(n_frames):
                img   = [paths[(t + i) % n_frames] for i in range(m)]
                label = [paths[(t + m + j) % n_frames] for j in range(n)]
                self.files_input.append(img)
                self.files_output.append(label)

            _, exh = _REF_ID[patient]
            ref_vol_file = paths[exh]

            if not (self.test or self.valid):
                self.files_input = [
                    [ele for ele in sub if ele != ref_vol_file]
                    for sub in self.files_input
                ]
                indices = [i for i, v in enumerate(self.files_input) if len(v) != m]
                self.files_input  = [v for i, v in enumerate(self.files_input)  if i not in indices]
                self.files_output = [v for i, v in enumerate(self.files_output) if i not in indices]

            self.ref_vol_files[patient] = ref_vol_file

    def __len__(self):
        return len(self.files_input)

    def __getitem__(self, idx):
        def _load(path):
            vol = nib.load(path).get_fdata()
            vol = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
            vol = F.interpolate(vol, scale_factor=[1, 0.5, 0.5], mode="trilinear").squeeze()
            return (vol - vol.mean()) / vol.std()

        input_volume_list  = [_load(f) for f in self.files_input[idx]]
        output_volume_list = [_load(f) for f in self.files_output[idx]]

        patient = self.files_input[idx][0].split("/")[-2]
        ref_volume = _load(self.ref_vol_files[patient])

        return ref_volume, input_volume_list, output_volume_list, self.files_output[idx]

class CachedDVF_Dataset(Dataset):
    """
    Wraps NAVIGATOR_4D_Dataset_multitime and attaches pre-computed DVFs.
    Returns:
        ref_volume, input_volume_list, current_volume_list,
        dvf_list (List[Tensor] shape (3,D,H,W)), vol_files
    """

    def __init__(self, base_dataset: NAVIGATOR_4D_Dataset_multitime, cache_dir: str):
        self.base    = base_dataset
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        ref_vol, input_vols, current_vols, vol_files = self.base[idx]

        dvf_list = []
        for filepath in vol_files:
            cache_path = os.path.join(self.cache_dir, _vol_key(filepath) + ".npy")
            arr = np.load(cache_path).astype(np.float32)
            # stored as (1,3,D,H,W) → squeeze batch dim for collation
            dvf_list.append(torch.from_numpy(arr).squeeze(0))

        return ref_vol, input_vols, current_vols, dvf_list, vol_files
