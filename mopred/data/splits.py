"""
Patient-level cross-validation split functions.
``make_folds_3fold`` produces train/val/test splits for the liver navigator dataset;
``make_folds_acdc`` does the same for the ACDC cardiac dataset.
"""
from __future__ import annotations

import os
import random
from sklearn.model_selection import KFold
import numpy as np


def make_folds_originalversion():
    case_list = [
        "CoMoDo01b",
        "CoMoDo02",
        "CoMoDo03",
        "CoMoDo04",
        "CoMoDo05",
        "CoMoDo06",
        "CoMoDo08b",
        "CoMoDo09",
        "CoMoDo10",
        "CoMoDo11",
        "CoMoDo12",
        "CoMoDo13",
        "CoMoDo15",
        "CoMoDo16",
        "CoMoDo17",
        "CoMoDo18",
        "CoMoDo19",
        "CoMoDo20",
        "CoMoDo21",
        "CoMoDo22",
        "CoMoDo24",
        "CoMoDo25",
        "CoMoDo26",
        "CoMoDo27",
        "CoMoDo28",
    ]

    kf = KFold(n_splits=24, shuffle=True, random_state=123)  # 8
    kf.get_n_splits(case_list)
    out = kf.split(case_list)
    train = []
    valid = []
    test = []
    random.seed(123)
    for train_idx, test_idx in out:
        train.append(np.take(case_list, train_idx))
        valid_idx = random.sample(list(train_idx), 1)  # 2
        valid.append(np.take(case_list, valid_idx))
        train[-1] = np.delete(train[-1], valid_idx)
        test.append(np.take(case_list, test_idx))
        
    return train, valid, test


def make_folds_3fold(n_splits=3, ratio_train=6, val_ratio=0.1, base_seed=123):
    case_list = np.array([
        "CoMoDo01b", "CoMoDo02", "CoMoDo03", "CoMoDo04", "CoMoDo05",
        "CoMoDo06", "CoMoDo08b", "CoMoDo09", "CoMoDo10", "CoMoDo11",
        "CoMoDo12", "CoMoDo13", "CoMoDo15", "CoMoDo16", "CoMoDo17",
        "CoMoDo18", "CoMoDo19", "CoMoDo20", "CoMoDo21", "CoMoDo22",
        "CoMoDo24", "CoMoDo25", "CoMoDo26", "CoMoDo27", "CoMoDo28",
    ])

    kf = KFold(n_splits=ratio_train, shuffle=True, random_state=base_seed)

    train, valid, test = [], [], []
    for i, (train_idx, test_val_idx) in enumerate(kf.split(case_list)):
        train.append(case_list[train_idx])

        n_val = len(test_val_idx) // 2
        rng = np.random.RandomState(base_seed + i)
        val_idx = rng.choice(test_val_idx, size=n_val, replace=False)
        valid.append(case_list[val_idx])

        # Train: remainder
        test_idx = np.setdiff1d(test_val_idx, val_idx)
        test.append(case_list[test_idx])
        
    # print(valid)    
    train_split= train[:n_splits-1] + [train[n_splits+2]]
    valid_split= valid[:n_splits-1] + [valid[n_splits+2]]
    test_split= test[:n_splits-1] + [test[n_splits+2]]
    # print(valid_split)
    
    return train_split, valid_split, test_split
    # return train[:n_splits], valid[:n_splits], test[:n_splits]


def _acdc_nb_frames(patient: str, data_dir: str) -> int:
    cfg_path = os.path.join(data_dir, patient, "Info.cfg")
    try:
        for line in open(cfg_path):
            if line.startswith("NbFrame:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def make_folds_acdc(n_splits=3, ratio_train=7, val_ratio=0.15, base_seed=123,
                    data_dir="/volatile/data/ACDC/database/full_dataset",
                    patient_subset=None, min_nb_frames=17):
    """KFold-based fold logic applied to the ACDC patientXXX IDs.

    Patients with NbFrame < min_nb_frames are excluded before splitting.
    KFold(ratio_train) defines the test fold (~1/ratio_train of total).
    val_ratio * total patients are drawn from the remaining train fold for val.
    With defaults ratio_train=7, val_ratio=0.15: ~71/15/14% split on 128 patients.
    """
    case_list = np.array(sorted(
        d for d in os.listdir(data_dir) if d.startswith("patient")
    ))

    # Exclude patients with too few frames (same threshold as acdc_4d.py _MIN_NB_FRAME)
    case_list = np.array([
        p for p in case_list if _acdc_nb_frames(p, data_dir) >= min_nb_frames
    ])

    if patient_subset is not None:
        subset_set = set(patient_subset)
        case_list = np.array([p for p in case_list if p in subset_set])

    n_val = round(len(case_list) * val_ratio)
    kf = KFold(n_splits=ratio_train, shuffle=True, random_state=base_seed)

    train, valid, test = [], [], []
    for i, (train_idx, test_idx) in enumerate(kf.split(case_list)):
        rng = np.random.RandomState(base_seed + i)
        val_idx = rng.choice(train_idx, size=n_val, replace=False)
        actual_train_idx = np.setdiff1d(train_idx, val_idx)
        train.append(case_list[actual_train_idx])
        valid.append(case_list[val_idx])
        test.append(case_list[test_idx])
    return train[:n_splits], valid[:n_splits], test[:n_splits]
