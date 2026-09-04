"""
Utilities for reassembling per-ROI NIfTI prediction volumes into full-FOV outputs.
"""
import numpy as np
import nibabel as nib
import os
from os import listdir
from os.path import isfile, join


def assemble_roi_volumes(path_volumes, path_save, target_imgs=False):

    onlyfiles = [f for f in listdir(path_volumes) if isfile(join(path_volumes, f))]
    if ".directory" in onlyfiles:
        onlyfiles.remove(".directory")

    if target_imgs:
        onlyfiles.sort(key=lambda x: int(x[2:-7]))
    else:
        onlyfiles.sort()

    all_V = np.zeros((4, 8, 8, len(onlyfiles)))

    for f in range(len(onlyfiles)):
        if "DVF" in (path_volumes + onlyfiles[f]):
            continue
        else:
            V = nib.load(path_volumes + onlyfiles[f]).get_fdata()
            all_V[:, :, :, f] = V

    all_V = nib.Nifti1Image(
        all_V,
        affine=[[3.5, 0, 0, 0], [0, 1.7 * 2, 0, 0], [0, 0, 1.7 * 2, 0], [0, 0, 0, 1]],
    )
    nib.save(all_V, path_save)


def assemble_volumes(path_volumes, path_save, target_imgs=False, downsampled=False):

    onlyfiles = [f for f in listdir(path_volumes) if isfile(join(path_volumes, f))]
    if ".directory" in onlyfiles:
        onlyfiles.remove(".directory")
    if target_imgs:
        onlyfiles.sort(key=lambda x: int(x[2:-7]))
    else:
        onlyfiles.sort()

    if downsampled:
        all_V = np.zeros((32, 64, 64, len(onlyfiles)))
    else:
        all_V = np.zeros((32, 128, 128, len(onlyfiles)))

    for f in range(len(onlyfiles)):
        V = nib.load(path_volumes + onlyfiles[f]).get_fdata()
        all_V[:, :, :, f] = V

    if downsampled:
        all_V = nib.Nifti1Image(
            all_V,
            affine=[
                [3.5, 0, 0, 0],
                [0, 1.7 * 2, 0, 0],
                [0, 0, 1.7 * 2, 0],
                [0, 0, 0, 1],
            ],
        )
    else:
        all_V = nib.Nifti1Image(
            all_V, affine=[[3.5, 0, 0, 0], [0, 1.7, 0, 0], [0, 0, 1.7, 0], [0, 0, 0, 1]]
        )
    nib.save(all_V, path_save)


if __name__ == "__main__":

    path = "/home/liset/PycharmProjects/AE_Nav/VM-AE_sc/04_25/s2_new_tmnet/test/09_07/22.31.16_18.48.31_Nav_dataset_new_tmnet_s2_test_042_5_model_best.pth_Nav_dataset/tracking/"
    path_volumes = "CoMoDo25/"
    path_save = path_volumes[:-1] + ".nii.gz"
    assemble_volumes(
        path + path_volumes, path + path_save, target_imgs=False, downsampled=True
    )
