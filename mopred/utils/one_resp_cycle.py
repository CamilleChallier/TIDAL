"""
Utilities for extracting and visualising one respiratory cycle from 4D liver volumes.
Applies per-patient spatial crops (``crop_record``) and saves individual NIfTI frames.
"""
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# --------------------------------------------
crop_record = {
    "CoMoDo01b": [(0, 0 + 128), (20, 20 + 128)],
    "CoMoDo02": [(0, 0 + 128), (20, 20 + 128)],
    "CoMoDo03": [(0, 0 + 128), (25, 25 + 128)],
    "CoMoDo04": [(5, 5 + 128), (20, 20 + 128)],
    "CoMoDo05": [(0, 0 + 128), (25, 25 + 128)],
    "CoMoDo06": [(0, 0 + 128), (15, 15 + 128)],
    "CoMoDo08b": [(0, 0 + 128), (22, 22 + 128)],
    "CoMoDo09": [(13, 13 + 128), (20, 20 + 128)],
    "CoMoDo10": [(0, 0 + 128), (15, 15 + 128)],
    "CoMoDo11": [(0, 0 + 128), (25, 25 + 128)],
    "CoMoDo12": [(0, 0 + 128), (20, 20 + 128)],
    "CoMoDo13": [(0, 0 + 128), (13, 13 + 128)],
    "CoMoDo15": [(5, 5 + 128), (22, 22 + 128)],
    "CoMoDo16": [(0, 0 + 128), (20, 20 + 128)],
    "CoMoDo17": [(0, 0 + 128), (22, 22 + 128)],
    "CoMoDo18": [(0, 0 + 128), (23, 23 + 128)],
    "CoMoDo19": [(0, 0 + 128), (22, 22 + 128)],
    "CoMoDo20": [(0, 0 + 128), (20, 20 + 128)],
    "CoMoDo21": [(0, 0 + 128), (15, 15 + 128)],
    "CoMoDo22": [(33, 33 + 128), (20, 20 + 128)],
    "CoMoDo24": [(0, 0 + 128), (22, 22 + 128)],
    "CoMoDo25": [(0, 0 + 128), (15, 15 + 128)],
    "CoMoDo26": [(0, 0 + 128), (16, 16 + 128)],
    "CoMoDo27": [(0, 0 + 128), (22, 22 + 128)],
    "CoMoDo28": [(0, 0 + 128), (9, 9 + 128)],
}

# Always in the first 4D dataset, ie. t_0.nii.gz
temp_crop = {
    "CoMoDo01b": (9, 17),
    "CoMoDo02": (15, 24),
    "CoMoDo03": (7, 17),
    "CoMoDo04": (2, 17),
    "CoMoDo05": (2, 11),
    "CoMoDo06": (9, 18),
    "CoMoDo08b": (4, 19),
    "CoMoDo09": (3, 11),
    "CoMoDo10": (10, 24),
    "CoMoDo11": (0, 10),
    "CoMoDo12": (0, 10),
    "CoMoDo13": (0, 13),
    "CoMoDo15": (4, 13),
    "CoMoDo16": (7, 14),
    "CoMoDo17": (3, 13),
    "CoMoDo18": (3, 17),
    "CoMoDo19": (7, 17),
    "CoMoDo20": (3, 13),
    "CoMoDo21": (5, 15),
    "CoMoDo22": (18, 27),
    "CoMoDo24": (13, 25),
    "CoMoDo25": (9, 19),
    "CoMoDo26": (6, 20),
    "CoMoDo27": (5, 14),
    "CoMoDo28": (0, 18),
}


def crop_save_times(path_orig_data, path_save, crop_record, temp_crop):

    affine_matrix = [
        [3.5, 0, 0, 0],
        [0, 1.70454, 0, 0],
        [0, 0, 1.70454, 0],
        [0, 0, 0, 1],
    ]

    for dirName, subdirList, fileList in os.walk(path_orig_data):
        for case in subdirList:

            V = nib.load(
                path_orig_data + case + "/4D/" + case + "_t_0.nii.gz"
            ).get_fdata()  # [32, 176, 176, 33]

            x, y = crop_record["CoMoDo" + str(case)]
            V_crop = V[
                :, 176 - y[1] : 176 - y[0], 176 - x[1] : 176 - x[0], :31
            ]  # [32, 128, 128, 31]

            # Temporal cropping
            from_a, to_b = temp_crop["CoMoDo" + str(case)]
            V_crop = V_crop[:, :, :, from_a : to_b + 1]

            new_image = nib.Nifti1Image(V_crop, affine=affine_matrix)
            new_image.to_filename(path_save + str(case) + ".nii.gz")

            print(case + "  shape: " + str(V_crop.shape))

        break


if __name__ == "__main__":
    path_orig_data = "/media/liset/DATA/Cloud/comodo_project/4D(2Dnav)_CoMoDoV2/"
    path_save = "/media/liset/LYF/annotations_William/"
    crop_save_times(path_orig_data, path_save, crop_record, temp_crop)
