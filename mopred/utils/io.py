"""
Low-level I/O helpers for the TIDAL pipeline.
Includes ``cond_mkdir``, ``custom_load`` (PyTorch checkpoint loader with
optional state-dict key remapping), and NIfTI save utilities.
"""
import os
import torch
import numpy as np
import nibabel as nib


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def custom_load(model, path, device="cuda:0"):
    whole_dict = torch.load(path, map_location=device, weights_only=False)
    if "model" in whole_dict:
        model.load_state_dict(whole_dict["model"])
    else:
        # Legacy VoxelMorph checkpoints: flat dict with a buffer key to strip.
        print("No model key in state_dict")
        sd = {k: v for k, v in whole_dict.items() if k != "spatial_transform.grid"}
        model.load_state_dict(sd)


def custom_save(model, path):
    if type(model) == torch.nn.DataParallel:
        model = model.module

    whole_dict = {"model": model.state_dict()}
    torch.save(whole_dict, path)


def save_tensor_as_nifti(
    tensor, description, val_vol_dir, epoch=-1, iter=-1, aff=np.eye(4)
):
    if epoch >= 0:
        # When used during validation
        save_dir = os.path.join(val_vol_dir, "epoch_%d" % epoch)
    else:
        # When used during testing
        save_dir = os.path.join(val_vol_dir, description)
    cond_mkdir(save_dir)
    # Check if tensor is on gpu
    numpy_tensor = tensor.detach().cpu().numpy()
    if "DVF" in description:
        # Put channel dimension last. Helps with DVF visualization.
        x = np.expand_dims(numpy_tensor[0, :, :, :], -1)
        y = np.expand_dims(numpy_tensor[1, :, :, :], -1)
        z = np.expand_dims(numpy_tensor[2, :, :, :], -1)
        numpy_tensor = np.concatenate([x, y, z], -1)

    new_image = nib.Nifti1Image(numpy_tensor, affine=aff)
    filename = os.path.join(save_dir, (description + "-%05s") % iter)
    # filename = os.path.join(save_dir, (description + '-iter_%05s.nii.gz') % iter)
    nib.save(new_image, filename)
