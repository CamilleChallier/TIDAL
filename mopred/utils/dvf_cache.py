"""
utils/dvf_cache.py
==================
Pre-compute and cache DVFs from a frozen registration model (VoxelMorph or MambaMorph).
Cache directories encode the model type and checkpoint content so that liver vs ACDC,
VoxelMorph vs MambaMorph, and different checkpoints never share the same cache.
"""
from __future__ import annotations
import os
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..data.data_loaders import NAVIGATOR_4D_Dataset_multitime
from ..utils.io import cond_mkdir


def _vol_key(filepath: str) -> str:
    """Stable cache key: SHA1 of the absolute path."""
    return hashlib.sha1(os.path.abspath(filepath).encode()).hexdigest()


def _checkpoint_hash(ckpt_path: str) -> str:
    """SHA1 of checkpoint file bytes (content-based, rename-safe), truncated to 12 chars."""
    h = hashlib.sha1()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def make_dvf_cache_dir(base_root: str, reg_model: str, ckpt_path: str) -> str:
    """
    Return a cache directory that uniquely identifies the (dataset, model, checkpoint) triple.

    base_root  : typically os.path.dirname(cfg.data_dir) — differs per dataset (liver / ACDC)
    reg_model  : "voxelmorph" | "mambamorph"
    ckpt_path  : absolute path to the frozen registration checkpoint

    Result: base_root/dvf_cache/{reg_model}_{ckpt_hash12}/
    """
    tag = f"{reg_model}_{_checkpoint_hash(ckpt_path)}"
    cache_dir = os.path.join(base_root, "dvf_cache", tag)
    print(f"[DVF cache] dir: {cache_dir}")
    return cache_dir


def build_dvf_cache(
    data_dir: str,
    sequence_list: list,
    vm: torch.nn.Module,
    cache_dir: str,
    device: torch.device,
    nb_inputs: int = 3,
    nb_pred: int = 1,
) -> None:
    """
    Iterate over every (ref, current) pair and save DVFs as .npy files.
    Skips pairs that are already cached.
    """
    cond_mkdir(cache_dir)
    
    split_id  = hashlib.sha1("".join(sorted(sequence_list)).encode()).hexdigest()[:16]
    sentinel  = os.path.join(cache_dir, f".complete_{split_id}.ok")
    if os.path.exists(sentinel):
        print(f"[DVF cache] Cache already complete for this split — skipping.")
        return
    
    vm.eval()

    dataset = NAVIGATOR_4D_Dataset_multitime(
        data_dir,
        nb_inputs=nb_inputs,
        sequence_list=sequence_list,
        nb_pred=nb_pred,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f"[DVF cache] Computing DVFs for {len(dataset)} samples → {cache_dir}")

    with torch.no_grad():
        for ref_vol, _, current_vol_list, vol_files in loader:
            ref_vol = ref_vol.unsqueeze(1).to(device)

            for t, (cur_vol, cur_file) in enumerate(
                zip(current_vol_list, vol_files)
            ):
                # vol_files[t] is a list of length batch_size (here 1)
                filepath = cur_file[0]
                cache_path = os.path.join(cache_dir, _vol_key(filepath) + ".npy")

                if os.path.exists(cache_path):                    
                    continue  # already computed

                cur_vol = cur_vol.unsqueeze(1).to(device)
                dvf = vm(ref_vol, cur_vol)  # (1, 3, D, H, W)
                np.save(cache_path, dvf.cpu().numpy().astype(np.float32))
                
    open(sentinel, "w").close()
    print(f"[DVF cache] Done — sentinel written: {sentinel}")


def load_dvf(filepath: str, cache_dir: str, device: torch.device) -> torch.Tensor:
    """Load a cached DVF tensor for the given volume filepath."""
    cache_path = os.path.join(cache_dir, _vol_key(filepath) + ".npy")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No cached DVF for '{filepath}'.\n"
            f"Run build_dvf_cache() first.  Expected: {cache_path}"
        )
    arr = np.load(cache_path).astype(np.float32)
    return torch.from_numpy(arr).to(device)  # (1, 3, D, H, W)