"""Lightweight config helpers — no model imports so this is safe to import in
any environment (including the `mamba` env that lacks monai/the full VAE zoo)."""
from __future__ import annotations

import argparse
import os

import yaml


def load_config(path: str) -> argparse.Namespace:
    """Load a YAML config and flatten model sub-dicts with a ``{key}_`` prefix.

    Example::

        mia_vae:
          base_ch: 32        →  cfg.mia_vae_base_ch = 32
        causal_dit:
          latent_ch: 64      →  cfg.causal_dit_latent_ch = 64
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    flat: dict = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}_{sub_k}"] = sub_v
        else:
            flat[k] = v
    ns = argparse.Namespace(**flat)
    ns._config_path = os.path.abspath(path)
    return ns


def _apply_overrides(cfg: argparse.Namespace, overrides: list[str]) -> None:
    """Parse KEY=VALUE strings and patch *cfg* in-place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Bad --override syntax (expected KEY=VALUE): {item!r}")
        key, raw_val = item.split("=", 1)
        for cast in (int, float, lambda v: {"true": True, "false": False}[v.lower()]):
            try:
                val = cast(raw_val)
                break
            except (ValueError, KeyError):
                val = raw_val
        setattr(cfg, key, val)
        print(f"[override] {key} = {val!r}")
