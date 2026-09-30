"""Model construction from an in-memory run config through the shared config builders.

The builders in ``neugk_jax.translate`` / ``neugk_jax.gyroswin.models`` read a config
mapping (``{"model": ..., "dataset": ...}``); a builder that only accepts a YAML path gets
one temporary file.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable

import yaml

from neugk_jax.utils import config_dict


def run_config(cfg, ds=None) -> dict:
    """``{"model", "dataset"}`` plain dict of a run config; ``ds`` fixes the resolution and zf layout."""
    out = {"model": config_dict(cfg.get("model")), "dataset": config_dict(cfg.get("dataset"))}
    if ds is not None:
        out["dataset"]["resolution"] = [int(r) for r in ds.resolution]
        out["dataset"]["separate_zf"] = bool(ds.separate_zf)
    return out


def call_builder(builder: Callable, source: dict, *args, **kwargs):
    try:
        return builder(source, *args, **kwargs)
    except TypeError as e:
        if "PathLike" not in str(e):
            raise
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(source, f)
        return builder(path, *args, **kwargs)
    finally:
        os.remove(path)


def build_ae(cfg, ds, *, key):
    from neugk_jax.translate import build_ae_from_config
    return call_builder(build_ae_from_config, run_config(cfg, ds), key=key,
                        legacy_double_shortcut=bool(cfg.model.get("legacy_swin_shortcut", False)))


def build_dit(cfg, ae, *, key):
    from neugk_jax.translate import build_dit_from_config
    return call_builder(build_dit_from_config, run_config(cfg), ae, key=key)


def build_gyroswin(cfg, ds, *, key):
    from neugk_jax.gyroswin.models import build_gyroswin_from_config
    return call_builder(build_gyroswin_from_config, run_config(cfg, ds), key=key,
                        legacy_double_shortcut=bool(cfg.model.get("legacy_swin_shortcut", False)))
