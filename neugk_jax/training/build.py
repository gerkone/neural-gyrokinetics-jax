"""Model construction from an in-memory run config through the shared config builders."""

from __future__ import annotations

from neugk_jax.utils import config_dict


def run_config(cfg, ds=None) -> dict:
    """``{"model", "dataset", "training"}`` plain dict of a run config; ``ds`` fixes resolution and zf."""
    out = {k: config_dict(cfg.get(k)) for k in ("model", "dataset", "training")}
    if ds is not None:
        out["dataset"]["resolution"] = [int(r) for r in ds.resolution]
        out["dataset"]["separate_zf"] = bool(ds.separate_zf)
    return out


def build_ae(cfg, ds, *, key):
    from neugk_jax.translate import build_ae_from_config
    return build_ae_from_config(run_config(cfg, ds), key=key,
                                legacy_double_shortcut=bool(cfg.model.get("legacy_swin_shortcut", False)))


def build_dit(cfg, ae, *, key):
    from neugk_jax.translate import build_dit_from_config
    return build_dit_from_config(run_config(cfg), ae, key=key)


def build_gyroswin(cfg, ds, *, key):
    from neugk_jax.gyroswin.models import build_gyroswin_from_config
    return build_gyroswin_from_config(run_config(cfg, ds), key=key,
                                      legacy_double_shortcut=bool(cfg.model.get("legacy_swin_shortcut", False)))
