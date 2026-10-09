"""PINC losses: the field-solve integrals and ky spectra of a prediction against its target's."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from neugk_jax.evaluate.integrals import spectral_integrals
from neugk_jax.losses import per_sample_rel_l1
from neugk_jax.utils import recombine_zf


def relative_l1_snap(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(per_sample_rel_l1(pred, target))


def served_spectral_loss(pred, target, mode_std=None, eps: float = 1e-8) -> jnp.ndarray:
    pl, tl = jnp.log1p(jnp.abs(pred)), jnp.log1p(jnp.abs(target))
    std = jnp.std(tl, axis=0, ddof=1) if mode_std is None else mode_std
    return jnp.mean(jnp.abs((pl - tl) / (std + eps)))


def pinc_integrals(geom_t: dict, df: jnp.ndarray, *, ds: float) -> dict:
    """Batched :func:`spectral_integrals` of a denormalized df ``(B, C, vp, mu, s, x, y)``.

    ``geom_t`` is a batched :func:`precompute_geometry` dict; separate-zf layouts are recombined.
    """

    return jax.vmap(lambda g, d: spectral_integrals(g, d, ds=ds))(geom_t, recombine_zf(df, axis=1))


# per-term comparators and potential (phi real, phi_c complex) of the ae fine-tune
LOSS_TYPES = {
    "potential": "phi",
    "phi_int": "relative_l1",
    "flux_int": "relative_l1",
    "kyspec": "log_std_l1",
    "qspec": "log_std_l1",
}


def _flat(x):
    return x.reshape(x.shape[0], -1)


def compare(kind: str, pred, target, *, mode_std=None, scale=None) -> jnp.ndarray:
    """Batch mean of one PINC comparator.

    ``relative_l1``: per-snapshot L1 over the target's; ``l1``: mean absolute error; ``std_l1``:
    the mean absolute error over ``scale`` per snapshot (the target's std when ``None``);
    ``log_std_l1``: :func:`served_spectral_loss` over ``mode_std``.
    """
    if kind == "relative_l1":
        return relative_l1_snap(pred, target)
    if kind == "log_std_l1":
        return served_spectral_loss(pred, target, mode_std)
    err = jnp.mean(jnp.abs(_flat(pred - target)), axis=-1)
    if kind == "l1":
        return jnp.mean(err)
    if kind == "std_l1":
        return jnp.mean(err / (jnp.std(_flat(target), axis=-1) if scale is None else scale))
    raise ValueError(f"unknown pinc loss type {kind!r}")


def pinc_terms(
    p: dict, t: dict, mode_std: Optional[dict] = None, loss_types: Optional[dict] = None
) -> dict[str, jnp.ndarray]:
    """PINC loss terms of prediction integrals ``p`` against target integrals ``t``.

    ``phi_int`` (on ``loss_types["potential"]``), ``flux_int = mean|pflux| + eflux`` and the
    ``kyspec`` / ``qspec`` spectra are compared with ``loss_types`` (over :data:`LOSS_TYPES`); a
    ``std_l1`` kyspec is that of the potential z-scored by the target's std. ``phi_int_mse`` and
    ``flux_int_l1`` (``mean|pflux| + mean|eflux - eflux_t|``) are gradient-free monitors.
    """
    mode_std = mode_std or {}
    kinds = {**LOSS_TYPES, **(loss_types or {})}
    pflux = jnp.mean(jnp.abs(p["pflux"]))
    pot = kinds["potential"]
    phi_var = jnp.var(_flat(t[pot]), axis=-1)
    return {
        "phi_int": compare(kinds["phi_int"], p[pot], t[pot]),
        "flux_int": pflux + compare(kinds["flux_int"], p["eflux"][:, None], t["eflux"][:, None]),
        "kyspec": compare(
            kinds["kyspec"],
            p["kyspec"],
            t["kyspec"],
            mode_std=mode_std.get("kyspec"),
            scale=phi_var,
        ),
        "qspec": compare(kinds["qspec"], p["qspec"], t["qspec"], mode_std=mode_std.get("qspec")),
        "phi_int_mse": jax.lax.stop_gradient(jnp.mean((p["phi"] - t["phi"]) ** 2)),
        "flux_int_l1": jax.lax.stop_gradient(pflux + jnp.mean(jnp.abs(p["eflux"] - t["eflux"]))),
    }


def pinc_losses(geom_t, pred_df, tgt_df, *, ds: float, mode_std=None):
    p = pinc_integrals(geom_t, pred_df, ds=ds)
    t = jax.lax.stop_gradient(pinc_integrals(geom_t, tgt_df, ds=ds))
    return pinc_terms(p, t, mode_std)
