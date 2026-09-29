"""Loss functions. AE/diffusion training uses relative-norm MSE on df,
mirroring upstream ``neugk/losses.py:relative_norm_mse``."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp


def mse_df(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)


def relative_norm_mse(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    """``mean_b ||pred - target||² / (||target||² + eps)``.

    Mirrors ``neugk/losses.py:relative_norm_mse`` (squared variant). Batch
    axis 0 is preserved as the reduction axis; everything else flattened.
    Lands in the 1-10 range when target is z-scored unit-variance.
    """
    assert pred.shape == target.shape, f"shape mismatch {pred.shape} != {target.shape}"
    if pred.ndim > 1:
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
    diff_sq = jnp.sum((pred - target) ** 2, axis=-1)
    tgt_sq = jnp.sum(target ** 2, axis=-1)
    return jnp.mean(diff_sq / (tgt_sq + eps))


def df_loss(pred: jnp.ndarray, target: jnp.ndarray, *, separate_zf: bool = False) -> jnp.ndarray:
    """Upstream ``df`` loss: plain MSE on zf slot + relative-norm MSE elsewhere.

    Matches ``neugk/losses.py:LossWrapper.forward`` lines 178-185 when
    ``separate_zf=True``. Channel slots 0:2 are the zf split, 2: are the
    other components. Without separate_zf falls back to relative-norm MSE.
    """
    if separate_zf and pred.shape[1] >= 4:
        zf_loss = jnp.mean((pred[:, :2] - target[:, :2]) ** 2)
        other_loss = relative_norm_mse(pred[:, 2:], target[:, 2:])
        return zf_loss + other_loss
    return relative_norm_mse(pred, target)


def l1(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(pred - target.reshape(pred.shape)))


def per_sample_mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    diff = (pred - target) ** 2
    return diff.reshape(diff.shape[0], -1).mean(axis=-1)


def integral_losses(
    geom_t: dict,
    pred_df: jnp.ndarray,
    pred_phi: Optional[jnp.ndarray],
    tgt_phi: jnp.ndarray,
    tgt_flux: jnp.ndarray,
    *,
    real_potens: bool = True,
) -> dict[str, jnp.ndarray]:
    """Physics-integral losses on denormalized batches, as upstream ``LossWrapper.integral_loss``.

    ``pred_df`` is ``(B, C, vp, mu, s, x, y)`` (a separate-zf ``C=4`` layout is
    recombined), ``geom_t`` a batched :func:`precompute_geometry` dict. Returns
    ``phi_int = mse(phi(df), tgt_phi)`` and
    ``flux_int = mean(pflux^2) + mse(eflux(df, phi), tgt_flux)``.
    """
    from neugk_jax.evaluate.integrals import flux_integral
    from neugk_jax.utils import recombine_zf

    pred_df = recombine_zf(pred_df, axis=1)

    def one(g, d, p):
        return flux_integral(g, d, p, real_potens=real_potens)

    if pred_phi is None:
        phi_int, (pflux, eflux, _) = jax.vmap(lambda g, d: one(g, d, None))(geom_t, pred_df)
    else:
        phi_int, (pflux, eflux, _) = jax.vmap(one)(geom_t, pred_df, pred_phi)
    tgt_phi = tgt_phi.reshape(phi_int.shape)
    return {
        "phi_int": jnp.mean((phi_int - tgt_phi) ** 2),
        "flux_int": jnp.mean(pflux**2) + jnp.mean((eflux - tgt_flux.reshape(eflux.shape)) ** 2),
    }
