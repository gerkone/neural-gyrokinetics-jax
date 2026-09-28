"""Latent rectified flow matching (Gaussian prior, continuous time, OT-coupled).

Mirrors ``FlowMatchingRunner`` in the upstream torch repo
(``neugk/diffusion/run.py:552-617``):

* ``x0 ~ N(0, I)``
* ``x1 = encoded_df * latent_scale``
* ``t ~ sigmoid(N(0, 1))`` (continuous time path)
* optional minibatch optimal transport (Hungarian) couples ``x0`` and
  ``x1`` across the batch before training
* ``xt = t * x1 + (1 - t) * x0`` and ``v_target = x1 - x0``
* train: ``MSE(model(xt, t, cond), v_target)``
* sample: Euler integration over ``[0, 1]`` of the learned velocity field

The flow-matching loss/training is independent of the autoencoder
specifics — caller passes the latent batch directly.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def sample_prior(key, shape, dtype=jnp.float32):
    """Gaussian prior x0."""
    return jr.normal(key, shape, dtype=dtype)


def sample_time(key, batch: int, dtype=jnp.float32):
    """``t ~ sigmoid(N(0, 1))`` — biases mass toward the middle of [0, 1]."""
    return jax.nn.sigmoid(jr.normal(key, (batch,), dtype=dtype))


def minibatch_ot(x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
    """Optimal-transport coupling of ``x0`` and ``x1`` across the batch axis.

    Uses scipy's Hungarian algorithm via ``jax.pure_callback``, so the coupling
    also works inside a jit'd training step — fine for the small batch sizes flow
    matching typically uses (upstream torch does the same scipy call).

    Bug fix vs upstream: ``scipy.optimize.linear_sum_assignment`` returns
    ``(row_ind, col_ind)`` where ``row_ind`` is always the identity
    permutation for a square cost matrix. Upstream's ``x0[row_ind]`` is
    therefore a no-op, silently disabling OT coupling during training.
    The correct permutation is over ``col_ind``: pair ``x0[i]`` with
    ``x1[col_ind[i]]``. Equivalently we re-order ``x0`` so that the
    returned ``x0_new[i]`` is the OT-match for the original ``x1[i]``,
    which is ``x0[argsort(col_ind)]``.
    """
    bs = x0.shape[0]
    x0_flat = x0.reshape(bs, -1)
    x1_flat = x1.reshape(bs, -1)
    # ||a-b||^2 = |a|^2 + |b|^2 - 2a.b: one gemm, no (B, B, D) intermediate
    sq0 = jnp.sum(x0_flat ** 2, axis=-1)
    sq1 = jnp.sum(x1_flat ** 2, axis=-1)
    cost = jnp.sqrt(jnp.maximum(sq0[:, None] + sq1[None, :] - 2.0 * (x0_flat @ x1_flat.T), 0.0))

    def _assign(cost_np):
        import scipy.optimize
        _, col = scipy.optimize.linear_sum_assignment(np.asarray(cost_np))
        return np.argsort(col).astype(np.int32)

    perm = jax.pure_callback(_assign, jax.ShapeDtypeStruct((bs,), jnp.int32), cost)
    return x0[perm]


def fm_forward_loss(
    model_fn: Callable,
    latents: jnp.ndarray,
    cond: Optional[jnp.ndarray],
    *,
    key,
    latent_scale: float = 1.0,
    use_ot: bool = True,
    pair_fn: Optional[Callable] = None,
    loss_mask: Optional[jnp.ndarray] = None,
    aux_loss_fn: Optional[Callable] = None,
    time_fn: Optional[Callable] = None,
) -> jnp.ndarray:
    """One flow-matching training step (returns the scalar loss).

    ``model_fn(xt, t_scalar, cond_per_sample)`` is the *per-sample* DiT
    forward — caller vmaps the model over the batch.

    Optional hooks (all default off — the latent path is unchanged):

    * ``pair_fn(key, x0, x1) -> x0`` — per-sample set coupling (e.g. within-set
      atom matching for splat banks); replaces ``minibatch_ot`` when set.
    * ``loss_mask`` — broadcastable to ``x1``; masks dead channels (weighted mean).
    * ``aux_loss_fn(x1_hat, x1, t) -> scalar`` — auxiliary loss on the predicted
      clean sample ``x1_hat = xt + (1 - t)·v̂`` (e.g. a differentiable render loss).
    * ``time_fn(key, batch) -> t`` — replaces the sigmoid-normal time sampler
      (e.g. a heavy tail near t=1 for targets whose fine structure only exists
      in a thin neighborhood of the data).
    """
    bs = latents.shape[0]
    k_prior, k_t, k_pair = jr.split(key, 3)
    x1 = latents * latent_scale
    x0 = sample_prior(k_prior, x1.shape, dtype=x1.dtype)
    if pair_fn is not None:
        x0 = pair_fn(k_pair, x0, x1)
    elif use_ot:
        x0 = minibatch_ot(x0, x1)
    t = time_fn(k_t, bs) if time_fn is not None else sample_time(k_t, bs, dtype=x1.dtype)
    t_b = t.reshape(-1, *[1] * (x1.ndim - 1))
    xt = t_b * x1 + (1.0 - t_b) * x0
    target_v = x1 - x0
    # vmap over the batch — model_fn is per-sample
    pred = jax.vmap(model_fn)(xt, t, cond) if cond is not None else jax.vmap(model_fn)(xt, t)
    err = (pred - target_v) ** 2
    if loss_mask is None:
        loss = jnp.mean(err)
    else:
        w = jnp.broadcast_to(loss_mask, err.shape)
        loss = jnp.sum(err * w) / jnp.maximum(jnp.sum(w), 1.0)
    if aux_loss_fn is not None:
        x1_hat = xt + (1.0 - t_b) * pred
        loss = loss + aux_loss_fn(x1_hat, x1, t)
    return loss


def _euler_step(velocity, x, ti, dti):
    return velocity(x, ti) * dti


def _midpoint_step(velocity, x, ti, dti):
    k1 = velocity(x, ti)
    return velocity(x + 0.5 * dti * k1, ti + 0.5 * dti) * dti


def _heun_step(velocity, x, ti, dti):
    """Explicit trapezoid: exact for a velocity field that is linear along the path."""
    k1 = velocity(x, ti)
    k2 = velocity(x + dti * k1, ti + dti)
    return 0.5 * dti * (k1 + k2)


def _rk4_step(velocity, x, ti, dti):
    k1 = velocity(x, ti)
    k2 = velocity(x + 0.5 * dti * k1, ti + 0.5 * dti)
    k3 = velocity(x + 0.5 * dti * k2, ti + 0.5 * dti)
    k4 = velocity(x + dti * k3, ti + dti)
    return (dti / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


_STEPPERS = {
    "euler": _euler_step,
    "midpoint": _midpoint_step,
    "heun": _heun_step,
    "rk4": _rk4_step,
}
NFE_PER_STEP = {"euler": 1, "midpoint": 2, "heun": 2, "rk4": 4}


def euler_sample(
    model_fn: Callable,
    *,
    key,
    shape: tuple[int, ...],
    cond: Optional[jnp.ndarray] = None,
    steps: int = 10,
    latent_scale: float = 1.0,
    dtype=jnp.float32,
    prior_fn: Optional[Callable] = None,
    time_warp: float = 1.0,
    method: str = "euler",
) -> jnp.ndarray:
    """Integrate the velocity field over ``[0, 1]``.

    ``method`` selects the explicit scheme (``euler``, ``midpoint``, ``heun``, ``rk4``); compare
    schemes at matched NFE via :data:`NFE_PER_STEP`, not at matched step count.

    Fused as a single ``jax.lax.scan`` so the whole sampling roll-out is
    one jit'd kernel — avoids the per-step host-device sync the previous
    Python ``for`` loop incurred. ``shape = (B, *latent_grid, z_dim)``
    matches the encoder's output. Returns a sample in the data scale
    (divides by ``latent_scale`` at the end to undo the encoder's whitening).
    """
    bs = shape[0]
    # prior_fn overrides the gaussian source (e.g. structured/tied noise)
    x0 = prior_fn(key, shape) if prior_fn is not None else sample_prior(key, shape, dtype=dtype)
    # time_warp > 1 concentrates integration steps near t=1 (t = 1 - (1-u)^p)
    u = jnp.linspace(0.0, 1.0, steps + 1, dtype=dtype)
    t_grid = 1.0 - (1.0 - u) ** time_warp if time_warp != 1.0 else u
    dts = t_grid[1:] - t_grid[:-1]
    ts = t_grid[:-1]

    if cond is not None:
        def velocity(x, ti):
            return jax.vmap(model_fn)(x, jnp.full((bs,), ti, dtype=dtype), cond)
    else:
        def velocity(x, ti):
            return jax.vmap(model_fn)(x, jnp.full((bs,), ti, dtype=dtype))

    increment = _STEPPERS[method]

    def step(x, ti_dti):
        ti, dti = ti_dti
        return x + increment(velocity, x, ti, dti), None

    x, _ = jax.lax.scan(step, x0, (ts, dts))
    return x / latent_scale

