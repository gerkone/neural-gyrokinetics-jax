"""Separable splat synthesis: the whole atom bank on the full 5-D grid with two GEMMs.

JAX port of ``origin/gyrosplats:neugk/pinc/gyrosplats/model/render.py``. Each atom
factorizes into (velocity Gaussian) x (spatial Gaussian x carrier), so instead of one
dense ``(n_voxel, N)`` matrix we build a velocity factor ``(Nv, N)`` and a physical
factor ``(Np, N)`` and contract once. Fully differentiable and vmap-friendly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from neugk_jax.gyrosplats.splat import PHYS, VEL, YIDX, L_phys, L_vel, SplatParams


def subgrids(field_shape) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Velocity (Nv, 2) and physical (Np, 3) subgrids for a tensor-product index grid.

    Built directly from the shape (each axis ``arange(n)/n``) without materializing
    the full 5-D grid; matches the torch ``subgrids(grid, shape)`` on the grid built
    by ``load_snapshot``.
    """
    axes = [np.arange(n, dtype=np.float32) / n for n in field_shape]
    vel = np.stack(
        [g.reshape(-1) for g in np.meshgrid(*axes[:2], indexing="ij")], axis=1
    )
    phys = np.stack(
        [g.reshape(-1) for g in np.meshgrid(*axes[2:], indexing="ij")], axis=1
    )
    return jnp.asarray(vel), jnp.asarray(phys)


def _whiten(L: jnp.ndarray, diff: jnp.ndarray) -> jnp.ndarray:
    """Solve L u = diff per atom. L (N, d, d); diff (G, N, d) -> u (G, N, d)."""
    # batch axis first for solve_triangular: (N, d, G)
    u = solve_triangular(L, diff.transpose(1, 2, 0), lower=True)
    return u.transpose(2, 0, 1)


def factors(
    p: SplatParams, vel_grid: jnp.ndarray, phys_grid: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-atom factor matrices: V (Nv, N) velocity, Pre/Pim (Np, N) carrier-modulated physical."""
    Lp, Lv = L_phys(p), L_vel(p)
    dv = vel_grid[:, None, :] - p.mu[None, :, VEL]  # (Nv, N, 2)
    uv = _whiten(Lv, dv)
    V = jnp.exp(-0.5 * (uv**2).sum(-1))  # (Nv, N)

    dp = phys_grid[:, None, :] - p.mu[None, :, PHYS]  # (Np, N, 3) — (s, x, y)
    up = _whiten(Lp, dp)
    P = jnp.exp(-0.5 * (up**2).sum(-1))
    y = phys_grid[:, YIDX - 2]
    theta = p.ky[None, :] * (y[:, None] - p.mu[None, :, YIDX])
    return V, P * jnp.cos(theta), P * jnp.sin(theta)


def render(
    p: SplatParams,
    vel_grid: jnp.ndarray,
    phys_grid: jnp.ndarray,
    amps: jnp.ndarray | None = None,
    *,
    atom_chunk: int | None = None,
) -> jnp.ndarray:
    """-> (Nv, Np, 2). Channel 0 = Re[Σᵢ cᵢ Gᵢ e^{iθᵢ}], channel 1 = Im[...], θᵢ = kyᵢ(y − μ_y,i).

    ``atom_chunk`` accumulates the two contractions over atom chunks with a scan —
    the full-grid physical difference tensor is (N, 3, Np) ≈ 0.8 GB at N=1597,
    Np=43520, so chunking bounds peak memory for eval-time full renders.
    """
    a_all = p.amps if amps is None else amps
    if atom_chunk is None:
        V, Pre, Pim = factors(p, vel_grid, phys_grid)
        a, b = a_all[:, 0], a_all[:, 1]
        out_re = jnp.einsum("vi,pi->vp", V, Pre * a - Pim * b)
        out_im = jnp.einsum("vi,pi->vp", V, Pim * a + Pre * b)
        return jnp.stack([out_re, out_im], axis=-1)

    n = p.mu.shape[0]
    pad = (-n) % atom_chunk
    # pad with zero-amp atoms (identity cholesky) so the scan sees equal chunks
    def _pad(x, fill=0.0):
        return jnp.concatenate([x, jnp.full((pad, *x.shape[1:]), fill, x.dtype)], 0)

    pp = SplatParams(
        mu=_pad(p.mu),
        L_phys_raw=_pad(p.L_phys_raw),
        L_vel_raw=_pad(p.L_vel_raw),
        amps=_pad(p.amps),
        ky=_pad(p.ky),
    )
    aa = _pad(a_all)
    n_chunks = (n + pad) // atom_chunk
    chunked = jax.tree_util.tree_map(
        lambda x: x.reshape(n_chunks, atom_chunk, *x.shape[1:]), (pp, aa)
    )

    # checkpoint the chunk body: without it the scan saves every chunk's
    # whitening intermediates for the backward (O(100 gb) at full grid)
    @jax.checkpoint
    def step(acc, chunk):
        cp, ca = chunk
        out = render(cp, vel_grid, phys_grid, ca)
        return acc + out, None

    init = jnp.zeros((vel_grid.shape[0], phys_grid.shape[0], 2), dtype=p.mu.dtype)
    out, _ = jax.lax.scan(step, init, chunked)
    return out


def to_field(sep: jnp.ndarray, field_shape) -> jnp.ndarray:
    """(Nv, Np, 2) -> the standard field layout (2, v∥, μ, s, x, y)."""
    return sep.transpose(2, 0, 1).reshape(2, *field_shape)


def to_sep(field: jnp.ndarray) -> jnp.ndarray:
    """(2, v∥, μ, s, x, y) -> (Nv, Np, 2)."""
    shape = field.shape[1:]
    nv, np_ = shape[0] * shape[1], shape[2] * shape[3] * shape[4]
    return field.reshape(2, nv, np_).transpose(1, 2, 0)
