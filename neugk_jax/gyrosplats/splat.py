"""The Splat: a bank of N Gaussian/Gabor atoms, 17 parameters each.

Pure-function JAX port of the torch reference
(``origin/gyrosplats:neugk/pinc/gyrosplats/model/atoms.py``). Per atom:

* ``mu (5)`` — center in ``(v∥, μ, s, x, y)``, each axis index/size-normalized to [0, 1)
* ``L_phys_raw (6)`` — raw lower-triangular 3x3 Cholesky factor of the SPATIAL
  covariance over ``(s, x, y)``; diagonals pass through softplus so any 6 reals
  decode to a valid positive-definite shape
* ``L_vel_raw (3)`` — same for the 2x2 velocity covariance over ``(v∥, μ)``
* ``amps (2)`` — complex amplitude ``c = a + i·b``
* ``ky (1)`` — carrier wavenumber along the binormal y (period 1, bin m ⇔ ky = 2π·m)

Only the forward/decode surface is ported; the fitting-time initializers
(``random_init``/``field_init``/``graft_*``) stay upstream.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

# column layout of the 5-d coordinates
VEL, PHYS, YIDX = slice(0, 2), slice(2, 5), 4

N_CHANNELS = 17
MU_SL, LPHYS_SL, LVEL_SL, AMPS_SL, KY_IDX = (
    slice(0, 5),
    slice(5, 11),
    slice(11, 14),
    slice(14, 16),
    16,
)


# channels 0..13 (mu + both cholesky factors) are the shared group envelope;
# 14..16 (amps + ky) are per-carrier
ENVELOPE_CH = 14


def bank_structure(bins):
    """(n_atoms,) carrier modes -> (env_idx (n_env,), grp_idx (n_groups, tied_k))."""
    import numpy as np

    bins = np.asarray(bins)
    env_idx = np.where(bins == 0)[0]
    carrier_idx = np.where(bins != 0)[0]
    tied_k = int(len(np.unique(bins[carrier_idx]))) if carrier_idx.size else 1
    assert carrier_idx.size % tied_k == 0, "carrier count not divisible by bin count"
    grp_idx = carrier_idx.reshape(-1, tied_k)
    if carrier_idx.size:
        assert (np.diff(bins[grp_idx], axis=1) > 0).all(), "group rows must cycle the bin set"
    return env_idx, grp_idx


def tie_group_channels(x, grp_idx):
    """Broadcast each group's first-row envelope channels (0..13) to all its rows.

    x (n_tokens, 17) — atom rows plus trailing extras untouched. With tied
    source noise and a tied velocity head, group envelopes stay exactly tied
    along the whole flow trajectory.
    """
    g = x[grp_idx]  # (n_groups, tied_k, 17)
    g = g.at[:, :, :ENVELOPE_CH].set(g[:, :1, :ENVELOPE_CH])
    return x.at[grp_idx].set(g)


class SplatParams(NamedTuple):
    """One atom bank. All leaves share the leading (N,) atom axis."""

    mu: jnp.ndarray  # (N, 5)
    L_phys_raw: jnp.ndarray  # (N, 6)
    L_vel_raw: jnp.ndarray  # (N, 3)
    amps: jnp.ndarray  # (N, 2)
    ky: jnp.ndarray  # (N,)


def tri(flat: jnp.ndarray, d: int) -> jnp.ndarray:
    """(N, d(d+1)/2) -> (N, d, d) lower-triangular Cholesky, softplus-positive diagonal."""
    n = flat.shape[0]
    # static index sets (numpy at trace time)
    diag = np.arange(d)
    L = jnp.zeros((n, d, d), dtype=flat.dtype)
    L = L.at[:, diag, diag].set(jnp.clip(jnp.logaddexp(flat[:, :d], 0.0), min=1e-5))
    if d > 1:
        r, c = np.tril_indices(d, k=-1)
        L = L.at[:, r, c].set(flat[:, d:])
    return L


def inv_softplus(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Inverse of softplus, matching the torch reference."""
    x = jnp.clip(x, min=eps)
    return x + jnp.log(-jnp.expm1(-x))


def L_phys(p: SplatParams) -> jnp.ndarray:
    return tri(p.L_phys_raw, 3)


def L_vel(p: SplatParams) -> jnp.ndarray:
    return tri(p.L_vel_raw, 2)


def pack(p: SplatParams) -> jnp.ndarray:
    """SplatParams -> (N, 17) flat atom tokens."""
    return jnp.concatenate(
        [p.mu, p.L_phys_raw, p.L_vel_raw, p.amps, p.ky[:, None]], axis=1
    )


def unpack(tokens: jnp.ndarray) -> SplatParams:
    """(N, 17) flat atom tokens -> SplatParams."""
    return SplatParams(
        mu=tokens[:, MU_SL],
        L_phys_raw=tokens[:, LPHYS_SL],
        L_vel_raw=tokens[:, LVEL_SL],
        amps=tokens[:, AMPS_SL],
        ky=tokens[:, KY_IDX],
    )
