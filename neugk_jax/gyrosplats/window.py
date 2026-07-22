"""v7 "windows" state: (871, 16) tokens per snapshot.

Rows 0..749 are the free envelope atoms (16 ch = mu(5), L_phys_raw(6),
L_vel_raw(3), amps(2); ky is dropped — it is zero-bin jitter). Rows 750..870
are one "window" token per tied carrier group (14 ch = the group's 7 carrier
amplitudes in bin order [Re_7, Im_7, ..., Re_13, Im_13] + 2 masked pad ch).

The carriers' geometry (mu / L_phys / L_vel) is a dataset-wide FROZEN scaffold
(validated: cross-time/per-group max dev ~0.02) and their ky is exactly 2*pi*m,
so only the 14 amplitude channels per group are generative. ``extract_scaffold``
pulls the scaffold once at dataset init; ``pack_windows`` / ``unpack_windows``
convert between the (1597, 17) atom bank and the (871, 16) state.

The optional DCT-II option rewrites the 14 window channels into an orthonormal
DCT basis over the 7 bins (separately for Re and Im) — an exactly invertible
channel transform applied before z-scoring.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from neugk_jax.gyrosplats.splat import SplatParams, bank_structure

# state channel layout (shared row width = 16)
N_STATE_CH = 16
ENV_MU_SL = slice(0, 5)
ENV_LPHYS_SL = slice(5, 11)
ENV_LVEL_SL = slice(11, 14)
ENV_AMPS_SL = slice(14, 16)
WIN_AMP_CH = 14  # 7 bins x (Re, Im)
WIN_PAD_SL = slice(14, 16)

TWO_PI = float(2.0 * np.pi)


class Scaffold(NamedTuple):
    """Frozen carrier geometry, shared per tied group."""

    mu: jnp.ndarray  # (n_groups, 5)
    L_phys_raw: jnp.ndarray  # (n_groups, 6)
    L_vel_raw: jnp.ndarray  # (n_groups, 3)
    ky: jnp.ndarray  # (tied_k,) = 2*pi*m in the group's bin order
    env_idx: jnp.ndarray  # (n_env,) rows of envelope atoms in the 1597 bank
    grp_idx: jnp.ndarray  # (n_groups, tied_k) rows of carriers in the 1597 bank


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix D (n, n); inverse is D.T."""
    k = np.arange(n)[:, None]
    m = np.arange(n)[None, :]
    d = np.cos(np.pi * (2 * m + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    d[0] *= np.sqrt(0.5)
    return d.astype(np.float64)


def _reim(win14: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # interleaved [Re_0, Im_0, ...] -> (..., 7) Re and (..., 7) Im
    return win14[..., 0::2], win14[..., 1::2]


def _interleave(re: jnp.ndarray, im: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([re, im], axis=-1).reshape(*re.shape[:-1], -1)


def window_dct(win14: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Forward DCT over the 7 bins, separately on Re and Im (interleaved layout)."""
    re, im = _reim(win14)
    return _interleave(re @ d.T, im @ d.T)


def window_idct(coef14: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Inverse of ``window_dct`` (orthonormal: D.T)."""
    cre, cim = _reim(coef14)
    return _interleave(cre @ d, cim @ d)


def extract_scaffold(atoms_stack: np.ndarray, bins: np.ndarray) -> Scaffold:
    """Frozen scaffold from a few snapshots (T, 1597, 17) of a training trajectory."""
    env_idx, grp_idx = bank_structure(bins)  # (n_env,), (n_groups, tied_k)
    car = atoms_stack[:, grp_idx.reshape(-1), :]  # (T, n_groups*tied_k, 17)
    tied_k = grp_idx.shape[1]
    car = car.reshape(atoms_stack.shape[0], grp_idx.shape[0], tied_k, 17)
    # per-group scaffold: mean over time and over the tied carriers
    mu = car[..., 0:5].mean(axis=(0, 2))
    lp = car[..., 5:11].mean(axis=(0, 2))
    lv = car[..., 11:14].mean(axis=(0, 2))
    # ky is exactly 2*pi*m in the group's bin order
    ky = TWO_PI * bins[grp_idx[0]].astype(np.float64)
    return Scaffold(
        mu=jnp.asarray(mu, jnp.float32),
        L_phys_raw=jnp.asarray(lp, jnp.float32),
        L_vel_raw=jnp.asarray(lv, jnp.float32),
        ky=jnp.asarray(ky, jnp.float32),
        env_idx=jnp.asarray(env_idx),
        grp_idx=jnp.asarray(grp_idx),
    )


def scaffold_max_dev(atoms_stack: np.ndarray, bins: np.ndarray) -> float:
    """Max abs deviation of individual carrier mu/L rows from the shared scaffold."""
    _, grp_idx = bank_structure(bins)
    tied_k = grp_idx.shape[1]
    car = atoms_stack[:, grp_idx.reshape(-1), :].reshape(
        atoms_stack.shape[0], grp_idx.shape[0], tied_k, 17
    )
    geom = car[..., 0:14]  # mu + both cholesky factors
    ref = geom.mean(axis=(0, 2), keepdims=True)
    return float(np.abs(geom - ref).max())


def pack_windows(atoms: np.ndarray, scaffold: Scaffold) -> np.ndarray:
    """(1597, 17) atom bank -> (871, 16) raw state (before DCT / normalization)."""
    env_idx = np.asarray(scaffold.env_idx)
    grp_idx = np.asarray(scaffold.grp_idx)
    n_win = grp_idx.shape[0]
    state = np.zeros((env_idx.shape[0] + n_win, N_STATE_CH), dtype=np.float32)
    # envelope rows: mu, L_phys, L_vel, amps (drop ky at ch 16)
    state[: env_idx.shape[0], :14] = atoms[env_idx, :14]
    state[: env_idx.shape[0], 14:16] = atoms[env_idx, 14:16]
    # window rows: interleaved carrier amplitudes per group
    amps = atoms[grp_idx.reshape(-1), 14:16].reshape(n_win, -1)  # (n_win, 14)
    state[env_idx.shape[0] :, :WIN_AMP_CH] = amps
    return state


def unpack_windows(state: jnp.ndarray, scaffold: Scaffold) -> SplatParams:
    """(871, 16) raw state + scaffold -> physical (1597, 17) SplatParams."""
    env_idx = scaffold.env_idx
    grp_idx = scaffold.grp_idx
    n_env = env_idx.shape[0]
    n_win, tied_k = grp_idx.shape
    n_atoms = bank_total(scaffold)

    env = state[:n_env]
    win = state[n_env:]

    full = jnp.zeros((n_atoms, 17), dtype=state.dtype)
    # envelope atoms (ky stays 0)
    env_atom = jnp.concatenate([env[:, :14], env[:, 14:16], jnp.zeros((n_env, 1))], axis=1)
    full = full.at[env_idx].set(env_atom)

    # carrier atoms: broadcast scaffold, insert per-group amps
    amps = win[:, :WIN_AMP_CH].reshape(n_win, tied_k, 2)  # (g, k, 2)
    mu = jnp.broadcast_to(scaffold.mu[:, None, :], (n_win, tied_k, 5))
    lp = jnp.broadcast_to(scaffold.L_phys_raw[:, None, :], (n_win, tied_k, 6))
    lv = jnp.broadcast_to(scaffold.L_vel_raw[:, None, :], (n_win, tied_k, 3))
    ky = jnp.broadcast_to(scaffold.ky[None, :, None], (n_win, tied_k, 1))
    car_atom = jnp.concatenate([mu, lp, lv, amps, ky], axis=-1)  # (g, k, 17)
    full = full.at[grp_idx].set(car_atom)

    from neugk_jax.gyrosplats.splat import unpack

    return unpack(full)


def bank_total(scaffold: Scaffold) -> int:
    """Total number of atoms in the reconstructed bank."""
    return int(scaffold.env_idx.shape[0] + scaffold.grp_idx.size)
