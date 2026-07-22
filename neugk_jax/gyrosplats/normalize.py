"""Invertible scalings: the zf field normalization and the token channel normalization.

Field side (JAX port of ``origin/gyrosplats:.../model/normalize.py``): atoms always fit
a z-scored field; ``zf`` z-scores the zonal (ky=0) and fluctuating parts separately so
the smooth high-amplitude zonal mode does not starve the transport-carrying fluctuations.

Token side (new): per-channel normalization of the (N, 17) atom tokens for flow
matching. mu is affine-mapped to [-1, 1] (NOT z-scored — keeps the 5-D matching cost
isotropic), Cholesky/amp channels are z-scored with dataset stats, and ky is reduced to
the z-scored drift around the per-slot carrier bin ``Δky = ky − 2π·m``. Heavy-tailed
channels can be asinh-compressed after z-scoring (exactly invertible).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from neugk_jax.gyrosplats.splat import KY_IDX, SplatParams, unpack

# the binormal y is the last field axis
YAXIS = -1

TWO_PI = float(2.0 * np.pi)


@dataclass
class ZfStats:
    """Per-snapshot zf normalization scalars (kind 'zf' or 'global')."""

    zonal_mean: jnp.ndarray
    zonal_std: jnp.ndarray
    fluc_mean: jnp.ndarray
    fluc_std: jnp.ndarray
    kind: str = "zf"


def _split(field_arr):
    # the ky=0 part: mean over the binormal axis
    zonal = jnp.broadcast_to(field_arr.mean(axis=YAXIS, keepdims=True), field_arr.shape)
    return zonal, field_arr - zonal


def zf_normalize(field_arr: jnp.ndarray, kind: str = "zf", eps: float = 1e-8):
    if kind == "global":
        # ddof=1 matches torch's unbiased std used at fit time
        m, s = field_arr.mean(), jnp.clip(field_arr.std(ddof=1), min=eps)
        return (field_arr - m) / s, ZfStats(m, s, m, s, "global")
    zonal, fluc = _split(field_arr)
    zm, zs = zonal.mean(), jnp.clip(zonal.std(ddof=1), min=eps)
    fm, fs = fluc.mean(), jnp.clip(fluc.std(ddof=1), min=eps)
    return (zonal - zm) / zs + (fluc - fm) / fs, ZfStats(zm, zs, fm, fs, "zf")


def zf_denormalize(norm_field: jnp.ndarray, st: ZfStats) -> jnp.ndarray:
    if st.kind == "global":
        return norm_field * st.zonal_std + st.zonal_mean
    zonal_n, fluc_n = _split(norm_field)
    return (zonal_n * st.zonal_std + st.zonal_mean) + (
        fluc_n * st.fluc_std + st.fluc_mean
    )


@dataclass(frozen=True)
class TokenStats:
    """Dataset-level per-channel stats for the 17 token channels + the zf-stats token.

    ``mean``/``std`` follow the pack() channel layout; the mu entries are fixed to
    (0.5, 0.5) — i.e. the affine map 2x − 1 — and the ky entry holds the Δky drift
    stats around the per-slot bin. ``asinh_channels`` are compressed with
    ``asinh(z / s) * s`` after z-scoring. ``stat_mean``/``stat_std`` normalize the
    global zf-stats token channels [zonal_mean, log zonal_std, log fluc_std].
    """

    mean: np.ndarray  # (17,)
    std: np.ndarray  # (17,)
    stat_mean: np.ndarray  # (3,)
    stat_std: np.ndarray  # (3,)
    asinh_channels: tuple[int, ...] = ()
    asinh_scale: float = 3.0

    @classmethod
    def load(cls, path, asinh_channels=(), asinh_scale=3.0) -> "TokenStats":
        z = np.load(path)
        return cls(
            mean=z["mean"].astype(np.float32),
            std=z["std"].astype(np.float32),
            stat_mean=z["stat_mean"].astype(np.float32),
            stat_std=z["stat_std"].astype(np.float32),
            asinh_channels=tuple(asinh_channels),
            asinh_scale=float(asinh_scale),
        )


def _asinh_mask(ts: TokenStats, n_ch: int):
    mask = np.zeros(n_ch, dtype=bool)
    for c in ts.asinh_channels:
        mask[c] = True
    return mask


def normalize_tokens(
    tokens: jnp.ndarray, bins: jnp.ndarray, ts: TokenStats
) -> jnp.ndarray:
    """(N, 17) raw atom tokens -> normalized fm space. ``bins`` (N,) int carrier modes."""
    offset = jnp.asarray(ts.mean)[None, :] + 0.0 * tokens
    offset = offset.at[:, KY_IDX].add(TWO_PI * bins.astype(tokens.dtype))
    z = (tokens - offset) / jnp.asarray(ts.std)[None, :]
    mask = _asinh_mask(ts, tokens.shape[1])
    if mask.any():
        s = ts.asinh_scale
        z = jnp.where(jnp.asarray(mask)[None, :], jnp.arcsinh(z / s) * s, z)
    return z


def denormalize_tokens(
    tokens_n: jnp.ndarray, bins: jnp.ndarray, ts: TokenStats
) -> SplatParams:
    """Normalized fm tokens (N, 17) -> physical SplatParams (single decode choke point)."""
    z = tokens_n
    mask = _asinh_mask(ts, tokens_n.shape[1])
    if mask.any():
        s = ts.asinh_scale
        z = jnp.where(jnp.asarray(mask)[None, :], jnp.sinh(z / s) * s, z)
    raw = z * jnp.asarray(ts.std)[None, :] + jnp.asarray(ts.mean)[None, :]
    raw = raw.at[:, KY_IDX].add(TWO_PI * bins.astype(raw.dtype))
    return unpack(raw)


# ---------------------------------------------------------------- v7 window state

@dataclass(frozen=True)
class WindowStats:
    """Channel-wise stats for the (n_tokens, 16) v7 state.

    Envelope mu channels (0:5) use the fixed affine 2x-1 (mean 0.5, std 0.5);
    every other envelope channel and all window channels are plain per-channel
    z-scored with stats fit on the run's TRAINING trajectories. Window channels
    are the DCT coefficients when ``dct`` is set. Pad channels (14, 15) of the
    window rows are inert (mean 0, std 1) and masked out of the loss.
    """

    env_mean: np.ndarray  # (16,)
    env_std: np.ndarray  # (16,)
    win_mean: np.ndarray  # (16,)
    win_std: np.ndarray  # (16,)
    n_env: int
    dct: bool = False
    dct_mat: np.ndarray | None = None  # (tied_k, tied_k)


def _apply_window_dct(state_raw, ws: WindowStats, inverse: bool):
    from neugk_jax.gyrosplats.window import WIN_AMP_CH, window_dct, window_idct

    if not ws.dct:
        return state_raw
    d = jnp.asarray(ws.dct_mat)
    win = state_raw[ws.n_env :]
    amp = win[:, :WIN_AMP_CH]
    amp = window_idct(amp, d) if inverse else window_dct(amp, d)
    win = win.at[:, :WIN_AMP_CH].set(amp)
    return state_raw.at[ws.n_env :].set(win)


def normalize_state(state_raw: jnp.ndarray, ws: WindowStats) -> jnp.ndarray:
    """(n_tokens, 16) raw v7 state -> normalized (DCT then per-channel z-score)."""
    state = _apply_window_dct(state_raw, ws, inverse=False)
    n = ws.n_env
    env = (state[:n] - jnp.asarray(ws.env_mean)) / jnp.asarray(ws.env_std)
    win = (state[n:] - jnp.asarray(ws.win_mean)) / jnp.asarray(ws.win_std)
    return jnp.concatenate([env, win], axis=0)


def denormalize_state(state_n: jnp.ndarray, ws: WindowStats) -> jnp.ndarray:
    """Normalized v7 state -> raw state (exact inverse of ``normalize_state``)."""
    n = ws.n_env
    env = state_n[:n] * jnp.asarray(ws.env_std) + jnp.asarray(ws.env_mean)
    win = state_n[n:] * jnp.asarray(ws.win_std) + jnp.asarray(ws.win_mean)
    state = jnp.concatenate([env, win], axis=0)
    return _apply_window_dct(state, ws, inverse=True)


def normalize_zf_scalars(zf: np.ndarray, ts: TokenStats) -> np.ndarray:
    """(4,) [zonal_mean, zonal_std, fluc_mean, fluc_std] -> (3,) normalized token channels."""
    vals = np.array([zf[0], np.log(zf[1]), np.log(zf[3])], dtype=np.float32)
    return (vals - ts.stat_mean) / ts.stat_std


def denormalize_zf_scalars(vals_n: jnp.ndarray, ts: TokenStats) -> ZfStats:
    """(3,) normalized stats-token channels -> ZfStats (fluc_mean is zero by construction)."""
    vals = vals_n * jnp.asarray(ts.stat_std) + jnp.asarray(ts.stat_mean)
    return ZfStats(
        zonal_mean=vals[0],
        zonal_std=jnp.exp(vals[1]),
        fluc_mean=jnp.zeros((), vals.dtype),
        fluc_std=jnp.exp(vals[2]),
        kind="zf",
    )
