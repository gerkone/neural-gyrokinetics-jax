"""Coordinate-generated patch embedding and DeepONet unpatch.

Drop-in replacements of ``PatchEmbed`` / ``PatchExpand`` (the ``unpatch``) on the mu-folded
``(species, vpar, s, x, y, c mu)`` grid. The weights of every point of a patch come from a SiLU MLP of
its physical coordinates: the offset from the token centre along s, x, y (index offset times the grid
spacing), the absolute vpar and mu, and the real/imag channel. Tokens sit on the patch lattice of
``PatchEmbed``; the decoder reconstructs every patch as ``sum_r branch(token)_r * basis(point)_r``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.models.utils import MLP, leaky_relu, silu

# vpar, mu, s, x, y scales of the coordinate features (v_th, mu units, field-line s, rho, rho)
FEATURE_SCALE = (3.0, 4.0, 0.15, 5.0, 50.0)


class PatchGrid(eqx.Module):
    """Point coordinates of one patch for every vpar token, in the ``(pv, ps, px, py, c, mu)`` order."""

    buffer_fields = ("vpar", "mu", "offsets", "channel", "weight")

    vpar: jnp.ndarray
    mu: jnp.ndarray
    offsets: jnp.ndarray
    channel: jnp.ndarray
    weight: jnp.ndarray
    spacing: tuple[float, ...] = eqx.field(static=True)
    scale: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, patch, padded, n_channels: int, grid: dict, scale=FEATURE_SCALE):
        _, pv, ps, px, py = patch
        vp_pad = padded[1]
        n_mu = len(grid["mu"])
        vpar = np.asarray(grid["vpar"], np.float64)
        # padded vpar points continue the uniform grid and get zero weight
        vpar = np.concatenate([vpar, vpar[-1] + (vpar[1] - vpar[0]) * np.arange(1, vp_pad - len(vpar) + 1)])
        intvp = np.pad(np.asarray(grid["intvp"], np.float64), (0, vp_pad - len(grid["intvp"])))
        intmu = np.asarray(grid["intmu"], np.float64)
        idx = np.meshgrid(*(np.arange(n) for n in (pv, ps, px, py, n_channels, n_mu)), indexing="ij")
        iv, i_s, ix, iy, ic, im = (a.reshape(-1) for a in idx)
        tv = vp_pad // pv
        rows = np.arange(tv)[:, None] * pv + iv[None]
        self.vpar = jnp.asarray(vpar[rows] / scale[0], jnp.float32)
        self.mu = jnp.asarray(np.asarray(grid["mu"], np.float64)[im] / scale[1], jnp.float32)
        self.offsets = jnp.asarray(
            np.stack([i_s - (ps - 1) / 2, ix - (px - 1) / 2, iy - (py - 1) / 2], -1), jnp.float32
        )
        self.channel = jnp.asarray(np.eye(n_channels)[ic], jnp.float32)
        w = intvp[rows] * intmu[im][None]
        self.weight = jnp.asarray(w / w[w > 0].mean(), jnp.float32)
        self.spacing = tuple(float(v) for v in grid["spacing"])
        self.scale = tuple(scale)

    def features(self, spacing=None) -> jnp.ndarray:
        """``(T_v, P, 5 + C)`` features; ``spacing`` (ds, dx, dy) defaults to the nominal one."""
        sp = jnp.asarray(self.spacing if spacing is None else spacing, jnp.float32)
        off = self.offsets * sp / jnp.asarray(self.scale[2:], jnp.float32)
        tv, p = self.vpar.shape
        return jnp.concatenate(
            [
                self.vpar[..., None],
                jnp.broadcast_to(self.mu[None, :, None], (tv, p, 1)),
                jnp.broadcast_to(off[None], (tv, p, 3)),
                jnp.broadcast_to(self.channel[None], (tv, p, self.channel.shape[-1])),
            ],
            -1,
        )


def _fold(x, patch, n_channels):
    # (ns, Vp, Sp, Xp, Yp, c mu) -> (ns, tv, ts, tx, ty, P) in the PatchGrid point order
    ns, vp, sp, xp, yp, ch = x.shape
    _, pv, ps, px, py = patch
    x = x.reshape(ns, vp // pv, pv, sp // ps, ps, xp // px, px, yp // py, py, n_channels, ch // n_channels)
    x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6, 8, 9, 10)
    return x.reshape(*x.shape[:5], -1)


def _unfold(x, patch, n_channels, n_mu):
    ns, tv, ts, tx, ty, _ = x.shape
    _, pv, ps, px, py = patch
    x = x.reshape(ns, tv, ts, tx, ty, pv, ps, px, py, n_channels, n_mu)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4, 8, 9, 10)
    return x.reshape(ns, tv * pv, ts * ps, tx * px, ty * py, n_channels * n_mu)


class FieldPatchEmbed(eqx.Module):
    """``PatchEmbed`` whose first layer is ``W[t_v, point] = kernel(features(point)) * weight(point)``."""

    grid: PatchGrid
    kernel: MLP
    mix: Optional[MLP]
    act: Callable = eqx.field(static=True)
    patch_size: tuple[int, ...] = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)

    def __init__(
        self,
        padded: Sequence[int],
        patch_size: Sequence[int],
        in_channels: int,
        embed_dim: int,
        grid: dict,
        *,
        key,
        mlp_depth: int = 2,
        mlp_ratio: float = 8.0,
        act_fn=leaky_relu,
        hidden: int = 256,
        depth: int = 2,
    ):
        k_kernel, k_mix = jr.split(key)
        self.patch_size = tuple(patch_size)
        self.grid_size = tuple(s // p for s, p in zip(padded, patch_size))
        self.n_channels = in_channels // len(grid["mu"])
        self.grid = PatchGrid(patch_size, padded, self.n_channels, grid)
        first = int(embed_dim * mlp_ratio) if mlp_depth > 1 else embed_dim
        self.kernel = MLP([5 + self.n_channels] + [hidden] * depth + [first], key=k_kernel, act_fn=silu)
        mix = [first] + [first] * (mlp_depth - 2) + [embed_dim]
        self.mix = MLP(mix, key=k_mix, act_fn=act_fn, use_bias=False) if mlp_depth > 1 else None
        self.act = act_fn

    def __call__(self, x: jnp.ndarray, spacing=None) -> jnp.ndarray:
        w = self.kernel(self.grid.features(spacing)) * self.grid.weight[..., None]
        p = _fold(x, self.patch_size, self.n_channels)
        h = jnp.einsum("ntsxyp,tph->ntsxyh", p, w) / math.sqrt(p.shape[-1])
        return h if self.mix is None else self.mix(self.act(h))


class FieldUnpatch(eqx.Module):
    """DeepONet ``PatchExpand``: ``f(point) = sum_r expansion(token)_r * basis(features(point))_r``."""

    grid: PatchGrid
    expansion: MLP
    basis: MLP
    modulation: Optional[object]
    patch_size: tuple[int, ...] = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_mu: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        padded: Sequence[int],
        patch_size: Sequence[int],
        out_channels: int,
        grid: dict,
        *,
        key,
        mlp_ratio: float = 8.0,
        rank: int = 256,
        hidden: int = 256,
        depth: int = 2,
        cond_dim: Optional[int] = None,
    ):
        k_exp, k_basis, k_mod = jr.split(key, 3)
        self.patch_size = tuple(patch_size)
        self.n_mu = len(grid["mu"])
        self.n_channels = out_channels // self.n_mu
        self.grid = PatchGrid(patch_size, padded, self.n_channels, grid)
        branch_hidden = int(math.prod(patch_size) * mlp_ratio)
        self.expansion = MLP([dim, branch_hidden, rank], key=k_exp, act_fn=leaky_relu)
        self.basis = MLP([5 + self.n_channels] + [hidden] * depth + [rank], key=k_basis, act_fn=silu)
        if cond_dim:
            from neugk_jax.models.swin import Film

            self.modulation = Film(cond_dim, dim, key=k_mod)
        else:
            self.modulation = None

    def __call__(self, z: jnp.ndarray, cond: Optional[jnp.ndarray] = None, spacing=None) -> jnp.ndarray:
        if self.modulation is not None:
            z = self.modulation(z, cond)
        out = jnp.einsum("ntsxyr,tpr->ntsxyp", self.expansion(z), self.basis(self.grid.features(spacing)))
        return _unfold(out, self.patch_size, self.n_channels, self.n_mu)
