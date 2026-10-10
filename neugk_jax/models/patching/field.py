"""Field patching: patch embedding and unpatch whose weights are functions of the point coordinates.

The layers are drop-in replacements of ``PatchEmbed`` / ``LinearUnpatch`` on channel-last inputs
``(*spatial, C)``, with an optional per-sample ``geometry`` (the spacings of the relative axes) at call
time. The coordinates of the points of a patch come from a :class:`PointGrid`; ``with_grid`` swaps it,
so one set of weights serves every grid (resolution, data type) of the same layout.

An embedding projects every channel of a patch onto a basis of the point coordinates by quadrature,
``h_k = mean_p w_p b_k(p) x_p``, and a head (linear at ``mlp_depth=1``) maps the projections to the
token; the unpatch maps the token to codes and synthesizes every channel, ``x_p = sum_k c_k b_k(p)``.
The same basis serves every channel.

- :class:`SmoothPatchEmbed` / :class:`SmoothUnpatch`: ``b_kr(p) = phi_k(p) K_r(p)``, fixed low-order
  cosines of the position in the patch times learned point filters, linear in the DCT modes of the patch.
- :class:`DCTPatchEmbed` / :class:`DCTUnpatch`: a Tucker decomposition of the patch whose per-axis
  factors are truncated DCTs re-mixed by a hypernetwork of the axis context.
"""

from __future__ import annotations

import abc
import copy
import math
from typing import Mapping, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.base import GridDecoderBase, GridEncoderBase
from neugk_jax.models.ops import _normalize_patch, fold_patches, unfold_patches
from neugk_jax.models.patching.points import PointGrid
from neugk_jax.models.utils import MLP, leaky_relu, silu

# coordinates and quadrature


def cosine_basis(pos: jnp.ndarray, modes: Sequence[int]) -> jnp.ndarray:
    """``(..., prod(modes))`` separable cosines of the cell-centred positions ``pos`` ``(..., n)``."""
    phi = jnp.ones((*pos.shape[:-1], 1))
    for i, m in enumerate(modes):
        b = jnp.cos(jnp.arange(m) * jnp.pi * (pos[..., i : i + 1] + 1) / 2)
        phi = (phi[..., :, None] * b[..., None, :]).reshape(*pos.shape[:-1], -1)
    return phi


def _cosines(u: jnp.ndarray, n: int, cut: int, orthonormal: bool = False) -> jnp.ndarray:
    """``(..., p, n)`` DCT-II cosines ``cos(m pi (u + 1) / 2)`` of the cell-centred coordinates ``u``,
    zero from ``cut``; ``orthonormal`` scales them to unit mean square."""
    m = jnp.arange(n)
    c = jnp.cos(m * jnp.pi * (u[..., None] + 1) / 2) * (m < cut)
    return c * jnp.where(m > 0, math.sqrt(2.0), 1.0) if orthonormal else c


def _window_coords(ax) -> jnp.ndarray:
    """``(..., p)`` coordinates of an axis within its window: relative axes as they are, absolute and
    folded nodes mapped to [-1, 1] over the window padded by half a node spacing at both ends."""
    u = ax.coords
    if ax.kind == "relative" or u.shape[-1] < 2:
        return u if ax.kind == "relative" else jnp.zeros_like(u)
    lo = u[..., :1] - (u[..., 1:2] - u[..., :1]) / 2
    hi = u[..., -1:] + (u[..., -1:] - u[..., -2:-1]) / 2
    return 2 * (u - lo) / (hi - lo) - 1


def _letters(n: int, abs_axes: Sequence[int]) -> tuple[str, str]:
    grid = "abcdefgh"[:n]
    return grid, "".join(grid[i] for i in abs_axes)


# heads and grids


def _zero_last(mlp: MLP) -> MLP:
    last = mlp.layers[-1].inner
    zeros = (jnp.zeros_like(last.weight), None if last.bias is None else jnp.zeros_like(last.bias))
    return eqx.tree_at(
        lambda m: (m.layers[-1].inner.weight, m.layers[-1].inner.bias),
        mlp,
        zeros,
        is_leaf=lambda x: x is None,
    )


def _head(n_in: int, n_out: int, depth: int, hidden: int, act_fn, *, key) -> MLP:
    return MLP([n_in, *[hidden] * (depth - 1), n_out], key=key, act_fn=act_fn, use_bias=False)


def _expansion(dim, width, depth, ratio, zero_init, *, key) -> MLP:
    head = _head(dim, width, depth, int(dim * ratio), leaky_relu, key=key)
    return _zero_last(head) if zero_init else head


def _embed_grid(base_resolution, patch_size, in_channels, grid):
    patch = _normalize_patch(patch_size)
    grid_size = tuple(s // p for s, p in zip(base_resolution, patch))
    return patch, grid_size, PointGrid(base_resolution, patch, in_channels, grid)


def _unpatch_init(dim, grid_size, expand_by, out_channels, grid, cond_dim, key, flags):
    if any(flags):
        raise NotImplementedError("field unpatch has no norm, conv or patch skip")
    expand_by = _normalize_patch(expand_by)
    target = tuple(g * e for g, e in zip(grid_size, expand_by))
    modulation = None
    if cond_dim:
        from neugk_jax.models.swin import Film

        modulation = Film(cond_dim, dim, key=key)
    return expand_by, target, PointGrid(target, expand_by, out_channels, grid), modulation


# smooth: point filters


def _filter_init(grid: PointGrid, rank: int, key) -> jnp.ndarray:
    """``(*bands, rank)`` coefficients of a point filter, the bands the points per window of ``grid``."""
    bands = tuple(ax.coords.shape[-1] for ax in grid.axes)
    return jr.normal(key, (*bands, rank)) / math.sqrt(math.prod(bands))


def _filter_values(a: jnp.ndarray, grid: PointGrid) -> jnp.ndarray:
    """``(*T_abs, P, rank)`` point filter ``K_r(p) = sum_m A_mr prod_d c_{m_d}(u_d(p))`` at the points of
    ``grid``: the orthonormal tensor-product DCT modes of the window-local coordinates, cut at the modes
    the grid represents."""
    n_ax = len(grid.axes)
    pts, mds = "abcdefgh"[:n_ax], "ijklmnop"[:n_ax]
    k, cur, toks = a, list(mds), ""
    for d, (ax, b) in enumerate(zip(grid.axes, a.shape[:-1])):
        u = _window_coords(ax)
        c = _cosines(u, b, min(ax.cap, u.shape[-1]), orthonormal=True)
        # an absolute axis has its modes per token row
        tok = "ABCDEFGH"[d] if c.ndim == 3 else ""
        new = cur.copy()
        new[d] = pts[d]
        k = jnp.einsum(
            f"{toks}{''.join(cur)}z,{tok}{pts[d]}{mds[d]}->{toks}{tok}{''.join(new)}z", k, c
        )
        cur, toks = new, toks + tok
    return k.reshape(*k.shape[: len(toks)], -1, k.shape[-1])


# dct: per-axis bases and tucker products


def _dct_params(grid, ranks, key, hidden, modes):
    """``(ranks, a0, hyper)``: per spatial and folded axis the mixing ``A0`` (the identity) and the
    hypernetwork of the axis context (last layer at zero), none on a folded axis."""
    ranks = tuple(int(r) for r in (ranks if ranks is not None else (*grid.patch, *grid.n_fold)))
    if len(ranks) != len(grid.axes):
        raise ValueError(f"{len(ranks)} dct ranks for {len(grid.axes)} axes")
    n_modes = max(modes, *ranks)
    a0, hyper = [], []
    for ax, r, k in zip(grid.axes, ranks, jr.split(key, len(grid.axes))):
        a0.append(jnp.eye(n_modes, r))
        n_ctx = {"relative": 1, "absolute": modes, "folded": 0}[ax.kind]
        hyper.append(
            _zero_last(MLP([n_ctx, hidden, n_modes * r], key=k, act_fn=silu)) if n_ctx else None
        )
    return ranks, tuple(a0), tuple(hyper)


def _dct_bases(grid, geometry, ranks, a0, hyper, modes) -> list[jnp.ndarray]:
    """Per spatial and folded axis the ``(..., p, r)`` basis ``B_r(u) = sum_m A_mr c_m(u)`` of the
    window-local coordinate at unit quadrature rms, ``A = A0 + hyper(context)``: the context is the log
    patch scale of a relative axis and the cosine-encoded window centre of an absolute axis."""
    n_modes, out = max(modes, *ranks), []
    for k, (ax, r) in enumerate(zip(grid.axes, ranks)):
        u = _window_coords(ax)
        a = a0[k]
        if hyper[k] is not None:
            if ax.kind == "absolute":
                centre = jnp.mean(ax.coords, -1)
                ctx = jnp.cos(jnp.arange(modes) * jnp.pi * (centre[..., None] + 1) / 2)
            else:
                ctx = grid.scale(geometry)[grid.rel_axes.index(k)][None]
            a = a + hyper[k](ctx).reshape(*ctx.shape[:-1], n_modes, r)
        b = jnp.einsum("...pm,...mr->...pr", _cosines(u, n_modes, min(ax.cap, u.shape[-1])), a)
        w = ax.weight[..., None]
        ms = jnp.sum(w * b**2, -2, keepdims=True) / jnp.maximum(jnp.sum(w, -2, keepdims=True), 1e-6)
        # a basis function cut to zero must not reach the sqrt at zero
        out.append(b / (jnp.sqrt(jnp.maximum(ms, 1e-12)) + 1e-6))
    return out


def _mode_products(
    x: jnp.ndarray, grid: PointGrid, bases: Sequence[jnp.ndarray], analysis: bool
) -> jnp.ndarray:
    """One basis contraction per axis of ``(*T, *slots, c, *folded slots)``: points to ranks (``analysis``) or back."""
    n, m = len(grid.patch), len(grid.n_fold)
    t, pts, rks = "ABCDEFGH"[:n], "abcdefgh"[:n] + "qstu"[:m], "ijklmnop"[:n] + "vwxy"[:m]
    cur = list(pts if analysis else rks)
    for k, b in enumerate(bases):
        src, dst = cur.copy(), cur.copy()
        dst[k] = rks[k] if analysis else pts[k]
        # an absolute axis has a basis per token row
        tok = t[k] if k < n and b.ndim == 3 else ""
        lhs = f"{t}{''.join(src[:n])}z{''.join(src[n:])}"
        rhs = f"{t}{''.join(dst[:n])}z{''.join(dst[n:])}"
        x = jnp.einsum(f"{lhs},{tok}{pts[k]}{rks[k]}->{rhs}", x, b)
        cur = dst
    return x


# base classes


def _with_grid(self, grid: PointGrid):
    """The same weights on another grid (resolution or data type)."""
    out = copy.copy(self)
    # the grid's static fields change with it, which tree_at keeps
    object.__setattr__(out, "grid", grid)
    return out


class FieldPatchEmbed(GridEncoderBase):
    """``PatchEmbed`` whose per-point weights come from the point coordinates of ``grid``.

    Subclasses give :meth:`basis` (its values at the points) and :meth:`project` (the patches to their
    projections onto it); the head ``mix`` maps the projections to the token.
    """

    grid: eqx.AbstractVar[PointGrid]
    mix: eqx.AbstractVar[MLP]
    patch_size: eqx.AbstractVar[tuple[int, ...]]
    grid_size: eqx.AbstractVar[tuple[int, ...]]

    with_grid = _with_grid

    @abc.abstractmethod
    def basis(self, geometry=None):
        """The basis at the points of the grid."""

    @abc.abstractmethod
    def project(self, patches: jnp.ndarray, geometry=None) -> jnp.ndarray:
        """``(*T, width)`` projections of the folded patches ``(*T, P * C)``."""

    def __call__(self, x: jnp.ndarray, geometry=None) -> jnp.ndarray:
        return self.mix(self.project(fold_patches(x, self.grid.patch), geometry))


class FieldUnpatch(GridDecoderBase):
    """Unpatch (as ``LinearUnpatch``) rebuilding every patch from per-token codes.

    The head ``expansion`` (last layer zero-initialized with ``zero_init``) maps the token to the
    codes; subclasses give :meth:`basis` and :meth:`synthesize`.
    """

    grid: eqx.AbstractVar[PointGrid]
    expansion: eqx.AbstractVar[MLP]
    modulation: eqx.AbstractVar[Optional[eqx.Module]]
    out_dim: eqx.AbstractVar[int]

    with_grid = _with_grid

    @abc.abstractmethod
    def basis(self, geometry=None):
        """The basis at the points of the grid."""

    @abc.abstractmethod
    def synthesize(self, codes: jnp.ndarray, geometry=None) -> jnp.ndarray:
        """``(*T, P * C)`` folded patches of the codes ``(*T, width)``."""

    def __call__(
        self, z: jnp.ndarray, cond: Optional[jnp.ndarray] = None, geometry=None
    ) -> jnp.ndarray:
        """``cond`` modulates the tokens (film)."""
        if self.modulation is not None:
            z = self.modulation(z, cond)
        grid = self.grid
        out = self.synthesize(self.expansion(z), geometry)
        return unfold_patches(
            out, grid.patch, out_channels=grid.n_channels * math.prod(grid.n_fold)
        )


# smooth layers


class SmoothPatchEmbed(FieldPatchEmbed):
    """Smooth field patch embedding: ``h_kcr = mean_p w_p phi_k(p) K_r(p) x_pc`` per channel ``c``.

    ``phi_k`` are the ``code_modes`` low-order cosines of the position in the patch (the patch mean and
    its first variations along every axis) and ``K_r`` the ``rank`` point filters shared by all
    channels, learned linear combinations of the DCT modes of the patch (``filter``), so band-limited
    to the modes the patch resolves. Like PaiNN's filters, linear in a fixed band-limited basis; unlike
    them, the basis is the tensor-product DCT of the offsets in the patch (not a radial basis of the
    distance) and the filters pool the points of a patch into its token rather than pass messages
    between nodes. ``w_p`` are the quadrature weights; the head ``mix`` maps the
    ``code_modes x C x rank`` projections to the token.
    """

    grid: PointGrid
    filter: jnp.ndarray
    mix: MLP
    patch_size: tuple[int, ...] = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)
    code_modes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        base_resolution: Sequence[int],
        patch_size: Sequence[int],
        in_channels: int,
        embed_dim: int,
        *,
        key,
        rank: int = 96,
        code_modes: Optional[Sequence[int]] = None,
        mlp_depth: int = 1,
        mlp_ratio: float = 1.0,
        act_fn=leaky_relu,
        grid: Optional[Mapping] = None,
    ):
        k_filter, k_mix = jr.split(key)
        self.patch_size, self.grid_size, self.grid = _embed_grid(
            base_resolution, patch_size, in_channels, grid
        )
        self.code_modes = tuple(code_modes or (2 if p >= 4 else 1 for p in self.patch_size))
        self.filter = _filter_init(self.grid, rank, k_filter)
        width = math.prod(self.code_modes) * self.grid.n_channels * rank
        self.mix = _head(width, embed_dim, mlp_depth, int(embed_dim * mlp_ratio), act_fn, key=k_mix)

    def basis(self, geometry=None):
        """``(*T_abs, P, rank)`` filter values ``K_r(p)``."""
        return _filter_values(self.filter, self.grid)

    def project(self, patches, geometry=None):
        grid = self.grid
        g, a = _letters(len(grid.patch), grid.abs_axes)
        w = self.basis(geometry) * grid.weight[..., None]
        phi = cosine_basis(grid.pos, self.code_modes)
        # (*T, P * C) in (spatial, channel, folded) order -> (*T, P, C)
        x = patches.reshape(*patches.shape[:-1], -1, grid.n_channels, math.prod(grid.n_fold))
        x = jnp.moveaxis(x, -2, -1).reshape(*patches.shape[:-1], -1, grid.n_channels)
        h = jnp.einsum(f"{g}pz,{a}pr,pk->{g}kzr", x, w, phi, optimize="optimal")
        return (h / w.shape[-2]).reshape(*h.shape[: len(g)], -1)


class SmoothUnpatch(FieldUnpatch):
    """Smooth field unpatch: ``x_pc = sum_kr c_kcr phi_k(p) psi_r(p) / rank``, the codes ``c`` from the
    token, ``phi_k`` the low-order cosines of the position in the patch and ``psi_r`` a point filter
    shared by all channels (as in :class:`SmoothPatchEmbed`)."""

    grid: PointGrid
    filter: jnp.ndarray
    expansion: MLP
    modulation: Optional[eqx.Module]
    out_dim: int = eqx.field(static=True)
    expand_by: tuple[int, ...] = eqx.field(static=True)
    target_grid_size: tuple[int, ...] = eqx.field(static=True)
    code_modes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        *,
        key,
        expand_by: Sequence[int],
        out_channels: int,
        rank: int = 96,
        code_modes: Optional[Sequence[int]] = None,
        zero_init: bool = True,
        mlp_depth: int = 1,
        mlp_ratio: float = 1.0,
        norm: bool = False,
        use_conv: bool = False,
        patch_skip: bool = False,
        cond_dim: Optional[int] = None,
        grid: Optional[Mapping] = None,
    ):
        k_exp, k_filter, k_mod = jr.split(key, 3)
        self.out_dim = out_channels
        flags = (norm, use_conv, patch_skip)
        self.expand_by, self.target_grid_size, self.grid, self.modulation = _unpatch_init(
            dim, grid_size, expand_by, out_channels, grid, cond_dim, k_mod, flags
        )
        self.code_modes = tuple(code_modes or (2 if p >= 4 else 1 for p in self.expand_by))
        self.filter = _filter_init(self.grid, rank, k_filter)
        width = math.prod(self.code_modes) * self.grid.n_channels * rank
        self.expansion = _expansion(dim, width, mlp_depth, mlp_ratio, zero_init, key=k_exp)

    def basis(self, geometry=None):
        """``(*T_abs, P, rank)`` filter values ``psi_r(p)``."""
        return _filter_values(self.filter, self.grid)

    def synthesize(self, codes, geometry=None):
        grid = self.grid
        g, a = _letters(len(grid.patch), grid.abs_axes)
        psi = self.basis(geometry)
        c = codes.reshape(*codes.shape[:-1], math.prod(self.code_modes), grid.n_channels, -1)
        phi = cosine_basis(grid.pos, self.code_modes)
        out = jnp.einsum(f"{g}kzr,{a}pr,pk->{g}pz", c, psi, phi, optimize="optimal")
        out = out.reshape(*out.shape[:-2], -1, math.prod(grid.n_fold), grid.n_channels)
        return jnp.moveaxis(out, -1, -2).reshape(*codes.shape[:-1], -1) / psi.shape[-1]


# dct layers


class DCTPatchEmbed(FieldPatchEmbed):
    """DCT field patch embedding: a Tucker decomposition of every patch with DCT factors.

    The core ``h = x x_1 B^1 x_2 ... x_n B^n`` contracts every spatial and folded axis of the patch
    with its basis ``B^d`` (``ranks[d]`` functions, weighted by the quadrature weights) and keeps the
    channels; the bases are DCT-II modes of the window-local coordinates re-mixed by a hypernetwork of
    the axis context (the log patch scale of a relative axis, the window centre of an absolute axis),
    starting at the first ``ranks[d]`` modes. The head ``mix`` maps the ``prod(ranks) x C`` core to the
    token.
    """

    grid: PointGrid
    a0: tuple
    hyper: tuple
    mix: MLP
    patch_size: tuple[int, ...] = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)
    ranks: tuple[int, ...] = eqx.field(static=True)
    modes: int = eqx.field(static=True)

    def __init__(
        self,
        base_resolution: Sequence[int],
        patch_size: Sequence[int],
        in_channels: int,
        embed_dim: int,
        *,
        key,
        ranks: Optional[Sequence[int]] = None,
        hidden: int = 64,
        modes: int = 16,
        mlp_depth: int = 1,
        mlp_ratio: float = 1.0,
        act_fn=leaky_relu,
        grid: Optional[Mapping] = None,
    ):
        k_bases, k_mix = jr.split(key)
        self.patch_size, self.grid_size, self.grid = _embed_grid(
            base_resolution, patch_size, in_channels, grid
        )
        self.modes = modes
        self.ranks, self.a0, self.hyper = _dct_params(self.grid, ranks, k_bases, hidden, modes)
        width = math.prod(self.ranks) * self.grid.n_channels
        self.mix = _head(width, embed_dim, mlp_depth, int(embed_dim * mlp_ratio), act_fn, key=k_mix)

    def basis(self, geometry=None):
        """Per spatial and folded axis the ``(..., p, r)`` basis ``B^d``."""
        return _dct_bases(self.grid, geometry, self.ranks, self.a0, self.hyper, self.modes)

    def project(self, patches, geometry=None):
        grid, lead = self.grid, patches.shape[: len(self.grid.patch)]
        x = patches.reshape(*lead, *grid.patch, grid.n_channels, *grid.n_fold)
        weighted = [b * ax.weight[..., None] for b, ax in zip(self.basis(geometry), grid.axes)]
        out = _mode_products(x, grid, weighted, analysis=True)
        return out.reshape(*lead, -1) / (math.prod(grid.patch) * math.prod(grid.n_fold))


class DCTUnpatch(FieldUnpatch):
    """DCT field unpatch: a Tucker core per token from the token, synthesized by the per-axis DCT bases
    (``x = c x_1 B^1 x_2 ... x_n B^n``, as in :class:`DCTPatchEmbed`)."""

    grid: PointGrid
    a0: tuple
    hyper: tuple
    expansion: MLP
    modulation: Optional[eqx.Module]
    out_dim: int = eqx.field(static=True)
    expand_by: tuple[int, ...] = eqx.field(static=True)
    target_grid_size: tuple[int, ...] = eqx.field(static=True)
    ranks: tuple[int, ...] = eqx.field(static=True)
    modes: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        *,
        key,
        expand_by: Sequence[int],
        out_channels: int,
        ranks: Optional[Sequence[int]] = None,
        hidden: int = 64,
        modes: int = 16,
        zero_init: bool = True,
        mlp_depth: int = 1,
        mlp_ratio: float = 1.0,
        norm: bool = False,
        use_conv: bool = False,
        patch_skip: bool = False,
        cond_dim: Optional[int] = None,
        grid: Optional[Mapping] = None,
    ):
        k_exp, k_bases, k_mod = jr.split(key, 3)
        self.out_dim = out_channels
        flags = (norm, use_conv, patch_skip)
        self.expand_by, self.target_grid_size, self.grid, self.modulation = _unpatch_init(
            dim, grid_size, expand_by, out_channels, grid, cond_dim, k_mod, flags
        )
        self.modes = modes
        self.ranks, self.a0, self.hyper = _dct_params(self.grid, ranks, k_bases, hidden, modes)
        width = math.prod(self.ranks) * self.grid.n_channels
        self.expansion = _expansion(dim, width, mlp_depth, mlp_ratio, zero_init, key=k_exp)

    def basis(self, geometry=None):
        """Per spatial and folded axis the ``(..., p, r)`` basis ``B^d``."""
        return _dct_bases(self.grid, geometry, self.ranks, self.a0, self.hyper, self.modes)

    def synthesize(self, codes, geometry=None):
        grid, n = self.grid, len(self.grid.patch)
        core = codes.reshape(*codes.shape[:n], *self.ranks[:n], grid.n_channels, *self.ranks[n:])
        out = _mode_products(core, grid, self.basis(geometry), analysis=False)
        return out.reshape(*codes.shape[:n], -1)
