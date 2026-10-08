"""Field patching: patch embedding and unpatch whose weights are functions of the point coordinates.

The field layers are drop-in replacements of ``PatchEmbed`` / ``LinearUnpatch`` on channel-last inputs
``(*spatial, C)``, with an optional per-sample ``geometry`` (the spacings of the relative axes) at call
time. The coordinates come from a :class:`PointGrid`; ``with_grid`` swaps it, so one set of weights
serves every grid (resolution, data type) whose feature and code shapes match.

An encoder projects a patch onto a basis of the point coordinates, ``h_k = mean_p w_p b_k(p) x_p``, and
a decoder synthesizes it from the same kind of basis, ``x_p = sum_k c_k b_k(p)``; a channel MLP maps the
projections ``h`` to the token and the token to the codes ``c`` (linear at depth 1). The bases:
``Smooth*`` (a filter MLP of the point features times low-order cosines of the position in the patch)
and ``Tucker*`` (tensor products of per-axis bases, with per-axis ranks).
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

AXIS_BASES = ("learned", "cosine")


def _with_cond(feats: jnp.ndarray, cond, n: int) -> jnp.ndarray:
    if not n:
        return feats
    cond = jnp.zeros((n,)) if cond is None else jnp.asarray(cond, feats.dtype)
    return jnp.concatenate([feats, jnp.broadcast_to(cond, (*feats.shape[:-1], n))], -1)


def _letters(n: int, abs_axes: Sequence[int]) -> tuple[str, str]:
    grid = "abcdefgh"[:n]
    return grid, "".join(grid[i] for i in abs_axes)


def _default_modes(patch: Sequence[int]) -> tuple[int, ...]:
    return tuple(2 if p >= 4 else 1 for p in patch)


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


def cosine_basis(pos: jnp.ndarray, modes: Sequence[int]) -> jnp.ndarray:
    """``(..., prod(modes))`` separable cosines of the cell-centred positions ``pos`` ``(..., n)``."""
    phi = jnp.ones((*pos.shape[:-1], 1))
    for i, m in enumerate(modes):
        b = jnp.cos(jnp.arange(m) * jnp.pi * (pos[..., i : i + 1] + 1) / 2)
        phi = (phi[..., :, None] * b[..., None, :]).reshape(*pos.shape[:-1], -1)
    return phi


class PointFilter(eqx.Module):
    """``(*T_abs, P, out)`` SiLU MLP of the point features (encoded coordinates, channel, scale, condition)."""

    net: MLP
    modes: int = eqx.field(static=True)
    n_cond: int = eqx.field(static=True)

    def __init__(
        self, grid: PointGrid, out: int, *, key, hidden: int, depth: int, modes: int, n_cond: int
    ):
        n_in = grid.n_coords * modes + grid.n_channels + len(grid.rel_axes) + n_cond
        self.net = MLP([n_in, *[hidden] * depth, out], key=key, act_fn=silu)
        self.modes, self.n_cond = modes, n_cond

    def __call__(self, grid: PointGrid, geometry=None, cond=None) -> jnp.ndarray:
        return self.net(_with_cond(grid.features(geometry, self.modes), cond, self.n_cond))


class AxisBases(eqx.Module):
    """Separable bases: per spatial and folded axis a ``(..., p, r)`` basis over the axis points.

    ``learned``: a 1D filter MLP of the encoded axis coordinate (and the log half-width of a relative
    axis), every basis function scaled to unit quadrature rms over the axis points. ``cosine``:
    orthonormal cosines cut at the resolution on the relative axes (absolute and folded axes stay
    learned).
    """

    nets: tuple
    ranks: tuple[int, ...] = eqx.field(static=True)
    modes: int = eqx.field(static=True)
    n_cond: int = eqx.field(static=True)

    def __init__(
        self,
        grid: PointGrid,
        ranks: Optional[Sequence[int]],
        *,
        key,
        basis: str,
        hidden: int,
        modes: int,
        n_cond: int = 0,
    ):
        ranks = tuple(ranks) if ranks is not None else (*grid.patch, *grid.n_fold)
        if len(ranks) != len(grid.axes):
            raise ValueError(f"{len(ranks)} tucker ranks for {len(grid.axes)} axes")
        if basis not in AXIS_BASES:
            raise ValueError(f"axis_basis={basis!r}; one of {AXIS_BASES}")
        self.ranks = tuple(int(r) for r in ranks)
        self.modes = modes
        self.n_cond = n_cond
        nets = []
        for ax, r, k in zip(grid.axes, self.ranks, jr.split(key, len(grid.axes))):
            if basis == "cosine" and ax.kind == "relative":
                nets.append(None)
            else:
                n_in = modes + (ax.kind == "relative") + n_cond
                nets.append(MLP([n_in, hidden, hidden, r], key=k, act_fn=silu))
        self.nets = tuple(nets)

    def __call__(self, grid: PointGrid, geometry=None, cond=None) -> list[jnp.ndarray]:
        out = []
        for k, (ax, net, r) in enumerate(zip(grid.axes, self.nets, self.ranks)):
            if net is None:
                m = jnp.arange(r)
                c = jnp.cos(m * jnp.pi * (ax.coords[..., None] + 1) / 2) * jnp.where(
                    m > 0, math.sqrt(2.0), 1.0
                )
                out.append(c * (m < ax.cap))
            else:
                b = net(_with_cond(grid.axis_features(k, geometry, self.modes), cond, self.n_cond))
                # unit quadrature rms per basis function, so the tensor product keeps unit scale
                w = ax.weight[..., None]
                rms = jnp.sqrt(
                    jnp.sum(w * b**2, -2, keepdims=True)
                    / jnp.maximum(jnp.sum(w, -2, keepdims=True), 1e-6)
                )
                out.append(b / (rms + 1e-6))
        return out

    def width(self, grid: PointGrid) -> int:
        return math.prod(self.ranks) * grid.n_channels


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
        tok = t[k] if k < n and k in grid.abs_axes else ""
        lhs = f"{t}{''.join(src[:n])}z{''.join(src[n:])}"
        rhs = f"{t}{''.join(dst[:n])}z{''.join(dst[n:])}"
        x = jnp.einsum(f"{lhs},{tok}{pts[k]}{rks[k]}->{rhs}", x, b)
        cur = dst
    return x


def tucker_project(x: jnp.ndarray, grid: PointGrid, bases: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Quadrature of the folded patches ``(*T, P*C)`` against the tensor-product basis, ``(*T, prod(r) * c)``."""
    n = len(grid.patch)
    lead = x.shape[:n]
    x = x.reshape(*lead, *grid.patch, grid.n_channels, *grid.n_fold)
    weighted = [b * ax.weight[..., None] for b, ax in zip(bases, grid.axes)]
    out = _mode_products(x, grid, weighted, analysis=True)
    return out.reshape(*lead, -1) / (math.prod(grid.patch) * math.prod(grid.n_fold))


def tucker_synthesize(
    core: jnp.ndarray, grid: PointGrid, bases: Sequence[jnp.ndarray], ranks: Sequence[int]
) -> jnp.ndarray:
    """Folded patches ``(*T, P*C)`` of the cores ``(*T, prod(r) * c)``."""
    n = len(grid.patch)
    lead = core.shape[:n]
    core = core.reshape(*lead, *ranks[:n], grid.n_channels, *ranks[n:])
    return _mode_products(core, grid, bases, analysis=False).reshape(*lead, -1)


class _Field(eqx.Module):
    grid: PointGrid

    def with_grid(self, grid: PointGrid):
        """The same weights on another grid (resolution or data type)."""
        out = copy.copy(self)
        # the grid's static fields change with it, which tree_at keeps
        object.__setattr__(out, "grid", grid)
        return out


class FieldPatchEmbed(_Field, GridEncoderBase):
    """``PatchEmbed`` whose per-point weights come from the point coordinates.

    Subclasses give the ``basis`` and :meth:`project` (patches to their projections onto the basis);
    the channel MLP ``mix`` maps the projections to the token (linear at ``mlp_depth=1``).
    """

    basis: eqx.AbstractVar[eqx.Module]
    mix: eqx.AbstractVar[MLP]
    patch_size: eqx.AbstractVar[tuple[int, ...]]
    grid_size: eqx.AbstractVar[tuple[int, ...]]

    @abc.abstractmethod
    def project(self, patches: jnp.ndarray, geometry=None, point_cond=None) -> jnp.ndarray:
        """``(*T, width)`` projections of the folded patches ``(*T, P*C)``."""

    def __call__(self, x: jnp.ndarray, geometry=None, point_cond=None) -> jnp.ndarray:
        """``point_cond`` (``cond_features`` long) conditions the basis."""
        return self.mix(self.project(fold_patches(x, self.grid.patch), geometry, point_cond))


class FieldUnpatch(_Field, GridDecoderBase):
    """Unpatch (as ``LinearUnpatch``) rebuilding every patch from per-token codes.

    The channel MLP ``expansion`` (linear at ``mlp_depth=1``, last layer zero-initialized with
    ``zero_init``) maps the token to the codes; subclasses give the ``basis`` and :meth:`synthesize`.
    """

    basis: eqx.AbstractVar[eqx.Module]
    expansion: eqx.AbstractVar[MLP]
    modulation: eqx.AbstractVar[Optional[eqx.Module]]
    out_dim: eqx.AbstractVar[int]

    @abc.abstractmethod
    def synthesize(self, codes: jnp.ndarray, geometry=None, point_cond=None) -> jnp.ndarray:
        """``(*T, P*C)`` folded patches of the codes ``(*T, width)``."""

    def __call__(
        self, z: jnp.ndarray, cond: Optional[jnp.ndarray] = None, geometry=None, point_cond=None
    ) -> jnp.ndarray:
        """``cond`` modulates the tokens (film), ``point_cond`` (``cond_features`` long) the basis."""
        if self.modulation is not None:
            z = self.modulation(z, cond)
        grid = self.grid
        out = self.synthesize(self.expansion(z), geometry, point_cond)
        return unfold_patches(
            out, grid.patch, out_channels=grid.n_channels * math.prod(grid.n_fold)
        )


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


def _expansion(dim, width, depth, ratio, zero_init, *, key) -> MLP:
    head = _head(dim, width, depth, int(dim * ratio), leaky_relu, key=key)
    return _zero_last(head) if zero_init else head


class SmoothPatchEmbed(FieldPatchEmbed):
    """``h_kr = mean_p w_p phi_k(p) K_r(p) x_p``: the ``code_modes`` cosines phi_k of the position in the
    patch times the ``rank`` outputs of the point filter K, quadrature weights w_p."""

    basis: PointFilter
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
        rank: int = 256,
        hidden: int = 256,
        depth: int = 2,
        code_modes: Optional[Sequence[int]] = None,
        modes: int = 16,
        cond_features: int = 0,
        mlp_depth: int = 1,
        mlp_ratio: float = 1.0,
        act_fn=leaky_relu,
        grid: Optional[Mapping] = None,
    ):
        k_basis, k_mix = jr.split(key)
        self.patch_size, self.grid_size, self.grid = _embed_grid(
            base_resolution, patch_size, in_channels, grid
        )
        self.code_modes = tuple(code_modes or _default_modes(self.patch_size))
        self.basis = PointFilter(
            self.grid,
            rank,
            key=k_basis,
            hidden=hidden,
            depth=depth,
            modes=modes,
            n_cond=cond_features,
        )
        width = rank * math.prod(self.code_modes)
        self.mix = _head(width, embed_dim, mlp_depth, int(embed_dim * mlp_ratio), act_fn, key=k_mix)

    def project(self, patches, geometry=None, point_cond=None):
        grid = self.grid
        g, a = _letters(len(grid.patch), grid.abs_axes)
        w = self.basis(grid, geometry, point_cond) * grid.weight[..., None]
        phi = cosine_basis(grid.pos, self.code_modes)
        h = jnp.einsum(f"{g}p,{a}pr,pk->{g}kr", patches, w, phi, optimize="optimal")
        return (h / patches.shape[-1]).reshape(*h.shape[:-2], -1)


class SmoothUnpatch(FieldUnpatch):
    """``f(p) = sum_kr phi_k(p) c_kr psi_r(p)``: the cosines phi_k of the position in the patch times the
    ``code_rank`` outputs of the point basis psi, codes from the token."""

    basis: PointFilter
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
        code_rank: int = 128,
        hidden: int = 256,
        depth: int = 2,
        code_modes: Optional[Sequence[int]] = None,
        modes: int = 16,
        cond_features: int = 0,
        zero_init: bool = True,
        mlp_depth: int = 1,
        mlp_ratio: float = 1.0,
        norm: bool = False,
        use_conv: bool = False,
        patch_skip: bool = False,
        cond_dim: Optional[int] = None,
        grid: Optional[Mapping] = None,
    ):
        k_exp, k_basis, k_mod = jr.split(key, 3)
        self.out_dim = out_channels
        self.expand_by, self.target_grid_size, self.grid, self.modulation = _unpatch_init(
            dim,
            grid_size,
            expand_by,
            out_channels,
            grid,
            cond_dim,
            k_mod,
            (norm, use_conv, patch_skip),
        )
        self.code_modes = tuple(code_modes or _default_modes(self.expand_by))
        self.basis = PointFilter(
            self.grid,
            code_rank,
            key=k_basis,
            hidden=hidden,
            depth=depth,
            modes=modes,
            n_cond=cond_features,
        )
        width = code_rank * math.prod(self.code_modes)
        self.expansion = _expansion(dim, width, mlp_depth, mlp_ratio, zero_init, key=k_exp)

    def synthesize(self, codes, geometry=None, point_cond=None):
        grid = self.grid
        g, a = _letters(len(grid.patch), grid.abs_axes)
        psi = self.basis(grid, geometry, point_cond)
        c = codes.reshape(*codes.shape[:-1], math.prod(self.code_modes), -1)
        phi = cosine_basis(grid.pos, self.code_modes)
        out = jnp.einsum(f"{g}kr,{a}pr,pk->{g}p", c, psi, phi, optimize="optimal")
        return out / psi.shape[-1]


class TuckerPatchEmbed(FieldPatchEmbed):
    """Projection of the patch onto the tensor product of per-axis bases (``ranks`` per spatial and
    folded axis, default the patch and node counts; ``axis_basis`` learned or fixed cosines)."""

    basis: AxisBases
    mix: MLP
    patch_size: tuple[int, ...] = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        base_resolution: Sequence[int],
        patch_size: Sequence[int],
        in_channels: int,
        embed_dim: int,
        *,
        key,
        ranks: Optional[Sequence[int]] = None,
        axis_basis: str = "learned",
        axis_hidden: int = 64,
        modes: int = 16,
        cond_features: int = 0,
        mlp_depth: int = 1,
        mlp_ratio: float = 1.0,
        act_fn=leaky_relu,
        grid: Optional[Mapping] = None,
    ):
        k_basis, k_mix = jr.split(key)
        self.patch_size, self.grid_size, self.grid = _embed_grid(
            base_resolution, patch_size, in_channels, grid
        )
        self.basis = AxisBases(
            self.grid,
            ranks,
            key=k_basis,
            basis=axis_basis,
            hidden=axis_hidden,
            modes=modes,
            n_cond=cond_features,
        )
        width = self.basis.width(self.grid)
        self.mix = _head(width, embed_dim, mlp_depth, int(embed_dim * mlp_ratio), act_fn, key=k_mix)

    def project(self, patches, geometry=None, point_cond=None):
        return tucker_project(patches, self.grid, self.basis(self.grid, geometry, point_cond))


class TuckerUnpatch(FieldUnpatch):
    """A core of the per-axis ranks from the token, synthesized by the per-axis bases."""

    basis: AxisBases
    expansion: MLP
    modulation: Optional[eqx.Module]
    out_dim: int = eqx.field(static=True)
    expand_by: tuple[int, ...] = eqx.field(static=True)
    target_grid_size: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        *,
        key,
        expand_by: Sequence[int],
        out_channels: int,
        ranks: Optional[Sequence[int]] = None,
        axis_basis: str = "learned",
        axis_hidden: int = 64,
        modes: int = 16,
        cond_features: int = 0,
        zero_init: bool = True,
        mlp_depth: int = 1,
        mlp_ratio: float = 1.0,
        norm: bool = False,
        use_conv: bool = False,
        patch_skip: bool = False,
        cond_dim: Optional[int] = None,
        grid: Optional[Mapping] = None,
    ):
        k_exp, k_basis, k_mod = jr.split(key, 3)
        self.out_dim = out_channels
        self.expand_by, self.target_grid_size, self.grid, self.modulation = _unpatch_init(
            dim,
            grid_size,
            expand_by,
            out_channels,
            grid,
            cond_dim,
            k_mod,
            (norm, use_conv, patch_skip),
        )
        self.basis = AxisBases(
            self.grid,
            ranks,
            key=k_basis,
            basis=axis_basis,
            hidden=axis_hidden,
            modes=modes,
            n_cond=cond_features,
        )
        width = self.basis.width(self.grid)
        self.expansion = _expansion(dim, width, mlp_depth, mlp_ratio, zero_init, key=k_exp)

    def synthesize(self, codes, geometry=None, point_cond=None):
        return tucker_synthesize(
            codes, self.grid, self.basis(self.grid, geometry, point_cond), self.basis.ranks
        )
