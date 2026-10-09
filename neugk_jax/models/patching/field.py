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
  cosines of the position in the patch times learned point filters.
- :class:`DCTPatchEmbed` / :class:`DCTUnpatch`: a Tucker decomposition of the patch whose per-axis
  factors are truncated DCTs, optionally re-mixed by a hypernetwork of the axis context.
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


def _expansion(dim, width, depth, ratio, zero_init, *, key) -> MLP:
    head = _head(dim, width, depth, int(dim * ratio), leaky_relu, key=key)
    return _zero_last(head) if zero_init else head


def cosine_basis(pos: jnp.ndarray, modes: Sequence[int]) -> jnp.ndarray:
    """``(..., prod(modes))`` separable cosines of the cell-centred positions ``pos`` ``(..., n)``."""
    phi = jnp.ones((*pos.shape[:-1], 1))
    for i, m in enumerate(modes):
        b = jnp.cos(jnp.arange(m) * jnp.pi * (pos[..., i : i + 1] + 1) / 2)
        phi = (phi[..., :, None] * b[..., None, :]).reshape(*pos.shape[:-1], -1)
    return phi


def _dct_modes(u: jnp.ndarray, n: int, cut: int) -> jnp.ndarray:
    """``(..., p, n)`` orthonormal DCT-II modes of the cell-centred coordinates ``u``, zero from ``cut``."""
    m = jnp.arange(n)
    c = jnp.cos(m * jnp.pi * (u[..., None] + 1) / 2) * jnp.where(m > 0, math.sqrt(2.0), 1.0)
    return c * (m < cut)


def _window_coords(ax) -> jnp.ndarray:
    """``(..., p)`` coordinates of an axis within its window: relative axes as they are, absolute and
    folded nodes mapped to [-1, 1] over the window padded by half a node spacing at both ends."""
    u = ax.coords
    if ax.kind == "relative" or u.shape[-1] < 2:
        return u if ax.kind == "relative" else jnp.zeros_like(u)
    lo = u[..., :1] - (u[..., 1:2] - u[..., :1]) / 2
    hi = u[..., -1:] + (u[..., -1:] - u[..., -2:-1]) / 2
    return 2 * (u - lo) / (hi - lo) - 1


def _unit_rms(b: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
    """``b`` ``(..., p, r)`` with every basis function at unit quadrature rms over the points."""
    w = weight[..., None]
    ms = jnp.sum(w * b**2, -2, keepdims=True) / jnp.maximum(jnp.sum(w, -2, keepdims=True), 1e-6)
    # a basis function cut to zero must not reach the sqrt at zero
    return b / (jnp.sqrt(jnp.maximum(ms, 1e-12)) + 1e-6)


def _split_channels(x: jnp.ndarray, grid: PointGrid) -> jnp.ndarray:
    """Folded patches ``(*T, P * C)`` as ``(*T, P, C)``, the points in :class:`PointGrid` order."""
    x = x.reshape(*x.shape[:-1], -1, grid.n_channels, math.prod(grid.n_fold))
    return jnp.moveaxis(x, -2, -1).reshape(*x.shape[:-3], -1, grid.n_channels)


def _merge_channels(x: jnp.ndarray, grid: PointGrid) -> jnp.ndarray:
    x = x.reshape(*x.shape[:-2], -1, math.prod(grid.n_fold), grid.n_channels)
    return jnp.moveaxis(x, -1, -2).reshape(*x.shape[:-3], -1)


class PointFilter(eqx.Module):
    """``(*T_abs, P, out)`` SiLU MLP of the point features (encoded coordinates, log patch scale)."""

    net: MLP
    modes: int = eqx.field(static=True)

    def __init__(self, grid: PointGrid, out: int, *, key, hidden: int, depth: int, modes: int):
        n_in = grid.n_coords * modes + len(grid.rel_axes)
        self.net = MLP([n_in, *[hidden] * depth, out], key=key, act_fn=silu)
        self.modes = modes

    def __call__(self, grid: PointGrid, geometry=None) -> jnp.ndarray:
        return self.net(grid.features(geometry, self.modes))


class CosineFilter(eqx.Module):
    """``(*T_abs, P, out)`` band-limited point filter, linear in the DCT modes of the patch.

    ``K_r(p) = sum_m A_mr prod_d c_{m_d}(u_d(p))``: a learned combination of the orthonormal
    tensor-product DCT-II modes of the window-local coordinates, ``bands`` modes per spatial and
    folded axis (default the points per window of the construction grid), cut at the modes the grid
    represents.
    """

    a: jnp.ndarray
    bands: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, grid: PointGrid, out: int, *, key, bands: Optional[Sequence[int]] = None):
        self.bands = tuple(bands) if bands else tuple(ax.coords.shape[-1] for ax in grid.axes)
        self.a = jr.normal(key, (*self.bands, out)) / math.sqrt(math.prod(self.bands))

    def __call__(self, grid: PointGrid, geometry=None) -> jnp.ndarray:
        n_ax = len(grid.axes)
        pts, mds = "abcdefgh"[:n_ax], "ijklmnop"[:n_ax]
        k, cur, toks = self.a, list(mds), ""
        for d, (ax, b) in enumerate(zip(grid.axes, self.bands)):
            u = _window_coords(ax)
            c = _dct_modes(u, b, min(ax.cap, u.shape[-1]))
            # an absolute axis has its modes per token row
            tok = "ABCDEFGH"[d] if c.ndim == 3 else ""
            new = cur.copy()
            new[d] = pts[d]
            k = jnp.einsum(
                f"{toks}{''.join(cur)}z,{tok}{pts[d]}{mds[d]}->{toks}{tok}{''.join(new)}z", k, c
            )
            cur, toks = new, toks + tok
        return k.reshape(*k.shape[: len(toks)], -1, k.shape[-1])


class DCTBases(eqx.Module):
    """Per spatial and folded axis a ``(..., p, r)`` basis ``B_r(u) = sum_m A_mr c_m(u)`` of the
    window-local coordinate ``u``, ``c_m`` the DCT-II modes cut at the ones the grid represents.

    ``A`` is the identity (the first ``r`` modes), or with ``learned`` ``A = A0 + hyper(context)``,
    ``A0`` starting at the identity and the hypernetwork's last layer at zero. The context of a
    relative axis is the log patch scale (with ``position`` also the cosine-encoded patch centre over
    the box, a basis per token), of an absolute axis the cosine-encoded window centre; a folded axis
    has ``A0`` alone. Every basis function is scaled to unit quadrature rms.
    """

    a0: tuple
    hyper: tuple
    ranks: tuple[int, ...] = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    encoding: int = eqx.field(static=True)
    learned: bool = eqx.field(static=True)
    position: bool = eqx.field(static=True)

    def __init__(
        self,
        grid: PointGrid,
        ranks: Optional[Sequence[int]],
        *,
        key,
        hidden: int,
        modes: int,
        learned: bool,
        position: bool,
    ):
        ranks = tuple(ranks) if ranks is not None else (*grid.patch, *grid.n_fold)
        if len(ranks) != len(grid.axes):
            raise ValueError(f"{len(ranks)} dct ranks for {len(grid.axes)} axes")
        if position and not learned:
            raise ValueError("the patch position needs the learned dct")
        self.ranks = tuple(int(r) for r in ranks)
        self.n_modes, self.encoding = max(modes, *self.ranks), modes
        self.learned, self.position = learned, position
        a0, hyper = [], []
        if learned:
            for ax, r, k in zip(grid.axes, self.ranks, jr.split(key, len(grid.axes))):
                a0.append(jnp.eye(self.n_modes, r))
                n_ctx = {"relative": 1 + position * modes, "absolute": modes, "folded": 0}[ax.kind]
                net = MLP([n_ctx, hidden, self.n_modes * r], key=k, act_fn=silu) if n_ctx else None
                hyper.append(_zero_last(net) if net is not None else None)
        self.a0, self.hyper = tuple(a0), tuple(hyper)

    def _encoded(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.cos(jnp.arange(self.encoding) * jnp.pi * (x[..., None] + 1) / 2)

    def _context(self, grid: PointGrid, k: int, geometry) -> jnp.ndarray:
        ax = grid.axes[k]
        if ax.kind == "absolute":
            return self._encoded(jnp.mean(ax.coords, -1))
        ctx = grid.scale(geometry)[grid.rel_axes.index(k)][None]
        if not self.position:
            return ctx
        centres = grid.centres(k)
        return jnp.concatenate(
            [jnp.broadcast_to(ctx, (len(centres), 1)), self._encoded(centres)], -1
        )

    def __call__(self, grid: PointGrid, geometry=None) -> list[jnp.ndarray]:
        out = []
        for k, (ax, r) in enumerate(zip(grid.axes, self.ranks)):
            u = _window_coords(ax)
            modes = _dct_modes(u, self.n_modes, min(ax.cap, u.shape[-1]))
            a = self.a0[k] if self.learned else jnp.eye(self.n_modes, r)
            if self.learned and self.hyper[k] is not None:
                ctx = self._context(grid, k, geometry)
                a = a + self.hyper[k](ctx).reshape(*ctx.shape[:-1], self.n_modes, r)
            out.append(_unit_rms(jnp.einsum("...pm,...mr->...pr", modes, a), ax.weight))
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
        # a basis per token along its axis: absolute axes, and relative ones with the patch position
        tok = t[k] if k < n and b.ndim == 3 else ""
        lhs = f"{t}{''.join(src[:n])}z{''.join(src[n:])}"
        rhs = f"{t}{''.join(dst[:n])}z{''.join(dst[n:])}"
        x = jnp.einsum(f"{lhs},{tok}{pts[k]}{rks[k]}->{rhs}", x, b)
        cur = dst
    return x


def tucker_project(x: jnp.ndarray, grid: PointGrid, bases: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Tucker cores ``(*T, prod(r) * c)`` of the folded patches ``(*T, P * C)``: weighted mode products."""
    n = len(grid.patch)
    lead = x.shape[:n]
    x = x.reshape(*lead, *grid.patch, grid.n_channels, *grid.n_fold)
    weighted = [b * ax.weight[..., None] for b, ax in zip(bases, grid.axes)]
    out = _mode_products(x, grid, weighted, analysis=True)
    return out.reshape(*lead, -1) / (math.prod(grid.patch) * math.prod(grid.n_fold))


def tucker_synthesize(
    core: jnp.ndarray, grid: PointGrid, bases: Sequence[jnp.ndarray], ranks: Sequence[int]
) -> jnp.ndarray:
    """Folded patches ``(*T, P * C)`` of the Tucker cores ``(*T, prod(r) * c)``."""
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
    def project(self, patches: jnp.ndarray, geometry=None) -> jnp.ndarray:
        """``(*T, width)`` projections of the folded patches ``(*T, P * C)``."""

    def __call__(self, x: jnp.ndarray, geometry=None) -> jnp.ndarray:
        return self.mix(self.project(fold_patches(x, self.grid.patch), geometry))


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


def _smooth_filter(grid, rank, key, hidden, depth, modes, band_limited) -> eqx.Module:
    if band_limited:
        return CosineFilter(grid, rank, key=key)
    return PointFilter(grid, rank, key=key, hidden=hidden, depth=depth, modes=modes)


class SmoothPatchEmbed(FieldPatchEmbed):
    """Smooth field patch embedding: ``h_kcr = mean_p w_p phi_k(p) K_r(p) x_pc`` per channel ``c``.

    ``phi_k`` are the ``code_modes`` low-order cosines of the position in the patch (the patch mean and
    its first variations along every axis) and ``K_r`` the ``rank`` outputs of a point filter shared by
    all channels: a SiLU MLP of the encoded point coordinates and log patch scales, or with
    ``band_limited`` a :class:`CosineFilter` (linear in the DCT modes of the patch). ``w_p`` are the
    quadrature weights; the head ``mix`` maps the ``code_modes x C x rank`` projections to the token.
    """

    basis: eqx.Module
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
        hidden: int = 256,
        depth: int = 2,
        code_modes: Optional[Sequence[int]] = None,
        modes: int = 16,
        band_limited: bool = False,
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
        self.basis = _smooth_filter(self.grid, rank, k_basis, hidden, depth, modes, band_limited)
        width = math.prod(self.code_modes) * self.grid.n_channels * rank
        self.mix = _head(width, embed_dim, mlp_depth, int(embed_dim * mlp_ratio), act_fn, key=k_mix)

    def project(self, patches, geometry=None):
        grid = self.grid
        g, a = _letters(len(grid.patch), grid.abs_axes)
        w = self.basis(grid, geometry) * grid.weight[..., None]
        phi = cosine_basis(grid.pos, self.code_modes)
        x = _split_channels(patches, grid)
        h = jnp.einsum(f"{g}pz,{a}pr,pk->{g}kzr", x, w, phi, optimize="optimal")
        return (h / w.shape[-2]).reshape(*h.shape[: len(g)], -1)


class SmoothUnpatch(FieldUnpatch):
    """Smooth field unpatch: ``x_pc = sum_kr c_kcr phi_k(p) psi_r(p) / rank``, the codes ``c`` from the
    token, ``phi_k`` the low-order cosines of the position in the patch and ``psi_r`` a point filter
    shared by all channels (as in :class:`SmoothPatchEmbed`)."""

    basis: eqx.Module
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
        hidden: int = 256,
        depth: int = 2,
        code_modes: Optional[Sequence[int]] = None,
        modes: int = 16,
        band_limited: bool = False,
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
        self.basis = _smooth_filter(self.grid, rank, k_basis, hidden, depth, modes, band_limited)
        width = math.prod(self.code_modes) * self.grid.n_channels * rank
        self.expansion = _expansion(dim, width, mlp_depth, mlp_ratio, zero_init, key=k_exp)

    def synthesize(self, codes, geometry=None):
        grid = self.grid
        g, a = _letters(len(grid.patch), grid.abs_axes)
        psi = self.basis(grid, geometry)
        c = codes.reshape(*codes.shape[:-1], math.prod(self.code_modes), grid.n_channels, -1)
        phi = cosine_basis(grid.pos, self.code_modes)
        out = jnp.einsum(f"{g}kzr,{a}pr,pk->{g}pz", c, psi, phi, optimize="optimal")
        return _merge_channels(out, grid) / psi.shape[-1]


class DCTPatchEmbed(FieldPatchEmbed):
    """DCT field patch embedding: a Tucker decomposition of every patch with DCT factors.

    The core ``h = x x_1 B^1 x_2 ... x_n B^n`` contracts every spatial and folded axis of the patch
    with its basis ``B^d`` (``ranks[d]`` functions, weighted by the quadrature weights) and keeps the
    channels; the bases are the first DCT-II modes of the window-local coordinates, or with
    ``learned`` the modes re-mixed by a hypernetwork of the axis context (:class:`DCTBases`). The head
    ``mix`` maps the ``prod(ranks) x C`` core to the token.
    """

    basis: DCTBases
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
        learned: bool = True,
        position: bool = False,
        hidden: int = 64,
        modes: int = 16,
        mlp_depth: int = 1,
        mlp_ratio: float = 1.0,
        act_fn=leaky_relu,
        grid: Optional[Mapping] = None,
    ):
        k_basis, k_mix = jr.split(key)
        self.patch_size, self.grid_size, self.grid = _embed_grid(
            base_resolution, patch_size, in_channels, grid
        )
        self.basis = DCTBases(
            self.grid,
            ranks,
            key=k_basis,
            hidden=hidden,
            modes=modes,
            learned=learned,
            position=position,
        )
        width = self.basis.width(self.grid)
        self.mix = _head(width, embed_dim, mlp_depth, int(embed_dim * mlp_ratio), act_fn, key=k_mix)

    def project(self, patches, geometry=None):
        return tucker_project(patches, self.grid, self.basis(self.grid, geometry))


class DCTUnpatch(FieldUnpatch):
    """DCT field unpatch: a Tucker core per token from the token, synthesized by the per-axis DCT bases
    (``x = c x_1 B^1 x_2 ... x_n B^n``, as in :class:`DCTPatchEmbed`)."""

    basis: DCTBases
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
        learned: bool = True,
        position: bool = False,
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
        self.basis = DCTBases(
            self.grid,
            ranks,
            key=k_basis,
            hidden=hidden,
            modes=modes,
            learned=learned,
            position=position,
        )
        width = self.basis.width(self.grid)
        self.expansion = _expansion(dim, width, mlp_depth, mlp_ratio, zero_init, key=k_exp)

    def synthesize(self, codes, geometry=None):
        return tucker_synthesize(
            codes, self.grid, self.basis(self.grid, geometry), self.basis.ranks
        )
