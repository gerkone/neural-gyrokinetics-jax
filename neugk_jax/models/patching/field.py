"""Field patching: patch embedding and unpatch whose weights are functions of the point coordinates.

``FieldPatchEmbed`` / ``FieldUnpatch`` are drop-in replacements of ``PatchEmbed`` / ``LinearUnpatch`` on
channel-last inputs ``(*spatial, C)``, with an optional per-sample ``geometry`` (the spacings of the
relative axes) at call time. The coordinates come from a :class:`PointGrid`; ``with_grid`` swaps it, so
one set of weights serves every grid (resolution, data type) whose feature and code shapes match.

Every encoder projects a patch onto a basis of the point coordinates, ``h_k = mean_p w_p b_k(p) x_p``,
and every decoder synthesizes it from the same kind of basis, ``x_p = sum_k c_k b_k(p)``; a channel
MLP maps the projections ``h`` to the token and the token to the codes ``c`` (linear at depth 1).
The bases: ``smooth`` (a filter MLP of the point features times low-order cosines of the position in
the patch) and ``tucker`` (tensor products of per-axis bases, ``ranks`` per axis).
"""

from __future__ import annotations

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

ENCODERS = ("smooth", "tucker")
DECODERS = ENCODERS

FIELD_OPTIONS = {
    # cosine modes per coordinate of the point encoding, cut at the resolution
    "modes": 16,
    # filter mlp: output rank, hidden width, hidden layers
    "rank": 256,
    "hidden": 256,
    "depth": 2,
    "encoder": "smooth",
    "decoder": "smooth",
    # smooth: cosine modes of the position in the patch per axis (default 2 on axes of >= 4 points), decoder code rank
    "code_modes": None,
    "code_rank": 128,
    # tucker: ranks per spatial and folded axis (default the patch and node counts)
    "ranks": None,
    # tucker: per-axis basis (learned, or fixed cosines on the relative axes) and its mlp width
    "axis_basis": "learned",
    "axis_hidden": 64,
    # length of a per-call descriptor (e.g. of the species) fed to the filter mlps as conditioning, 0: none
    "cond_features": 0,
    "zero_init": True,
}


def field_options(options: Mapping) -> dict:
    unknown = set(options) - set(FIELD_OPTIONS)
    if unknown:
        raise ValueError(
            f"unknown field patching options {sorted(unknown)}; one of {sorted(FIELD_OPTIONS)}"
        )
    opts = {**FIELD_OPTIONS, **options}
    if opts["encoder"] not in ENCODERS:
        raise ValueError(f"encoder={opts['encoder']!r}; one of {ENCODERS}")
    if opts["decoder"] not in DECODERS:
        raise ValueError(f"decoder={opts['decoder']!r}; one of {DECODERS}")
    if opts["axis_basis"] not in ("learned", "cosine"):
        raise ValueError(f"axis_basis={opts['axis_basis']!r}; one of learned, cosine")
    return opts


def _n_features(grid: PointGrid, opts: Mapping) -> int:
    return (
        grid.n_coords * opts["modes"] + grid.n_channels + len(grid.rel_axes) + opts["cond_features"]
    )


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


def _point_filter(grid: PointGrid, opts: Mapping, out: int, *, key) -> MLP:
    dims = [_n_features(grid, opts)] + [opts["hidden"]] * opts["depth"] + [out]
    return MLP(dims, key=key, act_fn=silu)


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
        ranks: Sequence[int],
        *,
        key,
        basis: str,
        hidden: int,
        modes: int,
        n_cond: int = 0,
    ):
        if len(ranks) != len(grid.axes):
            raise ValueError(f"{len(ranks)} tucker ranks for {len(grid.axes)} axes")
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


def _tucker_ranks(grid: PointGrid, ranks) -> tuple[int, ...]:
    return tuple(ranks) if ranks is not None else (*grid.patch, *grid.n_fold)


def _axis_basis(grid: PointGrid, opts: Mapping, n_cond: int, *, key) -> tuple[AxisBases, int]:
    ranks = _tucker_ranks(grid, opts["ranks"])
    bases = AxisBases(
        grid,
        ranks,
        key=key,
        basis=opts["axis_basis"],
        hidden=opts["axis_hidden"],
        modes=opts["modes"],
        n_cond=n_cond,
    )
    return bases, math.prod(ranks) * grid.n_channels


class _Field(eqx.Module):
    grid: PointGrid

    def with_grid(self, grid: PointGrid):
        """The same weights on another grid (resolution or data type)."""
        out = copy.copy(self)
        # the grid's static fields change with it, which tree_at keeps
        object.__setattr__(out, "grid", grid)
        return out

    def _features(self, geometry, point_cond) -> jnp.ndarray:
        return _with_cond(self.grid.features(geometry, self.modes), point_cond, self.n_cond)


class FieldPatchEmbed(_Field, GridEncoderBase):
    """``PatchEmbed`` whose per-point weights come from the point coordinates.

    ``smooth``: ``h_kr = mean_p w_p phi_k(p) K_r(features_p) x_p`` for the ``code_modes`` cosines phi_k of
    the position in the patch and the filter MLP K. ``tucker``: the projection onto the tensor product
    of the per-axis bases. Then the channel MLP ``mix`` (linear at ``mlp_depth=1``).
    """

    filters: eqx.Module
    mix: MLP
    patch_size: tuple[int, ...] = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)
    encoder: str = eqx.field(static=True)
    modes: int = eqx.field(static=True)
    n_cond: int = eqx.field(static=True)
    code_modes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        base_resolution: Sequence[int],
        patch_size: Sequence[int],
        in_channels: int,
        embed_dim: int,
        *,
        key,
        mlp_depth: int = 1,
        mlp_ratio: float = 1.0,
        act_fn=leaky_relu,
        grid: Optional[Mapping] = None,
        **options,
    ):
        opts = field_options(options)
        k_kernel, k_mix = jr.split(key)
        self.patch_size = _normalize_patch(patch_size)
        self.grid_size = tuple(s // p for s, p in zip(base_resolution, self.patch_size))
        self.grid = PointGrid(base_resolution, self.patch_size, in_channels, grid)
        self.encoder = opts["encoder"]
        self.modes = opts["modes"]
        self.n_cond = opts["cond_features"]
        self.code_modes = ()
        if self.encoder == "smooth":
            self.code_modes = tuple(opts["code_modes"] or _default_modes(self.patch_size))
            self.filters = _point_filter(self.grid, opts, opts["rank"], key=k_kernel)
            width = opts["rank"] * math.prod(self.code_modes)
        else:
            self.filters, width = _axis_basis(self.grid, opts, self.n_cond, key=k_kernel)
        self.mix = _head(width, embed_dim, mlp_depth, int(embed_dim * mlp_ratio), act_fn, key=k_mix)

    def __call__(self, x: jnp.ndarray, geometry=None, point_cond=None) -> jnp.ndarray:
        grid = self.grid
        p = fold_patches(x, grid.patch)
        if self.encoder == "tucker":
            h = tucker_project(p, grid, self.filters(grid, geometry, point_cond))
        else:
            g, a = _letters(len(grid.patch), grid.abs_axes)
            w = self.filters(self._features(geometry, point_cond)) * grid.weight[..., None]
            phi = cosine_basis(grid.pos, self.code_modes)
            h = jnp.einsum(f"{g}p,{a}pr,pk->{g}kr", p, w, phi, optimize="optimal") / p.shape[-1]
            h = h.reshape(*h.shape[:-2], -1)
        return self.mix(h)


class FieldUnpatch(_Field, GridDecoderBase):
    """Unpatch (as ``LinearUnpatch``) rebuilding every patch from per-token codes.

    The channel MLP ``expansion`` (linear at ``mlp_depth=1``, last layer zero-initialized) maps the
    token to the codes. ``smooth``: ``f(p) = sum_kr phi_k(p) c_kr psi_r(p)`` for the cosines phi_k of
    the position in the patch and the basis MLP psi. ``tucker``: a core of the per-axis ranks,
    synthesized by the per-axis bases.
    """

    expansion: MLP
    basis: eqx.Module
    modulation: Optional[eqx.Module]
    expand_by: tuple[int, ...] = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    target_grid_size: tuple[int, ...] = eqx.field(static=True)
    decoder: str = eqx.field(static=True)
    modes: int = eqx.field(static=True)
    n_cond: int = eqx.field(static=True)
    code_modes: tuple[int, ...] = eqx.field(static=True)
    ranks: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        *,
        key,
        expand_by: Sequence[int],
        out_channels: int,
        mlp_depth: int = 1,
        mlp_ratio: float = 1.0,
        norm: bool = False,
        use_conv: bool = False,
        patch_skip: bool = False,
        cond_dim: Optional[int] = None,
        grid: Optional[Mapping] = None,
        **options,
    ):
        if norm or use_conv or patch_skip:
            raise NotImplementedError("field unpatch has no norm, conv or patch skip")
        opts = field_options(options)
        k_exp, k_basis, k_mod = jr.split(key, 3)
        self.expand_by = _normalize_patch(expand_by)
        self.out_dim = out_channels
        self.target_grid_size = tuple(g * e for g, e in zip(grid_size, self.expand_by))
        self.decoder = opts["decoder"]
        self.grid = PointGrid(self.target_grid_size, self.expand_by, out_channels, grid)
        self.modes = opts["modes"]
        self.n_cond = opts["cond_features"]
        self.code_modes = ()
        self.ranks = ()
        if self.decoder == "smooth":
            self.code_modes = tuple(opts["code_modes"] or _default_modes(self.expand_by))
            self.basis = _point_filter(self.grid, opts, opts["code_rank"], key=k_basis)
            width = opts["code_rank"] * math.prod(self.code_modes)
        else:
            self.basis, width = _axis_basis(self.grid, opts, self.n_cond, key=k_basis)
            self.ranks = self.basis.ranks
        self.expansion = _head(dim, width, mlp_depth, int(dim * mlp_ratio), leaky_relu, key=k_exp)
        if opts["zero_init"]:
            self.expansion = _zero_last(self.expansion)
        if cond_dim:
            from neugk_jax.models.swin import Film

            self.modulation = Film(cond_dim, dim, key=k_mod)
        else:
            self.modulation = None

    def __call__(
        self, z: jnp.ndarray, cond: Optional[jnp.ndarray] = None, geometry=None, point_cond=None
    ) -> jnp.ndarray:
        """``cond`` modulates the tokens (film), ``point_cond`` (``cond_features`` long) the filters."""
        if self.modulation is not None:
            z = self.modulation(z, cond)
        grid = self.grid
        channels = grid.n_channels * math.prod(grid.n_fold)
        c = self.expansion(z)
        if self.decoder == "tucker":
            out = tucker_synthesize(c, grid, self.basis(grid, geometry, point_cond), self.ranks)
        else:
            g, a = _letters(len(grid.patch), grid.abs_axes)
            psi = self.basis(self._features(geometry, point_cond))
            c = c.reshape(*z.shape[:-1], math.prod(self.code_modes), -1)
            phi = cosine_basis(grid.pos, self.code_modes)
            out = (
                jnp.einsum(f"{g}kr,{a}pr,pk->{g}p", c, psi, phi, optimize="optimal") / psi.shape[-1]
            )
        return unfold_patches(out, grid.patch, out_channels=channels)
