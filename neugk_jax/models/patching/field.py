"""Field patching: patch embedding and unpatch whose weights are functions of the point coordinates.

``FieldPatchEmbed`` / ``FieldUnpatch`` are drop-in replacements of ``PatchEmbed`` / ``PatchExpand`` on
channel-last inputs ``(*spatial, C)``, with an optional per-sample ``geometry`` (the spacings of the
relative axes) at call time. The coordinates come from a :class:`PointGrid`; ``with_grid`` swaps it, so
one set of weights serves every grid (resolution, data type) whose feature and code shapes match.

Encoders: ``kernel`` (a filter MLP of the point features gives rank-R weights per point), ``smooth``
(the kernel filters modulated by a cosine basis of the position in the patch, K x R functionals) or ``tucker``
(per-axis bases, a multilinear projection onto their tensor product). Decoders: ``deeponet`` (branch
codes times a filter-MLP basis over the patch), ``hier`` (anchor sub-blocks with tied codes), ``smooth``
(codes blended over the patch by a cosine basis of the position) or ``tucker`` (a core synthesized by
the per-axis bases).
"""

from __future__ import annotations

import copy
import math
from typing import Callable, Mapping, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.base import GridDecoderBase, GridEncoderBase
from neugk_jax.models.patching.ops import _normalize_patch, fold_patches, unfold_patches
from neugk_jax.models.patching.points import PointGrid, n_encoded
from neugk_jax.models.utils import MLP, Linear, leaky_relu, silu

ENCODERS = ("kernel", "smooth", "tucker")
DECODERS = ("deeponet", "hier", "smooth", "tucker")

FIELD_OPTIONS = {
    # point encoding: fourier (n_freq octaves), cosine (modes, cut at the resolution) or ipe (cell-averaged cosines)
    "encoding": "fourier",
    "n_freq": 5,
    "modes": 16,
    # filter mlp: output rank, hidden width, hidden layers
    "rank": 256,
    "hidden": 256,
    "depth": 2,
    "encoder": "kernel",
    "decoder": "hier",
    "branch": 1024,
    # hier / smooth: anchors per axis, code modes per axis (tied codes), code rank
    "anchors": None,
    "code_modes": None,
    "code_rank": 128,
    # hier block decoder: deeponet (linear in the code) or field (shift-modulated mlp of block-local coordinates)
    "local": "deeponet",
    "local_width": 64,
    "local_depth": 2,
    # tucker: ranks per spatial and folded axis (default the patch and node counts), axis basis (learned or cosine) and its mlp width
    "ranks": None,
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
    if opts["local"] not in ("deeponet", "field"):
        raise ValueError(f"local={opts['local']!r}; one of deeponet, field")
    return opts


def _n_features(grid: PointGrid, opts: Mapping) -> int:
    return (
        n_encoded(grid.n_coords, opts["encoding"], opts["n_freq"], opts["modes"])
        + grid.n_channels
        + len(grid.rel_axes)
        + opts["cond_features"]
    )


def _with_cond(feats: jnp.ndarray, cond, n: int) -> jnp.ndarray:
    if not n:
        return feats
    cond = jnp.zeros((n,)) if cond is None else jnp.asarray(cond, feats.dtype)
    return jnp.concatenate([feats, jnp.broadcast_to(cond, (*feats.shape[:-1], n))], -1)


def _letters(n: int, abs_axes: Sequence[int]) -> tuple[str, str]:
    grid = "abcdefgh"[:n]
    return grid, "".join(grid[i] for i in abs_axes)


def _default_anchors(patch: Sequence[int]) -> tuple[int, ...]:
    return tuple(2 if p >= 4 and p % 2 == 0 else 1 for p in patch)


def unpatch_anchors(
    decoder: str, patch: Sequence[int], anchors: Optional[Sequence[int]] = None
) -> Optional[tuple[int, ...]]:
    """Anchor blocks per axis of an unpatch grid (``hier`` only)."""
    return (
        tuple(anchors or _default_anchors(_normalize_patch(patch))) if decoder == "hier" else None
    )


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


def cosine_basis(pos: jnp.ndarray, modes: Sequence[int]) -> jnp.ndarray:
    """``(..., prod(modes))`` separable cosine basis of the cell-centred positions ``pos`` ``(..., n)``."""
    phi = jnp.ones((*pos.shape[:-1], 1))
    for i, m in enumerate(modes):
        b = jnp.cos(jnp.arange(m) * jnp.pi * (pos[..., i : i + 1] + 1) / 2)
        phi = (phi[..., :, None] * b[..., None, :]).reshape(*pos.shape[:-1], -1)
    return phi


class AxisBases(eqx.Module):
    """Separable bases: per spatial and folded axis a ``(..., p, r)`` basis over the axis points.

    ``learned``: a 1D filter MLP of the encoded axis coordinate (and the log half-width of a relative
    axis), every basis function scaled to unit quadrature rms over the axis points. ``cosine``: orthonormal cosines cut at the resolution on the relative axes (absolute and
    folded axes stay learned).
    """

    nets: tuple
    ranks: tuple[int, ...] = eqx.field(static=True)
    encoding: tuple = eqx.field(static=True)
    n_cond: int = eqx.field(static=True)

    def __init__(
        self,
        grid: PointGrid,
        ranks: Sequence[int],
        *,
        key,
        basis: str,
        hidden: int,
        encoding: tuple,
        n_cond: int = 0,
    ):
        if len(ranks) != len(grid.axes):
            raise ValueError(f"{len(ranks)} tucker ranks for {len(grid.axes)} axes")
        self.ranks = tuple(int(r) for r in ranks)
        self.encoding = encoding
        self.n_cond = n_cond
        nets = []
        for ax, r, k in zip(grid.axes, self.ranks, jr.split(key, len(grid.axes))):
            if basis == "cosine" and ax.kind == "relative":
                nets.append(None)
            else:
                n_in = n_encoded(1, *encoding) + (ax.kind == "relative") + n_cond
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
                b = net(
                    _with_cond(grid.axis_features(k, geometry, *self.encoding), cond, self.n_cond)
                )
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

    ``kernel``: ``h = mean_p w_p K(features_p) x_p`` (rank R), then the channel MLP. ``smooth``:
    ``h_k = mean_p phi_k(p) w_p K(features_p) x_p`` for the ``code_modes`` cosines phi_k of the position
    in the patch (K x R functionals). ``tucker``: the
    multilinear projection onto the per-axis bases, then the channel MLP.
    """

    filters: eqx.Module
    mix: MLP
    act: Callable = eqx.field(static=True)
    patch_size: tuple[int, ...] = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)
    encoder: str = eqx.field(static=True)
    encoding: tuple = eqx.field(static=True)
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
        mlp_depth: int = 2,
        mlp_ratio: float = 8.0,
        act_fn=leaky_relu,
        grid: Optional[Mapping] = None,
        **options,
    ):
        opts = field_options(options)
        k_kernel, k_mix = jr.split(key)
        self.patch_size = _normalize_patch(patch_size)
        self.grid_size = tuple(s // p for s, p in zip(base_resolution, self.patch_size))
        self.grid = PointGrid(base_resolution, self.patch_size, in_channels, grid)
        self.encoding = (opts["encoding"], opts["n_freq"], opts["modes"])
        self.encoder = opts["encoder"]
        self.n_cond = opts["cond_features"]
        self.code_modes = (
            tuple(opts["code_modes"] or _default_modes(self.patch_size))
            if self.encoder == "smooth"
            else ()
        )
        if self.encoder in ("kernel", "smooth"):
            dims = (
                [_n_features(self.grid, opts)] + [opts["hidden"]] * opts["depth"] + [opts["rank"]]
            )
            self.filters = MLP(dims, key=k_kernel, act_fn=silu)
            width = opts["rank"] * max(1, math.prod(self.code_modes))
        else:
            ranks = _tucker_ranks(self.grid, opts["ranks"])
            self.filters = AxisBases(
                self.grid,
                ranks,
                key=k_kernel,
                basis=opts["axis_basis"],
                hidden=opts["axis_hidden"],
                encoding=self.encoding,
                n_cond=self.n_cond,
            )
            width = math.prod(ranks) * self.grid.n_channels
        mix = [width] + [int(embed_dim * mlp_ratio)] * (mlp_depth - 1) + [embed_dim]
        self.mix = MLP(mix, key=k_mix, act_fn=act_fn, use_bias=False)
        self.act = act_fn

    def __call__(self, x: jnp.ndarray, geometry=None, point_cond=None) -> jnp.ndarray:
        p = fold_patches(x, self.grid.patch)
        if self.encoder == "tucker":
            return self.mix(
                tucker_project(p, self.grid, self.filters(self.grid, geometry, point_cond))
            )
        w = (
            self.filters(
                _with_cond(self.grid.features(geometry, *self.encoding), point_cond, self.n_cond)
            )
            * self.grid.weight[..., None]
        )
        g, a = _letters(len(self.grid.patch), self.grid.abs_axes)
        if self.encoder == "smooth":
            phi = cosine_basis(self.grid.pos, self.code_modes)
            h = jnp.einsum(f"{g}p,{a}pr,pk->{g}kr", p, w, phi, optimize="optimal") / p.shape[-1]
            h = h.reshape(*h.shape[:-2], -1)
        else:
            h = jnp.einsum(f"{g}p,{a}pr->{g}r", p, w) / p.shape[-1]
        return self.mix(self.act(h))


class FieldUnpatch(_Field, GridDecoderBase):
    """``PatchExpand`` (unpatch) rebuilding every patch from per-token codes.

    ``deeponet``: ``f(p) = <branch(z), basis(p)>`` over the whole patch.
    ``hier``: the patch splits into ``anchors`` blocks per axis; the branch gives ``prod(code_modes)``
    code modes, mixed into one code per anchor by a cosine basis of the anchor position, and every
    block is a DeepONet of its anchor code over block-relative coordinates.
    ``local="field"``: every hier block is a shift-modulated MLP of its block-local coordinates,
    ``h_1 = act(psi(p) + s_1)``, ``h_l = act(W_l h_{l-1} + s_l)``, ``f(p) = <v, h_L>``, with the shifts
    ``s_l`` and readout ``v`` from the anchor code.
    ``smooth``: as ``hier`` with the cosine basis evaluated at every point instead of the anchors,
    ``f(p) = sum_k phi_k(p) <h_k(z), basis(p)>`` (no blocks, any patch size).
    ``tucker``: the branch gives a core of the per-axis ranks, synthesized by the per-axis bases.
    """

    expansion: MLP
    basis: eqx.Module
    local: Optional[list]
    modulation: Optional[eqx.Module]
    expand_by: tuple[int, ...] = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    target_grid_size: tuple[int, ...] = eqx.field(static=True)
    decoder: str = eqx.field(static=True)
    code_modes: tuple[int, ...] = eqx.field(static=True)
    ranks: tuple[int, ...] = eqx.field(static=True)
    encoding: tuple = eqx.field(static=True)
    n_cond: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        *,
        key,
        expand_by: Sequence[int],
        out_channels: int,
        mlp_depth: int = 1,
        mlp_ratio: float = 8.0,
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
        k_exp, k_basis, k_mod, k_local = jr.split(key, 4)
        self.expand_by = _normalize_patch(expand_by)
        self.out_dim = out_channels
        self.target_grid_size = tuple(g * e for g, e in zip(grid_size, self.expand_by))
        self.decoder = opts["decoder"]
        anchors = unpatch_anchors(self.decoder, self.expand_by, opts["anchors"])
        self.grid = PointGrid(self.target_grid_size, self.expand_by, out_channels, grid, anchors)
        self.encoding = (opts["encoding"], opts["n_freq"], opts["modes"])
        self.n_cond = opts["cond_features"]
        self.code_modes = ()
        self.ranks = ()
        self.local = None
        if self.decoder == "tucker":
            self.ranks = _tucker_ranks(self.grid, opts["ranks"])
            self.basis = AxisBases(
                self.grid,
                self.ranks,
                key=k_basis,
                basis=opts["axis_basis"],
                hidden=opts["axis_hidden"],
                encoding=self.encoding,
                n_cond=self.n_cond,
            )
            width = math.prod(self.ranks) * self.grid.n_channels
        else:
            if self.decoder == "hier":
                self.code_modes = tuple(opts["code_modes"] or anchors)
            elif self.decoder == "smooth":
                self.code_modes = tuple(opts["code_modes"] or _default_modes(self.expand_by))
            rank = opts["rank"] if self.decoder == "deeponet" else opts["code_rank"]
            basis_out = rank
            if self.decoder == "hier" and opts["local"] == "field":
                # code: one shift per layer and the readout; basis: the first-layer coordinate embedding
                w, d = opts["local_width"], opts["local_depth"]
                rank, basis_out = (d + 1) * w, w
                self.local = [Linear(w, w, key=k) for k in jr.split(k_local, d - 1)]
            width = rank * max(1, math.prod(self.code_modes))
            dims = [_n_features(self.grid, opts)] + [opts["hidden"]] * opts["depth"] + [basis_out]
            self.basis = MLP(dims, key=k_basis, act_fn=silu)
        self.expansion = MLP([dim, opts["branch"], width], key=k_exp, act_fn=leaky_relu)
        if opts["zero_init"]:
            self.expansion = _zero_last(self.expansion)
        if cond_dim:
            from neugk_jax.models.swin import Film

            self.modulation = Film(cond_dim, dim, key=k_mod)
        else:
            self.modulation = None

    def _local_field(self, codes: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
        """``(*T, A, Q)`` block fields of the anchor codes ``(*T, A, C)``, rematerialized per slice of the first two token axes."""
        grid, w = self.grid, psi.shape[-1]
        n_lead = min(2, codes.ndim - 2)
        # psi carries the token axes it depends on (the absolute ones), broadcast over the others
        lead = [k for k in range(n_lead) if k in grid.abs_axes]
        shape = [
            codes.shape[k] if k in grid.abs_axes else 1 for k in range(n_lead, len(grid.patch))
        ]

        def block(idx):
            c = codes[tuple(idx)]
            ps = (psi[tuple(idx[k] for k in lead)] if lead else psi).reshape(
                *shape, *psi.shape[-3:]
            )
            h = silu(ps + c[..., None, :w])
            for j, layer in enumerate(self.local, 1):
                h = silu(layer(h) + c[..., None, j * w : (j + 1) * w])
            return jnp.einsum("...qw,...w->...q", h, c[..., -w:]) / w

        sizes = codes.shape[:n_lead]
        idx = jnp.stack(jnp.meshgrid(*(jnp.arange(n) for n in sizes), indexing="ij"), -1).reshape(
            -1, n_lead
        )
        out = jax.lax.map(jax.checkpoint(lambda i: block(tuple(i[k] for k in range(n_lead)))), idx)
        return out.reshape(*sizes, *out.shape[1:])

    def codes(self, z: jnp.ndarray) -> jnp.ndarray:
        """``(..., n_anchor, rank)`` anchor codes of the hier decoder."""
        h = self.expansion(z).reshape(*z.shape[:-1], math.prod(self.code_modes), -1)
        return jnp.einsum("ak,...kr->...ar", cosine_basis(self.grid.unit, self.code_modes), h)

    def __call__(
        self, z: jnp.ndarray, cond: Optional[jnp.ndarray] = None, geometry=None, point_cond=None
    ) -> jnp.ndarray:
        """``cond`` modulates the tokens (film), ``point_cond`` (``cond_features`` long) the filters."""
        if self.modulation is not None:
            z = self.modulation(z, cond)
        grid = self.grid
        channels = grid.n_channels * math.prod(grid.n_fold)
        if self.decoder == "tucker":
            out = tucker_synthesize(
                self.expansion(z), grid, self.basis(grid, geometry, point_cond), self.ranks
            )
            return unfold_patches(out, grid.patch, out_channels=channels)
        psi = self.basis(
            _with_cond(grid.features(geometry, *self.encoding), point_cond, self.n_cond)
        )
        g, a = _letters(len(grid.patch), grid.abs_axes)
        if self.decoder == "deeponet":
            out = jnp.einsum(f"{g}r,{a}pr->{g}p", self.expansion(z), psi) / psi.shape[-1]
        elif self.decoder == "smooth":
            h = self.expansion(z).reshape(*z.shape[:-1], math.prod(self.code_modes), -1)
            phi = cosine_basis(grid.pos, self.code_modes)
            out = (
                jnp.einsum(f"{g}kr,{a}pr,pk->{g}p", h, psi, phi, optimize="optimal") / psi.shape[-1]
            )
        else:
            psi = psi[..., grid.perm, :].reshape(*psi.shape[:-2], grid.n_anchor, -1, psi.shape[-1])
            if self.local is None:
                out = jnp.einsum(f"{g}xr,{a}xqr->{g}xq", self.codes(z), psi) / psi.shape[-1]
            else:
                out = self._local_field(self.codes(z), psi)
            out = out.reshape(*out.shape[:-2], -1)[..., grid.inv]
        return unfold_patches(out, grid.patch, out_channels=channels)
