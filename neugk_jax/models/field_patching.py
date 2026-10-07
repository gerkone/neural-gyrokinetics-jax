"""Field patching: patch embedding and unpatch with weights generated from point coordinates.

``FieldPatchEmbed`` / ``FieldUnpatch`` are drop-in replacements of ``PatchEmbed`` / ``PatchExpand`` on
channel-last inputs ``(*spatial, C)``, with an optional per-sample ``geometry`` (the spacings of the
relative axes) at call time. A grid spec describes the coordinates: every spatial axis is ``relative``
(offset from the centre of its block, in [-1, 1], with a physical spacing) or ``absolute`` (node values
mapped to [-1, 1], with quadrature weights), and absolute axes may be folded into the channels as
``(c, *folded)``. Without a spec every axis is relative with unit spacing. Every point of a patch is
encoded (fourier or cosine features of its coordinates, its channel, the log physical half-width of its
block) and filter MLPs of these features give the embedding weights and the unpatch basis.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.models.patching import _normalize_patch, fold_patches, unfold_patches
from neugk_jax.models.utils import MLP, leaky_relu, silu

FIELD_OPTIONS = {
    # filter mlp: output rank, hidden width, hidden layers
    "rank": 256,
    "hidden": 256,
    "depth": 2,
    # point encoding: fourier (n_freq octaves) or cosine (modes, capped at the block resolution)
    "encoding": "fourier",
    "n_freq": 5,
    "modes": 16,
    # unpatch: deeponet over the whole patch or hier (anchor sub-blocks with tied codes)
    "decoder": "hier",
    "branch": 1024,
    "anchors": None,
    "code_modes": None,
    "code_rank": 128,
    "zero_init": True,
}


def field_options(options: Mapping) -> dict:
    unknown = set(options) - set(FIELD_OPTIONS)
    if unknown:
        raise ValueError(f"unknown field patching options {sorted(unknown)}; one of {sorted(FIELD_OPTIONS)}")
    return {**FIELD_OPTIONS, **options}


def _half(n: int) -> float:
    return (n - 1) / 2 if n > 1 else 1.0


def _unit(nodes: np.ndarray, ref: np.ndarray) -> np.ndarray:
    lo, hi = ref.min(), ref.max()
    return 2.0 * (nodes - lo) / max(hi - lo, 1e-12) - 1.0


class PointGrid(eqx.Module):
    """Coordinates of the points of one patch, split into anchor blocks, in ``fold_patches`` order.

    ``coords`` holds the absolute coordinates per token along the absolute spatial axes
    ``(*T_abs, P, n_abs)``; ``offsets`` the index offsets of the relative axes from the centre of
    each point's block; ``perm`` sorts the points by block and ``unit`` places the block centres
    in [-1, 1] of the patch.
    """

    buffer_fields = ("coords", "offsets", "channel", "weight", "unit", "perm", "inv")

    coords: jnp.ndarray
    offsets: jnp.ndarray
    channel: jnp.ndarray
    weight: jnp.ndarray
    unit: jnp.ndarray
    perm: jnp.ndarray
    inv: jnp.ndarray
    abs_axes: tuple[int, ...] = eqx.field(static=True)
    half: tuple[float, ...] = eqx.field(static=True)
    caps: tuple[int, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    reference: tuple[float, ...] = eqx.field(static=True)
    n_anchor: int = eqx.field(static=True)

    def __init__(self, padded: Sequence[int], patch: Sequence[int], channels: int, spec: Optional[Mapping], anchors: Sequence[int]):
        patch = _normalize_patch(patch)
        n = len(patch)
        spec = dict(spec or {})
        axes = list(spec.get("axes") or [{"kind": "relative"}] * n)
        folded = list(spec.get("folded") or [])
        if len(axes) != n:
            raise ValueError(f"grid spec has {len(axes)} axes for {n} spatial axes")
        n_fold = [len(f["nodes"]) for f in folded]
        if channels % max(1, math.prod(n_fold)):
            raise ValueError(f"{channels} channels do not hold the folded axes {n_fold}")
        n_c = channels // math.prod(n_fold)
        sub = [p // a for p, a in zip(patch, anchors)]
        if any(p % a for p, a in zip(patch, anchors)):
            raise ValueError(f"anchors {tuple(anchors)} do not divide the patch {patch}")
        idx = np.meshgrid(*(np.arange(k) for k in (*patch, n_c, *n_fold)), indexing="ij")
        idx = [i.reshape(-1) for i in idx]
        offs, ic, ifold = idx[:n], idx[n], idx[n + 1 :]
        anchor = np.ravel_multi_index([o // q for o, q in zip(offs, sub)], anchors)
        first = np.array([np.flatnonzero(anchor == k)[0] for k in range(math.prod(anchors))])
        centre = [(o // q) * q + (q - 1) / 2 for o, q in zip(offs, sub)]
        unit = [(c[first] - (p - 1) / 2) / _half(p) for c, p in zip(centre, patch)]

        rel = [i for i, a in enumerate(axes) if a.get("kind", "relative") == "relative"]
        self.abs_axes = tuple(i for i in range(n) if i not in rel)
        tokens = [padded[i] // patch[i] for i in self.abs_axes]
        # absolute spatial axes: node values per token and patch offset, padded nodes continue the grid with zero weight
        coords, weight = [], np.ones((*tokens, len(anchor)))
        for j, i in enumerate(self.abs_axes):
            nodes = np.asarray(axes[i]["nodes"], np.float64)
            w = np.asarray(axes[i].get("weights", np.ones_like(nodes)), np.float64)
            extra = padded[i] - len(nodes)
            step = nodes[-1] - nodes[-2] if len(nodes) > 1 else 1.0
            full = np.concatenate([nodes, nodes[-1] + step * np.arange(1, extra + 1)])
            w = np.concatenate([w, np.zeros(extra)])
            pos = np.arange(tokens[j])[:, None] * patch[i] + offs[i][None]
            shape = [1] * len(tokens) + [len(anchor)]
            shape[j] = tokens[j]
            coords.append(np.broadcast_to(_unit(full[pos], nodes).reshape(shape), (*tokens, len(anchor))))
            weight = weight * w[pos].reshape(shape)
        for f, i in zip(folded, ifold):
            nodes = np.asarray(f["nodes"], np.float64)
            coords.append(np.broadcast_to(_unit(nodes, nodes)[i], (*tokens, len(anchor))))
            weight = weight * np.asarray(f.get("weights", np.ones_like(nodes)), np.float64)[i]
        self.coords = jnp.asarray(np.stack(coords, -1) if coords else np.zeros((*tokens, len(anchor), 0)), jnp.float32)
        self.weight = jnp.asarray(weight / weight[weight > 0].mean(), jnp.float32)
        self.offsets = jnp.asarray(np.stack([offs[i] - centre[i] for i in rel], -1) if rel else np.zeros((len(anchor), 0)), jnp.float32)
        self.channel = jnp.asarray(np.eye(n_c)[ic], jnp.float32)
        self.unit = jnp.asarray(np.stack(unit, -1), jnp.float32)
        self.half = tuple(_half(sub[i]) for i in rel)
        self.caps = tuple(sub[i] for i in rel) + tuple(len(axes[i]["nodes"]) for i in self.abs_axes) + tuple(n_fold)
        self.spacing = tuple(float(axes[i].get("spacing", 1.0)) for i in rel)
        # physical half-width of a full patch at the nominal spacing, unless given
        self.reference = tuple(float(axes[i].get("reference", _half(patch[i]) * s)) for i, s in zip(rel, self.spacing))
        perm = np.argsort(anchor, kind="stable")
        self.perm, self.inv = jnp.asarray(perm), jnp.asarray(np.argsort(perm))
        self.n_anchor = len(first)

    @property
    def n_coords(self) -> int:
        return self.offsets.shape[-1] + self.coords.shape[-1]

    def features(self, geometry, encoding: str, n_freq: int, modes: int) -> jnp.ndarray:
        """``(*T_abs, P, F)`` encoded coordinates, channel and log physical half-width of every point."""
        sp = jnp.asarray(self.spacing if geometry is None else geometry, jnp.float32)
        lead = self.coords.shape[:-1]
        rel = jnp.broadcast_to(self.offsets / jnp.asarray(self.half, jnp.float32), (*lead, self.offsets.shape[-1]))
        x = jnp.concatenate([rel, self.coords], -1)
        if encoding == "fourier":
            z = x[..., None] * (jnp.pi * 2.0 ** jnp.arange(n_freq))
            enc = [x, jnp.sin(z).reshape(*lead, -1), jnp.cos(z).reshape(*lead, -1)]
        elif encoding == "cosine":
            k = jnp.arange(modes)
            # modes above the resolution of each coordinate are cut
            cap = jnp.arange(modes)[None] < jnp.asarray(self.caps)[:, None]
            enc = [(jnp.cos(k * jnp.pi * (x[..., None] + 1) / 2) * cap).reshape(*lead, -1)]
        else:
            raise ValueError(f"encoding={encoding!r}; one of fourier, cosine")
        scale = jnp.log(jnp.asarray(self.half, jnp.float32) * sp / jnp.asarray(self.reference, jnp.float32))
        return jnp.concatenate([*enc, jnp.broadcast_to(self.channel, (*lead, self.channel.shape[-1])), jnp.broadcast_to(scale, (*lead, scale.shape[-1]))], -1)


def _n_features(grid: PointGrid, opts: Mapping) -> int:
    per = 1 + 2 * opts["n_freq"] if opts["encoding"] == "fourier" else opts["modes"]
    return grid.n_coords * per + grid.channel.shape[-1] + len(grid.spacing)


def _letters(n: int, abs_axes: Sequence[int]) -> tuple[str, str]:
    grid = "abcdefgh"[:n]
    return grid, "".join(grid[i] for i in abs_axes)


def _default_anchors(patch: Sequence[int]) -> tuple[int, ...]:
    return tuple(2 if p >= 4 and p % 2 == 0 else 1 for p in patch)


class FieldPatchEmbed(eqx.Module):
    """``PatchEmbed`` whose per-point weights are a filter MLP of the point coordinates, times the quadrature weight."""

    grid: PointGrid
    kernel: MLP
    mix: MLP
    act: Callable = eqx.field(static=True)
    patch_size: tuple[int, ...] = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)
    encoding: tuple = eqx.field(static=True)

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
        self.grid = PointGrid(base_resolution, self.patch_size, in_channels, grid, (1,) * len(self.patch_size))
        self.encoding = (opts["encoding"], opts["n_freq"], opts["modes"])
        dims = [_n_features(self.grid, opts)] + [opts["hidden"]] * opts["depth"] + [opts["rank"]]
        self.kernel = MLP(dims, key=k_kernel, act_fn=silu)
        mix = [opts["rank"]] + [int(embed_dim * mlp_ratio)] * (mlp_depth - 1) + [embed_dim]
        self.mix = MLP(mix, key=k_mix, act_fn=act_fn, use_bias=False)
        self.act = act_fn

    def __call__(self, x: jnp.ndarray, geometry=None) -> jnp.ndarray:
        w = self.kernel(self.grid.features(geometry, *self.encoding)) * self.grid.weight[..., None]
        p = fold_patches(x, self.patch_size)
        g, a = _letters(len(self.patch_size), self.grid.abs_axes)
        h = jnp.einsum(f"{g}p,{a}pr->{g}r", p, w) / p.shape[-1]
        return self.mix(self.act(h))


class FieldUnpatch(eqx.Module):
    """``PatchExpand`` (unpatch) rebuilding every patch from per-token codes and a filter-MLP basis.

    ``decoder="deeponet"``: ``f(point) = sum_r branch(token)_r basis(point)_r`` over the whole patch.
    ``decoder="hier"``: the patch splits into ``anchors`` blocks per axis; the branch gives
    ``prod(code_modes)`` code modes per token, mixed into one code per anchor by a cosine basis of
    the anchor position, and each block is a DeepONet of its anchor code over block-relative coordinates.
    """

    grid: PointGrid
    expansion: MLP
    basis: MLP
    modulation: Optional[eqx.Module]
    expand_by: tuple[int, ...] = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    target_grid_size: tuple[int, ...] = eqx.field(static=True)
    decoder: str = eqx.field(static=True)
    code_modes: tuple[int, ...] = eqx.field(static=True)
    encoding: tuple = eqx.field(static=True)

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
        k_exp, k_basis, k_mod = jr.split(key, 3)
        self.expand_by = _normalize_patch(expand_by)
        self.out_dim = out_channels
        self.target_grid_size = tuple(g * e for g, e in zip(grid_size, self.expand_by))
        self.decoder = opts["decoder"]
        if self.decoder not in ("deeponet", "hier"):
            raise ValueError(f"decoder={self.decoder!r}; one of deeponet, hier")
        hier = self.decoder == "hier"
        anchors = tuple(opts["anchors"] or _default_anchors(self.expand_by)) if hier else (1,) * len(self.expand_by)
        self.code_modes = tuple(opts["code_modes"] or anchors)
        self.grid = PointGrid(self.target_grid_size, self.expand_by, out_channels, grid, anchors)
        self.encoding = (opts["encoding"], opts["n_freq"], opts["modes"])
        rank = opts["code_rank"] if hier else opts["rank"]
        width = math.prod(self.code_modes) * rank if hier else rank
        self.expansion = MLP([dim, opts["branch"], width], key=k_exp, act_fn=leaky_relu)
        if opts["zero_init"]:
            last = self.expansion.layers[-1].inner
            self.expansion = eqx.tree_at(lambda m: (m.layers[-1].inner.weight, m.layers[-1].inner.bias), self.expansion, (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)))
        dims = [_n_features(self.grid, opts)] + [opts["hidden"]] * opts["depth"] + [rank]
        self.basis = MLP(dims, key=k_basis, act_fn=silu)
        if cond_dim:
            from neugk_jax.models.swin import Film

            self.modulation = Film(cond_dim, dim, key=k_mod)
        else:
            self.modulation = None

    def codes(self, z: jnp.ndarray) -> jnp.ndarray:
        h = self.expansion(z).reshape(*z.shape[:-1], math.prod(self.code_modes), -1)
        phi = jnp.ones((self.grid.n_anchor, 1))
        for i, m in enumerate(self.code_modes):
            basis = jnp.cos(jnp.arange(m) * jnp.pi * (self.grid.unit[:, i : i + 1] + 1) / 2)
            phi = (phi[:, :, None] * basis[:, None, :]).reshape(self.grid.n_anchor, -1)
        return jnp.einsum("ak,...kr->...ar", phi, h)

    def __call__(self, z: jnp.ndarray, cond: Optional[jnp.ndarray] = None, geometry=None) -> jnp.ndarray:
        if self.modulation is not None:
            z = self.modulation(z, cond)
        psi = self.basis(self.grid.features(geometry, *self.encoding))
        g, a = _letters(len(self.expand_by), self.grid.abs_axes)
        if self.decoder == "deeponet":
            out = jnp.einsum(f"{g}r,{a}pr->{g}p", self.expansion(z), psi) / psi.shape[-1]
        else:
            psi = psi[..., self.grid.perm, :].reshape(*psi.shape[:-2], self.grid.n_anchor, -1, psi.shape[-1])
            out = jnp.einsum(f"{g}xr,{a}xqr->{g}xq", self.codes(z), psi) / psi.shape[-1]
            out = out.reshape(*out.shape[:-2], -1)[..., self.grid.inv]
        return unfold_patches(out, self.expand_by, out_channels=self.out_dim)
