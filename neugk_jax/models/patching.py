"""N-dimensional patch operations.

The three modules below all share a single spatial primitive — N-D
**fold / unfold** (a.k.a. im2col / col2im) — and differ only in the linear
mixer applied to the channel axis. Conceptually:

* ``PatchEmbed``  =  fold + MLP project up
* ``PatchMerge``  =  fold (patch=2) + norm + linear project up
* ``PatchExpand`` =  linear/MLP project + unfold + (optional crop)

Inputs are unbatched and shaped ``(*spatial, channels)``. Batched callers
``jax.vmap`` over the leading axis.
"""

from __future__ import annotations

from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.utils import RMSNorm
from neugk_jax.models.utils import MLP, LayerNorm, Linear, gelu, leaky_relu


def _norm(dim: int, *, rms: bool, affine: bool = True):
    return RMSNorm(dim) if rms else LayerNorm(dim, elementwise_affine=affine)


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _normalize_patch(patch_size: Sequence[int]) -> tuple[int, ...]:
    """Replace 0/None entries with 1 (axis kept unchanged)."""
    return tuple(p if p and p > 0 else 1 for p in patch_size)




def fold_patches(x: jnp.ndarray, patch_size: Sequence[int]) -> jnp.ndarray:
    """``(*spatial, c) → (*grid, prod(patch)*c)`` where ``grid_i = spatial_i // patch_i``.

    Generic N-D im2col: reshape each axis into ``(grid, patch)``, transpose
    so all grid axes come first and all patch axes second, then flatten the
    patch + channel suffix. Axes with ``patch=1`` are passthrough.
    """
    ps = _normalize_patch(patch_size)
    n = len(ps)
    spatial = x.shape[:n]
    c = x.shape[-1]
    new_shape = []
    for s, p in zip(spatial, ps):
        new_shape.extend([s // p, p])
    new_shape.append(c)
    x = x.reshape(new_shape)
    perm = list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2)) + [2 * n]
    x = jnp.transpose(x, perm)
    grid = tuple(s // p for s, p in zip(spatial, ps))
    return x.reshape(*grid, -1)


def unfold_patches(
    x: jnp.ndarray,
    expand_by: Sequence[int],
    *,
    out_channels: Optional[int] = None,
) -> jnp.ndarray:
    """``(*grid, prod(expand)*out_c) → (*expanded, out_c)``. Inverse of ``fold_patches``."""
    eb = _normalize_patch(expand_by)
    n = len(eb)
    grid = x.shape[:n]
    feat = x.shape[-1]
    if out_channels is None:
        out_channels = feat // _prod(eb)
    x = x.reshape(*grid, *eb, out_channels)
    perm = []
    for i in range(n):
        perm.append(i)
        perm.append(i + n)
    perm.append(2 * n)
    x = jnp.transpose(x, perm)
    return x.reshape(*[g * e for g, e in zip(grid, eb)], out_channels)




def pad_to_blocks(
    x: jnp.ndarray, block_size: Sequence[int]
) -> tuple[jnp.ndarray, tuple[int, ...]]:
    """Right-pad each spatial axis with zeros to a multiple of block_size."""
    bs = _normalize_patch(block_size)
    spatial = x.shape[: len(bs)]
    pads = []
    for s, b in zip(spatial, bs):
        r = s % b
        pads.append(0 if r == 0 else b - r)
    pad_width = [(0, p) for p in pads] + [(0, 0)] * (x.ndim - len(bs))
    return jnp.pad(x, pad_width), tuple(pads)


def unpad(
    x: jnp.ndarray, pad_amounts: Sequence[int], base_resolution: Sequence[int]
) -> jnp.ndarray:
    """Trim back to ``base_resolution`` along the leading axes."""
    slices = [slice(0, s) for s in base_resolution]
    slices += [slice(None)] * (x.ndim - len(base_resolution))
    return x[tuple(slices)]




class PatchEmbed(eqx.Module):
    """Fold + MLP channel mixer.

    Input  ``(*spatial, in_channels)`` → output ``(*grid, embed_dim)``.
    The MLP is stored under ``patch`` to match upstream torch's naming
    (``patch_embed.patch.mlp.{0,3}.{weight,bias}``).
    """

    patch: MLP
    norm: Optional[object]
    patch_size: tuple[int, ...] = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

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
        norm: bool = False,
        rms_norm: bool = False,
        act_fn=leaky_relu,
    ):
        ps = _normalize_patch(patch_size)
        self.patch_size = ps
        self.grid_size = tuple(s // p for s, p in zip(base_resolution, ps))
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        patch_elems = _prod(ps) * in_channels
        # hidden = embed_dim * mlp_ratio, no max-clamp (matches torch)
        hidden = int(embed_dim * mlp_ratio)
        dims = [patch_elems] + [hidden] * (mlp_depth - 1) + [embed_dim]
        # PatchEmbed MLP uses the model act_fn (config: GELU); bias=False
        self.patch = MLP(dims, key=key, act_fn=act_fn, use_bias=False)
        self.norm = _norm(embed_dim, rms=rms_norm) if norm else None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = fold_patches(x, self.patch_size)
        x = self.patch(x)
        if self.norm is not None:
            x = self.norm(x)
        return x




class PatchMerge(eqx.Module):
    """Fold with patch=2 + norm + linear up-project.

    Halves every spatial axis with size ≥ 2; channels become ``dim * c_multiplier``.
    """

    proj: Linear
    norm: object
    grid_size: tuple[int, ...] = eqx.field(static=True)
    target_grid_size: tuple[int, ...] = eqx.field(static=True)
    merge_mask: tuple[bool, ...] = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    patch_size: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        *,
        key,
        c_multiplier: int = 2,
        rms_norm: bool = False,
    ):
        gs = tuple(grid_size)
        merge_mask = tuple(g >= 2 for g in gs)
        self.grid_size = gs
        self.merge_mask = merge_mask
        self.patch_size = tuple(2 if m else 1 for m in merge_mask)
        # ceil so odd-length axes round up; forward pads them to the next multiple
        self.target_grid_size = tuple(
            (g + p - 1) // p for g, p in zip(gs, self.patch_size)
        )
        n_merged = sum(merge_mask)
        in_features = dim * (2**n_merged)
        out_features = dim * c_multiplier
        self.in_dim = dim
        self.out_dim = out_features
        self.norm = _norm(in_features, rms=rms_norm)
        self.proj = Linear(in_features, out_features, key=key, use_bias=False)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # pad odd-length axes to a multiple of 2, matching upstream torch PatchMerge.forward
        x, _ = pad_to_blocks(x, self.patch_size)
        x = fold_patches(x, self.patch_size)
        x = self.norm(x)
        return self.proj(x)




class StridedConvTranspose(eqx.Module):
    """ConvTranspose with stride == kernel (non-overlapping) == the patch-expand op.

    Stores the weight in **torch ConvTranspose layout** ``(in, out, *kernel)`` so the
    checkpoint loads with a direct copy. Output channel placement matches torch:
    ``out[*(g_i*k_i), oc] = sum_ic x[*g, ic] * W[ic, oc, *k]`` (kernel block per grid
    cell), implemented as einsum + interleave-reshape (same layout as ``unfold_patches``).
    """

    weight: jax.Array  # (in, out, *kernel)
    bias: jax.Array    # (out,)
    expand_by: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, in_ch: int, out_ch: int, expand_by: Sequence[int], *, key):
        eb = tuple(expand_by)
        self.expand_by = eb
        self.weight = jr.normal(key, (in_ch, out_ch, *eb)) * 0.02
        self.bias = jnp.zeros((out_ch,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (*grid, in) -> (*grid*expand, out)
        s = len(self.expand_by)
        gnd = x.ndim - 1
        kl = "".join(chr(ord("p") + i) for i in range(s))
        y = jnp.einsum(f"...i,io{kl}->...o{kl}", x, self.weight)  # (*grid, out, *k)
        # interleave grid axis i with kernel axis i, channel (out) last
        perm = []
        for i in range(gnd):
            perm += [i, gnd + 1 + i]
        perm.append(gnd)  # out channel
        y = jnp.transpose(y, perm)  # (g0, k0, g1, k1, ..., out)
        new_shape = [x.shape[i] * self.expand_by[i] for i in range(gnd)] + [y.shape[-1]]
        y = y.reshape(new_shape)
        return y + self.bias


class PatchExpand(eqx.Module):
    """MLP channel mixer + unfold + optional crop.

    Upsamples spatial axes by ``expand_by`` and reduces channels by
    ``c_multiplier`` (or sets them to ``out_channels`` when given). The
    MLP is stored under ``expansion`` to match upstream's naming
    (``unpatch.expansion.mlp.0.weight`` etc.).
    """

    expansion: object  # MLP (mlp patch) or eqx.nn.ConvTranspose (conv patch)
    proj_concat: Optional[Linear]
    modulation: Optional[object]  # Film, when cond_dim given (unpatch)
    norm: Optional[object]
    grid_size: tuple[int, ...] = eqx.field(static=True)
    target_grid_size: tuple[int, ...] = eqx.field(static=True)
    expand_by: tuple[int, ...] = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    out_channels: int | None = eqx.field(static=True)
    use_conv: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        *,
        key,
        c_multiplier: int = 2,
        expand_by: int | Sequence[int] = 2,
        target_grid_size: Optional[Sequence[int]] = None,
        out_channels: Optional[int] = None,
        mlp_depth: int = 1,
        mlp_ratio: float = 8.0,
        norm: bool = True,
        rms_norm: bool = False,
        use_conv: bool = False,
        patch_skip: bool = False,
        cond_dim: Optional[int] = None,
    ):
        gs = tuple(grid_size)
        if isinstance(expand_by, int):
            if target_grid_size is not None:
                # ceil so we never undershoot the target (crop after unfold)
                expand_by = [
                    max(1, -(-t // max(1, g)))
                    for g, t in zip(gs, target_grid_size)
                ]
            else:
                expand_by = [expand_by if g > 1 else 1 for g in gs]
        eb = _normalize_patch(expand_by)
        if target_grid_size is not None:
            target_grid_size = tuple(target_grid_size)
        else:
            target_grid_size = tuple(g * e for g, e in zip(gs, eb))
        self.grid_size = gs
        self.target_grid_size = target_grid_size
        self.expand_by = eb

        self.in_dim = dim
        self.out_channels = out_channels
        if out_channels is not None:
            inner = out_channels * _prod(eb)
            self.out_dim = out_channels
        else:
            inner = max(1, (dim * _prod(eb)) // c_multiplier)
            self.out_dim = max(1, dim // c_multiplier)

        self.use_conv = use_conv
        kexp, kpc, kmod = jr.split(key, 3)
        if use_conv:
            # stride==kernel ConvTranspose, torch layout (in, out, *kernel) -> direct copy
            self.expansion = StridedConvTranspose(dim, self.out_dim, eb, key=kexp)
        else:
            # hidden = prod(expand_by) * mlp_ratio, not dim * mlp_ratio (matches torch)
            hidden = int(_prod(eb) * mlp_ratio)
            dims = [dim] + [hidden] * (mlp_depth - 1) + [inner]
            # upstream PatchExpand uses LeakyReLU + bias=True
            self.expansion = MLP(dims, key=kexp, act_fn=leaky_relu, use_bias=True)
        # patch-skip residual projection (Linear(2*dim->dim) + LeakyReLU) and FiLM
        self.proj_concat = Linear(2 * dim, dim, key=kpc) if patch_skip else None
        if cond_dim:
            from neugk_jax.models.swin import Film
            self.modulation = Film(cond_dim, dim, key=kmod)
        else:
            self.modulation = None
        # norm runs over out_dim channels after unfold, matching torch PatchExpand.forward
        self.norm = _norm(self.out_dim, rms=rms_norm) if norm else None

    def __call__(self, x: jnp.ndarray, cond: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # torch order: proj_concat (skip residual) -> FiLM -> expansion -> crop -> norm
        if self.proj_concat is not None:
            x = leaky_relu(self.proj_concat(x))
        if self.modulation is not None:
            x = self.modulation(x, cond)
        if self.use_conv:
            x = self.expansion(x)           # (*grid, c) -> (*expanded, out)
        else:
            x = self.expansion(x)
            x = unfold_patches(x, self.expand_by, out_channels=self.out_dim)
        # crop any overshoot from ceiling the expand factor
        slices = [slice(0, t) for t in self.target_grid_size] + [slice(None)]
        x = x[tuple(slices)]
        if self.norm is not None:
            x = self.norm(x)
        return x
