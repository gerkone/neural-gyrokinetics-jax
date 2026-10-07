"""N-dimensional fold / unfold (im2col / col2im) and block padding on channel-last ``(*spatial, C)`` inputs."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import jax.numpy as jnp


def _normalize_patch(patch_size: Sequence[int]) -> tuple[int, ...]:
    # 0/none entries become 1
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
    new_shape = []
    for s, p in zip(spatial, ps):
        new_shape.extend([s // p, p])
    x = x.reshape(*new_shape, x.shape[-1])
    perm = list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2)) + [2 * n]
    x = jnp.transpose(x, perm)
    return x.reshape(*(s // p for s, p in zip(spatial, ps)), -1)


def unfold_patches(
    x: jnp.ndarray, expand_by: Sequence[int], *, out_channels: Optional[int] = None
) -> jnp.ndarray:
    """``(*grid, prod(expand)*out_c) → (*expanded, out_c)``. Inverse of ``fold_patches``."""
    eb = _normalize_patch(expand_by)
    n = len(eb)
    grid = x.shape[:n]
    if out_channels is None:
        out_channels = x.shape[-1] // math.prod(eb)
    x = x.reshape(*grid, *eb, out_channels)
    perm = [a for i in range(n) for a in (i, i + n)] + [2 * n]
    x = jnp.transpose(x, perm)
    return x.reshape(*[g * e for g, e in zip(grid, eb)], out_channels)


def pad_amounts(spatial: Sequence[int], block_size: Sequence[int]) -> tuple[int, ...]:
    return tuple(-s % b for s, b in zip(spatial, _normalize_patch(block_size)))


def pad_to_blocks(x: jnp.ndarray, block_size: Sequence[int]) -> jnp.ndarray:
    pads = pad_amounts(x.shape[: len(block_size)], block_size)
    return jnp.pad(x, [(0, p) for p in pads] + [(0, 0)] * (x.ndim - len(pads)))


def unpad(x: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
    return x[tuple(slice(0, s) for s in shape)]
