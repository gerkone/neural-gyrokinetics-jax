"""Linear patch embedding / unpatch.

Both share a single spatial primitive — N-D
**fold / unfold** (a.k.a. im2col / col2im) — and differ only in the
mixer applied to the channel axis. Conceptually:

* ``PatchEmbed``    =  fold + MLP project up
* ``LinearUnpatch`` =  linear/MLP project + unfold + (optional crop), a ``TokenExpand`` onto the grid

Inputs are unbatched and shaped ``(*spatial, channels)``. Batched callers
``jax.vmap`` over the leading axis.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import equinox as eqx
import jax.numpy as jnp

from neugk_jax.models.base import GridDecoderBase, GridEncoderBase
from neugk_jax.models.ops import _normalize_patch, fold_patches
from neugk_jax.models.tokens import TokenExpand
from neugk_jax.models.utils import MLP, leaky_relu


class PatchEmbed(GridEncoderBase):
    """Fold + MLP channel mixer.

    Input  ``(*spatial, in_channels)`` → output ``(*grid, embed_dim)``.
    The MLP is stored under ``patch`` (``patch_embed.patch.mlp.{0,3}.{weight,bias}``).
    """

    patch: MLP
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
        mlp_depth: int = 2,
        mlp_ratio: float = 8.0,
        act_fn=leaky_relu,
    ):
        ps = _normalize_patch(patch_size)
        self.patch_size = ps
        self.grid_size = tuple(s // p for s, p in zip(base_resolution, ps))
        # hidden = embed_dim * mlp_ratio, no max-clamp
        hidden = int(embed_dim * mlp_ratio)
        dims = [math.prod(ps) * in_channels] + [hidden] * (mlp_depth - 1) + [embed_dim]
        self.patch = MLP(dims, key=key, act_fn=act_fn, use_bias=False)

    def __call__(self, x: jnp.ndarray, geometry=None) -> jnp.ndarray:
        return self.patch(fold_patches(x, self.patch_size))


class LinearUnpatch(TokenExpand, GridDecoderBase):
    """Linear unpatch: a :class:`TokenExpand` of the last tokens onto the grid (``out_channels`` per point)."""

    def __call__(
        self, x: jnp.ndarray, cond: Optional[jnp.ndarray] = None, geometry=None
    ) -> jnp.ndarray:
        return TokenExpand.__call__(self, x, cond)
