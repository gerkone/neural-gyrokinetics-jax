"""The Transolver layer: a stack of physics-attention blocks over the flattened tokens."""

from __future__ import annotations

from typing import Callable, Optional

from neugk_jax.models.attention.transolver import TransolverBlock
from neugk_jax.models.swin import BlockStack, block_stack
from neugk_jax.models.utils import gelu


def transolver_layer(
    dim: int,
    depth: int,
    num_heads: int,
    *,
    key,
    cond_dim: Optional[int] = None,
    cond_mode: str = "film",
    slice_num: int = 64,
    mlp_ratio: float = 4.0,
    drop_path: float = 0.0,
    act_fn: Callable = gelu,
    use_checkpoint: bool = False,
    qkv_bias: bool = False,
    qk_norm: bool = False,
    norm_affine: bool = False,
    rms_norm: bool = False,
) -> BlockStack:
    """``depth`` Transolver blocks over the flattened ``(*grid, dim)`` tokens; conditioning by film only."""
    kw = dict(
        slice_num=slice_num,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        act_fn=act_fn,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        norm_affine=norm_affine,
        rms_norm=rms_norm,
    )

    def plain(i, k):
        return TransolverBlock(dim, num_heads, key=k, **kw)

    def dit(i, k):
        raise NotImplementedError(
            "DiT-conditioned Transolver blocks are not implemented; use cond_mode='film'"
        )

    return block_stack(
        plain,
        dit,
        depth,
        dim,
        key=key,
        cond_dim=cond_dim,
        cond_mode=cond_mode,
        tokens=True,
        use_checkpoint=use_checkpoint,
    )
