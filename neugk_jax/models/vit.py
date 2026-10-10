"""The ViT layer: a stack of global-attention blocks over the flattened tokens."""

from __future__ import annotations

from typing import Callable, Optional

from neugk_jax.models.attention.vit import DiTViTBlock, ViTBlock
from neugk_jax.models.swin import BlockStack, block_stack
from neugk_jax.models.utils import gelu


def vit_layer(
    dim: int,
    depth: int,
    num_heads: int,
    *,
    key,
    cond_dim: Optional[int] = None,
    cond_mode: str = "dit",
    mlp_ratio: float = 4.0,
    drop_path: float = 0.0,
    act_fn: Callable = gelu,
    use_checkpoint: bool = False,
    qkv_bias: bool = False,
    qk_norm: bool = False,
    gated_attention: bool = False,
    norm_affine: bool = False,
    rms_norm: bool = False,
    attention: str = "einsum",
) -> BlockStack:
    """``depth`` ViT blocks over the flattened ``(*grid, dim)`` tokens."""
    common = dict(
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        act_fn=act_fn,
        qkv_bias=qkv_bias,
        attention=attention,
    )
    attn_kw = dict(
        qk_norm=qk_norm, gated_attention=gated_attention, norm_affine=norm_affine, rms_norm=rms_norm
    )

    def plain(i, k):
        return ViTBlock(dim, num_heads, key=k, **attn_kw, **common)

    def dit(i, k):
        return DiTViTBlock(dim, num_heads, cond_dim, key=k, **attn_kw, **common)

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
