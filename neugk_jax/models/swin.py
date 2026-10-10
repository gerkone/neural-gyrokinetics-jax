"""Token layers (block stacks with film / DiT conditioning) and the Swin layer of shifted-window blocks."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.attention.swin import DiTSwinBlock, SwinBlock
from neugk_jax.models.base import TokenLayerBase
from neugk_jax.models.utils import Linear, gelu, split_key


class Film(eqx.Module):
    """FiLM modulation ``x * (scale + 1) + shift``; one ``Linear(cond_dim -> 2*dim)`` gives both."""

    modulation: Linear

    def __init__(self, cond_dim: int, dim: int, *, key):
        self.modulation = Linear(cond_dim, 2 * dim, key=key)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        scale, shift = jnp.split(self.modulation(cond), 2, axis=-1)
        return x * (scale + 1.0) + shift


def run_blocks(blocks, films, x, cond, *args, modulated=False, checkpoint=False, key, inference):
    """Run ``blocks`` in sequence, FiLM-modulating each block input with ``films`` when given.

    Modulated (DiT) blocks take ``cond`` as their second argument; ``args`` follow it.
    """
    for i, (blk, k) in enumerate(zip(blocks, split_key(key, len(blocks)))):
        if films is not None:
            x = films[i](x, cond)
        call = eqx.filter_checkpoint(blk) if checkpoint else blk
        lead = (cond,) if modulated else ()
        x = call(x, *lead, *args, key=k, inference=inference)
    return x


class BlockStack(TokenLayerBase):
    """``depth`` transformer blocks run in sequence, plain, FiLM- or DiT-conditioned.

    ``conditioning`` holds one ``Film`` per block applied to the block input; DiT blocks
    (``modulated``) take the condition themselves. ``tokens`` flattens the spatial axes into
    one token axis around the blocks (global attention).
    """

    blocks: list
    conditioning: Optional[list]
    modulated: bool = eqx.field(static=True)
    tokens: bool = eqx.field(static=True)
    use_checkpoint: bool = eqx.field(static=True)

    @property
    def needs_pos_embed(self) -> bool:
        return bool(self.blocks) and not self.blocks[0].positional

    def __call__(self, x, condition=None, *, key=None, inference=True):
        shape = x.shape
        if self.tokens:
            x = x.reshape(-1, shape[-1])
        x = run_blocks(
            self.blocks,
            self.conditioning,
            x,
            condition,
            modulated=self.modulated,
            checkpoint=self.use_checkpoint,
            key=key,
            inference=inference,
        )
        return x.reshape(shape)


def block_stack(
    plain: Callable,
    dit: Callable,
    depth: int,
    dim: int,
    *,
    key,
    cond_dim: Optional[int],
    cond_mode: str,
    tokens: bool,
    use_checkpoint: bool,
) -> BlockStack:
    """Stack of ``depth`` blocks; ``plain(i, key)`` / ``dit(i, key)`` build block ``i``.

    No ``cond_dim`` gives plain blocks, ``cond_mode="film"`` plain blocks behind one Film
    each, anything else DiT-modulated blocks.
    """
    keys = jr.split(key, depth)
    films = None
    if cond_dim and cond_mode == "film":
        fkeys = jr.split(jr.fold_in(key, 1), depth)
        films = [Film(cond_dim, dim, key=fkeys[i]) for i in range(depth)]
    modulated = bool(cond_dim) and cond_mode != "film"
    make = dit if modulated else plain
    blocks = [make(i, keys[i]) for i in range(depth)]
    return BlockStack(blocks, films, modulated, tokens, use_checkpoint)


def swin_layer(
    dim: int,
    depth: int,
    num_heads: int,
    grid_size: Sequence[int],
    window_size: Sequence[int],
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
    use_rpb: bool = False,
    gated_attention: bool = False,
    norm_affine: bool = False,
    rms_norm: bool = False,
    legacy_double_shortcut: bool = False,
    attention: str = "einsum",
) -> BlockStack:
    """``depth`` Swin blocks alternating non-shifted / shifted windows."""
    common = dict(
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        act_fn=act_fn,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        use_rpb=use_rpb,
        gated_attention=gated_attention,
        rms_norm=rms_norm,
        attention=attention,
    )

    def plain(i, k):
        return SwinBlock(
            dim,
            num_heads,
            grid_size,
            window_size,
            key=k,
            shift=bool(i % 2),
            norm_affine=norm_affine,
            legacy_double_shortcut=legacy_double_shortcut,
            **common,
        )

    def dit(i, k):
        return DiTSwinBlock(
            dim, num_heads, cond_dim, grid_size, window_size, key=k, shift=bool(i % 2), **common
        )

    return block_stack(
        plain,
        dit,
        depth,
        dim,
        key=key,
        cond_dim=cond_dim,
        cond_mode=cond_mode,
        tokens=False,
        use_checkpoint=use_checkpoint,
    )
