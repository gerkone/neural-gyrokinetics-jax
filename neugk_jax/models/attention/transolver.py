"""Transolver blocks: pre-norm transformer blocks with physics attention on ``(n_tokens, dim)`` tokens."""

from __future__ import annotations

from typing import Callable

import jax.random as jr

from neugk_jax.models.attention.physics import PhysicsAttention
from neugk_jax.models.base import AttentionBlockBase
from neugk_jax.models.utils import MLP, DropPath, gelu, make_norm, split_key


class TransolverBlock(AttentionBlockBase):
    """``x + attn(norm1(x))``, then ``x + mlp(norm2(x))``, with physics attention over all tokens."""

    flat_tokens = True

    norm1: object
    norm2: object
    attn: PhysicsAttention
    mlp: MLP
    drop_path: DropPath

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key,
        slice_num: int = 64,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        norm_affine: bool = False,
        rms_norm: bool = False,
    ):
        katt, kmlp = jr.split(key, 2)
        self.norm1 = make_norm(dim, rms=rms_norm, affine=norm_affine)
        self.norm2 = make_norm(dim, rms=rms_norm, affine=norm_affine)
        self.attn = PhysicsAttention(
            dim, num_heads, key=katt, slice_num=slice_num, qkv_bias=qkv_bias, qk_norm=qk_norm
        )
        self.mlp = MLP([dim, max(int(dim * mlp_ratio), dim), dim], key=kmlp, act_fn=act_fn)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x, *, key=None, inference=True):
        k1, k2, k3 = split_key(key, 3)
        x = x + self.drop_path(
            self.attn(self.norm1(x), key=k3, inference=inference), key=k1, inference=inference
        )
        return x + self.drop_path(self.mlp(self.norm2(x)), key=k2, inference=inference)
