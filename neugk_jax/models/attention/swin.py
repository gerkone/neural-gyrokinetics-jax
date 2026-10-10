"""Shifted-window (Swin) attention blocks on ``(*grid, dim)`` tokens."""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.models.attention.mha import MultiHeadSelfAttention
from neugk_jax.models.base import AttentionBlockBase
from neugk_jax.models.ops import fold_patches, pad_amounts, pad_to_blocks, unfold_patches, unpad
from neugk_jax.models.utils import MLP, DiTModulation, DropPath, gelu, make_norm, split_key


def _effective_window(grid_size, window_size):
    """Per-axis effective window: never larger than the grid; 0/None → no partition."""
    return tuple(g if w is None or w == 0 or w >= g else w for g, w in zip(grid_size, window_size))


def _shift_size(grid_size, window_size, eff_window, shift: bool) -> tuple[int, ...]:
    # half-window shift, skipped on axes where the window covers the whole grid
    return tuple(
        (e // 2) if (shift and g > w and e > 1) else 0
        for g, w, e in zip(grid_size, window_size, eff_window)
    )


def window_partition(x: jnp.ndarray, window_size: Sequence[int]) -> jnp.ndarray:
    return fold_patches(x, window_size).reshape(-1, math.prod(window_size), x.shape[-1])


def window_reverse(
    windows: jnp.ndarray, window_size: Sequence[int], spatial: Sequence[int]
) -> jnp.ndarray:
    grid = tuple(s // w for s, w in zip(spatial, window_size))
    dim = windows.shape[-1]
    return unfold_patches(windows.reshape(*grid, -1), window_size, out_channels=dim)


def _build_shift_mask(
    spatial: Sequence[int], window_size: Sequence[int], shift: Sequence[int]
) -> Optional[jnp.ndarray]:
    """Pre-compute the cyclic-shift attention mask on the *padded* grid.

    Returns ``(num_windows, W, W)`` additive bias (0 / -inf) or None when
    no shift applies on any axis.
    """
    if all(s == 0 for s in shift):
        return None
    padded = tuple(s + p for s, p in zip(spatial, pad_amounts(spatial, window_size)))
    # build a per-position region id by counting slices on each axis
    region = np.zeros(padded, dtype=np.int32)
    for axis, (size, w, sh) in enumerate(zip(padded, window_size, shift)):
        if sh == 0:
            continue
        slices = [(0, size - w), (size - w, size - sh), (size - sh, size)]
        for i, (lo, hi) in enumerate(slices):
            idx = [slice(None)] * len(padded)
            idx[axis] = slice(lo, hi)
            region[tuple(idx)] += i * (10**axis)
    win = window_partition(jnp.asarray(region)[..., None], window_size)[..., 0]
    mask = win[:, :, None] != win[:, None, :]  # (n_win, W, W)
    return jnp.where(mask, -1e9, 0.0).astype(jnp.float32)


def _window_attention(attn, x, window_size, shift_size, attn_mask):
    """Shifted-window self-attention of ``x`` (*spatial, dim): pad, roll, partition, attend, undo."""
    spatial = x.shape[:-1]
    h = pad_to_blocks(x, window_size)
    padded = h.shape[:-1]
    axes = list(range(len(padded)))
    shifted = any(s > 0 for s in shift_size)
    if shifted:
        h = jnp.roll(h, shift=[-s for s in shift_size], axis=axes)
    windows = window_partition(h, window_size)
    if attn_mask is not None:
        windows = jax.vmap(lambda w, m: attn(w, attn_bias=m[None]))(windows, attn_mask)
    else:
        windows = jax.vmap(attn)(windows)
    h = window_reverse(windows, window_size, padded)
    if shifted:
        h = jnp.roll(h, shift=list(shift_size), axis=axes)
    return unpad(h, spatial)


class SwinBlock(AttentionBlockBase):
    """One shifted-window transformer block (SwinV2 post-norm).

    Attention runs on the un-normed input and ``norm1`` normalizes its output; the MLP branch
    is ``norm2(drop_path(mlp(x_res1)))``, combined as ``x_res1 + mlp_out``.
    ``legacy_double_shortcut=True`` returns ``2·x_res1 + mlp_out`` instead, for checkpoints
    trained with that topology.
    """

    positional = True

    norm1: object
    norm2: object
    attn: MultiHeadSelfAttention
    mlp: MLP
    drop_path: DropPath
    window_size: tuple[int, ...] = eqx.field(static=True)
    shift_size: tuple[int, ...] = eqx.field(static=True)
    attn_mask: Optional[jax.Array]
    buffer_fields = ("attn_mask",)
    legacy_double_shortcut: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        *,
        key,
        shift: bool = False,
        legacy_double_shortcut: bool = False,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = False,
        rms_norm: bool = False,
    ):
        eff_w = _effective_window(grid_size, window_size)
        self.window_size = eff_w
        self.shift_size = _shift_size(grid_size, window_size, eff_w, shift)
        self.norm1 = make_norm(dim, rms=rms_norm, affine=norm_affine)
        self.norm2 = make_norm(dim, rms=rms_norm, affine=norm_affine)
        katt, kmlp = jr.split(key, 2)
        self.attn = MultiHeadSelfAttention(
            dim,
            num_heads,
            key=katt,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_rpb=use_rpb,
            gated_attention=gated_attention,
            window_size=eff_w,
        )
        self.mlp = MLP([dim, max(int(dim * mlp_ratio), dim), dim], key=kmlp, act_fn=act_fn)
        self.drop_path = DropPath(drop_path)
        self.attn_mask = _build_shift_mask(tuple(grid_size), eff_w, self.shift_size)
        self.legacy_double_shortcut = legacy_double_shortcut

    def __call__(self, x: jnp.ndarray, *, key=None, inference: bool = True) -> jnp.ndarray:
        h = _window_attention(self.attn, x, self.window_size, self.shift_size, self.attn_mask)
        key1, key2 = split_key(key, 2)
        x_res1 = x + self.drop_path(self.norm1(h), key=key1, inference=inference)
        mlp_out = self.norm2(self.drop_path(self.mlp(x_res1), key=key2, inference=inference))
        if self.legacy_double_shortcut:
            return 2.0 * x_res1 + mlp_out
        return x_res1 + mlp_out


class DiTSwinBlock(AttentionBlockBase):
    """SwinBlock with DiT-style 6-way conditioning modulation (pre-norm, gated residuals)."""

    modulated = True
    positional = True

    norm1: object
    norm2: object
    attn: MultiHeadSelfAttention
    mlp: MLP
    drop_path: DropPath
    mod: DiTModulation
    window_size: tuple[int, ...] = eqx.field(static=True)
    shift_size: tuple[int, ...] = eqx.field(static=True)
    attn_mask: Optional[jax.Array]
    buffer_fields = ("attn_mask",)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        *,
        key,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        rms_norm: bool = False,
    ):
        eff_w = _effective_window(grid_size, window_size)
        self.window_size = eff_w
        self.shift_size = _shift_size(grid_size, window_size, eff_w, shift)
        # dit modulation provides scale/shift, so the norm is non-affine
        self.norm1 = make_norm(dim, rms=rms_norm, affine=False)
        self.norm2 = make_norm(dim, rms=rms_norm, affine=False)
        katt, kmlp, kmod = jr.split(key, 3)
        self.attn = MultiHeadSelfAttention(
            dim,
            num_heads,
            key=katt,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_rpb=use_rpb,
            gated_attention=gated_attention,
            window_size=eff_w,
        )
        self.mlp = MLP([dim, max(int(dim * mlp_ratio), dim), dim], key=kmlp, act_fn=act_fn)
        self.drop_path = DropPath(drop_path)
        self.mod = DiTModulation(cond_dim, dim, key=kmod)
        self.attn_mask = _build_shift_mask(tuple(grid_size), eff_w, self.shift_size)

    def __call__(
        self, x: jnp.ndarray, cond: jnp.ndarray, *, key=None, inference=True
    ) -> jnp.ndarray:
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self.mod(cond)
        h = self.norm1(x) * (1.0 + scale_msa) + shift_msa
        h = _window_attention(self.attn, h, self.window_size, self.shift_size, self.attn_mask)
        key1, key2 = split_key(key, 2)
        x = x + gate_msa * self.drop_path(h, key=key1, inference=inference)
        h2 = self.mlp(self.norm2(x) * (1.0 + scale_mlp) + shift_mlp)
        return x + gate_mlp * self.drop_path(h2, key=key2, inference=inference)
