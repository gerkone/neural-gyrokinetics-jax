"""Interfaces of the swappable model parts: grid encoders / decoders, token blocks and token layers.

A U-Net or autoencoder is built from

* a :class:`GridEncoderBase` (patch embedding, ``(*spatial, C) -> (*grid, dim)``),
* :class:`TokenLayerBase` stages (``(*grid, dim) -> (*grid, dim)``, stacks of
  :class:`AttentionBlockBase` blocks, e.g. Swin, ViT or Transolver),
* :class:`TokenResamplerBase` steps between stages (token merging / expansion, token space only),
* a :class:`GridDecoderBase` (unpatch, ``(*grid, dim) -> (*spatial, C)``),

and any implementation of an interface can replace another. All inputs are unbatched and channel-last.
"""

from __future__ import annotations

import abc
from typing import ClassVar, Optional

import equinox as eqx
import jax.numpy as jnp


class GridEncoderBase(eqx.Module):
    """Patch embedding of a channel-last grid ``(*spatial, C)`` into tokens ``(*grid, dim)``.

    ``geometry`` is an optional per-sample description of the grid (e.g. the physical spacings);
    encoders that do not use it ignore it.
    """

    patch_size: eqx.AbstractVar[tuple[int, ...]]
    grid_size: eqx.AbstractVar[tuple[int, ...]]

    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray, geometry=None) -> jnp.ndarray:
        raise NotImplementedError


class GridDecoderBase(eqx.Module):
    """Tokens ``(*grid, dim)`` back to a channel-last grid ``(*target_grid_size, C)``.

    ``cond`` is an optional condition embedding (film), ``geometry`` as for :class:`GridEncoderBase`.
    """

    target_grid_size: eqx.AbstractVar[tuple[int, ...]]

    @abc.abstractmethod
    def __call__(
        self, z: jnp.ndarray, cond: Optional[jnp.ndarray] = None, geometry=None
    ) -> jnp.ndarray:
        raise NotImplementedError


class TokenResamplerBase(eqx.Module):
    """Tokens ``(*grid, dim)`` to tokens ``(*target_grid_size, out_dim)`` (merging or expansion)."""

    target_grid_size: eqx.AbstractVar[tuple[int, ...]]
    out_dim: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class AttentionBlockBase(eqx.Module):
    """One residual token-mixing block on ``(*grid, dim)`` or ``(n, dim)`` tokens.

    ``modulated`` blocks (DiT) take the condition embedding as their second argument;
    ``flat_tokens`` blocks expect ``(n, dim)`` (the layer flattens the grid around them);
    ``positional`` blocks know where their tokens are (windows, relative position bias), the
    others are permutation equivariant and need a positional embedding on their input.
    """

    modulated: ClassVar[bool] = False
    flat_tokens: ClassVar[bool] = False
    positional: ClassVar[bool] = False

    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray, *args, key=None, inference: bool = True) -> jnp.ndarray:
        raise NotImplementedError


class TokenLayerBase(eqx.Module):
    """A stage of token-mixing blocks on ``(*grid, dim)``, optionally conditioned, shape preserving.

    ``needs_pos_embed``: the layer does not locate its tokens, so as the first layer after the grid
    encoder it needs a positional embedding of the tokens.
    """

    @property
    @abc.abstractmethod
    def needs_pos_embed(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self,
        x: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        *,
        key=None,
        inference: bool = True,
    ) -> jnp.ndarray:
        raise NotImplementedError
