"""Attention primitives and the token-mixing blocks built on them.

``mha``: multi-head self / cross attention. ``swin``: shifted-window blocks on ``(*grid, dim)``.
``vit``: global-attention blocks on ``(n_tokens, dim)``. Every block implements
:class:`~neugk_jax.models.base.AttentionBlockBase`.
"""

from neugk_jax.models.attention.mha import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    einsum_attention,
)
from neugk_jax.models.attention.swin import DiTSwinBlock, SwinBlock
from neugk_jax.models.attention.vit import DiTViTBlock, ViTBlock

__all__ = [
    "DiTSwinBlock",
    "DiTViTBlock",
    "MultiHeadCrossAttention",
    "MultiHeadSelfAttention",
    "SwinBlock",
    "ViTBlock",
    "einsum_attention",
]
