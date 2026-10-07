"""The swappable model parts implement their interfaces."""

from __future__ import annotations

import jax.random as jr
import pytest

from neugk_jax.models.attention import DiTSwinBlock, DiTViTBlock, SwinBlock, ViTBlock
from neugk_jax.models.base import (
    AttentionBlockBase,
    GridDecoderBase,
    GridEncoderBase,
    TokenLayerBase,
    TokenResamplerBase,
)
from neugk_jax.models.patching import FieldPatchEmbed, FieldUnpatch, LinearUnpatch, PatchEmbed
from neugk_jax.models.swin import swin_layer
from neugk_jax.models.tokens import TokenExpand, TokenMerge
from neugk_jax.models.vit import vit_layer


def test_patching_interfaces():
    assert issubclass(PatchEmbed, GridEncoderBase) and issubclass(FieldPatchEmbed, GridEncoderBase)
    assert issubclass(LinearUnpatch, GridDecoderBase) and issubclass(FieldUnpatch, GridDecoderBase)
    with pytest.raises(TypeError):
        GridEncoderBase()


def test_block_interfaces():
    for blk in (SwinBlock, DiTSwinBlock, ViTBlock, DiTViTBlock):
        assert issubclass(blk, AttentionBlockBase)
    assert DiTSwinBlock.modulated and DiTViTBlock.modulated and not SwinBlock.modulated
    assert ViTBlock.flat_tokens and not SwinBlock.flat_tokens
    assert SwinBlock.positional and DiTSwinBlock.positional and not ViTBlock.positional


def test_layers_need_a_position_embedding_without_windows():
    swin = swin_layer(16, 2, 2, (4, 4), (2, 2), key=jr.PRNGKey(0))
    vit = vit_layer(16, 2, 2, key=jr.PRNGKey(0))
    assert isinstance(swin, TokenLayerBase) and isinstance(vit, TokenLayerBase)
    assert not swin.needs_pos_embed and vit.needs_pos_embed


def test_token_resamplers_are_token_space():
    assert issubclass(TokenMerge, TokenResamplerBase) and issubclass(
        TokenExpand, TokenResamplerBase
    )
    assert issubclass(LinearUnpatch, TokenExpand) and not issubclass(TokenExpand, GridDecoderBase)
