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
from neugk_jax.models.patching import CConvPatchEmbed, CConvUnpatch, LinearUnpatch, PatchEmbed
from neugk_jax.models.swin import swin_layer
from neugk_jax.models.tokens import TokenExpand, TokenMerge
from neugk_jax.models.vit import vit_layer


def test_patching_interfaces():
    assert issubclass(PatchEmbed, GridEncoderBase) and issubclass(CConvPatchEmbed, GridEncoderBase)
    assert issubclass(LinearUnpatch, GridDecoderBase) and issubclass(CConvUnpatch, GridDecoderBase)
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


def _unet(**kw):
    from neugk_jax.models.gk_unet import SwinNDUnet

    return SwinNDUnet(
        space=3,
        dim=16,
        base_resolution=(8, 6, 8),
        in_channels=3,
        out_channels=3,
        patch_size=(2, 3, 4),
        window_size=(2, 2, 2),
        depth=[1, 1],
        num_heads=[2, 2],
        num_layers=2,
        middle_depth=1,
        middle_num_heads=2,
        key=jr.PRNGKey(0),
        **kw,
    )


def _forward(m, x):
    z = m.patch_encode(x)
    skips = []
    for blk in m.down_blocks:
        z, s = blk(z)
        skips.append(s)
    z = m.middle_upscale(m.middle(z))
    for blk, s in zip(m.up_blocks, skips[::-1]):
        z = blk(z, s)
    return m.patch_decode(z)


@pytest.mark.parametrize(
    "layer,pe", [({"kind": "transolver", "slice_num": 4}, "sincos"), ("vit", "ape"), ("swin", None)]
)
def test_unet_stages_take_any_token_layer(layer, pe):
    m = _unet(layer=layer, middle_layer=layer, token_pe=pe)
    x = jr.normal(jr.PRNGKey(1), (3, 8, 6, 8))
    assert _forward(m, x).shape == x.shape
    assert (m.token_pe is None) == (pe is None)


def test_unet_needs_a_position_embedding_for_global_layers():
    with pytest.raises(ValueError, match="token_pe"):
        _unet(layer="transolver")


def test_default_unet_has_the_legacy_leaves():
    import jax

    m = _unet()
    assert m.token_pe is None
    leaves = jax.tree_util.tree_leaves_with_path(m)
    assert not any("token_pe" in jax.tree_util.keystr(p) for p, _ in leaves)


def test_ae_bottleneck_layer_spec():
    import jax.numpy as jnp

    from neugk_jax.pinc import Swin5DAE

    ae = Swin5DAE(
        space=5,
        decouple_mu=True,
        dim=16,
        base_resolution=(8, 4, 4, 8, 4),
        in_channels=2,
        out_channels=2,
        patch_size=(2, 0, 2, 4, 2),
        window_size=(2, 0, 2, 2, 2),
        depth=1,
        num_heads=2,
        num_layers=1,
        bottleneck_dim=8,
        bottleneck_depth=1,
        bottleneck_num_heads=2,
        bottleneck_layer={"kind": "transolver", "slice_num": 4},
        key=jr.PRNGKey(0),
    )
    x = jr.normal(jr.PRNGKey(1), (2, 8, 4, 4, 8, 4))
    assert ae(x)["df"].shape == x.shape and bool(jnp.isfinite(ae(x)["df"]).all())
