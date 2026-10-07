"""Physics attention, Transolver blocks / layers and the swappable token layers."""

from __future__ import annotations

import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neugk_jax.models.attention import PhysicsAttention, TransolverBlock
from neugk_jax.models.base import AttentionBlockBase, TokenLayerBase
from neugk_jax.models.layers import token_layer
from neugk_jax.models.transolver import transolver_layer

AGENTAL = "/system/user/publicwork/galletti/git/active-learning-agent/src"


def test_physics_attention_shape_and_slices():
    attn = PhysicsAttention(32, 4, key=jr.PRNGKey(0), slice_num=16)
    x = jr.normal(jr.PRNGKey(1), (100, 32))
    assert attn(x).shape == (100, 32)
    w = attn.slice_weights(x)
    assert w.shape == (4, 100, 16)
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-5)


def test_physics_attention_is_permutation_equivariant():
    # tokens carry no order: permuting them permutes the output
    attn = PhysicsAttention(16, 2, key=jr.PRNGKey(0), slice_num=8)
    x = jr.normal(jr.PRNGKey(1), (50, 16))
    perm = jr.permutation(jr.PRNGKey(2), 50)
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(attn(x[perm]), attn(x)[perm], atol=1e-5)


def test_physics_attention_matches_the_agental_port():
    pytest.importorskip("jaxtyping")
    sys.path.insert(0, AGENTAL)
    try:
        from agentAL.surrogate.physics_attention import PhysicsAttentionIrregularMesh
    except ImportError:
        pytest.skip("agentAL not available")
    finally:
        sys.path.remove(AGENTAL)
    ref = PhysicsAttentionIrregularMesh(32, heads=4, dim_head=8, slice_num=16, key=jr.PRNGKey(3))
    ours = PhysicsAttention(32, 4, key=jr.PRNGKey(4), slice_num=16)
    pairs = [
        ("in_x", "in_project_x"),
        ("in_fx", "in_project_fx"),
        ("in_slice", "in_project_slice"),
        ("to_q", "to_q"),
        ("to_k", "to_k"),
        ("to_v", "to_v"),
        ("proj", "to_out"),
    ]
    for mine, theirs in pairs:
        lin = getattr(ref, theirs)
        ours = eqx.tree_at(lambda m, n=mine: getattr(m, n).inner.weight, ours, lin.weight)
        if lin.bias is not None:
            ours = eqx.tree_at(lambda m, n=mine: getattr(m, n).inner.bias, ours, lin.bias)
    ours = eqx.tree_at(lambda m: m.temperature, ours, ref.temperature[0] * 1.3)
    ref = eqx.tree_at(lambda m: m.temperature, ref, ref.temperature * 1.3)
    x = jr.normal(jr.PRNGKey(5), (64, 32))
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(ours(x), ref(x[None])[0], atol=1e-5)


def test_transolver_layer_on_a_grid_with_film():
    x = jr.normal(jr.PRNGKey(0), (4, 3, 5, 16))
    plain = transolver_layer(16, 2, 2, key=jr.PRNGKey(1), slice_num=8)
    assert plain(x).shape == x.shape
    film = transolver_layer(16, 2, 2, key=jr.PRNGKey(1), slice_num=8, cond_dim=6, cond_mode="film")
    c0, c1 = jnp.zeros(6), jnp.ones(6)
    assert float(jnp.abs(film(x, c0) - film(x, c1)).max()) > 0
    with pytest.raises(NotImplementedError, match="film"):
        transolver_layer(16, 1, 2, key=jr.PRNGKey(1), cond_dim=6, cond_mode="dit")


@pytest.mark.parametrize("spec", ["swin", "vit", {"kind": "transolver", "slice_num": 8}])
def test_token_layers_are_swappable(spec):
    # the swin arguments build every layer kind; all map (*grid, dim) to (*grid, dim)
    shared = dict(
        grid_size=(4, 4, 6),
        window_size=(2, 2, 3),
        mlp_ratio=2.0,
        drop_path=0.1,
        qkv_bias=True,
        qk_norm=True,
        use_rpb=True,
        gated_attention=False,
        norm_affine=True,
        rms_norm=False,
        legacy_double_shortcut=False,
        cond_dim=8,
        cond_mode="film",
    )
    layer = token_layer(spec, 16, 2, 2, key=jr.PRNGKey(0), **shared)
    assert isinstance(layer, TokenLayerBase)
    x = jr.normal(jr.PRNGKey(1), (4, 4, 6, 16))
    out = layer(x, jnp.ones(8), key=jr.PRNGKey(2), inference=False)
    assert out.shape == x.shape and bool(jnp.isfinite(out).all())


def test_spec_options_are_checked():
    with pytest.raises(TypeError):
        token_layer({"kind": "vit", "slice_num": 8}, 16, 1, 2, key=jr.PRNGKey(0))
    with pytest.raises(ValueError, match="kind"):
        token_layer("conv", 16, 1, 2, key=jr.PRNGKey(0))


def test_transolver_interfaces():
    assert issubclass(TransolverBlock, AttentionBlockBase)
    assert (
        TransolverBlock.flat_tokens
        and not TransolverBlock.positional
        and not TransolverBlock.modulated
    )
    assert transolver_layer(16, 1, 2, key=jr.PRNGKey(0)).needs_pos_embed
