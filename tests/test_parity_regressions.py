"""Regression guards for torch-parity port bugs."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from omegaconf import OmegaConf


def test_swin_layer_forwards_rms_norm():
    from neugk_jax.models.swin import SwinLayer
    from neugk_jax.models.utils import RMSNorm

    grid, win, dim = (4, 8), (2, 4), 8
    layer = SwinLayer(2, dim, depth=2, num_heads=2, grid_size=grid, window_size=win,
                      key=jr.PRNGKey(2), rms_norm=True)
    assert all(isinstance(b.norm1, RMSNorm) and isinstance(b.norm2, RMSNorm)
               for b in layer.blocks)
    layer_ln = SwinLayer(2, dim, depth=1, num_heads=2, grid_size=grid, window_size=win,
                         key=jr.PRNGKey(2), rms_norm=False)
    assert not isinstance(layer_ln.blocks[0].norm1, RMSNorm)
    layer_legacy = SwinLayer(2, dim, depth=1, num_heads=2, grid_size=grid, window_size=win,
                             key=jr.PRNGKey(2), legacy_double_shortcut=True)
    assert layer_legacy.blocks[0].legacy_double_shortcut


def test_weight_decay_mask_and_coupling():
    from neugk_jax.training.runner import build_optimizer, weight_decay_mask

    params = {"cond_embed": jnp.ones(3), "blocks": {"w": jnp.ones(3)}}
    mask = weight_decay_mask(params, ["cond"])
    assert mask == {"cond_embed": False, "blocks": {"w": True}}
    assert not any(jax.tree_util.tree_leaves(weight_decay_mask(params, "all")))

    tcfg = OmegaConf.create({"weight_decay": 0.1, "exclude_from_wd": ["cond"], "clip_grad": False})
    zero = jax.tree_util.tree_map(jnp.zeros_like, params)
    for decoupled in (False, True):
        opt = build_optimizer(optax.constant_schedule(1e-2), tcfg, params, decoupled=decoupled)
        upd, _ = opt.update(zero, opt.init(params), params)
        # excluded leaves see no decay, the rest shrink
        assert jnp.all(upd["cond_embed"] == 0)
        assert jnp.all(upd["blocks"]["w"] < 0)


def _zero_mlp(blk):
    zeroed = jax.tree_util.tree_map(lambda a: jnp.zeros_like(a) if eqx.is_array(a) else a, blk.mlp)
    return eqx.tree_at(lambda b: b.mlp, blk, zeroed)


@pytest.mark.parametrize("kind", ["swin", "dit_swin", "vit", "dit_vit"])
def test_legacy_double_shortcut_every_block(kind):
    from neugk_jax.models.swin import DiTSwinBlock, SwinBlock
    from neugk_jax.models.vit import DiTViTBlock, ViTBlock

    dim, cond_dim = 8, 6
    cond = jr.normal(jr.PRNGKey(3), (cond_dim,))
    if kind in ("swin", "dit_swin"):
        x = jr.normal(jr.PRNGKey(0), (4, 8, dim))
    else:
        x = jr.normal(jr.PRNGKey(0), (16, dim))

    def build(legacy):
        k = jr.PRNGKey(1)
        if kind == "swin":
            return SwinBlock(dim, 2, (4, 8), (2, 4), key=k, legacy_double_shortcut=legacy)
        if kind == "dit_swin":
            return DiTSwinBlock(dim, 2, cond_dim, (4, 8), (2, 4), key=k,
                                legacy_double_shortcut=legacy)
        if kind == "vit":
            return ViTBlock(dim, 2, key=k, legacy_double_shortcut=legacy)
        return DiTViTBlock(dim, 2, cond_dim, key=k, legacy_double_shortcut=legacy)

    def run(blk):
        args = (x,) if kind in ("swin", "vit") else (x, cond)
        return blk(*args, inference=True)

    single, legacy = build(False), build(True)
    x_res1 = run(_zero_mlp(single))
    # zeroed mlp leaves the post-attention residual: single -> x_res1, legacy -> 2*x_res1
    assert jnp.allclose(run(_zero_mlp(legacy)), 2.0 * x_res1, atol=1e-5)
    assert jnp.allclose(run(legacy) - run(single), x_res1, atol=1e-5)


def test_layers_forward_legacy_flag():
    from neugk_jax.models.swin import DiTSwinLayer, FilmSwinLayer
    from neugk_jax.models.vit import DiTLayer, FilmViTLayer, ViTLayer

    kw = dict(key=jr.PRNGKey(0), legacy_double_shortcut=True)
    layers = [
        DiTSwinLayer(2, 8, 2, 2, (4, 8), (2, 4), cond_dim=6, **kw),
        FilmSwinLayer(2, 8, 2, 2, (4, 8), (2, 4), cond_dim=6, **kw),
        ViTLayer(2, 8, 2, 2, (4, 4), **kw),
        DiTLayer(2, 8, 2, 2, (4, 4), cond_dim=6, **kw),
        FilmViTLayer(2, 8, 2, 2, (4, 4), cond_dim=6, **kw),
    ]
    for layer in layers:
        assert all(b.legacy_double_shortcut for b in layer.blocks), type(layer).__name__
