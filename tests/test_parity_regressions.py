"""Regression guards for torch-parity port bugs."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from omegaconf import OmegaConf


def test_swin_legacy_double_shortcut_and_rms_norm():
    """Guards the two port bugs that made pre-e79b021 checkpoints unusable."""
    from neugk_jax.models.swin import SwinBlock, SwinLayer
    from neugk_jax.models.utils import RMSNorm

    grid, win, dim = (4, 8), (2, 4), 8
    x = jr.normal(jr.PRNGKey(0), (*grid, dim))
    kw = dict(mlp_ratio=2.0, drop_path=0.0, rms_norm=True)
    single = SwinBlock(dim, 2, grid, win, key=jr.PRNGKey(1), **kw)
    legacy = SwinBlock(dim, 2, grid, win, key=jr.PRNGKey(1),
                       legacy_double_shortcut=True, **kw)

    # zero the MLP so the block output IS the post-attention residual x_res1:
    # single -> x_res1, legacy -> 2*x_res1 (the pre-e79b021 upstream topology)
    def _zero_mlp(blk):
        zeroed = jax.tree_util.tree_map(
            lambda a: jnp.zeros_like(a) if eqx.is_array(a) else a, blk.mlp)
        return eqx.tree_at(lambda b: b.mlp, blk, zeroed)
    x_res1 = _zero_mlp(single)(x, inference=True)
    assert jnp.allclose(_zero_mlp(legacy)(x, inference=True), 2.0 * x_res1, atol=1e-6)
    # with the MLP live the two differ by exactly x_res1
    assert jnp.allclose(legacy(x, inference=True) - single(x, inference=True),
                        x_res1, atol=1e-5)

    # SwinLayer must forward rms_norm to its blocks (it used to land in **_unused)
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
