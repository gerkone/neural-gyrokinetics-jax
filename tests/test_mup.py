"""muP of the swin AE: width rule, per-leaf multipliers, readout scale and zero init."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from omegaconf import OmegaConf

from neugk_jax.models.build import build_ae_from_config
from neugk_jax.models.mup import build_multipliers
from neugk_jax.models.utils import trainable_mask
from neugk_jax.training.runner import build_optimizer

BASE = (8, 4, 4, 8, 4)


def cfg(width=64):
    patch = dict(
        patch_size=[2, 0, 2, 4, 2], window_size=[2, 0, 2, 2, 2], merging_depth=2, unmerging_depth=2,
        merging_hidden_ratio=1.0, unmerging_hidden_ratio=1.0, c_multiplier=1,
    )
    mup = dict(enable=True, head_dim=16, bottleneck_ratio=2, base_width=32, delta_width=48, output_mult=1.0)
    return {
        "model": {
            "latent_dim": width, "num_layers": 1, "init_weights": "kaiming_uniform", "patch": patch, "mup": mup,
            "vit": {"num_heads": [4], "depth": [1]}, "bottleneck": {"dim": 8, "depth": 1, "num_heads": 2},
        },
        "dataset": {"resolution": list(BASE), "separate_zf": False},
        "training": {"learning_rate": 2.4e-3, "weight_decay": 1e-6, "adam_b2": 0.95},
    }


def test_mup_model_and_multipliers():
    c = cfg()
    build = lambda w: build_ae_from_config(c, key=jr.PRNGKey(0), width=w)
    model = build(64)
    # heads and bottleneck follow the width; the readout scales by base / width and starts at zero
    assert model.bottleneck_dim == 32
    unpatch = model.backbone.unpatch
    assert unpatch.in_mult == 0.5
    assert float(jnp.abs(unpatch.expansion.layers[0].weight).max()) == 0.0
    assert float(jnp.abs(unpatch.expansion.layers[1].weight).max()) > 0.0
    mask = trainable_mask(model)
    lr, wd = build_multipliers(build, model, mask, 32, 48)
    named = {jax.tree_util.keystr(k): v for k, v in jax.tree_util.tree_flatten_with_path(lr)[0]}
    assert named[".backbone.down_blocks[0].swin.blocks[0].attn.qkv.inner.weight"] == 0.5
    assert named[".backbone.patch_embed.patch.layers[0].inner.weight"] == 1.0
    assert named[".backbone.unpatch.expansion.layers[0].inner.weight"] == 1.0
    assert sorted(set(named.values())) == [0.5, 1.0]
    wd_named = {jax.tree_util.keystr(k): v for k, v in jax.tree_util.tree_flatten_with_path(wd)[0]}
    assert wd_named[".backbone.down_blocks[0].swin.blocks[0].attn.qkv.inner.weight"] == 2.0


def test_mup_adam_step():
    c = cfg()
    build = lambda w: build_ae_from_config(c, key=jr.PRNGKey(0), width=w)
    model = build(64)
    mask = trainable_mask(model)
    mults = build_multipliers(build, model, mask, 32, 48)
    opt = build_optimizer(
        optax.constant_schedule(2.4e-3), OmegaConf.create(c["training"]), model,
        decoupled=False, b2=0.95, mask=mask, multipliers=mults,
    )
    params, static = eqx.partition(model, mask)
    state = opt.init(params)
    x = jr.normal(jr.PRNGKey(1), (2, *BASE))
    loss = lambda p: jnp.mean((eqx.combine(p, static)(x)["df"] - x) ** 2)
    # zero readout: the first reconstruction is exactly zero
    assert float(jnp.abs(model(x)["df"]).max()) == 0.0
    grads = eqx.filter_grad(loss)(params)
    updates, _ = opt.update(grads, state, params)
    leaves = jax.tree_util.tree_leaves(updates)
    assert all(bool(jnp.isfinite(u).all()) for u in leaves)
    # adam moves every trainable leaf by about lr times its multiplier in the first step
    qkv = lambda t: t.backbone.down_blocks[0].swin.blocks[0].attn.qkv.weight
    readout = lambda t: t.backbone.unpatch.expansion.layers[0].weight
    assert float(jnp.abs(qkv(updates)).max()) < 0.6 * float(jnp.abs(readout(updates)).max())
