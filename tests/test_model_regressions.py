"""Regression guards: legacy residual flag, rms_norm forwarding, weight-decay masking."""

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
    layer = SwinLayer(
        2,
        dim,
        depth=2,
        num_heads=2,
        grid_size=grid,
        window_size=win,
        key=jr.PRNGKey(2),
        rms_norm=True,
    )
    assert all(isinstance(b.norm1, RMSNorm) and isinstance(b.norm2, RMSNorm) for b in layer.blocks)
    layer_ln = SwinLayer(
        2,
        dim,
        depth=1,
        num_heads=2,
        grid_size=grid,
        window_size=win,
        key=jr.PRNGKey(2),
        rms_norm=False,
    )
    assert not isinstance(layer_ln.blocks[0].norm1, RMSNorm)
    layer_legacy = SwinLayer(
        2,
        dim,
        depth=1,
        num_heads=2,
        grid_size=grid,
        window_size=win,
        key=jr.PRNGKey(2),
        legacy_double_shortcut=True,
    )
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
            return DiTSwinBlock(
                dim, 2, cond_dim, (4, 8), (2, 4), key=k, legacy_double_shortcut=legacy
            )
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


def test_layers_reject_unknown_kwargs():
    from neugk_jax.models.swin import DiTSwinLayer, FilmSwinLayer, SwinLayer
    from neugk_jax.models.vit import DiTLayer, FilmViTLayer, ViTLayer

    common = dict(key=jr.PRNGKey(0), rms_nrom=True)
    with pytest.raises(TypeError):
        SwinLayer(2, 8, depth=1, num_heads=2, grid_size=(4, 4), window_size=(2, 2), **common)
    with pytest.raises(TypeError):
        DiTSwinLayer(
            2, 8, depth=1, num_heads=2, grid_size=(4, 4), window_size=(2, 2), cond_dim=4, **common
        )
    with pytest.raises(TypeError):
        FilmSwinLayer(
            2, 8, depth=1, num_heads=2, grid_size=(4, 4), window_size=(2, 2), cond_dim=4, **common
        )
    with pytest.raises(TypeError):
        ViTLayer(2, 8, depth=1, num_heads=2, grid_size=(4, 4), **common)
    with pytest.raises(TypeError):
        DiTLayer(2, 8, depth=1, num_heads=2, grid_size=(4, 4), cond_dim=4, **common)
    with pytest.raises(TypeError):
        FilmViTLayer(2, 8, depth=1, num_heads=2, grid_size=(4, 4), cond_dim=4, **common)


def test_ae_backbone_has_no_dead_middle():
    from neugk_jax.autoencoders import Swin5DAE

    ae = Swin5DAE(
        decouple_mu=True,
        dim=8,
        base_resolution=[4, 4, 4, 16, 8],
        in_channels=2,
        out_channels=2,
        patch_size=[2, 0, 2, 4, 2],
        window_size=[2, 0, 2, 2, 2],
        depth=[1],
        num_heads=[2],
        num_layers=1,
        bottleneck_dim=8,
        bottleneck_depth=1,
        bottleneck_num_heads=2,
        c_multiplier=1,
        key=jr.PRNGKey(0),
    )
    assert ae.backbone.middle is None and ae.backbone.middle_upscale is None
    assert ae(jnp.zeros((2, 4, 4, 4, 16, 8)))["df"].shape == (2, 4, 4, 4, 16, 8)


def test_ae_dit_builders_accept_mappings(tmp_path):
    import yaml

    from neugk_jax.translate import build_ae_from_config, build_dit_from_config

    ae_cfg = {
        "model": {
            "latent_dim": 8,
            "num_layers": 1,
            "decouple_mu": True,
            "patch": {
                "patch_size": [2, 0, 2, 4, 2],
                "window_size": [2, 0, 2, 2, 2],
                "c_multiplier": 1,
                "merging_depth": 1,
                "unmerging_depth": 1,
            },
            "vit": {"num_heads": [2], "depth": [1], "drop_path": 0.3},
            "bottleneck": {"dim": 8, "depth": 1, "num_heads": 2},
        },
        "dataset": {"separate_zf": False, "resolution": [4, 4, 4, 16, 8]},
    }
    path = tmp_path / "ae.yaml"
    path.write_text(yaml.safe_dump(ae_cfg))
    ae_path = build_ae_from_config(str(path), key=jr.PRNGKey(0))
    ae = build_ae_from_config(OmegaConf.create(ae_cfg), key=jr.PRNGKey(0))
    assert jax.tree_util.tree_structure(ae) == jax.tree_util.tree_structure(ae_path)
    assert ae.middle_pre.drop_path == 0.3
    dit_cfg = {
        "model": {
            "latent_dim": 16,
            "conditioning": ["itg", "dg"],
            "vit": {"num_heads": 2, "depth": 1},
        }
    }
    dit = build_dit_from_config(dit_cfg, ae, key=jr.PRNGKey(0))
    assert dit.backbone.mlp_ratio == 2.0 and dit.backbone.drop_path == 0.1
    assert dit.cond_embed is not None


def test_conditioning_slots_follow_sorted_names():
    import numpy as np

    from neugk_jax.training.runner import conditioning_slots

    ds_conds = sorted(["itg", "dg", "s_hat", "q", "timestep"])
    for order in (["itg", "dg", "s_hat", "q"], ["q", "s_hat", "dg", "itg"]):
        slots = conditioning_slots(ds_conds, order)
        assert [ds_conds[i] for i in slots] == ["dg", "itg", "q", "s_hat"]
    assert conditioning_slots(ds_conds, []) is None
    assert np.asarray(conditioning_slots(ds_conds, ["timestep"])).tolist() == [4]
