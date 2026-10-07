"""Field patching: drop-in swap for the linear patch embedding / unpatch."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neugk_jax.models.build import build_ae_from_config
from neugk_jax.models.field_patching import FieldPatchEmbed, FieldUnpatch, field_options
from neugk_jax.models.gk_unet import SwinNDUnet
from neugk_jax.models.patching import PatchEmbed, PatchExpand
from neugk_jax.pinc import Swin5DAE

BASE = (8, 4, 4, 8, 4)


def grid_5d():
    return {
        "axes": [
            {"kind": "absolute", "nodes": list(np.linspace(-3, 3, 8)), "weights": [1.0] * 8},
            {"kind": "relative", "spacing": 0.06},
            {"kind": "relative", "spacing": 0.6},
            {"kind": "relative", "spacing": 15.0},
        ],
        "folded": [{"nodes": list(np.linspace(0.1, 2.0, 4) ** 2), "weights": [1.0] * 4}],
    }


def small_ae(**kw):
    return Swin5DAE(
        space=5,
        decouple_mu=True,
        dim=16,
        base_resolution=BASE,
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
        merging_depth=2,
        unmerging_depth=1,
        merging_hidden_ratio=2.0,
        unmerging_hidden_ratio=2.0,
        key=jr.PRNGKey(0),
        **kw,
    )


def unet(patching, **kw):
    return SwinNDUnet(
        space=3,
        dim=16,
        base_resolution=(8, 6, 8),
        in_channels=3,
        out_channels=3,
        patch_size=(2, 3, 4),
        window_size=(2, 2, 2),
        depth=1,
        num_heads=2,
        num_layers=1,
        patching=patching,
        patching_kwargs=kw or None,
        key=jr.PRNGKey(0),
    )


def test_linear_is_the_default():
    model = unet("linear")
    assert type(model.patch_embed) is PatchEmbed and type(model.unpatch) is PatchExpand


@pytest.mark.parametrize("decoder", ["deeponet", "hier"])
def test_generic_nd_swap(decoder):
    linear, field = unet("linear"), unet("field", decoder=decoder, rank=16, hidden=16, code_rank=8, branch=32)
    assert isinstance(field.patch_embed, FieldPatchEmbed) and isinstance(field.unpatch, FieldUnpatch)
    x = jr.normal(jr.PRNGKey(1), (3, 8, 6, 8))
    z = field.patch_encode(x)
    assert z.shape == linear.patch_encode(x).shape
    assert field.patch_decode(z).shape == x.shape


@pytest.mark.parametrize("decoder,encoding", [("deeponet", "fourier"), ("hier", "fourier"), ("hier", "cosine")])
def test_adiabatic_5d_grid(decoder, encoding):
    opts = dict(decoder=decoder, encoding=encoding, rank=16, hidden=16, code_rank=8, branch=32, grid=grid_5d())
    ae = small_ae(patching="field", patching_kwargs=opts)
    x = jr.normal(jr.PRNGKey(1), (2, *BASE))
    out = ae(x)["df"]
    assert out.shape == x.shape
    # zero-initialized code layer: the reconstruction starts at zero
    assert float(jnp.abs(out).max()) == 0.0
    # the per-sample geometry (spacings of s, x, y) enters the patch weights
    z0 = ae.encode(x)
    z1 = ae.encode(x, geometry=jnp.asarray([0.12, 1.2, 30.0]))
    assert float(jnp.abs(z1 - z0).max()) > 0
    loss = lambda m: jnp.mean((m(x)["df"] - x) ** 2)
    grads = eqx.filter_grad(loss)(ae)
    leaves = [g for g in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))]
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    assert float(jnp.abs(grads.backbone.unpatch.expansion.layers[-1].weight).max()) > 0


def test_build_from_config():
    patch = dict(
        patch_size=[2, 0, 2, 4, 2], window_size=[2, 0, 2, 2, 2], merging_depth=2, unmerging_depth=1,
        merging_hidden_ratio=2.0, unmerging_hidden_ratio=2.0, c_multiplier=2,
        type="field", field={"rank": 16, "hidden": 16, "code_rank": 8, "branch": 32}, grid=grid_5d(),
    )
    cfg = {
        "model": {
            "latent_dim": 16, "num_layers": 1, "patch": patch,
            "vit": {"num_heads": [2], "depth": [1]}, "bottleneck": {"dim": 8, "depth": 1, "num_heads": 2},
        },
        "dataset": {"resolution": list(BASE), "separate_zf": False},
    }
    ae = build_ae_from_config(cfg, key=jr.PRNGKey(0))
    assert isinstance(ae.backbone.patch_embed, FieldPatchEmbed)
    assert ae(jnp.zeros((2, *BASE)))["df"].shape == (2, *BASE)


def test_unknown_option_raises():
    with pytest.raises(ValueError, match="unknown field patching options"):
        field_options({"ranks": 4})
