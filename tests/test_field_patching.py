"""Field patching: drop-in swap for the linear patch embedding / unpatch."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neugk_jax.models.build import build_ae_from_config
from neugk_jax.models.gk_unet import SwinNDUnet, patching_options
from neugk_jax.models.patching import (
    PATCHINGS,
    FieldPatchEmbed,
    FieldUnpatch,
    LinearUnpatch,
    PatchEmbed,
    PointGrid,
    SmoothPatchEmbed,
    fold_patches,
)
from neugk_jax.models.patching.field import AxisBases, tucker_project, tucker_synthesize
from neugk_jax.pinc import Swin5DAE

BASE = (8, 4, 4, 8, 4)
SMALL = {"smooth": dict(rank=8, hidden=8, code_rank=4), "tucker": dict(axis_hidden=8)}


def field_pair(kind, patch=(2, 5), base=(4, 10), channels=3, dim=8, **kw):
    """Embedding / unpatch of a field kind on a 2D grid, each with the options it takes."""
    embed_cls, unpatch_cls = PATCHINGS[kind]
    e_kw, u_kw = patching_options(embed_cls, unpatch_cls, {**SMALL[kind], **kw})
    grid_size = tuple(b // p for b, p in zip(base, patch))
    embed = embed_cls(base, patch, channels, dim, key=jr.PRNGKey(0), **e_kw)
    unpatch = unpatch_cls(
        dim, grid_size, key=jr.PRNGKey(1), expand_by=patch, out_channels=channels, **u_kw
    )
    return embed, unpatch


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
    assert type(model.patch_embed) is PatchEmbed and type(model.unpatch) is LinearUnpatch


@pytest.mark.parametrize("kind", ["smooth", "tucker"])
def test_generic_nd_swap(kind):
    linear = unet("linear")
    field = unet(kind, **SMALL[kind])
    assert isinstance(field.patch_embed, FieldPatchEmbed) and isinstance(
        field.unpatch, FieldUnpatch
    )
    x = jr.normal(jr.PRNGKey(1), (3, 8, 6, 8))
    z = field.patch_encode(x)
    assert z.shape == linear.patch_encode(x).shape
    assert field.patch_decode(z).shape == x.shape


@pytest.mark.parametrize("kind", ["smooth", "tucker"])
def test_adiabatic_5d_grid(kind):
    ae = small_ae(patching=kind, patching_kwargs={**SMALL[kind], "grid": grid_5d()})
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
    unpatch = grads.backbone.unpatch
    assert float(jnp.abs(unpatch.expansion.layers[-1].weight).max()) > 0


@pytest.mark.parametrize("kind", ["smooth", "tucker"])
def test_default_heads_are_linear_in_the_data(kind):
    embed, unpatch = field_pair(kind, zero_init=False)
    assert len(embed.mix.layers) == 1 and len(unpatch.expansion.layers) == 1
    x, y = jr.normal(jr.PRNGKey(2), (2, 4, 10, 3))
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(embed(2 * x - y), 2 * embed(x) - embed(y), atol=1e-4)
        z = embed(x)
        np.testing.assert_allclose(unpatch(3 * z), 3 * unpatch(z), atol=1e-4)


def test_cell_centred_coordinates_keep_physical_positions():
    # a patch of 10 cells at spacing 1 and of 5 cells at spacing 2 cover the same interval
    fine = PointGrid((10,), (10,), 1, {"axes": [{"kind": "relative", "spacing": 1.0}]})
    coarse = PointGrid((5,), (5,), 1, {"axes": [{"kind": "relative", "spacing": 2.0}]})
    x_fine = np.asarray(fine.offsets[:, 0]) * fine.half[0] * fine.spacing[0]
    x_coarse = np.asarray(coarse.offsets[:, 0]) * coarse.half[0] * coarse.spacing[0]
    np.testing.assert_allclose(x_coarse, (x_fine[0::2] + x_fine[1::2]) / 2, atol=1e-6)
    np.testing.assert_allclose(fine.scale(), coarse.scale(), atol=1e-6)


def test_tucker_cosine_full_rank_is_exact():
    # orthonormal cosines at full rank: projection then synthesis is the identity on every patch
    spec = {"axes": [{"kind": "relative"}] * 3}
    grid = PointGrid((6, 4, 10), (3, 2, 5), 2, spec)
    ranks = grid.patch
    bases = AxisBases(grid, ranks, key=jr.PRNGKey(0), basis="cosine", hidden=8, modes=16)(grid)
    p = fold_patches(jr.normal(jr.PRNGKey(1), (6, 4, 10, 2)), grid.patch)
    with jax.default_matmul_precision("highest"):
        back = tucker_synthesize(tucker_project(p, grid, bases), grid, bases, ranks)
    np.testing.assert_allclose(back, p, atol=1e-5)


@pytest.mark.parametrize("kind", ["smooth", "tucker"])
def test_one_set_of_weights_on_two_resolutions(kind):
    # x refined by 2 with the same physical patch: the weights of one grid run on the other
    spec = lambda dx: {
        "axes": [
            {"kind": "relative", "spacing": 0.1},
            {"kind": "relative", "spacing": dx, "reference": 0.5},
        ]
    }
    extra = {"smooth": dict(code_modes=[1, 2]), "tucker": dict(ranks=[2, 10])}[kind]
    embed, unpatch = field_pair(kind, grid=spec(0.2), zero_init=False, **extra)
    fine = PointGrid((4, 20), (2, 10), 3, spec(0.1))
    x = jr.normal(jr.PRNGKey(2), (4, 20, 3))
    z = embed.with_grid(fine)(x)
    assert z.shape == (2, 2, 8)
    assert unpatch.with_grid(fine)(z).shape == x.shape


def test_build_from_config():
    patch = dict(
        patch_size=[2, 0, 2, 4, 2],
        window_size=[2, 0, 2, 2, 2],
        merging_depth=2,
        unmerging_depth=1,
        merging_hidden_ratio=2.0,
        unmerging_hidden_ratio=2.0,
        c_multiplier=2,
        type="smooth",
        field={"rank": 16, "hidden": 16, "code_rank": 8},
        grid=grid_5d(),
    )
    cfg = {
        "model": {
            "latent_dim": 16,
            "num_layers": 1,
            "patch": patch,
            "vit": {"num_heads": [2], "depth": [1]},
            "bottleneck": {"dim": 8, "depth": 1, "num_heads": 2},
        },
        "dataset": {"resolution": list(BASE), "separate_zf": False},
    }
    ae = build_ae_from_config(cfg, key=jr.PRNGKey(0))
    assert isinstance(ae.backbone.patch_embed, SmoothPatchEmbed)
    assert ae(jnp.zeros((2, *BASE)))["df"].shape == (2, *BASE)


def test_unknown_option_raises():
    with pytest.raises(ValueError, match="unknown patching options"):
        unet("smooth", rnk=4)
    # an option of the other field kind is unknown too
    with pytest.raises(ValueError, match="unknown patching options"):
        unet("tucker", code_rank=4)


@pytest.mark.parametrize("kind", ["smooth", "tucker"])
def test_point_conditioning_enters_the_filters(kind):
    embed, unpatch = field_pair(kind, cond_features=3, zero_init=False)
    x = jr.normal(jr.PRNGKey(2), (4, 10, 3))
    ion, electron = jnp.asarray([1.0, 0.0, 0.0]), jnp.asarray([-1.0, -1.0, 0.3])
    z_ion, z_el = embed(x, point_cond=ion), embed(x, point_cond=electron)
    assert float(jnp.abs(z_ion - z_el).max()) > 0
    assert (
        float(jnp.abs(unpatch(z_ion, point_cond=ion) - unpatch(z_ion, point_cond=electron)).max())
        > 0
    )


def test_band_limit_keeps_the_trained_modes_on_a_finer_grid():
    # data resolving 5 modes per patch, sampled on 10 points: cosines above the band are cut
    spec = {"axes": [{"kind": "relative", "band": 5}]}
    fine = PointGrid((10,), (10,), 1, spec)
    assert fine.axes[0].cap == 5 and fine.caps[0] == 5
    bases = AxisBases(fine, (10,), key=jr.PRNGKey(0), basis="cosine", hidden=8, modes=16)(fine)
    assert (
        float(jnp.abs(bases[0][:, 5:]).max()) == 0.0 and float(jnp.abs(bases[0][:, :5]).max()) > 0
    )
