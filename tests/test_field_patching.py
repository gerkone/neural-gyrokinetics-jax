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
from neugk_jax.models.patching.field import DCTBases, tucker_project, tucker_synthesize
from neugk_jax.pinc import Swin5DAE

BASE = (8, 4, 4, 8, 4)
SMALL = {"smooth": dict(rank=8, hidden=8), "dct": dict(hidden=8)}
KINDS = ["smooth", "dct"]


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


def refined(dx):
    """2D spec, s at spacing 0.1 and x at ``dx``, with a fixed reference half-width."""
    return {
        "axes": [
            {"kind": "relative", "spacing": 0.1},
            {"kind": "relative", "spacing": dx, "reference": 0.5},
        ]
    }


def shifted(module):
    """``module`` off its zero-initialized layers."""
    return jax.tree_util.tree_map(lambda a: a + 0.1 if eqx.is_inexact_array(a) else a, module)


def test_linear_is_the_default():
    model = unet("linear")
    assert type(model.patch_embed) is PatchEmbed and type(model.unpatch) is LinearUnpatch


@pytest.mark.parametrize("kind", KINDS)
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


@pytest.mark.parametrize("kind", KINDS)
def test_adiabatic_5d_grid(kind):
    ae = small_ae(patching=kind, patching_kwargs={**SMALL[kind], "grid": grid_5d()})
    x = jr.normal(jr.PRNGKey(1), (2, *BASE))
    out = ae(x)["df"]
    assert out.shape == x.shape
    # zero-initialized code layer: the reconstruction starts at zero
    assert float(jnp.abs(out).max()) == 0.0
    # the per-sample geometry enters the smooth filters; the dct hypernetworks start at zero
    z0 = ae.encode(x)
    z1 = ae.encode(x, geometry=jnp.asarray([0.12, 1.2, 30.0]))
    assert (float(jnp.abs(z1 - z0).max()) > 0) == (kind == "smooth")
    grads = eqx.filter_grad(lambda m: jnp.mean((m(x)["df"] - x) ** 2))(ae)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    assert float(jnp.abs(grads.backbone.unpatch.expansion.layers[-1].weight).max()) > 0


@pytest.mark.parametrize("kind", KINDS)
def test_default_heads_are_linear_in_the_data(kind):
    embed, unpatch = field_pair(kind, zero_init=False)
    assert len(embed.mix.layers) == 1 and len(unpatch.expansion.layers) == 1
    x, y = jr.normal(jr.PRNGKey(2), (2, 4, 10, 3))
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(embed(2 * x - y), 2 * embed(x) - embed(y), atol=1e-4)
        z = embed(x)
        np.testing.assert_allclose(unpatch(3 * z), 3 * unpatch(z), atol=1e-4)


@pytest.mark.parametrize("kind", KINDS)
def test_one_set_of_weights_on_two_resolutions(kind):
    # x refined by 2 with the same physical patch: the weights of one grid run on the other
    extra = {"smooth": dict(code_modes=[1, 2]), "dct": dict(ranks=[2, 10])}[kind]
    embed, unpatch = field_pair(kind, grid=refined(0.2), zero_init=False, **extra)
    fine = PointGrid((4, 20), (2, 10), 3, refined(0.1))
    x = jr.normal(jr.PRNGKey(2), (4, 20, 3))
    z = embed.with_grid(fine)(x)
    assert z.shape == (2, 2, 8)
    assert unpatch.with_grid(fine)(z).shape == x.shape


@pytest.mark.parametrize("kind", KINDS)
def test_one_basis_for_every_channel(kind):
    # rolling the channels of the input rolls the channels of the projections
    embed, unpatch = field_pair(kind, zero_init=False)
    x = jr.normal(jr.PRNGKey(2), (4, 10, 3))
    # smooth projections are (code mode, channel, rank), dct cores (*ranks, channel)
    shape, axis = (
        ((2, 2, -1, 3, 8), -2) if kind == "smooth" else ((2, 2, *embed.basis.ranks, 3), -1)
    )
    h, rolled = (
        embed.project(fold_patches(v, embed.grid.patch)).reshape(shape)
        for v in (x, jnp.roll(x, 1, -1))
    )
    assert jnp.allclose(rolled, jnp.roll(h, 1, axis), atol=1e-4)
    assert unpatch(embed(x)).shape == x.shape


def test_cell_centred_coordinates_keep_physical_positions():
    # a patch of 10 cells at spacing 1 and of 5 cells at spacing 2 cover the same interval
    fine = PointGrid((10,), (10,), 1, {"axes": [{"kind": "relative", "spacing": 1.0}]})
    coarse = PointGrid((5,), (5,), 1, {"axes": [{"kind": "relative", "spacing": 2.0}]})
    x_fine = np.asarray(fine.offsets[:, 0]) * fine.half[0] * fine.spacing[0]
    x_coarse = np.asarray(coarse.offsets[:, 0]) * coarse.half[0] * coarse.spacing[0]
    np.testing.assert_allclose(x_coarse, (x_fine[0::2] + x_fine[1::2]) / 2, atol=1e-6)
    np.testing.assert_allclose(fine.scale(), coarse.scale(), atol=1e-6)


def test_filter_features_do_not_depend_on_the_grid():
    # the cell centres of a 4-point patch are also points of a 12-point patch: same features there
    coarse = PointGrid(
        (4,), (4,), 1, {"axes": [{"kind": "relative", "spacing": 0.3, "reference": 0.6}]}
    )
    fine = PointGrid(
        (12,), (12,), 1, {"axes": [{"kind": "relative", "spacing": 0.1, "reference": 0.6}]}
    )
    f_coarse, f_fine = coarse.features(None, 16), fine.features(None, 16)
    np.testing.assert_allclose(f_coarse, f_fine[1::3], atol=1e-6)
    # dct bases still stop at the modes a grid can hold
    assert coarse.axes[0].cap == 4 and fine.axes[0].cap == 12


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
        field={"rank": 16, "hidden": 16},
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
        unet("dct", code_modes=(1, 1))


def test_fixed_dct_at_full_rank_is_exact():
    # orthonormal dct modes at full rank: projection then synthesis is the identity on every patch
    grid = PointGrid((6, 4, 10), (3, 2, 5), 2, {"axes": [{"kind": "relative"}] * 3})
    bases = DCTBases(
        grid, grid.patch, key=jr.PRNGKey(0), hidden=8, modes=16, learned=False, position=False
    )
    p = fold_patches(jr.normal(jr.PRNGKey(1), (6, 4, 10, 2)), grid.patch)
    with jax.default_matmul_precision("highest"):
        b = bases(grid)
        back = tucker_synthesize(tucker_project(p, grid, b), grid, b, grid.patch)
    np.testing.assert_allclose(back, p, atol=1e-4)


def test_dct_modes_stop_at_the_band():
    # data resolving 5 modes per patch, sampled on 10 points: modes above the band are cut
    fine = PointGrid((10,), (10,), 1, {"axes": [{"kind": "relative", "band": 5}]})
    assert fine.axes[0].cap == 5 and fine.bands[0] == 5
    b = DCTBases(fine, (10,), key=jr.PRNGKey(0), hidden=8, modes=16, learned=False, position=False)
    b = b(fine)[0]
    assert float(jnp.abs(b[:, 5:]).max()) == 0.0 and float(jnp.abs(b[:, :5]).max()) > 0


def test_learned_dct_starts_as_the_fixed_dct():
    grid = PointGrid((6, 10), (3, 5), 2, {"axes": [{"kind": "relative"}] * 2})
    kw = dict(hidden=8, modes=16, position=False)
    with jax.default_matmul_precision("highest"):
        fixed = DCTBases(grid, (3, 4), key=jr.PRNGKey(0), learned=False, **kw)(grid)
        learned = DCTBases(grid, (3, 4), key=jr.PRNGKey(1), learned=True, **kw)(grid)
    for a, b in zip(fixed, learned):
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_fixed_dct_has_no_learned_basis():
    embed, _ = field_pair("dct", ranks=[2, 4], learned=False)
    assert jax.tree_util.tree_leaves(eqx.filter(embed.basis, eqx.is_inexact_array)) == []
    assert embed(jr.normal(jr.PRNGKey(2), (4, 10, 3))).shape == (2, 2, 8)


def test_learned_dct_off_init_on_a_refined_grid():
    embed, unpatch = field_pair("dct", grid=refined(0.2), zero_init=False, ranks=[2, 4])
    embed = shifted(embed)
    fine = PointGrid((4, 20), (2, 10), 3, refined(0.1))
    z = embed.with_grid(fine)(jr.normal(jr.PRNGKey(2), (4, 20, 3)))
    assert z.shape == (2, 2, 8) and unpatch.with_grid(fine)(z).shape == (4, 20, 3)


def test_patch_position_starts_as_the_shared_model():
    base = field_pair("dct", zero_init=False, ranks=[2, 4])
    pos = field_pair("dct", zero_init=False, ranks=[2, 4], position=True)
    x = jr.normal(jr.PRNGKey(2), (4, 10, 3))
    # tf32 tolerance: the per-token bases contract in another order
    z = base[0](x)
    assert jnp.allclose(z, pos[0](x), atol=1e-3)
    assert jnp.allclose(base[1](z), pos[1](z), atol=1e-3)


def test_patch_position_gives_each_token_its_bases():
    embed, _ = field_pair("dct", ranks=[2, 4], position=True)
    embed = shifted(embed)
    bases = embed.basis(embed.grid)
    assert bases[1].shape == (2, 5, 4)
    assert not jnp.allclose(bases[1][0], bases[1][1])
    assert embed(jr.normal(jr.PRNGKey(2), (4, 10, 3))).shape == (2, 2, 8)


def test_patch_position_needs_the_learned_dct():
    with pytest.raises(ValueError):
        field_pair("dct", learned=False, position=True)


def test_band_limited_smooth_projects_the_same_on_a_finer_grid():
    embed, _ = field_pair("smooth", grid=refined(0.2), zero_init=False, band_limited=True)
    a = jr.normal(jr.PRNGKey(3), (2, 3, 3))

    def field(q):
        # per x patch a cosine polynomial of degree < 3, constant in s
        u = (2 * jnp.arange(q) + 1) / q - 1
        c = jnp.cos(jnp.arange(3)[:, None] * jnp.pi * (u + 1) / 2)
        f = jnp.einsum("pmc,mu->puc", a, c).reshape(-1, 3)
        return jnp.broadcast_to(f, (4, *f.shape))

    coarse = PointGrid((4, 10), (2, 5), 3, refined(0.2))
    fine = PointGrid((4, 20), (2, 10), 3, refined(0.1))
    h = [
        embed.with_grid(g).project(fold_patches(field(q), g.patch))
        for g, q in ((coarse, 5), (fine, 10))
    ]
    assert float(jnp.linalg.norm(h[0] - h[1]) / jnp.linalg.norm(h[0])) < 1e-3


def test_band_limited_smooth_on_the_5d_grid():
    kw = {**SMALL["smooth"], "band_limited": True, "grid": grid_5d()}
    ae = small_ae(patching="smooth", patching_kwargs=kw)
    x = jr.normal(jr.PRNGKey(1), (2, *BASE))
    grads = eqx.filter_grad(lambda m: jnp.mean((m(x)["df"] - x) ** 2))(ae)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
