"""Continuous-convolution patching: drop-in swap for the linear patch embedding / unpatch."""

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
    BandLimitedPatchEmbed,
    CConvPatchEmbed,
    CConvUnpatch,
    LinearUnpatch,
    PatchEmbed,
    PointGrid,
    TuckerPatchEmbed,
    TuckerUnpatch,
    fold_patches,
)
from neugk_jax.pinc import Swin5DAE

BASE = (8, 4, 4, 8, 4)
SMALL = {"cconv": dict(rank=8), "tucker": dict(hidden=8)}
KINDS = ["cconv", "tucker"]


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
    assert isinstance(field.patch_embed, CConvPatchEmbed) and isinstance(
        field.unpatch, CConvUnpatch
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
    extra = {"cconv": dict(code_modes=[1, 2]), "tucker": dict(ranks=[2, 10])}[kind]
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
    # cconv projections are (code mode, channel, rank), tucker cores (*ranks, channel)
    shape, axis = ((2, 2, -1, 3, 8), -2) if kind == "cconv" else ((2, 2, *embed.ranks, 3), -1)
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
    x_fine = np.asarray(fine.pos[:, 0]) * fine.half[0] * fine.spacing[0]
    x_coarse = np.asarray(coarse.pos[:, 0]) * coarse.half[0] * coarse.spacing[0]
    np.testing.assert_allclose(x_coarse, (x_fine[0::2] + x_fine[1::2]) / 2, atol=1e-6)
    np.testing.assert_allclose(fine.scale(), coarse.scale(), atol=1e-6)


def test_build_from_config():
    patch = dict(
        patch_size=[2, 0, 2, 4, 2],
        window_size=[2, 0, 2, 2, 2],
        merging_depth=2,
        unmerging_depth=1,
        merging_hidden_ratio=2.0,
        unmerging_hidden_ratio=2.0,
        c_multiplier=2,
        type="cconv",
        field={"rank": 16},
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
    assert isinstance(ae.backbone.patch_embed, BandLimitedPatchEmbed)
    assert ae(jnp.zeros((2, *BASE)))["df"].shape == (2, *BASE)


def test_unknown_option_raises():
    with pytest.raises(ValueError, match="unknown patching options"):
        unet("cconv", rnk=4)
    # an option of the other field kind is unknown too
    with pytest.raises(ValueError, match="unknown patching options"):
        unet("tucker", code_modes=(1, 1))


def tucker_on(grid_shape, patch, ranks, spec):
    """A tucker embedding of one channel on its own grid."""
    return TuckerPatchEmbed(
        grid_shape, patch, 1, 4, key=jr.PRNGKey(0), ranks=ranks, hidden=8, grid=spec
    )


def test_tucker_at_full_rank_starts_exact():
    # orthonormal dct modes at full rank: projection then synthesis is the identity on every patch
    spec, patch = {"axes": [{"kind": "relative"}] * 3}, (3, 2, 5)
    embed = tucker_on((6, 4, 10), patch, patch, spec)
    unpatch = TuckerUnpatch(
        4, (2, 2, 2), key=jr.PRNGKey(1), expand_by=patch, out_channels=1, ranks=patch, grid=spec
    )
    p = fold_patches(jr.normal(jr.PRNGKey(1), (6, 4, 10, 1)), patch)
    with jax.default_matmul_precision("highest"):
        back = unpatch.synthesize(embed.project(p))
    np.testing.assert_allclose(back, p, atol=1e-4)


def test_tucker_modes_stop_at_the_band():
    # data resolving 5 modes per patch, sampled on 10 points: modes above the band are cut
    embed = tucker_on((10,), (10,), (10,), {"axes": [{"kind": "relative", "band": 5}]})
    assert embed.grid.axes[0].cap == 5
    b = embed.kernel()[0]
    assert float(jnp.abs(b[:, 5:]).max()) == 0.0 and float(jnp.abs(b[:, :5]).max()) > 0


def test_tucker_off_init_on_a_refined_grid():
    embed, unpatch = field_pair("tucker", grid=refined(0.2), zero_init=False, ranks=[2, 4])
    embed = shifted(embed)
    fine = PointGrid((4, 20), (2, 10), 3, refined(0.1))
    z = embed.with_grid(fine)(jr.normal(jr.PRNGKey(2), (4, 20, 3)))
    assert z.shape == (2, 2, 8) and unpatch.with_grid(fine)(z).shape == (4, 20, 3)


def test_cconv_projects_the_same_on_a_finer_grid():
    # band-limited filters: exact quadrature on every grid that resolves the modes of the field
    embed, _ = field_pair("cconv", grid=refined(0.2), zero_init=False)
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


def test_cconv_on_the_5d_grid():
    kw = {**SMALL["cconv"], "grid": grid_5d()}
    ae = small_ae(patching="cconv", patching_kwargs=kw)
    x = jr.normal(jr.PRNGKey(1), (2, *BASE))
    grads = eqx.filter_grad(lambda m: jnp.mean((m(x)["df"] - x) ** 2))(ae)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
