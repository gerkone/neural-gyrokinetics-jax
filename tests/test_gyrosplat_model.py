"""GyrosplatDiT (flat per-atom conditioned transformer): shapes, grads, init, equivariance."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neugk_jax.gyrosplats.model import GyrosplatDiT

N_ENV, TIED_GROUPS, TIED_K = 10, 3, 7
N_ATOMS = N_ENV + TIED_GROUPS * TIED_K


def _bins():
    return np.concatenate(
        [np.zeros(N_ENV, np.int32), np.tile(np.arange(7, 14, dtype=np.int32), TIED_GROUPS)]
    )


@pytest.fixture(scope="module")
def model():
    return GyrosplatDiT(
        bins=_bins(), n_cond=4, n_hidden=64, n_layers=2, n_head=4, type_dim=8,
        key=jr.PRNGKey(0),
    )


def _inputs(key=0, n_tokens=N_ATOMS):
    ks = jr.split(jr.PRNGKey(key), 3)
    x = jr.normal(ks[0], (n_tokens, 17))
    t = jnp.float32(0.3)
    cond = jr.normal(ks[2], (4,))
    return x, t, cond


def test_forward_shape_finite(model):
    x, t, cond = _inputs()
    out = model(x, t, cond)
    assert out.shape == (N_ATOMS, 17)
    assert jnp.isfinite(out).all()


def test_grads_finite(model):
    x, t, cond = _inputs()

    def loss(m):
        return jnp.mean(m(x, t, cond) ** 2)

    _, grads = eqx.filter_value_and_grad(loss)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert leaves
    assert all(jnp.isfinite(g).all() for g in leaves)


def test_adaln_zero_init_identity(model):
    """at init the modulation is zero, so conditioning must not affect the output."""
    x, t, cond = _inputs()
    out1 = model(x, t, cond)
    out2 = model(x, jnp.float32(0.9), -3.0 * cond)
    np.testing.assert_allclose(np.asarray(out1), np.asarray(out2), atol=1e-6)


def test_permutation_equivariance_within_type(model):
    """permuting same-type atom tokens permutes outputs identically."""
    x, t, cond = _inputs()
    perm = np.arange(N_ATOMS)
    perm[:N_ENV] = np.random.default_rng(0).permutation(N_ENV)  # envelope block only
    out = model(x, t, cond)
    out_p = model(x[jnp.asarray(perm)], t, cond)
    np.testing.assert_allclose(np.asarray(out)[perm], np.asarray(out_p), atol=1e-5)


def test_bin_identity_breaks_cross_type_symmetry(model):
    """swapping an envelope token with a carrier token must NOT commute (type embed)."""
    x, t, cond = _inputs()
    perm = np.arange(N_ATOMS)
    perm[0], perm[N_ENV] = N_ENV, 0  # swap envelope 0 with a bin-7 carrier
    out = model(x, t, cond)
    out_p = model(x[jnp.asarray(perm)], t, cond)
    assert not np.allclose(np.asarray(out)[perm], np.asarray(out_p), atol=1e-5)
