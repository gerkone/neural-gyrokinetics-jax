"""v7 "windows" layout: scaffold extraction, pack/unpack + DCT invertibility,
state shape, normalization roundtrip, and GyrosplatWindowDiT over all option
combos (base / dct / rope / dct+rope)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_gyrosplat_dataset import N_ATOMS, N_ENV, TIED_GROUPS, TIED_K, _bins, make_synth_cache

from neugk_jax.dataset.gyrosplat import GyrosplatDataset
from neugk_jax.gyrosplats.model import GyrosplatWindowDiT
from neugk_jax.gyrosplats.normalize import denormalize_state, normalize_state
from neugk_jax.gyrosplats.splat import pack
from neugk_jax.gyrosplats.window import (
    N_STATE_CH,
    dct_matrix,
    extract_scaffold,
    pack_windows,
    scaffold_max_dev,
    unpack_windows,
    window_dct,
    window_idct,
)


@pytest.fixture()
def synth_cache(tmp_path):
    return make_synth_cache(tmp_path)


def _atoms(synth_cache, name="iteration_1"):
    return np.load(synth_cache / name / "atoms.npy")


def test_dct_orthonormal_and_roundtrip():
    d = dct_matrix(TIED_K)
    np.testing.assert_allclose(d @ d.T, np.eye(TIED_K), atol=1e-12)
    w = jr.normal(jr.PRNGKey(0), (5, 2 * TIED_K))
    rt = window_idct(window_dct(w, jnp.asarray(d)), jnp.asarray(d))
    np.testing.assert_allclose(np.asarray(rt), np.asarray(w), atol=1e-5)


def test_scaffold_constant(synth_cache):
    atoms = _atoms(synth_cache)
    bins = _bins()
    # frozen synthetic scaffold -> deviation must be ~0
    assert scaffold_max_dev(atoms[:4], bins) < 1e-5
    scaf = extract_scaffold(atoms[:4], bins)
    assert scaf.mu.shape == (TIED_GROUPS, 5)
    assert scaf.L_phys_raw.shape == (TIED_GROUPS, 6)
    assert scaf.L_vel_raw.shape == (TIED_GROUPS, 3)
    assert scaf.ky.shape == (TIED_K,)
    np.testing.assert_allclose(
        np.asarray(scaf.ky), 2 * np.pi * np.arange(7, 14), atol=1e-4
    )


def test_pack_unpack_roundtrip(synth_cache):
    atoms = _atoms(synth_cache)
    bins = _bins()
    scaf = extract_scaffold(atoms[:4], bins)
    state = pack_windows(atoms[2], scaf)
    assert state.shape == (N_ENV + TIED_GROUPS, N_STATE_CH)
    full = np.asarray(pack(unpack_windows(jnp.asarray(state), scaf)))
    env = np.asarray(scaf.env_idx)
    grp = np.asarray(scaf.grp_idx).reshape(-1)
    # envelope mu/L/amps exact, envelope ky zeroed
    np.testing.assert_allclose(full[env, :16], atoms[2][env, :16], atol=1e-5)
    np.testing.assert_allclose(full[env, 16], 0.0, atol=1e-6)
    # carrier amps exact, carrier ky == 2*pi*m
    np.testing.assert_allclose(full[grp, 14:16], atoms[2][grp, 14:16], atol=1e-5)
    np.testing.assert_allclose(full[grp, 16], 2 * np.pi * bins[grp], atol=1e-4)


def test_pack_amps_exact_all_carriers(synth_cache):
    """Amp round-trip must be EXACT for envelopes and all carriers (regression guard):
    window packing must scatter each group's 14 channels back to its 7 carriers with
    the [Re_7,Im_7,...,Re_13,Im_13] interleave, never collapse to 121 rows."""
    atoms = _atoms(synth_cache)
    bins = _bins()
    scaf = extract_scaffold(atoms[:4], bins)
    env = np.asarray(scaf.env_idx)
    gr = np.asarray(scaf.grp_idx).reshape(-1)
    for t in (0, 2, 4):
        full = np.asarray(pack(unpack_windows(jnp.asarray(pack_windows(atoms[t], scaf)), scaf)))
        # every one of the 847 carriers keeps its own amplitude, exactly
        np.testing.assert_allclose(full[gr, 14:16], atoms[t][gr, 14:16], atol=1e-6)
        np.testing.assert_allclose(full[env, 14:16], atoms[t][env, 14:16], atol=1e-6)


def test_dataset_decode_exact_when_scaffold_matches(synth_cache):
    """On the synthetic cache the carrier geometry IS frozen, so the FULL dataset
    pipeline (pack -> normalize -> denormalize -> scaffold decode) must reproduce the
    original bank exactly (amps + geometry), except envelope ky which is dropped."""
    ds = GyrosplatDataset(cache_path=str(synth_cache), layout="windows")
    atoms = _atoms(synth_cache)
    gr = np.asarray(ds.scaffold.grp_idx).reshape(-1)
    env = np.asarray(ds.scaffold.env_idx)
    p = ds.decode_state(jnp.asarray(ds[3].tokens))
    full = np.asarray(pack(p))
    np.testing.assert_allclose(full[env, :16], atoms[3][env, :16], atol=1e-4)
    np.testing.assert_allclose(full[gr, :16], atoms[3][gr, :16], atol=1e-4)


def test_normalize_state_roundtrip(synth_cache):
    for dct in (False, True):
        ds = GyrosplatDataset(cache_path=str(synth_cache), layout="windows", dct=dct)
        raw = pack_windows(_atoms(synth_cache)[0], ds.scaffold)
        norm = normalize_state(jnp.asarray(raw), ds.window_stats)
        back = denormalize_state(norm, ds.window_stats)
        np.testing.assert_allclose(np.asarray(back), raw, atol=1e-4)


def test_windows_dataset(synth_cache):
    ds = GyrosplatDataset(cache_path=str(synth_cache), layout="windows")
    assert ds.n_tokens == N_ENV + TIED_GROUPS
    assert ds.n_channels == N_STATE_CH
    s = ds[0]
    assert s.tokens.shape == (N_ENV + TIED_GROUPS, N_STATE_CH)
    # window pad channels masked
    mask = ds.loss_mask
    assert np.all(mask[N_ENV:, 14:] == 0.0)
    assert np.all(mask[:N_ENV] == 1.0)
    # decode returns a full 1597-equivalent bank
    p = ds.decode_state(jnp.asarray(s.tokens))
    assert p.mu.shape == (N_ATOMS, 5)


def _model(rope=False, key=0):
    scaf_mu = np.asarray(jr.uniform(jr.PRNGKey(99), (TIED_GROUPS, 5)))
    return GyrosplatWindowDiT(
        scaffold_mu=scaf_mu, n_env=N_ENV, n_cond=4, n_hidden=64, n_layers=2,
        n_head=4, type_dim=8, win_embed_dim=8, rope=rope, key=jr.PRNGKey(key),
    )


def _inputs(key=0):
    ks = jr.split(jr.PRNGKey(key), 2)
    x = jr.normal(ks[0], (N_ENV + TIED_GROUPS, N_STATE_CH))
    return x, jnp.float32(0.3), jr.normal(ks[1], (4,))


@pytest.mark.parametrize("rope", [False, True])
def test_windows_model_fwd_bwd(rope):
    model = _model(rope=rope)
    x, t, cond = _inputs()
    out = model(x, t, cond)
    assert out.shape == (N_ENV + TIED_GROUPS, N_STATE_CH)
    assert jnp.isfinite(out).all()

    def loss(m):
        return jnp.mean(m(x, t, cond) ** 2)

    _, grads = eqx.filter_value_and_grad(loss)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert leaves and all(jnp.isfinite(g).all() for g in leaves)


@pytest.mark.parametrize("rope", [False, True])
def test_windows_adaln_zero_identity(rope):
    # adaln-zero: at init conditioning must not affect the output
    model = _model(rope=rope)
    x, t, cond = _inputs()
    o1 = model(x, t, cond)
    o2 = model(x, jnp.float32(0.9), -2.0 * cond)
    np.testing.assert_allclose(np.asarray(o1), np.asarray(o2), atol=1e-6)


@pytest.mark.parametrize("rope", [False, True])
def test_windows_envelope_permutation_equivariance(rope):
    # permuting envelope rows permutes their outputs; window rows stay fixed
    model = _model(rope=rope)
    x, t, cond = _inputs()
    perm = np.arange(N_ENV + TIED_GROUPS)
    perm[:N_ENV] = np.random.default_rng(0).permutation(N_ENV)
    out = model(x, t, cond)
    out_p = model(x[jnp.asarray(perm)], t, cond)
    np.testing.assert_allclose(np.asarray(out)[perm], np.asarray(out_p), atol=1e-4)
    # window outputs (fixed identity) unchanged
    np.testing.assert_allclose(
        np.asarray(out)[N_ENV:], np.asarray(out_p)[N_ENV:], atol=1e-4
    )
