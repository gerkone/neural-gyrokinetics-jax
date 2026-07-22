"""GyrosplatDataset: cache structure, normalization roundtrip, conditioning order.

Runs against a tiny synthetic cache built in tmp_path; the real-cache checks are
gated on NEUGK_GYROSPLAT_CACHE.
"""

from __future__ import annotations

import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

from neugk_jax.dataset.gyrosplat import (
    N_STAT_CHANNELS,
    STATS_TYPE_ID,
    GyrosplatDataset,
    GyrosplatSample,
    collate,
)
from neugk_jax.gyrosplats.splat import N_CHANNELS

REAL_CACHE = os.environ.get("NEUGK_GYROSPLAT_CACHE", "/local00/bioinf/galletti/gyrosplats_cache")

N_ENV, TIED_GROUPS, TIED_K = 6, 4, 7
N_ATOMS = N_ENV + TIED_GROUPS * TIED_K
TWO_PI = 2.0 * np.pi


def _bins():
    tail = np.tile(np.arange(7, 14, dtype=np.int32), TIED_GROUPS)
    return np.concatenate([np.zeros(N_ENV, dtype=np.int32), tail])


def make_synth_cache(tmp_path):
    """Tiny synthetic gyrosplat cache — shared with the e2e test."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    bins = _bins()
    np.save(tmp_path / "bins.npy", bins)
    t_steps = 5
    # dataset-wide FROZEN carrier scaffold (mu + both cholesky factors, ch 0:14),
    # shared per tied group — mirrors the validated real-data fact so the v7
    # "windows" layout can extract a constant scaffold from this cache
    carrier = np.arange(N_ENV, N_ATOMS).reshape(TIED_GROUPS, TIED_K)
    scaf = rng.normal(size=(TIED_GROUPS, 14)).astype(np.float32)
    scaf[:, :5] = rng.uniform(size=(TIED_GROUPS, 5))
    for name, seed in (("iteration_1", 1), ("iteration_2", 2)):
        d = tmp_path / name
        d.mkdir()
        rng = np.random.default_rng(seed)
        atoms = rng.normal(size=(t_steps, N_ATOMS, 17)).astype(np.float32)
        atoms[..., :5] = rng.uniform(size=(t_steps, N_ATOMS, 5))
        atoms[..., 16] = TWO_PI * bins[None, :] + 0.01 * rng.normal(size=(t_steps, N_ATOMS))
        # freeze carrier geometry (only their amps ch 14:16 vary across time)
        for g in range(TIED_GROUPS):
            atoms[:, carrier[g], :14] = scaf[g]
        np.save(d / "atoms.npy", atoms)
        zf = np.abs(rng.normal(size=(t_steps, 4))).astype(np.float32) + 0.5
        zf[:, 2] = 0.0
        np.save(d / "zfstats.npy", zf)
        np.save(d / "params.npy", np.array([10.0, 2.5, 4.5, 3.0], dtype=np.float32))  # itg dg q s_hat
        np.save(d / "flux.npy", rng.normal(size=(90,)).astype(np.float32))
        (d / "meta.json").write_text(json.dumps({"traj": name, "steps": []}))
    # channel stats mirroring the converter
    all_atoms = np.concatenate(
        [np.load(tmp_path / n / "atoms.npy") for n in ("iteration_1", "iteration_2")]
    )
    all_zf = np.concatenate(
        [np.load(tmp_path / n / "zfstats.npy") for n in ("iteration_1", "iteration_2")]
    )
    flat = all_atoms.reshape(-1, 17).astype(np.float64)
    mean, std = flat.mean(0), flat.std(0)
    mean[:5], std[:5] = 0.5, 0.5
    dky = (all_atoms[..., 16] - TWO_PI * bins[None, :]).reshape(-1)
    mean[16], std[16] = dky.mean(), max(dky.std(), 1e-8)
    stat_vals = np.stack([all_zf[:, 0], np.log(all_zf[:, 1]), np.log(all_zf[:, 3])], 1)
    np.savez(
        tmp_path / "channel_stats.npz",
        mean=mean.astype(np.float32),
        std=np.maximum(std, 1e-8).astype(np.float32),
        stat_mean=stat_vals.mean(0).astype(np.float32),
        stat_std=np.maximum(stat_vals.std(0), 1e-8).astype(np.float32),
    )
    (tmp_path / "converted.json").write_text(json.dumps(["iteration_1", "iteration_2"]))
    return tmp_path


@pytest.fixture()
def synth_cache(tmp_path):
    return make_synth_cache(tmp_path)


def test_dataset_shapes_and_types(synth_cache):
    ds = GyrosplatDataset(cache_path=str(synth_cache), offset=80)
    assert len(ds) == 10
    s = ds[0]
    assert s.tokens.shape == (N_ATOMS, N_CHANNELS)
    assert s.conditioning.shape == (4,)
    assert set(np.unique(ds.type_ids[:N_ENV])) == {0}
    batch = collate([ds[0], ds[1]])
    assert isinstance(batch, GyrosplatSample)
    assert batch.tokens.shape == (2, N_ATOMS, N_CHANNELS)
    # stats-token variant keeps the trailing global token
    ds2 = GyrosplatDataset(cache_path=str(synth_cache), stats_token=True)
    assert ds2[0].tokens.shape == (N_ATOMS + 1, N_CHANNELS)
    assert ds2.type_ids[-1] == STATS_TYPE_ID


def test_conditioning_sorted_order(synth_cache):
    # params.npy is [itg, dg, q, s_hat]; sorted conditioning order is (dg, itg, q, s_hat)
    ds = GyrosplatDataset(cache_path=str(synth_cache))
    s = ds[0]
    assert ds.conditions == ["dg", "itg", "q", "s_hat"]
    np.testing.assert_allclose(s.conditioning, [2.5, 10.0, 4.5, 3.0])


def test_normalized_moments_and_roundtrip(synth_cache):
    ds = GyrosplatDataset(cache_path=str(synth_cache))
    toks = np.stack([ds[i].tokens for i in range(len(ds))])
    # per-channel z-scored (except mu channels which are affine to ~[-1,1])
    m, s = toks.reshape(-1, 17).mean(0), toks.reshape(-1, 17).std(0)
    assert np.all(np.abs(m[5:]) < 0.15), m
    assert np.all(np.abs(s[5:] - 1.0) < 0.15), s
    assert toks[..., :5].min() >= -1.5 and toks[..., :5].max() <= 1.5

    raw = np.asarray(ds.atoms[0][0])
    p = ds.denormalize_atom_tokens(jnp.asarray(ds[0].tokens))
    rt = np.concatenate(
        [p.mu, p.L_phys_raw, p.L_vel_raw, p.amps, np.asarray(p.ky)[:, None]], axis=1
    )
    np.testing.assert_allclose(rt, raw, atol=1e-5)


def test_asinh_roundtrip(synth_cache):
    ds = GyrosplatDataset(cache_path=str(synth_cache), asinh_channels=(14, 15))
    raw = np.asarray(ds.atoms[0][0])
    p = ds.denormalize_atom_tokens(jnp.asarray(ds[0].tokens))
    np.testing.assert_allclose(np.asarray(p.amps), raw[:, 14:16], atol=1e-4)


def test_frozen_ky_mask(synth_cache):
    ds = GyrosplatDataset(cache_path=str(synth_cache), ky_mode="frozen")
    assert np.all(ds[0].tokens[:, 16] == 0.0)
    assert np.all(ds.loss_mask[:, 16] == 0.0)
    ds2 = GyrosplatDataset(cache_path=str(synth_cache), stats_token=True)
    mask = ds2.loss_mask
    assert np.all(mask[-1, N_STAT_CHANNELS:] == 0.0)
    assert np.all(mask[-1, :N_STAT_CHANNELS] == 1.0)


@pytest.mark.skipif(not os.path.isdir(REAL_CACHE), reason="real gyrosplat cache absent")
def test_real_cache_loads():
    ds = GyrosplatDataset(cache_path=REAL_CACHE, trajectories="iteration_{13,131,235}")
    assert len(ds) == 3 * 186
    s = ds[0]
    assert s.tokens.shape == (1597, 17)
    assert np.isfinite(s.tokens).all()
    # normalized atom channels should be roughly standardized on the real cache
    m = s.tokens.mean(0)
    assert np.all(np.abs(m) < 2.0)
