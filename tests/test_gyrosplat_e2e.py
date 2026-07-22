"""End-to-end gyrosplat flow matching: full train + eval loop on a tiny cache.

Mirrors ``test_e2e_loop.py``: BaseRunner.__call__ runs an epoch, checkpoints
save + resume, pairing modes produce valid couplings.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from omegaconf import OmegaConf
from test_gyrosplat_dataset import make_synth_cache


def _cfg(cache, out_path, **overrides):
    cfg = OmegaConf.create(
        {
            "workflow": "gyrosplat",
            "seed": 0,
            "output_path": str(out_path),
            "model": {
                "name": "gyrosplat_fm",
                "n_hidden": 32,
                "n_layers": 1,
                "n_head": 2,
                "slice_num": 4,
                "type_dim": 4,
                "pairing": "sliced",
                "ot_fraction": 0.8,
            },
            "dataset": {
                "name": "gyrosplat",
                "cache_path": str(cache),
                "training_trajectories": ["iteration_1"],
                "validation_trajectories": ["iteration_2"],
                "conditions": ["itg", "dg", "s_hat", "q"],
                "offset": 80,
                "ky_mode": "delta",
            },
            "training": {
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-3,
                "final_learning_rate": 1e-6,
                "weight_decay": 0.0,
                "clip_grad": True,
                "clip_to": 1.0,
            },
            "validation": {"validate_every_n_epochs": 1, "eval_sampling": False},
            "logging": None,
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def test_runner_one_epoch_and_resume(tmp_path):
    from neugk_jax.gyrosplats.runner import GyrosplatFMRunner

    cache = make_synth_cache(tmp_path / "cache")
    out = tmp_path / "run"
    runner = GyrosplatFMRunner(_cfg(cache, out), output_path=str(out))
    runner()
    assert (out / "ckp.eqx").exists()
    logs = runner.evaluate(1)
    assert np.isfinite(logs["fm_loss"])

    # resume: a fresh runner picks up the checkpoint
    runner2 = GyrosplatFMRunner(_cfg(cache, out), output_path=str(out))
    assert runner2.start_epoch == 1


def test_sample_and_decode(tmp_path):
    from neugk_jax.gyrosplats.runner import GyrosplatFMRunner

    cache = make_synth_cache(tmp_path / "cache")
    out = tmp_path / "run"
    runner = GyrosplatFMRunner(_cfg(cache, out), output_path=str(out))
    cond = jnp.stack([jnp.asarray(runner.val_ds[0].conditioning)] * 2)
    toks = runner.sample(key=jr.PRNGKey(0), batch=2, cond=cond, steps=3)
    assert toks.shape == (2, runner.train_ds.n_tokens, 17)
    p, st = runner.decode(toks[0])
    assert p.mu.shape == (runner.train_ds.n_atoms, 5)
    assert float(st.zonal_std) > 0.0 and float(st.fluc_std) > 0.0


@pytest.mark.parametrize("mode", ["sliced", "morton", "sinkhorn", "hungarian"])
def test_pairing_valid_within_population(tmp_path, mode):
    from neugk_jax.gyrosplats.pairing import _populations, make_pair_fn

    cache = make_synth_cache(tmp_path / "cache")
    bins = np.load(cache / "bins.npy")
    pair_fn = make_pair_fn(bins, mode=mode, ot_fraction=1.0)
    n_tok = bins.shape[0] + 1
    key = jr.PRNGKey(0)
    x0 = jr.normal(key, (2, n_tok, 17))
    x1 = jr.normal(jr.PRNGKey(1), (2, n_tok, 17))
    out = np.asarray(pair_fn(key, x0, x1))
    x0_np = np.asarray(x0)
    for b in range(2):
        # rows are a permutation of the originals, within populations only
        for idx in _populations(bins):
            got = out[b][idx]
            src = x0_np[b][idx]
            # set equality by sorting rows lexicographically
            np.testing.assert_allclose(
                np.sort(got, axis=0), np.sort(src, axis=0), atol=1e-6
            )
        # stats token untouched
        np.testing.assert_allclose(out[b][-1], x0_np[b][-1])
        # paired mu closer (or equal) on average than independent
        mu_dist = np.abs(out[b][:-1, :5] - np.asarray(x1)[b][:-1, :5]).mean()
        mu_dist_ind = np.abs(x0_np[b][:-1, :5] - np.asarray(x1)[b][:-1, :5]).mean()
        assert mu_dist <= mu_dist_ind + 1e-6


def test_independent_mode_is_none():
    from neugk_jax.gyrosplats.pairing import make_pair_fn

    assert make_pair_fn(np.zeros(4, np.int32), mode="independent") is None


@pytest.mark.parametrize("dct,rope", [(False, False), (True, False), (False, True), (True, True)])
def test_windows_runner_one_epoch(tmp_path, dct, rope):
    """v7 windows layout: full train step + sample + decode for all option combos."""
    from neugk_jax.gyrosplats.runner import GyrosplatFMRunner

    cache = make_synth_cache(tmp_path / "cache")
    out = tmp_path / "run"
    cfg = _cfg(
        cache, out,
        model={"n_hidden": 32, "n_head": 2, "dct": dct, "rope": rope, "pairing": "morton"},
        dataset={"layout": "windows"},
    )
    runner = GyrosplatFMRunner(cfg, output_path=str(out))
    runner()
    assert (out / "ckp.eqx").exists()
    cond = jnp.stack([jnp.asarray(runner.val_ds[0].conditioning)] * 2)
    toks = runner.sample(key=jr.PRNGKey(0), batch=2, cond=cond, steps=3)
    assert toks.shape == (2, runner.train_ds.n_tokens, runner.train_ds.n_channels)
    p, st = runner.decode(toks[0])
    assert p.mu.shape == (runner.train_ds.n_atoms, 5)
    assert np.isfinite(np.asarray(p.amps)).all()
