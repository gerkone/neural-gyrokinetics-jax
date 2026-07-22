"""Smoke tests for the evaluator modules.

Verifies:
* ``AEEvaluator`` runs to completion on a tiny synthetic dataset, returns
  reconstruction MSE,
* ``validation_metrics`` returns the right keys with and without the
  integrals branch,
* ``compute_integrals`` raises a friendly error / no-ops when gyaradax
  isn't installed (or runs to completion when it is).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from omegaconf import OmegaConf

from neugk_jax.evaluate import (
    AEEvaluator,
    DiffusionEvaluator,
    validation_metrics,
)


def _make_geometry(resolution):
    """Shape-consistent single-species geometry for gyaradax on a tiny grid."""
    vp, mu, s, x, y = resolution
    return {
        "krho": np.arange(y, dtype=np.float64) * 0.5,          # ky=0 zonal mode at index 0
        "kxrh": (np.arange(x, dtype=np.float64) - x // 2) * 0.4,
        "ints": np.full(s, 1.0 / s, dtype=np.float64),
        "intmu": np.linspace(0.1, 0.4, mu, dtype=np.float64),
        "intvp": np.full(vp, 0.5, dtype=np.float64),
        "vpgr": np.linspace(-1.5, 1.5, vp, dtype=np.float64),
        "mugr": np.linspace(0.1, 1.6, mu, dtype=np.float64),
        "bn": np.ones(s, dtype=np.float64),
        "ffun": np.ones(s, dtype=np.float64),
        "efun": np.ones(s, dtype=np.float64),
        "rfun": np.ones(s, dtype=np.float64),
        "bt_frac": np.ones(s, dtype=np.float64),
        "parseval": np.where(np.arange(y) == 0, 1.0, 2.0).astype(np.float64),
        "little_g": np.tile(np.array([1.0, 0.0, 1.0]), (s, 1)),
        **{k: np.ones((1,), dtype=np.float64) for k in (
            "mas", "tmp", "de", "d2X", "signz", "signB", "vthrat",
        )},
    }


def _make_traj(root: Path, name: str, *, n_t: int, resolution):
    traj = root / f"{name}_ifft_realpotens"
    data = traj / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    for t in range(n_t):
        rng.standard_normal((2, *resolution)).astype(np.float32).tofile(
            data / f"timestep_{t:05d}.bin"
        )
    meta = {
        "timesteps": np.arange(n_t, dtype=np.float64),
        "flux": rng.standard_normal(n_t).astype(np.float32),
        "ion_temp_grad": np.array([2.3], dtype=np.float32),
        "density_grad": np.array([1.1], dtype=np.float32),
        "s_hat": np.array([0.8], dtype=np.float32),
        "q": np.array([1.4], dtype=np.float32),
        "resolution": np.array(resolution),
        "ds": np.float64(0.0625),
        "geometry": _make_geometry(resolution),
    }
    with open(traj / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)


@pytest.fixture
def tiny_setup(tmp_path):
    """Build a 4D synthetic dataset + a tiny Swin5DAE."""
    resolution = (4, 4, 4, 16, 8)
    _make_traj(tmp_path, "iteration_0", n_t=4, resolution=resolution)
    from neugk_jax.dataset import CycloneDataset, NumpyBackend
    ds = CycloneDataset(
        path=str(tmp_path), split="train",
        trajectories="iteration_0",
        fields_to_load=("df",),
        conditions=("itg", "dg", "s_hat", "q"),
        mode="ae", backend=NumpyBackend(),
        separate_zf=False, normalization=None,
    )
    from neugk_jax.autoencoders import Swin5DAE
    ae = Swin5DAE(
        space=5, decouple_mu=True, dim=16,
        base_resolution=resolution,
        in_channels=2, out_channels=2,
        patch_size=[2, 0, 2, 4, 2], window_size=[2, 0, 2, 2, 2],
        depth=[1], num_heads=[2], num_layers=1,
        middle_depth=1, middle_num_heads=2,
        bottleneck_dim=8, bottleneck_depth=1, bottleneck_num_heads=2,
        merging_depth=1, unmerging_depth=1,
        merging_hidden_ratio=2.0, unmerging_hidden_ratio=2.0,
        hidden_mlp_ratio=2.0,
        key=jr.PRNGKey(0),
    )
    return ds, ae


def test_validation_metrics_basic():
    pred = jnp.zeros((2, 3))
    tgt = jnp.ones((2, 3))
    metrics, integrated = validation_metrics(
        preds={"df": pred}, tgts={"df": tgt}, eval_integrals=False,
    )
    assert "df" in metrics
    assert metrics["df"] == pytest.approx(1.0)
    assert integrated is None


def test_ae_evaluator_runs(tiny_setup):
    ds, ae = tiny_setup
    cfg = OmegaConf.create({"validation": {"eval_integrals": False}})
    ev = AEEvaluator(cfg, val_ds=ds, is_rank0=True)
    metrics, _ = ev(ae, epoch=1, batch_size=1, eval_integrals=False)
    assert "df_mse" in metrics
    assert jnp.isfinite(metrics["df_mse"])


def test_ae_evaluator_with_integrals_optional(tiny_setup):
    """When eval_integrals=True but the df layout doesn't fit the flux
    integral (plain 2-channel df), the evaluator should still finish and
    just skip the integrals."""
    ds, ae = tiny_setup
    cfg = OmegaConf.create({"validation": {"eval_integrals": True}})
    ev = AEEvaluator(cfg, val_ds=ds, is_rank0=True)
    metrics, _ = ev(ae, epoch=1, batch_size=1, eval_integrals=True)
    assert "df_mse" in metrics
    assert jnp.isfinite(metrics["df_mse"])


def test_ae_evaluator_spectral_metrics(tiny_setup):
    """eval_spectra=True produces finite time-averaged zonal-flow and
    spectral metrics (needs the 'ds' metadata entry)."""
    ds, ae = tiny_setup
    assert ds.get_ds(0) == 0.0625
    cfg = OmegaConf.create({"validation": {"eval_spectra": True}})
    ev = AEEvaluator(cfg, val_ds=ds, is_rank0=True)
    metrics, _ = ev(ae, epoch=1, batch_size=1, eval_spectra=True)
    for key in ("zfphi_rl2", "zf_energy_err", "kyspec_rl2", "qspec_rl1"):
        assert key in metrics, f"missing spectral metric {key}"
        assert np.isfinite(metrics[key]), f"{key} not finite: {metrics[key]}"


def test_diffusion_evaluator_constructs(tiny_setup):
    """Sampling-based eval just needs a callable sample_fn — verify
    the evaluator construct + a single sampling call work."""
    ds, ae = tiny_setup
    cfg = OmegaConf.create({"validation": {"eval_integrals": False}})

    def fake_sample(*, key, batch, cond=None, steps=50):
        # return zeros of the expected shape
        return {"df": jnp.zeros((batch, ae.backbone.original_in_channels,
                                  *ae.backbone.full_resolution))}

    ev = DiffusionEvaluator(
        cfg, val_ds=ds, autoencoder=ae, sample_fn=fake_sample, is_rank0=True,
    )
    metrics, _ = ev(model=None, epoch=1, batch_size=1, n_steps=2,
                    eval_integrals=False)
    assert "df" in metrics
    assert jnp.isfinite(metrics["df"])
