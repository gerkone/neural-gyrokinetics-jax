"""Shared-initialization neural fields and the representation-interpolation script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import COND, RES, make_traj, tiny_ae, tiny_vq_cfg
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def nf_cfg(tmp_path, density_lr: float) -> OmegaConf:
    weights = {k: 1.0 for k in ("df", "phi_int", "flux_int", "kyspec", "qspec")}
    density = {"epochs": 1, "lr": density_lr, "min_lr": 0.0, "weight_decay": 0.0}
    density.update(batch_size=64, subsample=[1.0, 1.0], scan_steps=4)
    pinc = {"epochs": 1, "lr": 1e-3, "min_lr": 1e-8, "warmup": 1, "weight_decay": 0.0}
    return OmegaConf.create(
        {
            "workflow": "nf",
            "output_path": str(tmp_path / "out"),
            "model": {"name": "mlp", "dim": 16, "n_layers": 3, "skips": True},
            "dataset": {
                "path": str(tmp_path),
                "backend": "numpy",
                "training_trajectories": [],
                "validation_trajectories": ["iteration_0", "iteration_1"],
                "timesteps": [0, 2],
                "spectral_offset": 1,
            },
            "training": {
                "pool_size": 2,
                "density": density,
                "pinc": {**pinc, "loss_weights": weights},
                "shared_init": {
                    "enabled": True,
                    "rounds": 2,
                    "epochs": 1,
                    "lr": 1e-2,
                    "weight_decay": 0.0,
                },
            },
            "logging": {"mode": "disabled"},
        }
    )


@pytest.fixture(scope="module")
def shared_run(tmp_path_factory):
    from neugk_jax.pinc.nf_runner import NFRunner

    tmp_path = tmp_path_factory.mktemp("interp")
    rng = np.random.default_rng(0)
    for i in range(2):
        spec = {k: rng.uniform(0.0, 1.0, (4, RES[-1])) for k in ("kyspec", "fluxspec")}
        make_traj(tmp_path, f"iteration_{i}", n_t=4, **spec)
    # a zero density lr keeps every density field at its trajectory's shared initialization
    cfg = nf_cfg(tmp_path, density_lr=0.0)
    NFRunner(cfg, output_path=cfg.output_path)()
    return tmp_path


def leaves(path):
    from neugk_jax.pinc.neural_field import MLPNF
    from neugk_jax.training.checkpoint import load_model_only

    m = load_model_only(path, MLPNF(RES, key=jr.PRNGKey(0), dim=16, n_layers=3))
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))]


def test_shared_init_aligns_the_fields_of_a_trajectory(shared_run):
    out = shared_run / "out"
    shared = {i: leaves(out / "shared_init" / f"iteration_{i}.eqx") for i in range(2)}
    assert not all(np.array_equal(a, b) for a, b in zip(shared[0], shared[1]))
    for i in range(2):
        for t in (0, 2):
            (path,) = out.glob(f"best_mlp_iteration_{i}_t{t}_x*.eqx")
            assert all(np.array_equal(a, b) for a, b in zip(leaves(path), shared[i]))


def test_mean_weights_and_nf_midpoint(shared_run):
    from eval_interp import mean_weights, nf_midpoint

    from neugk_jax.pinc.benchmark.reconstructors import load_nf
    from neugk_jax.pinc.neural_field import sample_field
    from neugk_jax.pinc.nf_runner import snapshot_norm

    out = shared_run / "out"
    (pa,) = out.glob("best_int_mlp_iteration_0_t0_x*.eqx")
    (pb,) = out.glob("best_int_mlp_iteration_0_t2_x*.eqx")
    a, b = load_nf(str(pa), RES), load_nf(str(pb), RES)
    mid = mean_weights(a, b)
    np.testing.assert_allclose(
        mid.readout.weight, 0.5 * (a.readout.weight + b.readout.weight), rtol=1e-6
    )
    rng = np.random.default_rng(1)
    df_a, df_b = (jnp.asarray(rng.standard_normal((2, *RES)), jnp.float32) for _ in range(2))
    (sa, ha), (sb, hb) = snapshot_norm(df_a), snapshot_norm(df_b)
    ref = sample_field(mid, RES) * 0.5 * (sa + sb) + 0.5 * (ha + hb)
    np.testing.assert_allclose(nf_midpoint(str(pa), str(pb), df_a, df_b), ref, rtol=1e-6)


@pytest.mark.parametrize("vq", [False, True])
def test_latent_midpoint_of_equal_snapshots_is_the_reconstruction(vq):
    from eval_interp import _latent_midpoint

    from neugk_jax.models.build import build_ae_from_config

    if vq:
        cfg = {"model": tiny_vq_cfg("fsq"), "dataset": {"resolution": RES}}
        model = build_ae_from_config(cfg, key=jr.PRNGKey(0))
    else:
        model = tiny_ae()
    # the ae separates the zonal flow, the vq-vae does not
    x = jr.normal(jr.PRNGKey(1), (2 if vq else 4, *RES))
    np.testing.assert_allclose(
        _latent_midpoint(model, x, x, COND), model(x, COND)["df"], rtol=1e-5, atol=1e-5
    )


def test_eval_interp_script(shared_run, tmp_path):
    out = tmp_path / "interp.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_interp.py"), "--nf-ckpts"]
    cmd += [str(shared_run / "out"), "--path", str(shared_run), "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=ROOT)
    res = json.loads(out.read_text())
    assert res["trajectories"] == ["iteration_0", "iteration_1"]
    assert [r["t"] for r in res["rows"]] == [1, 1]
    keys = {"Extremes", "f (data)", "NF (weights)", "PINC-NF (weights)"}
    assert set(res["agg"]) == keys
    assert all(np.isfinite(v["psnr"][0]) for v in res["agg"].values())
