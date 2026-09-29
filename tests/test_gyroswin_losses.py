"""GyroSwin loss terms, loss-config resolution and the jittable flux-integral core."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neugk_jax.losses import l1, relative_norm_mse
from neugk_jax.training.loss_scheduler import LossConfig, compute_multi_task_loss


def test_relative_norm_mse_matches_formula():
    rng = np.random.default_rng(0)
    p, y = rng.standard_normal((3, 4, 5)), rng.standard_normal((3, 4, 5))
    ref = np.mean([np.sum((p[b] - y[b]) ** 2) / (np.sum(y[b] ** 2) + 1e-4) for b in range(3)])
    assert float(relative_norm_mse(jnp.asarray(p), jnp.asarray(y))) == pytest.approx(ref, rel=1e-5)


def test_l1_reshapes_target():
    p = jnp.asarray([[1.0], [2.0], [4.0]])
    y = jnp.asarray([0.0, 3.0, 4.0])
    assert float(l1(p, y)) == pytest.approx((1.0 + 1.0 + 0.0) / 3)


def test_multi_task_loss_forms():
    rng = np.random.default_rng(1)
    preds = {"df": jnp.asarray(rng.standard_normal((2, 4, 3))),
             "phi": jnp.asarray(rng.standard_normal((2, 5))),
             "fluxavg": jnp.asarray(rng.standard_normal((2, 1)))}
    tgts = {"df": jnp.asarray(rng.standard_normal((2, 4, 3))),
            "phi": jnp.asarray(rng.standard_normal((2, 5))),
            "fluxavg": jnp.asarray(rng.standard_normal(2))}
    w = {"df": 1.0, "phi": 0.5, "fluxavg": 2.0}
    total, parts = compute_multi_task_loss(preds, tgts, w, ("df", "phi", "fluxavg"))
    assert float(parts["df"]) == pytest.approx(float(relative_norm_mse(preds["df"], tgts["df"])))
    assert float(parts["fluxavg"]) == pytest.approx(
        float(jnp.mean(jnp.abs(preds["fluxavg"][:, 0] - tgts["fluxavg"]))))
    assert float(total) == pytest.approx(float(parts["df"] + 0.5 * parts["phi"] + 2 * parts["fluxavg"]),
                                         rel=1e-6)
    _, zf = compute_multi_task_loss(preds, tgts, w, ("df",), separate_zf_loss=True)
    ref = jnp.mean((preds["df"][:, :2] - tgts["df"][:, :2]) ** 2) + relative_norm_mse(
        preds["df"][:, 2:], tgts["df"][:, 2:])
    assert float(zf["df"]) == pytest.approx(float(ref), rel=1e-6)


def test_loss_config_matches_upstream_rules():
    sched = {"df": {"type": "linear", "start": 0, "end": 0, "start_fraction": 0, "end_fraction": 1.0},
             "phi": None, "flux": None,
             "fluxavg": {"type": "linear", "start": 1, "end": 1, "start_fraction": 0, "end_fraction": 1.0},
             "phi_int": None, "flux_int": None, "phi_cross": None, "flux_cross": None}
    cfg = LossConfig({"df": 1.0, "phi": 0.1, "flux": 0.0, "fluxavg": 0.0},
                     {"phi_int": 0.0, "flux_int": 0.0, "phi_cross": 0.0, "flux_cross": 0.0}, sched)
    assert cfg.outputs == ("df", "phi", "fluxavg")
    assert cfg.active == ("df", "phi", "fluxavg")
    assert cfg.flux_key == "fluxavg"
    # a schedule replaces the static weight
    assert cfg.weights_at(0.3) == {"df": 0.0, "phi": 0.1, "fluxavg": 1.0}
    assert cfg.integrals == ()


def test_loss_config_rejects_bad_keys():
    with pytest.raises(ValueError, match="unknown loss keys"):
        LossConfig({"df": 1.0, "avgflux": 1.0})
    with pytest.raises(ValueError, match="cross losses"):
        LossConfig({"df": 1.0}, {"phi_cross": 0.5})
    with pytest.raises(ValueError, match="both flux"):
        LossConfig({"df": 1.0, "flux": 1.0, "fluxavg": 1.0})


@pytest.mark.skipif(
    not os.environ.get("NEUGK_CYCLONE_PATH"),
    reason="set NEUGK_CYCLONE_PATH to compare the integral core to torch FluxIntegral",
)
def test_flux_integral_matches_torch():
    import sys
    sys.path.insert(0, "/system/user/publicwork/galletti/git/neural-gyrokinetics-gitlab")
    import torch
    from neugk.physics.integrals import FluxIntegral

    from neugk_jax.dataset import CycloneDataset, NumpyBackend
    from neugk_jax.evaluate.integrals import flux_integral, precompute_geometry

    ds = CycloneDataset(path=os.environ["NEUGK_CYCLONE_PATH"], trajectories="iteration_0",
                        fields_to_load=("df", "phi"), backend=NumpyBackend())
    s = ds[3]
    geom = ds.metadata[0]["geometry"]
    df, phi = np.asarray(s.df, np.float32), np.asarray(s.phi, np.float32)
    gt = precompute_geometry(geom)
    core = jax.jit(flux_integral)
    j_phi, (j_pf, j_ef, _) = core(gt, jnp.asarray(df))
    _, (j_pf2, j_ef2, _) = core(gt, jnp.asarray(df), jnp.asarray(phi))

    integ = FluxIntegral(real_potens=True)
    g_t = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32)[None] for k, v in geom.items()}
    with torch.no_grad():
        t_phi, (t_pf, t_ef, _) = integ(g_t, torch.from_numpy(df)[None])
        _, (t_pf2, t_ef2, _) = integ(g_t, torch.from_numpy(df)[None], torch.from_numpy(phi)[None])
    t_phi = t_phi[0].numpy()
    assert j_phi.shape == t_phi.shape
    assert np.linalg.norm(np.asarray(j_phi) - t_phi) <= 1e-5 * np.linalg.norm(t_phi)
    assert float(j_ef) == pytest.approx(float(t_ef[0]), rel=1e-5)
    assert float(j_ef2) == pytest.approx(float(t_ef2[0]), rel=1e-5)
    assert float(j_pf2) == pytest.approx(float(t_pf2[0]), rel=1e-5)
    # the solved-phi particle flux vanishes analytically; both sides sit at round-off
    assert abs(float(j_pf) - float(t_pf[0])) <= 1e-6 * abs(float(t_ef[0]))
