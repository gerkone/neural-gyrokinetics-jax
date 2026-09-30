"""The jittable flux-integral core against the torch ``FluxIntegral``."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("neugk")


@pytest.mark.skipif(
    not os.environ.get("NEUGK_CYCLONE_PATH"),
    reason="set NEUGK_CYCLONE_PATH to compare the integral core to torch FluxIntegral",
)
def test_flux_integral_matches_torch():
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
