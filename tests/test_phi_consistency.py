"""Stored potentials equal the field solve of the stored df (real data only)."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("NEUGK_CYCLONE_PATH"),
                                reason="set NEUGK_CYCLONE_PATH (needs real data)")


def test_stored_phi_matches_field_solve():
    from neugk_jax.dataset import CycloneDataset, NumpyBackend
    from neugk_jax.evaluate.integrals import flux_integral, precompute_geometry

    ds = CycloneDataset(path=os.environ["NEUGK_CYCLONE_PATH"], trajectories="iteration_0",
                        fields_to_load=("df", "phi"), backend=NumpyBackend())
    solve = jax.jit(flux_integral)
    gt = precompute_geometry(ds.metadata[0]["geometry"])
    for i in (0, len(ds) // 2, len(ds) - 1):
        s = ds[i]
        phi, _ = solve(gt, jnp.asarray(s.df, jnp.float32))
        a, b = np.asarray(phi).ravel(), np.asarray(s.phi).ravel()
        assert np.linalg.norm(a - b) <= 1e-4 * np.linalg.norm(b)
