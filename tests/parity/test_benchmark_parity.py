"""Compression benchmark metrics and codecs against the torch evaluation (``neugk.pinc.eval``)."""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import RES, make_geometry

from neugk_jax.dataset.backend import complete_geometry
from neugk_jax.evaluate.integrals import precompute_geometry
from neugk_jax.pinc.benchmark import metrics as M
from neugk_jax.pinc.benchmark.runner import host_diag


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


@pytest.fixture(scope="module")
def snapshot():
    rng = np.random.default_rng(0)
    gt = rng.standard_normal((2, *RES)).astype(np.float32)
    pred = gt + 0.1 * rng.standard_normal(gt.shape).astype(np.float32)
    return pred, gt, complete_geometry(make_geometry(RES)), 0.0625


def test_snapshot_metrics_and_spectra_match_torch(snapshot):
    import torch
    from neugk.physics.diagnostics import velocity_moment_errors
    from neugk.pinc.eval import metrics as T

    pred, gt, geom, ds = snapshot
    tg = {k: torch.as_tensor(np.asarray(v)) for k, v in geom.items()}
    tp, tt = torch.from_numpy(pred), torch.from_numpy(gt)
    p_phi, p_ef = T.integrate(tp, tg)
    g_phi, g_ef = T.integrate(tt, tg)
    ref = {**T.ml_eval(tp, tt, p_phi, g_phi, p_ef, g_ef), **velocity_moment_errors(tp, tt, tg)}
    ref_diag = T.spectral_diagnostics(tp, tg, ds)

    with jax.default_matmul_precision("highest"):
        gt_t = precompute_geometry(geom)
        p = M.snapshot_fields(gt_t, jnp.asarray(pred), ds)
        g = M.snapshot_fields(gt_t, jnp.asarray(gt), ds)
        out = {
            k: float(v)
            for k, v in M.snapshot_metrics(jnp.asarray(pred), jnp.asarray(gt), p, g).items()
        }
    out.update(M.velocity_moment_errors(pred, gt, M.moment_weights(geom)))
    assert set(out) == set(ref)
    for k in ref:
        assert _rel(out[k], ref[k]) < 1e-4, k
    diag = host_diag(p, geom)
    assert list(diag) == list(ref_diag)
    for k in ref_diag:
        assert _rel(diag[k], ref_diag[k].numpy()) < 1e-5, k


def test_time_averaged_spectral_metrics_match_torch():
    import torch
    from neugk.pinc.eval.metrics import time_averaged_spectral_metrics

    rng = np.random.default_rng(1)

    def diags(n):
        return [
            {
                "kyspec": rng.lognormal(size=12).astype(np.float32),
                "qspec": rng.lognormal(size=12).astype(np.float32),
                **{k: rng.standard_normal(9) for k in ("zfphi", "zfflow", "zfshear")},
            }
            for _ in range(n)
        ]

    pred, gt = diags(4), diags(4)
    to_t = lambda ds: [{k: torch.from_numpy(v) for k, v in d.items()} for d in ds]  # noqa: E731
    ref = time_averaged_spectral_metrics(to_t(pred), to_t(gt))
    out = M.time_averaged_spectral_metrics(pred, gt)
    assert set(out) == set(ref)
    for k in ref:
        assert _rel(out[k], ref[k]) < 1e-5, k


def test_optical_flow_epe_matches_torch():
    import torch
    from neugk.pinc.eval.metrics import temporal_epe

    rng = np.random.default_rng(2)
    gt = [rng.standard_normal((2, *RES)).astype(np.float32) for _ in range(3)]
    pred = [g + 0.2 * rng.standard_normal(g.shape).astype(np.float32) for g in gt]
    ref = temporal_epe([torch.from_numpy(a) for a in gt], [torch.from_numpy(a) for a in pred])
    assert _rel(M.temporal_epe(gt, pred), ref) < 1e-5


@pytest.mark.parametrize("codec", ["zfp", "sz3"])
def test_codec_search_matches_torch(codec):
    if codec == "sz3" and importlib.util.find_spec("pysz") is None:
        pytest.skip("pysz not installed")
    import torch
    from neugk.pinc.eval.reconstructors import _encode_at_cr

    from neugk_jax.pinc.benchmark.codecs import SHAPE, encode_at_cr

    rng = np.random.default_rng(3)
    df = (rng.standard_normal(SHAPE) * np.exp(rng.standard_normal(SHAPE[:3] + (1, 1, 1)))).astype(
        np.float32
    )
    recon, nbytes, knob = encode_at_cr(codec, df, 300.0)
    t_recon, t_nbytes, t_knob = _encode_at_cr(codec, torch.from_numpy(df), 300.0)
    assert (nbytes, knob) == (t_nbytes, t_knob)
    np.testing.assert_array_equal(recon, t_recon.numpy())
