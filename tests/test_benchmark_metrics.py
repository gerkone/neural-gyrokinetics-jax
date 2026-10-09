"""Spectral metrics of the compression benchmark: RMSLE and Wasserstein-1 distance."""

import numpy as np
import pytest

from neugk_jax.pinc.benchmark import metrics as M


def _spectral(pred, gt):
    zf = {k: np.ones(4) for k in ("zfphi", "zfflow", "zfshear")}
    return M.time_averaged_spectral_metrics(
        [{"kyspec": pred, "qspec": gt, **zf}], [{"kyspec": gt, "qspec": gt, **zf}]
    )


def test_rmsle_and_w1_measure_shape():
    g = np.array([50.0, 8.0, 4.0, 2.0, 1.0, 0.5])
    same = _spectral(g, g)
    assert same["kyspec_rmsle"] == 0.0 and same["kyspec_w1"] == 0.0
    # a wrong zonal mode is seen over every ky and not over ky > 0
    zonal = g.copy()
    zonal[0] *= 10
    out = _spectral(zonal, g)
    assert out["kyspec_rmsle"] == pytest.approx(np.sqrt(1 / 6)) and out["kyspec_w1"] > 0
    assert out["kyspec_rmsle_nz"] == 0.0 and out["kyspec_w1_nz"] == 0.0
    # a uniform factor 10 on ky > 0 is one decade everywhere and the same shape
    scaled = np.concatenate([g[:1], 10 * g[1:]])
    out = _spectral(scaled, g)
    assert out["kyspec_rmsle_nz"] == pytest.approx(1.0)
    assert out["kyspec_w1_nz"] == pytest.approx(0.0, abs=1e-12)
    # all the mass moved by one bin
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert M.wasserstein1(a, b) == pytest.approx(1 / 3)


def test_rmsle_floor_clips_the_tail():
    g = np.array([1.0, 1e-3, 1e-4])
    assert M.rmsle(np.array([1.0, 1e-2, 1e-2]), g) == pytest.approx(0.0)
