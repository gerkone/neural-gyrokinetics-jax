"""Spectral metrics of the compression benchmark: log-spectral and Wasserstein-1 distances over ky > 0."""

import numpy as np
import pytest

from neugk_jax.pinc.benchmark import metrics as M


def test_lsd_and_w1_skip_the_zonal_mode_and_measure_shape():
    g = np.array([50.0, 8.0, 4.0, 2.0, 1.0, 0.5])
    same = M.time_averaged_spectral_metrics(
        [{"kyspec": g, "qspec": g, **_zf()}], [{"kyspec": g, "qspec": g, **_zf()}]
    )
    assert same["kyspec_lsd"] == 0.0 and same["kyspec_w1"] == 0.0
    # a wrong zonal mode leaves both untouched while rel-L2 sees it
    zonal = g.copy()
    zonal[0] *= 10
    out = M.time_averaged_spectral_metrics(
        [{"kyspec": zonal, "qspec": g, **_zf()}], [{"kyspec": g, "qspec": g, **_zf()}]
    )
    assert out["kyspec_lsd"] == 0.0 and out["kyspec_w1"] == 0.0 and out["kyspec_rl2"] > 1
    # a uniform factor 10 on ky > 0 is one decade everywhere and the same shape
    scaled = np.concatenate([g[:1], 10 * g[1:]])
    out = M.time_averaged_spectral_metrics(
        [{"kyspec": scaled, "qspec": g, **_zf()}], [{"kyspec": g, "qspec": g, **_zf()}]
    )
    assert out["kyspec_lsd"] == pytest.approx(1.0) and out["kyspec_w1"] == pytest.approx(
        0.0, abs=1e-12
    )
    # all the mass moved by one bin
    a, b = np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0])
    assert M.wasserstein1(a[1:], b[1:]) == pytest.approx(1 / 3)


def test_lsd_floor_clips_the_tail():
    g = np.array([1.0, 1e-5, 1e-6])
    assert M.log_spectral_distance(np.array([1.0, 1e-4, 1e-4]), g) == pytest.approx(0.0)


def _zf():
    return {k: np.ones(4) for k in ("zfphi", "zfflow", "zfshear")}
