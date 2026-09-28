"""Parity of the spectral/zonal-flow metrics port against upstream torch.

Compares ``neugk_jax.evaluate.metrics`` (gyaradax-backed) against
``neugk.pinc.eval.metrics`` (torch FluxIntegral) on a few real snapshots.
Runs only when ``NEUGK_CYCLONE_PATH`` points at real data::

    NEUGK_CYCLONE_PATH=/local00/bioinf/galletti/preprocessed_kvikio \
        python -m pytest tests/test_spectral_metrics_parity.py -x -q -s
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEUGK_CYCLONE_PATH"),
    reason="set NEUGK_CYCLONE_PATH to run spectral-metrics parity (needs real data)",
)

# upstream torch repo (parent of this jax port)
TORCH_REPO = "/system/user/publicwork/galletti/git/neural-gyrokinetics-gitlab"
N_STEPS = 4
SEED = 0


@pytest.fixture(scope="module")
def snapshots():
    """(gt, pred, geom, ds) — a few real timesteps + a perturbed 'prediction'."""
    from neugk_jax.dataset import CycloneDataset, NumpyBackend
    ds = CycloneDataset(
        path=os.environ["NEUGK_CYCLONE_PATH"], split="train",
        trajectories="iteration_0",
        fields_to_load=("df",),
        conditions=("itg", "dg", "s_hat", "q"),
        mode="ae", backend=NumpyBackend(),
        separate_zf=False, normalization=None,
    )
    gt = np.stack([np.asarray(ds[i].df, dtype=np.float64) for i in range(N_STEPS)])
    rng = np.random.default_rng(SEED)
    pred = gt + 0.01 * gt.std() * rng.standard_normal(gt.shape)
    geom = {
        k: np.atleast_1d(np.asarray(v, dtype=np.float64))
        for k, v in ds.metadata[0]["geometry"].items()
    }
    return gt, pred, geom, ds.get_ds(0)


def _torch_side(gt, pred, geom, ds_val):
    """Per-snapshot diagnostics + time-averaged metrics via the torch stack."""
    sys.path.insert(0, TORCH_REPO)
    import torch
    from neugk.pinc.eval import metrics as tmetrics
    torch.set_num_threads(8)
    geom_t = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in geom.items()}

    def diags(dfs):
        out = []
        for df in dfs:
            df_t = torch.as_tensor(df).unsqueeze(0)  # (sp=1, 2, vp, mu, s, x, y)
            out.append(tmetrics.spectral_diagnostics(df_t, geom_t, ds=ds_val))
        return out

    d_pred, d_gt = diags(pred), diags(gt)
    tam = tmetrics.time_averaged_spectral_metrics(d_pred, d_gt)
    to_np = lambda ds_: [{k: np.asarray(v.cpu()) for k, v in d.items()} for d in ds_]
    return to_np(d_pred), to_np(d_gt), tam


def _jax_side(gt, pred, geom, ds_val):
    from neugk_jax.evaluate import metrics as jmetrics
    d_pred = jmetrics.spectral_diagnostics(pred, geom, ds_val)
    d_gt = jmetrics.spectral_diagnostics(gt, geom, ds_val)
    return d_pred, d_gt, jmetrics.time_averaged_spectral_metrics(d_pred, d_gt)


def _rel(a, b):
    return abs(a - b) / (abs(b) + 1e-12)


def test_spectral_metrics_parity(snapshots):
    gt, pred, geom, ds_val = snapshots
    assert ds_val is not None
    tp, tg, tam = _torch_side(gt, pred, geom, ds_val)
    jp, jg, jam = _jax_side(gt, pred, geom, ds_val)

    print("\n--- time-averaged spectral metrics (torch vs jax) ---")
    for k in sorted(jam):
        if k in tam:
            print(f"{k:>16}: torch={tam[k]: .8e}  jax={jam[k]: .8e}  rel={_rel(jam[k], tam[k]):.2e}")

    # jax replicates torch's zonal-correction quirk (kx index 0 special-cased on the ky=0 column, see integrals._torch_zonal_quirk)
    for k in ("zfphi_rl2", "zfflow_rl2", "zfshear_rl2", "zf_energy_err"):
        assert _rel(jam[k], tam[k]) < 1e-4, (
            f"{k}: torch={tam[k]} jax={jam[k]} rel={_rel(jam[k], tam[k]):.3e}"
        )

    # kyspec is |phi|^2-based: scale-free metrics must agree tightly
    for k in ("kyspec_pc", "kyspec_sc", "kyspec_rl2", "kyspec_rl1", "kyspec_wd"):
        assert _rel(jam[k], tam[k]) < 1e-4, (
            f"{k}: torch={tam[k]} jax={jam[k]} rel={_rel(jam[k], tam[k]):.3e}"
        )

    # torch pev_fluxes double-counts ints and ships the ny-parseval factor; gyaradax corrects both, and for this dataset the two torch bugs cancel to a ratio of 1
    q_t = np.stack([d["qspec"] for d in tg], 0).mean(0)
    q_j = np.stack([d["qspec"] for d in jg], 0).mean(0)
    mask = np.abs(q_j) > 1e-12 * np.abs(q_j).max()
    ratio = q_t[mask] / q_j[mask]
    ratio_mean, ratio_std = float(ratio.mean()), float(ratio.std())
    print(f"\nqspec scale ratio torch/jax: mean={ratio_mean:.10f} std={ratio_std:.3e} "
          f"min={ratio.min():.10f} max={ratio.max():.10f} (over {mask.sum()} of {q_j.size} modes)")
    assert abs(ratio_mean - 1.0) < 1e-4, (
        f"qspec torch/jax scale drifted from 1: mean={ratio_mean}"
    )
    assert ratio_std / abs(ratio_mean) < 1e-4, (
        f"qspec torch/jax ratio scatter above bessel-noise level: "
        f"mean={ratio_mean} std={ratio_std}"
    )
    for k in ("qspec_pc", "qspec_sc", "qspec_rl2", "qspec_rl1", "qspec_wd"):
        assert _rel(jam[k], tam[k]) < 1e-4, (
            f"{k}: torch={tam[k]} jax={jam[k]} rel={_rel(jam[k], tam[k]):.3e}"
        )

    # per-snapshot zonal profiles agree pointwise (phi-solve parity)
    for d_t, d_j in zip(tg, jg):
        for k in ("zfphi", "zfflow", "zfshear"):
            num = np.linalg.norm(d_t[k] - d_j[k])
            den = np.linalg.norm(d_t[k]) + 1e-30
            assert num / den < 1e-4, f"{k} profile rel-L2 {num / den:.3e}"
