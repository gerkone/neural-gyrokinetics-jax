"""Byte-level parity of the JAX CycloneDataset (``ae`` and ``next`` modes) against the torch loaders."""

from __future__ import annotations

import os

import numpy as np
import pytest

from neugk_jax.dataset import CycloneDataset, NumpyBackend

pytest.importorskip("neugk")


def _stats(flux_mean=1.0, flux_std=2.0, avg_mean=-1.0, avg_std=4.0):
    def one(m, s):
        return {"full": {"mean": np.asarray(m, np.float32), "std": np.asarray(s, np.float32),
                         "min": np.asarray(m - s, np.float32), "max": np.asarray(m + s, np.float32)}}
    return {"df": one(0.5, 2.0), "phi": one(-0.5, 3.0),
            "flux": one(flux_mean, flux_std), "fluxavg": one(avg_mean, avg_std)}


_NORM = {k: {"type": "zscore"} for k in ("df", "phi", "flux", "fluxavg")}


@pytest.mark.skipif(
    not os.environ.get("NEUGK_CYCLONE_PATH"),
    reason="set NEUGK_CYCLONE_PATH to run torch-loader parity (needs real data)",
)
@pytest.mark.parametrize("separate_zf", [False, True])
def test_byte_equal_to_torch_loader(separate_zf):
    """Compare ``df`` bytes against the upstream torch ``CycloneAEDataset``.

    Parameterised over ``separate_zf`` so we catch divergences in either
    the raw read path or the channel-axis pre-processing.

    Required env vars::
        NEUGK_CYCLONE_PATH=/local00/bioinf/galletti/preprocessed_kvikio
        NEUGK_CYCLONE_TRAJS=iteration_{0-1}
    """
    from neugk.dataset.backend import KvikIOBackend as TorchKvikIO
    from neugk.dataset.cyclone_diff import CycloneAEDataset as TorchAE

    path = os.environ["NEUGK_CYCLONE_PATH"]
    trajectories = os.environ.get("NEUGK_CYCLONE_TRAJS", "iteration_{0-1}")

    t_ds = TorchAE(
        backend=TorchKvikIO(rank=0, use_kvikio=False),
        path=path, split="train",
        trajectories=trajectories,
        partial_holdouts={},
        fields_to_load=["df"],
        probe_targets=[],
        # upstream get_dataset sorts before constructing; the jax dataset sorts internally
        conditions=sorted(["itg", "dg", "s_hat", "q"]),
        normalization=None,
        offset=0,
        bundle_seq_length=1,
        spatial_ifft=True,
        real_potens=True,
        separate_zf=separate_zf,
    )
    j_ds = CycloneDataset(
        path=path, split="train",
        trajectories=trajectories,
        fields_to_load=("df",),
        conditions=("itg", "dg", "s_hat", "q"),
        mode="ae",
        backend=NumpyBackend(),
        separate_zf=separate_zf,
    )
    assert len(t_ds) == len(j_ds), f"lengths differ: torch={len(t_ds)} jax={len(j_ds)}"
    # match by (file basename, timestep) — ordering differs between stacks (torch: set, jax: sorted)
    t_lookup = {
        (os.path.basename(t_ds.files[fid]), int(t_idx)): flat
        for flat, (fid, t_idx) in t_ds.flat_index_to_file_and_tstep.items()
    }
    for i in (0, 1, len(j_ds) // 2, len(j_ds) - 1):
        j_fid, j_t = j_ds.flat_index_to_file_and_tstep[i]
        key = (os.path.basename(j_ds.files[j_fid]), int(j_t))
        ti = t_lookup[key]
        ts = t_ds[ti]
        js = j_ds[i]
        t_df = np.asarray(ts.df)
        j_df = np.asarray(js.df)
        assert t_df.shape == j_df.shape, (
            f"shape mismatch at jax-idx {i} (separate_zf={separate_zf}): "
            f"torch={t_df.shape} jax={j_df.shape}"
        )
        # post-separate_zf path subtracts a mean ⇒ tiny rounding allowed
        atol = 1e-6 if separate_zf else 0.0
        diff = np.abs(t_df - j_df).max()
        assert diff <= atol, (
            f"df differs at jax-idx {i} key={key} "
            f"(separate_zf={separate_zf}), max|diff|={diff}"
        )
        t_cond = np.asarray(ts.conditioning) if ts.conditioning is not None else None
        j_cond = np.asarray(js.conditioning)
        if t_cond is not None:
            assert np.array_equal(t_cond, j_cond), f"conditioning differs at {key}"


@pytest.mark.skipif(
    not os.environ.get("NEUGK_CYCLONE_PATH"),
    reason="set NEUGK_CYCLONE_PATH to run torch-loader parity (needs real data)",
)
@pytest.mark.parametrize("separate_zf", [False, True])
def test_next_step_matches_torch_loader(separate_zf):
    """Byte-level parity of ``mode="next"`` against the upstream torch ``CycloneDataset``."""
    from neugk.dataset.backend import KvikIOBackend as TorchKvikIO
    from neugk.dataset.cyclone import CycloneDataset as TorchCyclone

    path = os.environ["NEUGK_CYCLONE_PATH"]
    trajectories = os.environ.get("NEUGK_CYCLONE_TRAJS", "iteration_{0-1}")
    stats = _stats(95.0, 48.0, 92.0, 45.0)
    stats["df"]["full"].update(mean=np.asarray(1e-4, np.float32), std=np.asarray(5e-3, np.float32))
    stats["phi"]["full"].update(mean=np.asarray(-0.1, np.float32), std=np.asarray(70.0, np.float32))
    common = dict(path=path, split="train", trajectories=trajectories, normalization=_NORM,
                  normalization_scope="dataset", offset=80, bundle_seq_length=1,
                  spatial_ifft=True, real_potens=True, separate_zf=separate_zf)
    t_ds = TorchCyclone(
        backend=TorchKvikIO(rank=0, use_kvikio=False), active_keys=["re", "im"],
        fields_to_load=["df", "phi", "fluxavg"], probe_targets=[], partial_holdouts={},
        normalization_stats=stats, **common,
    )
    j_ds = CycloneDataset(fields_to_load=("df", "phi"), conditions=("itg", "dg", "s_hat", "q"),
                          mode="next", backend=NumpyBackend(), normalization_stats=stats, **common)
    assert len(t_ds) == len(j_ds)
    t_lookup = {
        (os.path.basename(t_ds.files[fid]), int(t)): int(flat)
        for flat, (fid, t) in t_ds.flat_index_to_file_and_tstep.items()
    }
    atol = 1e-5 if separate_zf else 1e-6
    for i in (0, 1, len(j_ds) // 2, len(j_ds) - 1):
        fid, t = j_ds.flat_index_to_file_and_tstep[i]
        ts = t_ds[t_lookup[(os.path.basename(j_ds.files[fid]), int(t))]]
        js = j_ds[i]
        for a, b in ((ts.df, js.df), (ts.y_df, js.y_df), (ts.phi, js.phi), (ts.y_phi, js.y_phi)):
            a, b = np.squeeze(np.asarray(a)), np.squeeze(np.asarray(b))
            assert a.shape == b.shape
            assert np.abs(a - b).max() <= atol * max(1.0, np.abs(a).max())
        assert float(ts.y_flux) == pytest.approx(float(js.y_flux), rel=1e-6)
        assert float(ts.y_fluxavg) == pytest.approx(float(js.y_fluxavg), rel=1e-6)
        assert float(ts.timestep) == pytest.approx(float(js.timestep))
