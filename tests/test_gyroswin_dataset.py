"""CycloneDataset ``mode="next"``: next-step index rule, normalization, torch parity."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from neugk_jax.dataset import CycloneDataset, NumpyBackend

RES = (2, 2, 2, 4, 2)


def _make_traj(root: Path, name: str, *, n_t: int, seed: int):
    traj = root / f"{name}_ifft_realpotens"
    data = traj / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # every df/phi entry of step t equals t + seed/100, so a read identifies (trajectory, step)
    for t in range(n_t):
        np.full((2, *RES), t + seed / 100, np.float32).tofile(data / f"timestep_{t:05d}.bin")
        np.full((RES[3], RES[2], RES[4]), -(t + seed / 100), np.float32).tofile(data / f"poten_{t:05d}.bin")
    meta = {
        "timesteps": np.arange(n_t, dtype=np.float64) * 0.5 + seed,
        "flux": rng.standard_normal(n_t).astype(np.float32),
        "ion_temp_grad": np.array([2.3], np.float32),
        "density_grad": np.array([1.1], np.float32),
        "s_hat": np.array([0.8], np.float32),
        "q": np.array([1.4], np.float32),
        "resolution": np.array(RES),
        "geometry": {"krho": np.ones((1,))},
    }
    with open(traj / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)
    return meta


@pytest.fixture
def two_trajs(tmp_path):
    metas = [_make_traj(tmp_path, f"iteration_{i}", n_t=6 + i, seed=10 * (i + 1)) for i in range(2)]
    return tmp_path, metas


def _stats(flux_mean=1.0, flux_std=2.0, avg_mean=-1.0, avg_std=4.0):
    def one(m, s):
        return {"full": {"mean": np.asarray(m, np.float32), "std": np.asarray(s, np.float32),
                         "min": np.asarray(m - s, np.float32), "max": np.asarray(m + s, np.float32)}}
    return {"df": one(0.5, 2.0), "phi": one(-0.5, 3.0),
            "flux": one(flux_mean, flux_std), "fluxavg": one(avg_mean, avg_std)}


_NORM = {k: {"type": "zscore"} for k in ("df", "phi", "flux", "fluxavg")}


def test_next_step_index_never_crosses_trajectories(two_trajs):
    root, metas = two_trajs
    ds = CycloneDataset(path=str(root), trajectories=["iteration_0", "iteration_1"],
                        fields_to_load=("df", "phi"), mode="next", backend=NumpyBackend(),
                        offset=1)
    # per file: (n_t - offset) - 2 * bundle + 1 samples
    assert len(ds) == (6 - 1 - 1) + (7 - 1 - 1)
    for i in range(len(ds)):
        s = ds[i]
        fid, t = int(s.file_index), int(s.timestep_index)
        tag = (10 * (fid + 1)) / 100
        assert np.allclose(s.df, t + 1 + tag)
        assert np.allclose(s.y_df, t + 2 + tag)
        assert np.allclose(s.y_phi, -(t + 2 + tag))
        assert s.y_flux == pytest.approx(metas[fid]["flux"][t + 2])
        assert s.y_fluxavg == pytest.approx(np.mean(metas[fid]["flux"][1:][-80:]))
        assert s.timestep == pytest.approx(metas[fid]["timesteps"][t + 1])


def test_next_step_tail_offset_keeps_rollout_targets(two_trajs):
    root, _ = two_trajs
    ds = CycloneDataset(path=str(root), trajectories=["iteration_0"], fields_to_load=("df",),
                        mode="next", backend=NumpyBackend(), tail_offset=2, split="val")
    assert len(ds) == 6 - 2 - 1
    last = ds[len(ds) - 1]
    # every rollout step capped by num_ts still has a target on disk
    steps = ds.num_ts(0) - int(last.timestep_index) - 1
    for t in range(steps + 2):
        ds.get_at_time(0, int(last.timestep_index) + t)


def test_next_step_normalization(two_trajs):
    root, metas = two_trajs
    ds = CycloneDataset(path=str(root), trajectories=["iteration_0"], fields_to_load=("df", "phi"),
                        mode="next", backend=NumpyBackend(), normalization=_NORM,
                        normalization_stats=_stats())
    s, raw = ds[2], ds.get_at_time(0, 2, normalized=False)
    assert np.allclose(s.df, (raw.df - 0.5) / 2.0)
    assert np.allclose(s.y_df, (raw.y_df - 0.5) / 2.0)
    assert np.allclose(s.y_phi, (raw.y_phi + 0.5) / 3.0)
    assert s.y_flux == pytest.approx((metas[0]["flux"][3] - 1.0) / 2.0)
    assert s.y_fluxavg == pytest.approx((np.mean(metas[0]["flux"][1:][-80:]) + 1.0) / 4.0)
    assert np.allclose(ds.denormalize(0, fluxavg=s.y_fluxavg), raw.y_fluxavg, atol=1e-6)


def test_timestep_condition(two_trajs):
    root, metas = two_trajs
    ds = CycloneDataset(path=str(root), trajectories=["iteration_0", "iteration_1"],
                        fields_to_load=("df",), mode="next", backend=NumpyBackend(),
                        conditions=("timestep", "itg"))
    assert len(ds.files) == 2
    assert ds.conditions == ["itg", "timestep"]
    s = ds[3]
    assert s.conditioning[1] == pytest.approx(metas[int(s.file_index)]["timesteps"][int(s.timestep_index)])
    assert ds.get_timestep(int(s.file_index), int(s.timestep_index)) == pytest.approx(s.conditioning[1])


def test_scale_shift_batches_per_trajectory_stats(two_trajs):
    root, _ = two_trajs
    stats = _stats()
    stats["flux"] = {0: {"mean": np.ones(1, np.float32), "std": np.full(1, 2.0, np.float32)},
                     1: {"mean": np.zeros(1, np.float32), "std": np.full(1, 3.0, np.float32)}}
    ds = CycloneDataset(path=str(root), trajectories=["iteration_0", "iteration_1"], mode="next",
                        backend=NumpyBackend(), normalization=_NORM, normalization_stats=stats,
                        normalization_scope="trajectory")
    scale, shift = ds.scale_shift([0, 1, 0], "flux", 0)
    assert scale.shape == (3,) and np.allclose(scale, [2.0, 3.0, 2.0])
    assert np.allclose(shift, [1.0, 0.0, 1.0])


@pytest.mark.skipif(
    not os.environ.get("NEUGK_CYCLONE_PATH"),
    reason="set NEUGK_CYCLONE_PATH to run torch-loader parity (needs real data)",
)
@pytest.mark.parametrize("separate_zf", [False, True])
def test_next_step_matches_torch_loader(separate_zf):
    """Byte-level parity of ``mode="next"`` against the upstream torch ``CycloneDataset``."""
    import sys
    sys.path.insert(0, "/system/user/publicwork/galletti/git/neural-gyrokinetics-gitlab")
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
