"""Species-axis AE on an adiabatic + kinetic dataset mix: windows, stems, stats, augmentation, weights."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import RES, make_geometry, make_traj, tiny_ae_model_cfg
from omegaconf import OmegaConf

from neugk_jax.dataset.cyclone import avg_flux
from neugk_jax.dataset.factory import build_splits, save_run_stats
from neugk_jax.losses import part_weight
from neugk_jax.models.build import build_ae
from neugk_jax.models.swin import _build_shift_mask, _effective_window, _shift_size

NORM = {"df": {"type": "zscore", "agg_axes": [2, 4, 5, 6]}}
CONDITIONS = ["dg", "etg", "itg", "kinetic", "q", "s_hat", "temp_ratio"]


def make_kinetic_traj(root: Path, name: str, *, n_t: int, scale: float = 1.0) -> None:
    """Two-species trajectory in the ``(re/im, species, vpar, mu, s, x, y)`` layout."""
    traj = Path(root) / f"{name}_ifft_realpotens"
    (traj / "data").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    for t in range(n_t):
        (scale * rng.standard_normal((2, 2, *RES))).astype(np.float32).tofile(traj / "data" / f"timestep_{t:05d}.bin")
    meta = {
        "timesteps": np.arange(n_t, dtype=np.float64),
        "flux": rng.standard_normal((n_t, 6)),
        "n_species": 2,
        "ion_species_index": 0,
        "df_shape": (2, 2, *RES),
        "resolution": np.array(RES),
        "ion_temp_grad": np.array([4.0]),
        "electron_temp_grad": np.array([3.0]),
        "density_grad": np.array([1.0]),
        "temp_ratio": np.array([0.8]),
        "s_hat": np.array([0.8]),
        "q": np.array([1.4]),
        "ds": np.float64(0.0625),
        "geometry": make_geometry(RES),
    }
    with open(traj / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)


@pytest.fixture
def mix_root(tmp_path):
    adiab, kin = tmp_path / "adiab", tmp_path / "kin"
    for i in range(3):
        make_traj(adiab, f"iteration_{i}", n_t=3)
    for i in range(3):
        make_kinetic_traj(kin, f"iteration_{500 + i}", n_t=3)
    for i in range(2):
        make_kinetic_traj(kin, f"iteration_{600 + i}", n_t=1, scale=1e-6)
    return tmp_path


def mix_cfg(root: Path, *, stable: bool = False, augment=None):
    parts = {
        # adiabatic listed first on purpose: the stem with the most species must still be primary
        "adiabatic": {
            "path": str(root / "adiab"),
            "training_trajectories": "iteration_{0-1}",
            "validation_trajectories": "iteration_2",
        },
        "kinetic": {
            "path": str(root / "kin"),
            "training_trajectories": "iteration_{500-501}",
            "validation_trajectories": "iteration_502",
        },
    }
    mixing = {"adiabatic": 0.5, "kinetic": 0.5}
    if stable:
        parts["kinetic_stable"] = {
            "path": str(root / "kin"),
            "stem": "kinetic",
            "stats_from": "kinetic",
            "augment": augment or {},
            "training_trajectories": "iteration_600",
            "validation_trajectories": "iteration_601",
        }
        mixing = {"adiabatic": 0.4, "kinetic": 0.4, "kinetic_stable": {"share": 0.2, "loss_weight": 0.25}}
    return OmegaConf.create(
        {
            "dataset": {
                "backend": "numpy",
                "species_axis": True,
                "separate_zf": False,
                "offset": 0,
                "input_fields": ["df"],
                "conditions": CONDITIONS,
                "normalization": NORM,
                "mixing": mixing,
                "parts": parts,
            },
            "model": {
                **tiny_ae_model_cfg(decoder_conditioning=CONDITIONS),
                "stems": {"adiabatic": {"patch_size": [2, 0, 2, 4, 2]}, "kinetic": {"patch_size": [2, 0, 2, 4, 2]}},
            },
        }
    )


def test_species_axis_window_is_never_shifted(mix_root):
    grid, cfg_window = (2, 4, 4), (math.inf, 2, 2)
    eff = _effective_window(grid, cfg_window)
    shift = _shift_size(grid, cfg_window, eff, True)
    assert eff[0] == 2 and shift == (0, 1, 1)
    mask = np.asarray(_build_shift_mask(grid, eff, shift))
    # tokens differing only in species share a region: never masked from each other
    tokens = np.arange(int(np.prod(eff))).reshape(eff)
    for w in range(mask.shape[0]):
        for i in tokens[0].ravel():
            assert mask[w, i, i + int(np.prod(eff[1:]))] == 0.0
    # and no block of the kinetic autoencoder shifts its species axis
    cfg = mix_cfg(mix_root)
    train, _ = build_splits(cfg.dataset)
    model = build_ae(cfg, train, key=jr.PRNGKey(0))
    shifts = [b.shift_size for b in jax.tree_util.tree_leaves(model, is_leaf=lambda x: hasattr(x, "shift_size"))
              if hasattr(b, "shift_size")]
    assert shifts and all(sh[0] == 0 for sh in shifts)


def test_mix_stems_order_and_shapes(mix_root):
    cfg = mix_cfg(mix_root, stable=True)
    train, val = build_splits(cfg.dataset)
    assert {n: p.stem for n, p in train.parts.items()} == {
        "adiabatic": "adiabatic", "kinetic": "kinetic", "kinetic_stable": "kinetic"
    }
    model = build_ae(cfg, train, key=jr.PRNGKey(0))
    assert model.primary_stem == "kinetic"
    for name in ("adiabatic", "kinetic", "kinetic_stable"):
        sample = train.parts[name][0]
        out = model(jnp.asarray(sample.df), jnp.asarray(sample.conditioning)[jnp.asarray([0, 1, 2, 3, 4, 5, 6])])
        assert out["df"].shape == sample.df.shape, name


def test_stats_from_data_shared_and_resumed(mix_root, tmp_path):
    cfg = mix_cfg(mix_root, stable=True)
    train, val = build_splits(cfg.dataset)
    kin, stable = train.parts["kinetic"].stats["df"]["full"], train.parts["kinetic_stable"].stats["df"]["full"]
    np.testing.assert_array_equal(kin["mean"], stable["mean"])
    np.testing.assert_array_equal(kin["std"], val.parts["kinetic"].stats["df"]["full"]["std"])
    # the statistics are those of the unnormalized training frames, not metadata moments
    frames = np.stack([np.fromfile(mix_root / "kin" / f"iteration_{i}_ifft_realpotens" / "data" / f"timestep_{t:05d}.bin", np.float32)
                       .reshape(2, 2, *RES) for i in (500, 501) for t in range(2)])
    np.testing.assert_allclose(kin["mean"].reshape(-1), frames.mean(axis=(0, 3, 5, 6, 7)).reshape(-1), rtol=1e-4, atol=1e-5)
    # a resumed run reads its own copy first
    run_dir = tmp_path / "run_stats"
    save_run_stats(train, str(run_dir))
    assert sorted(p.name for p in run_dir.iterdir()) == ["adiabatic.pkl", "kinetic.pkl", "kinetic_stable.pkl"]
    with open(run_dir / "kinetic.pkl", "rb") as f:
        saved = pickle.load(f)
    saved["df"]["full"]["mean"] = saved["df"]["full"]["mean"] + 1.0
    with open(run_dir / "kinetic.pkl", "wb") as f:
        pickle.dump(saved, f)
    resumed, _ = build_splits(cfg.dataset, run_stats_dir=str(run_dir))
    np.testing.assert_allclose(resumed.parts["kinetic"].stats["df"]["full"]["mean"], kin["mean"] + 1.0)


def test_missing_stats_path_is_an_error(mix_root):
    from neugk_jax.dataset.backend import make_backend
    from neugk_jax.dataset.cyclone import CycloneDataset

    with pytest.raises(FileNotFoundError):
        CycloneDataset(
            path=str(mix_root / "kin"), trajectories="iteration_500", species_axis=True,
            normalization=NORM, normalization_stats=str(mix_root / "nope.pkl"),
            backend=make_backend(OmegaConf.create({"backend": "numpy"})),
        )


def test_augmentation_off_unless_set(mix_root):
    off, _ = build_splits(mix_cfg(mix_root, stable=True).dataset)
    a, b = off.parts["kinetic_stable"][0].df, off.parts["kinetic_stable"][0].df
    np.testing.assert_array_equal(a, b)
    on, on_val = build_splits(mix_cfg(mix_root, stable=True, augment={"roll_y": True, "amplitude": [2.0, 4.0]}).dataset)
    ds = on.parts["kinetic_stable"]
    assert on_val.parts["kinetic_stable"].augment == {}
    raw = ds.transform(np.fromfile(mix_root / "kin" / "iteration_600_ifft_realpotens" / "data" / "timestep_00000.bin",
                                   np.float32).reshape(1, 2, 2, *RES), np.asarray([0]), normalize=False)
    ref = off.parts["kinetic_stable"].transform(
        np.fromfile(mix_root / "kin" / "iteration_600_ifft_realpotens" / "data" / "timestep_00000.bin", np.float32)
        .reshape(1, 2, 2, *RES), np.asarray([0]), normalize=False)
    ratio = np.linalg.norm(raw) / np.linalg.norm(ref)
    assert 2.0 <= ratio <= 4.0
    # a roll along y: the same values in every (.., y) line, scaled
    np.testing.assert_allclose(np.sort(raw, axis=-1) / ratio, np.sort(ref, axis=-1), rtol=1e-5, atol=1e-12)


def test_loss_weight_served_and_applied(mix_root):
    train, _ = build_splits(mix_cfg(mix_root, stable=True).dataset)
    assert float(train.parts["kinetic_stable"][0].loss_weight) == 0.25
    assert float(train.parts["kinetic"][0].loss_weight) == 1.0
    assert float(part_weight({})) == 1.0
    assert float(part_weight({"loss_weight": jnp.full(4, 0.25)})) == 0.25


def test_single_snapshot_runs_are_served(mix_root):
    train, _ = build_splits(mix_cfg(mix_root, stable=True).dataset)
    assert len(train.parts["kinetic_stable"]) == 1
    assert len(train.parts["kinetic"]) == 4


def test_avg_flux_single_snapshot():
    np.testing.assert_array_equal(avg_flux(np.array([[1.5, 2.5]])), [1.5, 2.5])
    assert avg_flux(np.array([9.0, 1.0, 3.0])) == 2.0


def test_species_never_merged_or_mixed(mix_root):
    from neugk_jax.models.patching import PatchExpand, PatchMerge

    cfg = mix_cfg(mix_root)
    train, _ = build_splits(cfg.dataset)
    model = build_ae(cfg, train, key=jr.PRNGKey(0))
    is_layer = lambda x: isinstance(x, (PatchMerge, PatchExpand))
    layers = [m for m in jax.tree_util.tree_leaves(model, is_leaf=is_layer) if is_layer(m)]
    merges = [m for m in layers if isinstance(m, PatchMerge)]
    assert merges and all(m.patch_size[0] == 1 for m in merges)
    assert all(m.expand_by[0] == 1 for m in layers if isinstance(m, PatchExpand))
    # every stem keeps its species extent at every stage
    assert all(g[0] == 2 for g in model.backbone.grid_sizes)
    assert all(g[0] == 1 for g in model.stem_backbones["adiabatic"].grid_sizes)
    # three species: merged along space only, and a species' merged tokens see only its own inputs
    merge = PatchMerge(4, (3, 8, 8), key=jr.PRNGKey(1), merge_mask=[False, True, True])
    assert merge.patch_size == (1, 2, 2) and merge.target_grid_size == (3, 4, 4)
    x = jr.normal(jr.PRNGKey(2), (3, 8, 8, 4))
    y0, y1 = merge(x), merge(x.at[1].add(jr.normal(jr.PRNGKey(3), x.shape[1:])))
    np.testing.assert_array_equal(y0[0], y1[0])
    np.testing.assert_array_equal(y0[2], y1[2])
    assert not np.allclose(y0[1], y1[1])
