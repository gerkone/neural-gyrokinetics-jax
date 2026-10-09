"""Neural fields: model, full-grid decoding and the pool runner on synthetic trajectories."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from helpers import RES, make_traj
from omegaconf import OmegaConf


def test_sample_field_matches_pointwise_and_differentiates():
    from neugk_jax.pinc.neural_field import MLPNF, grid_coords, n_params, sample_field

    m = MLPNF(RES, key=jr.PRNGKey(0), dim=16, n_layers=3)
    width = round(16 / len(RES))
    assert n_params(m) == sum(RES) * width + (5 * width + 1) * 16 + 17 * 16 + 17 * 2
    field = sample_field(m, RES)
    coords = grid_coords(RES, jnp.arange(int(np.prod(RES))))
    np.testing.assert_allclose(field, m(coords).T.reshape(2, *RES), rtol=1e-6, atol=1e-6)
    grads = jax.grad(lambda mm: jnp.sum(sample_field(mm, RES) ** 2))(m)
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(grads))


def test_nf_runner_trains_pools_and_resumes(tmp_path):
    from neugk_jax.pinc.nf_runner import NFRunner

    rng = np.random.default_rng(0)
    for i in range(2):
        make_traj(
            tmp_path,
            f"iteration_{i}",
            n_t=4,
            kyspec=rng.uniform(0.0, 3.0, (4, RES[-1])),
            fluxspec=rng.uniform(0.0, 1.0, (4, RES[-1])),
        )
    weights = {k: 1.0 for k in ("df", "phi_int", "flux_int", "kyspec", "qspec")}
    cfg = OmegaConf.create(
        {
            "workflow": "nf",
            "output_path": str(tmp_path / "out"),
            "model": {
                "name": "mlp",
                "dim": 16,
                "n_layers": 3,
                "skips": True,
                "embed_type": "discrete",
                "act_fn": "silu",
            },
            "dataset": {
                "path": str(tmp_path),
                "backend": "numpy",
                "training_trajectories": [],
                "validation_trajectories": ["iteration_0", "iteration_1"],
                "timesteps": [1, 3, 9],
                "spectral_offset": 1,
            },
            "training": {
                "pool_size": 2,
                "density": {
                    "epochs": 2,
                    "lr": 1e-2,
                    "min_lr": 1e-12,
                    "weight_decay": 1e-8,
                    "batch_size": 64,
                    "subsample": [0.5, 1.0],
                    "scan_steps": 4,
                },
                "pinc": {
                    "epochs": 3,
                    "lr": 1e-3,
                    "min_lr": 1e-8,
                    "warmup": 1,
                    "weight_decay": 1e-12,
                    "loss_weights": weights,
                },
            },
            "logging": {"mode": "disabled"},
        }
    )
    r = NFRunner(cfg, output_path=cfg.output_path)
    # timestep 9 is past the 4 stored frames
    assert len(r.jobs) == 4
    r()
    out = tmp_path / "out"
    for prefix in ("best_", "int_", "best_int_"):
        assert len(list(out.glob(f"{prefix}mlp_iteration_*_t*_x*.eqx"))) == 4, prefix
    assert len(NFRunner(cfg, output_path=cfg.output_path).jobs) == 0
    # without the pinc fields, a restart warm-starts them from the saved density fields
    density = {f: f.read_bytes() for f in out.glob("best_mlp_*.eqx")}
    for f in [*out.glob("int_*.eqx"), *out.glob("best_int_*.eqx")]:
        f.unlink()
    r = NFRunner(cfg, output_path=cfg.output_path)
    assert len(r.jobs) == 4 and all(r.has_density(*j) for j in r.jobs)
    r()
    assert len(list(out.glob("best_int_mlp_*.eqx"))) == 4
    assert all(f.read_bytes() == b for f, b in density.items())
