"""Representation interpolation against the torch ``scripts/pinc/run_interp.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

GRID = (6, 4, 5, 9, 4)


def torch_run_interp(root: Path):
    spec = importlib.util.spec_from_file_location("run_interp", root / "scripts/pinc/run_interp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def torch_nf(seed: int):
    from neugk.pinc.neural_fields.models.mlp import MLPNF as TorchNF

    torch.manual_seed(seed)
    kw = dict(n_layers=5, dim=64, act_fn=torch.nn.SiLU, skips=True, embed_type="discrete")
    return TorchNF(5, 2, grid_size=GRID, **kw)


def test_scores_and_weight_midpoint_match_torch(torch_repo_root):
    from eval_interp import df_scores, mean_weights

    from neugk_jax.pinc.neural_field import MLPNF, from_torch_state, grid_coords, sample_field

    ref = torch_run_interp(torch_repo_root)
    rng = np.random.default_rng(0)
    gt = rng.standard_normal((2, *GRID)).astype(np.float32)
    pred = gt + 0.1 * rng.standard_normal(gt.shape).astype(np.float32)
    expected = ref.metrics(torch.from_numpy(pred), torch.from_numpy(gt))
    out = df_scores(pred, jnp.asarray(gt))
    for k in ("psnr", "l1"):
        np.testing.assert_allclose(out[k], expected[k], rtol=1e-5)

    a, b = torch_nf(0), torch_nf(1)
    sa, sb = ({k: v.clone() for k, v in m.state_dict().items()} for m in (a, b))
    a.load_state_dict({k: 0.5 * (sa[k] + sb[k]) for k in sa})
    coords = np.asarray(grid_coords(GRID, jnp.arange(int(np.prod(GRID)))))
    with torch.no_grad():
        expected = a(torch.from_numpy(coords).long()).numpy().T.reshape(2, *GRID)

    def jax_nf(state):
        model = MLPNF(GRID, key=jr.PRNGKey(0))
        return from_torch_state(model, {k: v.numpy() for k, v in state.items()})

    mid = mean_weights(jax_nf(sa), jax_nf(sb))
    np.testing.assert_allclose(sample_field(mid, GRID), expected, rtol=1e-5, atol=1e-5)
