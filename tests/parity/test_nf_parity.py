"""The JAX neural field against the torch ``MLPNF`` through the state_dict mapping."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch


def test_mlpnf_forward_and_state_dict_match_torch():
    from neugk.pinc.neural_fields.models.mlp import MLPNF as TorchNF

    from neugk_jax.pinc.neural_field import MLPNF, from_torch_state, grid_coords, torch_state

    grid = (6, 4, 5, 9, 4)
    tm = TorchNF(
        5,
        2,
        grid_size=grid,
        n_layers=5,
        dim=64,
        act_fn=torch.nn.SiLU,
        skips=True,
        embed_type="discrete",
    )
    jm = from_torch_state(
        MLPNF(grid, key=jr.PRNGKey(0)), {k: v.numpy() for k, v in tm.state_dict().items()}
    )
    coords = np.asarray(grid_coords(grid, jnp.arange(int(np.prod(grid)))))
    with torch.no_grad():
        ref = tm(torch.from_numpy(coords).long()).numpy()
    np.testing.assert_allclose(np.asarray(jm(jnp.asarray(coords))), ref, rtol=1e-5, atol=1e-5)
    tm2 = TorchNF(
        5,
        2,
        grid_size=grid,
        n_layers=5,
        dim=64,
        act_fn=torch.nn.SiLU,
        skips=True,
        embed_type="discrete",
    )
    tm2.load_state_dict({k: torch.from_numpy(v) for k, v in torch_state(jm).items()}, strict=True)
    with torch.no_grad():
        np.testing.assert_allclose(tm2(torch.from_numpy(coords).long()).numpy(), ref, rtol=1e-6)
