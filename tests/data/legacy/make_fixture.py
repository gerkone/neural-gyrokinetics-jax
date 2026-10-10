"""Writes the legacy checkpoint fixtures: small models built and saved with the pre-restructure code.

Run with the code of commit 9e5081b on ``PYTHONPATH`` (before the attention / token / patching
restructure). Every fixture is ``<name>.eqx`` (a checkpoint as ``save_checkpoint`` writes it) and
``<name>.npz`` (a fixed input and the model output on it).
Usage: make_fixture.py <out_dir>
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.models.gk_unet import SwinNDUnet
from neugk_jax.pinc import Swin5DAE
from neugk_jax.training.checkpoint import CheckpointState, save_checkpoint


def ae():
    model = Swin5DAE(
        space=5,
        decouple_mu=True,
        dim=16,
        base_resolution=(8, 4, 4, 8, 4),
        in_channels=2,
        out_channels=2,
        patch_size=(2, 0, 2, 4, 2),
        window_size=(2, 0, 2, 2, 2),
        depth=1,
        num_heads=2,
        num_layers=1,
        bottleneck_dim=8,
        bottleneck_depth=1,
        bottleneck_num_heads=2,
        merging_depth=2,
        unmerging_depth=1,
        merging_hidden_ratio=2.0,
        unmerging_hidden_ratio=2.0,
        use_rpb=True,
        key=jr.PRNGKey(0),
    )
    x = jr.normal(jr.PRNGKey(1), (2, 8, 4, 4, 8, 4))
    return model, (x,), lambda m, x: m(x)["df"]


def unet():
    model = SwinNDUnet(
        space=3,
        dim=16,
        base_resolution=(8, 6, 8),
        in_channels=3,
        out_channels=3,
        patch_size=(2, 3, 4),
        window_size=(2, 2, 2),
        depth=[1, 1],
        num_heads=[2, 2],
        num_layers=2,
        middle_depth=1,
        middle_num_heads=2,
        n_cond=2,
        cond_mode="dit",
        use_rpb=True,
        qk_norm=True,
        key=jr.PRNGKey(0),
    )
    x, c = jr.normal(jr.PRNGKey(1), (3, 8, 6, 8)), jnp.asarray([0.3, -1.2])

    def fwd(m, x, c):
        cond = m.condition(c)
        z = m.patch_encode(x)
        skips = []
        for blk in m.down_blocks:
            z, s = blk(z, cond)
            skips.append(s)
        z = m.middle_upscale(m.middle(z, cond))
        for blk, s in zip(m.up_blocks, skips[::-1]):
            z = blk(z, s, cond)
        return m.patch_decode(z, cond)

    return model, (x, c), fwd


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with jax.default_matmul_precision("highest"):
        for name, make in (("ae", ae), ("unet", unet)):
            model, inputs, fwd = make()
            y = fwd(model, *inputs)
            save_checkpoint(
                out / f"{name}.eqx",
                CheckpointState(
                    model=model, opt_state=None, epoch=0, loss=0.0, meta={"fixture": name}
                ),
            )
            np.savez(
                out / f"{name}.npz",
                y=np.asarray(y),
                **{f"in{i}": np.asarray(a) for i, a in enumerate(inputs)},
            )
            print(name, y.shape)


if __name__ == "__main__":
    main(*sys.argv[1:])
