"""Per-snapshot neural field: a coordinate MLP over the ``(vpar, mu, s, x, y)`` grid of one df.

``MLPNF`` maps integer grid coordinates to the (re, im) df value through one learned table per
axis (concatenated), residual SiLU blocks and a linear readout, the torch ``MLPNF`` with
``embed_type="discrete"``. ``sample_field`` decodes the whole grid in rematerialized slices so
the backward of a full-field loss keeps only the field, not the activations.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.utils import Linear, gelu, relu, silu

ACTIVATIONS = {"silu": silu, "gelu": gelu, "relu": relu}


def _xavier(linear: Linear, key) -> Linear:
    w = jax.nn.initializers.glorot_uniform()(key, linear.weight.shape)
    linear = eqx.tree_at(lambda m: m.inner.weight, linear, w)
    if linear.use_bias:
        linear = eqx.tree_at(lambda m: m.inner.bias, linear, jnp.zeros_like(linear.bias))
    return linear


class GridEmbed(eqx.Module):
    """One ``N(0, 1)`` table of ``round(dim / ndim)`` features per grid axis, concatenated."""

    tables: list[jax.Array]

    def __init__(self, dim: int, grid_size: Sequence[int], *, key):
        width = round(dim / len(grid_size))
        keys = jr.split(key, len(grid_size))
        self.tables = [jr.normal(k, (int(n), width)) for k, n in zip(keys, grid_size)]

    @property
    def dim(self) -> int:
        return sum(t.shape[1] for t in self.tables)

    def __call__(self, idx: jax.Array) -> jax.Array:
        # one-hot lookup: the backward is a matmul, not a scatter contended on the small tables
        return jnp.concatenate(
            [
                jnp.matmul(
                    jax.nn.one_hot(idx[..., i], t.shape[0], dtype=t.dtype),
                    t,
                    precision=jax.lax.Precision.HIGHEST,
                )
                for i, t in enumerate(self.tables)
            ],
            axis=-1,
        )


class MLPNF(eqx.Module):
    """Coordinate MLP ``(..., ndim)`` integer coordinates -> ``(..., out_dim)``.

    ``n_layers - 1`` hidden linears of width ``dim``; with ``skips`` every square block is
    residual. The activation follows every hidden linear but the last, which is residual
    before it.
    """

    embed: GridEmbed
    layers: list[Linear]
    readout: Linear
    skips: bool = eqx.field(static=True)
    act: Callable = eqx.field(static=True)

    def __init__(
        self,
        grid_size: Sequence[int],
        *,
        key,
        dim: int = 64,
        n_layers: int = 5,
        out_dim: int = 2,
        skips: bool = True,
        act_fn: str = "silu",
    ):
        k_embed, k_out, *k_layers = jr.split(key, n_layers + 1)
        self.embed = GridEmbed(dim, grid_size, key=k_embed)
        dims = [self.embed.dim] + [dim] * (n_layers - 1)
        self.layers = [
            _xavier(Linear(i, o, key=k), k) for (i, o), k in zip(zip(dims, dims[1:]), k_layers)
        ]
        self.readout = _xavier(Linear(dim, out_dim, key=k_out), k_out)
        self.skips = skips
        self.act = ACTIVATIONS[act_fn]

    def __call__(self, idx: jax.Array) -> jax.Array:
        x = self.embed(idx)
        last = len(self.layers) - 1
        for i, lyr in enumerate(self.layers):
            h = lyr(x)
            if i < last:
                h = self.act(h)
            x = x + h if self.skips and h.shape == x.shape else h
        return self.readout(self.act(x))


def build_nf(mcfg, grid_size: Sequence[int], *, key) -> MLPNF:
    if mcfg.get("name", "mlp") != "mlp" or mcfg.get("embed_type", "discrete") != "discrete":
        raise NotImplementedError("only the mlp neural field with the discrete embedding")
    return MLPNF(
        grid_size,
        key=key,
        dim=int(mcfg.dim),
        n_layers=int(mcfg.n_layers),
        skips=bool(mcfg.get("skips", True)),
        act_fn=mcfg.get("act_fn", "silu"),
    )


def n_params(model) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


def grid_coords(grid_size: Sequence[int], flat: jax.Array) -> jax.Array:
    """Integer coordinates ``(n, ndim)`` of the flat grid indices ``flat``."""
    return jnp.stack(jnp.unravel_index(flat, tuple(grid_size)), axis=-1).astype(jnp.int32)


def sample_field(model: MLPNF, grid_size: Sequence[int]) -> jax.Array:
    """The field ``(out_dim, *grid_size)``, decoded one leading-axis slice at a time."""
    n0, rest = grid_size[0], tuple(grid_size[1:])
    inner = jnp.arange(math.prod(rest))

    @jax.checkpoint
    def one(i):
        coords = jnp.concatenate(
            [jnp.full((inner.size, 1), i, jnp.int32), grid_coords(rest, inner)], axis=-1
        )
        return model(coords)

    out = jax.lax.map(one, jnp.arange(n0))
    return jnp.moveaxis(out, -1, 0).reshape(-1, *grid_size)


def torch_state(model: MLPNF) -> dict:
    """The torch ``MLPNF`` state_dict (numpy) of a JAX neural field."""
    import numpy as np

    out = {f"coord_embed.embeds.{i}.weight": t for i, t in enumerate(model.embed.tables)}
    for i, lyr in enumerate(model.layers):
        out[f"net.blocks.{i}.block.0.weight"] = lyr.weight
        out[f"net.blocks.{i}.block.0.bias"] = lyr.bias
    out["readout.weight"], out["readout.bias"] = model.readout.weight, model.readout.bias
    return {k: np.asarray(jax.device_get(v), np.float32) for k, v in out.items()}


def from_torch_state(model: MLPNF, state: dict) -> MLPNF:
    """``model`` with the arrays of a torch ``MLPNF`` state_dict (strict)."""
    ref = torch_state(model)
    if set(state) != set(ref):
        raise KeyError(f"state_dict keys differ: {sorted(set(state) ^ set(ref))[:5]}")
    n = len(model.layers)
    model = eqx.tree_at(
        lambda m: m.embed.tables,
        model,
        [
            jnp.asarray(state[f"coord_embed.embeds.{i}.weight"])
            for i in range(len(model.embed.tables))
        ],
    )
    for i in range(n):
        model = eqx.tree_at(
            lambda m: m.layers[i].inner.weight,
            model,
            jnp.asarray(state[f"net.blocks.{i}.block.0.weight"]),
        )
        model = eqx.tree_at(
            lambda m: m.layers[i].inner.bias,
            model,
            jnp.asarray(state[f"net.blocks.{i}.block.0.bias"]),
        )
    model = eqx.tree_at(
        lambda m: m.readout.inner.weight, model, jnp.asarray(state["readout.weight"])
    )
    return eqx.tree_at(lambda m: m.readout.inner.bias, model, jnp.asarray(state["readout.bias"]))
