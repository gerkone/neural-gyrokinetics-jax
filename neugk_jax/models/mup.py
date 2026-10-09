"""muP (maximal update parametrization) of the swin autoencoders, after the torch reference.

Only the width is muP-governed: ``width_dims`` derives every width-dependent dim from one width
(head dim fixed, heads and bottleneck proportional). Comparing the shapes of the model built at
``base_width`` and ``delta_width`` marks the dims of every parameter that grow with width; a
parameter with two such dims is matrix-like and gets the learning rate divided, and the coupled
weight decay multiplied, by its fan-in over the base fan-in, as ``mup.MuAdam``. The readout (the
first linear of every unpatch expansion, whose output width is fixed) is zero-initialized and sees
its input scaled by ``output_mult / width_mult``, as ``mup.MuReadout``.
"""

from __future__ import annotations

from typing import Callable

import jax


def width_dims(width: int, head_dim: int = 64, bottleneck_ratio: int = 2) -> dict:
    if width % head_dim:
        raise ValueError(f"width {width} must be divisible by head_dim {head_dim}")
    return {
        "latent_dim": width,
        "num_heads": width // head_dim,
        "bottleneck_dim": max(1, width // bottleneck_ratio),
        "bottleneck_num_heads": width // head_dim,
    }


def multipliers(model, mask, base, delta) -> tuple:
    """``(lr_mult, wd_mult)`` pytrees over the leaves of ``model`` that ``mask`` marks trainable.

    ``base`` / ``delta`` are the same model at the base / delta width, possibly abstract
    (``eqx.filter_eval_shape``); leaves are matched by position.
    """
    import equinox as eqx

    leaves, treedef = jax.tree_util.tree_flatten(model)
    shapes = lambda t: [getattr(x, "shape", None) for x in jax.tree_util.tree_leaves(t)]
    base_shapes, delta_shapes = shapes(base), shapes(delta)
    if not len(leaves) == len(base_shapes) == len(delta_shapes):
        raise ValueError("the base / delta models do not have the leaves of the model")
    lr, wd = [], []
    for leaf, b, d in zip(leaves, base_shapes, delta_shapes):
        inf = [i for i, (x, y) in enumerate(zip(b, d)) if x != y] if b is not None else []
        if len(inf) > 2:
            raise NotImplementedError(f"muP leaf with {len(inf)} width dims {leaf.shape}")
        mult = leaf.shape[-1] / b[-1] if len(inf) == 2 else 1.0
        lr.append(1.0 / mult)
        wd.append(mult)
    unflat = lambda v: eqx.filter(jax.tree_util.tree_unflatten(treedef, v), mask)
    return unflat(lr), unflat(wd)


def build_multipliers(
    build: Callable[[int], object], model, mask, base_width: int, delta_width: int
) -> tuple:
    """:func:`multipliers` with ``build(width)`` the model at ``width`` (built abstractly)."""
    import equinox as eqx

    return multipliers(
        model,
        mask,
        eqx.filter_eval_shape(build, base_width),
        eqx.filter_eval_shape(build, delta_width),
    )
