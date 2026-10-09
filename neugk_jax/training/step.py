"""Training step: the optimizer update, its static spec and the jitted step on a train state.

``PartitionedTrainStep`` splits the static structure of the model and optimizer state off once and
runs a plain ``jax.jit`` on the arrays, a donated :class:`TrainState` with explicit shardings
(as the MaxText train step). ``train_step`` is the same step under ``eqx.filter_jit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.random as jr
import numpy as np

from neugk_jax.utils import count_trace


def train_update(model, opt_state, loss_fn, optimizer, mask, *, has_aux: bool = False):
    """One optimizer step on the leaves ``mask`` marks trainable; buffers stay fixed."""
    params, static = eqx.partition(model, mask)
    out, grads = eqx.filter_value_and_grad(
        lambda p: loss_fn(eqx.combine(p, static)), has_aux=has_aux
    )(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.combine(eqx.apply_updates(params, updates), static), opt_state, out


@dataclass(frozen=True, eq=False)
class StepSpec:
    """Static part of a train step: loss hook, optimizer, trainable mask and post-update hook."""

    name: str
    loss_fn: Callable
    optimizer: Any
    mask: Any
    post_update: Callable


def _train_step(inputs, model, opt_state, spec: StepSpec):
    """``inputs = (batch, ctx, key)``; ``ctx`` holds the run-constant device tables (not donated).

    A ``"state"`` entry of the loss aux is not logged but handed to
    ``spec.post_update(model, state, key) -> (model, logs)`` after the optimizer step.
    """
    count_trace(f"train_step:{spec.name}")
    batch, ctx, key = inputs

    def loss(m):
        return spec.loss_fn(m, {**ctx, **batch}, key)

    model, opt_state, (value, aux) = train_update(
        model, opt_state, loss, spec.optimizer, spec.mask, has_aux=True
    )
    aux = dict(aux)
    model, extra = spec.post_update(model, aux.pop("state", None), jr.fold_in(key, 1))
    return model, opt_state, {"total": value, **aux, **extra}


train_step = eqx.filter_jit(_train_step, donate="all-except-first")


@jax.tree_util.register_dataclass
@dataclass
class TrainState:
    """Array leaves of the model and optimizer state; their structure lives in the step."""

    params: list
    opt_state: list


def _split(tree):
    arrays, static = eqx.partition(tree, eqx.is_array)
    leaves, treedef = jax.tree_util.tree_flatten(arrays)
    return leaves, treedef, static


class PartitionedTrainStep:
    """``train_step`` as a plain ``jax.jit`` on a :class:`TrainState`, the structure split once.

    ``init`` partitions a model and optimizer state into a ``TrainState``, a call
    ``step(state, (batch, ctx, key), i) -> (state, logs)`` runs one update with the step key
    ``fold_in(key, i)`` derived inside the program, and ``restore`` rebuilds the model and optimizer
    state. ``sharding`` (the replicated sharding of the state) pins the state's input and output
    shardings; non-array inputs are static, as under ``eqx.filter_jit``.

    ``eqx.filter_jit`` re-partitions and re-flattens the whole model and optimizer pytree of
    ``Module`` nodes on every call, then rebuilds them from the outputs, and blocks until the step
    is done. Here the structure is split once per run and a step only passes flat lists of arrays
    through jax's C++ dispatch, so the host queues the next step while the device runs this one.
    The same pattern is the recommended one in:

    1. Equinox, "Low-overhead training loops":
       https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops
    2. MaxText, the train step jitted on the state with explicit shardings and the state donated:
       https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/train_utils.py
    3. Flax NNX, "Functional training loop" (``nnx.split`` once, plain ``jax.jit``, ``nnx.merge``
       inside): https://flax.readthedocs.io/en/latest/guides/performance.html
    """

    def __init__(self, spec: StepSpec, model, opt_state, *, sharding: Optional[Any] = None):
        _, self._m_def, self._m_static = _split(model)
        _, self._o_def, self._o_static = _split(opt_state)

        def step(state, dyn, i, static):
            batch, ctx, key = eqx.combine(dyn, jax.tree_util.tree_unflatten(*static))
            model, opt_state = self.restore(state)
            model, opt_state, logs = _train_step(
                (batch, ctx, jr.fold_in(key, i)), model, opt_state, spec
            )
            return self.init(model, opt_state), logs

        shardings = (
            {}
            if sharding is None
            else dict(in_shardings=(sharding, None, None), out_shardings=(sharding, None))
        )
        self._step = jax.jit(step, static_argnums=3, donate_argnums=0, **shardings)

    def init(self, model, opt_state) -> TrainState:
        return TrainState(_split(model)[0], _split(opt_state)[0])

    def restore(self, state: TrainState):
        model = eqx.combine(jax.tree_util.tree_unflatten(self._m_def, state.params), self._m_static)
        opt_state = eqx.combine(
            jax.tree_util.tree_unflatten(self._o_def, state.opt_state), self._o_static
        )
        return model, opt_state

    def matches(self, model, opt_state) -> bool:
        (_, m_def, m_static), (_, o_def, o_static) = _split(model), _split(opt_state)
        return (m_def, o_def) == (self._m_def, self._o_def) and bool(
            eqx.tree_equal((m_static, o_static), (self._m_static, self._o_static))
        )

    def __call__(self, state: TrainState, inputs, i: int):
        dyn, static = eqx.partition(inputs, eqx.is_array)
        leaves, treedef = jax.tree_util.tree_flatten(static)
        return self._step(state, dyn, np.int32(i), (treedef, tuple(leaves)))
