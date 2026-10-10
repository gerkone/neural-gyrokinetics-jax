"""Checkpoints written before the attention / token / patching restructure, on the current models.

A checkpoint pickles the model's array pytree, so it names the classes it was built from. The
legacy reader loads it without any of them (every ``neugk_jax`` class becomes a plain state
holder), flattens the arrays in jax's leaf order (module fields in order, dicts by sorted key) and
grafts them onto a model built with the current code from the same config, checking every shape.
Usage: python -m neugk_jax.training.legacy <old.eqx> <new.eqx> <run config.yaml>  (autoencoders)
"""

from __future__ import annotations

import dataclasses
import pickle
import sys
from typing import Any

import equinox as eqx
import numpy as np

from neugk_jax.training.checkpoint import CheckpointState, _graft, save_checkpoint


class _State:
    """Stand-in for a pickled ``neugk_jax`` object: keeps its state, needs no class."""

    def __setstate__(self, state):
        if isinstance(state, tuple):
            state = {k: v for part in state if isinstance(part, dict) for k, v in part.items()}
        self.__dict__.update(state)


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "neugk_jax":
            return type(name, (_State,), {})
        return super().find_class(module, name)


def read_legacy(path) -> dict:
    """The checkpoint bundle with every ``neugk_jax`` object as a :class:`_State`."""
    with open(path, "rb") as f:
        return _LegacyUnpickler(f).load()


def legacy_leaves(tree: Any) -> list[np.ndarray]:
    """The arrays of a legacy pytree in jax's leaf order."""
    out = []

    def visit(node):
        if isinstance(node, np.ndarray):
            out.append(node)
        elif isinstance(node, _State):
            for v in node.__dict__.values():
                visit(v)
        elif isinstance(node, eqx.Module):
            for f in dataclasses.fields(node):
                visit(getattr(node, f.name, None))
        elif isinstance(node, dict):
            for k in sorted(node):
                visit(node[k])
        elif isinstance(node, (list, tuple)):
            for v in node:
                visit(v)

    visit(tree)
    return out


def convert(path, template) -> CheckpointState:
    """A legacy checkpoint on ``template`` (a current model of the same config); no optimizer state."""
    bundle = read_legacy(path)
    model = _graft(legacy_leaves(bundle["model_leaves"]), template)
    meta = {**(bundle.get("meta") or {}), "converted_from": str(path)}
    return CheckpointState(
        model=model,
        opt_state=None,
        epoch=int(bundle["epoch"]),
        loss=float(bundle["loss"]),
        meta=meta,
    )


def main(old, new, config):
    import jax.random as jr

    from neugk_jax.models.build import build_ae_from_config

    template = build_ae_from_config(config, key=jr.PRNGKey(0))
    save_checkpoint(new, convert(old, template))
    print(f"{old} -> {new}")


if __name__ == "__main__":
    main(*sys.argv[1:])
