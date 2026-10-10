"""Checkpoints written by the pre-restructure code (commit 9e5081b) load exactly on the current models."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from neugk_jax.training.legacy import convert, legacy_leaves, read_legacy

DATA = Path(__file__).parent / "data" / "legacy"


def _fixtures():
    spec = importlib.util.spec_from_file_location("make_fixture", DATA / "make_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {"ae": mod.ae, "unet": mod.unet}


@pytest.mark.parametrize("name", ["ae", "unet"])
def test_legacy_checkpoint_reproduces_its_outputs(name):
    template, inputs, fwd = _fixtures()[name]()
    state = convert(DATA / f"{name}.eqx", template)
    ref = np.load(DATA / f"{name}.npz")
    with jax.default_matmul_precision("highest"):
        y = fwd(state.model, *[ref[f"in{i}"] for i in range(len(inputs))])
    np.testing.assert_allclose(np.asarray(y), ref["y"], atol=1e-5)
    assert state.meta["fixture"] == name


def test_legacy_reader_needs_no_old_classes():
    bundle = read_legacy(DATA / "ae.eqx")
    leaves = legacy_leaves(bundle["model_leaves"])
    template, _, _ = _fixtures()["ae"]()
    current = [x for x in jax.tree_util.tree_leaves(template) if eqx.is_array(x)]
    assert [np.shape(a) for a in leaves] == [np.shape(b) for b in current]
