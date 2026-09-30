"""Every persistent torch leaf of the reference checkpoints is placed on the JAX template."""

from __future__ import annotations

import glob
import os

import jax.random as jr
import pytest

ROOT = os.environ.get("NEUGK_CKPT_ROOT", "/restricteddata/ukaea/checkpoints/neurips26")


def _run(name):
    hits = sorted(glob.glob(os.path.join(ROOT, name, "*", "best.pth")), key=len)
    if not hits:
        pytest.skip(f"no {name} checkpoint under {ROOT}")
    return os.path.dirname(hits[0])


def test_ae_translation_is_complete():
    from neugk_jax.translate import build_ae_from_config, load_torch_state, translate_ae

    run = _run("AE_noCond")
    model = build_ae_from_config(os.path.join(run, "config.yaml"), key=jr.PRNGKey(0))
    _, missing, unused = translate_ae(model, load_torch_state(os.path.join(run, "best.pth")))
    assert not missing and not unused, (missing[:5], unused[:5])
