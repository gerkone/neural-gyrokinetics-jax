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


@pytest.mark.parametrize("name", ["GyroSwin_tiny", "GyroSwin_cold", "GyroSwin_warm"])
def test_gyroswin_translation_is_complete(name, tmp_path):
    import yaml

    from neugk_jax.gyroswin.models.gyroswin import build_gyroswin_from_config
    from neugk_jax.translate import load_torch_state, translate_gyroswin

    run = _run(name)
    full = yaml.safe_load(open(os.path.join(run, "config.yaml")))
    cfg = tmp_path / "cfg.yaml"
    ds = {"separate_zf": full["dataset"].get("separate_zf", True),
          "real_potens": full["dataset"].get("real_potens", True),
          "resolution": [32, 8, 16, 85, 32]}
    cfg.write_text(yaml.safe_dump({"model": full["model"], "dataset": ds}))
    model = build_gyroswin_from_config(str(cfg), key=jr.PRNGKey(0))
    sd = load_torch_state(os.path.join(run, "best.pth"))
    _, missing, unused = translate_gyroswin(model, sd)
    assert not missing, missing[:5]
    # torch registers shared modules under several names; only byte-distinct keys count
    used_bytes = {sd[k].tobytes() for k in set(sd) - set(unused)}
    real = [k for k in unused if sd[k].tobytes() not in used_bytes]
    # the outer flux cond embed is never called by torch's forward
    assert all(k.startswith("flux_head.cond_embed.") for k in real), real
    assert model.flux_head is not None
    assert model.flux_head.use_cond == bool(full["model"]["swin"].get("flux_conditioning"))


def test_conditioned_flux_decoder_forward_parity():
    import jax.numpy as jnp
    import numpy as np
    import torch
    from neugk.gyroswin.models.x_layers import FluxDecoder as TorchFluxDecoder

    from neugk_jax.gyroswin.models.x_layers import FluxDecoder
    from neugk_jax.translate import translate_gyroswin

    torch.manual_seed(0)
    left_dims, right_dims, n_cond = [32, 16], [64, 32], 3
    tmod = TorchFluxDecoder(left_dims, right_dims, num_heads=4, depth=1, n_cond=n_cond,
                            cond_embed_dim=128, drop=0.1, attn_drop=0.1).eval()
    sd = {k: v.detach().numpy() for k, v in tmod.state_dict().items()}
    jmod = FluxDecoder(left_dims, right_dims, 4, 1, key=jr.PRNGKey(0), n_cond=n_cond)
    jmod, missing, unused = translate_gyroswin(jmod, sd)
    assert not missing and all(k.startswith("cond_embed.") for k in unused), (missing, unused)

    rng = np.random.default_rng(0)
    grids = [(2, 3, 2), (4, 6, 4)]
    cond = rng.standard_normal(n_cond).astype(np.float32)
    lefts = [rng.standard_normal((*g, d)).astype(np.float32) for g, d in zip(grids, left_dims)]
    rights = [rng.standard_normal((*g, d)).astype(np.float32) for g, d in zip(grids, right_dims)]
    with torch.no_grad():
        tc = torch.from_numpy(cond)[None]
        tl = [tmod.mix(i, torch.from_numpy(a)[None], torch.from_numpy(b)[None], cond=tc)
              for i, (a, b) in enumerate(zip(lefts, rights))]
        tflux = tmod(tl).numpy()
    jl = [jmod.mix(i, jnp.asarray(a), jnp.asarray(b), jnp.asarray(cond))
          for i, (a, b) in enumerate(zip(lefts, rights))]
    jflux = np.asarray(jmod(jl))
    np.testing.assert_allclose(jflux, tflux, rtol=1e-4, atol=1e-5)
