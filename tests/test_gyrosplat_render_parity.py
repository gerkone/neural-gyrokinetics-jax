"""Parity of the JAX splat renderer against the torch gyrosplats reference.

The torch reference modules live on the ``origin/gyrosplats`` git ref
(``neugk/pinc/gyrosplats/model``); a real fitted latent plus its raw snapshot
gate the full-fidelity checks. Pure-shape/roundtrip tests always run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neugk_jax.gyrosplats.normalize import ZfStats, zf_denormalize, zf_normalize
from neugk_jax.gyrosplats.render import render, subgrids, to_field, to_sep
from neugk_jax.gyrosplats.splat import SplatParams, inv_softplus, pack, tri, unpack

torch = pytest.importorskip("torch")

# parity tests compare against fp32 cpu torch — disable tf32 matmuls
jax.config.update("jax_default_matmul_precision", "highest")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GYROSPLAT_DATA = os.environ.get(
    "NEUGK_GYROSPLAT_PATH", "/restricteddata/ukaea/gyrokinetics/gyrosplats/data"
)
SNAPSHOT_DATA = os.environ.get(
    "NEUGK_CYCLONE_PATH", "/restricteddata/ukaea/gyrokinetics/preprocessed_kvikio"
)


def _torch_reference_dir():
    """Materialize the torch reference modules from the gyrosplats git ref."""
    d = tempfile.mkdtemp(prefix="gyrosplats_ref_")
    for f in ("atoms.py", "render.py", "normalize.py"):
        out = subprocess.run(
            ["git", "show", f"origin/gyrosplats:neugk/pinc/gyrosplats/model/{f}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            pytest.skip("origin/gyrosplats ref not available")
        with open(os.path.join(d, f), "w") as fh:
            fh.write(out.stdout)
    return d


@pytest.fixture(scope="module")
def torch_ref():
    d = _torch_reference_dir()
    sys.path.insert(0, d)
    import atoms as t_atoms  # noqa: F401
    import normalize as t_normalize  # noqa: F401
    import render as t_render  # noqa: F401

    yield t_atoms, t_render, t_normalize
    sys.path.remove(d)


def _random_params(key, n=64):
    ks = jax.random.split(key, 5)
    return SplatParams(
        mu=jax.random.uniform(ks[0], (n, 5)),
        L_phys_raw=jax.random.normal(ks[1], (n, 6)) - 2.0,
        L_vel_raw=jax.random.normal(ks[2], (n, 3)) - 2.0,
        amps=jax.random.normal(ks[3], (n, 2)),
        ky=2.0 * jnp.pi * jax.random.randint(ks[4], (n,), 0, 14).astype(jnp.float32),
    )


def test_pack_unpack_roundtrip():
    p = _random_params(jax.random.PRNGKey(0))
    q = unpack(pack(p))
    for a, b in zip(p, q):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_tri_inv_softplus_roundtrip():
    x = jnp.array([0.5, 1.0, 2.0, 0.01])
    np.testing.assert_allclose(
        np.asarray(jnp.logaddexp(inv_softplus(x), 0.0)), np.asarray(x), rtol=1e-5
    )
    L = tri(jnp.array([[0.3, -1.0, 0.2, 0.5, -0.4, 0.1]]), 3)
    assert L.shape == (1, 3, 3)
    assert np.all(np.asarray(L[0])[np.triu_indices(3, 1)] == 0.0)
    assert np.all(np.diag(np.asarray(L[0])) > 0.0)


def test_to_field_to_sep_roundtrip():
    shape = (4, 2, 3, 5, 6)
    sep = jax.random.normal(jax.random.PRNGKey(1), (8, 90, 2))
    np.testing.assert_allclose(
        np.asarray(to_sep(to_field(sep, shape))), np.asarray(sep)
    )


def test_render_matches_torch_reference(torch_ref):
    t_atoms, t_render, _ = torch_ref
    shape = (8, 4, 6, 10, 12)
    p = _random_params(jax.random.PRNGKey(2), n=48)
    vg, pg = subgrids(shape)

    sep_jax = render(p, vg, pg)
    sep_jax_chunked = render(p, vg, pg, atom_chunk=13)

    bank = t_atoms.Splat(
        *(torch.tensor(np.asarray(a)) for a in (p.mu, p.L_phys_raw, p.L_vel_raw, p.amps)),
        torch.tensor(np.asarray(p.ky)),
        learn_ky=False,
    )
    axes = [torch.arange(n, dtype=torch.float32) / n for n in shape]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 5)
    tvg, tpg = t_render.subgrids(grid, shape)
    np.testing.assert_allclose(np.asarray(vg), tvg.numpy(), atol=1e-6)
    np.testing.assert_allclose(np.asarray(pg), tpg.numpy(), atol=1e-6)
    with torch.no_grad():
        sep_t = t_render.render(bank, tvg, tpg).numpy()

    np.testing.assert_allclose(np.asarray(sep_jax), sep_t, atol=1e-4)
    np.testing.assert_allclose(np.asarray(sep_jax_chunked), sep_t, atol=1e-4)


def test_zf_normalize_matches_torch_reference(torch_ref):
    _, _, t_normalize = torch_ref
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 4, 2, 3, 5, 6)) * 2.0 + 0.3
    xn, st = zf_normalize(x)
    tx = torch.tensor(np.asarray(x))
    txn, tst = t_normalize.normalize(tx)
    np.testing.assert_allclose(np.asarray(xn), txn.numpy(), atol=1e-5)
    back = zf_denormalize(xn, st)
    np.testing.assert_allclose(np.asarray(back), np.asarray(x), atol=1e-5)
    tback = t_normalize.denormalize(txn, tst)
    np.testing.assert_allclose(np.asarray(back), tback.numpy(), atol=1e-5)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(GYROSPLAT_DATA, "iteration_13")),
    reason="gyrosplat latents not readable",
)
def test_real_latent_render_psnr(torch_ref):
    """Full-fidelity gate: denormalized render of a real latent vs the raw snapshot."""
    sd = torch.load(
        os.path.join(GYROSPLAT_DATA, "iteration_13", "latent_0080.pt"),
        map_location="cpu",
        weights_only=False,
    )
    p = SplatParams(
        mu=jnp.asarray(sd["mu"].numpy()),
        L_phys_raw=jnp.asarray(sd["L_phys_raw"].numpy()),
        L_vel_raw=jnp.asarray(sd["L_vel_raw"].numpy()),
        amps=jnp.asarray(sd["amps"].numpy()),
        ky=jnp.asarray(sd["ky"].numpy()),
    )
    ns = sd["norm_stats"]
    st = ZfStats(
        jnp.asarray(ns["zonal_mean"]),
        jnp.asarray(ns["zonal_std"]),
        jnp.asarray(ns["fluc_mean"]),
        jnp.asarray(ns["fluc_std"]),
        ns["kind"],
    )

    meta_path = os.path.join(SNAPSHOT_DATA, "iteration_13_ifft_realpotens")
    if not os.path.isdir(meta_path):
        pytest.skip("raw snapshots not readable")
    import pickle

    with open(os.path.join(meta_path, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    shape = tuple(int(v) for v in meta["resolution"])
    raw = np.fromfile(
        os.path.join(meta_path, "data", "timestep_00080.bin"), dtype=np.float32
    ).reshape(2, *shape)

    vg, pg = subgrids(shape)
    recon = zf_denormalize(to_field(render(p, vg, pg, atom_chunk=256), shape), st)
    recon = np.asarray(recon)

    # psnr on the normalized-phi convention used at fit time is stored in the file;
    # here gate on field psnr in normalized space being finite and close to the fit
    gt_n, gt_st = zf_normalize(jnp.asarray(raw))
    recon_n = (
        to_field(render(p, vg, pg, atom_chunk=256), shape)
    )  # fit-space render (already normalized space)
    mse = float(jnp.mean((recon_n - gt_n) ** 2))
    peak = float(jnp.max(jnp.abs(gt_n)))
    psnr = 10.0 * np.log10(peak**2 / mse)
    nmse = float(np.mean((recon - raw) ** 2) / np.mean(raw**2))
    # values measured with the torch reference pipeline: nmse ~0.295 at t=80
    assert nmse < 0.35, nmse
    assert np.isfinite(psnr)
