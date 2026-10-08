"""Quantities of a trained smooth field patching model on a real frame, for the smooth diagram.

Run with the patching benchmark code of the model on ``PYTHONPATH`` (its ``bench`` dir included).
Usage: extract_smooth.py <cache_dir> <model.eqx> <out.npz>
"""

import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import bench2 as b
from neugk_jax.models.patching import fold_patches, pad_to_blocks
from neugk_jax.models.patching.field import cosine_basis

PART, SET, FRAME = "adiabatic", "adiabatic_ood", 3
OPTS = {"encoder": "smooth", "decoder": "smooth", "encoding": "cosine", "code_modes": [2, 2, 3, 2], "mix_depth": 1, "mix_act": False, "branch_depth": 1}


def main(cache_dir, model_path, out):
    cache = b.Cache(cache_dir)
    geo = b.Geometry(cache)
    model = b.build(cache, geo, "field", dict(OPTS))
    model = eqx.tree_deserialise_leaves(model_path, model)
    df, (geometry, species, _) = cache.load(SET, FRAME)
    df = df.astype(jnp.float32)
    enc = model.embed.with_grid(model.enc_grids[PART])
    dec = model.unpatch.with_grid(model.dec_grids[PART])
    grid, patch = enc.grid, enc.grid.patch
    xs = pad_to_blocks(b.to_points(df)[0], patch)
    p = fold_patches(xs, patch)
    z = enc(xs, geometry, species[0])
    feats = grid.features(geometry, *enc.encoding)
    K = enc.filters(feats)
    phi = cosine_basis(grid.pos, enc.code_modes)
    psi = dec.basis(dec.grid.features(geometry, *dec.encoding))
    codes = dec.expansion(z)
    n_k = phi.shape[-1]

    # the patch with the most energy; slices through its points at fixed vpar, y, channel and mu
    energy = jnp.sum(p**2, -1)
    tok = tuple(int(i) for i in np.unravel_index(int(jnp.argmax(energy)), energy.shape))
    n_c, n_mu = grid.n_channels, grid.n_fold[0]
    shape = (*patch, n_c, n_mu)
    ivp, iy, ic, imu = 1, 1, 0, 3

    def sl(v):
        # (P,) in fold order -> (s, x) slice of the patch
        return np.asarray(v).reshape(shape)[ivp, :, :, iy, ic, imu]

    k_maps = np.stack([sl(K[tok[0], :, r]) for r in range(K.shape[-1])])
    phi_maps = np.stack([sl(phi[:, k]) for k in range(n_k)])
    psi_maps = np.stack([sl(psi[tok[0], :, r]) for r in range(psi.shape[-1])])
    c = np.asarray(codes[tok]).reshape(n_k, -1)
    terms = np.einsum("kr,kij,rij->krij", c, phi_maps, psi_maps) / psi.shape[-1]
    recon = terms.sum((0, 1))
    full = np.asarray(dec(z, None, geometry, species[0]))
    full_p = np.asarray(fold_patches(full, patch))[tok]
    truth = sl(p[tok])

    # the full (s, x) plane of the frame at the slice's vpar, y and mu, the patch it sits in
    plane = np.asarray(xs)[tok[0] * patch[0] + ivp, :, :, tok[3] * patch[3] + iy, ic * n_mu + imu]

    # the same filters on the 2x finer x grid of the resolution test
    res_grid = model.enc_grids[f"{PART}_res"]
    _, (g_res, _, _) = cache.load(f"{PART}_res", FRAME)
    k_res = enc.with_grid(res_grid).filters(res_grid.features(g_res, *enc.encoding))
    shape_res = (*res_grid.patch, n_c, n_mu)
    k_res_maps = np.asarray(k_res[tok[0]]).reshape(*shape_res, -1)[ivp, :, :, iy, ic, imu].transpose(2, 0, 1)

    h = np.asarray(jnp.einsum("p,pr,pk->kr", p[tok], K[tok[0]] * grid.weight[tok[0]][:, None], phi) / p.shape[-1])
    np.savez(
        out,
        plane=plane,
        patch=np.asarray(patch),
        token=np.asarray(tok),
        truth=truth,
        recon=recon,
        recon_model=sl(full_p),
        k_maps=k_maps,
        k_res_maps=k_res_maps,
        phi_maps=phi_maps,
        psi_maps=psi_maps,
        terms=terms,
        codes=c,
        h=h,
        z=np.asarray(z[tok]),
        pos=np.asarray(grid.pos).reshape(*shape, -1)[ivp, :, :, iy, ic, imu],
        pos_res=np.asarray(res_grid.pos).reshape(*shape_res, -1)[ivp, :, :, iy, ic, imu],
        code_modes=np.asarray(enc.code_modes),
    )
    print("token", tok, "relative recon error of the slice", np.linalg.norm(recon - truth) / np.linalg.norm(truth))


if __name__ == "__main__":
    with jax.default_matmul_precision("highest"):
        main(*sys.argv[1:])
