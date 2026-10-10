"""Quantities of a trained band-limited continuous-convolution model on a real frame, for its diagram.

Run with the patching benchmark code on ``PYTHONPATH`` (its ``bench`` dir included); the trained
parameters come from ``export_params.py``.
Usage: extract_cconv.py <cache_dir> <params.npz> <out.npz> <bench options...>
"""

import sys

import bench2 as b
import jax
import jax.numpy as jnp
import numpy as np
from export_params import load

from neugk_jax.models.patching import fold_patches, pad_to_blocks
from neugk_jax.models.patching.cconv import cosine_basis

PART, SET, FRAME = "adiabatic", "adiabatic_ood", 3
# slice through the points of a patch: vpar index, y index, channel, mu index
IVP, IY, IC, IMU = 1, 1, 0, 3


def split_channels(p, grid):
    """Folded patches ``(*T, P * C)`` as ``(*T, P, C)``, the points in :class:`PointGrid` order."""
    p = p.reshape(*p.shape[:-1], -1, grid.n_channels, int(np.prod(grid.n_fold)))
    return jnp.moveaxis(p, -2, -1).reshape(*p.shape[:-3], -1, grid.n_channels)


def main(cache_dir, params, out, *args):
    cache = b.Cache(cache_dir)
    model = load(b.build(cache, b.Geometry(cache), "field", b.parse(args)), params)
    df, (geometry, _, _) = cache.load(SET, FRAME)
    df, geometry = df.astype(jnp.float32), model.grid_geometry(PART, geometry)
    enc = model.embed.with_grid(model.enc_grids[PART])
    dec = model.unpatch.with_grid(model.dec_grids[PART])
    grid, patch = enc.grid, enc.grid.patch
    xs = pad_to_blocks(b.to_points(df)[0], patch)
    x = split_channels(fold_patches(xs, patch), grid)
    z = enc(xs, geometry)
    K = enc.kernel(geometry)
    psi = dec.kernel(geometry)
    phi = cosine_basis(grid.pos, enc.code_modes)
    n_k, n_c, n_mu = phi.shape[-1], grid.n_channels, grid.n_fold[0]

    # the patch with the most energy
    energy = jnp.sum(x**2, (-2, -1))
    tok = tuple(int(i) for i in np.unravel_index(int(jnp.argmax(energy)), energy.shape))
    shape = (*patch, n_mu)
    row = lambda a: a[tok[0]] if a.ndim == 3 else a

    def sl(v):
        # (P,) in point order -> (s, x) slice of the patch
        return np.asarray(v).reshape(shape)[IVP, :, :, IY, IMU]

    k_maps = np.stack([sl(row(K)[:, r]) for r in range(K.shape[-1])])
    psi_maps = np.stack([sl(row(psi)[:, r]) for r in range(psi.shape[-1])])
    phi_maps = np.stack([sl(phi[:, k]) for k in range(n_k)])
    codes = np.asarray(dec.expansion(z[tok])).reshape(n_k, n_c, -1)[:, IC]
    terms = np.einsum("kr,kij,rij->krij", codes, phi_maps, psi_maps) / psi.shape[-1]
    full = split_channels(fold_patches(dec(z, None, geometry), patch), grid)[tok]
    truth = sl(x[tok][:, IC])
    plane = np.asarray(xs)[tok[0] * patch[0] + IVP, :, :, tok[3] * patch[3] + IY, IC * n_mu + IMU]
    np.savez(
        out,
        plane=plane,
        patch=np.asarray(patch),
        token=np.asarray(tok),
        truth=truth,
        recon=terms.sum((0, 1)),
        recon_model=sl(full[:, IC]),
        k_maps=k_maps,
        phi_maps=phi_maps,
        psi_maps=psi_maps,
        terms=terms,
        codes=codes,
        z=np.asarray(z[tok]),
        code_modes=np.asarray(enc.code_modes),
        rank=K.shape[-1],
        bands=np.asarray(enc.filter.shape[:-1]),
        channels=n_c,
        token_dim=z.shape[-1],
    )
    err = np.linalg.norm(terms.sum((0, 1)) - truth) / np.linalg.norm(truth)
    print("token", tok, "relative error of the slice", err)


if __name__ == "__main__":
    with jax.default_matmul_precision("highest"):
        main(*sys.argv[1:])
