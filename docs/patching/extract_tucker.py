"""Quantities of a trained Tucker continuous-convolution model on a real frame, for its diagram.

Run with the patching benchmark code on ``PYTHONPATH`` (its ``bench`` dir included); the trained
parameters come from ``export_params.py``.
Usage: extract_tucker.py <cache_dir> <params.npz> <out.npz> <bench options...>
"""

import sys

import bench2 as b
import jax
import jax.numpy as jnp
import numpy as np
from export_params import load

from neugk_jax.models.patching import fold_patches, pad_to_blocks
from neugk_jax.models.patching.cconv import _cosines, _window_coords

PART, SET, FRAME = "adiabatic", "adiabatic_ood", 3
# slice through the points of a patch: vpar index, y index, channel, mu index
IVP, IY, IC, IMU = 1, 1, 0, 3


def main(cache_dir, params, out, *args):
    cache = b.Cache(cache_dir)
    model = load(b.build(cache, b.Geometry(cache), "field", b.parse(args)), params)
    df, (geometry, _, _) = cache.load(SET, FRAME)
    df, geometry = df.astype(jnp.float32), model.grid_geometry(PART, geometry)
    enc = model.embed.with_grid(model.enc_grids[PART])
    dec = model.unpatch.with_grid(model.dec_grids[PART])
    grid, patch = enc.grid, enc.grid.patch
    ranks = enc.ranks
    xs = pad_to_blocks(b.to_points(df)[0], patch)
    p = fold_patches(xs, patch)
    energy = jnp.sum(p**2, -1)
    tok = tuple(int(i) for i in np.unravel_index(int(jnp.argmax(energy)), energy.shape))
    n_c, n_mu = grid.n_channels, grid.n_fold[0]

    # per-axis bases of the token (a basis per token row on an axis with token-dependent bases)
    def take(bases):
        return [np.asarray(v[tok[k]] if v.ndim == 3 else v) for k, v in enumerate(bases)]

    eb, db = take(enc.kernel(geometry)), take(dec.kernel(geometry))
    modes = []
    for ax, r in zip(grid.axes, ranks):
        u = _window_coords(ax)
        m = _cosines(u, r, min(ax.cap, u.shape[-1]), orthonormal=True)
        modes.append(np.asarray(m[0] if m.ndim == 3 else m))

    core_shape = (*ranks[:4], n_c, ranks[4])
    core_enc = np.asarray(enc.project(p, geometry)[tok]).reshape(core_shape)
    z = enc(xs, geometry)
    codes = np.asarray(dec.expansion(z[tok])).reshape(core_shape)
    # the decoder core reduced to the (s, x) plane at the slice's vpar, y, channel and mu
    c_sx = np.einsum("aijbm,a,b,m->ij", codes[:, :, :, :, IC], db[0][IVP], db[3][IY], db[4][IMU])
    terms = np.einsum("ij,si,xj->ijsx", c_sx, db[1], db[2])
    sl = lambda v: np.asarray(v).reshape(*patch, n_c, n_mu)[IVP, :, :, IY, IC, IMU]
    full = np.asarray(fold_patches(dec(z, None, geometry), patch))[tok]
    plane = np.asarray(xs)[tok[0] * patch[0] + IVP, :, :, tok[3] * patch[3] + IY, IC * n_mu + IMU]
    e = np.abs(core_enc[:, :, :, :, IC]).sum((1, 2))
    a, bb, m = np.unravel_index(int(np.argmax(e)), e.shape)
    np.savez(
        out,
        plane=plane,
        patch=np.asarray(patch),
        token=np.asarray(tok),
        truth=sl(p[tok]),
        recon=terms.sum((0, 1)),
        recon_model=sl(full),
        bases_enc=np.array(eb, dtype=object),
        bases_dec=np.array(db, dtype=object),
        modes=np.array(modes, dtype=object),
        core_enc_sx=core_enc[a, :, :, bb, IC, m],
        core_dec_sx=c_sx,
        terms=terms,
        z=np.asarray(z[tok]),
        ranks=np.asarray(ranks),
        channels=n_c,
        token_dim=z.shape[-1],
    )
    err = np.linalg.norm(terms.sum((0, 1)) - sl(p[tok])) / np.linalg.norm(sl(p[tok]))
    print("token", tok, "relative error of the slice", err)


if __name__ == "__main__":
    with jax.default_matmul_precision("highest"):
        main(*sys.argv[1:])
