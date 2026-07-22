"""Per-trajectory dataset compression (JAX driver). Pure-JAX; NO torch, NO neugk.

    python -m neugk_jax.gyrowave.compress.process_trajectory \
        --version {real|semispectral} --traj iteration_13 \
        --support {per_snapshot|shared} --alpha 1 --beta 1 --gamma 1 --out <cache.npz>

Per trajectory:
  * geometry (magnetic equilibrium) is loaded once from the trajectory's metadata and the
    fast operator layer (fastops.TrajOps) is built from it -> pure JAX linear algebra.
  * 185 snapshots at OFFSET 80 (t = 80..264, range(80,265)); pre-saturation t<80 skipped.
  * fp32 throughout; K-files cast fp64->fp32 on load.
  * per-snapshot physics-importance support (df + phi + flux importance, iterative refine);
    STORED coefficients are phi-aware -> flux-GN.
  * writes a variable-length coordinate-tagged token cache (same npz format the runner reads).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import jax.numpy as jnp
import numpy as np

from neugk_jax.gyrowave.compress import fastops, pipeline

TRAJ_TS = list(range(80, 265))                    # 185 snapshots, offset 80

# per-trajectory real .bin + metadata dir (overridden by set_trajectory)
DATA_DIR = "/local00/bioinf/galletti/preprocessed_kvikio/iteration_13_ifft_realpotens"


def _squeeze0(v):
    a = np.asarray(v)
    if a.ndim > 0 and a.shape[0] == 1:
        a = a.reshape(a.shape[1:])
    return a


def load_meta(data_dir=None):
    data_dir = data_dir or DATA_DIR
    with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:
        return pickle.load(f)


def build_geom(meta):
    """Per-trajectory geometry dict (numpy fp64), matching common.load_snapshot['geom']."""
    geom = {k: _squeeze0(v).astype(np.float64) for k, v in meta["geometry"].items()}
    for k in ("adiabatic", "de", "beta", "nlapar", "nlbpar"):
        geom.setdefault(k, np.array(1.0, dtype=np.float64))
    return geom


def set_trajectory(traj, base="/local00/bioinf/galletti/preprocessed_kvikio"):
    """Redirect ALL data/geometry to `traj` (e.g. 'iteration_100'). The flux/phi operators
    depend on per-trajectory geometry (magnetic equilibrium), so this MUST run before TrajOps
    is built. Returns (meta, geom)."""
    global DATA_DIR
    DATA_DIR = os.path.join(base, f"{traj}_ifft_realpotens")   # real .bin + metadata
    fastops.RAW = f"/restricteddata/ukaea/gyrokinetics/raw/{traj}"  # semispectral K-files
    fastops._KS = None                                        # reset K-listing cache
    meta = load_meta()
    geom = build_geom(meta)
    return meta, geom


def load_raw(version, ts):
    """real field (2,vp,mu,s,x,y) fp32, or raw K (semispectral)."""
    if version == "real":
        arr = np.fromfile(os.path.join(DATA_DIR, "data", f"timestep_{ts:05d}.bin"),
                          dtype=np.float32).reshape(fastops.SHAPE)
        return jnp.asarray(arr)
    return fastops.load_K(ts, raw=fastops.RAW)


def run(version, traj, support_mode, weights, out_path, n_refine=1, limit=None, verbose=True,
        cr=400.0):
    meta, geom = set_trajectory(traj)             # redirect paths + per-trajectory geometry
    ops = fastops.TrajOps(version, geom, geom["vpgr"], geom["mugr"], verbose=verbose)
    budget = int(fastops.N_TOTAL // cr // 2)      # CR = compression ratio vs raw field DOF
    tslist = TRAJ_TS if limit is None else TRAJ_TS[:limit]

    shared_idx = None
    if support_mode == "shared":
        raw0 = load_raw(version, tslist[0])
        c_full = ops.fwd(raw0)
        shared_idx = pipeline.select_support(ops, c_full, budget, weights, n_refine=n_refine)

    coord_tbl = fastops.get_frame_coords()        # (M_TOT,5)
    collected = []
    for j, ts in enumerate(tslist):
        raw = load_raw(version, ts)
        tok, fid, t = pipeline.process_snapshot(
            ops, raw, budget, weights, n_refine=n_refine, do_phi=True,
            support_idx=shared_idx)
        fid["ts"] = ts
        collected.append((ts, tok, fid, t))
        if verbose and (j % 20 == 0 or j < 2):
            print(f"  [{version}] {j + 1}/{len(tslist)} ts={ts} df={fid['df_psnr']:.1f} "
                  f"phi={fid['phi_psnr']:.1f} flux={fid['flux_relerr']:.3f} "
                  f"zonal={fid['zonal_phi_ratio']:.2f} ntok={len(tok['idx'])}", flush=True)

    collected.sort(key=lambda r: r[0])
    tok_re, tok_im, tok_coord, offsets, fids = [], [], [], [0], []
    for ts, tok, fid, t in collected:
        idx = tok["idx"]
        tok_re.append(tok["re"])
        tok_im.append(tok["im"])
        tok_coord.append(coord_tbl[idx])
        offsets.append(offsets[-1] + len(idx))
        fids.append(fid)

    cache = {
        "version": version, "traj": traj, "support_mode": support_mode,
        "weights": np.array(weights, np.float32), "CR": float(cr), "budget_Kc": budget,
        "cshape": np.array(fastops.CSHAPE),
        "timesteps": np.array([f["ts"] for f in fids], np.int32),
        "tok_offsets": np.array(offsets, np.int64),
        "tok_coord": np.concatenate(tok_coord).astype(np.int16),   # (Ntot,5): h,l,s_wav,kx_wav,ky
        "tok_re": np.concatenate(tok_re).astype(np.float32),
        "tok_im": np.concatenate(tok_im).astype(np.float32),
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.savez(out_path, **cache)

    def agg(k):
        return float(np.mean([f[k] for f in fids]))

    ntok = np.diff(offsets)
    summary = {
        "version": version, "traj": traj, "n_snap": len(fids),
        "fidelity_all": {"df_psnr": agg("df_psnr"), "phi_psnr": agg("phi_psnr"),
                         "flux_relerr": agg("flux_relerr"),
                         "zonal_phi_ratio": agg("zonal_phi_ratio")},
        "ntok_mean": float(ntok.mean()), "ntok_min": int(ntok.min()),
        "ntok_max": int(ntok.max()), "out_path": out_path,
    }
    return summary, fids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="real", choices=["real", "semispectral"])
    ap.add_argument("--traj", default="iteration_13")
    ap.add_argument("--support", default="per_snapshot", choices=["per_snapshot", "shared"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cr", type=float, default=400.0)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = a.out or os.path.join(here, "cache", f"tokens_{a.traj}_{a.version}.npz")
    w = (a.alpha, a.beta, a.gamma)
    summ, _ = run(a.version, a.traj, a.support, w, out, limit=a.limit, cr=a.cr)
    print(json.dumps(summ, indent=1))
