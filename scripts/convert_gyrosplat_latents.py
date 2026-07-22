"""One-time conversion of fitted gyrosplat latents (.pt) into the training cache.

Reads ``<data>/iteration_N/latent_*.pt`` (torch) and writes per-trajectory numpy
caches plus global channel statistics:

    <cache>/iteration_N/atoms.npy      (T, 1597, 17) f32, pack() channel order
    <cache>/iteration_N/zfstats.npy    (T, 4)  [zonal_mean, zonal_std, fluc_mean, fluc_std]
    <cache>/iteration_N/params.npy     (4,)    [itg, dg, q, s_hat]
    <cache>/iteration_N/flux.npy       full heat-flux timeseries
    <cache>/iteration_N/meta.json      timesteps + fit quality per step
    <cache>/bins.npy                   (1597,) int32 per-slot carrier mode m
    <cache>/channel_stats.npz          token normalization stats (see TokenStats)

The bank structure (750 envelope atoms + 121 tied groups x 7 carriers at bins 7..13)
is hard-asserted per file; ky is stored raw — the cache keeps physical units, all
normalization happens in the dataset via TokenStats. This is the only
torch-dependent step of the gyrosplat pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

N_ENV = 750
TIED_GROUPS = 121
TIED_K = 7
CARRIER_BINS = list(range(7, 14))
N_ATOMS = N_ENV + TIED_GROUPS * TIED_K
TWO_PI = 2.0 * np.pi


class StructureMismatch(Exception):
    """Trajectory fitted with a different bank recipe (e.g. envelope-only n=1000)."""


def _expected_bins() -> np.ndarray:
    # graft_tied layout: envelope block, then groups of 7 rows cycling bins 7..13
    tail = np.tile(np.array(CARRIER_BINS, dtype=np.int32), TIED_GROUPS)
    return np.concatenate([np.zeros(N_ENV, dtype=np.int32), tail])


def convert_trajectory(traj_dir: str, out_dir: str) -> dict:
    """Convert one iteration dir; returns summary stats for the global pass."""
    import torch

    files = sorted(f for f in os.listdir(traj_dir) if f.startswith("latent_"))
    assert files, f"no latents in {traj_dir}"
    atoms, meta_rows = [], []
    bins_ref = _expected_bins()
    for f in files:
        sd = torch.load(os.path.join(traj_dir, f), map_location="cpu", weights_only=False)
        flat = torch.cat(
            [sd["mu"], sd["L_phys_raw"], sd["L_vel_raw"], sd["amps"], sd["ky"][:, None]],
            dim=1,
        ).numpy().astype(np.float32)
        if flat.shape != (N_ATOMS, 17):
            raise StructureMismatch(f"{traj_dir}/{f}: shape {flat.shape}")
        m = np.round(sd["ky"].numpy() / TWO_PI).astype(np.int32)
        if not np.array_equal(m, bins_ref):
            raise StructureMismatch(f"bank structure violated in {traj_dir}/{f}")
        atoms.append(flat)
        meta_rows.append(
            {
                "t": int(sd["t"]),
                "psnr_phi": float(sd.get("psnr_phi", np.nan)),
                "flux_relL1": float(sd.get("flux_relL1", np.nan)),
            }
        )
    atoms = np.stack(atoms)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "atoms.npy"), atoms)
    for name in ("params.npy", "flux.npy"):
        src = os.path.join(traj_dir, name)
        if os.path.exists(src):
            np.save(os.path.join(out_dir, name), np.load(src))
    # some fits are incomplete (fewer latents than zfstats rows): align zf rows
    # to each latent's stored timestep (t - first fitted timestep)
    zf_full = np.load(os.path.join(traj_dir, "zfstats.npy")).astype(np.float32)
    ts = [row["t"] for row in meta_rows]
    t0 = min(ts)
    idx = [t - t0 for t in ts]
    assert max(idx) < zf_full.shape[0], (traj_dir, max(ts), zf_full.shape)
    zf = zf_full[idx]
    np.save(os.path.join(out_dir, "zfstats.npy"), zf)
    assert zf.shape[0] == atoms.shape[0], (zf.shape, atoms.shape)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump({"traj": os.path.basename(traj_dir), "steps": meta_rows}, fh, indent=1)
    return {"atoms": atoms, "zfstats": zf}


def compute_channel_stats(all_atoms: np.ndarray, all_zf: np.ndarray) -> dict:
    """Token normalization constants (see gyrosplats.normalize.TokenStats).

    mu channels use the fixed affine map (mean 0.5, std 0.5 -> 2x-1, isotropic);
    the ky channel stores the drift stats around the per-slot bin.
    """
    flat = all_atoms.reshape(-1, 17).astype(np.float64)
    mean = flat.mean(0)
    std = flat.std(0)
    mean[:5], std[:5] = 0.5, 0.5
    bins = _expected_bins().astype(np.float64)
    dky = (all_atoms[..., 16] - TWO_PI * bins[None, :]).reshape(-1)
    mean[16], std[16] = dky.mean(), max(dky.std(), 1e-8)
    stat_vals = np.stack(
        [all_zf[:, 0], np.log(all_zf[:, 1]), np.log(all_zf[:, 3])], axis=1
    )
    return {
        "mean": mean.astype(np.float32),
        "std": np.maximum(std, 1e-8).astype(np.float32),
        "stat_mean": stat_vals.mean(0).astype(np.float32),
        "stat_std": np.maximum(stat_vals.std(0), 1e-8).astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="/restricteddata/ukaea/gyrokinetics/gyrosplats/data")
    ap.add_argument("--cache", required=True)
    ap.add_argument(
        "--trajs",
        nargs="*",
        default=None,
        help="iteration names; default: _train_trajs_valid.json + all readable",
    )
    ap.add_argument("--skip-unreadable", action="store_true")
    ap.add_argument(
        "--fold-stats",
        action="store_true",
        help="bake each snapshot's fluc_std into the amps (exact for flux/spectra: "
        "ky=0 carries zero flux; zonal amplitude approximated by the fluc scale). "
        "decoding then needs NO per-snapshot stats",
    )
    args = ap.parse_args()

    if args.trajs:
        names = args.trajs
    else:
        valid = os.path.join(args.data, "_train_trajs_valid.json")
        with open(valid) as fh:
            names = [f"iteration_{i}" for i in json.load(fh)]

    os.makedirs(args.cache, exist_ok=True)
    np.save(os.path.join(args.cache, "bins.npy"), _expected_bins())

    atoms_all, zf_all, converted = [], [], []
    for name in names:
        traj_dir = os.path.join(args.data, name)
        try:
            out = convert_trajectory(traj_dir, os.path.join(args.cache, name))
        except PermissionError:
            if args.skip_unreadable:
                print(f"skip (unreadable): {name}")
                continue
            raise
        except StructureMismatch as e:
            # a handful of trajectories carry envelope-only banks from an older
            # fitting recipe — exclude rather than crash, and report them
            print(f"skip (structure mismatch): {name} — {e}")
            continue
        if args.fold_stats:
            # amps live in per-snapshot fit space; fold the fluctuation scale in
            fs = out["zfstats"][:, 3:4]  # (T, 1)
            out["atoms"][:, :, 14:16] *= fs[:, :, None]
            np.save(os.path.join(args.cache, name, "atoms.npy"), out["atoms"])
        atoms_all.append(out["atoms"])
        zf_all.append(out["zfstats"])
        converted.append(name)
        print(f"converted {name}: atoms {out['atoms'].shape}")

    assert converted, "nothing converted"
    stats = compute_channel_stats(np.concatenate(atoms_all), np.concatenate(zf_all))
    np.savez(os.path.join(args.cache, "channel_stats.npz"), **stats)
    with open(os.path.join(args.cache, "converted.json"), "w") as fh:
        json.dump(converted, fh, indent=1)
    print(f"channel_stats over {len(converted)} trajs -> {args.cache}/channel_stats.npz")


if __name__ == "__main__":
    main()
