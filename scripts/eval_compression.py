"""CLI: the compression benchmark (Table 1 steady state, transition windows) over test trajectories.

The (trajectory, timesteps) set and the iso-CR target come from the PINC-NF checkpoints in
``--ckpts`` (``nf_ckps_intdiag`` for the steady state, ``nf_ckps_transition`` for the transition
windows). Rows are written per (method, trajectory) and resume; ``eval1k_<LABEL>.json`` is rebuilt
at the end. Methods: ``nf``, ``nf-pinc``, ``pinn`` (``--pinn-ckpts``), the codecs ``zfp``, ``sz3``,
``wavelet``, ``pca``, ``jpeg2000`` at the neural-field CR, and autoencoders ``--ae LABEL=RUN[:CKPT]``
(a JAX run directory, or a torch ``.pth`` of the ``pinc_revival`` AE).

    python scripts/eval_compression.py --methods nf,nf-pinc,zfp --ckpts nf_ckps_intdiag \\
        --path /system/user/publicwork/galletti/pinc_revival_eval --gpus 0,1 --outdir out
    python scripts/eval_compression.py --ae PINC-AE=runs/pinc_ae:ckp.eqx --ckpts nf_ckps_transition \\
        --path ... --outdir out_transition --dump-diags
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CODEC_METHODS = ("zfp", "sz3", "wavelet", "pca", "jpeg2000")
NF_METHODS = ("nf", "nf-pinc", "pinn")


def labels_of(args) -> list[tuple[str, str]]:
    """``(method id, label)`` of the requested methods."""
    out = []
    for m in [m for m in args.methods.split(",") if m]:
        if m in CODEC_METHODS:
            out.append((m, m.upper()))
        elif m in NF_METHODS:
            out.append((m, "PINN" if m == "pinn" else m))
        else:
            raise SystemExit(f"unknown method {m!r}")
    out += [(f"ae:{spec}", spec.split("=", 1)[0]) for spec in args.ae]
    return out


def work_items(args):
    from neugk_jax.pinc.benchmark.reconstructors import NF_PREFIX, discover

    weights, cr = discover(args.ckpts, NF_PREFIX["nf-pinc"])
    trajs = sorted(weights, key=lambda s: int(s.split("_")[1]))
    if args.trajs:
        keep = set(args.trajs.split(","))
        trajs = [t for t in trajs if t in keep]
    return {t: sorted(weights[t]) for t in trajs}, cr


def build(method: str, args, nf_cr, probe_traj: str):
    from neugk_jax.pinc.benchmark import reconstructors as R

    if method in CODEC_METHODS:
        return R.Traditional(method.upper(), method, target_cr=float(nf_cr))
    if method in ("nf", "nf-pinc"):
        return R.NeuralField(method, R.discover(args.ckpts, R.NF_PREFIX[method])[0])
    if method == "pinn":
        return R.NeuralField("PINN", R.discover(args.pinn_ckpts, R.NF_PREFIX["nf-pinc"])[0])
    label, spec = method.removeprefix("ae:").split("=", 1)
    run, _, ckpt = spec.partition(":")
    model, dataset, slots = R.build_autoencoder(
        run, path=args.path, traj=probe_traj, ckpt=ckpt or None
    )
    return R.Autoencoder(label, model, dataset, slots)


def worker(args, shard: list[str]) -> None:
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    from neugk_jax.pinc.benchmark.runner import GroundTruth, evaluate, row_file, write_row

    ts_by_traj, nf_cr = work_items(args)
    methods = labels_of(args)
    gt = GroundTruth(args.path, shard)
    cache = {}
    for traj in shard:
        todo = [
            (m, lbl)
            for m, lbl in methods
            if args.overwrite or not os.path.exists(row_file(args.outdir, lbl, traj))
        ]
        if not todo:
            continue
        ts = ts_by_traj[traj]
        tgt = gt.trajectory(traj, ts)
        for m, lbl in todo:
            t0 = time.time()
            try:
                if m not in cache:
                    cache[m] = build(m, args, nf_cr, shard[0])
                row, diags = evaluate(cache[m], traj, ts, tgt)
                row.update(method=lbl, traj=traj, n_timesteps=len(ts))
            except Exception as e:
                traceback.print_exc()
                row, diags = dict(method=lbl, traj=traj, error=f"{type(e).__name__}: {e}"), None
            write_row(args.outdir, lbl, traj, row, diags if args.dump_diags else None)
            print(
                f"  {lbl} {traj}: psnr={row.get('psnr', float('nan')):.2f} "
                f"phi_psnr={row.get('phi_psnr', float('nan')):.2f} "
                f"eflux_l1={row.get('eflux_l1', float('nan')):.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="")
    ap.add_argument("--ae", action="append", default=[], help="LABEL=RUN[:CKPT], repeatable")
    ap.add_argument(
        "--ckpts", required=True, help="neural-field checkpoints defining the snapshot set"
    )
    ap.add_argument("--pinn-ckpts", default="nf_ckps_pinn")
    ap.add_argument("--path", required=True, help="evaluation data root")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gpus", default="0", help="one worker per entry (repeat an id for more)")
    ap.add_argument("--trajs", default="", help="comma list restricting the trajectories")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dump-diags", action="store_true", help="per-timestep spectra pkls")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--worker", type=int, default=-1, help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.outdir, "rows"), exist_ok=True)
    labels = [lbl for _, lbl in labels_of(args)]

    from neugk_jax.pinc.benchmark.runner import aggregate

    if not args.aggregate_only:
        ts_by_traj, nf_cr = work_items(args)
        trajs = list(ts_by_traj)
        gpus = [g for g in args.gpus.split(",") if g]
        if args.worker >= 0:
            worker(args, trajs[args.worker :: len(gpus)])
            return
        print(f"{len(trajs)} trajectories, methods {labels}, neural-field cr {nf_cr}x", flush=True)
        procs = [
            subprocess.Popen(
                [sys.executable, *sys.argv, "--worker", str(i)],
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": g,
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                },
            )
            for i, g in enumerate(gpus)
        ]
        for p in procs:
            p.wait()
    for label, agg in aggregate(args.outdir, labels).items():
        print(
            f"{label}: n={agg['n_traj']} (+{agg['n_err']} err) "
            + " ".join(
                f"{k}={agg[k][0]:.4g}"
                for k in ("psnr", "phi_psnr", "eflux_l1", "endpoint")
                if k in agg
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
