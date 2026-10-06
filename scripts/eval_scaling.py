"""CLI: rate-distortion points of the compression benchmark for the scaling figure.

``nf``: the NF (density, ``mlp_*``) and NF-PINC (``int_mlp_*``) fields of ``<scale-dir>/<cr>x/`` and
``ae``: autoencoder runs (``--ae FAMILY@LABEL=RUN[:CKPT]``), one entry per variant pooled over the
trajectories, appended to ``<outdir>/scaling.pkl`` as ``{family: [{"cr", "name", metric...}]}``.
``codecs``: each (codec, target CR, trajectory) encoded per snapshot at iso-CR, rows
``{traj, t, raw, bytes, knob, psnr, l1, phi_l1, phi_psnr, eflux_l1, sec}`` in ``<outdir>/codecs.pkl``
keyed by ``(codec, target)``; CPU parallel. Every family and key is skipped when already stored.

    python scripts/eval_scaling.py nf --scale-dir nf_ckps_scale --crs 50,1168 --path ... --outdir out
    python scripts/eval_scaling.py ae --ae AE@AE_502=runs/scaling_ae_502/20261004 --path ... --outdir out
    python scripts/eval_scaling.py codecs --codecs zfp,sz3 --crs 50,1168 --workers 30 --path ... --outdir out
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_TRAJS = (
    "iteration_{0,8,20,36,41,48,55,65,73,79,85,94,100,104,108,113,117,121,125,130,134,138,142,146,"
    "151,155,159,163,168,172,176,180,185,189,193,197,202,206,210,214,218,223,227,231,235,240,244,"
    "248,252,257,261,265,269,274,278,282,286,291,295,299}"
)
NF_CRS = "10,50,200,500,1168,2000,5000,10000,50000"


def _store(path: str):
    return pickle.load(open(path, "rb")) if os.path.exists(path) else {}


def _save(path: str, obj) -> None:
    with open(path + ".tmp", "wb") as f:
        pickle.dump(obj, f)
    os.replace(path + ".tmp", path)


def nf_weights(cr_dir: str, prefix: str) -> dict[int, dict]:
    """``{cr: {traj: {t: path}}}`` of the ``<prefix>_<traj>_t<t>_x<cr>.pt`` checkpoints in ``cr_dir``."""
    import re

    rx = re.compile(rf"^{prefix}_(iteration_\d+)_t(\d+)_x(\d+)\.pt$")
    out: dict[int, dict] = defaultdict(lambda: defaultdict(dict))
    for f in os.listdir(cr_dir) if os.path.isdir(cr_dir) else []:
        m = rx.match(f)
        if m:
            out[int(m.group(3))][m.group(1)][int(m.group(2))] = os.path.join(cr_dir, f)
    return {cr: dict(w) for cr, w in out.items()}


def run_families(args, families: dict) -> None:
    """Evaluate ``{family: [reconstructor]}`` and append the entries to ``scaling.pkl``."""
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    from neugk_jax.pinc.benchmark.runner import GroundTruth, scaling_entry

    out = os.path.join(args.outdir, "scaling.pkl")
    gt = GroundTruth(args.path, args.trajs)
    for fam, recs in families.items():
        store = _store(out)
        done = {e["name"] for e in store.get(fam, [])}
        for rec in recs:
            if rec.name in done:
                continue
            t0 = time.time()
            try:
                entry = scaling_entry(rec, args.trajs, args.timesteps, gt)
            except Exception as e:
                print(f"[scaling] {fam} {rec.name}: skipped ({type(e).__name__}: {e})", flush=True)
                continue
            store = _store(out)
            store[fam] = sorted(store.get(fam, []) + [entry], key=lambda e: e["cr"])
            _save(out, store)
            print(
                f"[scaling] {fam} {rec.name}: cr {entry['cr']:.1f} psnr {entry['psnr']:.2f} "
                f"eflux_l1 {entry['eflux_l1']:.3f} ({time.time() - t0:.0f}s)",
                flush=True,
            )


def cmd_nf(args) -> None:
    from neugk_jax.pinc.benchmark.reconstructors import NeuralField

    families = {}
    for fam, prefix in (("NF", "mlp"), ("NF-PINC", "int_mlp")):
        recs = []
        for cr in args.crs:
            for ncr, w in sorted(
                nf_weights(os.path.join(args.scale_dir, f"{cr}x"), prefix).items()
            ):
                keep = {t: w[t] for t in args.trajs if t in w}
                recs.append(NeuralField(f"NF_x{ncr}", keep))
        if recs:
            families[fam] = recs
    run_families(args, families)


def cmd_ae(args) -> None:
    from neugk_jax.pinc.benchmark.reconstructors import Autoencoder, build_autoencoder

    families = defaultdict(list)
    for spec in args.ae:
        fam_label, run = spec.split("=", 1)
        fam, label = fam_label.split("@", 1)
        run, _, ckpt = run.partition(":")
        model, dataset, slots = build_autoencoder(
            run, path=args.path, traj=args.trajs[0], ckpt=ckpt or None
        )
        families[fam].append(Autoencoder(label, model, dataset, slots))
    run_families(args, dict(families))


def codec_job(job):
    codec, target, traj, timesteps, path = job
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp

    from neugk_jax.pinc.benchmark.codecs import encode_at_cr
    from neugk_jax.pinc.benchmark.metrics import snapshot_fields, snapshot_metrics
    from neugk_jax.pinc.benchmark.runner import GroundTruth

    gt = GroundTruth(path, [traj]).trajectory(traj, timesteps)
    rows, warm = [], None
    for t, df, g in zip(timesteps, gt.frames, gt.fields):
        t0 = time.time()
        recon, size, warm = encode_at_cr(codec, df, float(target), warm=warm)
        p = snapshot_fields(gt.geom_t, jnp.asarray(recon), gt.ds)
        m = snapshot_metrics(jnp.asarray(recon), jnp.asarray(df), p, g)
        r = dict(traj=f"{traj}.h5", t=int(t), raw=int(df.nbytes), bytes=int(size), knob=warm)
        r.update({k: float(m[k]) for k in ("psnr", "l1", "phi_l1", "phi_psnr", "eflux_l1")})
        r["sec"] = time.time() - t0
        rows.append(r)
    return codec, target, rows


def cmd_codecs(args) -> None:
    import multiprocessing as mp

    out = os.path.join(args.outdir, "codecs.pkl")
    res = _store(out)
    n = len(args.trajs) * len(args.timesteps)
    jobs = [
        (c, cr, tr, args.timesteps, args.path)
        for c in args.codecs.split(",")
        for cr in args.crs
        if len(res.get((c, cr), [])) < n
        for tr in args.trajs
    ]
    print(f"{len(jobs)} jobs on {args.workers} workers", flush=True)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    acc: dict = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for c, cr, rows in pool.imap_unordered(codec_job, jobs):
            acc.setdefault((c, cr), []).extend(rows)
            if len(acc[(c, cr)]) == n:
                res[(c, cr)] = acc.pop((c, cr))
                raw = sum(r["raw"] for r in res[(c, cr)])
                nb = sum(r["bytes"] for r in res[(c, cr)])
                print(f"[{time.time() - t0:7.0f}s] {c}@{cr}: pooled cr {raw / nb:.1f}", flush=True)
                _save(out, res)


def main():
    from neugk_jax.dataset.backend import expand_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("nf", "ae", "codecs"))
    ap.add_argument("--path", required=True, help="evaluation data root")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trajs", default=TEST_TRAJS, help="trajectory spec (ranges allowed)")
    ap.add_argument("--timesteps", default="110,160,210,260")
    ap.add_argument("--crs", default=NF_CRS)
    ap.add_argument("--scale-dir", default="nf_ckps_scale", help="nf: <cr>x/ checkpoint dirs")
    ap.add_argument(
        "--ae", action="append", default=[], help="ae: FAMILY@LABEL=RUN[:CKPT], repeatable"
    )
    ap.add_argument("--codecs", default="wavelet,jpeg2000,pca,sz3,zfp")
    ap.add_argument("--workers", type=int, default=16, help="codecs: cpu worker processes")
    args = ap.parse_args()
    args.trajs = [t.removesuffix(".h5") for t in expand_spec(args.trajs)]
    args.timesteps = [int(t) for t in args.timesteps.split(",") if t]
    args.crs = [int(c) for c in args.crs.split(",") if c]
    os.makedirs(args.outdir, exist_ok=True)
    {"nf": cmd_nf, "ae": cmd_ae, "codecs": cmd_codecs}[args.mode](args)


if __name__ == "__main__":
    main()
