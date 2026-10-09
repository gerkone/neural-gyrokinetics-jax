"""CLI: representation interpolation of the neural fields and autoencoders over test trajectories.

For consecutive extremes ``a < b`` of a trajectory's neural fields in ``--nf-ckpts`` (trained with
``python main.py experiment=nf_interp``), the snapshot ``(a + b) // 2`` is predicted by linear
interpolation of the representations at the extremes: the neural-field weights, denormalized with
the mean of the extremes' z-scores, and the autoencoder latents (VQ-VAE latents before
quantization). ``Extremes`` scores the snapshot ``a`` and ``f (data)`` the data-space mean. PSNR and
L1 of f are those of the compression benchmark; ``--out`` gets the rows and their mean / std.

    python scripts/eval_interp.py --nf-ckpts nf_interp --path <eval data> \\
        --ae AE=runs/ae --ae PINC-AE=runs/pinc_ae:best.eqx --out interp.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neugk_jax.pinc.benchmark.reconstructors import NF_PREFIX, build_autoencoder, discover, load_nf
from neugk_jax.pinc.benchmark.runner import GroundTruth
from neugk_jax.pinc.neural_field import sample_field
from neugk_jax.pinc.nf_runner import snapshot_norm
from neugk_jax.utils import recombine_zf

NF_ROWS = {"NF": NF_PREFIX["nf"], "PINC-NF": NF_PREFIX["nf-pinc"]}


def df_scores(pred, gt) -> dict:
    pred, gt = jnp.asarray(pred, jnp.float32).reshape(gt.shape), jnp.asarray(gt)
    mse = jnp.mean((pred - gt) ** 2)
    return {
        "psnr": float(10 * jnp.log10(jnp.max(gt) ** 2 / mse)),
        "l1": float(jnp.mean(jnp.abs(pred - gt))),
    }


def mean_weights(a, b):
    """The neural field with the mean of the parameters of ``a`` and ``b``."""
    params = jax.tree_util.tree_map(
        lambda x, y: 0.5 * (x + y), eqx.filter(a, eqx.is_array), eqx.filter(b, eqx.is_array)
    )
    return eqx.combine(params, eqx.partition(a, eqx.is_array)[1])


def nf_midpoint(path_a: str, path_b: str, df_a, df_b):
    grid = tuple(int(n) for n in df_a.shape[1:])
    model = mean_weights(load_nf(path_a, grid), load_nf(path_b, grid))
    (scale_a, shift_a), (scale_b, shift_b) = snapshot_norm(df_a), snapshot_norm(df_b)
    return sample_field(model, grid) * 0.5 * (scale_a + scale_b) + 0.5 * (shift_a + shift_b)


@eqx.filter_jit
def _latent_midpoint(model, x_a, x_b, cond):
    z = 0.5 * (model.encode(x_a, cond) + model.encode(x_b, cond))
    return model.decode(model.bottleneck(z)[0], cond)["df"]


def ae_midpoint(model, ds, slots, ta: int, tb: int):
    """Decoded mean latent of snapshots ``ta`` and ``tb``, on the conditioning of ``ta``."""
    x_a, x_b = (jnp.asarray(ds.read_frame(0, t)["df"]) for t in (ta, tb))
    cond = None
    if slots is not None:
        cond = jnp.asarray(ds.conditioning(0, ds.metadata[0]["timesteps"][ta])[slots])
    pred = _latent_midpoint(model, x_a, x_b, cond)[None]
    pred = ds.norm.denormalize("df", pred, jnp.zeros((1,), jnp.int32))
    return recombine_zf(pred, axis=1)[0]


def aggregate(rows: list[dict]) -> dict:
    agg = defaultdict(dict)
    for key in [k for k in rows[0] if isinstance(rows[0][k], dict)]:
        for m in ("psnr", "l1"):
            v = np.array([r[key][m] for r in rows])
            agg[key][m] = [float(v.mean()), float(v.std())]
    return dict(agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nf-ckpts", required=True, help="neural fields at the extremes")
    ap.add_argument("--path", required=True, help="evaluation data root")
    ap.add_argument("--ae", action="append", default=[], help="LABEL=RUN[:CKPT], repeatable")
    ap.add_argument("--trajs", default="", help="comma list restricting the trajectories")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")

    nfs = {name: discover(args.nf_ckpts, prefix)[0] for name, prefix in NF_ROWS.items()}
    trajs = sorted(nfs["PINC-NF"], key=lambda s: int(s.split("_")[1]))
    if args.trajs:
        trajs = [t for t in trajs if t in set(args.trajs.split(","))]
    gt = GroundTruth(args.path, trajs)

    aes = {}
    for spec in args.ae:
        label, run = spec.split("=", 1)
        run, _, ckpt = run.partition(":")
        model, make, slots = build_autoencoder(
            run, path=args.path, traj=trajs[0], ckpt=ckpt or None
        )
        aes[label] = (model, make, slots)

    rows, pairs = [], {}
    for traj in trajs:
        ext = sorted(nfs["PINC-NF"][traj])
        pairs[traj] = [(a, b, (a + b) // 2) for a, b in zip(ext[:-1], ext[1:])]
        datasets = {label: make(traj) for label, (_, make, _) in aes.items()}
        for ta, tb, tc in pairs[traj]:
            f = {t: gt.ds.read_frame(gt.fid[traj], t)["df"] for t in (ta, tb, tc)}
            f = {t: jnp.asarray(np.asarray(v, np.float32)) for t, v in f.items()}
            row = {
                "traj": traj,
                "t": tc,
                "Extremes": df_scores(f[ta], f[tc]),
                "f (data)": df_scores(0.5 * (f[ta] + f[tb]), f[tc]),
            }
            for name, weights in nfs.items():
                pred = nf_midpoint(weights[traj][ta], weights[traj][tb], f[ta], f[tb])
                row[f"{name} (weights)"] = df_scores(pred, f[tc])
            for label, (model, _, slots) in aes.items():
                pred = ae_midpoint(model, datasets[label], slots, ta, tb)
                row[f"{label} (latents)"] = df_scores(pred, f[tc])
            rows.append(row)
            scores = {k: round(v["psnr"], 2) for k, v in row.items() if isinstance(v, dict)}
            print(traj, tc, scores, flush=True)

    agg = aggregate(rows)
    with open(args.out, "w") as fh:
        json.dump({"pairs": pairs, "trajectories": trajs, "agg": agg, "rows": rows}, fh, indent=1)
    for key, v in agg.items():
        print(f"{key:28s} & {v['psnr'][0]:.1f}$_{{\\pm {v['psnr'][1]:.1f}}}$ \\\\")


if __name__ == "__main__":
    main()
