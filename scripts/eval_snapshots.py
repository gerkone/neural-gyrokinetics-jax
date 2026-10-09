"""CLI: an AE / PINC-AE over the evaluation snapshots, with the ``neugk.pinc.eval`` metrics.

The snapshots are ``dataset.validation_trajectories`` x ``dataset.timesteps`` of the ``nf``
dataset config; the AE normalization comes from the ``pinc_revival`` dataset config (or
``--stats``). Per snapshot, as ``neugk.pinc.eval.metrics.ml_eval``: PSNR of the recombined
physical df and of the real potential, ``|sum Q_pred - sum Q_gt|``, df rel-L2. Reported as the
mean (and std) over trajectories of the per-trajectory means.

    python scripts/eval_snapshots.py --base <base ae .pth / jax run> \\
        [--lora <jax pinc run | torch peft .pth>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from hydra import compose, initialize_config_dir

from neugk_jax.dataset.factory import build_dataset
from neugk_jax.evaluate.base import GeometryCache, take_rows
from neugk_jax.evaluate.integrals import flux_integral
from neugk_jax.losses import per_sample_rel_l2
from neugk_jax.models.build import ae_conditioning, build_ae
from neugk_jax.pinc.eval import reconstruct, select_conditions
from neugk_jax.training.data import BatchLoader, stack_fields
from neugk_jax.training.ddp import init_distributed, shard_batch
from neugk_jax.training.runner import conditioning_slots
from neugk_jax.utils import recombine_zf


def _psnr(p, t):
    flat = lambda a: a.reshape(a.shape[0], -1)  # noqa: E731
    mse = jnp.mean((flat(p) - flat(t)) ** 2, axis=-1)
    return 10.0 * jnp.log10(jnp.max(flat(t), axis=-1) ** 2 / mse)


@eqx.filter_jit
def snapshot_metrics(model, batch, norm, geom):
    x, fids = batch["df"], batch["file_index"]
    pred = reconstruct(model, x, batch.get("conditioning"), inference=True)["df"]
    p = recombine_zf(norm.denormalize("df", pred, fids), axis=1)
    t = recombine_zf(norm.denormalize("df", x, fids), axis=1)
    g = take_rows(geom, fids)
    p_phi, (_, p_q, _) = jax.vmap(flux_integral)(g, p)
    t_phi, (_, t_q, _) = jax.vmap(flux_integral)(g, t)
    return {
        "psnr": _psnr(p, t),
        "phi_psnr": _psnr(p_phi, t_phi),
        "eflux_l1": jnp.abs(p_q - t_q),
        "df_rel_l2": per_sample_rel_l2(p, t),
    }


def load_model(cfg, ds, base: str, lora: str | None):
    from neugk_jax.training.checkpoint import load_checkpoint, resolve_checkpoint
    from neugk_jax.translate import (
        attach_ae_lora,
        import_ae_lora,
        load_or_translate,
        load_torch_state,
    )

    model = load_or_translate(
        build_ae(cfg, ds, key=jr.PRNGKey(0)), str(resolve_checkpoint(base)), strict=True
    )
    if lora is None:
        return model
    model = attach_ae_lora(model, cfg.model.peft.lora, key=jr.PRNGKey(1))
    if lora.endswith(".pth"):
        return import_ae_lora(model, load_torch_state(lora))[0]
    return load_checkpoint(resolve_checkpoint(lora), model).model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base ae: torch .pth, jax .eqx or run directory")
    p.add_argument(
        "--lora", default=None, help="pinc adapters: jax run directory / .eqx, or torch peft .pth"
    )
    p.add_argument(
        "--stats", default=None, help="normalization stats pickle (default: the pinc_revival one)"
    )
    p.add_argument("--batch-size", type=int, default=16, help="per device")
    p.add_argument("--out", default=None, help="json of the per-snapshot rows")
    p.add_argument(
        "--precision", default="highest", help="jax_default_matmul_precision (tf32 otherwise)"
    )
    args, overrides = p.parse_known_args()
    jax.config.update("jax_default_matmul_precision", args.precision)

    configs = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=configs, version_base=None):
        cfg = compose("main", overrides=["experiment=pinc_revival", *overrides])
        nf = compose("main", overrides=["experiment=nf"]).dataset
    dist = init_distributed()
    stats = args.stats or cfg.dataset.normalization_stats
    ds = build_dataset(
        cfg.dataset,
        split="val",
        dist=dist,
        trajectories=list(nf.validation_trajectories),
        cond_filters=None,
        offset=0,
        subsample=1,
        normalization_stats=stats,
    )
    enc, dec = ae_conditioning(cfg.model)
    slots = conditioning_slots(ds.conditions, set(enc) | set(dec))
    model = load_model(cfg, ds, args.base, args.lora)
    index = {v: k for k, v in ds.flat_index_to_file_and_tstep.items()}
    snaps = [
        (f, int(t)) for f in range(len(ds.files)) for t in nf.timesteps if (f, int(t)) in index
    ]
    geom = GeometryCache(ds).table()
    gbs = args.batch_size * dist.device_count
    loader = BatchLoader(workers=8, prefetch=2)

    def load(ds_, idx, read):
        return select_conditions(
            stack_fields(read(ds_, idx), ("df", "file_index", "conditioning")), slots
        )

    from neugk_jax.training.data import BatchPlan

    flat = [index[s] for s in snaps]
    plans = []
    for b in range(0, len(flat), gbs):
        sel = flat[b : b + gbs]
        n = len(sel)
        plans.append(
            BatchPlan(np.asarray(sel + [sel[-1]] * (gbs - n)), np.arange(gbs) < n, b // gbs)
        )
    rows, t0 = [], time.perf_counter()
    for plan, batch, _ in loader.iterate(ds, plans, load, lambda b: shard_batch(dist, b)):
        batch.pop("mask")
        m = jax.device_get(snapshot_metrics(model, batch, ds.norm, geom))
        for i in np.flatnonzero(plan.mask):
            f, t = ds.flat_index_to_file_and_tstep[int(plan.indices[i])]
            rows.append(
                {
                    "traj": Path(ds.files[f]).name.split("_ifft")[0],
                    "t": int(t),
                    **{k: float(v[i]) for k, v in m.items()},
                }
            )
    loader.close()
    dt = time.perf_counter() - t0
    trajs = sorted({r["traj"] for r in rows})
    rate = len(rows) / dt
    print(f"{len(rows)} snapshots, {len(trajs)} trajectories, {dt:.0f}s ({rate:.1f} snapshots/s)")
    for k in ("psnr", "phi_psnr", "eflux_l1", "df_rel_l2"):
        per = [np.mean([r[k] for r in rows if r["traj"] == tr]) for tr in trajs]
        print(f"  {k:10s} {np.mean(per):.3f} +- {np.std(per):.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
