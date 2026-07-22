"""CLI: ID/OOD sampling eval for the latent flow-matching DiT.

Runs ``DiffusionEvaluator`` over the canonical ID split
(``iteration_{8,115,131,148,235,262}``) and the OOD split
(``ood_iteration_{0-4}``), printing per-split metrics and saving the
cross-section panels + ``avg_flux_UQ`` scatter to ``<output>/<split>/``.
See PARITY.md "ID / OOD eval script".
"""

from __future__ import annotations

import argparse
import json
import os
import re

SPLIT_TRAJECTORIES = {
    "id": "iteration_{8,115,131,148,235,262}",
    "ood": "ood_iteration_{0-4}",
}


def _save_plot(obj, path: str) -> None:
    # evaluator plots are wandb.Image when wandb is installed, bare figures otherwise
    img = getattr(obj, "image", None)
    if img is not None:
        img.save(path)
    elif hasattr(obj, "savefig"):
        obj.savefig(path, bbox_inches="tight", dpi=120)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae-ckpt", required=True, help="translated AE checkpoint (.eqx, or torch .pth)")
    p.add_argument("--dit-ckpt", required=True, help="translated DiT checkpoint (.eqx, or torch .pth)")
    p.add_argument("--config", required=True, help="DIFF_FLOW config.yaml (dataset + model.vit)")
    p.add_argument("--data-path", required=True, help="preprocessed cyclone dataset root")
    p.add_argument("--splits", default="id,ood", help="comma-separated subset of id,ood")
    p.add_argument("--output", required=True, help="directory for per-split metrics + panels")
    p.add_argument("--steps", type=int, default=50, help="euler sampling steps")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--subsample", type=int, default=1,
                   help="stride over val samples (upstream val_subsample; the paper uses 10)")
    args = p.parse_args()

    # heavy imports after argparse so --help stays instant
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np
    import yaml

    from neugk_jax.dataset import CycloneDataset, NumpyBackend
    from neugk_jax.diffusion.flow_matching import euler_sample
    from neugk_jax.evaluate import DiffusionEvaluator
    from neugk_jax.translate import build_ae_from_config, build_dit_from_config, load_or_translate

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg.get("dataset", {}) or {}

    # AE config lives next to the AE checkpoint — same convention as FlowMatchingRunner
    ae_cfg = os.path.join(os.path.dirname(args.ae_ckpt), "config.yaml")
    ae = load_or_translate(build_ae_from_config(ae_cfg, key=jr.PRNGKey(0)), args.ae_ckpt)
    dit = load_or_translate(build_dit_from_config(args.config, ae, key=jr.PRNGKey(0)), args.dit_ckpt)
    latent_shape = tuple(dit.latent_shape)
    print(f"loaded AE + DiT: latent_shape={latent_shape}")

    common = dict(
        path=args.data_path,
        split="val",
        fields_to_load=tuple(dcfg.get("input_fields", ("df",))),
        conditions=tuple(dcfg.get("conditions", ("itg", "dg", "s_hat", "q"))),
        mode="diff",
        backend=NumpyBackend(),
        separate_zf=dcfg.get("separate_zf", False),
        normalization=dcfg.get("normalization"),
        normalization_scope=dcfg.get("normalization_scope", "dataset"),
        normalization_stats=dcfg.get("normalization_stats"),
        offset=dcfg.get("offset", 0),
    )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in SPLIT_TRAJECTORIES]
    if unknown:
        raise SystemExit(f"unknown splits {unknown}; choose from {sorted(SPLIT_TRAJECTORIES)}")

    latent_scale = None
    for split in splits:
        ds = CycloneDataset(trajectories=SPLIT_TRAJECTORIES[split], **common)
        print(f"[{split}] {len(ds.files)} trajectories, {len(ds)} samples")

        if latent_scale is None:
            # 1 / std of the AE latents — mirrors the runner's latent_scale, estimated
            # from a handful of encoded samples spread over the first split
            idx = np.linspace(0, len(ds) - 1, num=min(8, len(ds)), dtype=int)
            z = jax.vmap(lambda x: ae.encode(x)[0])(
                jnp.stack([jnp.asarray(ds[int(i)].df) for i in idx])
            )
            latent_scale = float(1.0 / np.sqrt(max(float(np.var(np.asarray(z))), 1e-12)))
            print(f"latent_scale = {latent_scale:.4f}")

        def sample_fn(*, key, batch, cond=None, steps=50):
            # euler-integrate the DiT velocity field, then decode — FlowMatchingRunner.sample
            latents = euler_sample(
                lambda x, t, c: dit(x, t, c),
                key=key, shape=(batch, *latent_shape),
                cond=cond, steps=steps, latent_scale=latent_scale,
            )
            return jax.vmap(ae.decode)(latents)

        evaluator = DiffusionEvaluator(
            cfg, val_ds=ds, autoencoder=ae, sample_fn=sample_fn, is_rank0=True,
        )
        metrics, plots = evaluator(
            dit, epoch=0,
            batch_size=args.batch_size,
            n_steps=args.steps,
            val_subsample=args.subsample,
        )

        out_dir = os.path.join(args.output, split)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{split}] " + "  ".join(f"{k}={v:.6g}" for k, v in sorted(metrics.items())))
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        for name, plot in plots.items():
            fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") + ".png"
            _save_plot(plot, os.path.join(out_dir, fname))
        print(f"[{split}] wrote metrics + {len(plots)} panels to {out_dir}")


if __name__ == "__main__":
    main()
