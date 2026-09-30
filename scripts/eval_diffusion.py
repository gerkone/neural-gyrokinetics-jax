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
                   help="stride over val samples (the paper uses 10)")
    p.add_argument("--latent-scale", type=float, default=None,
                   help="latent_scale the DiT was TRAINED with (the runner prints it). "
                        "Omitted → estimated from this split, which mis-scales the sampler.")
    p.add_argument("--n-samples", type=int, default=1,
                   help="stochastic samples per condition (ensemble size for the flux UQ)")
    args = p.parse_args()

    # heavy imports after argparse so --help stays instant
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np
    from omegaconf import OmegaConf

    from neugk_jax.dataset.factory import build_dataset
    from neugk_jax.diffusion.runner import encode_batch
    from neugk_jax.evaluate import DiffusionEvaluator
    from neugk_jax.training.runner import conditioning_slots
    from neugk_jax.translate import (
        build_ae_from_config,
        build_dit_from_config,
        load_or_translate,
    )

    cfg = OmegaConf.load(args.config)
    dcfg = cfg.get("dataset") or OmegaConf.create({})
    with OmegaConf.read_write(dcfg):
        dcfg.path = args.data_path
        dcfg.backend = "numpy"

    # ae config lives next to the ae checkpoint, same convention as FlowMatchingRunner
    ae_cfg = os.path.join(os.path.dirname(args.ae_ckpt), "config.yaml")
    ae = load_or_translate(build_ae_from_config(ae_cfg, key=jr.PRNGKey(0)), args.ae_ckpt)
    dit = load_or_translate(build_dit_from_config(args.config, ae, key=jr.PRNGKey(0)), args.dit_ckpt)
    print(f"loaded AE + DiT: latent_shape={tuple(dit.latent_shape)}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in SPLIT_TRAJECTORIES]
    if unknown:
        raise SystemExit(f"unknown splits {unknown}; choose from {sorted(SPLIT_TRAJECTORIES)}")

    latent_scale = None
    for split in splits:
        with OmegaConf.read_write(dcfg):
            dcfg.validation_trajectories = SPLIT_TRAJECTORIES[split]
        ds = build_dataset(dcfg, split="val", mode="ae")
        print(f"[{split}] {len(ds.files)} trajectories, {len(ds)} samples")

        if latent_scale is None:
            if args.latent_scale is not None:
                latent_scale = float(args.latent_scale)
                print(f"latent_scale = {latent_scale:.4f} (from --latent-scale, as trained)")
            else:
                # 1 / std of the ae latents, estimated from a few samples of this split
                idx = np.linspace(0, len(ds) - 1, num=min(8, len(ds)), dtype=int)
                z = encode_batch(ae, jnp.stack([jnp.asarray(ds[int(i)].df) for i in idx]))
                latent_scale = float(1.0 / np.sqrt(max(float(jax.numpy.var(z)), 1e-12)))
                print(f"latent_scale = {latent_scale:.4f} (ESTIMATED from this split; pass "
                      "--latent-scale to match training)")

        evaluator = DiffusionEvaluator(
            cfg, val_ds=ds, autoencoder=ae, latent_scale=latent_scale,
            cond_slots=conditioning_slots(ds.conditions, list(cfg.model.get("conditioning") or [])),
            batch_size=args.batch_size, steps=args.steps, n_samples=args.n_samples,
            subsample=args.subsample,
        )
        metrics, plots = evaluator(dit, epoch=0)

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
