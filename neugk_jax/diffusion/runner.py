"""Latent flow-matching runner.

Reuses the AE dataset setup, loads a trained AE (frozen), caches its latents over both
splits, keeps the latent tables on device and trains a DiT on Gaussian -> latent flow
matching with the batch latents gathered inside the step.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
from omegaconf import open_dict

from neugk_jax.dataset import precompute_latents
from neugk_jax.diffusion.flow_matching import dit_flow_loss
from neugk_jax.diffusion.latents import (
    latent_arrays,
    latent_cache_meta,
    latent_cache_path,
    load_precomputed_latents,
)
from neugk_jax.losses import part_weight
from neugk_jax.models.build import build_dit
from neugk_jax.training.checkpoint import resolve_checkpoint
from neugk_jax.training.runner import BaseRunner, conditioning_slots
from neugk_jax.utils import to_dict


def _same_path(a, b) -> bool:
    return Path(a).resolve() == Path(b).resolve()


def check_ae_dataset(ae_dataset: dict, dataset: dict) -> None:
    """Raise unless ``dataset`` preprocesses df as the AE run's saved ``ae_dataset`` section.

    Compares ``separate_zf``, ``offset``, the ``normalization`` of every field normalized in
    ``dataset`` and the resolved ``normalization_stats`` paths; warns when only one side
    sets ``normalization_stats``.
    """
    import warnings

    bad = []
    for k, default in (("separate_zf", False), ("offset", 0)):
        a, b = ae_dataset.get(k, default), dataset.get(k, default)
        if a != b:
            bad.append((k, a, b))
    ae_norm = ae_dataset.get("normalization") or {}
    for field, spec in (dataset.get("normalization") or {}).items():
        if ae_norm.get(field) != spec:
            bad.append((f"normalization.{field}", ae_norm.get(field), spec))
    a, b = ae_dataset.get("normalization_stats"), dataset.get("normalization_stats")
    if isinstance(a, str) and isinstance(b, str):
        if not _same_path(a, b):
            bad.append(("normalization_stats", a, b))
    elif (a is None) != (b is None):
        warnings.warn(
            f"ae normalization_stats={a!r}, diffusion normalization_stats={b!r}: "
            "cannot check that both normalize df with the same statistics"
        )
    if bad:
        lines = "\n".join(f"  dataset.{k}: ae={a!r} diffusion={b!r}" for k, a, b in bad)
        raise ValueError(f"diffusion dataset does not match the AE run's dataset:\n{lines}")


def load_autoencoder(
    path, *, resolution=None, dataset: dict | None = None, legacy=False, ds=None
):
    """AE of a run directory or checkpoint file; ``dataset`` is checked against the run's.

    ``legacy`` forces the doubled swin residual, else ``model.legacy_swin_shortcut`` decides.
    The stems of a species-axis AE take their grids from ``ds`` (see ``run_config``).
    """
    from neugk_jax.models.build import build_ae_from_config, run_config
    from neugk_jax.translate import load_or_translate

    ae_file = resolve_checkpoint(path)
    ae_cfg = to_dict(str(ae_file.parent / "config.yaml"))
    if dataset is not None:
        check_ae_dataset(ae_cfg.get("dataset") or {}, dataset)
    if ds is not None and (ae_cfg.get("model") or {}).get("stems"):
        ae_cfg = run_config(ae_cfg, ds)
    template = build_ae_from_config(
        ae_cfg, key=jr.PRNGKey(0), resolution=resolution, legacy_double_shortcut=legacy or None
    )
    return load_or_translate(template, str(ae_file))


@eqx.filter_jit
def encode_batch(ae, df):
    return jax.vmap(ae.encode)(df)


def parts_of(ds) -> dict:
    # the datasets of a mix, or the dataset itself
    return getattr(ds, "parts", None) or {None: ds}


class FlowEvaluators:
    """The flow-loss evaluator and the optional sampling evaluator, called together."""

    def __init__(self, loss, sampling=None):
        self.loss, self.sampling = loss, sampling

    def __call__(self, model, *, epoch: int):
        out, plots = self.loss(model, epoch=epoch)
        if self.sampling is not None:
            metrics, plots = self.sampling(model, epoch=epoch)
            out.update(metrics)
        return out, plots


class FlowMatchingRunner(BaseRunner):
    """Trains a DiT on the latent distribution of a frozen AE via flow matching."""

    val_metrics = ("avg_flux_rmse", "fm_loss")
    decoupled_wd = True

    def setup_data(self) -> None:
        cfg = self.cfg
        if cfg.get("ae_checkpoint") is None:
            raise ValueError("diffusion workflow requires ae_checkpoint")
        # the latents are normalized with the statistics the AE was trained with
        ae_stats = resolve_checkpoint(cfg.ae_checkpoint).parent / "normalization_stats"
        self.build_data("ae", stats_dir=str(ae_stats) if ae_stats.is_dir() else None)
        dcfg = to_dict(cfg.dataset)
        self.ae = load_autoencoder(
            cfg.ae_checkpoint,
            resolution=self.train_ds.resolution,
            dataset=dcfg,
            legacy=bool(cfg.get("ae_legacy_swin_shortcut", False)),
            ds=self.train_ds,
        )
        self.latent_shape = (*self.ae.bottleneck_grid_size, int(self.ae.bottleneck_dim))
        ae_file = resolve_checkpoint(cfg.ae_checkpoint)
        for split_ds, key in ((self.train_ds, "latents_cache_train"), (self.val_ds, "latents_cache_val")):
            for ds in parts_of(split_ds).values():
                meta = latent_cache_meta(
                    ds, ae_file, normalization_stats=dcfg.get("normalization_stats")
                )
                path = None if hasattr(split_ds, "parts") else dcfg.get(key)
                shape = self.part_latent_shape(ds)
                if path:
                    load_precomputed_latents(ds, path, latent_shape=shape, meta=meta)
                    continue
                cache = latent_cache_path(
                    ds, ds.split, cfg.ae_checkpoint, decouple_mu=dcfg.get("norm_decouple_mu", False)
                )
                precompute_latents(
                    ds,
                    encode_fn=lambda df, _c: encode_batch(self.ae, df),
                    cache_file=cache,
                    batch_size=self.tcfg.get("precompute_batch", 2),
                    meta=meta,
                    latent_shape=shape,
                )
        self.cond_slots = conditioning_slots(
            self.train_ds.conditions, list(cfg.model.get("conditioning") or [])
        )
        if cfg.get("latent_scale") is not None:
            self.latent_scale = float(cfg.latent_scale)
        else:
            parts = list(parts_of(self.train_ds).values())
            # mean latent variance over all parts, each weighted by its sample count
            var = sum(float(np.mean(p.latent_stats.var)) * len(p) for p in parts) / sum(map(len, parts))
            self.latent_scale = float(1.0 / np.sqrt(max(var, 1e-12)))
            with open_dict(cfg):
                cfg.latent_scale = self.latent_scale
        self.use_ot = bool(cfg.model.get("minibatch_ot", True))
        if self.dist.is_rank0:
            print(f"latent_scale = {self.latent_scale:.4f}")

    def checkpoint_meta(self) -> dict:
        return {"latent_scale": self.latent_scale}

    def build_model(self, key):
        return build_dit(self.cfg, self.ae, key=key)

    def part_latent_shape(self, ds) -> tuple:
        # latents of a species-axis AE have the species of the data
        if getattr(ds, "species_axis", False) and hasattr(self.ae, "stem_inputs"):
            return (int(ds.n_species), *self.latent_shape[1:])
        return self.latent_shape

    @property
    def mixed(self) -> bool:
        return hasattr(self.train_ds, "parts")

    def _tables(self, ds) -> dict:
        z, cond = latent_arrays(ds)
        if self.cond_slots is None:
            return {"latents": z, "cond": None}
        return {"latents": z, "cond": cond[:, self.cond_slots]}

    def step_context(self) -> dict:
        # a mix has one latent shape per part: its batches carry their latents
        return {} if self.mixed else self._tables(self.train_ds)

    def load_batch(self, ds, indices, read) -> dict:
        if not self.mixed:
            return {"idx": np.asarray(indices, np.int32)}
        samples = read(ds, indices)
        out = {
            "z": np.stack([np.asarray(s.df, np.float32) for s in samples]),
            "loss_weight": np.asarray([s.loss_weight for s in samples], np.float32),
        }
        if self.cond_slots is not None:
            out["c"] = np.stack([s.conditioning for s in samples])[:, self.cond_slots]
        return out

    def loss_fn(self, model, batch, key):
        if "z" in batch:
            z, cond = batch["z"], batch.get("c")
        else:
            idx, cond = batch["idx"], batch["cond"]
            z, cond = batch["latents"][idx], None if cond is None else cond[idx]
        loss = dit_flow_loss(
            model,
            z,
            cond,
            key,
            latent_scale=self.latent_scale,
            use_ot=self.use_ot,
            train=True,
        )
        return loss * part_weight(batch), {"fm_loss": loss}

    def make_evaluator(self):
        from neugk_jax.diffusion.eval import DiffusionEvaluator, FlowLossEvaluator

        loss = FlowLossEvaluator(
            self.cfg,
            tables=self._tables(self.val_ds),
            latent_scale=self.latent_scale,
            use_ot=self.use_ot,
            seed=self.seed,
            **self.evaluator_kwargs(),
        )
        if not self.vcfg.get("eval_sampling", False):
            return FlowEvaluators(loss)
        # sampled latents are scored against the df snapshots
        kwargs = {**self.evaluator_kwargs(), "val_ds": self.val_ds.with_mode("ae")}
        sampling = DiffusionEvaluator(
            self.cfg,
            autoencoder=self.ae,
            latent_scale=self.latent_scale,
            cond_slots=self.cond_slots,
            latent_shape=self.part_latent_shape(self.val_ds),
            **kwargs,
        )
        return FlowEvaluators(loss, sampling)
