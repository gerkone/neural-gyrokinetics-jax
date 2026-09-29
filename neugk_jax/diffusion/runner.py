"""Latent flow-matching training pipeline.

Loads a translated/trained AE (frozen), precomputes latents over the
training set, then trains a DiT on Gaussian → latent flow matching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.dataset import CycloneDataset, KvikIOBackend, NumpyBackend, precompute_latents
from neugk_jax.diffusion.dit import DiT
from neugk_jax.diffusion.flow_matching import (
    euler_sample,
    fm_forward_loss,
)
from neugk_jax.diffusion.latents import latent_cache_path, load_precomputed_latents
from neugk_jax.models.utils import trainable_mask
from neugk_jax.training.ddp import global_batch_size, process_batch_indices, replicate, shard_batch
from neugk_jax.training.runner import BaseRunner, build_optimizer, train_update
from neugk_jax.training.schedulers import warmup_cosine
from neugk_jax.translate import force_f32


def resolve_ae_checkpoint(path) -> Path:
    """AE checkpoint file for a run directory (``best.eqx``, else ``best.pth``) or a file path."""
    p = Path(path)
    if p.is_dir():
        for name in ("best.eqx", "best.pth"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"no best.eqx / best.pth in {p}")
    return p


class FlowMatchingRunner(BaseRunner):
    """Trains a DiT to model the latent distribution via flow matching."""

    dataset_cls = CycloneDataset
    val_metrics = ("avg_flux_rmse", "fm_loss")

    def _dataset_kwargs(self) -> dict:
        return {}

    def setup_data(self) -> None:
        cfg = self.cfg
        # conditioning is gyroswin-specific; latent diffusion stays unconditional
        if cfg.model.get("conditioning") not in (None, [], ()):
            raise ValueError(
                "`model.conditioning` is set but the diffusion workflow does not accept "
                "scalar conditioning at the model level. Drop it from the config or "
                "switch to workflow=gyroswin."
            )
        ae_path = cfg.ae_checkpoint
        if ae_path is None:
            raise ValueError("diffusion workflow requires ae_checkpoint")
        from neugk_jax.translate import build_ae_from_config, load_or_translate
        ae_file = resolve_ae_checkpoint(ae_path)
        ae_template = build_ae_from_config(str(ae_file.parent / "config.yaml"), key=jr.PRNGKey(0))
        # .eqx loads directly; a torch .pth is translated on the fly
        self.ae = load_or_translate(ae_template, str(ae_file))

        backend = (
            KvikIOBackend(rank=self.dist.local_rank)
            if cfg.dataset.get("backend", "numpy") == "kvikio"
            else NumpyBackend()
        )
        common = dict(
            path=cfg.dataset.path,
            fields_to_load=tuple(cfg.dataset.get("input_fields", ("df",))),
            conditions=tuple(cfg.dataset.get("conditions", ("itg", "dg", "s_hat", "q"))),
            mode="ae",
            backend=backend,
            separate_zf=cfg.dataset.get("separate_zf", False),
            normalization=cfg.dataset.get("normalization"),
            normalization_scope=cfg.dataset.get("normalization_scope", "dataset"),
            normalization_stats=cfg.dataset.get("normalization_stats"),
            offset=cfg.dataset.get("offset", 0),
            lightweight_metadata=cfg.dataset.get("lightweight_metadata", False),
        )
        extra = self._dataset_kwargs()
        # the cond filters fix the file list, hence the fid ordering a latent cache uses
        self.train_ds = self.dataset_cls(
            split="train",
            trajectories=cfg.dataset.training_trajectories,
            cond_filters=self._omegaconf_to_dict(cfg.dataset.get("training_cond_filters")),
            **common,
            **extra,
        )
        self.val_ds = self.dataset_cls(
            split="val",
            trajectories=cfg.dataset.validation_trajectories,
            cond_filters=self._omegaconf_to_dict(cfg.dataset.get("eval_cond_filters")),
            **common,
            **extra,
        )

        # encode every sample once so training is just mse on cached latents
        encode = eqx.filter_jit(lambda ae, df: jax.vmap(lambda x: ae.encode(x)[0])(df))

        def encode_fn(df_batch, cond_batch):
            return encode(self.ae, df_batch)

        latent_shape = (*self.ae.bottleneck_grid_size, int(self.ae.bottleneck_dim))
        for ds, key in ((self.train_ds, "latents_cache_train"), (self.val_ds, "latents_cache_val")):
            path = cfg.dataset.get(key)
            if path:
                load_precomputed_latents(ds, path, latent_shape=latent_shape)
                if self.dist.is_rank0:
                    print(f"loaded {len(ds.precomputed_latents)} {ds.split} latents from {path}")
            else:
                cache = latent_cache_path(
                    ds, ds.split, ae_path, decouple_mu=cfg.dataset.get("norm_decouple_mu", False),
                    timestep_std_filter=cfg.dataset.get("timestep_std_filter"),
                )
                precompute_latents(ds, encode_fn=encode_fn, cache_file=cache,
                                   batch_size=cfg.training.get("precompute_batch", 2))

        # 1 / sqrt(mean variance)
        var = self.train_ds.latent_stats.var
        self.latent_scale = float(1.0 / np.sqrt(max(float(np.mean(var)), 1e-12)))
        if self.dist.is_rank0:
            print(f"latent_scale = {self.latent_scale:.4f}")

    def setup_components(self) -> None:
        cfg = self.cfg
        mcfg = cfg.model
        key = jr.PRNGKey(getattr(cfg, "seed", 0))
        grid = tuple(self.ae.bottleneck_grid_size)
        z_dim = int(self.ae.bottleneck_dim)
        self.latent_shape = (*grid, z_dim)
        self.model = DiT(
            space=len(grid),
            z_dim=z_dim,
            dim=mcfg.get("latent_dim", 512),
            grid_size=grid,
            depth=mcfg.vit.get("depth", 4),
            num_heads=mcfg.vit.get("num_heads", 8),
            n_cond=len(cfg.dataset.get("conditions", [])),
            key=key,
            mlp_ratio=mcfg.vit.get("mlp_ratio", 4.0),
            drop_path=mcfg.vit.get("drop_path", 0.0),
        )
        self.model = force_f32(self.model)
        steps_per_epoch = max(1, len(self.train_ds) // cfg.training.batch_size)
        total = cfg.training.n_epochs * steps_per_epoch
        self.schedule = warmup_cosine(
            peak_lr=cfg.training.learning_rate,
            total_steps=total,
            steps_per_epoch=steps_per_epoch,
            n_epochs=cfg.training.n_epochs,
            min_lr=cfg.training.get("final_learning_rate", 1e-6),
        )
        self.optimizer = build_optimizer(self.schedule, cfg.training, self.model, decoupled=True)
        self.trainable = trainable_mask(self.model)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, self.trainable))
        self.model = replicate(self.dist, self.model)
        self.opt_state = replicate(self.dist, self.opt_state)
        self.use_ot = bool(cfg.model.get("minibatch_ot", True))

    @eqx.filter_jit
    def _train_step(self, model, opt_state, latents, cond, key):
        fm_key, drop_key = jr.split(key)

        def loss_fn(m):
            def fwd(x, t, *rest):
                *c, k = rest
                return m(x, t, c[0] if c else None, key=k, inference=False)
            return fm_forward_loss(fwd, latents, cond, key=fm_key,
                                   latent_scale=self.latent_scale,
                                   use_ot=self.use_ot, dropout_key=drop_key)
        return train_update(model, opt_state, loss_fn, self.optimizer, self.trainable)

    def train_epoch(self, epoch: int, key) -> dict:
        cfg = self.cfg
        bs = global_batch_size(self.dist, cfg.training.batch_size)
        n = len(self.train_ds)
        idx_key, key = jr.split(key)
        idx = np.asarray(jr.permutation(idx_key, n))
        losses = []
        for start in range(0, n - bs + 1, bs):
            local = process_batch_indices(self.dist, idx[start:start + bs])
            samples = [self.train_ds[int(i)] for i in local]
            z = np.stack([np.asarray(s.df) for s in samples])
            cond = (
                np.stack([np.asarray(s.conditioning) for s in samples])
                if samples[0].conditioning is not None
                else None
            )
            z, cond = shard_batch(self.dist, (z, cond))
            step_key, key = jr.split(key)
            self.model, self.opt_state, loss = self._train_step(
                self.model, self.opt_state, z, cond, step_key,
            )
            losses.append(float(loss))
        return {"loss": sum(losses) / max(len(losses), 1)}

    def evaluate(self, epoch: int) -> dict:
        from neugk_jax.evaluate import DiffusionEvaluator

        def _sample(*, key, batch, cond=None, steps=50):
            return self.sample(key=key, batch=batch, cond=cond, steps=steps)

        # for cheap eval we also report the fm training-loss on the val set
        cfg = self.cfg
        bs = cfg.training.batch_size
        n = min(len(self.val_ds), bs * 4)
        losses = []
        key = jr.PRNGKey(epoch)
        for start in range(0, n - bs + 1, bs):
            samples = [self.val_ds[i] for i in range(start, start + bs)]
            z = jnp.stack([jnp.asarray(s.df) for s in samples])
            cond = (
                jnp.stack([jnp.asarray(s.conditioning) for s in samples])
                if samples[0].conditioning is not None
                else None
            )
            step_key, key = jr.split(key)
            losses.append(float(fm_forward_loss(
                lambda x, t, c: self.model(x, t, c),
                z, cond, key=step_key,
                latent_scale=self.latent_scale, use_ot=self.use_ot,
            )))
        out = {"fm_loss": sum(losses) / max(len(losses), 1)}

        # sample-based eval — only when explicitly enabled (slow on cpu)
        if cfg.validation.get("eval_sampling", False):
            ev = DiffusionEvaluator(
                cfg, val_ds=self.val_ds,
                autoencoder=self.ae,
                sample_fn=_sample,
                is_rank0=self.dist.is_rank0,
            )
            metrics, val_plots = ev(
                self.model, epoch=epoch,
                batch_size=bs,
                n_steps=cfg.validation.get("eval_sample_steps", 50),
                n_samples_per_traj=cfg.validation.get("eval_n_samples", 1),
                eval_integrals=cfg.validation.get("eval_integrals", True),
                eval_spectra=cfg.validation.get("eval_spectra", False),
                max_batches=cfg.validation.get("eval_max_batches", None),
            )
            out.update(metrics)
            # hand plots to the wandb logger
            if val_plots and self.dist.is_rank0:
                self.logger.log({f"val_plots/{k}": v for k, v in val_plots.items()},
                                step=epoch)
        return out

    def sample(self, *, key, batch: int, cond: Optional[jnp.ndarray] = None, steps: int = 50):
        latents = euler_sample(
            lambda x, t, c: self.model(x, t, c),
            key=key, shape=(batch, *self.latent_shape),
            cond=cond, steps=steps, latent_scale=self.latent_scale,
        )
        return jax.vmap(self.ae.decode)(latents)
