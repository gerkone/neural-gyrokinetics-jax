"""Latent flow-matching runner.

Reuses the AE dataset setup, loads a trained AE (frozen), caches its latents over both
splits, keeps the latent tables on device and trains a DiT on Gaussian -> latent flow
matching with the batch latents gathered inside the step.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from omegaconf import open_dict

from neugk_jax.autoencoders.runner import AERunner
from neugk_jax.dataset import precompute_latents
from neugk_jax.diffusion.flow_matching import fm_forward_loss
from neugk_jax.diffusion.latents import (
    latent_arrays,
    latent_cache_meta,
    latent_cache_path,
    load_precomputed_latents,
)
from neugk_jax.training.build import build_dit
from neugk_jax.training.ddp import local_view, replicate_local
from neugk_jax.training.runner import conditioning_slots
from neugk_jax.utils import config_dict, count_trace


def resolve_ae_checkpoint(path) -> Path:
    """AE checkpoint file for a run directory (``best.eqx``, else ``best.pth``) or a file path."""
    p = Path(path)
    if p.is_dir():
        for name in ("best.eqx", "best.pth"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"no best.eqx / best.pth in {p}")
    return p


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


def load_autoencoder(path, *, resolution=None, dataset: dict | None = None):
    """AE of a run directory or checkpoint file; ``dataset`` is checked against the run's."""
    from neugk_jax.translate import build_ae_from_config, load_config, load_or_translate

    ae_file = resolve_ae_checkpoint(path)
    ae_cfg = load_config(str(ae_file.parent / "config.yaml"))
    if dataset is not None:
        check_ae_dataset(ae_cfg.get("dataset") or {}, dataset)
    template = build_ae_from_config(ae_cfg, key=jr.PRNGKey(0), resolution=resolution)
    return load_or_translate(template, str(ae_file))


@eqx.filter_jit
def encode_batch(ae, df):
    return jax.vmap(lambda x: ae.encode(x)[0])(df)


def _fm_loss(model, z, cond, key, *, latent_scale, use_ot, train: bool, mask=None):
    fm_key, drop_key = jr.split(key)

    def fwd(x, t, *rest):
        c = rest[0] if cond is not None else None
        k = rest[-1] if train else None
        return model(x, t, c, key=k, inference=not train)

    return fm_forward_loss(
        fwd,
        z,
        cond,
        key=fm_key,
        latent_scale=latent_scale,
        use_ot=use_ot,
        dropout_key=drop_key if train else None,
        mask=mask,
    )


@eqx.filter_jit
def fm_val_loss(model, latents, cond, idx, mask, key, latent_scale: float, use_ot: bool):
    count_trace("fm_val_loss")
    c = None if cond is None else cond[idx]
    return _fm_loss(
        model,
        latents[idx],
        c,
        key,
        latent_scale=latent_scale,
        use_ot=use_ot,
        train=False,
        mask=mask,
    )


class FlowMatchingRunner(AERunner):
    """Trains a DiT on the latent distribution of a frozen AE via flow matching."""

    val_metrics = ("avg_flux_rmse", "fm_loss")
    decoupled_wd = True
    conditioned = True

    def train_dtype(self):
        return None

    def setup_data(self) -> None:
        cfg = self.cfg
        if cfg.get("ae_checkpoint") is None:
            raise ValueError("diffusion workflow requires ae_checkpoint")
        super().setup_data()
        dcfg = config_dict(cfg.dataset)
        self.ae = load_autoencoder(
            cfg.ae_checkpoint, resolution=self.train_ds.resolution, dataset=dcfg
        )
        self.latent_shape = (*self.ae.bottleneck_grid_size, int(self.ae.bottleneck_dim))
        ae_file = resolve_ae_checkpoint(cfg.ae_checkpoint)
        for ds, key in ((self.train_ds, "latents_cache_train"), (self.val_ds, "latents_cache_val")):
            meta = latent_cache_meta(
                ds, ae_file, normalization_stats=dcfg.get("normalization_stats")
            )
            path = dcfg.get(key)
            if path:
                load_precomputed_latents(ds, path, latent_shape=self.latent_shape, meta=meta)
            else:
                cache = latent_cache_path(
                    ds,
                    ds.split,
                    cfg.ae_checkpoint,
                    decouple_mu=dcfg.get("norm_decouple_mu", False),
                    timestep_std_filter=dcfg.get("timestep_std_filter"),
                )
                precompute_latents(
                    ds,
                    encode_fn=lambda df, _c: encode_batch(self.ae, df),
                    cache_file=cache,
                    batch_size=self.tcfg.get("precompute_batch", 2),
                    meta=meta,
                    latent_shape=self.latent_shape,
                )
        # df-mode view of the val split for sample targets
        self.val_df_ds = self.val_ds.with_mode("ae")
        self.cond_slots = conditioning_slots(
            self.train_ds.conditions, list(cfg.model.get("conditioning") or [])
        )
        if cfg.get("latent_scale") is not None:
            self.latent_scale = float(cfg.latent_scale)
        else:
            var = float(np.mean(self.train_ds.latent_stats.var))
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

    def _tables(self, ds) -> dict:
        z, cond = latent_arrays(ds)
        if self.cond_slots is None:
            return {"latents": z, "cond": None}
        return {"latents": z, "cond": cond[:, self.cond_slots]}

    def step_context(self) -> dict:
        return self._tables(self.train_ds)

    def load_batch(self, ds, indices, read) -> dict:
        return {"idx": np.asarray(indices, np.int32)}

    def loss_fn(self, model, batch, key):
        idx, cond = batch["idx"], batch["cond"]
        loss = _fm_loss(
            model,
            batch["latents"][idx],
            None if cond is None else cond[idx],
            key,
            latent_scale=self.latent_scale,
            use_ot=self.use_ot,
            train=True,
        )
        return loss, {"fm_loss": loss}

    def make_evaluator(self):
        self.val_tables = replicate_local(self.dist, self._tables(self.val_ds))
        if not (self.cfg.get("validation") or {}).get("eval_sampling", False):
            return None
        from neugk_jax.diffusion.eval import DiffusionEvaluator

        vcfg = self.cfg.validation
        return DiffusionEvaluator(
            self.cfg,
            val_ds=self.val_df_ds,
            autoencoder=self.ae,
            latent_scale=self.latent_scale,
            cond_slots=self.cond_slots,
            dist=self.dist,
            loader=self.loader,
            batch_size=vcfg.get("batch_size") or self.tcfg.batch_size,
        )

    def evaluate(self, epoch: int) -> tuple[dict, dict]:
        bs, n = self.tcfg.batch_size, len(self.val_ds)
        model = local_view(self.dist, self.model)
        key = jr.fold_in(jr.PRNGKey(self.cfg.get("seed", 0)), n)
        total = 0.0
        for i, start in enumerate(range(0, n, bs)):
            idx = np.arange(start, start + bs) % n
            mask = (np.arange(start, start + bs) < n).astype(np.float32)
            loss = fm_val_loss(
                model,
                self.val_tables["latents"],
                self.val_tables["cond"],
                jnp.asarray(idx, jnp.int32),
                jnp.asarray(mask),
                jr.fold_in(key, i),
                self.latent_scale,
                self.use_ot,
            )
            total += float(loss) * float(mask.sum())
        out = {"fm_loss": total / n if n else float("nan")}
        plots = {}
        if self.evaluator is not None:
            metrics, plots = self.evaluator(self.model, epoch=epoch)
            out.update(metrics)
        return out, plots
