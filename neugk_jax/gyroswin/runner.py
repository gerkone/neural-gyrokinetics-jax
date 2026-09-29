"""GyroSwin training runner: next-step multi-task training on df, phi and flux targets."""

from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml
from omegaconf import OmegaConf

from neugk_jax.dataset import CycloneDataset, KvikIOBackend, NumpyBackend
from neugk_jax.evaluate.integrals import precompute_geometry
from neugk_jax.gyroswin.models import build_gyroswin_from_config
from neugk_jax.losses import integral_losses
from neugk_jax.models.utils import trainable_mask
from neugk_jax.training.ddp import global_batch_size, process_batch_indices, replicate, shard_batch
from neugk_jax.training.loss_scheduler import LossConfig, compute_multi_task_loss
from neugk_jax.training.runner import BaseRunner, build_optimizer, train_update
from neugk_jax.training.schedulers import warmup_cosine

# sample ndim (without batch) per normalized field
_FIELD_NDIM = {"df": 6, "phi": 3, "flux": 0}


def gyroswin_loss(preds, tgts, weights, loss_cfg, *, integrals=None, separate_zf_loss=False):
    return compute_multi_task_loss(preds, tgts, weights, loss_cfg.active, extra=integrals,
                                   separate_zf_loss=separate_zf_loss)


def model_conditions(cfg) -> tuple:
    conds = cfg.model.get("conditioning")
    if conds is None:
        conds = cfg.dataset.get("conditions", ("itg", "dg", "s_hat", "q"))
    return tuple(conds or ())


class GyroSwinRunner(BaseRunner):
    """Trains GyroSwinMultitask to predict the state at ``t + 1`` from ``t``."""

    def setup_data(self) -> None:
        cfg = self.cfg
        m = cfg.model
        self.loss_cfg = LossConfig(m.get("loss_weights"), m.get("extra_loss_weights"),
                                   self._omegaconf_to_dict(m.get("loss_scheduler")))
        fields = set(cfg.dataset.get("input_fields", ("df",)))
        fields |= {k for k in self.loss_cfg.outputs if k in ("df", "phi")}
        if self.loss_cfg.integrals:
            fields |= {"df", "phi"}
        backend = (
            KvikIOBackend(rank=self.dist.local_rank)
            if getattr(cfg.dataset, "backend", "kvikio") == "kvikio"
            else NumpyBackend()
        )
        stats = cfg.dataset.get("normalization_stats")
        if stats is not None and not isinstance(stats, str):
            stats = self._omegaconf_to_dict(stats)
        common = dict(
            path=cfg.dataset.path,
            fields_to_load=tuple(sorted(fields)),
            conditions=model_conditions(cfg),
            mode="next",
            backend=backend,
            separate_zf=cfg.dataset.get("separate_zf", True),
            real_potens=cfg.dataset.get("real_potens", True),
            normalization=self._omegaconf_to_dict(cfg.dataset.get("normalization")),
            normalization_scope=cfg.dataset.get("normalization_scope", "dataset"),
            normalization_stats=stats,
            offset=cfg.dataset.get("offset", 0),
            rank=self.dist.process_id,
        )
        self.train_ds = CycloneDataset(
            split="train", trajectories=cfg.dataset.training_trajectories,
            cond_filters=self._omegaconf_to_dict(cfg.dataset.get("training_cond_filters")), **common,
        )
        # crop the trajectory tail so autoregressive rollout targets exist
        self.val_ds = CycloneDataset(
            split="val", trajectories=cfg.dataset.validation_trajectories,
            cond_filters=self._omegaconf_to_dict(cfg.dataset.get("eval_cond_filters")),
            tail_offset=int(cfg.validation.get("n_eval_steps", 1)), **common,
        )

    def setup_components(self) -> None:
        cfg = self.cfg
        cfg_d = OmegaConf.to_container(cfg, resolve=True)
        cfg_d.setdefault("dataset", {})
        cfg_d["dataset"].setdefault("resolution", list(self.train_ds.resolution))
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"model": cfg_d["model"], "dataset": cfg_d["dataset"]}, f)
            tmp_cfg = f.name
        self.model = build_gyroswin_from_config(
            tmp_cfg, key=jr.PRNGKey(getattr(cfg, "seed", 0)),
            legacy_double_shortcut=cfg.model.get("legacy_swin_shortcut", False),
        )
        if self.model.flux_key != self.loss_cfg.flux_key:
            raise ValueError(f"model flux head {self.model.flux_key!r} != loss flux key "
                             f"{self.loss_cfg.flux_key!r}")

        self.global_bs = global_batch_size(self.dist, cfg.training.batch_size)
        steps_per_epoch = max(1, len(self.train_ds) // self.global_bs)
        total = cfg.training.n_epochs * steps_per_epoch
        self.schedule = warmup_cosine(
            peak_lr=cfg.training.learning_rate, total_steps=total,
            steps_per_epoch=steps_per_epoch, n_epochs=cfg.training.n_epochs,
            min_lr=cfg.training.get("final_learning_rate", 1e-6),
        )
        self.optimizer = build_optimizer(self.schedule, cfg.training, self.model,
                                         decoupled=False, b2=0.95)
        self.trainable = trainable_mask(self.model)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, self.trainable))
        self.model = replicate(self.dist, self.model)
        self.opt_state = replicate(self.dist, self.opt_state)
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total
        self.separate_zf_loss = bool(cfg.model.get("extra_zf_loss", False)
                                     and cfg.dataset.get("separate_zf", True))
        self.real_potens = bool(cfg.dataset.get("real_potens", True))
        self._geom_cache: dict[int, dict] = {}
        self._shared_norm = None
        if self.loss_cfg.integrals and self._norm_is_shared(self.train_ds):
            self._shared_norm = replicate(self.dist, self._norm_params(self.train_ds, [0]))

    @staticmethod
    def _norm_is_shared(ds) -> bool:
        return ds.normalization is None or ds.normalization_scope in ("dataset", "sample")

    @staticmethod
    def _norm_params(ds, fids) -> dict:
        return {k: tuple(jnp.asarray(a) for a in ds.scale_shift(fids, k, nd))
                for k, nd in _FIELD_NDIM.items()}

    def geometry(self, ds, fids) -> dict:
        """Stacked :func:`precompute_geometry` tensors for a batch of file indices."""
        cache = self._geom_cache.setdefault(id(ds), {})
        for f in set(int(f) for f in fids):
            if f not in cache:
                cache[f] = precompute_geometry(ds.metadata[f]["geometry"])
        return {k: np.stack([cache[int(f)][k] for f in fids]) for k in cache[int(fids[0])]}

    def weights_at(self, step: int) -> dict:
        progress_remaining = max(0.0, 1.0 - step / max(self.total_steps, 1))
        return {k: jnp.asarray(v, jnp.float32)
                for k, v in self.loss_cfg.weights_at(progress_remaining).items()}

    @eqx.filter_jit
    def _train_step(self, model, opt_state, batch, weights, key, shared_norm=None):
        loss_cfg = self.loss_cfg
        bsz = batch["df"].shape[0]
        keys = jr.split(key, bsz)

        def loss_fn(m):
            preds = jax.vmap(lambda x, c, k: m(x, c, key=k, inference=False))(
                batch["df"], batch.get("cond"), keys)
            tgts = {k: batch.get(f"y_{k}") for k in ("df", "phi", "flux", "fluxavg")}
            ints = None
            if loss_cfg.integrals:
                norm = shared_norm if shared_norm is not None else batch["norm"]

                def den(x, field):
                    scale, shift = norm[field]
                    return x * scale + shift

                ints = integral_losses(
                    batch["geom"], den(preds["df"], "df"),
                    den(preds["phi"], "phi") if "phi" in preds else None,
                    den(tgts["phi"], "phi"), den(tgts["flux"], "flux"),
                    real_potens=self.real_potens,
                )
            return gyroswin_loss(preds, tgts, weights, loss_cfg, integrals=ints,
                                 separate_zf_loss=self.separate_zf_loss)

        return train_update(model, opt_state, loss_fn, self.optimizer, self.trainable, has_aux=True)

    def load_batch(self, ds, indices) -> dict:
        samples = [ds[int(i)] for i in indices]
        b = CycloneDataset.collate(samples)
        out = {"df": b.df, "cond": b.conditioning, "y_df": b.y_df, "y_phi": b.y_phi,
               "y_flux": b.y_flux, "y_fluxavg": b.y_fluxavg}
        if self.loss_cfg.integrals:
            fids = np.asarray(b.file_index)
            out["geom"] = self.geometry(ds, fids)
            if self._shared_norm is None:
                out["norm"] = self._norm_params(ds, fids)
        return {k: v for k, v in out.items() if v is not None}

    def train_epoch(self, epoch: int, key) -> tuple[dict, dict]:
        gbs = self.global_bs
        n = len(self.train_ds)
        perm_key, step_key = jr.split(key)
        idx = np.asarray(jr.permutation(perm_key, n))
        starts = list(range(0, n - gbs + 1, gbs))

        def _load(start):
            local = process_batch_indices(self.dist, idx[start:start + gbs])
            return shard_batch(self.dist, self.load_batch(self.train_ds, local))

        ex = ThreadPoolExecutor(max_workers=1)
        sums: dict[str, float] = {}
        t_data, t_step = [], []
        future = ex.submit(_load, starts[0]) if starts else None
        try:
            for i, start in enumerate(starts):
                t0 = time.perf_counter_ns()
                batch = future.result()
                t_data.append((time.perf_counter_ns() - t0) / 1e6)
                if i + 1 < len(starts):
                    future = ex.submit(_load, starts[i + 1])
                t0 = time.perf_counter_ns()
                weights = self.weights_at((epoch - 1) * self.steps_per_epoch + i)
                self.model, self.opt_state, (loss, parts) = self._train_step(
                    self.model, self.opt_state, batch, weights, jr.fold_in(step_key, i),
                    self._shared_norm,
                )
                sums["total"] = sums.get("total", 0.0) + float(loss)
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                t_step.append((time.perf_counter_ns() - t0) / 1e6)
        finally:
            ex.shutdown(wait=False)

        def median(ts):
            ts = sorted(ts[1:] or ts)
            return ts[len(ts) // 2] if ts else 0.0

        loss_logs = {k: v / max(len(starts), 1) for k, v in sums.items()}
        info = {"data_ms": median(t_data), "step_ms": median(t_step),
                "first_step_ms": t_step[0] if t_step else 0.0}
        return loss_logs, info

    def evaluate(self, epoch: int):
        from neugk_jax.gyroswin.eval import GyroSwinEvaluator
        ev = GyroSwinEvaluator(self.cfg, val_ds=self.val_ds, is_rank0=self.dist.is_rank0)
        return ev(self.model, epoch=epoch, batch_size=self.cfg.training.batch_size,
                  dist=self.dist, outputs=self.loss_cfg.outputs,
                  geometry_fn=lambda fids: self.geometry(self.val_ds, fids))
