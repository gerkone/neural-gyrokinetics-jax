"""Generic training loop shell used by both AE and diffusion runners.

Owns the boilerplate (epoch loop, checkpoint resume, eval cadence, logging
hand-off) so the workflow-specific runners only define ``setup_components``,
``train_step`` and ``evaluate``. Mirrors the upstream ``BaseRunner``.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import optax

from neugk_jax.models.utils import trainable_mask
from neugk_jax.training.checkpoint import (
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)
from neugk_jax.training.ddp import DistributedInfo, init_distributed, replicate
from neugk_jax.training.logging import Logger


def weight_decay_mask(params, exclude):
    """Pytree of bools, False where the leaf path contains any ``exclude`` substring."""
    if exclude == "all":
        return jax.tree_util.tree_map(lambda _: False, params)
    exclude = [e.lower() for e in exclude or []]

    def keep(path, _):
        name = jax.tree_util.keystr(path).lower()
        return not any(e in name for e in exclude)

    return jax.tree_util.tree_map_with_path(keep, params)


def build_optimizer(schedule, tcfg, model, *, decoupled: bool, b2: float = 0.999):
    """Clip + Adam chain; ``decoupled`` picks torch AdamW over Adam's coupled L2 decay."""
    wd = tcfg.get("weight_decay", 0.0)
    params = eqx.filter(model, trainable_mask(model))
    mask = weight_decay_mask(params, tcfg.get("exclude_from_wd", []))
    clip = (optax.clip_by_global_norm(tcfg.get("clip_to", 1.0))
            if tcfg.get("clip_grad", True) else optax.identity())
    if wd <= 0:
        return optax.chain(clip, optax.adam(schedule, b2=b2))
    if decoupled:
        return optax.chain(clip, optax.adamw(schedule, b2=b2, weight_decay=wd, mask=mask))
    # torch Adam adds wd * p to the gradient before the moment estimates
    return optax.chain(clip, optax.add_decayed_weights(wd, mask=mask),
                       optax.scale_by_adam(b2=b2), optax.scale_by_learning_rate(schedule))


def train_update(model, opt_state, loss_fn, optimizer, mask, *, has_aux: bool = False):
    """One optimizer step on the leaves ``mask`` marks trainable; buffers stay fixed."""
    params, static = eqx.partition(model, mask)
    out, grads = eqx.filter_value_and_grad(
        lambda p: loss_fn(eqx.combine(p, static)), has_aux=has_aux)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.combine(eqx.apply_updates(params, updates), static), opt_state, out


class BaseRunner(ABC):
    # validation metrics that select best.eqx, first present wins (lower is better)
    val_metrics: tuple[str, ...] = ("df", "df_mse")
    cfg: Any
    dist: DistributedInfo
    logger: Logger

    def __init__(self, cfg, *, output_path: str | None = None):
        self.cfg = cfg
        self.dist = init_distributed()
        self.logger = Logger(
            is_rank0=self.dist.is_rank0,
            cfg=getattr(cfg, "logging", None) and dict(cfg.logging) or None,
            mode=getattr(getattr(cfg, "logging", {}), "mode", "online")
            if getattr(cfg, "logging", None)
            else "disabled",
        )
        self.output_path = Path(output_path or getattr(cfg, "output_path", "outputs/run"))
        self.start_epoch = 0
        self.best_val = math.inf
        self.opt_state = None
        self.model = None
        self.setup_data()
        self.setup_components()
        self._maybe_resume()

    @staticmethod
    def _omegaconf_to_dict(node):
        """OmegaConf node → plain python containers (dataset code indexes them directly)."""
        if node is None:
            return None
        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(node, resolve=True)
        except Exception:
            return dict(node)

    @abstractmethod
    def setup_data(self) -> None: ...

    @abstractmethod
    def setup_components(self) -> None: ...

    @abstractmethod
    def train_epoch(self, epoch: int, key) -> dict: ...

    @abstractmethod
    def evaluate(self, epoch: int) -> dict: ...

    def _maybe_resume(self):
        ckpt = self.output_path / "ckp.eqx"
        if ckpt.exists():
            state = load_checkpoint(ckpt, self.model)
            self.model = state.model
            self.opt_state = state.opt_state
            self.start_epoch = state.epoch
            self.model = replicate(self.dist, self.model)
            self.opt_state = replicate(self.dist, self.opt_state)
            self.best_val = float((state.meta or {}).get("best_val", math.inf))
            if self.dist.is_rank0:
                print(f"resumed from epoch {self.start_epoch} (val={self.best_val:.4e})")

    def save_checkpoint(self, epoch: int, val: float, name: str = "ckp.eqx") -> None:
        if not self.dist.is_rank0:
            return
        save_checkpoint(
            self.output_path / name,
            CheckpointState(
                model=self.model,
                opt_state=self.opt_state,
                epoch=epoch,
                loss=val,
                meta={"best_val": self.best_val},
            ),
        )

    def _val_score(self, val_logs: dict) -> float:
        for k in self.val_metrics:
            if k in val_logs:
                return float(val_logs[k])
        raise KeyError(f"none of {self.val_metrics} in validation metrics {sorted(val_logs)}")

    def _current_lr(self, step: int) -> float | None:
        sched = getattr(self, "schedule", None)
        if sched is None:
            return None
        try:
            return float(sched(step))
        except Exception:
            return None

    def __call__(self) -> None:
        base_key = jax.random.PRNGKey(getattr(self.cfg, "seed", 0))
        val_every = getattr(self.cfg.validation, "validate_every_n_epochs", 1)
        last_val = math.nan
        for epoch in range(self.start_epoch + 1, self.cfg.training.n_epochs + 1):
            train_key = jax.random.fold_in(base_key, epoch)
            t0 = time.perf_counter()
            # train_epoch returns either loss_logs, or (loss_logs, info_dict)
            train_out = self.train_epoch(epoch, train_key)
            if isinstance(train_out, tuple) and len(train_out) == 2:
                loss_logs, info_dict = train_out
            else:
                loss_logs, info_dict = train_out, {}
            t_train = time.perf_counter() - t0

            val_logs, val_plots = {}, {}
            validating = epoch % val_every == 0 or epoch == 1
            if validating:
                val_out = self.evaluate(epoch)
                if isinstance(val_out, tuple) and len(val_out) == 2:
                    val_logs, val_plots = val_out
                else:
                    val_logs = val_out

            # build wandb-style log dict: train/* (losses + lr) | info/* (timing) | val_traj/*
            lr = self._current_lr(epoch * getattr(self, "steps_per_epoch", 1))
            train_ns = {f"train/{k}": v for k, v in loss_logs.items()}
            if lr is not None:
                train_ns["train/lr"] = lr
            info_ns = {f"info/{k}": v for k, v in info_dict.items()}
            val_ns = {f"val_traj/{k}": v for k, v in val_logs.items()}
            logs = {**train_ns, **info_ns, **val_ns, "epoch": epoch, "epoch_time_s": t_train}
            self.logger.log(logs, step=epoch, commit=not val_plots)
            if val_plots:
                self.logger.log(val_plots, step=epoch, commit=True)
            if self.dist.is_rank0:
                core = " ".join(
                    f"{k}={v:.4e}" for k, v in loss_logs.items() if isinstance(v, (int, float))
                )
                print(f"epoch {epoch:04d}  {core}  ({t_train:.1f}s)")

            if validating:
                last_val = self._val_score(val_logs)
                if last_val < self.best_val:
                    self.best_val = last_val
                    self.save_checkpoint(epoch, last_val, "best.eqx")
            save_every = getattr(self.cfg.training, "save_every_n_epochs", 1)
            if epoch % save_every == 0 or epoch == self.cfg.training.n_epochs:
                self.save_checkpoint(epoch, last_val, "ckp.eqx")

        self.logger.finish()
