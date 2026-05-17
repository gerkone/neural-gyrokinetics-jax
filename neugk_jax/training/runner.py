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
import jax.numpy as jnp
from tqdm import tqdm

from neugk_jax.training.checkpoint import (
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)
from neugk_jax.training.ddp import DistributedInfo, init_distributed
from neugk_jax.training.logging import Logger


class BaseRunner(ABC):
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
            self.best_val = state.loss
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
            ),
        )

    def __call__(self) -> None:
        key = jax.random.PRNGKey(getattr(self.cfg, "seed", 0))
        for epoch in range(self.start_epoch + 1, self.cfg.training.n_epochs + 1):
            key, train_key = jax.random.split(key)
            t0 = time.perf_counter()
            train_logs = self.train_epoch(epoch, train_key)
            t_train = time.perf_counter() - t0

            val_logs = {}
            if epoch % getattr(self.cfg.validation, "validate_every_n_epochs", 1) == 0:
                val_logs = self.evaluate(epoch)

            logs = {**{f"train/{k}": v for k, v in train_logs.items()},
                    **{f"val/{k}": v for k, v in val_logs.items()},
                    "epoch": epoch, "epoch_time_s": t_train}
            self.logger.log(logs, step=epoch)
            if self.dist.is_rank0:
                print(
                    f"epoch {epoch:04d}  "
                    + " ".join(f"{k}={v:.4e}" for k, v in train_logs.items() if isinstance(v, (int, float)))
                    + f"  ({t_train:.1f}s)"
                )

            val = val_logs.get("df", train_logs.get("loss", math.inf))
            if val < self.best_val:
                self.best_val = val
                self.save_checkpoint(epoch, val, "best.eqx")
            self.save_checkpoint(epoch, val, "ckp.eqx")

        self.logger.finish()
