"""GyroSwin training runner: next-step multi-task training on df, phi and flux targets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.dataset.factory import build_splits
from neugk_jax.evaluate.base import Denorm, geometry_table
from neugk_jax.losses import integral_losses
from neugk_jax.training.build import build_gyroswin
from neugk_jax.training.data import stack_fields
from neugk_jax.training.loss_scheduler import LossConfig, compute_multi_task_loss
from neugk_jax.training.runner import BaseRunner
from neugk_jax.utils import config_dict

TARGETS = ("df", "phi", "flux", "fluxavg")


def gyroswin_loss(preds, tgts, weights, loss_cfg, *, integrals=None, separate_zf_loss=False):
    return compute_multi_task_loss(
        preds, tgts, weights, loss_cfg.active, extra=integrals, separate_zf_loss=separate_zf_loss
    )


def model_conditions(cfg) -> tuple:
    conds = cfg.model.get("conditioning")
    if conds is None:
        conds = cfg.dataset.get("conditions", ("itg", "dg", "s_hat", "q"))
    return tuple(conds or ())


class GyroSwinRunner(BaseRunner):
    """Trains GyroSwinMultitask to predict the state at ``t + 1`` from ``t``."""

    adam_b2 = 0.95

    def setup_data(self) -> None:
        cfg = self.cfg
        m = cfg.model
        self.loss_cfg = LossConfig(
            m.get("loss_weights"), m.get("extra_loss_weights"), config_dict(m.get("loss_scheduler"))
        )
        fields = set(cfg.dataset.get("input_fields", ("df",)))
        fields |= {k for k in self.loss_cfg.outputs if k in ("df", "phi")}
        if self.loss_cfg.integrals:
            fields |= {"df", "phi"}
        # the val split ends n_eval_steps frames early; those frames are rollout targets
        tail = int((cfg.get("validation") or {}).get("n_eval_steps", 1))
        self.train_ds, self.val_ds = build_splits(
            cfg.dataset,
            dist=self.dist,
            mode="next",
            fields=tuple(sorted(fields)),
            conditions=model_conditions(cfg),
            val_overrides={"tail_offset": tail},
        )
        self.separate_zf_loss = bool(m.get("extra_zf_loss", False) and self.train_ds.separate_zf)
        self.real_potens = bool(cfg.dataset.get("real_potens", True))
        self._geom: dict[int, dict] = {}

    def build_model(self, key):
        model = build_gyroswin(self.cfg, self.train_ds, key=key)
        if model.flux_key != self.loss_cfg.flux_key:
            raise ValueError(
                f"model flux head {model.flux_key!r} != loss flux key "
                f"{self.loss_cfg.flux_key!r}"
            )
        return model

    def step_context(self) -> dict:
        if not self.loss_cfg.integrals:
            return {}
        return {"norm": Denorm.from_dataset(self.train_ds, TARGETS[:3])}

    def geometry(self, fids) -> dict:
        for f in set(int(f) for f in fids) - set(self._geom):
            self._geom[f] = {k: v[0] for k, v in geometry_table(self.train_ds, [f]).items()}
        return {
            k: np.stack([self._geom[int(f)][k] for f in fids]) for k in self._geom[int(fids[0])]
        }

    def weights_at(self, step: int) -> dict:
        progress_remaining = max(0.0, 1.0 - step / max(self.total_steps, 1))
        return {
            k: jnp.asarray(v, jnp.float32)
            for k, v in self.loss_cfg.weights_at(progress_remaining).items()
        }

    def step_extras(self, step: int) -> dict:
        return {"weights": self.weights_at(step)}

    def load_batch(self, ds, indices, read) -> dict:
        batch = stack_fields(
            read(ds, indices), ("df", "conditioning", "file_index", *(f"y_{k}" for k in TARGETS))
        )
        if self.loss_cfg.integrals:
            batch["geom"] = self.geometry(np.asarray(batch["file_index"]))
        return batch

    def loss_fn(self, model, batch, key):
        loss_cfg = self.loss_cfg
        x, cond = batch["df"], batch.get("conditioning")
        keys = jr.split(key, x.shape[0])
        preds = jax.vmap(lambda xi, ci, k: model(xi, ci, key=k, inference=False))(x, cond, keys)
        tgts = {k: batch.get(f"y_{k}") for k in TARGETS}
        ints = None
        if loss_cfg.integrals:
            norm, fids = batch["norm"], batch["file_index"]
            ints = integral_losses(
                batch["geom"],
                norm("df", preds["df"], fids),
                norm("phi", preds["phi"], fids) if "phi" in preds else None,
                norm("phi", tgts["phi"], fids),
                norm("flux", tgts["flux"], fids),
                real_potens=self.real_potens,
            )
        return gyroswin_loss(
            preds,
            tgts,
            batch["weights"],
            loss_cfg,
            integrals=ints,
            separate_zf_loss=self.separate_zf_loss,
        )

    def make_evaluator(self):
        from neugk_jax.gyroswin.eval import GyroSwinEvaluator

        vcfg = self.cfg.get("validation") or {}
        return GyroSwinEvaluator(
            self.cfg,
            val_ds=self.val_ds,
            dist=self.dist,
            loader=self.loader,
            batch_size=vcfg.get("batch_size") or self.tcfg.batch_size,
            outputs=self.loss_cfg.outputs,
        )
