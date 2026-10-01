"""AE training runner: reconstruction loss on df with Adam (coupled L2) + warmup-cosine."""

from __future__ import annotations

import jax
import jax.random as jr

from neugk_jax.dataset.factory import build_splits
from neugk_jax.losses import df_loss
from neugk_jax.training.build import build_ae
from neugk_jax.training.runner import BaseRunner


def train_dtype(cfg):
    # dataset.prefer_dtype, else training.amp dtype
    amp = cfg.training.get("amp") or {}
    return cfg.dataset.get("prefer_dtype") or (
        amp.get("dtype", "bf16") if amp.get("enable") else None
    )


class AERunner(BaseRunner):
    """Trains the Swin5DAE on cyclone df snapshots."""

    val_metrics = ("df_mse",)
    conditioned = False

    def train_dtype(self):
        return train_dtype(self.cfg)

    def setup_data(self) -> None:
        if not self.conditioned and self.cfg.model.get("conditioning") not in (None, [], ()):
            raise ValueError(
                f"`model.conditioning` is set but {type(self).__name__} is "
                "unconditional; drop it or use workflow=gyroswin"
            )
        self.train_ds, self.val_ds = build_splits(
            self.cfg.dataset, dist=self.dist, mode="ae", train_dtype=self.train_dtype()
        )
        self.separate_zf = bool(self.train_ds.separate_zf)

    def build_model(self, key):
        return build_ae(self.cfg, self.train_ds, key=key)

    def loss_fn(self, model, batch, key):
        x = batch["df"]
        keys = jr.split(key, x.shape[0])
        pred = jax.vmap(lambda xi, k: model(xi, key=k, inference=False)["df"])(x, keys)
        loss = df_loss(pred, x, separate_zf=self.separate_zf)
        return loss, {"df": loss}

    def make_evaluator(self):
        from neugk_jax.autoencoders.eval import AEEvaluator

        vcfg = self.cfg.get("validation") or {}
        return AEEvaluator(
            self.cfg,
            val_ds=self.val_ds,
            dist=self.dist,
            loader=self.loader,
            batch_size=vcfg.get("batch_size") or self.tcfg.batch_size,
        )
