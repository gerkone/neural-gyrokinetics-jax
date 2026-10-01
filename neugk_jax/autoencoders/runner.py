"""AE training runner: reconstruction loss on df with Adam (coupled L2) + warmup-cosine."""

from __future__ import annotations

import jax
import jax.random as jr

from neugk_jax.losses import df_loss
from neugk_jax.models.build import build_ae
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

    def setup_data(self) -> None:
        if self.cfg.model.get("conditioning") not in (None, [], ()):
            raise ValueError(
                "`model.conditioning` is set but the AE is unconditional; drop it or use "
                "workflow=gyroswin"
            )
        self.build_data("ae", train_dtype=train_dtype(self.cfg))
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

        return AEEvaluator(self.cfg, **self.evaluator_kwargs())
