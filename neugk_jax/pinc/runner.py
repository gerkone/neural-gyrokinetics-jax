"""AE training runners.

``AERunner``: reconstruction loss on df with Adam (coupled L2) + warmup-cosine. ``VQVAERunner``:
the Swin5DVQVAE on the ``model.loss_weights`` of the ``df`` + ``vq_commit`` losses, with the EMA
codebook update after every optimizer step.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.losses import df_loss, recon_loss
from neugk_jax.models.build import ae_conditioning, build_ae
from neugk_jax.models.utils import cast_floating
from neugk_jax.pinc.eval import (
    MAX_HISTOGRAM_CODES,
    AEEvaluator,
    VQVAEEvaluator,
    reconstruct,
    select_conditions,
)
from neugk_jax.pinc.quantizers import VectorQuantizer
from neugk_jax.pinc.vqvae import Swin5DVQVAE
from neugk_jax.training.data import stack_fields
from neugk_jax.training.runner import BaseRunner, conditioning_slots
from neugk_jax.utils import to_dict


def train_dtype(cfg):
    # dataset.prefer_dtype, else training.amp dtype
    amp = cfg.training.get("amp") or {}
    return cfg.dataset.get("prefer_dtype") or (
        amp.get("dtype", "bf16") if amp.get("enable") else None
    )


def read_loss_weights(mcfg, supported) -> dict[str, float]:
    """Nonzero ``model.loss_weights`` + ``model.extra_loss_weights``; raises unless all are supported."""
    weights = {**to_dict(mcfg.get("loss_weights")), **to_dict(mcfg.get("extra_loss_weights"))}
    if not weights:
        raise ValueError(f"model.loss_weights is required; weights over {tuple(supported)}")
    scheduled = [k for k, v in to_dict(mcfg.get("loss_scheduler")).items() if v]
    # zero weights switch a term off, whatever the workflow supports
    unknown = sorted(k for k, v in weights.items() if float(v) != 0 and k not in supported)
    unknown += sorted(scheduled)
    if unknown:
        raise ValueError(
            f"unsupported loss weights / schedules {unknown}; one of {tuple(supported)}"
        )
    return {k: float(v) for k, v in weights.items() if float(v) != 0.0}


class AERunner(BaseRunner):
    """Trains the Swin5DAE on cyclone df snapshots, with its encoder / decoder conditioning.

    ``training.loss_type`` picks the :func:`recon_loss` of df; unset, the zf-split
    :func:`df_loss`.
    """

    val_metrics = ("df_mse",)
    evaluator_cls = AEEvaluator
    default_loss_type = None
    # whether a quantized (model_type vqvae) model may be built
    accepts_vq = False

    def setup_data(self) -> None:
        self.build_data("ae", train_dtype=train_dtype(self.cfg))
        self.separate_zf = bool(self.train_ds.separate_zf)
        enc, dec = ae_conditioning(self.cfg.model)
        self.cond_slots = conditioning_slots(self.train_ds.conditions, set(enc) | set(dec))
        self.loss_type = self.tcfg.get("loss_type", self.default_loss_type)
        self.extra_zf = bool(self.cfg.model.get("extra_zf_loss", False)) and self.separate_zf
        # training.batch_transform: readers serve raw frames, normalized per batch in the step
        self.train_ds.batch_transform = bool(self.tcfg.get("batch_transform", False))
        # training.compute_dtype: forward in bf16 on a cast copy of the fp32 weights (torch autocast)
        dt = self.tcfg.get("compute_dtype")
        self.compute_dtype = {"bf16": jnp.bfloat16, "fp16": jnp.float16}.get(dt) if dt else None

    def place_batch(self, batch: dict) -> dict:
        # eager ops, bit-identical to the per-sample transform (a jitted fusion is not)
        if not self.train_ds.batch_transform:
            return batch
        return {**batch, "df": self.train_ds.transform(batch["df"], batch["file_index"])}

    def forward(self, model, x, cond, keys) -> dict:
        """Training reconstruction, in ``compute_dtype`` when set; the outputs come back fp32."""
        if self.compute_dtype is None:
            return reconstruct(model, x, cond, keys, inference=False)
        dt = self.compute_dtype
        model = cast_floating(model, dt)
        cond = None if cond is None else cond.astype(dt)
        out = reconstruct(model, x.astype(dt), cond, keys, inference=False)
        return {**out, "df": out["df"].astype(jnp.float32)}

    def checkpoint_meta(self) -> dict:
        return {"resolution": [int(r) for r in self.train_ds.resolution]}

    def build_model(self, key):
        model = build_ae(self.cfg, self.train_ds, key=key)
        if isinstance(model, Swin5DVQVAE) and not self.accepts_vq:
            raise ValueError(f"{type(self).__name__} cannot train a vq-vae; use workflow=vqvae")
        return model

    def load_batch(self, ds, indices, read) -> dict:
        fields = ("df", "file_index", "conditioning")
        return select_conditions(stack_fields(read(ds, indices), fields), self.cond_slots)

    def df_recon_loss(self, pred, x):
        if self.loss_type is None:
            return df_loss(pred, x, separate_zf=self.separate_zf)
        return recon_loss(pred, x, str(self.loss_type), separate_zf=self.extra_zf)

    def loss_fn(self, model, batch, key):
        x = batch["df"]
        keys = jr.split(key, x.shape[0])
        pred = self.forward(model, x, batch.get("conditioning"), keys)["df"]
        loss = self.df_recon_loss(pred, x)
        return loss, {"df": loss}

    def evaluator_kwargs(self) -> dict:
        return {**super().evaluator_kwargs(), "cond_slots": self.cond_slots}

    def make_evaluator(self):
        return self.evaluator_cls(self.cfg, **self.evaluator_kwargs())


class VQVAERunner(AERunner):
    """Trains the Swin5DVQVAE; the VQ codebook buffers follow the EMA rule outside the gradient."""

    evaluator_cls = VQVAEEvaluator
    default_loss_type = "mse"
    accepts_vq = True

    def setup_data(self) -> None:
        if self.cfg.model.get("model_type") != "vqvae":
            raise ValueError("workflow=vqvae needs model.model_type=vqvae")
        super().setup_data()
        self.loss_weights = read_loss_weights(self.cfg.model, ("df", "vq_commit"))

    def loss_fn(self, model, batch, key):
        x = batch["df"]
        keys = jr.split(key, x.shape[0])
        out = self.forward(model, x, batch.get("conditioning"), keys)
        losses = {
            "df": self.df_recon_loss(out["df"], x),
            "vq_commit": model.vq.batch_loss(out["vq_aux"]),
        }
        aux = dict(losses)
        if model.codebook_size <= MAX_HISTOGRAM_CODES:
            used = jnp.bincount(out["vq_indices"].reshape(-1), length=model.codebook_size) > 0
            aux["codebook_usage"] = jnp.mean(used.astype(jnp.float32))
        if isinstance(model.vq, VectorQuantizer):
            aux["state"] = (out["vq_aux"]["z"], out["vq_indices"])
        return sum(w * losses[k] for k, w in self.loss_weights.items()), aux

    def post_update(self, model, state, key):
        if state is None:
            return model, {}
        z, idx = state
        vq, n = model.vq.ema_update(z.reshape(-1, z.shape[-1]), idx.reshape(-1), key)
        return eqx.tree_at(lambda m: m.vq, model, vq), {"dead_codes": n.astype(jnp.float32)}
