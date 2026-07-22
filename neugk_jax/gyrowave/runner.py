"""gyrowave flow-matching runner.

Mirrors neugk_jax/diffusion/runner.py:FlowMatchingRunner — subclasses BaseRunner and reuses
neugk_jax.diffusion.flow_matching (fm_forward_loss, euler_sample). WaveletDiT's extra static
per-token COORD input is threaded through the FM `cond` slot as a ``(params, coords)`` pair,
so the existing vmap-based loss and Euler sampler apply unchanged.

Overfit/eval on one trajectory (constant params) is distribution-level: does the FM reproduce
the trajectory's heat-flux distribution (reconstructed via gyrowave.compress.fastops + gyaradax
flux).
"""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from neugk_jax.dataset import WaveCycloneDataset

# tokens = the compressed data representation (not a cache)
from neugk_jax.diffusion.flow_matching import euler_sample, fm_forward_loss
from neugk_jax.gyrowave import WaveletDiT
from neugk_jax.training.runner import BaseRunner


def _model_fn(model):
    # FM passes cond = (params (n_cond,), coords (N, n_coord)) per sample after vmap.
    return lambda x, t, c: model(x, c[1], t, c[0])


def sample_values(model, coords_n, params, n_samples=16, steps=50, seed=1):
    # TODO (JAX-native, no torch): flux eval needs scatter -> inverse (s,x) wavelet + inverse
    # HL moment matmul + inverse y-FFT -> field -> gyaradax_flux_integrals vs GT; requires
    # porting compress recon_field to JAX.
    cond = (jnp.broadcast_to(jnp.asarray(params), (n_samples, params.shape[0])),
            jnp.broadcast_to(jnp.asarray(coords_n[:1]), (n_samples,) + coords_n.shape[1:]))
    return euler_sample(_model_fn(model), key=jr.PRNGKey(seed),
                        shape=(n_samples, coords_n.shape[1], 2), cond=cond, steps=steps)


class GyrowaveFMRunner(BaseRunner):
    """Flow matching over wavelet-token sets. Same shape as FlowMatchingRunner:
    setup_data / setup_components / _train_step / train_epoch / evaluate / sample."""

    def setup_data(self) -> None:
        # WaveCycloneDataset pools all trajectories + global-whitens (single-cache = 1-traj
        # special case, reproduces the pre-multi-traj behavior exactly)
        self.data = WaveCycloneDataset.from_config(self.cfg.dataset)
        self.X_np, self.C_np, self.P_np = self.data.X, self.data.C, self.data.P
        self.nrm, self.S, self.N = self.data.nrm, self.data.S, self.data.N
        self.X, self.C = jnp.asarray(self.X_np), jnp.asarray(self.C_np)
        self.P = jnp.asarray(self.P_np)                              # (S_tot, n_cond) per-sample params
        self.n_cond = self.data.n_cond
        # representative params: sampling / single-traj compat (multi: cfg sample_trajectory)
        self.params = self.data.rep_params
        self.pj = jnp.asarray(self.params)

    def setup_components(self) -> None:
        cfg, m = self.cfg, self.cfg.model
        self.model = WaveletDiT(
            attn_kind=getattr(m, "attn_kind", "phys"), slice_num=getattr(m, "slice_num", 512),
            val_dim=2, n_coord=5, n_cond=self.n_cond, n_hidden=getattr(m, "n_hidden", 256),
            n_layers=getattr(m, "n_layers", 8), n_head=getattr(m, "n_head", 8),
            key=jr.PRNGKey(getattr(cfg, "seed", 0)))
        self.steps_per_epoch = max(1, self.S // cfg.training.batch_size)
        total = cfg.training.n_epochs * self.steps_per_epoch
        peak = cfg.training.learning_rate
        self.schedule = optax.warmup_cosine_decay_schedule(
            0.0, peak, min(500, max(1, total // 20)), total,
            getattr(cfg.training, "final_learning_rate", peak * 1e-2))
        wd = getattr(cfg.training, "weight_decay", 1e-4)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(getattr(cfg.training, "clip_to", 1.0)),
            optax.adamw(self.schedule, weight_decay=wd))
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

    @eqx.filter_jit
    def _train_step(self, model, opt_state, xb, cb, key):
        def loss_fn(m):
            return fm_forward_loss(_model_fn(m), xb, cb, key=key, use_ot=False)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        params, static = eqx.partition(model, eqx.is_array)
        g, _ = eqx.partition(grads, eqx.is_array)
        updates, opt_state = self.optimizer.update(g, opt_state, params)
        model = eqx.combine(eqx.apply_updates(params, updates), static)
        return model, opt_state, loss

    def train_epoch(self, epoch: int, key) -> dict:
        bs = self.cfg.training.batch_size
        perm = jr.permutation(key, self.S)
        losses = []
        for i in range(0, self.S, bs):
            idx = perm[i:i + bs]
            cb = (self.P[idx], self.C[idx])                          # PER-SAMPLE params + coords
            self.model, self.opt_state, l = self._train_step(
                self.model, self.opt_state, self.X[idx], cb, jr.fold_in(key, i))
            losses.append(float(l))
        return {"loss": float(np.mean(losses))}

    def evaluate(self, epoch: int) -> dict:
        # pure-JAX value-space check (no torch). Flux decode is the next wire-up (see sample_values).
        samp = np.asarray(sample_values(self.model, self.C_np, self.params, n_samples=16, seed=epoch))
        return {"samp_val_mean": float(samp.mean()), "samp_val_std": float(samp.std()),
                "real_val_std": float(self.X_np.std())}

    def sample(self, *, key, batch, cond, steps: int = 50):
        return euler_sample(_model_fn(self.model), key=key,
                            shape=(batch, self.N, 2), cond=cond, steps=steps)
