"""Gyrosplat flow-matching training pipeline.

Trains a GyrosplatDiT on Gaussian → splat-token flow matching, reusing the
latent-diffusion machinery: ``fm_forward_loss``/``euler_sample`` with the
splat-specific hooks (within-set atom pairing, stats-token loss mask, optional
coarse-grid render loss on the predicted clean splat).
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from neugk_jax.dataset.gyrosplat import GyrosplatDataset
from neugk_jax.diffusion.flow_matching import euler_sample, fm_forward_loss
from neugk_jax.gyrosplats.model import GyrosplatDiT
from neugk_jax.gyrosplats.normalize import (
    ZfStats,
    denormalize_tokens,
    denormalize_zf_scalars,
)
from neugk_jax.gyrosplats.pairing import make_pair_fn
from neugk_jax.gyrosplats.splat import bank_structure, tie_group_channels
from neugk_jax.training.runner import BaseRunner
from neugk_jax.training.schedulers import warmup_cosine


class GyrosplatFMRunner(BaseRunner):
    """Flow matching over Gaussian-splat parameter banks."""

    def setup_data(self) -> None:
        cfg = self.cfg
        common = dict(
            cache_path=cfg.dataset.cache_path,
            path=cfg.dataset.get("path"),
            geometry_path=cfg.dataset.get("geometry_path"),
            conditions=tuple(cfg.dataset.get("conditions", ("itg", "dg", "s_hat", "q"))),
            offset=cfg.dataset.get("offset", 80),
            stats_token=cfg.dataset.get("stats_token", False),
            ky_mode=cfg.dataset.get("ky_mode", "delta"),
            asinh_channels=tuple(cfg.dataset.get("asinh_channels", ())),
            asinh_scale=cfg.dataset.get("asinh_scale", 3.0),
            layout=cfg.dataset.get("layout", "atoms"),
            dct=cfg.model.get("dct", False),
            rank=self.dist.process_id,
        )
        # windows layout fits scaffold + channel stats from the TRAIN trajectories;
        # the val set reuses them by fitting on the same training-trajectory list
        stats_traj = cfg.dataset.training_trajectories
        self.train_ds = GyrosplatDataset(
            split="train", trajectories=cfg.dataset.training_trajectories,
            stats_trajectories=stats_traj, **common
        )
        self.val_ds = GyrosplatDataset(
            split="val", trajectories=cfg.dataset.validation_trajectories,
            stats_trajectories=stats_traj, **common
        )

    def setup_components(self) -> None:
        cfg = self.cfg
        mcfg = cfg.model
        ds = self.train_ds
        key = jr.PRNGKey(getattr(cfg, "seed", 0))
        self.layout = ds.layout
        if ds.layout == "windows":
            from neugk_jax.gyrosplats.model import GyrosplatWindowDiT

            self.model = GyrosplatWindowDiT(
                scaffold_mu=np.asarray(ds.scaffold.mu),
                n_env=ds.n_env,
                n_cond=len(ds.conditions),
                n_hidden=mcfg.get("n_hidden", 256),
                n_layers=mcfg.get("n_layers", 8),
                n_head=mcfg.get("n_head", 8),
                mlp_ratio=mcfg.get("mlp_ratio", 2),
                type_dim=mcfg.get("type_dim", 16),
                win_embed_dim=mcfg.get("win_embed_dim", 16),
                mu_fourier_bands=mcfg.get("mu_fourier_bands", 4),
                rope=mcfg.get("rope", False),
                key=key,
            )
        else:
            self.model = GyrosplatDiT(
                bins=ds.bins,
                n_cond=len(ds.conditions),
                n_hidden=mcfg.get("n_hidden", 256),
                n_layers=mcfg.get("n_layers", 8),
                n_head=mcfg.get("n_head", 8),
                mlp_ratio=mcfg.get("mlp_ratio", 2),
                type_dim=mcfg.get("type_dim", 16),
                mu_fourier_bands=mcfg.get("mu_fourier_bands", 4),
                key=key,
            )
        self.steps_per_epoch = max(1, len(ds) // cfg.training.batch_size)
        total = cfg.training.n_epochs * self.steps_per_epoch
        self.schedule = warmup_cosine(
            peak_lr=cfg.training.learning_rate,
            total_steps=total,
            steps_per_epoch=self.steps_per_epoch,
            n_epochs=cfg.training.n_epochs,
            min_lr=cfg.training.get("final_learning_rate", 1e-6),
        )
        wd = cfg.training.get("weight_decay", 0.0)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.training.get("clip_to", 1.0))
            if cfg.training.get("clip_grad", True)
            else optax.identity(),
            optax.adamw(self.schedule, weight_decay=wd) if wd > 0 else optax.adam(self.schedule),
        )
        params, _ = eqx.partition(self.model, eqx.is_array)
        self.opt_state = self.optimizer.init(params)

        if ds.layout == "windows":
            # pairing acts on the envelope rows only (single population of n_env);
            # the n_window carrier tokens are extra tokens that never permute
            self.pair_fn = make_pair_fn(
                np.zeros(ds.n_env, np.int32),
                mode=mcfg.get("pairing", "morton"),
                ot_fraction=mcfg.get("ot_fraction", 0.8),
                blend=mcfg.get("pairing_blend", "perturb"),
                morton_bits=mcfg.get("morton_bits", 6),
                sinkhorn_eps=mcfg.get("sinkhorn_eps", 0.05),
                sinkhorn_iters=mcfg.get("sinkhorn_iters", 100),
                tie_groups=False,
                n_extra_tokens=ds.n_window,
            )
            self.tie_groups = False
        else:
            self.pair_fn = make_pair_fn(
                ds.bins,
                mode=mcfg.get("pairing", "sinkhorn"),
                ot_fraction=mcfg.get("ot_fraction", 0.8),
                blend=mcfg.get("pairing_blend", "perturb"),
                morton_bits=mcfg.get("morton_bits", 6),
                sinkhorn_eps=mcfg.get("sinkhorn_eps", 0.05),
                sinkhorn_iters=mcfg.get("sinkhorn_iters", 100),
                tie_groups=mcfg.get("tie_groups", False),
                n_extra_tokens=int(ds.stats_token),
            )
            self.tie_groups = bool(mcfg.get("tie_groups", False))
            _, grp_idx = bank_structure(ds.bins)
            self._grp_idx = jnp.asarray(grp_idx)

        # flux-carrying phase structure only exists within ~2% of t=1: give the
        # training time distribution a heavy tail there and warp the sampler grid
        tail_frac = float(mcfg.get("t_tail_frac", 0.3))
        tail_lo = float(mcfg.get("t_tail_lo", 0.9))
        self.time_warp = float(mcfg.get("time_warp", 3.0))

        def time_fn(key, batch):
            k1, k2, k3 = jr.split(key, 3)
            base = jax.nn.sigmoid(jr.normal(k1, (batch,)))
            tail = jr.uniform(k2, (batch,), minval=tail_lo, maxval=1.0)
            use_tail = jr.bernoulli(k3, p=tail_frac, shape=(batch,))
            return jnp.where(use_tail, tail, base)

        self.time_fn = time_fn
        self.loss_mask = jnp.asarray(ds.loss_mask)

        self._bins = jnp.asarray(ds.bins)
        self._token_stats = ds.token_stats


        # the dataset is tiny (~60 mb normalized) — keep it resident on device;
        # the per-sample host normalize path costs more than the whole train step
        self._train_tokens, self._train_cond = self._device_dataset(self.train_ds)
        self._val_tokens, self._val_cond = self._device_dataset(self.val_ds)

    @staticmethod
    def _device_dataset(ds):
        toks = jnp.asarray(np.stack([ds[i].tokens for i in range(len(ds))]))
        cond = jnp.asarray(np.stack([ds[i].conditioning for i in range(len(ds))]))
        return toks, cond


    @eqx.filter_jit
    def _train_step(self, model, opt_state, tokens, cond, key):
        def loss_fn(m):
            return fm_forward_loss(
                lambda x, t, c: m(x, t, c),
                tokens,
                cond,
                key=key,
                use_ot=False,
                pair_fn=self.pair_fn,
                loss_mask=self.loss_mask,
                time_fn=self.time_fn,
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        params, static = eqx.partition(model, eqx.is_array)
        g_params, _ = eqx.partition(grads, eqx.is_array)
        updates, opt_state = self.optimizer.update(g_params, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return eqx.combine(params, static), opt_state, loss


    def train_epoch(self, epoch: int, key) -> dict:
        cfg = self.cfg
        bs = cfg.training.batch_size
        n = self._train_tokens.shape[0]
        idx_key, key = jr.split(key)
        idx = jr.permutation(idx_key, n)
        losses = []
        for start in range(0, n - bs + 1, bs):
            bidx = idx[start : start + bs]
            step_key, key = jr.split(key)
            self.model, self.opt_state, loss = self._train_step(
                self.model,
                self.opt_state,
                self._train_tokens[bidx],
                self._train_cond[bidx],
                step_key,
            )
            losses.append(loss)
        return {"loss": float(sum(losses)) / max(len(losses), 1)}

    def evaluate(self, epoch: int) -> dict:
        cfg = self.cfg
        bs = cfg.training.batch_size
        n = min(self._val_tokens.shape[0], bs * 4)
        losses = []
        key = jr.PRNGKey(epoch)
        for start in range(0, n - bs + 1, bs):
            tokens = self._val_tokens[start : start + bs]
            cond = self._val_cond[start : start + bs]
            step_key, key = jr.split(key)
            losses.append(
                float(
                    fm_forward_loss(
                        lambda x, t, c: self.model(x, t, c),
                        tokens,
                        cond,
                        key=step_key,
                        use_ot=False,
                        pair_fn=self.pair_fn,
                        loss_mask=self.loss_mask,
                    )
                )
            )
        out = {"fm_loss": sum(losses) / max(len(losses), 1)}

        if cfg.validation.get("eval_sampling", False):
            from neugk_jax.gyrosplats.eval import GyrosplatEvaluator

            ev = GyrosplatEvaluator(
                cfg, val_ds=self.val_ds, sample_fn=self.sample, is_rank0=self.dist.is_rank0
            )
            metrics, val_plots = ev(
                self.model,
                epoch=epoch,
                batch_size=min(bs, 8),
                n_steps=cfg.validation.get("eval_sample_steps", 50),
                n_ensemble=cfg.validation.get("eval_n_samples", 32),
                gt_stride=cfg.validation.get("eval_gt_stride", 12),
                max_trajectories=cfg.validation.get("eval_max_trajectories", None),
            )
            out.update(metrics)
            if val_plots and self.dist.is_rank0:
                self.logger.log(
                    {f"val_plots/{k}": v for k, v in val_plots.items()}, step=epoch
                )
        return out

    def sample(self, *, key, batch: int, cond: Optional[jnp.ndarray] = None, steps: int = 50):
        """Sample normalized token banks (batch, n_tokens, 17)."""
        prior_fn = None
        if self.tie_groups:

            def prior_fn(k, shape):
                x0 = jr.normal(k, shape)
                return jax.vmap(lambda xb: tie_group_channels(xb, self._grp_idx))(x0)

        return euler_sample(
            lambda x, t, c: self.model(x, t, c),
            key=key,
            shape=(batch, self.train_ds.n_tokens, self.train_ds.n_channels),
            cond=cond,
            steps=steps,
            prior_fn=prior_fn,
            time_warp=self.time_warp,
        )

    def decode(self, tokens, fid: int = 0):
        """One sampled bank -> (SplatParams, ZfStats) in physical units.

        Without a stats token the zf stats come from the trajectory's time
        mean (teacher substitute until the theta-regressor lands)."""
        if self.train_ds.layout == "windows":
            p = self.train_ds.decode_state(tokens)
            zf = self.val_ds.traj_mean_zf_stats(fid)
            return p, ZfStats(*[jnp.asarray(v) for v in zf], "zf")
        if self.train_ds.stats_token:
            p = denormalize_tokens(tokens[:-1], self._bins, self._token_stats)
            st = denormalize_zf_scalars(tokens[-1, :3], self._token_stats)
            return p, st
        p = denormalize_tokens(tokens, self._bins, self._token_stats)
        zf = self.val_ds.traj_mean_zf_stats(fid)
        st = ZfStats(*[jnp.asarray(v) for v in zf], "zf")
        return p, st
