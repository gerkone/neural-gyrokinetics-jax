"""Latent flow matching conditioned on the paired linear-run field.

Same pipeline as :class:`~neugk_jax.diffusion.runner.FlowMatchingRunner` (frozen AE,
precomputed latents, rectified flow) with the scalar ``(itg, dg, s_hat, q)``
conditioning replaced by a code encoded from the trajectory's linear eigenmode field.

The linear field is constant within a trajectory, so ``training.cond_group_size`` lets a
batch draw ``batch_size / g`` trajectories × ``g`` timesteps: the field encoder then runs
once per group instead of once per sample, and the codes are repeated across the group.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from neugk_jax.dataset import LinearCondCycloneDataset
from neugk_jax.diffusion.flow_matching import euler_sample, fm_forward_loss
from neugk_jax.diffusion.lincond_dit import LinearCondDiT, LinearFieldEncoder
from neugk_jax.diffusion.runner import FlowMatchingRunner
from neugk_jax.training.schedulers import warmup_cosine
from neugk_jax.translate import force_f32


class LinearCondFlowMatchingRunner(FlowMatchingRunner):
    """Trains a :class:`LinearCondDiT` on latents, conditioned on linear fields."""

    dataset_cls = LinearCondCycloneDataset

    def setup_data(self) -> None:
        super().setup_data()
        # the val split must normalize with the TRAINING profile, not its own
        prof = getattr(self.train_ds, "linear_profile", None)
        if prof is not None and self.val_ds is not self.train_ds:
            self.val_ds.linear_profile = prof
            self.val_ds._linear_cache.clear()
            self.val_ds._preload_linear(int(self.cfg.dataset.get("linear_workers", 8)))
            if self.dist.is_rank0:
                print(f"val linear fields re-normalized with the train profile {prof.shape}")

    def _dataset_kwargs(self) -> dict:
        dcfg = self.cfg.dataset
        return dict(
            raw_root=dcfg.get("raw_root", "/restricteddata/ukaea/gyrokinetics/raw"),
            linear_roots=dcfg.get("linear_roots"),
            linear_to_real=dcfg.get("linear_to_real", True),
            linear_separate_zf=dcfg.get("linear_separate_zf"),
            linear_normalize=dcfg.get("linear_normalize", "rms"),
            linear_preload=dcfg.get("linear_preload", True),
            linear_cache_size=dcfg.get("linear_cache_size"),
            linear_workers=dcfg.get("linear_workers", 8),
            linear_required=dcfg.get("linear_required", True),
            linear_dtype=dcfg.get("linear_dtype", "float32"),
            linear_rescale_per_traj=dcfg.get("linear_rescale_per_traj", False),
        )

    def setup_components(self) -> None:
        cfg = self.cfg
        mcfg = cfg.model
        key = jr.PRNGKey(getattr(cfg, "seed", 0))
        grid = tuple(self.ae.bottleneck_grid_size)
        z_dim = int(self.ae.bottleneck_dim)
        self.latent_shape = (*grid, z_dim)
        self.cond_group_size = int(cfg.training.get("cond_group_size", 1))
        assert cfg.training.batch_size % self.cond_group_size == 0, \
            "batch_size must be a multiple of cond_group_size"

        ecfg = mcfg.get("linear_encoder", {})
        k_enc, k_dit = jr.split(key, 2)
        encoder = LinearFieldEncoder(
            base_resolution=list(self.train_ds.resolution),
            in_channels=int(self.train_ds.linear_channels),
            patch_size=list(ecfg.get("patch_size", [4, 0, 4, 5, 4])),
            window_size=list(ecfg.get("window_size", [4, 0, 4, 9, 4])),
            dim=int(ecfg.get("dim", 256)),
            depth=list(ecfg.get("depth", [2, 2])),
            num_heads=list(ecfg.get("num_heads", [8, 8])),
            code_dim=int(ecfg.get("code_dim", 256)),
            num_layers=int(ecfg.get("num_layers", 2)),
            decouple_mu=bool(ecfg.get("decouple_mu", True)),
            c_multiplier=int(ecfg.get("c_multiplier", 1)),
            pool=ecfg.get("pool", "max"),
            drop_path=float(ecfg.get("drop_path", 0.0)),
            mlp_ratio=float(ecfg.get("mlp_ratio", 2.0)),
            merging_depth=int(ecfg.get("merging_depth", 2)),
            merging_hidden_ratio=float(ecfg.get("merging_hidden_ratio", 1.0)),
            qkv_bias=bool(ecfg.get("qkv_bias", False)),
            qk_norm=bool(ecfg.get("qk_norm", True)),
            use_rpb=bool(ecfg.get("use_rpb", True)),
            gated_attention=bool(ecfg.get("gated_attention", True)),
            key=k_enc,
        )
        cond_mode = mcfg.get("cond_mode", "adaln")
        self.model = force_f32(LinearCondDiT(
            space=len(grid),
            z_dim=z_dim,
            dim=mcfg.get("latent_dim", 512),
            grid_size=grid,
            depth=mcfg.vit.get("depth", 4),
            num_heads=mcfg.vit.get("num_heads", 8),
            linear_encoder=encoder,
            cond_mode=cond_mode,
            key=k_dit,
            mlp_ratio=mcfg.vit.get("mlp_ratio", 4.0),
            drop_path=mcfg.vit.get("drop_path", 0.0),
        ))
        if self.dist.is_rank0:
            n_par = sum(x.size for x in jax.tree_util.tree_leaves(
                eqx.filter(self.model, eqx.is_array)))
            entry = ("pooled code -> scale/shift/gate" if cond_mode == "adaln"
                     else f"{encoder.grid_sizes[-1]} tokens -> cross-attention")
            print(f"LinearCondDiT: {n_par / 1e6:.2f}M params, cond_mode={cond_mode} "
                  f"({entry}), encoder grids={encoder.grid_sizes}")
            print(f"conditioning: linear field ONLY (no scalar path in the model; the "
                  f"dataset's {len(self.train_ds.conditions)} scalars are logging + "
                  f"latent-cache checks)")

        steps_per_epoch = max(1, len(self.train_ds) // cfg.training.batch_size)
        total = cfg.training.n_epochs * steps_per_epoch
        self.schedule = warmup_cosine(
            peak_lr=cfg.training.learning_rate,
            total_steps=total,
            steps_per_epoch=steps_per_epoch,
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
        self.use_ot = bool(cfg.model.get("minibatch_ot", True))
        self.log_every = int(cfg.training.get("log_every_n_steps", 200))
        self._lin_dev_train = self._device_linear(self.train_ds)
        self._lin_dev_val = self._device_linear(self.val_ds)
        if self.dist.is_rank0 and self._lin_dev_train:
            gb = sum(x.size * x.dtype.itemsize for x in self._lin_dev_train.values()) / 1e9
            print(f"linear fields resident on device: {len(self._lin_dev_train)} trajectories, "
                  f"{gb:.1f} GB")
        # latents on device too: a per-sample host gather is what pins the GPU idle
        self._z_train, self._lin_train_rows = self._device_latents(self.train_ds,
                                                                  self._lin_dev_train)
        self._z_val, self._lin_val_rows = self._device_latents(self.val_ds, self._lin_dev_val)

    def _device_latents(self, ds, lin_dev):
        """Pack the split's latents (and the field cache) into indexable device arrays."""
        if not self.cfg.dataset.get("latents_device_cache", True) or ds.precomputed_latents is None:
            return None, None
        keys = ds.flat_index_to_file_and_tstep
        shape = ds.precomputed_latents[keys[0]]["x"].shape
        host = np.empty((len(keys), *shape), dtype=np.float32)
        for flat, key in keys.items():
            host[flat] = ds.precomputed_latents[key]["x"]
        z = jnp.asarray(host)
        del host
        rows = None
        if lin_dev:
            fids = sorted(lin_dev)
            rows = ({fid: i for i, fid in enumerate(fids)},
                    jnp.stack([lin_dev[f] for f in fids]))
        if self.dist.is_rank0:
            print(f"{ds.split} latents resident on device: {z.shape} "
                  f"({z.size * 4 / 1e9:.1f} GB)")
        return z, rows

    def _device_linear(self, ds) -> dict:
        # one device-resident copy per trajectory; the field never changes

        if not self.cfg.dataset.get("linear_device_cache", True):
            return {}
        return {fid: jnp.asarray(ds.get_linear(fid)) for fid in ds.metadata}

    def _linear_batch(self, ds, dev_cache, samples, fids):
        if dev_cache:
            return jnp.stack([dev_cache[int(f)] for f in fids])
        return jnp.stack([jnp.asarray(s.linear) for s in samples])

    # batching

    def _flat_by_fid(self, ds) -> dict[int, np.ndarray]:
        by_fid = defaultdict(list)
        for flat, (fid, _) in ds.flat_index_to_file_and_tstep.items():
            by_fid[fid].append(flat)
        return {k: np.asarray(v) for k, v in by_fid.items()}

    def _grouped_batches(self, ds, bs: int, g: int, key):
        """Yield lists of ``bs`` flat indices; consecutive blocks of ``g`` share a trajectory."""
        n_steps = max(1, len(ds) // bs)
        if g == 1:
            idx = np.asarray(jr.permutation(key, len(ds)))
            for i in range(n_steps):
                yield idx[i * bs:(i + 1) * bs].tolist()
            return
        by_fid = self._flat_by_fid(ds)
        fids = np.asarray(sorted(by_fid))
        rng = np.random.default_rng(int(jr.randint(key, (), 0, 2**31 - 1)))
        n_groups = bs // g
        for _ in range(n_steps):
            picked = rng.choice(fids, size=n_groups, replace=n_groups > len(fids))
            batch = []
            for fid in picked:
                pool = by_fid[int(fid)]
                batch.extend(rng.choice(pool, size=g, replace=g > len(pool)).tolist())
            yield batch

    @eqx.filter_jit
    def _train_step(self, model, opt_state, latents, lin, key):
        g = self.cond_group_size

        def loss_fn(m):
            codes = jax.vmap(m.encode_cond)(lin)
            if g > 1:
                codes = jnp.repeat(codes, g, axis=0)
            return fm_forward_loss(lambda x, t, c: m(x, t, c), latents, codes, key=key,
                                   latent_scale=self.latent_scale, use_ot=self.use_ot)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        params, static = eqx.partition(model, eqx.is_array)
        g_params, _ = eqx.partition(grads, eqx.is_array)
        updates, opt_state = self.optimizer.update(g_params, opt_state, params)
        params = eqx.apply_updates(params, updates)
        model = eqx.combine(params, static)
        return model, opt_state, loss

    def train_epoch(self, epoch: int, key) -> dict:
        cfg = self.cfg
        bs = cfg.training.batch_size
        g = self.cond_group_size
        idx_key, key = jr.split(key)
        losses = []
        t0 = time.perf_counter()
        fmap = self.train_ds.flat_index_to_file_and_tstep
        for step, sel in enumerate(self._grouped_batches(self.train_ds, bs, g, idx_key)):
            sel = np.asarray(sel)
            # one field per group — the encoder never sees the same trajectory twice per step
            fids = [fmap[int(sel[i])][0] for i in range(0, bs, g)]
            z, lin = self._batch_arrays(self.train_ds, self._z_train,
                                        self._lin_train_rows, self._lin_dev_train, sel, fids, bs, g)
            step_key, key = jr.split(key)
            self.model, self.opt_state, loss = self._train_step(
                self.model, self.opt_state, z, lin, step_key,
            )
            losses.append(loss)                       # device scalar; synced below
            if self.dist.is_rank0 and self.log_every and (step + 1) % self.log_every == 0:
                recent = [float(x) for x in losses[-self.log_every:]]
                print(f"  [train] epoch {epoch} step {step + 1} "
                      f"loss={sum(recent) / len(recent):.4e} "
                      f"{(step + 1) / (time.perf_counter() - t0):.2f} it/s", flush=True)
        losses = [float(x) for x in losses]
        return {"loss": sum(losses) / max(len(losses), 1)}

    def _batch_arrays(self, ds, z_dev, lin_rows, lin_dev, sel, fids, bs, g):
        """(latents, one field per group) for a batch — device gathers when cached."""
        if z_dev is not None and lin_rows is not None:
            fid_to_row, lin_all = lin_rows
            rows = np.asarray([fid_to_row[int(f)] for f in fids])
            return z_dev[jnp.asarray(sel)], lin_all[jnp.asarray(rows)]
        samples = [ds[int(i)] for i in sel]
        z = jnp.stack([jnp.asarray(s.df) for s in samples])
        lin = self._linear_batch(ds, lin_dev, [samples[i] for i in range(0, bs, g)], fids)
        return z, lin

    def evaluate(self, epoch: int) -> dict:
        from neugk_jax.evaluate import DiffusionEvaluator

        def _sample(*, key, batch, cond=None, steps=50):
            return self.sample(key=key, batch=batch, cond=cond, steps=steps)

        cfg = self.cfg
        bs = cfg.training.batch_size
        n = min(len(self.val_ds), bs * 4)
        losses = []
        key = jr.PRNGKey(epoch)
        for start in range(0, n - bs + 1, bs):
            idxs = np.arange(start, start + bs)
            fids = [self.val_ds.flat_index_to_file_and_tstep[int(i)][0] for i in idxs]
            z, lin = self._batch_arrays(self.val_ds, self._z_val, self._lin_val_rows,
                                        self._lin_dev_val, idxs, fids, bs, 1)
            codes = jax.vmap(self.model.encode_cond)(lin)
            step_key, key = jr.split(key)
            losses.append(float(fm_forward_loss(
                lambda x, t, c: self.model(x, t, c),
                z, codes, key=step_key,
                latent_scale=self.latent_scale, use_ot=self.use_ot,
            )))
        out = {"fm_loss": sum(losses) / max(len(losses), 1)}

        if cfg.validation.get("eval_sampling", False):
            ev = DiffusionEvaluator(
                cfg, val_ds=self.val_ds,
                autoencoder=self.ae,
                sample_fn=_sample,
                is_rank0=self.dist.is_rank0,
                cond_field="linear",
            )
            metrics, val_plots = ev(
                self.model, epoch=epoch,
                batch_size=bs,
                n_steps=cfg.validation.get("eval_sample_steps", 50),
                n_samples_per_traj=cfg.validation.get("eval_n_samples", 1),
                eval_integrals=cfg.validation.get("eval_integrals", True),
                eval_spectra=cfg.validation.get("eval_spectra", False),
                max_batches=cfg.validation.get("eval_max_batches", None),
            )
            out.update(metrics)
            if val_plots and self.dist.is_rank0:
                self.logger.log({f"val_plots/{k}": v for k, v in val_plots.items()},
                                step=epoch)
        return out

    def sample(self, *, key, batch: int, cond: Optional[jnp.ndarray] = None, steps: int = 50):
        # encode once — the conditioning is constant along the integration path
        if cond is not None and cond.ndim == self.model.lin_encoder.field_ndim + 1:
            cond = jax.vmap(self.model.encode_cond)(cond)
        latents = euler_sample(
            lambda x, t, c: self.model(x, t, c),
            key=key, shape=(batch, *self.latent_shape),
            cond=cond, steps=steps, latent_scale=self.latent_scale,
        )
        return jax.vmap(self.ae.decode)(latents)
