"""Neural-field training: one ``MLPNF`` per ``(trajectory, timestep)`` snapshot, in vmapped pools.

``NFRunner`` enumerates the snapshots of ``dataset.validation_trajectories`` x ``dataset.timesteps``
(or the per-trajectory ``t0..t1`` windows, step 2, of a ``dataset.transition_windows`` json) and
trains them ``training.pool_size`` per device at a time, the pool stacked along the data axis
of the mesh. Each NF fits its own df, z-scored per (channel, mu):

1. ``density``: MSE on shuffled point batches over a growing subsample of the grid, AdamW
   with a per-epoch cosine.
2. ``pinc``: the decoded field against the same PINC terms as the AE fine-tune
   (:func:`neugk_jax.pinc.losses.pinc_terms`, the trajectory's spectral stats) plus the df
   MSE, AdamW with warmup-cosine; the parameters of the best PSNR(phi) are kept per NF.

Checkpoints ``<prefix><name>_<traj>_t<t>_x<cr>.eqx`` (prefix ``best_`` density, ``int_`` /
``best_int_`` pinc final / best) go to ``training.ckpt_dir`` (default ``output_path``); finished
snapshots are skipped, and snapshots with a density field in ``training.init_from`` (default: the
checkpoint directory) start the pinc phase from it, so a run resumes by restarting it.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from omegaconf import OmegaConf

from neugk_jax.dataset.factory import build_dataset
from neugk_jax.evaluate.base import GeometryCache
from neugk_jax.evaluate.integrals import spectral_integrals
from neugk_jax.pinc.losses import pinc_terms
from neugk_jax.pinc.neural_field import build_nf, grid_coords, n_params, sample_field
from neugk_jax.pinc.peft import SPECTRUM_STATS
from neugk_jax.pinc.runner import read_loss_weights
from neugk_jax.training.checkpoint import load_model_only, save_model_only
from neugk_jax.training.ddp import init_distributed, shard_batch
from neugk_jax.training.logging import Logger
from neugk_jax.training.runner import build_optimizer, configure_compilation_cache
from neugk_jax.utils import to_dict

LOSS_KEYS = ("df", "phi_int", "flux_int", "kyspec", "qspec")


def snapshot_norm(df: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(scale, shift)`` z-score of a ``(2, vp, mu, s, x, y)`` df, per (channel, mu)."""
    axes = (1, 3, 4, 5)
    shift = jnp.mean(df, axis=axes, keepdims=True)
    scale = jnp.std(df, axis=axes, keepdims=True, ddof=1)
    return jnp.maximum(scale, 1e-12), shift


def epoch_cosine(lr: float, min_lr: float, steps_per_epoch) -> optax.Schedule:
    """Cosine over epochs, constant within one (torch ``CosineAnnealingLR`` stepped per epoch)."""
    n = len(steps_per_epoch)
    lrs = jnp.asarray(
        [min_lr + (lr - min_lr) * (1 + math.cos(math.pi * e / n)) / 2 for e in range(n)]
    )
    bounds = jnp.cumsum(jnp.asarray(steps_per_epoch))
    return lambda count: lrs[jnp.minimum(jnp.searchsorted(bounds, count, side="right"), n - 1)]


def psnr(pred, target):
    return 10.0 * jnp.log10(jnp.max(target) ** 2 / jnp.mean((pred - target) ** 2))


def local_rows(tree):
    """Host numpy of the process-local rows of the row-sharded arrays of ``tree``."""

    def rows(x):
        if not isinstance(x, jax.Array):
            return x
        shards = sorted(x.addressable_shards, key=lambda s: s.index[0].start or 0)
        return np.concatenate([np.asarray(s.data) for s in shards])

    return jax.tree_util.tree_map(rows, tree)


@eqx.filter_jit(donate="all-except-first")
def density_chunk(inputs, models, opt_state, optimizer, n_steps: int):
    """``n_steps`` AdamW steps of the pool on point batches ``perm[:, step * B : (step + 1) * B]``."""
    fields, perm, start, batch = inputs
    grid, n = fields.shape[2:], perm.shape[1]
    params, static = eqx.partition(models, eqx.is_array)

    def one(m, f, idx):
        return jnp.mean((m(grid_coords(grid, idx)) - f.reshape(2, -1)[:, idx].T) ** 2)

    def step(carry, j):
        params, opt_state = carry
        offs = (start + j) * batch.shape[0]
        idx = jnp.take(perm, (offs + batch) % n, axis=1)

        def loss(p):
            return jnp.sum(eqx.filter_vmap(one)(eqx.combine(p, static), fields, idx))

        value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), value

    (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), jnp.arange(n_steps))
    return eqx.combine(params, static), opt_state, losses


def pinc_eval(model, data, *, ds: float, weights: dict, loss_types: dict, physical_df: bool):
    """PINC terms, weighted total and PSNR(phi) of one NF against its precomputed targets."""
    pred = sample_field(model, data["df"].shape[1:])
    pred_d = pred * data["scale"] + data["shift"]
    p = spectral_integrals(data["geom"], pred_d, ds=ds)
    batched = jax.tree_util.tree_map(lambda a: a[None], (p, data["tgt"]))
    terms = pinc_terms(*batched, data["mode_std"], loss_types)
    err = (pred - data["df"]) * (data["scale"] if physical_df else 1.0)
    terms["df"] = jnp.mean(err**2)
    pot = loss_types.get("potential", "phi")
    terms["phi_psnr"] = jax.lax.stop_gradient(psnr(p[pot], data["tgt"][pot]))
    return sum(w * terms[k] for k, w in weights.items()), terms


@eqx.filter_jit(donate="all-except-first")
def pinc_step(inputs, models, opt_state, best, optimizer, loss_cfg: dict):
    """One AdamW step of the pool on the PINC loss; keeps each NF's best-PSNR(phi) parameters.

    ``loss_cfg`` holds the static :func:`pinc_eval` options.
    """
    data = inputs
    params, static = eqx.partition(models, eqx.is_array)

    def loss(p):
        totals, terms = eqx.filter_vmap(lambda m, d: pinc_eval(m, d, **loss_cfg))(
            eqx.combine(p, static), data
        )
        return jnp.sum(totals), terms

    (_, terms), grads = jax.value_and_grad(loss, has_aux=True)(params)
    best_params, best_score = best
    better = terms["phi_psnr"] > best_score

    def pick(new, old):
        return jnp.where(better.reshape(-1, *[1] * (new.ndim - 1)), new, old)

    best = (
        jax.tree_util.tree_map(pick, params, best_params),
        jnp.maximum(terms["phi_psnr"], best_score),
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.combine(optax.apply_updates(params, updates), static), opt_state, best, terms


@eqx.filter_jit
def target_integrals(data, ds: float):
    phys = data["df"] * data["scale"] + data["shift"]
    return jax.vmap(lambda d, g: spectral_integrals(g, d, ds=ds))(phys, data["geom"])


@eqx.filter_jit
def permutations(keys, n: int):
    return jax.vmap(lambda k: jr.permutation(k, n))(keys)


def copy_arrays(tree):
    return jax.tree_util.tree_map(jnp.copy, eqx.filter(tree, eqx.is_array))


@eqx.filter_jit
def final_scores(models, data, ds: float, pot: str):
    def one(m, d):
        pred = sample_field(m, d["df"].shape[1:]) * d["scale"] + d["shift"]
        return psnr(spectral_integrals(d["geom"], pred, ds=ds)[pot], d["tgt"][pot])

    return eqx.filter_vmap(one)(models, data)


class NFRunner:
    """Trains every snapshot NF of the configured trajectories and timesteps, pool by pool."""

    def __init__(self, cfg, *, output_path: str | None = None):
        self.cfg = cfg
        configure_compilation_cache(cfg)
        self.dist = init_distributed()
        self.tcfg = cfg.training
        self.output_path = Path(cfg.training.get("ckpt_dir") or output_path or cfg.output_path)
        self.init_from = Path(cfg.training.get("init_from") or self.output_path)
        self.logger = Logger(
            is_rank0=self.dist.is_rank0, config=to_dict(cfg), logging=to_dict(cfg.get("logging"))
        )
        dcfg = cfg.dataset
        self.ds = build_dataset(
            dcfg,
            split="val",
            dist=self.dist,
            normalization=None,
            normalization_stats=None,
            cond_filters=None,
            conditions=(),
            offset=0,
        )
        self.spec_offset = int(dcfg.get("spectral_offset", 80))
        self.geoms = GeometryCache(self.ds)
        self.grid = tuple(int(r) for r in self.ds.resolution)
        pcfg = self.tcfg.pinc
        self.weights = read_loss_weights(
            OmegaConf.create({"loss_weights": pcfg.loss_weights}), LOSS_KEYS
        )
        self.ds_spacing = self._uniform_ds()
        self.loss_cfg = dict(
            ds=self.ds_spacing,
            weights=self.weights,
            loss_types=to_dict(pcfg.get("loss_types")),
            physical_df=pcfg.get("df_units", "normalized") == "physical",
        )
        self.name = cfg.model.get("name", "mlp")
        probe = build_nf(cfg.model, self.grid, key=jr.PRNGKey(0))
        self.cr = 2 * np.prod(self.grid) / n_params(probe)
        self.jobs = self._jobs()
        if self.dist.is_rank0:
            self.output_path.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, self.output_path / "config.yaml")
            print(
                f"{len(self.jobs)} snapshots to train, cr {self.cr:.1f}x, {n_params(probe)} params"
            )

    def _uniform_ds(self) -> float:
        vals = {self.ds.get_ds(f) for f in range(len(self.ds.files))}
        if None in vals or len(vals) != 1:
            raise ValueError(f"neural fields need one 'ds' in every trajectory, got {vals}")
        return float(vals.pop())

    def traj_name(self, fid: int) -> str:
        name = Path(self.ds.files[fid]).name.split(".")[0]
        return name.replace("_ifft", "").replace("_realpotens", "")

    def ckpt_path(self, prefix: str, fid: int, t: int, root: Path | None = None) -> Path:
        name = f"{prefix}{self.name}_{self.traj_name(fid)}_t{t}_x{int(self.cr)}.eqx"
        return (root or self.output_path) / name

    def timesteps(self, fid: int) -> list[int]:
        windows = self.cfg.dataset.get("transition_windows")
        if not windows:
            return [int(t) for t in self.cfg.dataset.timesteps]
        with open(windows) as f:
            w = json.load(f).get(self.traj_name(fid))
        return [] if w is None or w.get("bad") else list(range(int(w["t0"]), int(w["t1"]) + 1, 2))

    def _jobs(self) -> list[tuple[int, int]]:
        jobs = [
            (fid, t)
            for fid in range(len(self.ds.files))
            for t in self.timesteps(fid)
            if t < self.ds.num_ts(fid) and not self.ckpt_path("best_int_", fid, t).exists()
        ]
        # warm-startable snapshots first, so whole pools skip the density phase
        jobs.sort(key=lambda j: not self.has_density(*j))
        # this process's share; pools are sharded over its devices
        return jobs[self.dist.process_id :: self.dist.num_processes]

    def has_density(self, fid: int, t: int) -> bool:
        return self.ckpt_path("best_", fid, t, self.init_from).exists()

    def load_pool(self, jobs):
        template = build_nf(self.cfg.model, self.grid, key=jr.PRNGKey(0))
        models = [
            load_model_only(self.ckpt_path("best_", f, t, self.init_from), template)
            for f, t in jobs
        ]
        stacked = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *[eqx.filter(m, eqx.is_array) for m in models]
        )
        return shard_batch(
            self.dist, eqx.combine(stacked, eqx.partition(template, eqx.is_array)[1])
        )

    def pool_data(self, jobs) -> dict:
        """Stacked per-NF fields, normalization, geometry, spectral stats and pinc targets."""
        dfs = jnp.stack([jnp.asarray(self.ds.read_frame(f, t)["df"]) for f, t in jobs])
        scale, shift = jax.vmap(snapshot_norm)(dfs)
        fids = [f for f, _ in jobs]
        geom = self.geoms.stack(fids)
        mode_std = {
            k: np.stack([self.ds.spectral_stats(src, [f], self.spec_offset)["std"] for f in fids])
            for k, src in SPECTRUM_STATS.items()
        }
        data = {
            "df": (dfs - shift) / scale,
            "scale": scale,
            "shift": shift,
            "geom": geom,
            "mode_std": mode_std,
        }
        data = shard_batch(self.dist, data)
        data["tgt"] = target_integrals(data, self.ds_spacing)
        return data

    def init_pool(self, key, n: int):
        keys = jr.split(key, n)
        models = eqx.filter_vmap(lambda k: build_nf(self.cfg.model, self.grid, key=k))(keys)
        return shard_batch(self.dist, models)

    def density(self, models, data, key):
        dcfg = self.tcfg.density
        n = int(np.prod(self.grid))
        batch, chunk = int(dcfg.batch_size), int(dcfg.get("scan_steps", 64))
        subs = np.linspace(*dcfg.subsample, int(dcfg.epochs))
        steps = [chunk * max(1, math.ceil(math.ceil(n * s / batch) / chunk)) for s in subs]
        tcfg = {"weight_decay": dcfg.weight_decay, "clip_grad": False}
        opt = build_optimizer(
            epoch_cosine(dcfg.lr, dcfg.min_lr, steps), tcfg, models, decoupled=True
        )
        opt_state = opt.init(eqx.filter(models, eqx.is_array))
        arange = jnp.arange(batch)
        for e, n_steps in enumerate(steps):
            keys = jr.split(jr.fold_in(key, e), data["df"].shape[0])
            perm = permutations(shard_batch(self.dist, keys), n)
            losses = []
            for c in range(n_steps // chunk):
                models, opt_state, loss = density_chunk(
                    (data["df"], perm, jnp.int32(c * chunk), arange), models, opt_state, opt, chunk
                )
                losses.append(loss)
            self.log({"density/mse": float(jnp.mean(jnp.stack(losses))) / data["df"].shape[0]})
        return models

    def pinc(self, models, data):
        pcfg = self.tcfg.pinc
        schedule = optax.warmup_cosine_decay_schedule(
            0.0, pcfg.lr, int(pcfg.warmup), max(int(pcfg.epochs), int(pcfg.warmup) + 1), pcfg.min_lr
        )
        tcfg = {"weight_decay": pcfg.weight_decay, "clip_grad": False}
        opt = build_optimizer(schedule, tcfg, models, decoupled=True)
        opt_state = opt.init(eqx.filter(models, eqx.is_array))
        n_nf = data["df"].shape[0]
        best = (copy_arrays(models), jnp.full((n_nf,), -jnp.inf))
        for _ in range(int(pcfg.epochs)):
            models, opt_state, best, terms = pinc_step(
                data, models, opt_state, best, opt, self.loss_cfg
            )
            self.log({f"pinc/{k}": float(jnp.mean(v)) for k, v in terms.items()})
        final = final_scores(
            models, data, self.ds_spacing, self.loss_cfg["loss_types"].get("potential", "phi")
        )
        static = eqx.partition(models, eqx.is_array)[1]
        last_better = final > best[1]
        best_params = jax.tree_util.tree_map(
            lambda new, old: jnp.where(last_better.reshape(-1, *[1] * (new.ndim - 1)), new, old),
            eqx.filter(models, eqx.is_array),
            best[0],
        )
        return models, eqx.combine(best_params, static), jnp.maximum(final, best[1])

    def log(self, logs: dict) -> None:
        self._step += 1
        self.logger.log(logs, step=self._step)

    def save(self, jobs, n_real: int, named_models: dict) -> None:
        for prefix, models in named_models.items():
            rows = local_rows(models)
            for i, (fid, t) in enumerate(jobs[:n_real]):
                one = jax.tree_util.tree_map(
                    lambda x: x[i] if isinstance(x, np.ndarray) else x, rows
                )
                save_model_only(self.ckpt_path(prefix, fid, t), one)

    def __call__(self) -> None:
        pool = int(self.tcfg.pool_size) * self.dist.local_device_count
        key = jr.PRNGKey(int(self.cfg.get("seed", 0)))
        self._step = 0
        for p, start in enumerate(range(0, len(self.jobs), pool)):
            t0 = time.perf_counter()
            jobs = self.jobs[start : start + pool]
            n_real = len(jobs)
            # pad the last pool with repeats; only the real snapshots are saved
            jobs = jobs + [jobs[-1]] * (pool - n_real)
            data = self.pool_data(jobs)
            if all(self.has_density(*j) for j in jobs):
                models = self.load_pool(jobs)
            else:
                models = self.init_pool(jr.fold_in(key, 2 * p), pool)
                models = self.density(models, data, jr.fold_in(key, 2 * p + 1))
                self.save(jobs, n_real, {"best_": models})
            score = math.nan
            if self.weights and int(self.tcfg.pinc.epochs) > 0:
                last, best, score = self.pinc(models, data)
                self.save(jobs, n_real, {"int_": last, "best_int_": best})
                score = float(jnp.mean(score[:n_real]))
            if self.dist.is_rank0:
                print(
                    f"pool {p}: {n_real} nfs ({start + n_real}/{len(self.jobs)}), "
                    f"psnr(phi) {score:.2f}, {time.perf_counter() - t0:.0f}s"
                )
        self.logger.finish()
