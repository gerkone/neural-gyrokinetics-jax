"""GyroSwin evaluator: autoregressive rollout metrics on denormalized data.

Each validation sample is rolled out for up to ``n_eval_steps`` steps, capped
per trajectory; targets are looked up by ``(file_index, timestep_index + t)``.
Logs ``{field}_x{t}`` per step (relative-norm MSE for df/phi, MSE for
flux/fluxavg, optional ``phi_int``/``flux_int``) plus their step mean ``df``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.evaluate.base import BaseEvaluator
from neugk_jax.evaluate.integrals import flux_integral
from neugk_jax.training.ddp import eval_batch_owner
from neugk_jax.utils import recombine_zf


def _rel_norm_mse(p: np.ndarray, y: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p, y = p.reshape(p.shape[0], -1), y.reshape(y.shape[0], -1)
    return np.sum((p - y) ** 2, -1) / (np.sum(y**2, -1) + eps)


@eqx.filter_jit
def _fwd(m, x, c):
    return jax.vmap(lambda xi, ci: m(xi, ci, inference=True))(x, c)


@functools.partial(jax.jit, static_argnums=3)
def _integrate(g, d, p, real_potens):
    return jax.vmap(lambda gi, di, pi: flux_integral(gi, di, pi, real_potens=real_potens))(g, d, p)


def _process_local(tree):
    # a process-local copy of replicated global arrays, so owners can evaluate independently
    def local(x):
        if isinstance(x, jax.Array) and not x.is_fully_addressable:
            return x.addressable_data(0)
        return x
    return jax.tree_util.tree_map(local, tree)


class GyroSwinEvaluator(BaseEvaluator):
    """``n_eval_steps`` autoregressive rollout over the (tail-cropped) validation set."""

    def __call__(self, model: Any, *, epoch: int, batch_size: int = 1, dist=None,
                 outputs: Optional[Sequence[str]] = None,
                 geometry_fn: Optional[Callable] = None, **_):
        ds = self.val_ds
        n_eval = int(self.cfg.validation.get("n_eval_steps", 1))
        outputs = tuple(outputs or ("df", "phi"))
        fields = [k for k in ("df", "phi", "flux", "fluxavg") if k in outputs]
        eval_integrals = (bool(self.cfg.validation.get("eval_integrals", False))
                          and set(outputs) == {"df", "phi", "flux"} and geometry_fn is not None)
        real_potens = bool(self.cfg.dataset.get("real_potens", True))
        t_slot = ds.conditions.index("timestep") if "timestep" in ds.conditions else None
        if jax.process_count() > 1:
            model = _process_local(model)

        names = fields + (["phi_int", "flux_int"] if eval_integrals else [])
        running = {f"{k}_x{t}": 0.0 for k in names for t in range(1, n_eval + 1)}
        running.update({f"_n_x{t}": 0.0 for t in range(1, n_eval + 1)})
        val_plots: dict[str, object] = {}
        plot_drawn = False

        n = len(ds)
        for bi, start in enumerate(range(0, n, batch_size)):
            if dist is not None and not eval_batch_owner(dist, bi):
                continue
            samples = [ds[i] for i in range(start, min(start + batch_size, n))]
            fids = np.asarray([int(s.file_index) for s in samples])
            t0 = np.asarray([int(s.timestep_index) for s in samples])
            steps = np.minimum(n_eval, np.asarray([ds.num_ts(f) for f in fids]) - t0 - 1)
            x = jnp.stack([jnp.asarray(s.df) for s in samples])
            cond = None
            if samples[0].conditioning is not None:
                cond = np.stack([np.asarray(s.conditioning) for s in samples])
            geom = geometry_fn(fids) if eval_integrals else None

            for t in range(int(steps.max())):
                if t_slot is not None:
                    cond[:, t_slot] = [ds.get_timestep(f, ti + t) for f, ti in zip(fids, t0)]
                preds = _fwd(model, x, None if cond is None else jnp.asarray(cond))
                live = np.nonzero(steps > t)[0]
                tg = [ds.get_at_time(fids[b], t0[b] + t, normalized=False) for b in live]
                pred_d = {k: np.stack([np.asarray(ds.denormalize(int(fids[b]), **{k: np.asarray(preds[k][b])}))
                                       for b in live]) for k in fields}
                tgt_d = {k: np.stack([np.asarray(getattr(s, f"y_{k}")) for s in tg]) for k in fields}
                if "df" in fields:
                    pred_d["df"] = recombine_zf(pred_d["df"], axis=1)
                    tgt_d["df"] = recombine_zf(tgt_d["df"], axis=1)
                step = {}
                for k in fields:
                    p, y = pred_d[k], tgt_d[k].reshape(pred_d[k].shape)
                    step[k] = _rel_norm_mse(p, y) if k in ("df", "phi") else \
                        np.mean((p - y).reshape(len(live), -1) ** 2, -1)
                if eval_integrals:
                    g = {k: v[live] for k, v in geom.items()}
                    phi_i, (pflux, eflux, _) = _integrate(g, jnp.asarray(pred_d["df"]),
                                                          jnp.asarray(pred_d["phi"]), real_potens)
                    phi_i = np.asarray(phi_i)
                    tphi = tgt_d["phi"].reshape(phi_i.shape)
                    step["phi_int"] = np.mean((phi_i - tphi).reshape(len(live), -1) ** 2, -1)
                    step["flux_int"] = (np.asarray(pflux) ** 2
                                        + (np.asarray(eflux) - tgt_d["flux"].reshape(-1)) ** 2)
                for k, v in step.items():
                    running[f"{k}_x{t + 1}"] += float(np.sum(v))
                running[f"_n_x{t + 1}"] += float(len(live))

                if not plot_drawn and self.is_rank0 and t == 0:
                    plot_drawn = True
                    from neugk_jax.evaluate.plots import generate_val_plots
                    roll = {k: pred_d[k][0] for k in ("df", "phi") if k in pred_d}
                    gt = {k: tgt_d[k][0] for k in ("df", "phi") if k in tgt_d}
                    val_plots.update(generate_val_plots(
                        rollout=roll, gt=gt, phase="random draw",
                        ts=np.asarray(samples[live[0]].timestep).reshape(-1)))
                x = preds["df"]

        running, _ = self._sync(running, 0.0)
        metrics = {}
        for t in range(1, n_eval + 1):
            cnt = running[f"_n_x{t}"]
            if cnt > 0:
                for k in names:
                    metrics[f"{k}_x{t}"] = running[f"{k}_x{t}"] / cnt
        for k in names:
            per_step = [v for m, v in metrics.items() if m.startswith(f"{k}_x")]
            if per_step:
                metrics[k] = float(np.mean(per_step))
        return metrics, val_plots
