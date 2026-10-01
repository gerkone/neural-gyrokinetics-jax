"""GyroSwin evaluator: autoregressive rollout metrics on denormalized data.

Each validation sample is rolled out for up to ``n_eval_steps`` steps, capped per
trajectory; step ``t`` targets are the next-step targets at ``timestep_index + t``.
Logs ``{field}_x{t}`` per step (relative-norm MSE for df/phi, MSE for flux/fluxavg,
optional ``phi_int`` (MSE of the integrated phi) and ``flux_int_rel_err``
(``|eflux - flux| / |flux|`` of the integrated heat flux)), ``df_rel_l2_x{t}``/
``phi_rel_l2_x{t}`` and the step means ``{field}``. Batches keep one shape; rollout steps beyond a trajectory are masked.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.evaluate.base import BaseEvaluator, integrate, per_sample_mse, per_sample_rel_l2
from neugk_jax.training.data import stack_fields
from neugk_jax.training.ddp import replicate_local
from neugk_jax.utils import count_trace, recombine_zf

TARGETS = ("df", "phi", "flux", "fluxavg")


def _rel_norm_mse(p, y, eps: float = 1e-4):
    p, y = p.reshape(p.shape[0], -1), y.reshape(y.shape[0], -1)
    return jnp.sum((p - y) ** 2, -1) / (jnp.sum(y**2, -1) + eps)


@eqx.filter_jit
def gyroswin_eval_step(model, x, cond, tgt, fids, live, t, acc, denorm, geom, fields,
                       real_potens: bool):
    """One rollout step; adds the ``live``-masked metric sums at row ``t`` of ``acc``."""
    count_trace("gyroswin_eval_step")
    preds = jax.vmap(lambda xi, ci: model(xi, ci, inference=True))(x, cond)
    pred_d = {k: denorm(k, preds[k], fids) for k in fields}
    tgt_d = {k: denorm(k, tgt[k].reshape(preds[k].shape), fids) for k in fields}
    if "df" in fields:
        pred_d["df"], tgt_d["df"] = recombine_zf(pred_d["df"], axis=1), recombine_zf(tgt_d["df"], axis=1)
    values = {}
    for k in fields:
        if k in ("df", "phi"):
            values[k] = _rel_norm_mse(pred_d[k], tgt_d[k])
            values[f"{k}_rel_l2"] = per_sample_rel_l2(pred_d[k], tgt_d[k])
        else:
            values[k] = per_sample_mse(pred_d[k], tgt_d[k])
    if geom is not None:
        phi_i, (_, eflux, _) = integrate(geom, fids, pred_d["df"], pred_d["phi"],
                                         real_potens=real_potens)
        flux = tgt_d["flux"].reshape(-1)
        values["phi_int"] = per_sample_mse(phi_i, tgt_d["phi"].reshape(phi_i.shape))
        values["flux_int_rel_err"] = jnp.abs(eflux - flux) / (jnp.abs(flux) + 1e-12)
    out = {k: acc[k].at[t].add(jnp.sum(v * live)) for k, v in values.items()}
    out["_n"] = acc["_n"].at[t].add(jnp.sum(live))
    return preds["df"], out, pred_d, tgt_d


class GyroSwinEvaluator(BaseEvaluator):
    """``n_eval_steps`` autoregressive rollout over the (tail-cropped) validation set."""

    denorm_fields = TARGETS

    def __init__(self, cfg: Any, *, outputs: Optional[Sequence[str]] = None, **kwargs):
        super().__init__(cfg, **kwargs)
        ds = self.ds
        self.n_eval = int(self.vcfg.get("n_eval_steps", 1))
        outputs = tuple(outputs or ("df", "phi"))
        self.fields = tuple(k for k in TARGETS if k in outputs)
        self.eval_integrals = (bool(self.vcfg.get("eval_integrals", False))
                               and set(outputs) == {"df", "phi", "flux"})
        self.real_potens = bool(ds.real_potens)
        self.t_slot = ds.conditions.index("timestep") if "timestep" in ds.conditions else None
        names = []
        for k in self.fields:
            names += [k, f"{k}_rel_l2"] if k in ("df", "phi") else [k]
        self.names = tuple(names + (["phi_int", "flux_int_rel_err"] if self.eval_integrals else []))

    def load(self, ds, indices, read):
        return stack_fields(read(ds, indices), ("df", "conditioning", "file_index", "timestep",
                                                *(f"y_{k}" for k in TARGETS)))

    def _targets(self, fids, t0, t, steps) -> dict:
        tt = t0 + np.minimum(t, np.maximum(steps - 1, 0))
        rows = self.loader.map(lambda ft: self.ds.get_target(*ft), list(zip(fids, tt)))
        return {k: np.stack([np.asarray(r[k]) for r in rows]) for k in self.fields}

    def __call__(self, model: Any, *, epoch: int) -> tuple[dict[str, float], dict[str, Any]]:
        ds, n_eval = self.ds, self.n_eval
        model = self.local_model(model)
        geom = self.geometry if self.eval_integrals else None
        acc = replicate_local(self.dist, {k: jnp.zeros((n_eval,), jnp.float32)
                                          for k in (*self.names, "_n")})
        plots: dict[str, Any] = {}
        for plan, batch, _ in self.loader.iterate(ds, self.plans, self.load, self.place):
            ft = [ds.flat_index_to_file_and_tstep[int(i)] for i in plan.indices]
            fids = np.asarray([f for f, _ in ft])
            t0 = np.asarray([t for _, t in ft])
            steps = np.minimum(n_eval, np.asarray([ds.num_ts(f) for f in fids]) - t0 - 1)
            x, cond = batch["df"], batch.get("conditioning")
            tgt = {k: batch[f"y_{k}"] for k in self.fields}
            for t in range(int(steps[plan.mask > 0].max())):
                if t > 0:
                    tgt = self.place(self._targets(fids, t0, t, steps))
                if self.t_slot is not None:
                    ts = np.asarray([ds.get_timestep(f, ti + t) for f, ti in zip(fids, t0)])
                    cond = cond.at[:, self.t_slot].set(self.place({"ts": ts})["ts"])
                live = self.place({"m": (plan.mask * (steps > t)).astype(np.float32)})["m"]
                x, acc, pred_d, tgt_d = gyroswin_eval_step(
                    model, x, cond, tgt, batch["file_index"], live, jnp.int32(t), acc,
                    self.denorm, geom, self.fields, self.real_potens)
                if plan.number == 0 and t == 0 and self.is_rank0:
                    plots = self._plots(pred_d, tgt_d, batch)
        host = jax.device_get(acc)
        sums = self.sum_processes({f"{k}_x{t + 1}": float(host[k][t])
                                   for k in host for t in range(n_eval)})
        metrics = {}
        for t in range(1, n_eval + 1):
            cnt = sums[f"_n_x{t}"]
            if cnt > 0:
                metrics.update({f"{k}_x{t}": sums[f"{k}_x{t}"] / cnt for k in self.names})
        for k in self.names:
            per_step = [metrics[f"{k}_x{t}"] for t in range(1, n_eval + 1) if f"{k}_x{t}" in metrics]
            if per_step:
                metrics[k] = float(np.mean(per_step))
        return metrics, plots

    @staticmethod
    def _plots(pred_d, tgt_d, batch) -> dict[str, Any]:
        from neugk_jax.evaluate.plots import generate_val_plots
        roll = {k: pred_d[k][0] for k in ("df", "phi") if k in pred_d}
        gt = {k: tgt_d[k][0] for k in ("df", "phi") if k in tgt_d}
        ts = np.asarray(jax.device_get(batch["timestep"][:1])).reshape(-1)
        return generate_val_plots(rollout=roll, gt=gt, phase="random draw", ts=ts)
