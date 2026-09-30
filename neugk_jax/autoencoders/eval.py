"""AE evaluator: reconstruction MSE and relative L2, optional flux integrals and spectra, cross-section plots.

Metrics: ``df_mse`` (normalized), ``df_rel_l2`` (denormalized, zf recombined); with
``validation.eval_integrals`` also ``phi_int_mse``/``phi_int_rel_l2`` and
``flux_int_mse``/``flux_int_rel_err`` of the integrals of the denormalized reconstruction
against those of the target (cached after the first epoch).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.evaluate.base import (
    BaseEvaluator,
    accumulate,
    integrate,
    per_sample_mse,
    per_sample_rel_l2,
)
from neugk_jax.training.data import stack_fields
from neugk_jax.utils import count_trace, recombine_zf


@eqx.filter_jit
def target_integrals(x, fids, denorm, geom):
    count_trace("ae_target_integrals")
    phi, (_, eflux, _) = integrate(geom, fids, denorm("df", x, fids))
    return phi, eflux


@eqx.filter_jit
def ae_eval_step(model, batch, acc, denorm, geom, tgt_int):
    """Reconstruct one batch, add its masked metric sums to ``acc``; returns the denormalized pair."""
    count_trace("ae_eval_step")
    x, fids, mask = batch["df"], batch["file_index"], batch["mask"]
    pred = jax.vmap(lambda xi: model(xi, inference=True)["df"])(x)
    pred_d, tgt_d = denorm("df", pred, fids), denorm("df", x, fids)
    values = {
        "df_mse": per_sample_mse(pred, x),
        "df_rel_l2": per_sample_rel_l2(recombine_zf(pred_d, axis=1), recombine_zf(tgt_d, axis=1)),
    }
    phi = None
    if tgt_int is not None:
        phi_t, eflux_t = tgt_int
        phi, (_, eflux, _) = integrate(geom, fids, pred_d)
        values["phi_int_mse"] = per_sample_mse(phi, phi_t)
        values["phi_int_rel_l2"] = per_sample_rel_l2(phi, phi_t)
        values["flux_int_mse"] = (eflux - eflux_t) ** 2
        values["flux_int_rel_err"] = jnp.abs(eflux - eflux_t) / (jnp.abs(eflux_t) + 1e-12)
    return accumulate(acc, values, mask), pred_d, tgt_d, phi


class AEEvaluator(BaseEvaluator):
    """Reconstruction metrics over the validation set."""

    def __init__(self, cfg: Any, **kwargs):
        super().__init__(cfg, **kwargs)
        self.eval_integrals = bool(self.vcfg.get("eval_integrals", False))
        self.eval_spectra = bool(self.vcfg.get("eval_spectra", False))
        keys = ["df_mse", "df_rel_l2"]
        if self.eval_integrals:
            keys += ["phi_int_mse", "phi_int_rel_l2", "flux_int_mse", "flux_int_rel_err"]
        self.metric_keys = tuple(keys)
        self._tgt_int: dict[int, tuple] = {}

    @staticmethod
    def load(ds, indices, read):
        return stack_fields(read(ds, indices), ("df", "file_index", "timestep"))

    def __call__(self, model: Any, *, epoch: int) -> tuple[dict[str, float], dict[str, Any]]:
        model = self.local_model(model)
        geom = self.geometry if self.eval_integrals else None
        acc = self.zeros((*self.metric_keys, "_n"))
        plots: dict[str, Any] = {}
        spectra: dict[int, tuple] = {}
        for plan, batch, _ in self.loader.iterate(self.ds, self.plans, self.load, self.place):
            tgt_int = None
            if self.eval_integrals:
                if plan.number not in self._tgt_int:
                    self._tgt_int[plan.number] = target_integrals(
                        batch["df"], batch["file_index"], self.denorm, geom)
                tgt_int = self._tgt_int[plan.number]
            acc, pred_d, tgt_d, phi = ae_eval_step(model, batch, acc, self.denorm, geom, tgt_int)
            if self.eval_spectra:
                self._spectra(spectra, pred_d, tgt_d, plan)
            if plan.number == 0 and self.is_rank0:
                plots = self._plots(pred_d, tgt_d, phi, tgt_int, batch)
        sums = self.reduce(acc)
        metrics = {k: sums[k] / max(sums["_n"], 1.0) for k in self.metric_keys}
        if spectra:
            from neugk_jax.evaluate.metrics import merged_spectral_metrics
            metrics.update(merged_spectral_metrics(spectra))
        return metrics, plots

    def _spectra(self, store, pred_d, tgt_d, plan) -> None:
        from neugk_jax.evaluate.metrics import accumulate_spectral_diagnostics
        valid = plan.mask > 0
        if not accumulate_spectral_diagnostics(store, pred_d, tgt_d, plan_fids(self.ds, plan),
                                               self.ds, valid=valid):
            if self.is_rank0:
                print("[evaluate] eval_spectra requested but metadata has no 'ds'; skipping spectral metrics")
            self.eval_spectra = False

    @staticmethod
    def _plots(pred_d, tgt_d, phi, tgt_int, batch) -> dict[str, Any]:
        from neugk_jax.evaluate.plots import generate_val_plots
        rollout, gt = {"df": pred_d[0]}, {"df": tgt_d[0]}
        if phi is not None:
            rollout["phi"], gt["phi"] = phi[0], tgt_int[0][0]
        ts = np.asarray(jax.device_get(batch["timestep"][:1])).reshape(-1)
        return generate_val_plots(rollout=rollout, gt=gt, phase="random draw", ts=ts)


def plan_fids(ds, plan) -> np.ndarray:
    return np.asarray([ds.flat_index_to_file_and_tstep[int(i)][0] for i in plan.indices])
