"""AE evaluator: reconstruction MSE and relative L2, optional flux integrals and spectra, cross-section plots.

Metrics: ``df_mse`` (normalized), ``df_rel_l2`` (denormalized, zf recombined); with
``validation.eval_integrals`` also ``phi_int_mse``/``phi_int_rel_l2`` and
``flux_int_mse``/``flux_int_rel_err`` of the integrals of the denormalized reconstruction
against those of the target (cached after the first epoch).
"""

from __future__ import annotations

from typing import Any

import jax

from neugk_jax.evaluate.base import BaseEvaluator, accumulate, integrate, recon_metrics
from neugk_jax.losses import per_sample_mse, per_sample_rel_l2, rel_err
from neugk_jax.training.data import stack_fields
from neugk_jax.utils import traced_jit


@traced_jit("ae_target_integrals")
def target_integrals(x, fids, norm, geom):
    phi, (_, eflux, _) = integrate(geom, fids, norm.denormalize("df", x, fids))
    return phi, eflux


@traced_jit("ae_eval_step")
def ae_eval_step(model, batch, acc, norm, geom, tgt_int):
    """Reconstruct one batch, add its masked metric sums to ``acc``; returns the denormalized pair."""
    x, fids, mask = batch["df"], batch["file_index"], batch["mask"]
    pred = jax.vmap(lambda xi: model(xi, inference=True)["df"])(x)
    pred_d, tgt_d = norm.denormalize("df", pred, fids), norm.denormalize("df", x, fids)
    values = recon_metrics(pred, x, pred_d, tgt_d)
    phi = None
    if tgt_int is not None:
        phi_t, eflux_t = tgt_int
        phi, (_, eflux, _) = integrate(geom, fids, pred_d)
        values["phi_int_mse"] = per_sample_mse(phi, phi_t)
        values["phi_int_rel_l2"] = per_sample_rel_l2(phi, phi_t)
        values["flux_int_mse"] = (eflux - eflux_t) ** 2
        values["flux_int_rel_err"] = rel_err(eflux, eflux_t)
    return accumulate(acc, values, mask), pred_d, tgt_d, phi


class AEEvaluator(BaseEvaluator):
    """Reconstruction metrics over the validation set."""

    def __init__(self, cfg: Any, **kwargs):
        super().__init__(cfg, **kwargs)
        self.eval_spectra = self.spectra_available(bool(self.vcfg.get("eval_spectra", False)))
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
        spectra: dict[int, dict] = {}
        for plan, batch, _ in self.loader.iterate(self.ds, self.plans, self.load, self.place):
            tgt_int = None
            if self.eval_integrals:
                if plan.number not in self._tgt_int:
                    self._tgt_int[plan.number] = jax.device_get(
                        target_integrals(batch["df"], batch["file_index"], self.norm, geom)
                    )
                tgt_int = self.place(self._tgt_int[plan.number])
            acc, pred_d, tgt_d, phi = ae_eval_step(model, batch, acc, self.norm, geom, tgt_int)
            if self.eval_spectra:
                self.spectra(spectra, pred_d, tgt_d, plan)
            if plan.number == 0 and self.is_rank0:
                plots = self._plots(pred_d, tgt_d, phi, tgt_int, batch)
        metrics = self.finalize(self.reduce(acc), self.metric_keys)
        if self.eval_spectra:
            metrics.update(self.spectral_metrics(spectra))
        return metrics, plots

    def _plots(self, pred_d, tgt_d, phi, tgt_int, batch) -> dict[str, Any]:
        from neugk_jax.evaluate.plots import generate_val_plots

        rollout, gt = {"df": pred_d[0]}, {"df": tgt_d[0]}
        if phi is not None:
            rollout["phi"], gt["phi"] = phi[0], tgt_int[0][0]
        return generate_val_plots(rollout, gt, "random draw", ts=self.plot_time(batch))
