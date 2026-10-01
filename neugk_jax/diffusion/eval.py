"""Diffusion evaluator: sample latents, decode, integrate, per-trajectory flux RMSE.

* draw ``eval_n_samples`` flow-matching samples per validation condition
* decode and denormalize on device, integrate the heat flux of every sample
* compare against the df-mode target snapshot (``df_mse``, ``df_rel_l2``)
* aggregate the sampled fluxes per ``iteration_<id>`` trajectory (mean ± std) against the
  trajectory's ``avg_flux``: ``avg_flux_rmse``, ``avg_flux_rel_err`` and per-trajectory values
* emit the cross-section ``df`` plot and the ``avg_flux_UQ`` scatter
"""

from __future__ import annotations

import re
from typing import Any, Optional

import equinox as eqx
import jax
import jax.random as jr
import numpy as np

from neugk_jax.diffusion.flow_matching import euler_sample
from neugk_jax.evaluate.base import (
    BaseEvaluator,
    accumulate,
    integrate,
    per_sample_mse,
    per_sample_rel_l2,
)
from neugk_jax.training.data import stack_fields
from neugk_jax.training.ddp import replicate_local
from neugk_jax.utils import count_trace, recombine_zf

_TRAJ_RE = re.compile(r"iteration_\d+")


def _traj_id(path: str) -> Optional[str]:
    m = _TRAJ_RE.search(path)
    return m.group(0) if m else None


def _sample_decode(dit, ae, key, cond, n, steps, latent_scale):
    z = euler_sample(
        lambda x, t, c=None: dit(x, t, c),
        key=key,
        shape=(n, *dit.latent_shape),
        cond=cond,
        steps=steps,
        latent_scale=latent_scale,
    )
    return jax.vmap(lambda zi: ae.decode(zi)["df"])(z)


@eqx.filter_jit
def diffusion_eval_step(dit, ae, key, batch, acc, denorm, geom, steps: int, latent_scale: float):
    count_trace("diffusion_eval_step")
    x, fids, mask = batch["df"], batch["file_index"], batch["mask"]
    pred = _sample_decode(dit, ae, key, batch.get("cond"), x.shape[0], steps, latent_scale)
    pred_d, tgt_d = denorm("df", pred, fids), denorm("df", x, fids)
    values = {
        "df_mse": per_sample_mse(pred, x),
        "df_rel_l2": per_sample_rel_l2(recombine_zf(pred_d, axis=1), recombine_zf(tgt_d, axis=1)),
    }
    eflux = None
    if geom is not None:
        _, (_, eflux, _) = integrate(geom, fids, pred_d)
    return accumulate(acc, values, mask), eflux, pred_d, tgt_d


class DiffusionEvaluator(BaseEvaluator):
    """Sampling-based evaluator with per-trajectory flux UQ.

    ``val_ds`` serves df targets (mode "ae"); ``cond_slots`` selects the DiT conditioning
    from the dataset's condition vector. ``steps``, ``n_samples``, ``stride`` (every
    ``stride``-th validation sample is evaluated) and ``max_batches`` default to
    ``validation.eval_sample_steps`` / ``eval_n_samples`` / ``eval_stride`` /
    ``eval_max_batches``.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        val_ds: Any,
        autoencoder: Any,
        latent_scale: float,
        cond_slots: Optional[np.ndarray] = None,
        steps: Optional[int] = None,
        n_samples: Optional[int] = None,
        stride: Optional[int] = None,
        max_batches: Optional[int] = None,
        **kwargs,
    ):
        vcfg = (cfg.get("validation") if hasattr(cfg, "get") else None) or {}
        stride = int(stride or vcfg.get("eval_stride", 1))
        max_batches = max_batches if max_batches is not None else vcfg.get("eval_max_batches")
        super().__init__(
            cfg,
            val_ds=val_ds,
            indices=range(0, len(val_ds), stride),
            max_batches=max_batches,
            **kwargs,
        )
        self.ae = replicate_local(self.dist, autoencoder)
        self.latent_scale = float(latent_scale)
        self.cond_slots = cond_slots
        self.steps = int(steps or self.vcfg.get("eval_sample_steps", 50))
        self.n_samples = int(n_samples or self.vcfg.get("eval_n_samples", 1))
        self.eval_integrals = bool(self.vcfg.get("eval_integrals", True))
        self.eval_spectra = self.spectra_available(bool(self.vcfg.get("eval_spectra", False)))
        self.metric_keys = ("df_mse", "df_rel_l2")
        self.traj_ids = [_traj_id(f) for f in val_ds.files]

    def load(self, ds, indices, read):
        batch = stack_fields(read(ds, indices), ("df", "file_index", "timestep", "conditioning"))
        cond = batch.pop("conditioning", None)
        if self.cond_slots is not None:
            batch["cond"] = np.asarray(cond)[:, self.cond_slots]
        return batch

    def __call__(self, model: Any, *, epoch: int) -> tuple[dict[str, float], dict[str, Any]]:
        from neugk_jax.evaluate.plots import avg_flux_confidence, generate_val_plots

        model = self.local_model(model)
        geom = self.geometry if self.eval_integrals else None
        acc = self.zeros((*self.metric_keys, "_n"))
        key = jr.PRNGKey(epoch)
        fluxes, plots, spectra = [], {}, {}
        for plan, batch, _ in self.loader.iterate(self.ds, self.plans, self.load, self.place):
            fids = np.asarray(
                [self.ds.flat_index_to_file_and_tstep[int(i)][0] for i in plan.indices]
            )
            for s in range(self.n_samples):
                k = jr.fold_in(jr.fold_in(key, plan.number), s)
                acc, eflux, pred_d, tgt_d = diffusion_eval_step(
                    model,
                    self.ae,
                    k,
                    batch,
                    acc,
                    self.denorm,
                    geom,
                    self.steps,
                    self.latent_scale,
                )
                if eflux is not None:
                    fluxes.append((fids, plan.mask, eflux))
                if self.eval_spectra:
                    self._spectra(spectra, pred_d, tgt_d, fids, plan.mask)
            if plan.number == 0 and self.is_rank0:
                ts = np.asarray(jax.device_get(batch["timestep"][:1])).reshape(-1)
                plots.update(
                    generate_val_plots(
                        rollout={"df": pred_d[0]}, gt={"df": tgt_d[0]}, phase="val sample", ts=ts
                    )
                )
        sums = self.reduce({**acc, **self._traj_sums(fluxes)})
        metrics = {k: sums[k] / max(sums["_n"], 1.0) for k in self.metric_keys}
        if self.eval_spectra:
            metrics.update(self.spectral_metrics(spectra))
        if self.eval_integrals:
            metrics.update(self._flux_metrics(sums, plots, avg_flux_confidence))
        return metrics, plots

    def _traj_sums(self, fluxes) -> dict:
        # per-trajectory sum, square sum and count of the sampled fluxes, one fixed key per val file
        if not self.eval_integrals:
            return {}
        out = {}
        host = jax.device_get([e for _, _, e in fluxes])
        for f in range(len(self.ds.files)):
            vals = np.concatenate(
                [np.asarray(e)[(fids == f) & (m > 0)] for (fids, m, _), e in zip(fluxes, host)]
                or [np.zeros(0)]
            )
            vals = vals.astype(np.float64)
            out[f"_flux_sum/{f}"] = float(vals.sum())
            out[f"_flux_sq/{f}"] = float((vals**2).sum())
            out[f"_flux_n/{f}"] = float(len(vals))
        return out

    def _flux_metrics(self, sums, plots, confidence_plot) -> dict:
        fids = [f for f in range(len(self.ds.files)) if sums.get(f"_flux_n/{f}", 0) > 0]
        if not fids:
            return {}
        n = np.asarray([sums[f"_flux_n/{f}"] for f in fids])
        mean = np.asarray([sums[f"_flux_sum/{f}"] for f in fids]) / n
        std = np.sqrt(
            np.maximum(np.asarray([sums[f"_flux_sq/{f}"] for f in fids]) / n - mean**2, 0)
        )
        tgt = np.asarray([self.ds.get_avg_flux(f) for f in fids])
        names = [self.traj_ids[f] or str(f) for f in fids]
        out = {
            "avg_flux_rmse": float(np.sqrt(np.mean((mean - tgt) ** 2))),
            "avg_flux_rel_err": float(np.mean(np.abs(mean - tgt) / np.maximum(np.abs(tgt), 1e-9))),
        }
        for t, pm, ps, gt in zip(names, mean, std, tgt):
            out[f"avg_flux_pred/{t}"], out[f"avg_flux_std/{t}"] = float(pm), float(ps)
            out[f"avg_flux_gt/{t}"] = float(gt)
        if len(tgt) > 2:
            var = float(np.var(mean))
            out["avg_flux_corr"] = float(np.corrcoef(mean, tgt)[0, 1])
            out["avg_flux_slope"] = (
                float(np.cov(mean, tgt)[0, 1] / var) if var > 0 else float("nan")
            )
        if self.is_rank0:
            plots["avg_flux_UQ"] = confidence_plot(mean, std, tgt, names)
        return out

    def _spectra(self, store, pred_d, tgt_d, fids, mask) -> None:
        from neugk_jax.evaluate.metrics import accumulate_spectral_diagnostics

        accumulate_spectral_diagnostics(store, pred_d, tgt_d, fids, self.ds, valid=mask > 0)
