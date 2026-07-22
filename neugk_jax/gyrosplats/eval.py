"""Gyrosplat evaluator: ensemble sampling vs time-averaged ground truth.

The model is a distribution over saturated turbulent states given the drive
parameters — snapshots are exchangeable, so NOTHING is compared pairwise.
Per validation trajectory:

* sample ``n_ensemble`` banks at the trajectory's conditioning,
* render each and compute the gyaradax heat flux and spectra (qspec = flux
  per ky bin; kyspec = flux-surface-averaged |phi|^2 per ky),
* ensemble-average and compare against the trajectory's TIME-averaged ground
  truth (flux timeseries mean; spectra averaged over strided raw snapshots).

``avg_flux_rmse`` is the rmse over trajectories between the ensemble-mean
sampled flux and the gt time-averaged flux. Sample panels (plot_nd) are
emitted next to an arbitrary gt snapshot for visual judgment — the pairing in
those images is for reference only.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.evaluate.base import BaseEvaluator
from neugk_jax.evaluate.plots import avg_flux_confidence, generate_val_plots
from neugk_jax.gyrosplats.normalize import ZfStats, denormalize_tokens, zf_denormalize
from neugk_jax.gyrosplats.render import render, subgrids, to_field
from neugk_jax.utils import separate_zf


def _rl2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


class GyrosplatEvaluator(BaseEvaluator):
    """Ensemble-statistics evaluator for splat flow matching."""

    def __init__(self, cfg: Any, *, val_ds: Any, sample_fn: Callable, is_rank0: bool = True):
        super().__init__(cfg, val_ds=val_ds, is_rank0=is_rank0)
        self.sample_fn = sample_fn

    def _spectra(self, fields: list[np.ndarray], geom: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(flux (B,), qspec (B, ky), kyspec (B, ky)) for physical fields."""
        from neugk_jax.evaluate.integrals import gyaradax_spectral_fields

        sep = np.stack([np.asarray(separate_zf(jnp.asarray(f), axis=0)) for f in fields])
        phi, eflux = gyaradax_spectral_fields(sep, geom)
        qspec = np.asarray(eflux).sum(axis=1)  # (B, ky)
        ints = np.asarray(geom["ints"]).reshape(1, -1, 1, 1)
        kyspec = (np.abs(np.asarray(phi)) ** 2 * ints).sum(axis=(1, 2))  # (B, ky)
        return qspec.sum(axis=1), qspec, kyspec

    def __call__(
        self,
        model: Any,
        *,
        epoch: int,
        batch_size: int = 8,
        n_steps: int = 50,
        n_ensemble: int = 32,
        gt_stride: int = 12,
        max_trajectories: Optional[int] = None,
        **kwargs,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        ds = self.val_ds
        key = jr.PRNGKey(epoch)
        val_plots: dict[str, Any] = {}
        per_traj: dict[str, dict[str, float]] = {}

        shape = None
        n_traj = len(ds.files) if max_trajectories is None else min(len(ds.files), max_trajectories)
        for fid in range(n_traj):
            meta = ds.get_metadata(fid)
            if shape is None:
                shape = tuple(int(v) for v in meta["resolution"])
                vg, pg = subgrids(shape)
            geom = {k: np.asarray(v) for k, v in meta["geometry"].items()}
            # first flat index of this trajectory -> conditioning (constant per traj)
            flat0 = next(
                i for i, (f, _) in ds.flat_index_to_file_and_tstep.items() if f == fid
            )
            s0 = ds[flat0]
            cond_one = jnp.asarray(s0.conditioning)

            # --- sampled ensemble ---
            fields = []
            for start in range(0, n_ensemble, batch_size):
                bs = min(batch_size, n_ensemble - start)
                step_key, key = jr.split(key)
                toks = self.sample_fn(
                    key=step_key, batch=bs, cond=jnp.tile(cond_one[None], (bs, 1)), steps=n_steps
                )
                toks = np.asarray(toks)
                for b in range(bs):
                    if getattr(ds, "layout", "atoms") == "windows":
                        p = ds.decode_state(jnp.asarray(toks[b]))
                        # trajectory time-mean stats (uses the real fluc_mean, not 0)
                        st = ZfStats(*[jnp.asarray(v) for v in ds.traj_mean_zf_stats(fid)], "zf")
                    elif ds.stats_token:
                        from neugk_jax.gyrosplats.normalize import denormalize_zf_scalars

                        p = denormalize_tokens(
                            jnp.asarray(toks[b, :-1]), jnp.asarray(ds.bins), ds.token_stats
                        )
                        st = denormalize_zf_scalars(jnp.asarray(toks[b, -1, :3]), ds.token_stats)
                    else:
                        p = denormalize_tokens(
                            jnp.asarray(toks[b]), jnp.asarray(ds.bins), ds.token_stats
                        )
                        # no stats token: trajectory time-mean stats (teacher substitute)
                        st = ZfStats(*[jnp.asarray(v) for v in ds.traj_mean_zf_stats(fid)], "zf")
                    fields.append(
                        np.asarray(
                            zf_denormalize(
                                to_field(render(p, vg, pg, atom_chunk=256), shape), st
                            )
                        )
                    )
            smp_flux, smp_q, smp_ky = self._spectra(fields, geom)

            # --- gt time averages ---
            t_ids = list(range(0, ds.atoms[fid].shape[0], max(1, gt_stride)))
            gt_fields = [ds.get_gt_field(fid, t) for t in t_ids]
            gt_flux_series = np.asarray(ds.flux[fid][ds.offset :], dtype=np.float64)
            _, gt_q, gt_ky = self._spectra(gt_fields, geom)

            tid = ds.files[fid]
            per_traj[tid] = {
                "flux_mean": float(smp_flux.mean()),
                "flux_std": float(smp_flux.std()),
                "gt_avg_flux": float(gt_flux_series.mean()),
                "gt_flux_std": float(gt_flux_series.std()),
                "qspec_rl2": _rl2(smp_q.mean(0), gt_q.mean(0)),
                "kyspec_rl2": _rl2(smp_ky.mean(0), gt_ky.mean(0)),
            }

            # sample panels for visual judgment (gt side is an ARBITRARY snapshot)
            if fid == 0 and self.is_rank0:
                try:
                    val_plots.update(
                        generate_val_plots(
                            rollout={"df": fields[0]},
                            gt={"df": gt_fields[0]},
                            phase=f"sample vs arbitrary gt ({tid})",
                            to_wandb=True,
                        )
                    )
                except Exception as e:
                    print(f"[eval] sample panels skipped: {e}")

        traj_ids = sorted(per_traj.keys())
        pred_means = np.array([per_traj[t]["flux_mean"] for t in traj_ids])
        pred_stds = np.array([per_traj[t]["flux_std"] for t in traj_ids])
        tgt_vals = np.array([per_traj[t]["gt_avg_flux"] for t in traj_ids])
        metrics = {
            # THE headline number: ensemble-mean flux vs gt time-averaged flux
            "avg_flux_rmse": float(np.sqrt(np.mean((pred_means - tgt_vals) ** 2))),
            "avg_flux_rel_err": float(
                np.mean(np.abs(pred_means - tgt_vals) / np.maximum(np.abs(tgt_vals), 1e-9))
            ),
            "flux_std_ratio": float(
                np.mean(pred_stds / np.maximum([per_traj[t]["gt_flux_std"] for t in traj_ids], 1e-9))
            ),
            "qspec_rl2": float(np.mean([per_traj[t]["qspec_rl2"] for t in traj_ids])),
            "kyspec_rl2": float(np.mean([per_traj[t]["kyspec_rl2"] for t in traj_ids])),
        }
        if self.is_rank0:
            val_plots["avg_flux_UQ"] = avg_flux_confidence(
                pred_means, pred_stds, tgt_vals, traj_ids, to_wandb=True
            )
        return metrics, val_plots
