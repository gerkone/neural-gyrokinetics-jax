"""AE evaluator: reconstruction MSE + optional integrals via gyaradax.

Mirrors ``neugk/pinc/autoencoders/eval.py:AutoencoderEvaluator`` but
trimmed to the bits the user actually trains (recon metrics + integrals
+ cross-section plots). Linear probing is left as a follow-up.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from neugk_jax.evaluate.base import BaseEvaluator, validation_metrics


class AEEvaluator(BaseEvaluator):
    """Run ``model`` over the val set, return mean recon metrics + plot dict."""

    def __call__(
        self,
        model: Any,
        *,
        epoch: int,
        batch_size: int = 1,
        eval_integrals: bool = False,
        **kwargs,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        ds = self.val_ds
        n = len(ds)

        @eqx.filter_jit
        def fwd(m, x):
            return jax.vmap(lambda xi: m(xi)["df"])(x)

        running: dict[str, float] = {}
        n_acc = 0.0
        # match training-time placement: model replicated across devices → shard the val batch too
        local_dev = jax.local_device_count()
        data_shard = None
        if local_dev > 1:
            mesh = jax.sharding.Mesh(jax.devices(), ("dp",))
            data_shard = NamedSharding(mesh, P("dp"))

        # plot collection — one cross-section panel per epoch (random draw), one flux UQ scatter
        val_plots: dict[str, Any] = {}
        plot_drawn = False
        # per-trajectory flux UQ accumulator
        flux_pred_per_traj: dict[str, list[float]] = {}
        flux_tgt_per_traj: dict[str, float] = {}

        for start in range(0, n - batch_size + 1, batch_size):
            samples = [ds[i] for i in range(start, start + batch_size)]
            df = jnp.stack([jnp.asarray(s.df) for s in samples])
            if data_shard is not None:
                df = jax.device_put(df, data_shard)
            pred = fwd(model, df)

            geometry = None
            if eval_integrals and hasattr(ds, "get_batch_geometry"):
                fid = np.asarray([int(s.file_index) for s in samples])
                geom = ds.get_batch_geometry(fid)
                geometry = {k: jnp.asarray(v) for k, v in geom.items()}

            metrics, integrated = validation_metrics(
                preds={"df": pred},
                tgts={"df": df},
                eval_integrals=eval_integrals,
                geometry=geometry,
            )
            running, n_acc = self._accumulate(running, metrics, n_acc, n_new=batch_size)

            # accumulate per-traj fluxes for the UQ scatter
            if integrated is not None and integrated.get("eflux") is not None:
                eflux = np.asarray(integrated["eflux"]).reshape(batch_size, -1).sum(axis=-1)
                for s_i, s in enumerate(samples):
                    fid_i = int(s.file_index)
                    traj_id = self._traj_id_for(fid_i)
                    flux_pred_per_traj.setdefault(traj_id, []).append(float(eflux[s_i]))
                    flux_tgt_per_traj[traj_id] = float(getattr(s, "avg_flux", 0.0))

            # one cross-section panel per epoch (first eval batch)
            if not plot_drawn and self.is_rank0:
                try:
                    from neugk_jax.evaluate.plots import generate_val_plots
                    b_idx = 0
                    pred_d = np.asarray(ds.denormalize(int(samples[b_idx].file_index),
                                                       df=np.asarray(pred[b_idx])))
                    tgt_d = np.asarray(ds.denormalize(int(samples[b_idx].file_index),
                                                      df=np.asarray(df[b_idx])))
                    panels = generate_val_plots(
                        rollout={"df": pred_d},
                        gt={"df": tgt_d},
                        phase="random draw",
                        ts=np.asarray(samples[b_idx].timestep).reshape(-1),
                    )
                    val_plots.update(panels)
                    plot_drawn = True
                except Exception as e:
                    print(f"[evaluate] cross-section plot skipped: {e}")
                    plot_drawn = True  # don't retry every batch

        # final flux UQ scatter
        if flux_pred_per_traj and self.is_rank0:
            try:
                from neugk_jax.evaluate.plots import avg_flux_confidence
                traj_ids = sorted(flux_pred_per_traj)
                means = np.array([np.mean(flux_pred_per_traj[t]) for t in traj_ids])
                stds = np.array([np.std(flux_pred_per_traj[t]) for t in traj_ids])
                tgts_arr = np.array([flux_tgt_per_traj[t] for t in traj_ids])
                val_plots["avg_flux_UQ"] = avg_flux_confidence(
                    pred_means=means, pred_stds=stds, tgt_vals=tgts_arr, traj_ids=traj_ids,
                )
            except Exception as e:
                print(f"[evaluate] avg_flux_UQ plot skipped: {e}")

        running, n_acc = self._sync(running, n_acc)
        # rename to torch's canonical keys (``df`` → ``df_mse``, ``flux`` → ``flux_int_mse``, ``phi`` → ``phi_int_mse``)
        finalized = self._finalize(running, n_acc)
        renamed = {
            "df_mse" if k == "df" else
            "phi_int_mse" if k == "phi" else
            "flux_int_mse" if k == "flux" else k: v
            for k, v in finalized.items()
        }
        return renamed, val_plots

    def _traj_id_for(self, fid: int) -> str:
        files = getattr(self.val_ds, "files", None)
        if not files or fid >= len(files):
            return f"fid_{fid}"
        import os, re
        name = os.path.basename(files[fid]).replace("_ifft_realpotens", "")
        m = re.match(r"^(iteration_\d+)", name)
        return m.group(1) if m else name
