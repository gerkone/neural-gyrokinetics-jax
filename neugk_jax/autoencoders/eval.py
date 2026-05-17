"""AE evaluator: reconstruction MSE + optional integrals via gyaradax.

Mirrors ``neugk/pinc/autoencoders/eval.py:AutoencoderEvaluator`` but
trimmed to the bits the user actually trains (recon metrics + integrals).
Linear probing is left as a follow-up.
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
    """Run ``model`` over the val set, return mean recon metrics."""

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
        # match the training-time placement: if the model is replicated across multiple devices,
        # the eval batch must be sharded on the same mesh, otherwise eqx.filter_jit errors out
        local_dev = jax.local_device_count()
        data_shard = None
        if local_dev > 1:
            mesh = jax.sharding.Mesh(jax.devices(), ("dp",))
            data_shard = NamedSharding(mesh, P("dp"))
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

            metrics, _ = validation_metrics(
                preds={"df": pred},
                tgts={"df": df},
                eval_integrals=eval_integrals,
                geometry=geometry,
            )
            running, n_acc = self._accumulate(running, metrics, n_acc, n_new=batch_size)

        running, n_acc = self._sync(running, n_acc)
        return self._finalize(running, n_acc), {}
