"""gyrowave fixed-shared-support runner (Approach A).

Training is UNCHANGED (WaveletDiT, per-snapshot supports, values diffused / coords
conditioned). Only inference changes: the unknown per-sample support is replaced by a FIXED,
known support S* computed offline from the training data, so coords are known for every query
(no circularity). S* = union of every training per-snapshot support (config
``training.fixed_support = union``), optionally capped to the top-M by appearance frequency
(``fixed_support = topk``, ``training.topk_M``). All coords in S* were seen in training.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp, jax.random as jr

from neugk_jax.diffusion.flow_matching import euler_sample
from neugk_jax.gyrowave.runner import GyrowaveFMRunner, _model_fn


class GyrowaveFixedSupportRunner(GyrowaveFMRunner):
    """WaveletDiT trained as-is; sampled/evaluated on a fixed shared support S*."""

    def setup_data(self) -> None:
        super().setup_data()                                     # X, C, nrm, S, N (training unchanged)
        self._build_fixed_support()

    def _build_fixed_support(self) -> None:
        cshape = np.asarray(self.nrm["cshape"]).astype(np.int64)  # (5,)
        coords = self.nrm["coord_int"].reshape(-1, 5).astype(np.int64)  # (S*N, 5)
        flat = np.ravel_multi_index([coords[:, k] for k in range(5)], cshape)
        uniq, counts = np.unique(flat, return_counts=True)       # frequency across snapshots
        mode = getattr(self.cfg.training, "fixed_support", "union")
        if mode == "topk":
            M = int(getattr(self.cfg.training, "topk_M", len(uniq)))
            uniq = uniq[np.argsort(-counts)[:M]]
        elif mode != "union":
            raise ValueError(f"fixed_support must be 'union' or 'topk', got {mode!r}")
        self.Sstar_int = np.stack(np.unravel_index(uniq, cshape), 1).astype(np.int64)  # (M, 5)
        self.M = int(len(uniq))
        self.Sstar_coords_n = jnp.asarray(self.Sstar_int / cshape[None, :], dtype=jnp.float32)

    def _fixed_cond(self, batch):
        return (jnp.broadcast_to(self.pj, (batch, self.n_cond)),
                jnp.broadcast_to(self.Sstar_coords_n[None], (batch, self.M, 5)))

    def sample(self, *, key, batch, cond=None, steps: int = 50):
        cond = self._fixed_cond(batch) if cond is None else cond
        return euler_sample(_model_fn(self.model), key=key,
                            shape=(batch, self.M, 2), cond=cond, steps=steps)

    def evaluate(self, epoch: int) -> dict:
        steps = getattr(self.cfg.validation, "eval_sample_steps", 50)
        samp = np.asarray(self.sample(key=jr.PRNGKey(epoch), batch=16, steps=steps))
        return {"samp_val_mean": float(samp.mean()), "samp_val_std": float(samp.std()),
                "real_val_std": float(self.X_np.std()), "n_support": float(self.M)}
