"""Evaluator base: fixed-shape batch iteration, on-device denormalization, flux integrals and
masked metric accumulation with cross-process sync over a fixed key set.

Each evaluator is built once per run; its batch plan, normalization and geometry tables
are fixed, and its per-batch forward is a module-level jitted function.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.evaluate.integrals import flux_integral, precompute_geometry, require_geometry
from neugk_jax.training.data import BatchLoader, eval_plans
from neugk_jax.training.ddp import (
    DistributedInfo,
    init_distributed,
    local_view,
    replicate_local,
    shard_local,
)
from neugk_jax.utils import config_dict, recombine_zf

# sample ndim (without batch) per normalized field
FIELD_NDIM = {"df": 6, "phi": 3, "flux": 0, "fluxavg": 0}


class Denorm(eqx.Module):
    """Per-field ``(scale, shift)``: shared arrays, or per-trajectory tables gathered by file index."""

    params: dict
    shared: bool = eqx.field(static=True)

    @classmethod
    def from_dataset(cls, ds, fields: Sequence[str]) -> "Denorm":
        shared = ds.normalization is None or ds.normalization_scope in ("dataset", "sample")
        fids = [0] if shared else list(range(len(ds.files)))
        params = {k: tuple(jnp.asarray(a) for a in ds.scale_shift(fids, k, FIELD_NDIM[k]))
                  for k in fields}
        return cls(params, shared)

    def scale_shift(self, field: str, fids):
        scale, shift = self.params[field]
        if not self.shared:
            scale, shift = scale[fids], shift[fids]
        return scale, shift

    def __call__(self, field: str, x, fids):
        scale, shift = self.scale_shift(field, fids)
        return x * scale + shift


def geometry_table(ds, fids: Optional[Sequence[int]] = None) -> dict[str, np.ndarray]:
    """Stacked :func:`precompute_geometry` tensors of trajectories ``fids`` (default: all)."""
    fids = range(len(ds.files)) if fids is None else fids
    geoms = []
    for f in fids:
        g = ds.metadata[int(f)].get("geometry")
        require_geometry(g)
        geoms.append(precompute_geometry(g))
    return {k: np.stack([g[k] for g in geoms]) for k in geoms[0]}


def per_sample_mse(p, t):
    return jnp.mean((p - t).reshape(p.shape[0], -1) ** 2, axis=-1)


def per_sample_rel_l2(p, t, eps: float = 1e-12):
    p, t = p.reshape(p.shape[0], -1), t.reshape(t.shape[0], -1)
    return jnp.linalg.norm(p - t, axis=-1) / (jnp.linalg.norm(t, axis=-1) + eps)


def integrate(geom, fids, df, phi=None, *, real_potens: bool = True):
    """Batched flux integral of a denormalized df (separate-zf layouts are recombined)."""
    g = jax.tree_util.tree_map(lambda a: a[fids], geom)
    df = recombine_zf(df, axis=1)
    if phi is None:
        return jax.vmap(lambda gi, d: flux_integral(gi, d, real_potens=real_potens))(g, df)
    return jax.vmap(lambda gi, d, p: flux_integral(gi, d, p, real_potens=real_potens))(g, df, phi)


def accumulate(acc: dict, values: Mapping[str, jnp.ndarray], weight, count_key: str = "_n") -> dict:
    """Add the ``weight``-masked per-sample ``values`` and the weight total into ``acc``."""
    out = dict(acc)
    for k, v in values.items():
        out[k] = acc[k] + jnp.sum(v * weight)
    out[count_key] = acc[count_key] + jnp.sum(weight)
    return out


class BaseEvaluator:
    """Owns the fixed evaluation batch plan, the device tables and metric reduction.

    ``batch_size`` is per local device. Subclasses define ``metric_keys`` and
    ``__call__(model, *, epoch) -> (metrics, plots)``.
    """

    denorm_fields: tuple[str, ...] = ("df",)

    def __init__(self, cfg: Any, *, val_ds: Any, dist: Optional[DistributedInfo] = None,
                 batch_size: int = 1, loader: Optional[BatchLoader] = None,
                 indices: Optional[Sequence[int]] = None, max_batches: Optional[int] = None):
        self.cfg = cfg
        self.vcfg = config_dict(cfg.get("validation")) if hasattr(cfg, "get") else {}
        self.ds = val_ds
        self.dist = dist or init_distributed()
        self.batch_size = int(batch_size) * self.dist.local_device_count
        self.loader = loader or BatchLoader()
        idx = range(len(val_ds)) if indices is None else indices
        self.plans = eval_plans(self.dist, idx, self.batch_size, max_batches)
        self.denorm = replicate_local(self.dist, Denorm.from_dataset(val_ds, self.denorm_fields))
        self._geometry = None

    @property
    def is_rank0(self) -> bool:
        return self.dist.is_rank0

    @property
    def geometry(self) -> dict:
        if self._geometry is None:
            self._geometry = replicate_local(self.dist, geometry_table(self.ds))
        return self._geometry

    def place(self, batch):
        return shard_local(self.dist, batch)

    def local_model(self, model):
        return local_view(self.dist, model)

    def zeros(self, keys: Sequence[str]) -> dict:
        return replicate_local(self.dist, {k: jnp.zeros((), jnp.float32) for k in keys})

    def reduce(self, acc: dict) -> dict[str, float]:
        return self.sum_processes({k: float(v) for k, v in jax.device_get(acc).items()})

    def spectra_available(self, requested: bool) -> bool:
        """``requested`` unless a validation trajectory's metadata lacks the ``ds`` spacing."""
        if requested and any(self.ds.get_ds(f) is None for f in range(len(self.ds.files))):
            if self.is_rank0:
                print("[evaluate] eval_spectra requested but metadata has no 'ds'; "
                      "skipping spectral metrics")
            return False
        return requested

    def spectral_metrics(self, store: dict) -> dict[str, float]:
        """Spectral metrics of the per-trajectory sums of every process (collective)."""
        from neugk_jax.evaluate import metrics as m
        if self.dist.num_processes > 1:
            n_ky = int(self.ds.resolution[-1])
            packed = self.sum_process_arrays(m.pack_spectral_store(store, len(self.ds.files), n_ky))
            store = m.unpack_spectral_store(packed, n_ky)
        return m.merged_spectral_metrics(store)

    def sum_process_arrays(self, arr: np.ndarray) -> np.ndarray:
        if self.dist.num_processes <= 1:
            return arr
        from jax.experimental import multihost_utils
        return np.asarray(multihost_utils.process_allgather(arr)).sum(axis=0)

    def sum_processes(self, host: dict[str, float]) -> dict[str, float]:
        """Sum host scalars over processes; every process passes the same key set."""
        if self.dist.num_processes <= 1:
            return host
        keys = sorted(host)
        tot = self.sum_process_arrays(np.asarray([host[k] for k in keys], dtype=np.float64))
        return {k: float(v) for k, v in zip(keys, tot)}

    def __call__(self, model: Any, *, epoch: int) -> tuple[dict[str, float], dict[str, Any]]:
        raise NotImplementedError
