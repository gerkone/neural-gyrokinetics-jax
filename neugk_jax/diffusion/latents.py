"""Precompute and cache AE latents for the diffusion training mode.

After training the AE (M4), the diffusion training (M5) operates on
the AE's bottleneck latents instead of the raw distribution functions.
We encode every sample once, cache the result, and the dataset serves
the latents in mode='diff' instead of doing a costly forward through
the encoder on each step.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.utils import RunningMeanStd


def _tqdm(*args, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(*args, **kwargs)
    except ImportError:
        return args[0] if args else iter(())


def _cache_path(dataset, split: str, ae_tag: str) -> Path:
    """Deterministic name for a precomputed-latents pickle on disk."""
    basenames = sorted(os.path.basename(f) for f in dataset.files)
    h = hashlib.sha256("".join(basenames).encode()).hexdigest()[:12]
    name = f"diff_{split}_latents_offset{dataset.offset}_{h}_{ae_tag}.pkl"
    return Path(dataset.path) / name


def precompute_latents(
    dataset,
    *,
    encode_fn: Callable,
    ae_tag: str,
    batch_size: int = 4,
    device=None,
    overwrite: bool = False,
) -> None:
    """Encode the entire dataset through ``encode_fn`` and cache the result.

    ``encode_fn`` is called as ``encode_fn(df_batch, cond_batch) -> latent_batch``
    where ``df_batch`` has shape ``(B, C, *resolution)`` and ``latent_batch``
    has shape ``(B, *latent_grid, latent_channels)``. The dataset is mutated
    in place: ``dataset.precomputed_latents`` is populated and the mode is
    flipped to ``"diff"`` so subsequent ``__getitem__`` calls return cached
    latents instead of raw df reads.
    """
    cache_file = _cache_path(dataset, dataset.split, ae_tag)
    if cache_file.exists() and not overwrite:
        with open(cache_file, "rb") as f:
            dataset.precomputed_latents = pickle.load(f)
        dataset.mode = "diff"
        _compute_latent_stats(dataset)
        return

    latents_dict: dict[tuple[int, int], dict] = {}
    n = len(dataset)
    indices = list(range(n))
    pbar = _tqdm(range(0, n, batch_size), desc=f"precompute {dataset.split} latents")
    for start in pbar:
        sl = indices[start : start + batch_size]
        # gather a batch from the (currently mode='ae') dataset
        samples = [dataset[i] for i in sl]
        df_batch = jnp.stack([jnp.asarray(s.df) for s in samples])
        cond_batch = None
        if samples[0].conditioning is not None:
            cond_batch = jnp.stack([jnp.asarray(s.conditioning) for s in samples])
        z = encode_fn(df_batch, cond_batch)
        z_np = np.asarray(z)
        for i, s in zip(sl, samples):
            fid = int(s.file_index)
            t_idx = int(s.timestep_index)
            latents_dict[(fid, t_idx)] = {
                "x": z_np[sl.index(i)],
                "flux": np.asarray(s.flux),
                "timestep": np.asarray(s.timestep),
                "phi": np.asarray(s.phi) if s.phi is not None else None,
                "conditioning": np.asarray(s.conditioning) if s.conditioning is not None else None,
            }

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(latents_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    dataset.precomputed_latents = latents_dict
    dataset.mode = "diff"
    _compute_latent_stats(dataset)


def load_precomputed_latents(
    dataset, pickle_path: str | Path, *, verify: bool = True, remap: bool = True,
    latent_shape=None,
) -> None:
    """Populate ``dataset.precomputed_latents`` from a pre-existing pickle.

    Bypasses the ``encode_fn`` path in :func:`precompute_latents` when the
    cache has been computed externally (e.g. by the upstream torch
    pipeline). The pickle is expected to be ``dict[(file_idx, t_idx),
    dict]`` with the same per-entry schema written by
    :func:`precompute_latents` — at minimum ``x``, ``flux``, ``timestep``;
    optionally ``phi`` and the scalar conditioning fields.

    ``verify`` cross-checks the cache against this dataset: every indexed
    ``(fid, t_idx)`` must be present, and where the cache carries the scalar
    conditions they must agree with the trajectory metadata — a foreign cache
    whose file ordering differs would otherwise pair latents with the wrong
    trajectory silently. ``latent_shape``, when given, is checked too. ``remap``
    then re-keys such a cache onto this dataset's fids (see
    :func:`remap_latent_cache`) instead of failing.
    """
    p = Path(pickle_path)
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "rb") as f:
        cache = pickle.load(f)
    if verify:
        try:
            verify_latent_cache(dataset, cache, latent_shape=latent_shape)
        except ValueError:
            if not remap:
                raise
            cache = remap_latent_cache(dataset, cache)
            verify_latent_cache(dataset, cache, latent_shape=latent_shape)
    dataset.precomputed_latents = cache
    dataset.mode = "diff"
    _compute_latent_stats(dataset)


_COND_ALIASES = {"itg": "ion_temp_grad", "dg": "density_grad", "s_hat": "s_hat", "q": "q"}


def remap_latent_cache(dataset, cache: dict, *, tol: float = 1e-3) -> dict:
    """Re-key a cache built with a different file ordering onto this dataset's fids.

    Trajectories are matched on their ``(itg, dg, s_hat, q)`` tuple, which must be
    unique on both sides; the per-entry ``timestep`` is then checked against the
    trajectory metadata so a wrong pairing cannot slip through.
    """
    def _tuple_of(get):
        return np.array([float(np.squeeze(get(k))) for k in ("itg", "dg", "s_hat", "q")])

    ours = {}
    for fid, meta in dataset.metadata.items():
        key = tuple(np.round(_tuple_of(lambda k: meta[_COND_ALIASES[k]]), 6))
        if key in ours:
            raise ValueError(f"trajectories {ours[key]} and {fid} share conditions {key}; "
                             "cannot remap the cache by conditions")
        ours[key] = fid
    keys = np.array(list(ours))
    fids = list(ours.values())

    cache_fids = sorted({k[0] for k in cache})
    steps = {f: sorted(t for cf, t in cache if cf == f) for f in cache_fids}
    mapping: dict[int, int] = {}
    for cfid in cache_fids:
        entry = cache[(cfid, steps[cfid][0])]
        if not all(k in entry for k in _COND_ALIASES):
            raise ValueError("cache entries carry no scalar conditions; cannot remap")
        d = np.abs(keys - _tuple_of(lambda k: entry[k])).max(axis=1)
        j = int(np.argmin(d))
        if d[j] > tol:
            raise ValueError(f"cache trajectory {cfid} matches no dataset trajectory "
                             f"(closest distance {d[j]:.3g})")
        mapping[cfid] = fids[j]
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("cache -> dataset trajectory matching is not bijective")

    out = {}
    for (cfid, t), entry in cache.items():
        fid = mapping[cfid]
        if "timestep" in entry:
            want = float(dataset.metadata[fid]["timesteps"][t + dataset.offset])
            if not np.isclose(float(entry["timestep"]), want, rtol=1e-4):
                raise ValueError(
                    f"remapped cache entry ({cfid}->{fid}, t={t}) has timestep "
                    f"{float(entry['timestep'])} but the trajectory has {want}"
                )
        out[(fid, t)] = entry
    return out


def verify_latent_cache(dataset, cache: dict, *, latent_shape=None) -> None:
    """Raise unless ``cache`` matches this dataset's index and file ordering."""
    wanted = set(dataset.flat_index_to_file_and_tstep.values())
    missing = sorted(wanted - set(cache))
    if missing:
        raise ValueError(
            f"latent cache is missing {len(missing)} of {len(wanted)} (fid, t_idx) entries, "
            f"e.g. {missing[:5]} — the trajectory selection or offset does not match "
            "the cache (cache holds "
            f"{len({k[0] for k in cache})} trajectories x {len({k[1] for k in cache})} steps)"
        )
    any_key = next(iter(wanted))
    x = cache[any_key]["x"]
    if latent_shape is not None and tuple(x.shape) != tuple(latent_shape):
        raise ValueError(f"latent cache shape {tuple(x.shape)} != model latent {tuple(latent_shape)}")
    # the scalar conditions pin the fid -> trajectory mapping
    aliases = _COND_ALIASES
    bad = []
    for fid in sorted({k[0] for k in wanted}):
        entry = cache[(fid, sorted(t for f, t in wanted if f == fid)[0])]
        for short, meta_key in aliases.items():
            if short not in entry:
                continue
            want = float(np.squeeze(dataset.metadata[fid][meta_key]))
            got = float(np.squeeze(entry[short]))
            if not np.isclose(want, got, rtol=1e-4, atol=1e-6):
                bad.append((fid, short, want, got))
    if bad:
        raise ValueError(
            f"latent cache conditions disagree for {len(bad)} (fid, field) pairs, e.g. "
            f"{bad[:4]} — the cache was built with a different file ordering or filter set"
        )


def _compute_latent_stats(dataset) -> None:
    """Running stats over all cached latents — used to scale by 1/std at train time."""
    stats = None
    for s in dataset.precomputed_latents.values():
        x = s["x"]
        mean = np.mean(x, keepdims=True)
        var = np.var(x, keepdims=True)
        mn = np.min(x, keepdims=True)
        mx = np.max(x, keepdims=True)
        if stats is None:
            stats = RunningMeanStd(shape=mean.shape)
        stats.update(mean, var, mn, mx, count=1)
    dataset.latent_stats = stats
