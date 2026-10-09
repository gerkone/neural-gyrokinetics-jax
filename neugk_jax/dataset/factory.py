"""``CycloneDataset`` construction from a ``dataset`` config section."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from omegaconf import OmegaConf

from neugk_jax.dataset.backend import make_backend
from neugk_jax.dataset.cyclone import DEFAULT_CONDITIONS, CycloneDataset
from neugk_jax.dataset.mix import CycloneMix
from neugk_jax.dataset.normalization import load_stats, save_stats, stream_stats
from neugk_jax.utils import to_dict


def build_dataset(
    dcfg,
    *,
    split: str,
    dist=None,
    mode: str = "ae",
    fields: Optional[Sequence[str]] = None,
    conditions: Optional[Sequence[str]] = None,
    prefer_dtype: Optional[str] = None,
    stats: Optional[dict] = None,
    **overrides: Any,
) -> CycloneDataset:
    """Dataset for ``split`` ("train" or "val") from the ``dataset`` config.

    ``stats`` (an already loaded ``CycloneDataset.stats``) replaces
    ``normalization_stats`` so the pickle is read once per run.
    ``lightweight_metadata`` (read ``metadata_light``, without the per-trajectory df
    moments) defaults on when normalization stats are given.
    """
    norm_stats = dcfg.get("normalization_stats")
    if norm_stats is not None and not isinstance(norm_stats, str):
        norm_stats = to_dict(norm_stats)
    normalization = to_dict(dcfg.get("normalization")) or None
    stats_ready = stats is not None or (
        norm_stats is not None
        and not (isinstance(norm_stats, str) and not os.path.exists(norm_stats))
    )
    lightweight = bool(dcfg.get("lightweight_metadata", stats_ready))
    if lightweight and normalization is not None and not stats_ready:
        raise ValueError(
            "lightweight_metadata drops the per-trajectory df moments; set "
            "normalization_stats or disable it"
        )
    trajectories = dcfg.training_trajectories if split == "train" else dcfg.validation_trajectories
    filters = dcfg.get("training_cond_filters" if split == "train" else "eval_cond_filters")
    augment = to_dict(dcfg.get("augment")) if split == "train" else None
    kwargs = dict(
        path=dcfg.path,
        split=split,
        trajectories=trajectories if isinstance(trajectories, str) else list(trajectories),
        fields_to_load=tuple(fields or dcfg.get("input_fields", ("df",))),
        conditions=tuple(
            conditions if conditions is not None else dcfg.get("conditions", DEFAULT_CONDITIONS)
        ),
        mode=mode,
        separate_zf=bool(dcfg.get("separate_zf", False)),
        species_axis=bool(dcfg.get("species_axis", False)),
        kx_crop=dcfg.get("kx_crop"),
        normalization=normalization,
        normalization_scope=dcfg.get("normalization_scope", "dataset"),
        normalization_stats=stats if stats is not None else norm_stats,
        offset=int(dcfg.get("offset", 0)),
        subsample=int(
            dcfg.get("subsample", 1) if split == "train" else dcfg.get("val_subsample", 1)
        ),
        cond_filters=to_dict(filters) or None,
        augment=augment or None,
        backend=make_backend(
            dcfg,
            local_rank=dist.local_rank if dist else 0,
            prefer_dtype=prefer_dtype,
            lightweight_metadata=lightweight,
        ),
        rank=dist.process_id if dist else 0,
    )
    kwargs.update(overrides)
    return CycloneDataset(**kwargs)


def resolve_stats(dcfg, *, run_stats: Optional[str] = None, **kwargs) -> Optional[dict]:
    """Dataset-scope normalization statistics: the run's own copy ``run_stats``, else the
    ``normalization_stats`` file, else computed from the unnormalized training samples."""
    normalization = to_dict(dcfg.get("normalization")) or None
    if not normalization or dcfg.get("normalization_scope", "dataset") != "dataset":
        return None
    path = dcfg.get("normalization_stats")
    if path is not None and not isinstance(path, str):
        return to_dict(path)
    for p in (run_stats, path):
        if p is not None and os.path.exists(p):
            return load_stats(p, normalization)
    ds = build_dataset(dcfg, split="train", stats={}, **kwargs)
    return stream_stats(ds, normalization, int(dcfg.get("stats_stride", 1)))


def build_splits(
    dcfg,
    *,
    dist=None,
    train_dtype: Optional[str] = None,
    val_overrides=None,
    stats: Optional[dict] = None,
    run_stats_dir: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs,
) -> tuple[CycloneDataset, CycloneDataset]:
    """Train and val datasets sharing the training split's normalization statistics.

    With ``dataset.parts`` (one dataset section per data type, the other ``dataset`` keys as
    their defaults) both splits are a :class:`CycloneMix` of the parts; ``dataset.mixing``
    schedules the training share of every part. A part's ``stem`` names the model stem
    decoding it and ``stats_from`` another part whose statistics it uses. ``run_stats_dir``
    holds a run's copies of the statistics (``<part>.pkl``), preferred when present.
    """
    if dcfg.get("parts"):
        shared = {k: v for k, v in to_dict(dcfg).items() if k not in ("parts", "mixing")}
        parts = to_dict(dcfg.parts)
        splits, stems = {}, {}
        for part_name, part in parts.items():
            source = part.get("stats_from")
            if source is not None and source not in splits:
                raise ValueError(
                    f"dataset.parts.{part_name}.stats_from {source!r} must name an earlier part"
                )
            stems[part_name] = part.get("stem", part_name)
            splits[part_name] = build_splits(
                OmegaConf.create({**shared, **part}),
                dist=dist,
                train_dtype=train_dtype,
                val_overrides=val_overrides,
                stats=splits[source][0].stats if source is not None else None,
                run_stats_dir=run_stats_dir,
                name=part_name,
                **kwargs,
            )
        mixing = to_dict(dcfg.get("mixing")) or None
        return (
            CycloneMix({n: s[0] for n, s in splits.items()}, mixing=mixing, stems=stems),
            CycloneMix({n: s[1] for n, s in splits.items()}, stems=stems),
        )
    if stats is None:
        run_stats = (
            os.path.join(run_stats_dir, f"{name or 'dataset'}.pkl") if run_stats_dir else None
        )
        stats = resolve_stats(dcfg, run_stats=run_stats, dist=dist, **kwargs)
    train = build_dataset(
        dcfg, split="train", dist=dist, prefer_dtype=train_dtype, stats=stats, **kwargs
    )
    # dataset.prefer_dtype is the stored precision, so validation reads it too
    val = build_dataset(
        dcfg,
        split="val",
        dist=dist,
        stats=train.stats if stats is not None else None,
        prefer_dtype=dcfg.get("prefer_dtype"),
        **{**kwargs, **(val_overrides or {})},
    )
    return train, val


def save_run_stats(ds, directory: str, overwrite: bool = True) -> None:
    """Writes the statistics of ``ds`` (each part of a mix) as ``<part>.pkl`` in ``directory``."""
    os.makedirs(directory, exist_ok=True)
    parts = getattr(ds, "parts", None) or {"dataset": ds}
    for name, part in parts.items():
        path = os.path.join(directory, f"{name}.pkl")
        if part.stats and (overwrite or not os.path.exists(path)):
            save_stats(path, part.stats)


def write_stats(dcfg, *, dist=None) -> None:
    """Write the ``normalization_stats`` pickle of the dataset (every part of a mix).

    The statistics pool every ``dataset.stats_stride``-th unnormalized training sample, after
    the offset, kx crop and zonal-flow split; parts with ``stats_from`` have none of their own.
    """
    shared = {k: v for k, v in to_dict(dcfg).items() if k not in ("parts", "mixing")}
    parts = to_dict(dcfg.get("parts")) or {None: {}}
    for name, part in parts.items():
        if part.get("stats_from") is not None:
            continue
        cfg = OmegaConf.create({**shared, **part})
        path = cfg.get("normalization_stats")
        if not isinstance(path, str):
            raise ValueError(f"dataset part {name!r} has no normalization_stats path to write")
        ds = build_dataset(cfg, split="train", dist=dist, stats={})
        stats = stream_stats(ds, to_dict(cfg.get("normalization")), int(cfg.get("stats_stride", 1)))
        if dist is None or dist.is_rank0:
            save_stats(path, stats)
            print(f"[stats] {name or 'dataset'}: {len(ds)} samples -> {path}")
