"""Unified ``CycloneDataset`` for both AE training and latent diffusion.

``mode="ae"`` returns raw distribution-function tensors; ``mode="diff"``
returns precomputed latents (after running ``precompute_latents``);
``mode="next"`` additionally returns next-step targets (``y_df``, ``y_phi``,
``y_flux``, ``y_fluxavg``) for autoregressive training.

Returns a frozen ``CycloneSample`` dataclass; the frames carry the array type of the
backend (host numpy, or device jax for ``KvikIOBackend``), the scalars are numpy.
"""

from __future__ import annotations

import copy
import itertools
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from neugk_jax.dataset.backend import DataBackend, expand_spec
from neugk_jax.dataset.normalization import NormTable, load_stats, metadata_stats
from neugk_jax.utils import RunningStats
from neugk_jax.utils import separate_zf as separate_zf_fn

# heat-flux steps averaged into the trajectory flux
FLUX_AVG_WINDOW = 80
# condition names of the scalar trajectory parameters -> their metadata keys
COND_META_KEYS = {"itg": "ion_temp_grad", "dg": "density_grad", "s_hat": "s_hat", "q": "q"}
DEFAULT_CONDITIONS = tuple(COND_META_KEYS)
# kinetic-electron conditions -> (metadata key, value of an adiabatic-electron trajectory)
KINETIC_COND_KEYS = {"etg": ("electron_temp_grad", 0.0), "temp_ratio": ("temp_ratio", 1.0)}
# 1 for kinetic-electron trajectories, 0 for adiabatic ones
KINETIC_FLAG = "kinetic"


def avg_flux(flux):
    flux = np.asarray(flux)
    # a single-snapshot trajectory averages its only frame
    return np.mean(flux[1:][-FLUX_AVG_WINDOW:] if len(flux) > 1 else flux, axis=0)


def n_species(meta: dict) -> int:
    return int(meta.get("n_species", 1))


def heat_flux(meta: dict) -> np.ndarray:
    """Heat flux per timestep, ``(T,)`` for one species and ``(T, species)`` for several."""
    flux = np.asarray(meta["flux"], dtype=np.float64)
    ns = n_species(meta)
    if ns == 1:
        return flux
    # kinetic metadata: (T, species * (pflux, eflux, vflux))
    return flux.reshape(len(flux), ns, -1)[..., 1]


def _f32(x):
    return None if x is None else x.astype(np.float32)


@dataclass(frozen=True)
class CycloneSample:
    """A single dataset item. ``df`` is the raw distribution function (mode='ae')
    or a precomputed latent tensor (mode='diff')."""

    df: np.ndarray | None
    phi: np.ndarray | None
    flux: np.ndarray
    avg_flux: np.ndarray
    timestep: np.ndarray
    file_index: np.ndarray
    timestep_index: np.ndarray
    conditioning: np.ndarray | None
    # raw scalar conditions (also packed into conditioning if requested)
    itg: np.ndarray
    dg: np.ndarray
    s_hat: np.ndarray
    q: np.ndarray
    # next-step targets (mode="next")
    y_df: np.ndarray | None = None
    y_phi: np.ndarray | None = None
    y_flux: np.ndarray | None = None
    y_fluxavg: np.ndarray | None = None
    # data type of the sample (its dataset's index in a mix) and its number of species
    data_type: np.ndarray | None = None
    n_species: np.ndarray | None = None
    loss_weight: np.ndarray | None = None


class CycloneDataset:
    """Map-style dataset over preprocessed gyrokinetics trajectories.

    Parameters
    ----------
    path, trajectories
        Directory + trajectory spec (string with ``{1-5,7}`` ranges or list).
    backend
        The :class:`DataBackend` reading the trajectories (see ``make_backend``).
    split
        ``"train"`` or ``"val"``.
    fields_to_load
        Subset of ``("df", "phi")`` to read from disk. ``flux`` is always
        derivable from metadata.
    conditions
        Scalar fields to pack into ``CycloneSample.conditioning`` (e.g.
        ``("itg", "dg", "s_hat", "q")``).
    mode
        ``"ae"`` returns raw df reads; ``"diff"`` returns precomputed latents
        (after calling :func:`precompute_latents`); ``"next"`` returns the
        input at ``t`` plus normalized targets at ``t + bundle_seq_length``.
    normalization, normalization_scope, normalization_stats
        Normalization config, its scope (``"dataset"`` or per-trajectory),
        and optional precomputed stats.
    separate_zf
        Optional channel-axis preprocessing matching the AE config.
    bundle_seq_length
        Time-bundling stride. Default ``1`` (one timestep per sample).
    offset
        Number of leading timesteps to skip per trajectory; at least the ``first_frame``
        (first stored frame) of every trajectory.
    species_axis
        Serve df as ``(C, species, vpar, mu, s, x, y)`` (adiabatic trajectories get a
        unit species axis) and the flux per species. Required for kinetic trajectories.
    kx_crop
        Optional number of central kx modes kept of trajectories with more (df and phi are
        cropped in Fourier space on read). Unset, frames are served with their stored kx modes
        and a mismatch raises.
    data_type
        Data type id served with every sample (the dataset's index in a mix).
    loss_weight
        Weight of this dataset's samples in the training loss, served with every sample
        (set from ``dataset.mixing`` by a mix).
    augment
        Random symmetries applied to the raw df of every training sample, all off unless
        given: ``roll_y`` (a random roll along y, an exact symmetry of the flux tube) and
        ``amplitude: [lo, hi]`` (a log-uniform rescaling, exact for linear states only; it
        scales the fluxes by its square).
    """

    def __init__(
        self,
        *,
        path: str,
        backend: DataBackend,
        split: str = "train",
        trajectories: Optional[Any] = None,
        fields_to_load: Sequence[str] = ("df",),
        conditions: Sequence[str] = DEFAULT_CONDITIONS,
        mode: str = "ae",
        normalization: Optional[dict] = None,
        normalization_scope: str = "dataset",
        normalization_stats: Optional[dict | str] = None,
        cond_filters: Optional[dict] = None,
        bundle_seq_length: int = 1,
        offset: int = 0,
        tail_offset: int = 0,
        subsample: int = 1,
        separate_zf: bool = False,
        species_axis: bool = False,
        kx_crop: Optional[int] = None,
        data_type: int = 0,
        loss_weight: float = 1.0,
        augment: Optional[dict] = None,
        rank: int = 0,
    ):
        assert split in ("train", "val")
        assert mode in ("ae", "diff", "next")
        if mode == "next" and bundle_seq_length != 1:
            raise NotImplementedError("mode='next' supports bundle_seq_length=1 only")
        self.path = path
        self.split = split
        self.fields_to_load = list(fields_to_load)
        # sort alphabetically to keep conditioning slot order consistent across runs
        self.conditions = sorted(conditions)
        self.mode = mode
        self.normalization = normalization
        self.normalization_scope = normalization_scope
        self.normalization_stats = normalization_stats
        self.cond_filters = cond_filters or {}
        self.bundle_seq_length = bundle_seq_length
        self.offset = offset
        self.tail_offset = tail_offset
        self.subsample = subsample
        self.separate_zf = separate_zf
        self.species_axis = species_axis
        self.kx_crop = kx_crop
        self.data_type = data_type
        self.loss_weight = float(loss_weight)
        self.augment = {k: v for k, v in (augment or {}).items() if v}
        self._augment_calls = itertools.count()
        # model stem decoding this dataset (set by a mix)
        self.stem = None
        self._batch_transform = False
        self.backend = backend
        self.rank = rank

        # latent storage (mode="diff"); filled by precompute_latents()
        self.precomputed_latents: dict[tuple[int, int], dict] | None = None
        self.latent_stats: RunningStats | None = None

        if trajectories is None:
            raw = [
                p for p in (os.path.join(path, n) for n in os.listdir(path)) if backend.is_valid(p)
            ]
        else:
            raw = [os.path.join(path, n) for n in expand_spec(trajectories)]
        self.files = sorted({p for p in map(backend.trajectory_path, raw) if backend.is_valid(p)})
        if not self.files:
            raise RuntimeError(f"no trajectories found under {path}")

        with ThreadPoolExecutor(max_workers=8) as ex:
            metas = list(ex.map(backend.read_metadata, self.files))
        self.metadata: dict[int, dict] = {}
        kept_files = []
        # metadata keys every sample needs, plus any other conditioning field
        required = {*COND_META_KEYS.values(), "flux", "timesteps"}
        optional = {*KINETIC_COND_KEYS, KINETIC_FLAG, "timestep"}
        required |= {c for c in self.conditions if c not in COND_META_KEYS and c not in optional}
        for fp, meta in zip(self.files, metas):
            if not self._passes_cond_filter(meta):
                continue
            missing = sorted(k for k in required if k not in meta)
            if missing:
                # traj missing a conditioning/metadata field -> exclude it rather than crash
                if self.rank == 0:
                    warnings.warn(f"{fp}: missing metadata {missing}; excluding trajectory")
                continue
            if offset < int(meta.get("first_frame", 0)):
                raise ValueError(f"{fp}: frames before {meta['first_frame']} are not stored (offset={offset})")
            if n_species(meta) > 1 and not species_axis:
                raise ValueError(f"{fp}: {n_species(meta)} species need species_axis=True")
            fid = len(kept_files)
            kept_files.append(fp)
            self.metadata[fid] = meta
        self.files = kept_files

        self.flat_index_to_file_and_tstep: dict[int, tuple[int, int]] = {}
        self.file_num_timesteps: list[int] = []
        flat = 0
        for fid, meta in self.metadata.items():
            timesteps = meta["timesteps"][offset:]
            self.file_num_timesteps.append(len(timesteps))
            if tail_offset > 0:
                timesteps = timesteps[:-tail_offset]
            n = len(timesteps[::subsample]) - bundle_seq_length * 2 + 1
            # a single-snapshot trajectory still serves its frame unless a next-step target is needed
            if mode != "next" and len(timesteps) == 1:
                n = 1
            for t_idx in range(max(0, n)):
                self.flat_index_to_file_and_tstep[flat] = (fid, t_idx * subsample)
                flat += 1
        self.length = flat

        # resolution: assume same across files
        meta0 = self.metadata[0]
        self.n_species = n_species(meta0)
        stored = tuple(meta0["resolution"])
        self.stored_df_shape = tuple(meta0.get("df_shape", (2, *stored)))
        self.stored_phi_shape = (stored[3], stored[2], stored[4])
        nx = stored[3] if kx_crop is None else min(stored[3], int(kx_crop))
        self.resolution = (*stored[:3], nx, stored[4])
        self.df_shape = (2, self.n_species, *self.resolution) if species_axis else (2, *self.resolution)
        self.phi_resolution = (self.resolution[3], self.resolution[2], self.resolution[4])

        # normalization_stats when given, else metadata moments (direct construction only, build_dataset resolves data stats)
        insert = species_axis and len(self.stored_df_shape) == 6
        if isinstance(normalization_stats, (str, os.PathLike)) and not os.path.exists(normalization_stats):
            raise FileNotFoundError(f"normalization_stats {normalization_stats} does not exist")
        if isinstance(normalization_stats, (str, os.PathLike)):
            self.stats = load_stats(normalization_stats, normalization)
        elif normalization_stats is not None:
            self.stats = normalization_stats
        elif normalization:
            self.stats = metadata_stats(self.metadata, self.fields_to_load, normalization, insert)
        else:
            self.stats = {}
        if insert and "df" in self.stats:
            # single-species stats still without it get the unit species axis of the served df
            self.stats = {**self.stats, "df": {
                k: {n: np.expand_dims(a, 1) if np.ndim(a) == 6 else a for n, a in v.items()}
                for k, v in self.stats["df"].items()
            }}
        ndims = {"df": len(self.df_shape), "phi": len(self.phi_resolution), "flux": 0, "fluxavg": 0}
        self.norm = NormTable.from_stats(
            self.stats, normalization, normalization_scope, len(self.files), ndims
        )

    def _passes_cond_filter(self, meta: dict) -> bool:
        for cond_name, cond_range in self.cond_filters.items():
            where = None
            if "_" in cond_name:
                where, cond_name = cond_name.split("_", 1)
            if cond_name not in meta:
                return False
            cond = meta[cond_name]
            if not isinstance(cond_range[0], (list, tuple)):
                cond_range = [cond_range]
            if cond_name == "flux":
                bound = self.offset if self.offset > 0 else FLUX_AVG_WINDOW
                # ion heat flux
                cond = heat_flux(meta)
                cond = cond if cond.ndim == 1 else cond[:, 0]
                cond = float(np.mean(cond[:bound] if where == "first" else cond[-bound:]))
            if not any(lo <= cond <= hi for lo, hi in cond_range):
                return False
        return True

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> CycloneSample:
        return self.sample(*self.flat_index_to_file_and_tstep[index])

    def with_mode(self, mode: str) -> "CycloneDataset":
        view = copy.copy(self)
        view.mode = mode
        return view

    def _frames(self, handle, fid: int, t: int) -> dict:
        """Normalized float32 ``df``/``phi`` frames (those loaded) at raw frame index ``t``."""
        out, raw = {}, {}
        if "df" in self.fields_to_load:
            df = self.backend.read_df(handle, t, self.stored_df_shape)
            if self.batch_transform:
                raw["df"] = df
            else:
                out["df"] = self.transform(df[None], np.asarray([fid]), normalize=False)[0]
        if "phi" in self.fields_to_load:
            phi = self.backend.read_phi(handle, t, self.stored_phi_shape)
            if phi.shape[0] != self.resolution[3]:
                if self.kx_crop is None:
                    raise ValueError(f"phi has {phi.shape[0]} kx modes, the dataset {self.resolution[3]}; set kx_crop to crop")
                from neugk_jax.evaluate.fourier import crop_kx_phi

                phi = crop_kx_phi(phi, self.resolution[3])
            out["phi"] = phi
        return {**raw, **{k: _f32(self.norm.normalize(k, v, fid)) for k, v in out.items()}}

    @property
    def batch_transform(self) -> bool:
        """Serve raw df frames (16-bit shards as stored); :meth:`transform` runs per batch."""
        return self._batch_transform

    @batch_transform.setter
    def batch_transform(self, on: bool) -> None:
        self._batch_transform = bool(on)
        if hasattr(self.backend, "keep_half"):
            self.backend.keep_half = self._batch_transform

    def transform(self, df, fids, normalize: bool = True):
        """The df frame transform on a raw batch ``(B, 2, ...)``.

        float32, kx crop, unit species axis, zonal-flow split and (with ``normalize``)
        normalization.
        """
        x = df.astype(np.float32)
        if x.shape[-2] != self.resolution[3]:
            if self.kx_crop is None:
                raise ValueError(f"df has {x.shape[-2]} kx modes, the dataset {self.resolution[3]}; set kx_crop to crop")
            from neugk_jax.evaluate.fourier import crop_kx_df

            x = crop_kx_df(x, self.resolution[3], reim_axis=1)
        if self.species_axis and x.ndim == 7:
            x = x[:, :, None]
        if self.augment:
            x = self._augmented(x)
        x = separate_zf_fn(x, axis=1) if self.separate_zf else x
        return self.norm.normalize("df", x, fids).astype(np.float32) if normalize else x

    def _augmented(self, x):
        """``augment`` on a raw df batch ``(B, 2, ..., x, y)``, independently per sample."""
        rng = np.random.default_rng([self.rank, next(self._augment_calls)])
        xp = x.__array_namespace__() if hasattr(x, "__array_namespace__") else np
        rows = []
        for row in x:
            if self.augment.get("roll_y"):
                row = xp.roll(row, int(rng.integers(row.shape[-1])), axis=-1)
            if self.augment.get("amplitude"):
                lo, hi = (float(v) for v in self.augment["amplitude"])
                row = row * float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            rows.append(row)
        return xp.stack(rows)

    def read_frame(self, fid: int, t: int) -> dict:
        """The loaded fields of trajectory ``fid`` at raw frame index ``t``, normalized."""
        with self.backend.open(self.files[int(fid)]) as handle:
            return self._frames(handle, int(fid), int(t))

    def _targets(self, handle, fid: int, t_idx: int) -> dict:
        """Normalized next-step targets (frames, ``flux``, ``fluxavg``) of sample ``(fid, t_idx)``."""
        t = t_idx + self.offset + self.bundle_seq_length
        flux = self.flux(fid)
        out = self._frames(handle, fid, t)
        out["flux"] = np.asarray(flux[t], dtype=np.float32)
        out["fluxavg"] = np.asarray(avg_flux(flux), dtype=np.float32)
        for k in ("flux", "fluxavg"):
            out[k] = np.asarray(self.norm.normalize(k, out[k], fid), np.float32)
        return out

    def get_target(self, fid: int, t_idx: int) -> dict[str, np.ndarray]:
        fid, t_idx = int(fid), int(t_idx)
        with self.backend.open(self.files[fid]) as handle:
            return self._targets(handle, fid, t_idx)

    def sample(self, fid: int, t_idx: int) -> CycloneSample:
        """The sample of trajectory ``fid`` at timestep index ``t_idx`` in the current mode."""
        meta, t = self.metadata[fid], t_idx + self.offset
        flux, timestep = self.flux(fid)[t], meta["timesteps"][t]
        targets = {}
        if self.mode == "diff" and self.precomputed_latents is not None:
            cached = self.precomputed_latents[(fid, t_idx)]
            frames = {"df": cached["x"].astype(np.float32), "phi": cached.get("phi")}
            flux, timestep = cached.get("flux", flux), cached.get("timestep", timestep)
        else:
            with self.backend.open(self.files[fid]) as handle:
                frames = self._frames(handle, fid, t)
                if self.mode == "next":
                    targets = self._targets(handle, fid, t_idx)
        timestep = np.asarray(timestep, dtype=np.float32)
        return CycloneSample(
            df=frames.get("df"),
            phi=frames.get("phi"),
            flux=np.asarray(flux, dtype=np.float32),
            avg_flux=np.asarray(self.get_avg_flux(fid), dtype=np.float32),
            timestep=timestep,
            file_index=np.int64(fid),
            timestep_index=np.int64(t_idx),
            conditioning=self.conditioning(fid, timestep),
            **{
                k: np.asarray(np.squeeze(meta[v]), dtype=np.float32)
                for k, v in COND_META_KEYS.items()
            },
            **{f"y_{k}": v for k, v in targets.items()},
            data_type=np.int64(self.data_type),
            n_species=np.int64(self.n_species),
            loss_weight=np.float32(self.loss_weight),
        )

    def conditioning(self, fid: int, timestep) -> Optional[np.ndarray]:
        """The ``conditions`` of trajectory ``fid`` at ``timestep``, packed in sorted order."""
        if not self.conditions:
            return None
        meta = self.metadata[fid]
        return np.concatenate(
            [np.atleast_1d(np.asarray(self.condition_value(meta, k, timestep), np.float32)) for k in self.conditions]
        )

    @staticmethod
    def condition_value(meta: dict, name: str, timestep=None):
        """Value of condition ``name`` of a trajectory; adiabatic ones take the kinetic defaults."""
        if name == "timestep":
            return timestep
        if name == KINETIC_FLAG:
            return float(n_species(meta) > 1)
        if name in KINETIC_COND_KEYS:
            key, default = KINETIC_COND_KEYS[name]
            return np.squeeze(meta[key]) if key in meta else default
        return np.squeeze(meta[COND_META_KEYS.get(name, name)])

    def flux(self, fid: int) -> np.ndarray:
        """Heat flux of trajectory ``fid`` per timestep, ``(T, species)`` with ``species_axis``."""
        flux = heat_flux(self.metadata[fid])
        return flux[:, None] if self.species_axis and flux.ndim == 1 else flux

    def num_ts(self, fid: int) -> int:
        """Raw timesteps of trajectory ``fid`` after ``offset``, tail and subsampled ones included.

        Comparable with ``timestep_index`` (a raw index); the frame at raw index
        ``t < num_ts`` is on disk.
        """
        return self.file_num_timesteps[int(fid)]

    def get_timestep(self, fid: int, t_idx: int) -> np.ndarray:
        return np.asarray(
            self.metadata[int(fid)]["timesteps"][int(t_idx) + self.offset], dtype=np.float32
        )

    def get_avg_flux(self, fid: int):
        return avg_flux(self.flux(fid))

    def geometry(self, fid: int) -> Optional[dict]:
        """Geometry of trajectory ``fid`` on the served kx grid.

        Kinetic runs without a stored geometry take it (and ``ds``) from their source config.
        """
        meta = self.metadata[int(fid)]
        if "source_config" in meta:
            # gyaradax runs preprocessed before the geometry / ds were stored
            if "geometry" not in meta:
                from omegaconf import OmegaConf

                from neugk_jax.dataset.preprocess import kinetic_geometry

                meta["geometry"] = kinetic_geometry(OmegaConf.load(meta["source_config"]))
            meta.setdefault("ds", float(np.asarray(meta["geometry"]["ints"]).ravel()[0]))
        geom = meta.get("geometry")
        if geom is None or len(geom["kxrh"]) == self.resolution[3]:
            return geom
        if self.kx_crop is None:
            raise ValueError(f"geometry has {len(geom['kxrh'])} kx modes, the dataset {self.resolution[3]}; set kx_crop to crop")
        # central kx modes, as kept by kx_crop
        i0 = (len(geom["kxrh"]) - self.resolution[3]) // 2
        return {**geom, "kxrh": np.asarray(geom["kxrh"])[i0 : i0 + self.resolution[3]]}

    def get_ds(self, fid: int) -> float | None:
        # parallel (s) grid spacing; None when the trajectory metadata doesn't carry it
        if "source_config" in self.metadata[fid]:
            self.geometry(fid)
        ds = self.metadata[fid].get("ds")
        return None if ds is None else float(ds)

    def spectral_stats(self, key: str) -> dict[str, np.ndarray]:
        """Per-mode ``mean`` / ``std`` of ``log1p`` of a served spectrum (``kyspec``, ``fluxspec``).

        Pooled over the trajectories from ``offset`` on, each weighted by its timestep count.
        """
        missing = [self.files[f] for f, m in self.metadata.items() if key not in m]
        if missing:
            raise KeyError(f"no {key!r} in the metadata of {len(missing)} trajectories")
        rms = RunningStats(prior_count=1e-4)
        for meta in self.metadata.values():
            spec = np.log1p(np.asarray(meta[key], dtype=np.float64)[self.offset :])
            rms.merge(
                spec.mean(axis=0),
                spec.var(axis=0),
                spec.min(axis=0),
                spec.max(axis=0),
                count=len(meta["timesteps"][self.offset :]),
            )
        stats = rms.moments(np.float32)
        return {"mean": stats["mean"], "std": stats["std"]}
