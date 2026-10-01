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
import dataclasses
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
        Number of leading timesteps to skip per trajectory.
    """

    def __init__(
        self,
        *,
        path: str,
        backend: DataBackend,
        split: str = "train",
        trajectories: Optional[Any] = None,
        fields_to_load: Sequence[str] = ("df",),
        conditions: Sequence[str] = ("itg", "dg", "s_hat", "q"),
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

        # metadata loads are I/O bound and tiny
        with ThreadPoolExecutor(max_workers=8) as ex:
            metas = list(ex.map(backend.read_metadata, self.files))
        self.metadata: dict[int, dict] = {}
        kept_files = []
        # metadata keys _build_sample hard-requires, plus any non-alias conditioning field
        required = {"ion_temp_grad", "density_grad", "s_hat", "q", "flux", "timesteps"}
        required |= {c for c in self.conditions if c not in ("itg", "dg", "s_hat", "q", "timestep")}
        for fp, meta in zip(self.files, metas):
            if not self._passes_cond_filter(meta):
                continue
            missing = sorted(k for k in required if k not in meta)
            if missing:
                # traj missing a conditioning/metadata field -> exclude it rather than crash
                if self.rank == 0:
                    warnings.warn(f"{fp}: missing metadata {missing}; excluding trajectory")
                continue
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
            for t_idx in range(max(0, n)):
                self.flat_index_to_file_and_tstep[flat] = (fid, t_idx * subsample)
                flat += 1
        self.length = flat

        # resolution: assume same across files
        self.resolution = tuple(self.metadata[0]["resolution"])
        self.df_shape = (2, *self.resolution)
        self.phi_resolution = (self.resolution[3], self.resolution[2], self.resolution[4])

        # normalization_stats when given, else the per-trajectory metadata moments
        if isinstance(normalization_stats, (str, os.PathLike)):
            self.stats = load_stats(normalization_stats, normalization)
        elif normalization_stats is not None:
            self.stats = normalization_stats
        else:
            self.stats = metadata_stats(self.metadata, self.fields_to_load) if normalization else {}
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
                bound = self.offset if self.offset > 0 else 80
                cond = float(np.mean(cond[:bound] if where == "first" else cond[-bound:]))
            if not any(lo <= cond <= hi for lo, hi in cond_range):
                return False
        return True

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> CycloneSample:
        fid, t_idx = self.flat_index_to_file_and_tstep[index]
        if self.mode == "diff" and self.precomputed_latents is not None:
            return self._get_latent_sample(fid, t_idx)
        if self.mode == "next":
            return self._get_next_sample(fid, t_idx)
        return self._get_ae_sample(fid, t_idx)

    def with_mode(self, mode: str) -> "CycloneDataset":
        view = copy.copy(self)
        view.mode = mode
        return view

    def get_target(self, fid: int, t_idx: int) -> dict[str, np.ndarray]:
        """Normalized next-step targets ``y_*`` of ``(fid, t_idx)`` without reading the input."""
        fid, t_idx = int(fid), int(t_idx)
        meta = self.metadata[fid]
        gt_t = t_idx + self.offset + self.bundle_seq_length
        out: dict[str, np.ndarray] = {}
        with self.backend.open(self.files[fid]) as handle:
            if "df" in self.fields_to_load:
                y_df = self.backend.read_df(handle, gt_t, self.df_shape)
                if self.separate_zf:
                    y_df = separate_zf_fn(y_df, axis=0)
                out["df"] = y_df
            if "phi" in self.fields_to_load:
                out["phi"] = self.backend.read_phi(handle, gt_t, self.phi_resolution)
        out["flux"] = np.asarray(meta["flux"][gt_t], dtype=np.float32)
        out["fluxavg"] = np.asarray(np.mean(np.asarray(meta["flux"])[1:][-80:]), dtype=np.float32)
        return {k: _f32(self.norm.normalize(k, v, fid)) for k, v in out.items()}

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

    def _get_ae_sample(self, fid: int, t_idx: int) -> CycloneSample:
        meta = self.metadata[fid]
        original_t = t_idx + self.offset
        f_path = self.files[fid]
        df = phi = None
        with self.backend.open(f_path) as handle:
            if "df" in self.fields_to_load:
                df = self.backend.read_df(handle, original_t, self.df_shape)
                if self.separate_zf:
                    df = separate_zf_fn(df, axis=0)
            if "phi" in self.fields_to_load:
                phi = self.backend.read_phi(handle, original_t, self.phi_resolution)

        flux = np.asarray(meta["flux"][original_t], dtype=np.float32)
        timestep = np.asarray(meta["timesteps"][original_t], dtype=np.float32)
        if df is not None and self.normalization is not None:
            df = _f32(self.norm.normalize("df", df, fid))
        if phi is not None and self.normalization is not None:
            phi = _f32(self.norm.normalize("phi", phi, fid))

        return self._build_sample(fid, t_idx, df, phi, flux, timestep, meta)

    def _get_next_sample(self, fid: int, t_idx: int) -> CycloneSample:
        meta = self.metadata[fid]
        original_t = t_idx + self.offset
        gt_t = original_t + self.bundle_seq_length
        df = y_df = phi = y_phi = None
        with self.backend.open(self.files[fid]) as handle:
            if "df" in self.fields_to_load:
                df = self.backend.read_df(handle, original_t, self.df_shape)
                y_df = self.backend.read_df(handle, gt_t, self.df_shape)
                if self.separate_zf:
                    df = separate_zf_fn(df, axis=0)
                    y_df = separate_zf_fn(y_df, axis=0)
            if "phi" in self.fields_to_load:
                phi = self.backend.read_phi(handle, original_t, self.phi_resolution)
                y_phi = self.backend.read_phi(handle, gt_t, self.phi_resolution)

        flux = np.asarray(meta["flux"][original_t], dtype=np.float32)
        y_flux = np.asarray(meta["flux"][gt_t], dtype=np.float32)
        y_fluxavg = np.asarray(np.mean(np.asarray(meta["flux"])[1:][-80:]), dtype=np.float32)
        timestep = np.asarray(meta["timesteps"][original_t], dtype=np.float32)
        if self.normalization is not None:
            norm = self.norm.normalize
            if df is not None:
                df, y_df = norm("df", df, fid), norm("df", y_df, fid)
            if phi is not None:
                phi, y_phi = norm("phi", phi, fid), norm("phi", y_phi, fid)
            y_flux = norm("flux", y_flux, fid)
            y_fluxavg = norm("fluxavg", y_fluxavg, fid)
        sample = self._build_sample(fid, t_idx, _f32(df), _f32(phi), flux, timestep, meta)
        return dataclasses.replace(
            sample,
            y_df=_f32(y_df),
            y_phi=_f32(y_phi),
            y_flux=np.asarray(y_flux, np.float32),
            y_fluxavg=np.asarray(y_fluxavg, np.float32),
        )

    def _get_latent_sample(self, fid: int, t_idx: int) -> CycloneSample:
        cached = self.precomputed_latents[(fid, t_idx)]
        meta = self.metadata[fid]
        flux = np.asarray(cached.get("flux", meta["flux"][t_idx + self.offset]), dtype=np.float32)
        timestep = np.asarray(
            cached.get("timestep", meta["timesteps"][t_idx + self.offset]), dtype=np.float32
        )
        return self._build_sample(
            fid,
            t_idx,
            cached["x"].astype(np.float32),
            cached.get("phi"),
            flux,
            timestep,
            meta,
        )

    def _build_sample(self, fid, t_idx, df, phi, flux, timestep, meta) -> CycloneSample:
        itg = np.asarray(np.squeeze(meta["ion_temp_grad"]), dtype=np.float32)
        dg = np.asarray(np.squeeze(meta["density_grad"]), dtype=np.float32)
        s_hat = np.asarray(np.squeeze(meta["s_hat"]), dtype=np.float32)
        q = np.asarray(np.squeeze(meta["q"]), dtype=np.float32)
        cond_vec = None
        if self.conditions:
            packed = []
            local = {"itg": itg, "dg": dg, "s_hat": s_hat, "q": q, "timestep": timestep}
            for k in self.conditions:
                v = local.get(k)
                if v is None:
                    v = np.asarray(np.squeeze(meta[k]), dtype=np.float32)
                packed.append(np.atleast_1d(v))
            cond_vec = np.concatenate(packed).astype(np.float32)

        avg = float(np.mean(meta["flux"][-80:]))
        return CycloneSample(
            df=df,
            phi=phi,
            flux=flux,
            avg_flux=np.float32(avg),
            timestep=timestep,
            file_index=np.int64(fid),
            timestep_index=np.int64(t_idx),
            conditioning=cond_vec,
            itg=itg,
            dg=dg,
            s_hat=s_hat,
            q=q,
        )

    def get_avg_flux(self, fid: int) -> float:
        return float(np.mean(self.metadata[fid]["flux"][-80:]))

    def get_ds(self, fid: int) -> float | None:
        # parallel (s) grid spacing; None when the trajectory metadata doesn't carry it
        ds = self.metadata[fid].get("ds")
        return None if ds is None else float(ds)
