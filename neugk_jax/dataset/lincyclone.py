"""``LinearCondCycloneDataset`` — cyclone samples carrying the paired linear field.

Every nonlinear trajectory ``iteration_N`` has a linear GKW run ``iteration_N_Lin``
whose single ``FDS`` snapshot is the eigenmode of that operating point: one 5D field
per trajectory, identical for all of its timesteps. This dataset resolves that pairing
(from the ``linear_fds`` / ``linear_bin`` metadata keys written by
``preprocess --mode=linear``, or by searching the raw roots directly), keeps **one
array per trajectory** in a shared cache, and attaches a reference to it on every
:class:`~neugk_jax.dataset.cyclone.CycloneSample` — the field is never duplicated
per sample, only per batch when the collate stacks it.

The amplitude of a linear run is arbitrary (it grows as ``exp(γt)``), so only the mode
shape carries information: ``linear_normalize="rms"`` (default) divides each field by
its own RMS.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Optional, Sequence

import numpy as np

from neugk_jax.dataset.cyclone import CycloneDataset, CycloneSample
from neugk_jax.dataset.linear import (
    default_linear_roots,
    k_to_real,
    load_linear_field,
    resolve_linear_dir,
    strip_format_tag,
)
from neugk_jax.utils import separate_zf as separate_zf_fn


def _as_str(v) -> str:
    # npz metadata round-trips strings as 0-d numpy arrays
    return v if isinstance(v, str) else str(np.asarray(v).item())


class LinearCondCycloneDataset(CycloneDataset):
    """Cyclone dataset whose samples carry their trajectory's linear eigenmode field.

    Parameters
    ----------
    raw_root, linear_roots
        Where to look for ``<traj>_Lin`` dirs when the metadata has no ``linear_fds``.
    linear_to_real
        ``True`` applies the kx/ky inverse transform so the field matches the ``df``
        layout; ``False`` feeds the semispectral field as GKW wrote it, which is what
        the conditioning encoder wants (it never reconstructs). Selects which
        materialized flavour is read (``data/linear.bin`` vs ``data/linear_k.bin``).
    linear_separate_zf
        Split the field into ``[zf, rest]`` channels. Defaults to the ``separate_zf``
        used for ``df``, so the conditioner and the AE see the same channel layout.
    linear_normalize
        ``"rms"``: divide each field by its own RMS (kills the arbitrary ``exp(γt)``
        amplitude). ``"ky"`` / ``"kxky"``: that, then divide by a **shared** profile
        aggregated over all trajectories, which flattens the spectral decay without
        erasing any trajectory's own deviation from the mean spectrum (dividing by a
        per-trajectory spectrum would erase exactly the fingerprint we condition on).
        ``"zscore"`` | ``"none"`` also accepted. The profile needs the per-trajectory
        moments from ``preprocess --mode=linear --stats``.
    linear_preload
        Load every trajectory's field at construction (``linear_workers`` threads)
        instead of lazily on first use.
    linear_cache_size
        Cap on cached fields (FIFO eviction). ``None`` keeps all of them —
        ~89 MB fp32 per trajectory.
    linear_required
        Raise when a trajectory has no linear pair; otherwise serve zeros.
    """

    def __init__(
        self,
        *,
        raw_root: str = "/restricteddata/ukaea/gyrokinetics/raw",
        linear_roots: Optional[Sequence[str]] = None,
        linear_to_real: bool = True,
        linear_separate_zf: Optional[bool] = None,
        linear_normalize: str = "cky",
        linear_rescale_per_traj: bool = False,
        linear_profile: Optional[np.ndarray] = None,
        linear_preload: bool = True,
        linear_cache_size: Optional[int] = None,
        linear_workers: int = 8,
        linear_required: bool = True,
        linear_dtype: str = "float32",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert linear_normalize in ("rms", "ky", "cky", "kxky", "ckxky", "zscore", "none")
        self.linear_rescale_per_traj = linear_rescale_per_traj
        self._linear_profile_in = linear_profile
        self.linear_to_real = linear_to_real
        self.linear_separate_zf = (
            self.separate_zf if linear_separate_zf is None else linear_separate_zf
        )
        self.linear_normalize = linear_normalize
        self.linear_cache_size = linear_cache_size
        self.linear_required = linear_required
        self.linear_dtype = np.dtype(linear_dtype)
        self.linear_roots = list(linear_roots) if linear_roots else default_linear_roots(raw_root)

        # traj_id -> FDS (or materialized bin) source; one entry per trajectory, never per sample
        self.linear_files: dict[str, str] = {}
        self.linear_bins: dict[str, str] = {}   # real-space (kx/ky inverted)
        self.linear_k_bins: dict[str, str] = {}  # semispectral, as GKW wrote it
        self.fid_to_traj: dict[int, str] = {}
        missing = []
        for fid, f_path in enumerate(self.files):
            traj = strip_format_tag(os.path.basename(f_path.rstrip("/")))
            self.fid_to_traj[fid] = traj
            if traj in self.linear_files or traj in self.linear_bins:
                continue
            meta = self.metadata[fid]
            for key, store in (("linear_bin", self.linear_bins),
                               ("linear_k_bin", self.linear_k_bins)):
                rel = meta.get(key)
                if rel is not None:
                    store[traj] = os.path.abspath(os.path.join(f_path, _as_str(rel)))
            src = meta.get("linear_fds")
            src = _as_str(src) if src is not None else resolve_linear_dir(f_path, self.linear_roots)
            if src is None:
                if traj not in self.linear_bins and traj not in self.linear_k_bins:
                    missing.append(traj)
                continue
            self.linear_files[traj] = src
        if missing and self.linear_required:
            raise RuntimeError(
                f"no linear (_Lin) pair for {len(missing)} trajectories, e.g. {missing[:5]}; "
                "run `preprocess --mode=linear` or pass linear_required=False"
            )
        self.linear_missing = missing

        self._linear_cache: dict[str, np.ndarray] = {}
        self.linear_profile = self._build_linear_profile()
        self.linear_channels = 2 * (2 if self.linear_separate_zf else 1)
        if self.linear_separate_zf and not self.linear_to_real:
            raise ValueError(
                "linear_separate_zf splits on the ky MEAN, which is a real-space construct; "
                "in the semispectral field the zonal component is the ky=0 mode"
            )
        self.linear_field_shape = (self.linear_channels, *self.resolution)
        if linear_preload:
            self._preload_linear(max(1, linear_workers))

    # loading

    def _read_linear(self, traj: str) -> np.ndarray:
        # each materialized flavour serves one space; fall back to parsing the raw FDS
        store = self.linear_bins if self.linear_to_real else self.linear_k_bins
        bin_path = store.get(traj)
        if bin_path is not None and os.path.exists(bin_path):
            shape = (2, *self.resolution)
            arr = np.asarray(self.backend.read_field(os.path.dirname(bin_path), bin_path, shape))
        elif traj in self.linear_files:
            arr = load_linear_field(
                self.linear_files[traj], self.resolution, to_real=self.linear_to_real,
            )
        else:
            arr = np.zeros((2, *self.resolution), dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if self.linear_separate_zf:
            arr = separate_zf_fn(arr, axis=0)
        return self._normalize_linear(arr).astype(self.linear_dtype, copy=False)

    _PROFILE_KEY = {"ky": "linear_rms_ky", "cky": "linear_rms_cky",
                    "kxky": "linear_rms_kxky", "ckxky": "linear_rms_ckxky"}

    def _build_linear_profile(self) -> Optional[np.ndarray]:
        """One profile for the whole split: RMS pooled over every trajectory.

        ``cky`` gives shape ``(2, nky)`` (re/im x ky), ``ckxky`` ``(2, nkx, nky)``.
        Pooling is ``sqrt(mean_i rms_i**2)`` (equal element counts per trajectory). A
        profile passed in via ``linear_profile`` wins, so the val split can inherit the
        training one instead of building a different normalization from its own few
        trajectories.
        """
        if self._linear_profile_in is not None:
            return np.asarray(self._linear_profile_in, dtype=np.float32)
        key = self._PROFILE_KEY.get(self.linear_normalize)
        if key is None:
            return None
        acc, n = None, 0
        for meta in self.metadata.values():
            if key not in meta:
                continue
            r = np.asarray(meta[key], dtype=np.float64)
            if self.linear_rescale_per_traj:
                r = r / (float(np.squeeze(meta.get("linear_rms", 1.0))) or 1.0)
            acc = np.square(r) if acc is None else acc + np.square(r)
            n += 1
        if acc is None:
            raise RuntimeError(
                f"linear_normalize={self.linear_normalize!r} needs '{key}' in the trajectory "
                "metadata — run `preprocess --mode=linear --stats`"
            )
        if n != len(self.metadata) and self.rank == 0:
            warnings.warn(f"linear profile built from {n}/{len(self.metadata)} trajectories")
        return np.sqrt(acc / n).astype(np.float32)

    def _normalize_linear(self, arr: np.ndarray) -> np.ndarray:
        if self.linear_normalize == "zscore":
            mean = float(np.mean(arr, dtype=np.float64))
            std = float(np.std(arr, dtype=np.float64))
            return (arr - mean) / max(std, 1e-30)
        if self.linear_normalize == "none":
            return arr
        if self.linear_normalize == "rms" or self.linear_rescale_per_traj:
            scale = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))
            arr = arr / max(scale, 1e-30)
        if self.linear_profile is not None:
            p = self.linear_profile
            # (2, nky) -> (2,1,1,1,1,nky); (2, nkx, nky) -> (2,1,1,1,nkx,nky); ky-only -> trailing
            if p.ndim == 2:
                p = p[:, None, None, None, None, :]
            elif p.ndim == 3:
                p = p[:, None, None, None, :, :]
            arr = arr / np.maximum(p, 1e-12)
        return arr

    def _preload_linear(self, workers: int) -> None:
        trajs = sorted(set(self.linear_files) | set(self.linear_bins))
        if self.linear_cache_size is not None:
            trajs = trajs[: self.linear_cache_size]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for traj, arr in zip(trajs, ex.map(self._read_linear, trajs)):
                self._linear_cache[traj] = arr

    def get_linear(self, fid: int) -> np.ndarray:
        """The (shared, already normalized) linear field for trajectory ``fid``."""
        traj = self.fid_to_traj[int(fid)]
        arr = self._linear_cache.get(traj)
        if arr is None:
            arr = self._read_linear(traj)
            cap = self.linear_cache_size
            if cap is not None and len(self._linear_cache) >= cap:
                self._linear_cache.pop(next(iter(self._linear_cache)))
            self._linear_cache[traj] = arr
        return arr

    # sample assembly

    def _build_sample(self, fid, t_idx, df, phi, flux, timestep, meta) -> CycloneSample:
        sample = super()._build_sample(fid, t_idx, df, phi, flux, timestep, meta)
        return replace(sample, linear=self.get_linear(fid))


__all__ = ["LinearCondCycloneDataset", "k_to_real", "load_linear_field"]
