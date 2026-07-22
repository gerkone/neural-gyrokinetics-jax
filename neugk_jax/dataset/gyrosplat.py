"""``GyrosplatDataset``: fitted Gaussian-splat latents for flow matching.

Consumes the numpy cache written by ``scripts/convert_gyrosplat_latents.py``
(the only torch-dependent step). Each sample is a normalized token bank
``(1598, 17)``: 1597 atom slots (see ``neugk_jax.gyrosplats.splat.pack`` for
the channel order) plus one global zf-stats token whose first 3 channels hold
the normalized per-snapshot normalization scalars — generated jointly with the
splat so sampling is standalone-generative.

Conventions mirror ``CycloneDataset``: conditioning keys sorted alphabetically,
``iteration_{a-b}`` trajectory ranges (plus ``file:<name>.json`` id lists),
host-side numpy samples in a frozen dataclass.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from neugk_jax.dataset.backend import NumpyBackend, read_bin, resolve_trajectories
from neugk_jax.gyrosplats.normalize import (
    TokenStats,
    WindowStats,
    denormalize_state,
    denormalize_tokens,
    normalize_state,
    normalize_tokens,
    normalize_zf_scalars,
)
from neugk_jax.gyrosplats.splat import N_CHANNELS, SplatParams
from neugk_jax.gyrosplats.window import (
    N_STATE_CH,
    WIN_AMP_CH,
    dct_matrix,
    extract_scaffold,
    pack_windows,
    scaffold_max_dev,
    unpack_windows,
)

# params.npy column order on disk (differs from the sorted conditioning order)
PARAMS_ORDER = ("itg", "dg", "q", "s_hat")

# token type ids: 0 = envelope, 1..7 = carrier bins 7..13, 8 = zf-stats token
STATS_TYPE_ID = 8
N_STAT_CHANNELS = 3


@dataclass(frozen=True)
class GyrosplatSample:
    """One snapshot: normalized token bank + conditioning + raw zf stats."""

    tokens: np.ndarray  # (1598, 17) normalized
    conditioning: np.ndarray  # (n_cond,)
    zf_stats: np.ndarray  # (4,) raw [zonal_mean, zonal_std, fluc_mean, fluc_std]
    avg_flux: np.ndarray
    timestep: np.ndarray
    file_index: np.ndarray
    timestep_index: np.ndarray


def collate(batch: Sequence[GyrosplatSample]) -> GyrosplatSample:
    def stack(key: str):
        return np.stack([getattr(s, key) for s in batch])

    return GyrosplatSample(**{k: stack(k) for k in GyrosplatSample.__dataclass_fields__})


class GyrosplatDataset:
    """Map-style dataset over converted gyrosplat trajectories.

    Parameters
    ----------
    cache_path
        Cache dir from ``convert_gyrosplat_latents.py`` (atoms/bins/channel_stats).
    path
        Original gyrosplat data dir — only used to resolve ``file:`` trajectory specs.
    geometry_path
        Preprocessed snapshot dir (``*_ifft_realpotens``) for eval-time geometry
        and ground-truth fields; optional for training.
    conditions
        Scalar drive parameters packed (sorted) into ``conditioning``.
    offset
        First fitted timestep (fixed by the fitting run; flux averaging uses it).
    """

    def __init__(
        self,
        *,
        cache_path: str,
        path: Optional[str] = None,
        geometry_path: Optional[str] = None,
        split: str = "train",
        trajectories: Optional[Any] = None,
        conditions: Sequence[str] = ("itg", "dg", "s_hat", "q"),
        offset: int = 80,
        stats_token: bool = False,
        ky_mode: str = "delta",
        asinh_channels: Sequence[int] = (),
        asinh_scale: float = 3.0,
        layout: str = "atoms",
        dct: bool = False,
        stats_trajectories: Optional[Any] = None,
        scaffold_n_snaps: int = 8,
        scaffold_tol: float = 0.05,
        rank: int = 0,
    ):
        assert split in ("train", "val")
        assert ky_mode in ("delta", "frozen")
        assert layout in ("atoms", "windows")
        self.cache_path = cache_path
        self.path = path
        self.geometry_path = geometry_path
        self.split = split
        # sort alphabetically to keep conditioning slot order consistent across runs
        self.conditions = sorted(conditions)
        self.offset = offset
        self.stats_token = stats_token
        self.ky_mode = ky_mode
        self.layout = layout
        self.dct = bool(dct)
        self.rank = rank

        self.bins = np.load(os.path.join(cache_path, "bins.npy"))
        self.n_atoms = int(self.bins.shape[0])
        if layout == "atoms":
            self.token_stats = TokenStats.load(
                os.path.join(cache_path, "channel_stats.npz"),
                asinh_channels=asinh_channels,
                asinh_scale=asinh_scale,
            )
            self.n_channels = N_CHANNELS
            self.n_tokens = self.n_atoms + int(stats_token)
        else:
            # windows layout stats come from the run's TRAINING trajectories
            self.token_stats = None
            self.n_channels = N_STATE_CH

        names = self._resolve(trajectories)
        self.files: list[str] = []
        self.atoms: dict[int, np.ndarray] = {}
        self.zfstats: dict[int, np.ndarray] = {}
        self.params: dict[int, np.ndarray] = {}
        self.flux: dict[int, np.ndarray] = {}
        self.meta: dict[int, dict] = {}
        for name in names:
            d = os.path.join(cache_path, name)
            if not os.path.exists(os.path.join(d, "atoms.npy")):
                if rank == 0:
                    warnings.warn(f"{name}: not in cache; excluding trajectory")
                continue
            fid = len(self.files)
            self.files.append(name)
            # mmap: ~20 MB/traj, read lazily per sample
            self.atoms[fid] = np.load(os.path.join(d, "atoms.npy"), mmap_mode="r")
            self.zfstats[fid] = np.load(os.path.join(d, "zfstats.npy"))
            self.params[fid] = np.load(os.path.join(d, "params.npy"))
            self.flux[fid] = np.load(os.path.join(d, "flux.npy"))
            with open(os.path.join(d, "meta.json")) as fh:
                self.meta[fid] = json.load(fh)
            # true timestep per latent (fits can be incomplete/gappy)
            steps = self.meta[fid].get("steps") or []
            self.timesteps = getattr(self, "timesteps", {})
            self.timesteps[fid] = (
                [row["t"] for row in steps]
                if steps
                else list(range(offset, offset + self.atoms[fid].shape[0]))
            )
        if not self.files:
            raise RuntimeError(f"no converted trajectories found under {cache_path}")

        # flat index over (traj, timestep)
        self.flat_index_to_file_and_tstep: dict[int, tuple[int, int]] = {}
        flat = 0
        for fid in range(len(self.files)):
            for t_idx in range(self.atoms[fid].shape[0]):
                self.flat_index_to_file_and_tstep[flat] = (fid, t_idx)
                flat += 1
        self.length = flat

        # per-slot token type ids incl. the stats token (bins 7..13 -> 1..7)
        type_ids = np.where(self.bins == 0, 0, self.bins - 6).astype(np.int32)
        self.type_ids = np.concatenate([type_ids, [STATS_TYPE_ID]])

        if layout == "windows":
            self._setup_windows(stats_trajectories, scaffold_n_snaps, scaffold_tol)

    def _setup_windows(self, stats_trajectories, n_snaps, tol) -> None:
        """Extract the frozen scaffold + fit channel-wise stats from training trajs."""
        import jax.numpy as jnp
        # scaffold + stats come from the run's training trajectories (passed in for
        # val); fall back to this dataset's own trajectories
        names = self._resolve(stats_trajectories) if stats_trajectories is not None else self.files
        names = [n for n in names if os.path.exists(os.path.join(self.cache_path, n, "atoms.npy"))]
        first = np.load(os.path.join(self.cache_path, names[0], "atoms.npy"), mmap_mode="r")
        snaps = np.asarray(first[: min(n_snaps, first.shape[0])]).astype(np.float32)
        dev = scaffold_max_dev(snaps, self.bins)
        if dev > tol:
            raise RuntimeError(f"carrier scaffold not frozen: max dev {dev:.4f} > {tol}")
        if self.rank == 0:
            print(f"[gyrosplat/windows] scaffold max dev {dev:.4f} (tol {tol})")
        self.scaffold = extract_scaffold(snaps, self.bins)
        self.n_env = int(self.scaffold.env_idx.shape[0])
        self.n_window = int(self.scaffold.grp_idx.shape[0])
        self.n_tokens = self.n_env + self.n_window
        self.dct_mat = dct_matrix(int(self.scaffold.grp_idx.shape[1])) if self.dct else None

        # streaming per-channel mean/std over env rows and window rows separately
        d = jnp.asarray(self.dct_mat) if self.dct else None
        acc = {k: np.zeros(N_STATE_CH) for k in ("env_s", "env_ss", "win_s", "win_ss")}
        n_env_rows = n_win_rows = 0
        for name in names:
            atoms = np.asarray(np.load(os.path.join(self.cache_path, name, "atoms.npy")))
            states = np.stack([pack_windows(atoms[t], self.scaffold) for t in range(atoms.shape[0])])
            env = states[:, : self.n_env].reshape(-1, N_STATE_CH)
            win = states[:, self.n_env :].reshape(-1, N_STATE_CH)
            if self.dct:
                from neugk_jax.gyrosplats.window import window_dct

                win = np.asarray(win).copy()
                win[:, :WIN_AMP_CH] = np.asarray(window_dct(jnp.asarray(win[:, :WIN_AMP_CH]), d))
            acc["env_s"] += env.sum(0)
            acc["env_ss"] += (env.astype(np.float64) ** 2).sum(0)
            acc["win_s"] += win.sum(0)
            acc["win_ss"] += (win.astype(np.float64) ** 2).sum(0)
            n_env_rows += env.shape[0]
            n_win_rows += win.shape[0]

        def _mean_std(s, ss, n):
            m = s / n
            v = np.maximum(ss / n - m**2, 0.0)
            return m.astype(np.float32), np.maximum(np.sqrt(v), 1e-6).astype(np.float32)

        env_mean, env_std = _mean_std(acc["env_s"], acc["env_ss"], n_env_rows)
        win_mean, win_std = _mean_std(acc["win_s"], acc["win_ss"], n_win_rows)
        # envelope mu channels: fixed affine 2x-1 (NOT z-scored)
        env_mean[:5], env_std[:5] = 0.5, 0.5
        # window pad channels are inert (masked): identity normalization
        win_mean[WIN_AMP_CH:], win_std[WIN_AMP_CH:] = 0.0, 1.0
        self.window_stats = WindowStats(
            env_mean=env_mean, env_std=env_std, win_mean=win_mean, win_std=win_std,
            n_env=self.n_env, dct=self.dct, dct_mat=self.dct_mat,
        )

    def _resolve(self, trajectories) -> list[str]:
        if trajectories is None:
            with open(os.path.join(self.cache_path, "converted.json")) as fh:
                return json.load(fh)
        if isinstance(trajectories, str) and trajectories.startswith("file:"):
            root = self.path or self.cache_path
            with open(os.path.join(root, trajectories[5:])) as fh:
                return [f"iteration_{i}" for i in json.load(fh)]
        return [os.path.basename(p) for p in resolve_trajectories("", trajectories)]

    def __len__(self) -> int:
        return self.length

    @property
    def loss_mask(self) -> np.ndarray:
        """(n_tokens, C) — masks dead channels (stats token / Δky / window pad)."""
        if self.layout == "windows":
            mask = np.ones((self.n_tokens, N_STATE_CH), dtype=np.float32)
            mask[self.n_env :, WIN_AMP_CH:] = 0.0  # window pad channels
            return mask
        mask = np.ones((self.n_tokens, N_CHANNELS), dtype=np.float32)
        if self.stats_token:
            mask[-1, :] = 0.0
            mask[-1, :N_STAT_CHANNELS] = 1.0
        atom_rows = slice(None, -1) if self.stats_token else slice(None)
        if self.ky_mode == "frozen":
            mask[atom_rows, 16] = 0.0
        return mask

    def traj_mean_zf_stats(self, fid: int) -> np.ndarray:
        """time-mean zf stats of a trajectory (denorm substitute without a stats token)."""
        return self.zfstats[fid].mean(axis=0).astype(np.float32)

    def _tokens(self, fid: int, t_idx: int) -> np.ndarray:
        import jax.numpy as jnp

        if self.layout == "windows":
            state = pack_windows(np.asarray(self.atoms[fid][t_idx]), self.scaffold)
            return np.asarray(normalize_state(jnp.asarray(state), self.window_stats))

        atoms = jnp.asarray(np.asarray(self.atoms[fid][t_idx]))
        tok = np.array(normalize_tokens(atoms, jnp.asarray(self.bins), self.token_stats))
        if self.ky_mode == "frozen":
            tok[:, 16] = 0.0
        if not self.stats_token:
            return tok
        stats_tok = np.zeros((1, N_CHANNELS), dtype=np.float32)
        stats_tok[0, :N_STAT_CHANNELS] = normalize_zf_scalars(
            self.zfstats[fid][t_idx], self.token_stats
        )
        return np.concatenate([tok, stats_tok], axis=0)

    def __getitem__(self, index: int) -> GyrosplatSample:
        fid, t_idx = self.flat_index_to_file_and_tstep[index]
        p = self.params[fid]
        by_name = dict(zip(PARAMS_ORDER, p))
        cond = np.array([by_name[c] for c in self.conditions], dtype=np.float32)
        flux = self.flux[fid]
        return GyrosplatSample(
            tokens=self._tokens(fid, t_idx),
            conditioning=cond,
            zf_stats=self.zfstats[fid][t_idx].astype(np.float32),
            avg_flux=np.float32(np.mean(flux[self.offset :])),
            timestep=np.int32(self.timesteps[fid][t_idx]),
            file_index=np.int32(fid),
            timestep_index=np.int32(t_idx),
        )

    def denormalize_atom_tokens(self, tokens: Any) -> SplatParams:
        """Normalized atom tokens (1597, 17) -> physical SplatParams."""
        import jax.numpy as jnp

        return denormalize_tokens(tokens, jnp.asarray(self.bins), self.token_stats)

    def decode_state(self, state_n: Any) -> SplatParams:
        """Normalized (871, 16) v7 state -> physical (1597, 17) SplatParams."""
        import jax.numpy as jnp

        raw = denormalize_state(jnp.asarray(state_n), self.window_stats)
        return unpack_windows(raw, self.scaffold)

    # ---------------------------------------------------------------- eval helpers

    def _snapshot_dir(self, fid: int) -> str:
        assert self.geometry_path is not None, "geometry_path not set"
        return os.path.join(self.geometry_path, f"{self.files[fid]}_ifft_realpotens")

    def get_metadata(self, fid: int) -> dict:
        return NumpyBackend().read_metadata(self._snapshot_dir(fid), ("df",))

    def get_batch_geometry(self, file_indices: Sequence[int]) -> dict[int, dict]:
        return {int(f): self.get_metadata(int(f))["geometry"] for f in set(map(int, file_indices))}

    def get_gt_field(self, fid: int, t_idx: int) -> np.ndarray:
        """Raw (2, v∥, μ, s, x, y) snapshot at the fitted timestep index."""
        meta = self.get_metadata(fid)
        shape = tuple(int(v) for v in meta["resolution"])
        t = self.timesteps[fid][t_idx]
        path = os.path.join(self._snapshot_dir(fid), "data", f"timestep_{t:05d}.bin")
        return read_bin(path, (2, *shape))
