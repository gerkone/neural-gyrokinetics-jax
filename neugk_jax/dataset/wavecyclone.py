"""``WaveCycloneDataset`` — pooled multi-trajectory wavelet-token DATA REPRESENTATION.

Companion to :class:`neugk_jax.dataset.cyclone.CycloneDataset`, but for the gyrowave
flow-matching workflow. It serves whitened wavelet tokens — the *compressed data
representation* of each trajectory, produced once by ``gyrowave.compress.process_trajectory``
(a lossy encode of the dataset, NOT a recomputable cache), analogous to the diffusion
project's precomputed AE latents.

Lists trajectories from a config, resolves each trajectory's token file + physical params,
pools every snapshot across trajectories, and applies GLOBAL value whitening so the FM N(0,I)
prior is consistent across trajectories. Each pooled snapshot row carries ITS trajectory's
params (per-sample conditioning). Single-trajectory (``tokens`` + ``params``) is the 1-traj
special case and reproduces the pre-multi-traj behavior exactly.

Pure numpy (no torch, no jax) — the runner wraps ``.X/.C/.P`` in ``jnp.asarray``.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

# default per-trajectory params root (matches configs/dataset/gyrowave.yaml single-trajectory path)
DEFAULT_PARAMS_BASE = "/local00/bioinf/galletti/gyrosplats_deform"


def _read_tokens(path):
    """One trajectory's token file -> RAW values (S,N,2), int coords (S,N,5), cshape (5,) float,
    S, N. No whitening — pooled/global whitening is the caller's job so the FM prior stays
    consistent across trajectories."""
    z = np.load(path)
    off = z["tok_offsets"].astype(np.int64)
    S = int(len(z["timesteps"]))
    n = int(off[1] - off[0])
    assert np.all(np.diff(off) == n), "runner v1 assumes fixed token count per snapshot"
    vals = np.stack([z["tok_re"], z["tok_im"]], -1).reshape(S, n, 2).astype(np.float32)
    crd_int = z["tok_coord"].reshape(S, n, 5)                        # keep int coords for decode / S*
    cshape = np.array([int(x) for x in z["cshape"]], np.float32)
    return vals, crd_int, cshape, S, n


def pool_tokens(token_files, params_list):
    """Pool snapshots across trajectories -> X (S_tot,N,2) GLOBALLY-whitened values, C
    (S_tot,N,5) normalized coords, P (S_tot,n_cond) per-sample params, nrm
    (mu/sd/cshape + pooled int coords), S_tot, N.

    Value mu/sd are over the POOLED values (consistent N(0,I) FM prior). Coord norm uses the
    shared cshape — N (token count) and cshape are asserted identical across trajectories."""
    assert len(token_files) == len(params_list) and len(token_files) > 0, "need >=1 trajectory"
    vals_l, crd_l, coord_int_l, P_l = [], [], [], []
    N0, cshape0 = None, None
    for path, params in zip(token_files, params_list):
        vals, crd_int, cshape, S, n = _read_tokens(path)
        if N0 is None:
            N0, cshape0 = n, cshape
        else:
            assert n == N0, f"token count mismatch: {n} != {N0} for {path}"
            assert np.array_equal(cshape, cshape0), f"cshape mismatch for {path}"
        vals_l.append(vals)
        crd_l.append(crd_int.astype(np.float32) / cshape[None, None, :])
        coord_int_l.append(crd_int)
        if params is not None:
            p = np.asarray(params, np.float32).reshape(-1)
            P_l.append(np.broadcast_to(p, (S, p.shape[0])).copy())   # per-sample conditioning
    X_raw = np.concatenate(vals_l, 0)                                # (S_tot, N, 2)
    C = np.concatenate(crd_l, 0)                                     # (S_tot, N, 5)
    coord_int = np.concatenate(coord_int_l, 0)                       # (S_tot, N, 5)
    P = np.concatenate(P_l, 0) if P_l else None                      # (S_tot, n_cond)
    mu = X_raw.reshape(-1, 2).mean(0)
    sd = X_raw.reshape(-1, 2).std(0).clip(1e-6)
    X = (X_raw - mu) / sd                                            # global whiten (FM prior N(0,I))
    nrm = {"mu": mu, "sd": sd, "cshape": cshape0.astype(int), "coord_int": coord_int}
    return X, C, P, nrm, int(X.shape[0]), N0


def load_tokens(path):
    """Single-trajectory load (backward compat): whiten values over THIS trajectory's snapshots.
    Returns the pre-multi-traj 5-tuple (vals_n, crd_n, nrm, S, N)."""
    X, C, _, nrm, S, N = pool_tokens([path], [None])
    return X, C, nrm, S, N


class WaveCycloneDataset:
    """Pooled multi-trajectory wavelet-token representation (see module docstring).

    Attributes
    ----------
    X : (S_tot, N, 2) globally-whitened token values
    C : (S_tot, N, 5) coords normalized by cshape (~[0,1] per axis)
    P : (S_tot, n_cond) per-sample params — each row is its trajectory's params
    nrm : {mu, sd, cshape, coord_int (S_tot,N,5)} for de-whitening / decode / fixed support
    S, N : pooled snapshot count and per-snapshot token count
    token_files, params_paths, params_list, is_multi, rep_params
    """

    def __init__(
        self,
        *,
        tokens: Optional[str] = None,
        params=None,
        token_files: Optional[Sequence[str]] = None,
        tokens_dir: Optional[str] = None,
        trajectories: Optional[Sequence[int]] = None,
        version: str = "real",
        params_base: str = DEFAULT_PARAMS_BASE,
        sample_trajectory: int = 0,
    ):
        self.token_files, self.params_paths, self.is_multi = self._resolve(
            tokens, params, token_files, tokens_dir, trajectories, version, params_base)
        self.params_list = [np.load(p).astype(np.float32).reshape(-1) for p in self.params_paths]
        self.X, self.C, self.P, self.nrm, self.S, self.N = pool_tokens(self.token_files, self.params_list)
        self.n_cond = int(self.P.shape[1])
        # representative params: sampling / single-traj compat (multi: pick sample_trajectory)
        rep = int(sample_trajectory) if self.is_multi else 0
        self.rep_params = self.params_list[rep]

    @staticmethod
    def _resolve(tokens, params, token_files, tokens_dir, trajectories, version, params_base):
        """-> (token file paths, params paths, is_multi). Single-traj is ``tokens``+``params``
        (1-elem lists); multi-traj is ``tokens_dir``+``trajectories``+``version`` (files
        ``{tokens_dir}/tokens_iteration_{it}_{version}.npz``, params
        ``{params_base}/iteration_{it}/params.npy``) or an explicit ``token_files`` list."""
        if token_files is None and tokens_dir is not None and trajectories is not None:
            token_files = [os.path.join(str(tokens_dir), f"tokens_iteration_{int(it)}_{version}.npz")
                           for it in trajectories]
        if token_files is not None:                                  # multi-trajectory
            token_files = [str(c) for c in token_files]
            if trajectories is not None:
                base = str(params_base)
                params = [os.path.join(base, f"iteration_{int(it)}", "params.npy")
                          for it in trajectories]
            else:                                                    # explicit files + params list
                assert params is not None, \
                    "explicit `token_files` needs a matching `params` list or `trajectories`"
                params = [str(p) for p in params]
            assert len(params) == len(token_files), "params/token_files length mismatch"
            return token_files, params, True
        return [str(tokens)], [str(params)], False                   # single-traj (backward compat)

    @classmethod
    def from_config(cls, ds):
        """Build from a ``cfg.dataset`` (OmegaConf DictConfig or plain dict) using the
        ``tokens``/``token_files``/``tokens_dir`` keys."""
        g = ds.get
        return cls(
            tokens=g("tokens", None),
            params=g("params", None),
            token_files=g("token_files", None),
            tokens_dir=g("tokens_dir", None),
            trajectories=g("trajectories", None),
            version=g("version", "real"),
            params_base=g("params_base", DEFAULT_PARAMS_BASE),
            sample_trajectory=g("sample_trajectory", 0),
        )
