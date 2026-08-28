"""Pairing and loading of linear-run (``_Lin``) conditioning fields.

A linear GKW run writes a single eigenmode snapshot, ``FDS``, in the same float64
Fortran-order layout as the nonlinear K-files: ``(2, nvpar, nmu, ns, nkx, nky)`` =
(re/im, vpar, mu, s, kx, ky). One such field characterizes a trajectory's operating
point, so it serves as conditioning for every timestep of that trajectory.

Shared by ``preprocess --mode=linear`` (which records the pairing in the trajectory
metadata) and ``LinearCondCycloneDataset`` (which serves the fields).
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

_FORMAT_TAGS = (
    "_ifft_separate_zf_",
    "_ifft_realpotens",
    "_ifft",
)

LINEAR_META_KEYS = ("linear_fds", "linear_bin", "linear_shape", "linear_space")
# the rebuttal ``extra_*`` runs keep the eigenmode in ``_Lin_gkw`` (their ``_Lin`` is empty)
_LINEAR_SUFFIXES = ("_Lin", "_Lin_gkw")


def strip_format_tag(name: str) -> str:
    """``iteration_3_ifft_realpotens`` → ``iteration_3``."""
    for tag in _FORMAT_TAGS:
        i = name.find(tag)
        if i >= 0:
            return name[:i]
    return name


def default_linear_roots(raw_root: str) -> list[str]:
    # searched in this order for a ``<traj>_Lin`` sibling
    return [raw_root, os.path.join(raw_root, "ood"), os.path.join(raw_root, "neurips26_rebuttal")]


def resolve_linear_dir(traj_name: str, roots: Sequence[str]) -> str | None:
    """Locate the ``_Lin`` raw dir paired with a preprocessed trajectory dir.

    Tries ``<root>/<base>_Lin`` and — for names that flattened a raw subdir into a
    prefix (``ood_iteration_3`` ← ``ood/iteration_3``) — ``<root>/<prefix>/<rest>_Lin``.
    """
    base = strip_format_tag(os.path.basename(traj_name.rstrip("/")))
    cands = []
    for suffix in _LINEAR_SUFFIXES:
        for root in roots:
            cands.append(os.path.join(root, f"{base}{suffix}"))
            if "_" in base:
                prefix, rest = base.split("_", 1)
                cands.append(os.path.join(root, prefix, f"{rest}{suffix}"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "FDS")):
            return c
    return None


def k_to_real(k: np.ndarray) -> np.ndarray:
    # numpy twin of fastops.K_to_real / torch preprocess.do_ifft; kx is zero-centered
    kc = (k[0] + 1j * k[1]).astype(np.complex64)
    kc = np.fft.fftshift(kc, axes=(-2,))
    f = np.fft.ifftn(kc, axes=(-2, -1), norm="forward")
    return np.stack([f.real, f.imag]).astype(np.float32)


def load_linear_field(
    fds_path: str, resolution: Sequence[int], *, to_real: bool = True,
) -> np.ndarray:
    """Read a linear-run ``FDS`` snapshot → ``(2, *resolution)`` fp32.

    ``to_real=True`` applies the same kx/ky inverse transform the nonlinear
    trajectories went through, so the field is layout-identical to a ``df`` sample.
    """
    if os.path.isdir(fds_path):
        fds_path = os.path.join(fds_path, "FDS")
    ff = np.fromfile(fds_path, dtype=np.float64)
    expected = 2 * int(np.prod(resolution))
    if ff.size != expected:
        raise IOError(f"{fds_path}: expected {expected} elements, got {ff.size}")
    k = np.reshape(ff, (2, *resolution), order="F").astype(np.float32)
    return k_to_real(k) if to_real else np.ascontiguousarray(k)
