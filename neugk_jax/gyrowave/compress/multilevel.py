"""Experimental — hierarchical multilevel wavelet tokenization.

SEPARATE from the main flat-index tokenization (fastops.frame_coords / the (h,l,s_wav,kx_wav,ky)
token coord). Nothing here is wired into the deployed frame, fit, or model.

Each db4 wavelet axis (s: 16->16 and x/kx: 85->87; both db4, level-3, periodization) is a
concatenation of sub-bands ``[cA_L | cD_L | ... | cD_1]``. The main path labels a coefficient by
its FLAT index into that concatenation. Here we instead label it by an explicit
``(level, offset-within-level)`` — L0 (coarse approximation) .. Lk (finest detail) — so tokens
become hierarchical and a model can attach a *learnable per-level embedding* (the "L0, L1, ... Lk
wavelets" that can be learned).

Key property preserved: the db4 frame itself is UNCHANGED — still a fixed, linear, deterministic
map — so the gauge / flux-as-quadratic-form guarantees are untouched. Only the coordinate
featurization of the two wavelet axes changes.

Pure numpy / pywt. No torch, no neugk.
"""
from __future__ import annotations

import numpy as np
import pywt

from neugk_jax.gyrowave.compress import fastops


def axis_levels(n_signal, wavelet="db4", level=3, mode="periodization"):
    """Per-coefficient ``(level, offset)`` for a 1-D multilevel db4 analysis of a length-``n_signal``
    signal — matching ``fastops._db4_axis_mats`` (same wavelet/level/mode).

    Returns ``(lvl, off, bands)``:
      * ``lvl[i]`` / ``off[i]`` — level index (0 = coarse approx, 1..level = detail coarse->fine)
        and within-level offset of frame coefficient ``i``.
      * ``bands`` — ``[(label, size), ...]`` in frame order.
    ``M = sum(size) = len(lvl)`` is the wavelet-axis length in the frame (16 for s, 87 for kx)."""
    coeffs = pywt.wavedec(np.zeros(n_signal), wavelet, mode=mode, level=level)  # [cA_L, cD_L, .., cD_1]
    sizes = [len(c) for c in coeffs]
    labels = [f"L0 approx (cA{level})"] + [f"L{i + 1} detail (cD{level - i})" for i in range(level)]
    lvl = np.concatenate([np.full(s, i, dtype=np.int16) for i, s in enumerate(sizes)])
    off = np.concatenate([np.arange(s, dtype=np.int16) for s in sizes])
    return lvl, off, list(zip(labels, sizes))


def hierarchical_frame_coords(s_signal=16, x_signal=85, level=3):
    """Re-label the frame's two wavelet axes with explicit ``(level, offset)``.

    Returns ``(coords, meta)`` where ``coords`` is ``(M_TOT, 7)`` int16 with columns
    ``(h, l, s_lvl, s_off, kx_lvl, kx_off, ky)`` in the SAME atom order as
    ``fastops.get_frame_coords()`` (a 1:1 re-featurization, no reordering). ``meta`` carries the
    per-axis band structure and the number of distinct levels."""
    frame = np.asarray(fastops.get_frame_coords())               # (M_TOT,5): h,l,s_wav,kx_wav,ky
    s_lvl, s_off, s_bands = axis_levels(s_signal, level=level)
    x_lvl, x_off, x_bands = axis_levels(x_signal, level=level)
    if len(s_lvl) != fastops.CSHAPE[2] or len(x_lvl) != fastops.CSHAPE[3]:
        raise ValueError(f"band sizes {len(s_lvl)},{len(x_lvl)} != frame axes "
                         f"{fastops.CSHAPE[2]},{fastops.CSHAPE[3]}")
    h, l, sw, xw, ky = frame.T
    coords = np.stack([h, l, s_lvl[sw], s_off[sw], x_lvl[xw], x_off[xw], ky], 1).astype(np.int16)
    return coords, {"s_bands": s_bands, "kx_bands": x_bands, "n_levels": level + 1,
                    "coord_names": ["h", "l", "s_lvl", "s_off", "kx_lvl", "kx_off", "ky"]}


def level_of(flat_idx, n_signal, level=3, wavelet="db4", mode="periodization"):
    """Convenience: map flat wavelet indices (into one axis) -> their level indices."""
    lvl, _, _ = axis_levels(n_signal, wavelet=wavelet, level=level, mode=mode)
    return lvl[np.asarray(flat_idx)]
