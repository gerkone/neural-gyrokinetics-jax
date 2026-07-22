"""Zonal-flow / spectral turbulence metrics for validation.

Numpy port of ``neugk/pinc/eval/metrics.py`` (the PINC compression eval) on
top of the gyaradax integrals adapter. The psnr/ml_eval and optical-flow
``temporal_epe`` parts are intentionally not ported.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    # ordinal ranks (== scipy average ranks when there are no ties; spectra are
    # continuous so ties don't occur), then Pearson on the ranks.
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _pearson(ra, rb)


def _wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    # 1D W1 for equal-length, uniform-weight samples == mean|sorted(u)-sorted(v)|;
    # matches scipy.stats.wasserstein_distance(u, v) on the (same-length) spectra.
    return float(np.abs(np.sort(u) - np.sort(v)).mean())


def _zonal_profiles(
    phi_spec: np.ndarray, geom: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """GKW diagnos_zfshear trio from the spectral potential (s, kx, ky).

    zfphi is the flux-surface average (ints weights) of the zonal (ky=0) mode;
    zfflow/zfshear are its first/second radial derivatives (i*kx in spectral x),
    the E x B zonal flow and its shear rate (Dannert & Jenko, PoP 2005).
    """
    ints = np.asarray(geom["ints"], dtype=np.float64).reshape(-1, 1)
    zon = (phi_spec[:, :, 0] * ints).sum(0)  # (kx,) complex
    kx = np.asarray(geom["kxrh"], dtype=np.float64)

    def prof(z):
        return np.fft.ifft(np.fft.ifftshift(z), norm="forward").real

    return {"zfphi": prof(zon), "zfflow": prof(1j * kx * zon), "zfshear": prof(-(kx**2) * zon)}


def diagnostics(
    phi_fft_: np.ndarray,
    eflux_field: np.ndarray,
    ds: float,
    aggregate: str = "mid",
) -> Dict[str, np.ndarray]:
    """Turbulence diagnostics from the potential FFT and the heat-flux field.

    Port of ``neugk/physics/diagnostics.py:diagnostics`` (minus the unused
    ``phi_zf`` profile) with the axis arithmetic kept verbatim — including
    the mid slice taken at index ``shape[-3] // 2``. The last three axes of
    ``phi_fft_`` are ``(nx, *, ny)``; ``kxspec`` sums the y axis, ``kyspec``
    sums the x (``*``) axis. ``aggregate`` selects how the remaining nx axis
    is collapsed: ``"mean"`` sums it, ``"mid"`` takes the central slice,
    ``"none"`` keeps it.
    """
    diag: Dict[str, np.ndarray] = {}
    nx = phi_fft_.shape[-3]
    power = phi_fft_.real ** 2 + phi_fft_.imag ** 2

    kxspec = power.sum(axis=-1) * ds  # reduce y -> (..., nx, mid)
    kyspec = power.sum(axis=-2) * ds  # reduce mid -> (..., nx, ny)

    def _agg(spec):  # collapse the nx axis (now at -2)
        if aggregate == "mean":
            return spec.sum(axis=-2)
        if aggregate == "mid":
            return np.take(spec, nx // 2, axis=-2)
        return spec

    diag["kxspec"] = _agg(kxspec)
    diag["kyspec"] = _agg(kyspec)

    # heat-flux spectrum: sum everything except the trailing wavenumber axis
    # (upstream sums dims (0,1,2,3) of the 5D torch flux field; the gyaradax
    # eflux field is already reduced to (kx, ky), so this is sum(axis=0))
    diag["qspec"] = (
        eflux_field.sum(axis=tuple(range(eflux_field.ndim - 1)))
        if eflux_field.ndim >= 2
        else eflux_field.sum()
    )
    return diag


def spectral_diagnostics(
    df_batch: np.ndarray, geom: Dict[str, np.ndarray], ds: float
) -> List[Dict[str, np.ndarray]]:
    """Turbulence spectra (kxspec/kyspec/qspec) + zonal profiles per snapshot.

    ``df_batch`` is the (already denormalised) spatial df ``(B, 4, vp, mu, s,
    x, y)``; ``geom`` is a single-trajectory geometry dict. Returns one dict
    per batch element (the upstream torch function is per-snapshot).
    """
    from neugk_jax.evaluate.integrals import gyaradax_spectral_fields
    phi_spec, eflux = gyaradax_spectral_fields(df_batch, geom)
    out: List[Dict[str, np.ndarray]] = []
    for b in range(phi_spec.shape[0]):
        d = diagnostics(phi_spec[b], eflux[b], ds=ds)
        d.update(_zonal_profiles(phi_spec[b], geom))
        out.append(d)
    return out


def time_averaged_spectral_metrics(
    pred_diags: List[Dict[str, np.ndarray]],
    gt_diags: List[Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """Pearson/Spearman/Wasserstein/L1 on the time-averaged ky and Q spectra."""
    out: Dict[str, float] = {}
    for key in ("kyspec", "qspec"):
        p = np.stack([np.asarray(d[key]) for d in pred_diags], 0).mean(0)
        g = np.stack([np.asarray(d[key]) for d in gt_diags], 0).mean(0)
        out[f"{key}_pc"] = float(_pearson(p, g))
        out[f"{key}_sc"] = float(_spearman(p, g))
        out[f"{key}_l1"] = float(np.abs(p - g).sum())
        out[f"{key}_rl2"] = float(np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-12))  # relative L2
        out[f"{key}_rl1"] = float(np.abs(p - g).sum() / (np.abs(g).sum() + 1e-12))  # relative L1
        pn, gn = p / (p.sum() + 1e-12), g / (g.sum() + 1e-12)
        out[f"{key}_wd"] = float(_wasserstein_1d(pn, gn))
    # zonal-flow fidelity (gkw diagnos_zfshear quantities): the profiles are signed
    # and time-varying, so score per snapshot (rel-L2) and average over time.
    for key in ("zfphi", "zfflow", "zfshear"):
        if key in pred_diags[0]:
            rl2 = [
                float(np.linalg.norm(p[key] - g[key]) / (np.linalg.norm(g[key]) + 1e-12))
                for p, g in zip(pred_diags, gt_diags)
            ]
            out[f"{key}_rl2"] = sum(rl2) / len(rl2)
    if "zfphi" in pred_diags[0]:
        er = [
            float((p["zfphi"] ** 2).sum() / ((g["zfphi"] ** 2).sum() + 1e-12))
            for p, g in zip(pred_diags, gt_diags)
        ]
        out["zf_energy_err"] = abs(sum(er) / len(er) - 1)  # |E_pred/E_gt - 1|
    return out


# direction of improvement, for table formatting downstream
DIRECTION = {
    "l1": "min",
    "mse": "min",
    "psnr": "max",
    "bpp": "min",
    "cr": "max",
    "phi_l1": "min",
    "phi_psnr": "max",
    "eflux_l1": "min",
    "endpoint": "min",
    "kyspec_pc": "max",
    "qspec_pc": "max",
    "kyspec_sc": "max",
    "qspec_sc": "max",
    "kyspec_l1": "min",
    "qspec_l1": "min",
    "kyspec_wd": "min",
    "qspec_wd": "min",
    "kyspec_rl2": "min",
    "qspec_rl2": "min",
    "kyspec_rl1": "min",
    "qspec_rl1": "min",
    "density_l1": "min",
    "momentum_l1": "min",
    "energy_l1": "min",
    "free_energy_err": "min",
    "zfphi_rl2": "min",
    "zfflow_rl2": "min",
    "zfshear_rl2": "min",
    "zf_energy_err": "min",
}


# --------------------------------------------------------------------------- #
# evaluator glue — shared between the AE and diffusion evaluators
# --------------------------------------------------------------------------- #
def accumulate_spectral_diagnostics(
    store: Dict[int, tuple],
    df_pred: np.ndarray,
    df_tgt: np.ndarray,
    file_idx: np.ndarray,
    val_ds: Any,
) -> bool:
    """Append per-snapshot pred/gt diagnostics to ``store``, grouped by trajectory.

    ``df_pred``/``df_tgt`` must already be denormalised. Returns ``False``
    (without touching ``store``) when the dataset metadata carries no ``ds``
    so the caller can warn once and stop asking.
    """
    file_idx = np.asarray(file_idx)
    for fid in np.unique(file_idx):
        ds_val = val_ds.get_ds(int(fid))
        if ds_val is None:
            return False
        idx = np.where(file_idx == fid)[0]
        # single-trajectory geometry (strip the batch axis get_batch_geometry adds)
        geom = {k: np.asarray(v)[0] for k, v in val_ds.get_batch_geometry(np.asarray([fid])).items()}
        p_list, g_list = store.setdefault(int(fid), ([], []))
        p_list.extend(spectral_diagnostics(df_pred[idx], geom, ds_val))
        g_list.extend(spectral_diagnostics(df_tgt[idx], geom, ds_val))
    return True


def merged_spectral_metrics(store: Dict[int, tuple]) -> Dict[str, float]:
    """Per-trajectory time-averaged spectral metrics, mean over trajectories."""
    per_traj = [
        time_averaged_spectral_metrics(p, g) for p, g in store.values() if p and g
    ]
    if not per_traj:
        return {}
    return {k: float(np.mean([m[k] for m in per_traj])) for k in per_traj[0]}
