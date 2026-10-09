"""Compression benchmark metrics of a reconstructed df sequence against its ground truth.

Per snapshot (:func:`snapshot_fields` + :func:`snapshot_metrics`): df / potential MSE, L1 and PSNR,
the heat-flux error ``|sum Q_pred - sum Q_gt|`` and the velocity-moment errors. Per trajectory:
the optical-flow end-point error of the sequence (:func:`temporal_epe`) and the errors of the
time-averaged ky / heat-flux spectra and of the zonal-flow profiles
(:func:`time_averaged_spectral_metrics`). The spectra follow the GKW convention, summed over the
field line.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.evaluate.fourier import df_to_spec, spec_to_phi
from neugk_jax.evaluate.integrals import _pev_fluxes, _solve_fields

EPS = 1e-8
# spectral bins below this fraction of the ground-truth peak count as equal in the log distance
LSD_FLOOR = 1e-3


@jax.jit
def snapshot_fields(geom_t: dict, df: jnp.ndarray, ds) -> dict:
    """Field solve of one spatial df ``(2, vp, mu, s, x, y)``.

    Returns the real potential ``phi`` ``(x, s, y)``, the summed heat flux ``eflux``, the spectral
    potential ``phi_spec`` ``(s, kx, ky)`` and the diagnostics ``kxspec``, ``kyspec``, ``qspec``
    and ``phi_zf``.
    """
    spec = df_to_spec(df)
    phi_s, apar_s, bpar_s = _solve_fields(geom_t, spec)
    eflux = _pev_fluxes(geom_t, spec, phi_s, apar_s, bpar_s, axis=())[1]
    power = jnp.real(phi_s) ** 2 + jnp.imag(phi_s) ** 2
    zf = jnp.zeros_like(phi_s).at[..., 0].set(phi_s[..., 0])
    ns, _, ny = phi_s.shape
    return {
        "phi": spec_to_phi(jnp.transpose(phi_s, (1, 0, 2))),
        "eflux": jnp.sum(eflux),
        "phi_spec": phi_s,
        "kxspec": jnp.sum(jnp.sum(power, axis=-1) * ds, axis=0),
        "kyspec": jnp.sum(jnp.sum(power, axis=-2) * ds, axis=0),
        "qspec": jnp.sum(eflux, axis=(0, 1, 2, 3)),
        "phi_zf": jnp.fft.irfftn(
            jnp.fft.fftshift(zf, axes=0), axes=(0, 2), norm="forward", s=(ns, ny)
        ),
    }


@jax.jit
def snapshot_metrics(pred: jnp.ndarray, gt: jnp.ndarray, p: dict, g: dict) -> dict:
    """Reconstruction and integral metrics of one snapshot from its :func:`snapshot_fields`."""
    mse = jnp.mean((pred - gt) ** 2)
    phi_mse = jnp.mean((p["phi"] - g["phi"]) ** 2)
    return {
        "mse": mse,
        "l1": jnp.mean(jnp.abs(pred - gt)),
        "psnr": 10 * jnp.log10(jnp.max(gt) ** 2 / mse),
        "phi_mse": phi_mse,
        "phi_l1": jnp.mean(jnp.abs(p["phi"] - g["phi"])),
        "phi_psnr": 10 * jnp.log10(jnp.max(g["phi"]) ** 2 / phi_mse),
        "eflux_l1": jnp.abs(p["eflux"] - g["eflux"]),
    }


@jax.jit
def _velocity_moments(pred, gt, w: dict):
    pred, gt = pred.astype(jnp.float64), gt.astype(jnp.float64)
    dv = w["intvp"] * w["intmu"] * w["bn"]
    kernels = {
        "density_l1": 1.0,
        "momentum_l1": w["vpgr"],
        "energy_l1": w["vpgr"] ** 2 + 2.0 * w["mugr"] * w["bn"],
    }
    out = {}
    for k, kern in kernels.items():
        mp = jnp.sum(kern * pred * dv, axis=(1, 2))
        mg = jnp.sum(kern * gt * dv, axis=(1, 2))
        out[k] = jnp.sum(jnp.abs(mp - mg)) / (jnp.sum(jnp.abs(mg)) + EPS)
    fe_p, fe_g = jnp.sum(pred**2), jnp.sum(gt**2)
    out["free_energy_err"] = jnp.abs(fe_p - fe_g) / (fe_g + EPS)
    return out


def moment_weights(geometry: dict) -> dict:
    """Velocity weights of the raw ``geometry`` laid out against a ``(2, vp, mu, s, x, y)`` df."""
    shape = {"intvp": 1, "vpgr": 1, "intmu": 2, "mugr": 2, "bn": 3}
    out = {}
    for k, ax in shape.items():
        v = np.asarray(geometry[k], np.float64).reshape(-1)
        s = [1] * 6
        s[ax] = v.size
        out[k] = v.reshape(s)
    return out


def velocity_moment_errors(pred, gt, weights: dict) -> dict[str, float]:
    """Relative L1 of the density, momentum and energy moments and the free-energy error, in float64."""
    with jax.enable_x64(True):
        w = {k: jnp.asarray(v, jnp.float64) for k, v in weights.items()}
        out = _velocity_moments(jnp.asarray(pred), jnp.asarray(gt), w)
        return {k: float(v) for k, v in out.items()}


def zonal_profiles(phi_spec: np.ndarray, geometry: dict) -> dict[str, np.ndarray]:
    """GKW ``zfshear`` profiles: flux-surface averaged zonal potential, its first and second radial derivative."""
    ints = np.asarray(geometry["ints"], np.float64).reshape(-1, 1)
    kx = np.asarray(geometry["kxrh"], np.float64).reshape(-1)
    zon = (np.asarray(phi_spec)[:, :, 0] * ints).sum(0)

    def prof(z):
        return np.fft.ifft(np.fft.ifftshift(z), norm="forward").real

    return {"zfphi": prof(zon), "zfflow": prof(1j * kx * zon), "zfshear": prof(-(kx**2) * zon)}


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)


def _ranks(a):
    return np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(a.dtype)


def log_spectral_distance(p, g, floor: float = LSD_FLOOR) -> float:
    """RMS of ``log10(p / g)`` over the bins, both clipped at ``floor`` times the peak of ``g``."""
    eps = floor * float(np.max(g))
    return float(
        np.sqrt(np.mean((np.log10(np.maximum(p, eps)) - np.log10(np.maximum(g, eps))) ** 2))
    )


def wasserstein1(p, g) -> float:
    """1D Wasserstein-1 distance of the normalized non-negative spectra, in units of the grid length."""
    p, g = np.maximum(p, 0), np.maximum(g, 0)
    cp, cg = np.cumsum(p / (p.sum() + 1e-30)), np.cumsum(g / (g.sum() + 1e-30))
    return float(np.abs(cp - cg).sum() / len(g))


def time_averaged_spectral_metrics(pred_diags: Sequence[dict], gt_diags: Sequence[dict]) -> dict:
    """Errors of the time-averaged ky and heat-flux spectra and of the zonal-flow profiles.

    Spectra: Pearson, Spearman, L1, rel-L1 and rel-L2 over every ky, the mean gap of the sorted
    normalized values (``_wd``), and over ky > 0 (the zonal mode excluded) the log-spectral distance
    (``_lsd``) and the Wasserstein-1 distance along ky (``_w1``). The zonal-flow profiles are scored
    per snapshot (rel-L2, energy ratio) and averaged over time.
    """
    out: dict[str, float] = {}
    for key in ("kyspec", "qspec"):
        p = np.stack([d[key] for d in pred_diags]).astype(np.float32).mean(0)
        g = np.stack([d[key] for d in gt_diags]).astype(np.float32).mean(0)
        out[f"{key}_pc"] = float(_pearson(p, g))
        out[f"{key}_sc"] = float(_pearson(_ranks(p), _ranks(g)))
        out[f"{key}_l1"] = float(np.abs(p - g).sum())
        out[f"{key}_rl2"] = float(np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-12))
        out[f"{key}_rl1"] = float(np.abs(p - g).sum() / (np.abs(g).sum() + 1e-12))
        pn, gn = p / (p.sum() + 1e-12), g / (g.sum() + 1e-12)
        out[f"{key}_wd"] = float(np.abs(np.sort(pn) - np.sort(gn)).mean())
        pk, gk = p.reshape(-1)[1:].astype(np.float64), g.reshape(-1)[1:].astype(np.float64)
        out[f"{key}_lsd"] = log_spectral_distance(pk, gk)
        out[f"{key}_w1"] = wasserstein1(pk, gk)
    for key in ("zfphi", "zfflow", "zfshear"):
        rl2 = [
            np.linalg.norm(p[key] - g[key]) / (np.linalg.norm(g[key]) + 1e-12)
            for p, g in zip(pred_diags, gt_diags)
        ]
        out[f"{key}_rl2"] = float(sum(rl2) / len(rl2))
    er = [
        (p["zfphi"] ** 2).sum() / ((g["zfphi"] ** 2).sum() + 1e-12)
        for p, g in zip(pred_diags, gt_diags)
    ]
    out["zf_energy_err"] = float(abs(sum(er) / len(er) - 1))
    return out


def _gradient(u, axis: int):
    # central differences inside, one-sided at the edges (numpy edge_order=1, unit spacing)
    n = u.shape[axis]
    take = lambda a, b: jax.lax.slice_in_dim(u, a, b, axis=axis)  # noqa: E731
    inner = (take(2, n) - take(0, n - 2)) / 2.0
    first, last = take(1, 2) - take(0, 1), take(n - 1, n) - take(n - 2, n - 1)
    return jnp.concatenate([first, inner, last], axis=axis)


def _shift(u, axis: int, off: int):
    # zero-filled neighbour shift: out[i] = u[i - off]
    n = u.shape[axis]
    pad = [(0, 0)] * u.ndim
    if off > 0:
        pad[axis] = (1, 0)
        return jnp.pad(jax.lax.slice_in_dim(u, 0, n - 1, axis=axis), pad)
    pad[axis] = (0, 1)
    return jnp.pad(jax.lax.slice_in_dim(u, 1, n, axis=axis), pad)


def optical_flow_5d(x, alpha: float = 1.0, n_iters: int = 50):
    """Iterative Horn-Schunck optical flow of a ``(c, t, vp, mu, s, x, y)`` sequence, ``(5, t - 1, ...)``.

    Channel-mean intensity, central-difference spatial gradients and the zero-padded star-stencil
    average over the five spatial axes.
    """
    inten = jnp.mean(x, axis=0)
    x1, x2 = inten[:-1], inten[1:]
    xt = x2 - x1
    grads = jnp.stack([_gradient(0.5 * (x1 + x2), a) for a in range(1, 6)])
    denom = alpha**2 + jnp.sum(grads**2, axis=0)

    def step(_, v):
        avg = sum(_shift(v, a, 1) + _shift(v, a, -1) for a in range(2, 7)) * (1.0 / 10)
        return avg - grads * (jnp.sum(grads * avg, axis=0) + xt) / denom

    return jax.lax.fori_loop(0, n_iters, step, jnp.zeros_like(grads))


@jax.jit
def _epe(gt, pred):
    d = optical_flow_5d(gt) - optical_flow_5d(pred)
    return jnp.mean(jnp.sqrt(jnp.sum(d**2, axis=0)))


def temporal_epe(gt_dfs: Sequence, pred_dfs: Sequence) -> float:
    """End-point error of the optical flow of the predicted against the ground-truth sequence."""
    if len(gt_dfs) < 2:
        return float("nan")
    g = jnp.stack([jnp.asarray(d, jnp.float32) for d in gt_dfs], axis=1)
    p = jnp.stack([jnp.asarray(d, jnp.float32) for d in pred_dfs], axis=1)
    return float(_epe(g, p))
