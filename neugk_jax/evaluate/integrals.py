"""Flux integrals and spectral fields of a spatial df.

``precompute_geometry`` + ``flux_integral`` are the jittable single-sample field solve and
fluxes used in training and evaluation. ``gyaradax_spectral_fields`` keeps the per-mode
potential and heat flux for the spectral metrics via the ``gyaradax`` package (electrostatic
only); it runs in float64 inside a local ``jax.enable_x64`` context and undoes gyaradax's
global x64 switch on import.
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

REQUIRED_GEOMETRY = ("krho", "kxrh", "ints", "intmu", "intvp", "vpgr", "mugr", "bn",
                     "efun", "rfun", "bt_frac", "little_g")


def require_integrals(geometry: Optional[dict]) -> None:
    import importlib.util
    if importlib.util.find_spec("gyaradax") is None:
        raise ImportError("eval_integrals needs gyaradax (pip install -e '.[gyro]')")
    require_geometry(geometry)


def require_geometry(geometry: Optional[dict]) -> None:
    """Raise unless ``geometry`` carries every field :func:`precompute_geometry` needs."""
    if geometry is None:
        raise ValueError("eval_integrals needs trajectory geometry in the metadata")
    missing = [k for k in REQUIRED_GEOMETRY if k not in geometry]
    if missing:
        raise KeyError(f"geometry is missing {missing} required by the flux integrals")


def _import_gyaradax():
    import importlib.util
    if importlib.util.find_spec("gyaradax") is None:
        raise ImportError("flux integrals need gyaradax (pip install -e '.[gyro]')")
    prev = jax.config.jax_enable_x64
    import gyaradax.integrals  # noqa: F401
    jax.config.update("jax_enable_x64", prev)


def _x64(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        _import_gyaradax()
        with jax.enable_x64(True):
            return fn(*args, **kwargs)
    return wrapped


def _f64(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with jax.enable_x64(True):
            return fn(*args, **kwargs)
    return wrapped


@_x64
def gyaradax_spectral_fields(df_batch, geometry: dict, *, per_sample: bool = False):
    """Spectral potential + per-mode heat-flux field of a spatial df batch via gyaradax.

    ``df_batch`` is ``(B, 4, vp, mu, s, x, y)`` (separate-zf layout) or ``(B, 2, ...)``;
    ``geometry`` is one trajectory's geometry, or with ``per_sample`` a geometry whose
    leaves carry a leading batch axis. Returns ``(phi_spec, eflux_field)`` as host numpy
    arrays of shapes ``(B, s, kx, ky)`` (complex) and ``(B, kx, ky)``; ``parseval`` is
    replaced by the Hermitian factor ``where(|krho| < 1e-12, 1, 2)``.
    """
    require_integrals(geometry)
    df_batch = jnp.asarray(df_batch)
    df_rec = df_batch[:, :2] + df_batch[:, 2:] if df_batch.shape[1] == 4 else df_batch
    df_cplx = (df_rec[:, 0] + 1j * df_rec[:, 1]).astype(jnp.complex128)
    geom = {k: jnp.asarray(v) for k, v in geometry.items()}
    geom["parseval"] = jnp.where(jnp.abs(geom["krho"]) < 1e-12, 1.0, 2.0).astype(jnp.float64)
    fn = _gyaradax_spectral_per_sample if per_sample else _gyaradax_spectral_batched
    phi, eflux = fn(df_cplx, geom)
    return np.asarray(phi), np.asarray(eflux)


@jax.jit
def _gyaradax_spectral_one(df_one, geom):
    """Per-sample (spatial complex df, geom) → (phi_spec, per-(kx, ky) eflux field)."""
    from gyaradax.integrals import _phi_adiabatic, calculate_fluxes, geom_tensors
    spec = jnp.fft.fftn(df_one, axes=(-2, -1), norm="forward")
    spec = jnp.fft.ifftshift(spec, axes=-2)
    gt = geom_tensors(geom)
    phi = _phi_adiabatic(gt, spec)  # (s, kx, ky) complex
    phi = _zonal_correction(gt, geom, spec, phi)
    _pflux, eflux, _vflux = calculate_fluxes(gt, spec, phi, reduce=False)
    return phi, eflux


def _zonal_correction(gt, geom, spec, phi):
    """Zonal (kx_idx=0, ky_idx=0) phi mode with the zonal-profile correction.

    Fluxes are unaffected (eflux ∝ krho = 0 on the zonal column). Reduces to
    ``phi = phi_raw + Σ_s matz·phi_raw``.
    """
    de, signz, tmp = gt["de"], gt["signz"], gt["tmp"]
    intvp, intmu, bn = gt["intvp"], gt["intmu"], gt["bn"]
    bessel, gamma = gt["bessel"], gt["gamma"]
    # phi_raw = Σ_{v,mu} poisson_int · df at the quirk mode (kx=0-index, ky=0-index)
    poisson_int = signz * de * intmu * intvp * bessel * bn
    phi_raw = jnp.sum(poisson_int * spec, axis=(1, 2))[0, :, 0, 0]  # (s,)
    ints_s = jnp.asarray(geom["ints"], dtype=jnp.float64)
    gamma00 = gamma[0, 0, 0, :, 0, 0]  # (s,) — gamma is v/mu-independent
    sz, d, t = signz.ravel()[0], de.ravel()[0], tmp.ravel()[0]
    diagz = sz * (gamma00 - 1.0) / t
    matz = -ints_s / (sz * d * (diagz - 1.0 / t))
    phi_new = phi_raw + jnp.sum(matz * phi_raw)
    return phi.at[:, 0, 0].set(phi_new)


_gyaradax_spectral_batched = jax.jit(jax.vmap(_gyaradax_spectral_one, in_axes=(0, None)))
_gyaradax_spectral_per_sample = jax.jit(jax.vmap(_gyaradax_spectral_one, in_axes=(0, 0)))


_SCALAR_DEFAULTS = {
    "mas": 1.0, "tmp": 1.0, "d2X": 1.0, "signz": 1.0, "signB": 1.0, "adiabatic": 1.0,
    "de": 1.0, "vthrat": 1.0, "beta": 0.0, "nlapar": 0.0, "nlbpar": 0.0,
}


@_f64
def precompute_geometry(geometry: dict, dtype=np.float32) -> dict[str, np.ndarray]:
    """Broadcast-ready geometry tensors for :func:`flux_integral` (one trajectory).

    Computed in float64 (Bessel / scaled-I0 gyroaverage terms included) and cast
    to ``dtype``. Tensors are laid out against a ``(vp, mu, s, x, y)`` df;
    species scalars must be single-species and become 0-d arrays.
    """
    from jax.scipy.special import bessel_jn, i0e

    g = {k: np.asarray(v, dtype=np.float64) for k, v in geometry.items()}
    out = {
        "krho": g["krho"].reshape(1, 1, 1, 1, -1),
        "ints": g["ints"].reshape(1, 1, -1, 1, 1),
        "intmu": g["intmu"].reshape(1, -1, 1, 1, 1),
        "intvp": g["intvp"].reshape(-1, 1, 1, 1, 1),
        "vpgr": g["vpgr"].reshape(-1, 1, 1, 1, 1),
        "mugr": g["mugr"].reshape(1, -1, 1, 1, 1),
        "parseval": g["parseval"].reshape(1, 1, 1, 1, -1),
    }
    for k in ("bn", "efun", "rfun", "bt_frac"):
        out[k] = g[k].reshape(1, 1, -1, 1, 1)
    for k, default in _SCALAR_DEFAULTS.items():
        v = g.get(k, np.asarray(default))
        if v.size != 1:
            raise ValueError(f"flux_integral is single-species; geometry[{k!r}] has shape {v.shape}")
        out[k] = v.reshape(())
    kxrh = g["kxrh"].reshape(1, 1, 1, -1, 1)
    little_g = g["little_g"].T.reshape(3, 1, 1, -1, 1, 1)
    krho = out["krho"]
    krloc = np.sqrt(krho**2 * little_g[0] + 2 * krho * kxrh * little_g[1] + kxrh**2 * little_g[2])
    out["krloc"] = krloc
    z = out["mas"] * out["vthrat"] * krloc * np.sqrt(2.0 * out["mugr"] / out["bn"]) / out["signz"]
    j = np.asarray(bessel_jn(jnp.asarray(np.where(np.abs(z) < 1e-8, 1.0, z)), v=1))
    out["bessel"] = np.where(np.abs(z) < 1e-8, 1.0, j[0])
    out["bessel_bpar"] = np.where(np.abs(z) < 1e-8, 1.0, 2.0 * j[1] / np.where(np.abs(z) < 1e-8, 1.0, z))
    gam = 0.5 * (out["mas"] * out["vthrat"] * krloc / (out["signz"] * out["bn"])) ** 2
    out["gamma"] = np.asarray(i0e(jnp.asarray(gam)))
    return {k: np.ascontiguousarray(v, dtype=dtype) for k, v in out.items()}


def _df_fft(df: jnp.ndarray) -> jnp.ndarray:
    spec = jnp.fft.fftn(df[0] + 1j * df[1], axes=(-2, -1), norm="forward")
    return jnp.fft.ifftshift(spec, axes=-2)


def _phi_to_spc(phi: jnp.ndarray, out_shape: tuple, real_potens: bool) -> jnp.ndarray:
    if not real_potens:
        phi = phi[0] + 1j * phi[1]
    phi = jnp.fft.fftshift(jnp.fft.fftn(phi, axes=(0, 2), norm="forward"), axes=0)
    if phi.shape != out_shape:
        nx, _, ny = out_shape
        phi = phi[..., phi.shape[-1] // 2:]
        xpad = (phi.shape[0] - nx) // 2 + (1 if phi.shape[0] % 2 == 0 else 0)
        phi = phi[xpad:nx + xpad, :, :ny]
    return jnp.transpose(phi, (1, 0, 2))


def _spc_to_phi(spc: jnp.ndarray, real_potens: bool) -> jnp.ndarray:
    spc = jnp.fft.ifftshift(jnp.transpose(spc, (1, 0, 2)), axes=0)
    if real_potens:
        return jnp.fft.irfftn(spc, s=(spc.shape[0], spc.shape[2]), axes=(0, 2), norm="forward")
    phi = jnp.fft.ifftn(spc, axes=(0, 2), norm="forward")
    return jnp.stack([phi.real, phi.imag])


def _solve_fields(g: dict, spec: jnp.ndarray):
    signz, de, tmp, bn = g["signz"], g["de"], g["tmp"], g["bn"]
    ints, intvp, intmu, gamma = g["ints"], g["intvp"], g["intmu"], g["gamma"]
    adiabatic = g["adiabatic"]
    phi = jnp.sum(signz * de * intmu * intvp * g["bessel"] * bn * spec, axis=(0, 1), keepdims=True)

    diag = (signz**2 * de * (gamma - 1.0) / tmp)
    diag = diag.at[..., 0, 0].set(0.0) - adiabatic
    diag = -1.0 / jnp.where(diag == 0.0, 1.0, diag)

    # zonal-flow correction on the ky=0 column, kx index 0 excluded
    diagz = signz * (gamma - 1.0) / tmp
    matz = -ints / (signz * de * (diagz - 1.0 / tmp))
    matz = matz.at[..., 1:].set(0.0)
    maty = tmp / de + jnp.sum(-matz, axis=-3, keepdims=True)
    maty = maty.at[..., 0, :].set(1.0)
    maty = 1.0 / jnp.where(maty == 0.0, 1.0, maty)
    maty = maty.at[..., 1:].set(0.0)
    bufphi = jnp.sum(matz * phi, axis=(-3, -1), keepdims=True)
    phi = ((phi + maty * bufphi * adiabatic) * diag)[0, 0]

    krloc2 = g["krloc"] ** 2
    krloc2 = jnp.where(krloc2 == 0.0, 1.0, krloc2)
    apar_int = g["beta"] * signz * de * g["vthrat"] * intmu * intvp * g["vpgr"] * g["bessel"] * bn
    apar = jnp.sum(apar_int * spec, axis=(0, 1), keepdims=True) / krloc2 * g["nlapar"]
    bpar_int = g["beta"] * de * tmp * intmu * intvp * g["mugr"] * g["bessel_bpar"] * bn
    bpar = -jnp.sum(bpar_int * spec, axis=(0, 1), keepdims=True) / krloc2 * g["nlbpar"]
    return phi, apar[0, 0], bpar[0, 0]


def _pev_fluxes(g: dict, spec, phi, apar, bpar):
    vpgr, mugr, bn, ints, intmu, intvp = g["vpgr"], g["mugr"], g["bn"], g["ints"], g["intmu"], g["intvp"]
    chi = (g["bessel"] * phi - 2.0 * g["vthrat"] * vpgr * g["bessel"] * apar
           + 2.0 * mugr * g["tmp"] / g["signz"] * g["bessel_bpar"] * bpar)
    dum1 = jnp.imag(g["parseval"] * ints * g["efun"] * g["krho"] * spec * jnp.conj(chi))
    d3v = ints * g["d2X"] * intmu * bn * intvp
    pflux = jnp.sum(d3v * dum1 * g["de"])
    eflux = jnp.sum(d3v * (vpgr**2 * dum1 + 2.0 * mugr * bn * dum1) * g["de"] * g["tmp"])
    vflux = jnp.sum(d3v * dum1 * vpgr * g["rfun"] * g["bt_frac"] * g["signB"]
                    * g["de"] * g["mas"] * g["vthrat"] ** 2)
    return pflux, eflux, vflux


def flux_integral(geom_t: dict, df: jnp.ndarray, phi: Optional[jnp.ndarray] = None,
                  *, real_potens: bool = True):
    """Jittable single-sample fields and fluxes from a spatial df (real-space phi out).

    ``geom_t`` comes from :func:`precompute_geometry`; ``df`` is ``(2, vp, mu, s, x, y)``
    (real/imag, spatial x/y). Fields are solved from ``df``; an external ``phi``
    (``(x, s, y)`` real, or ``(2, x, s, y)``) replaces the solved one in the fluxes.
    Returns ``(phi_int, (pflux, eflux, vflux))`` with ``phi_int`` in the dataset
    layout ``(x, s, y)`` (``(2, x, s, y)`` without ``real_potens``). Runs in the
    input precision.
    """
    ns, nx, ny = df.shape[-3:]
    spec = _df_fft(df)
    phi_s, apar_s, bpar_s = _solve_fields(geom_t, spec)
    phi_f = phi_s if phi is None else _phi_to_spc(phi, (nx, ns, ny), real_potens)
    fluxes = _pev_fluxes(geom_t, spec, phi_f, apar_s, bpar_s)
    return _spc_to_phi(phi_s, real_potens), fluxes
