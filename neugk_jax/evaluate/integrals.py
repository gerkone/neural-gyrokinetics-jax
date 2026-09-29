"""Gyaradax adapter — phi / particle / heat / momentum fluxes from df.

Delegates to the pure-JAX ``gerkone/gyaradax`` package. The API:

    from gyaradax.integrals import get_integrals
    phi, (pflux, eflux, vflux) = get_integrals(df, geometry, params=None, ...)

Inputs are unbatched; we ``vmap`` over the batch axis. The public entry points
run in float64 inside a local ``jax.enable_x64`` context; gyaradax's global x64
switch on import is undone so the rest of the process stays fp32.

``gyaradax`` is electrostatic-only at the moment (no apar/bpar paths).
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

_GEOMETRY_KEYS = (
    "krho", "ints", "intmu", "intvp", "vpgr", "mugr",
    "bn", "ffun", "efun", "rfun", "bt_frac", "parseval",
    "mas", "tmp", "d2X", "signz", "signB", "de", "vthrat",
    "kxrh", "little_g",
)
_PARAM_KEYS = ("adiabatic", "beta", "nlapar", "nlbpar")



def _import_gyaradax():
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


def _split_geom_and_params(geometry: dict[str, jnp.ndarray]):
    geom = {k: geometry[k] for k in _GEOMETRY_KEYS if k in geometry}
    params_dict = {k: geometry[k] for k in _PARAM_KEYS if k in geometry}
    return geom, params_dict


@_x64
def compute_integrals(
    df: jnp.ndarray,
    geometry: dict[str, jnp.ndarray],
    *,
    params: Optional[Any] = None,
    adiabatic_electrons: bool = True,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Compute (phi, (pflux, eflux, vflux)) for one sample's distribution function.

    ``df`` shape (with adiabatic electrons): ``(vpar, mu, s, x, y)`` or
    ``(2, vpar, mu, s, x, y)`` for complex inputs (real / imag channels);
    ``gyaradax.get_integrals`` accepts both layouts.
    """
    from gyaradax.integrals import get_integrals
    geom, gparams = _split_geom_and_params(geometry)
    p = params
    if p is None and gparams:
        # pass scalars as-is; gyaradax builds its params object or falls back to compute_geometry defaults
        p = gparams
    return get_integrals(df, geom, params=p, adiabatic_electrons=adiabatic_electrons)


@_x64
def batched_integrals(
    df_batch: jnp.ndarray,
    geometry_batch: dict[str, jnp.ndarray],
    *,
    adiabatic_electrons: bool = True,
):
    def one(df, geom_one):
        return compute_integrals(df, geom_one, adiabatic_electrons=adiabatic_electrons)
    return jax.vmap(one)(df_batch, geometry_batch)


@_x64
def gyaradax_flux_integrals(
    df_batch: jnp.ndarray,
    geometry_one: dict,
):
    """Pure-JAX flux integral via gyaradax. Returns ``(phi, eflux)`` as
    host numpy arrays.

    Inputs:
        df_batch:     ``(B, 4, vp, mu, s, x, y)`` — denormalised AE-decoded
                      df with the separate-zf channel-of-4 layout.
        geometry_one: dict of per-trajectory geometry values (no batch axis).
                      Caller is expected to have stripped the batch axis
                      (batches in the eval loop are single-trajectory).

    Pipeline:
        recombine zf → real/imag → complex 5D (vp, mu, s, x, y) →
        forward FFT (x, y, norm='forward'; ifftshift on x) → gyaradax
        ``get_integrals`` adiabatic path.

    Overrides ``parseval`` with the Hermitian-symmetry factor
    ``where(|krho|<1e-12, 1, 2)``.
    """
    df_batch = jnp.asarray(df_batch)
    B = df_batch.shape[0]
    # recombine_zf: (B, 4, ...) → (B, 2, ...) → complex (B, vp, mu, s, x, y)
    df_rec = df_batch[:, :2] + df_batch[:, 2:]
    df_cplx = (df_rec[:, 0] + 1j * df_rec[:, 1]).astype(jnp.complex128)

    geom = {k: jnp.asarray(v) for k, v in geometry_one.items()}
    # batched geom (leaves carry leading batch axis matching B)? else single-traj path
    krho = geom.get("krho")
    if krho is not None and krho.ndim > 0 and krho.shape[0] == B:
        # batched geom — collapse if all samples share trajectory, else vmap over both
        if _same_traj(geom, B):
            geom = _strip_batch_axis(geom, B)
            geom["parseval"] = jnp.where(
                jnp.abs(geom["krho"]) < 1e-12, 1.0, 2.0,
            ).astype(jnp.float64)
            phi, eflux = _gyaradax_integ_batched(df_cplx, geom)
        else:
            # per-sample geom: override parseval per row (vectorized)
            geom["parseval"] = jnp.where(
                jnp.abs(geom["krho"]) < 1e-12, 1.0, 2.0,
            ).astype(jnp.float64)
            phi, eflux = _gyaradax_integ_batched_geom(df_cplx, geom)
    else:
        # caller passed a single-trajectory geom (no batch axis)
        geom["parseval"] = jnp.where(
            jnp.abs(jnp.asarray(geom["krho"])) < 1e-12, 1.0, 2.0,
        ).astype(jnp.float64)
        phi, eflux = _gyaradax_integ_batched(df_cplx, geom)
    return np.asarray(phi), np.asarray(eflux)


@_x64
def gyaradax_spectral_fields(
    df_batch: jnp.ndarray,
    geometry_one: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral potential + per-mode heat-flux field via gyaradax.

    Same pipeline as ``gyaradax_flux_integrals`` (recombine zf → complex df →
    forward FFT with the ifftshift-on-x convention) but keeps the per-mode
    fields instead of reducing to scalars:

    Returns ``(phi_spec, eflux_field)`` as host numpy arrays with shapes
    ``(B, s, kx, ky)`` (complex) and ``(B, kx, ky)``.

    Inputs:
        df_batch:     ``(B, 4, vp, mu, s, x, y)`` spatial df (separate-zf
                      channel-of-4 layout; a plain 2-channel df also works).
        geometry_one: single-trajectory geometry dict (no batch axis).

    Applies the same ``parseval`` override as ``gyaradax_flux_integrals``.
    """
    df_batch = jnp.asarray(df_batch)
    # recombine_zf only applies to the separate-zf channel-of-4 layout
    df_rec = df_batch[:, :2] + df_batch[:, 2:] if df_batch.shape[1] == 4 else df_batch
    df_cplx = (df_rec[:, 0] + 1j * df_rec[:, 1]).astype(jnp.complex128)

    geom = {k: jnp.asarray(v) for k, v in geometry_one.items()}
    geom["parseval"] = jnp.where(
        jnp.abs(jnp.asarray(geom["krho"])) < 1e-12, 1.0, 2.0,
    ).astype(jnp.float64)
    phi, eflux = _gyaradax_spectral_batched(df_cplx, geom)
    return np.asarray(phi), np.asarray(eflux)


@jax.jit
def _gyaradax_spectral_one(df_one, geom):
    """Per-sample (spatial complex df, geom) → (phi_spec, eflux_field).

    Same FFT convention as ``_gyaradax_integ_one`` but goes through the
    gyaradax internals directly so ``calculate_fluxes`` can keep the
    per-(kx, ky) flux field (``reduce=False``)."""
    from gyaradax.integrals import _phi_adiabatic, calculate_fluxes, geom_tensors
    spec = jnp.fft.fftn(df_one, axes=(-2, -1), norm="forward")
    spec = jnp.fft.ifftshift(spec, axes=-2)
    gt = geom_tensors(geom)
    phi = _phi_adiabatic(gt, spec)  # (s, kx, ky) complex
    phi = _torch_zonal_quirk(gt, geom, spec, phi)
    _pflux, eflux, _vflux = calculate_fluxes(gt, spec, phi, reduce=False)
    return phi, eflux


def _torch_zonal_quirk(gt, geom, spec, phi):
    """Special-case the zonal (kx_idx=0, ky_idx=0) phi mode to match the
    torch zonal-profile convention; fluxes are unaffected (eflux ∝ krho = 0
    on the zonal column). Reduces to ``phi = phi_raw + Σ_s matz·phi_raw``.
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


# vmap with shared geom — evaluators integrate one trajectory at a time
_gyaradax_spectral_batched = jax.jit(jax.vmap(_gyaradax_spectral_one, in_axes=(0, None)))


@jax.jit
def _gyaradax_integ_one(df_one, geom):
    """Per-sample (spectral df, geom) → (phi, eflux). FFT inside so the
    caller can stay in spatial layout. Jit'd once at module level."""
    from gyaradax.integrals import get_integrals
    spec = jnp.fft.fftn(df_one, axes=(-2, -1), norm="forward")
    spec = jnp.fft.ifftshift(spec, axes=-2)
    phi, (_pflux, eflux, _vflux) = get_integrals(
        spec, geom, adiabatic_electrons=True,
    )
    return phi, eflux


# vmap with shared geom (single trajectory in a batch) — the fast common case
_gyaradax_integ_batched = jax.jit(jax.vmap(_gyaradax_integ_one, in_axes=(0, None)))
# vmap with per-sample geom (handles mixed-trajectory batches at boundaries)
_gyaradax_integ_batched_geom = jax.jit(jax.vmap(_gyaradax_integ_one, in_axes=(0, 0)))


def _strip_batch_axis(geom: dict, batch_size: int) -> dict:
    """Drop the leading batch axis from a geom dict whose leaves were stacked
    across a batch (via ``get_batch_geometry``). Returns the geom of the first
    sample — only safe when all samples share the same trajectory."""
    out = {}
    for k, v in geom.items():
        arr = jnp.asarray(v)
        if arr.ndim and arr.shape[0] == batch_size:
            out[k] = arr[0]
        else:
            out[k] = arr
    return out


def _same_traj(geom: dict, batch_size: int) -> bool:
    """Cheap heuristic — check if the batched geom's per-sample slices are
    identical (i.e. all from the same trajectory)."""
    krho = geom.get("krho")
    if krho is None:
        return False
    arr = np.asarray(krho)
    if arr.ndim == 0 or arr.shape[0] != batch_size:
        return False
    return bool(np.all(arr == arr[0:1]))


def torch_flux_integrals(
    df_batch,
    geometry_batch: dict,
):
    """Compute ``(phi, eflux)`` by delegating to torch ``FluxIntegral``.

    Prefers CUDA, falls back to CPU with a capped thread count. The
    integrator is built once and cached. Returns ``(phi_np, eflux_np)``
    as host numpy arrays.
    """
    import numpy as _np
    import torch
    from neugk.integrals import FluxIntegral
    from neugk.utils import recombine_zf

    device = torch_flux_integrals._device
    if device is None:
        # cpu fallback with a capped thread count; override via TORCH_FLUX_DEVICE
        import os
        override = os.environ.get("TORCH_FLUX_DEVICE")
        if override:
            device = torch.device(override)
        else:
            device = torch.device("cpu")
        if device.type == "cpu":
            torch.set_num_threads(int(os.environ.get("TORCH_FLUX_THREADS", "16")))
            try:
                torch.set_num_interop_threads(2)
            except RuntimeError:
                pass
        torch_flux_integrals._device = device

    integrator = torch_flux_integrals._integrator
    if integrator is None:
        integrator = FluxIntegral(
            real_potens=True, spectral_potens=False, flux_fields=False,
            spectral_df=False, integral_precision="float64",
        ).to(device)
        torch_flux_integrals._integrator = integrator

    df_t = torch.as_tensor(_np.array(df_batch), device=device)
    if df_t.dim() == 7:
        df_t = recombine_zf(df_t, dim=1)  # (B, 2, vp, mu, s, x, y)
    df_t = df_t.unsqueeze(1).double()  # (B, sp=1, 2, vp, mu, s, x, y)

    geom_t = {
        k: torch.as_tensor(_np.array(v), device=device).double()
        for k, v in geometry_batch.items()
    }
    with torch.no_grad():
        phi, (_pflux, eflux, _vflux) = integrator(geom_t, df_t)
    return _np.asarray(phi.float().cpu()), _np.asarray(eflux.squeeze(-1).float().cpu())


torch_flux_integrals._integrator = None  # type: ignore[attr-defined]
torch_flux_integrals._device = None  # type: ignore[attr-defined]


_SCALAR_DEFAULTS = {
    "mas": 1.0, "tmp": 1.0, "d2X": 1.0, "signz": 1.0, "signB": 1.0, "adiabatic": 1.0,
    "de": 1.0, "vthrat": 1.0, "beta": 0.0, "nlapar": 0.0, "nlbpar": 0.0,
}


@_x64
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
    """Jittable single-sample port of torch ``FluxIntegral`` (spatial df, real-space phi out).

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
