"""Field solve, fluxes and spectra of a spatial df through the ``gyaradax`` integrals.

``precompute_geometry`` + ``flux_integral`` are the jittable single-sample field solve and
fluxes used in training and evaluation, ``spectral_integrals`` adds the ky spectra and
``gyaradax_spectral_fields`` keeps the per-mode potential and heat flux for the spectral
metrics. The df carries a species axis, ``(2, species, vp, mu, s, x, y)``; a df without it
is one species. One species is solved with adiabatic electrons, several kinetically, and
fluxes are per species. Electrostatic only.
"""

from __future__ import annotations

import functools
import importlib.util
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.evaluate.fourier import df_to_spec, phi_to_spec, spec_to_phi, spec_to_phi_complex
from neugk_jax.utils import recombine_zf

REQUIRED_GEOMETRY = (
    "krho",
    "kxrh",
    "ints",
    "intmu",
    "intvp",
    "vpgr",
    "mugr",
    "bn",
    "efun",
    "rfun",
    "bt_frac",
    "little_g",
)
SPECIES_KEYS = ("mas", "tmp", "de", "signz", "vthrat")


def require_geometry(geometry: Optional[dict]) -> None:
    """Raise unless ``geometry`` carries every field :func:`precompute_geometry` needs."""
    if geometry is None:
        raise ValueError("eval_integrals needs trajectory geometry in the metadata")
    missing = [k for k in REQUIRED_GEOMETRY if k not in geometry]
    if missing:
        raise KeyError(f"geometry is missing {missing} required by the flux integrals")


def _import_gyaradax():
    if importlib.util.find_spec("gyaradax") is None:
        raise ImportError("the flux integrals need gyaradax (pip install -e '.[gyro]')")
    from jax._src.config import enable_x64

    # the global value, also when called inside a jax.enable_x64 context
    prev = enable_x64.get_global()
    import gyaradax.integrals  # noqa: F401

    # gyaradax switches x64 on globally at import
    jax.config.update("jax_enable_x64", prev)


def _float64(fn):
    """Run ``fn`` inside a local ``jax.enable_x64`` context."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with jax.enable_x64(True):
            return fn(*args, **kwargs)

    return wrapped


def precompute_geometry(geometry: dict, dtype=np.float32) -> dict[str, np.ndarray]:
    """Gyaradax geometry tensors of one trajectory, computed in float64 and cast to ``dtype``.

    Per-species ``geom_tensors`` stacked on a leading species axis, plus the kinetic Poisson
    diagonal ``phi_diag`` for several species. Missing scalars take the
    :func:`complete_geometry` defaults; ``parseval`` is the Hermitian factor
    ``where(|krho| < 1e-12, 1, 2)``.
    """
    from neugk_jax.dataset.backend import complete_geometry

    _import_gyaradax()
    from gyaradax.integrals import geom_tensors, precompute_phi_kinetic

    g = {k: np.asarray(v, dtype=np.float64) for k, v in complete_geometry(geometry).items()}
    for k in SPECIES_KEYS:
        g[k] = np.atleast_1d(g[k])
    g["parseval"] = np.where(np.abs(g["krho"]) < 1e-12, 1.0, 2.0)
    ns = len(g["mas"])
    with jax.enable_x64(True):
        gj = {k: jnp.asarray(v) for k, v in g.items()}
        per_species = []
        for i in range(ns):
            gs = {**gj, **{k: gj[k][i : i + 1] for k in SPECIES_KEYS}}
            per_species.append({k: np.asarray(v) for k, v in geom_tensors(gs).items()})
        out = {k: np.stack([t[k] for t in per_species]) for k in per_species[0]}
        if ns > 1:
            out["phi_diag"] = np.asarray(precompute_phi_kinetic(gj)[1])
    cast = lambda v: v.astype(dtype) if v.dtype.kind == "f" else v
    return {k: np.ascontiguousarray(cast(v)) for k, v in out.items()}


def _with_species(df):
    # (2, vp, mu, s, x, y) is one species
    return df[:, None] if df.ndim == 6 else df


INDEX_KEYS = ("ixzero", "iyzero")


def _species(geom: dict, i: int) -> dict:
    # zonal-mode indices stay integers through dtype casts of the whole geometry
    return {
        k: v[i].astype(jnp.int32) if k in INDEX_KEYS else v[i]
        for k, v in geom.items()
        if k != "phi_diag"
    }


def _zonal_weights(signz, de, tmp, gamma, ints):
    return -ints / (signz * de * (signz * (gamma - 1.0) / tmp - 1.0 / tmp))


def _zonal_correction(gt, spec, phi):
    """Zonal (kx_idx=0, ky_idx=0) phi mode with the zonal-profile correction of GKW.

    Fluxes are unaffected (eflux ∝ krho = 0 on the zonal column). Reduces to
    ``phi = phi_raw + Σ_s matz·phi_raw``.
    """
    de, signz, tmp = gt["de"], gt["signz"], gt["tmp"]
    intvp, intmu, bn = gt["intvp"], gt["intmu"], gt["bn"]
    bessel, gamma = gt["bessel"], gt["gamma"]
    poisson_int = signz * de * intmu * intvp * bessel * bn
    phi_raw = jnp.sum(poisson_int * spec, axis=(1, 2))[0, :, 0, 0]
    gamma00 = gamma[0, 0, 0, :, 0, 0]
    sz, d, t = signz.ravel()[0], de.ravel()[0], tmp.ravel()[0]
    matz = _zonal_weights(sz, d, t, gamma00, gt["ints"].reshape(-1))
    return phi.at[:, 0, 0].set(phi_raw + jnp.sum(matz * phi_raw))


def solve_phi(geom: dict, spec: jnp.ndarray) -> jnp.ndarray:
    """Spectral potential ``(s, kx, ky)`` of a spectral df ``(species, vp, mu, s, kx, ky)``.

    One species is solved with adiabatic electrons, several with kinetic quasineutrality.
    """
    _import_gyaradax()
    from gyaradax.integrals import _phi_adiabatic

    if spec.shape[0] == 1:
        gt = _species(geom, 0)
        return _zonal_correction(gt, spec[0], _phi_adiabatic(gt, spec[0]))
    # gyaradax precompute_phi_kinetic weight, factored per species
    weight = (
        geom["signz"] * geom["de"] * geom["intmu"] * geom["intvp"] * geom["bessel"] * geom["bn"]
    )
    num = jnp.sum(weight[:, 0] * spec, axis=(0, 1, 2))
    return -num / geom["phi_diag"]


def species_fluxes(geom: dict, spec: jnp.ndarray, phi: jnp.ndarray, reduce: bool = True):
    """``(species, 3)`` (pflux, eflux, vflux), or ``(species, 3, kx, ky)`` without ``reduce``."""
    _import_gyaradax()
    from gyaradax.integrals import calculate_fluxes

    return jnp.stack(
        [
            jnp.stack(calculate_fluxes(_species(geom, i), spec[i], phi, reduce=reduce))
            for i in range(spec.shape[0])
        ]
    )


def flux_integral(geom: dict, df: jnp.ndarray, phi: Optional[jnp.ndarray] = None):
    """Jittable single-sample potential and fluxes of a spatial df.

    ``geom`` comes from :func:`precompute_geometry`. The potential is solved from ``df``; an
    external real ``phi`` ``(x, s, y)`` replaces it in the fluxes. Returns ``(phi_int,
    (pflux, eflux, vflux))``, ``phi_int`` in the dataset layout ``(x, s, y)`` and each flux
    ``(species,)``.
    """
    df = _with_species(df)
    ns, nx, ny = df.shape[-3:]
    spec = df_to_spec(df)
    phi_s = solve_phi(geom, spec)
    if phi is not None:
        shape = None if phi.shape == (nx, ns, ny) else (nx, ns, ny)
        phi_f = jnp.transpose(phi_to_spec(phi, shape), (1, 0, 2))
    fluxes = species_fluxes(geom, spec, phi_s if phi is None else phi_f)
    return spec_to_phi(jnp.transpose(phi_s, (1, 0, 2))), tuple(fluxes.T)


def flux_spectrum(geom: dict, df: jnp.ndarray) -> jnp.ndarray:
    """Heat flux per species and ky ``(species, ky)`` of a spatial df."""
    spec = df_to_spec(_with_species(df))
    return species_fluxes(geom, spec, solve_phi(geom, spec), reduce=False)[:, 1].sum(axis=-2)


def spectral_integrals(geom: dict, df: jnp.ndarray, *, ds: float) -> dict:
    """Jittable single-sample potential, fluxes and ky spectra from a spatial df.

    Returns ``phi`` (dataset layout), ``phi_c`` (re, im of the complex potential of the
    one-sided spectrum), ``pflux`` / ``eflux`` per species, ``kyspec = ds * sum_(s, kx)
    |phi_k|^2`` and ``qspec``, the heat flux per species and ky.
    """
    spec = df_to_spec(_with_species(df))
    phi_s = solve_phi(geom, spec)
    fields = species_fluxes(geom, spec, phi_s, reduce=False)
    phi_k = jnp.transpose(phi_s, (1, 0, 2))
    return {
        "phi": spec_to_phi(phi_k),
        "phi_c": spec_to_phi_complex(phi_k),
        "pflux": fields[:, 0].sum(axis=(-2, -1)),
        "eflux": fields[:, 1].sum(axis=(-2, -1)),
        "kyspec": ds * jnp.sum(jnp.real(phi_s) ** 2 + jnp.imag(phi_s) ** 2, axis=(0, 1)),
        "qspec": fields[:, 1].sum(axis=-2),
    }


def gyaradax_spectral_fields(df_batch, geometry: dict, *, per_sample: bool = False):
    """Spectral potential + per-mode heat-flux field of a spatial df batch in float64.

    ``df_batch`` is ``(B, C, [species,] vp, mu, s, x, y)`` with the zonal-flow parts of a
    separate-zf layout summed back; ``geometry`` is one trajectory's geometry, or with
    ``per_sample`` a :func:`precompute_geometry` dict whose leaves carry a leading batch
    axis. Returns
    ``(phi_spec, eflux_field)`` as host numpy arrays of shapes ``(B, s, kx, ky)`` (complex)
    and ``(B, species, kx, ky)``.
    """
    _import_gyaradax()
    if not per_sample:
        require_geometry(geometry)
    with jax.enable_x64(True):
        df = recombine_zf(jnp.asarray(df_batch), axis=1).astype(jnp.float64)
        if per_sample:
            geom = {k: jnp.asarray(v) for k, v in geometry.items()}
        else:
            geom = {k: jnp.asarray(v) for k, v in precompute_geometry(geometry, np.float64).items()}
        fn = _spectral_per_sample if per_sample else _spectral_batched
        phi, eflux = fn(df, geom)
        return np.asarray(phi), np.asarray(eflux)


def _spectral_one(df_one, geom):
    spec = df_to_spec(_with_species(df_one))
    phi = solve_phi(geom, spec)
    return phi, species_fluxes(geom, spec, phi, reduce=False)[:, 1]


_spectral_batched = jax.jit(jax.vmap(_spectral_one, in_axes=(0, None)))
_spectral_per_sample = jax.jit(jax.vmap(_spectral_one, in_axes=(0, 0)))
