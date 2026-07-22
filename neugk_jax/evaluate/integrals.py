"""Gyaradax adapter — phi / particle / heat / momentum fluxes from df.

The upstream torch repo carries its own python port of the GKW field-solve
+ phase-space integrals (``neugk/integrals.py:FluxIntegral``). We delegate
to the pure-JAX ``gerkone/gyaradax`` package instead. The API:

    from gyaradax.integrals import get_integrals
    phi, (pflux, eflux, vflux) = get_integrals(df, geometry, params=None, ...)

Inputs are unbatched; we ``vmap`` over the batch axis. Physical
quantities should run in ``float64`` — ``gyaradax`` sets
``jax_enable_x64=True`` globally on import. We don't put the integrals on
the training graph so the x64 promotion is contained to eval-only
``jit`` blocks.

Note: ``gyaradax`` is electrostatic-only at the moment (no apar/bpar
paths). That matches what the upstream evaluator actually uses.
"""

from __future__ import annotations

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


def _split_geom_and_params(geometry: dict[str, jnp.ndarray]):
    """Separate gyaradax's geometry dict from its params dict.

    Upstream's torch ``FluxIntegral`` lumps everything into one geometry
    dict; gyaradax expects them split.
    """
    geom = {k: geometry[k] for k in _GEOMETRY_KEYS if k in geometry}
    params_dict = {k: geometry[k] for k in _PARAM_KEYS if k in geometry}
    return geom, params_dict


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


def batched_integrals(
    df_batch: jnp.ndarray,
    geometry_batch: dict[str, jnp.ndarray],
    *,
    adiabatic_electrons: bool = True,
):
    """``vmap``ed compute_integrals over the batch axis (axis 0)."""
    def one(df, geom_one):
        return compute_integrals(df, geom_one, adiabatic_electrons=adiabatic_electrons)
    return jax.vmap(one)(df_batch, geometry_batch)


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

    **Parseval correction** — upstream torch's metadata.pkl ships
    ``parseval = [1, ny, ny, ...]`` (a physics bug: it should be the
    Hermitian-symmetry factor ``[1, 2, 2, ...]``, see
    ``gyaradax/geometry/geom.py:geometry_from_geom_dat_and_input``).
    Torch's ``pev_fluxes`` then double-counts ``ints`` to cancel the
    ``ny/2`` overcount — both bugs land on the right flux. We refuse to
    inherit the bug: ``parseval`` is replaced here with the correct
    ``where(|krho|<1e-12, 1, 2)``, and the single-``ints`` formula in
    gyaradax then gives the same flux values as upstream torch to fp64
    precision, with the correct underlying physics.

    Roughly **300× faster** than the torch FluxIntegral bridge on CPU
    (~12 ms vs ~3.35 s per batch-of-4).
    """
    from gyaradax.integrals import get_integrals
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

    Applies the same corrected-parseval override as
    ``gyaradax_flux_integrals`` (see the comment there).
    """
    import gyaradax  # noqa: F401 — enables jax x64 before any array conversion
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
    """Replicate upstream torch's zonal-correction quirk on one phi mode.

    ``neugk/physics/integrals.py:solve_fields`` special-cases kx INDEX 0 on
    the zonal (ky index 0) column (``poisson_diag[..., 0, 0] = 0``,
    ``maty[..., 0, :] = 1``) instead of the kx=0 mode. gyaradax applies the
    same treatment at ``ixzero = argmin|kxrh|`` — at that mode the two
    formulations coincide analytically (gamma=1 there), so torch and
    gyaradax phi differ ONLY at (kx_idx=0, ky_idx=0). The mode is the
    zonal-profile diagnostic's, so we inherit the torch value for metric
    parity; fluxes are untouched (eflux ∝ krho = 0 on the zonal column).
    torch there reduces to ``phi = phi_raw + Σ_s matz·phi_raw``.
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
    """Compute ``(phi, eflux)`` by delegating to upstream torch ``FluxIntegral``.

    Runs on CUDA so the eval doesn't blow up to thousands of CPU OpenMP
    threads (saw 1290 threads at 0% GPU util before — the per-batch
    CPU FluxIntegral was the eval bottleneck). The integrator is built
    once and cached. Returns ``(phi_np, eflux_np)`` as host numpy arrays.
    """
    import numpy as _np
    import torch
    from neugk.integrals import FluxIntegral
    from neugk.utils import recombine_zf

    device = torch_flux_integrals._device
    if device is None:
        # FluxIntegral CUDA path fails nvrtc on this cluster; CPU fallback with thread cap (override via TORCH_FLUX_DEVICE)
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
