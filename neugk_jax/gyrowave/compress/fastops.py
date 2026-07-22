"""fast_compress (JAX): per-trajectory dataset-compression operators.

Pure-JAX port of the torch reference (scratchpad/fast_compress/fastops.py). NO torch,
NO neugk (the torch repo). The physics geometry expansion + field solve replicate the
GKW formulas of ``neugk.physics.integrals.FluxIntegral`` in JAX, so W / pint / pdiag /
matz / maty / flux / phi match the torch operators to fp32 precision. Equivalence to the
pure-JAX gyaradax flux (``gerkone/gyaradax``) is confirmed as an independent cross-check
in the scratchpad verification.

Two versions share the operator layer (both score a REAL field):
  version="real"        : ifft/real bins (load .bin); transform HL x (s,x)db4 x fft-y
  version="semispectral": raw K files    (load K);    transform HL x (s)db4 x (kx)db4 x native-ky

Precision: the geometry Bessel/Gamma are a per-trajectory host precompute via
``scipy.special`` in float64 (j0/j1 + i0*exp, matching torch.special.bessel_j0/j1 to
~1e-9 — NB ``jax.scipy.special.bessel_jn`` is inaccurate for |arg|>~40 high-k modes),
then cast to float32; the transform + field solve run in float32 / complex64 with matmul
precision 'highest' (no tf32) to match torch's allow_tf32=False.
"""
from __future__ import annotations

import os
import warnings

import jax
import numpy as np
import scipy.special as _sp

# no tf32 in fp32 matmuls (parity with torch's allow_tf32=False)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp  # noqa: E402
import pywt  # noqa: E402

RES = (32, 8, 16, 85, 32)                       # nvpar, nmu, ns, nkx, nky
SHAPE = (2, *RES)
N_TOTAL = int(np.prod(SHAPE))                    # 22,282,240
FLUX_BAND = (1, 16)
# HL velocity moment count (Hermite x Laguerre). Parametric: velocity is the LEAST compressible
# axis and the (6,3)=18 default is the fidelity bottleneck (df 46 / phi 17 / flux 24% at full
# frame; each of h,l is an atom coord so higher rank flows straight into the dynamic support and
# the pruner keeps the phi/flux-critical moments per snapshot). Override BEFORE import via env
# GYROWAVE_H / GYROWAVE_L so every consumer (pipeline captures CSHAPE/M_TOT at import) agrees.
# Empirical (iter13, velocity-only round-trip): 8x4->df49/phi19/flux3%, 12x6->54/27/1%,
# 16x8->57/49/0.1%, 24x8->63/85/0%.
H = int(os.environ.get("GYROWAVE_H", "6"))
L = int(os.environ.get("GYROWAVE_L", "3"))
assert 1 <= H <= 32 and 1 <= L <= 8, f"HL rank out of range: H={H} L={L} (max 32,8 = velocity grid)"
CSHAPE = (H, L, 16, 87, 32)                      # (h, l, s_wav, kx_wav, ky)
M_TOT = int(np.prod(CSHAPE))                     # default (6,3): 801,792 complex frame atoms

# HL envelopes (fastops: g20 = exp(-2 v^2), e25 = exp(-2.5 mu))
ENV_V = {"g20": lambda v: np.exp(-2.0 * v ** 2)}
ENV_M = {"e25": lambda m: np.exp(-2.5 * m)}

# default raw K-file directory (overridden per trajectory by set_trajectory)
RAW = "/restricteddata/ukaea/gyrokinetics/raw/iteration_13"


# ------------------------------------------------------------------ HL axis bases
def _axis_basis_stable(grid, n, env):
    """Env-weighted monomials Gram-Schmidt (torch _axis_basis_stable, numpy fp64 -> jnp fp32)."""
    x = np.asarray(grid, np.float64)
    Q = np.zeros((len(x), n))
    q = env(x).astype(np.float64)
    q /= np.linalg.norm(q)
    Q[:, 0] = q
    for k in range(1, n):
        w = x * Q[:, k - 1]
        for _ in range(2):
            w -= Q[:, :k] @ (Q[:, :k].T @ w)
        Q[:, k] = w / np.linalg.norm(w)
    return jnp.asarray(Q, dtype=jnp.float32)


# ------------------------------------------------------------------ db4 wavelet mats
def _db4_axis_mats(n, wavelet="db4", mode="periodization", level=3):
    """1D multilevel db4 analysis W (m,n) and synthesis S (n,m) with S@W = I (run_wav.axis_mats).
    Built from pywt on unit vectors (torch-free); returns jnp fp32."""
    cols, slices = [], None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            c = pywt.wavedec(e, wavelet, mode=mode, level=level)
            arr, slices = pywt.coeffs_to_array(c)
            cols.append(arr)
        W = np.stack(cols, 1)
        m = W.shape[0]
        scols = []
        for j in range(m):
            a = np.zeros(m)
            a[j] = 1.0
            c = pywt.array_to_coeffs(a, slices, output_format="wavedec")
            scols.append(pywt.waverec(c, wavelet, mode=mode)[:n])
        S = np.stack(scols, 1)
    err = np.abs(S @ W - np.eye(n)).max()
    assert err < 1e-10, f"S@W != I for n={n}: {err}"
    return jnp.asarray(W, dtype=jnp.float32), jnp.asarray(S, dtype=jnp.float32)


# ------------------------------------------------------------------ geometry expansion
def geom_tensors(geom):
    """Replicate FluxIntegral._geom_tensors (5D layout vp,mu,s,kx,ky). Bessel/Gamma via
    scipy.special (numpy fp64 host precompute, matching torch.special.bessel_j0/j1 + i0*exp
    to ~1e-9; NB jax.scipy.special.bessel_jn is inaccurate for |arg|>~40 at high-k modes).
    Returns fp32 jnp tensors. geom: raw per-trajectory geometry (GKW keys)."""
    g = {k: np.asarray(v, dtype=np.float64) for k, v in geom.items()}
    gd = {}
    gd["krho"] = g["krho"].reshape(1, 1, 1, 1, -1)
    gd["ints"] = g["ints"].reshape(1, 1, -1, 1, 1)
    gd["intmu"] = g["intmu"].reshape(1, -1, 1, 1, 1)
    gd["intvp"] = g["intvp"].reshape(-1, 1, 1, 1, 1)
    gd["vpgr"] = g["vpgr"].reshape(-1, 1, 1, 1, 1)
    gd["mugr"] = g["mugr"].reshape(1, -1, 1, 1, 1)
    gd["bn"] = g["bn"].reshape(1, 1, -1, 1, 1)
    gd["efun"] = g["efun"].reshape(1, 1, -1, 1, 1)
    gd["rfun"] = g["rfun"].reshape(1, 1, -1, 1, 1)
    gd["bt_frac"] = g["bt_frac"].reshape(1, 1, -1, 1, 1)
    gd["parseval"] = g["parseval"].reshape(1, 1, 1, 1, -1)

    for k in ("mas", "tmp", "d2X", "signz", "signB", "adiabatic", "de",
              "vthrat", "beta", "nlapar", "nlbpar"):
        # torch expand_scalar after squeeze(0): scalar -> (1,1,1,1,1)
        gd[k] = g[k].reshape(1, 1, 1, 1, 1)

    kxrh = g["kxrh"].reshape(1, 1, 1, -1, 1)
    lg = g["little_g"]                                   # (s, 3)
    g0 = lg[:, 0].reshape(1, 1, -1, 1, 1)
    g1 = lg[:, 1].reshape(1, 1, -1, 1, 1)
    g2 = lg[:, 2].reshape(1, 1, -1, 1, 1)
    krloc = np.sqrt(gd["krho"] ** 2 * g0 + 2.0 * gd["krho"] * kxrh * g1 + kxrh ** 2 * g2)
    gd["krloc"] = krloc

    bessel = np.sqrt(2.0 * gd["mugr"] / gd["bn"]) / gd["signz"]
    bessel = gd["mas"] * gd["vthrat"] * krloc * bessel
    gd["bessel"] = _sp.j0(bessel)
    safe_b = np.where(np.abs(bessel) < 1e-8, 1.0, bessel)
    gd["bessel_bpar"] = np.where(np.abs(bessel) < 1e-8, 1.0, 2.0 * _sp.j1(safe_b) / safe_b)

    gamma = gd["mas"] * gd["vthrat"] * krloc
    gamma = 0.5 * (gamma / (gd["signz"] * gd["bn"])) ** 2
    gd["gamma"] = _sp.i0e(gamma)                          # i0(x)*exp(-x) for x>=0
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in gd.items()}


# ================================================================= operators
class TrajOps:
    """All per-trajectory operators. Geometry solved once in __init__ (pure JAX, no torch)."""

    def __init__(self, version, geom, vp_grid, mu_grid, verbose=True):
        assert version in ("real", "semispectral")
        self.version = version
        self.Qv = _axis_basis_stable(vp_grid, H, ENV_V["g20"])         # (32,6)
        self.Qm = _axis_basis_stable(mu_grid, L, ENV_M["e25"])         # (8,3)
        self.Ws, self.Ss = _db4_axis_mats(16)                          # s  16->16
        self.Wx, self.Sx = _db4_axis_mats(85)                          # x/kx 85->87

        self.gd = geom_tensors(geom)
        gd = self.gd
        d3v = gd["ints"] * gd["d2X"] * gd["intmu"] * gd["bn"] * gd["intvp"]
        W = (d3v * (gd["vpgr"] ** 2 + 2.0 * gd["mugr"] * gd["bn"]) * gd["de"] * gd["tmp"]
             * gd["parseval"] * gd["ints"] * gd["efun"] * gd["krho"] * gd["bessel"])
        self.W = jnp.broadcast_to(W, RES).astype(jnp.float32)
        pint = gd["signz"] * gd["de"] * gd["intmu"] * gd["intvp"] * gd["bessel"] * gd["bn"]
        self.pint = jnp.broadcast_to(pint, RES).astype(jnp.float32)

        # non-zonal poisson_diag (flux path)
        pdiag = jnp.sum((gd["signz"] ** 2) * gd["de"] * (gd["gamma"] - 1.0) / gd["tmp"],
                        axis=0, keepdims=True)
        pdiag = pdiag.at[..., 0, 0].set(0.0)
        pdiag = pdiag - gd["adiabatic"]
        pdiag = jnp.where(pdiag == 0.0, 1.0, pdiag)
        self.pdiag = jnp.squeeze(-1.0 / pdiag).astype(jnp.float32)     # (s,kx,ky)

        self._precompute_phi_operator()
        self.phi_atom_energy = self._build_phi_atom_energy()           # (M_TOT,) fp32

        if verbose:
            print(f"[TrajOps {version}] frame {M_TOT} complex atoms", flush=True)

    # -------------------------------------------------- transform (fwd / syn)
    def fwd(self, field_or_K):
        """input -> complex coeff tensor (H,L,16,87,32); ky in last axis."""
        x = field_or_K
        m = jnp.einsum("vh,ul,cvusxy->chlsxy", self.Qv, self.Qm, x)
        m = jnp.moveaxis(jnp.tensordot(m, self.Ws, axes=([3], [1])), -1, 3)
        m = jnp.moveaxis(jnp.tensordot(m, self.Wx, axes=([4], [1])), -1, 4)
        z = (m[0] + 1j * m[1]).astype(jnp.complex64)
        if self.version == "real":
            z = jnp.fft.fft(z, axis=-1, norm="ortho")     # y -> ky
        return z

    def _syn_coeff(self, Z):
        """complex coeff (H,L,16,87,32) -> intermediate before final reduction."""
        if self.version == "real":
            Z = jnp.fft.ifft(Z, axis=-1, norm="ortho")    # ky -> y
        m = jnp.stack([Z.real, Z.imag])
        m = jnp.moveaxis(jnp.tensordot(m, self.Ss, axes=([3], [1])), -1, 3)
        m = jnp.moveaxis(jnp.tensordot(m, self.Sx, axes=([4], [1])), -1, 4)
        return jnp.einsum("vh,ul,chlsxy->cvusxy", self.Qv, self.Qm, m)

    def recon_field(self, Z):
        """complex coeff tensor -> REAL field (2,vp,mu,s,x,y) for scoring (both versions)."""
        out = self._syn_coeff(Z)
        if self.version == "semispectral":
            out = K_to_real(out)
        return out

    # -------------------------------------------------- fast physics (no torch/neugk)
    def to_spec(self, field):
        """real field (2,vp,mu,s,x,y) -> spectral df (vp,mu,s,kx,ky) complex (_df_fft path)."""
        dfc = (field[0] + 1j * field[1]).astype(jnp.complex64)
        spec = jnp.fft.fftn(dfc, axes=(-2, -1), norm="forward")
        return jnp.fft.ifftshift(spec, axes=-2)

    def flux_band_of_field(self, field):
        """reduced flux Q(ky)[band] of a REAL field."""
        dff = self.to_spec(field)
        P = (self.W * dff).sum((0, 1))                                 # (s,kx,ky)
        phi_pre = (self.pint.astype(dff.dtype) * dff).sum((0, 1))
        phi = self.pdiag * phi_pre
        Q = (P.real * (-phi.imag) + P.imag * phi.real).sum((0, 1))     # Im(P conj(phi)) -> (ky,)
        return Q[slice(*FLUX_BAND)]

    def flux_total_of_field(self, field):
        dff = self.to_spec(field)
        P = (self.W * dff).sum((0, 1))
        phi_pre = (self.pint.astype(dff.dtype) * dff).sum((0, 1))
        phi = self.pdiag * phi_pre
        return (P.real * (-phi.imag) + P.imag * phi.real).sum()

    def _precompute_phi_operator(self):
        """Geom-only pieces of the electrostatic phi solve (poisson_diag + zonal matz/maty),
        replicating solve_fields' phi branch."""
        gd = self.gd
        signz, de, tmp, gamma = gd["signz"], gd["de"], gd["tmp"], gd["gamma"]
        ints, adiabatic = gd["ints"], gd["adiabatic"]
        ecf = jnp.exp(-jnp.zeros_like(ints))
        pdiag = jnp.sum(ecf * (signz ** 2) * de * (gamma - 1.0) / tmp, axis=0, keepdims=True)
        pdiag = pdiag.at[..., 0, 0].set(0.0)
        pdiag = pdiag - ecf * adiabatic
        pdiag = jnp.where(pdiag == 0.0, 1.0, pdiag)
        self.pdiag_full = (-1.0 / pdiag).astype(jnp.float32)           # (1,1,s,kx,ky)
        # zonal (single species)
        isz, igam, itmp, ide = signz, gamma, tmp, de
        diagz = isz * (igam - 1.0) * ecf / itmp
        matz = -ints / (isz * ide * (diagz - ecf / itmp))
        matz = matz.at[..., 1:].set(0.0)
        maty = jnp.sum(-matz * ecf, axis=-3, keepdims=True)
        maty = itmp / (ide * ecf) + maty / ecf
        maty = maty.at[..., 0, :].set(1.0)
        maty = jnp.where(maty == 0.0, 1.0, maty)
        maty = 1.0 / maty
        maty = maty.at[..., 1:].set(0.0)
        self.matz = matz.astype(jnp.complex64)
        self.maty = maty.astype(jnp.complex64)
        self.adiabatic_c = adiabatic.astype(jnp.complex64)
        self.pint_c = self.pint.astype(jnp.complex64)

    def solve_phi_spectral(self, dff):
        """Electrostatic phi (spectral, s,kx,ky) from spectral df; matches solve_fields[0]."""
        phi = (self.pint_c * dff).sum((0, 1), keepdims=True)           # (1,1,s,kx,ky)
        bufphi = (self.matz * phi).sum((-3, -1), keepdims=True)
        phi = phi + self.maty * bufphi * self.adiabatic_c
        return (phi * self.pdiag_full.astype(jnp.complex64)).reshape(phi.shape[-3:])

    def _spc_to_phi(self, spc):
        """spectral phi (s,kx,ky) -> real phi (2,x,s,y); _spc_to_phi, real_potens=False."""
        x = jnp.transpose(spc, (1, 0, 2))                              # (x,s,y)
        x = jnp.fft.ifftshift(x, axes=0)
        x = jnp.fft.ifftn(x, axes=(0, 2), norm="forward")
        return jnp.stack([x.real, x.imag])                            # (2,x,s,y)

    def phi_lin(self, field):
        """linear phi operator: real field -> phi vec (matches get_integrals[0])."""
        dff = self.to_spec(field)
        phi_i = self.solve_phi_spectral(dff)
        phi_r = self._spc_to_phi(phi_i)
        return phi_r.reshape(-1)

    # -------------------------------------------------- per-atom phi energy
    def _build_phi_atom_energy(self):
        PItilde = jnp.einsum("vh,ul,vusxy->hlsxy", self.Qv, self.Qm, self.pint)
        AA = (self.pdiag[None, None] * PItilde) ** 2                   # (H,L,s,kx,ky)
        eyeK = jnp.eye(87, dtype=jnp.float32)
        if self.version == "real":
            xreal = self.Sx @ eyeK                                    # (85,87) real-x pattern
            xc = xreal.astype(jnp.complex64)
        else:
            spk = (self.Sx @ eyeK).astype(jnp.complex64)              # (85,87) spectral kx
            spk = jnp.fft.fftshift(spk, axes=0)
            xc = jnp.fft.ifftn(spk, axes=(0,), norm="forward")
        Cx = jnp.fft.fft(xc, axis=0, norm="forward")
        Cx = jnp.fft.ifftshift(Cx, axes=0)
        Cx2 = Cx.real ** 2 + Cx.imag ** 2                             # (85,87)
        Ss2 = self.Ss ** 2                                            # (16,16)
        tmp_kx = jnp.einsum("hlsxy,xw->hlswy", AA, Cx2)               # (H,L,s,xw,ky)
        E = jnp.einsum("hlsAy,sB->hlBAy", tmp_kx, Ss2)                # (H,L,sw,xw,ky)
        return E.reshape(-1).astype(jnp.float32)


# ------------------------------------------------------------------ K-file IO
def K_to_real(K):
    """Exact replica of preprocess.do_ifft: K -> real field (2,vp,mu,s,x,y)."""
    Kc = (K[0] + 1j * K[1]).astype(jnp.complex64)
    Kc = jnp.fft.fftshift(Kc, axes=(-2,))
    F = jnp.fft.ifftn(Kc, axes=(-2, -1), norm="forward")
    return jnp.stack([F.real, F.imag])


_KS = None


def _ks_list(raw=None):
    global _KS
    raw = raw or RAW
    if _KS is None:
        files = os.listdir(raw)
        digit = sorted([f for f in files if f.isdigit()], key=int)
        kf = sorted([f for f in files if f.startswith("K") and not f.endswith(".dat")])
        _KS = kf + digit
    return _KS


def load_K(ts, raw=None):
    """Raw semispectral K (2,vp,mu,s,kx,ky) fp32 (cast fp64->fp32 on load)."""
    raw = raw or RAW
    ff = np.fromfile(os.path.join(raw, _ks_list(raw)[ts]), dtype=np.float64)
    arr = np.reshape(ff, SHAPE, order="F").astype("float32").copy()
    return jnp.asarray(arr)


# ------------------------------------------------------------------ coord table
def frame_coords():
    """(M_TOT, 5) int16 coordinate table for the frame: (h, l, s_wav, kx_wav, ky)."""
    hh, ll, sw, xw, yy = np.unravel_index(np.arange(M_TOT), CSHAPE)
    return np.stack([hh, ll, sw, xw, yy], 1).astype(np.int16)


_FRAME_COORDS = None


def get_frame_coords():
    global _FRAME_COORDS
    if _FRAME_COORDS is None:
        _FRAME_COORDS = frame_coords()
    return _FRAME_COORDS
