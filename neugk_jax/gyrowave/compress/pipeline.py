"""Per-snapshot compression (JAX): full-frame fit -> physics-importance support ->
flux-GN + phi-aware. Pure-JAX port of scratchpad/fast_compress/pipeline.py. NO torch,
NO neugk. All physics is the fastops.TrajOps operator layer.

flux-GN uses jax.jacrev for the exact (quadratic) Jacobian; phi-aware solves the SPD
normal system (R^T R + lam_eff G^T G) c = R^T f + lam_eff G^T phi_gt by CG, with each
matvec / RHS a single jax.grad of a quadratic / linear form (no materialised Jacobian).
lam_eff is a Hutchinson trace ratio (4 Rademacher probes; probes injectable so torch and
JAX can share identical probes in parity checks — defaults to numpy default_rng(7)).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.gyrowave.compress import fastops
from neugk_jax.gyrowave.compress.fastops import CSHAPE, M_TOT


# ------------------------------------------------------------------ recon on a support
def make_recon_S(ops, idx):
    """idx: int frame indices (active support). Returns (recon_fn, n) with
    c = [Re(z_S), Im(z_S)] real (2|S|,) -> REAL field."""
    idx = jnp.asarray(idx)
    n = int(idx.shape[0])

    @jax.jit
    def recon_fn(c):
        z = (c[:n] + 1j * c[n:]).astype(jnp.complex64)
        full = jnp.zeros(M_TOT, dtype=jnp.complex64).at[idx].add(z)
        return ops.recon_field(full.reshape(CSHAPE))

    return recon_fn, n


# ------------------------------------------------------------------ flux-GN (fast)
def gauss_newton_flux(c0, flux_of, q_target_band, n_steps=5, ridge=1e-3):
    """Exact port of core.gauss_newton_flux; flux_of(c)->band is the fast reduced flux."""
    qsc = float(jnp.maximum(jnp.abs(q_target_band).max(), 1e-12))
    Nb = int(q_target_band.shape[0])
    c = jnp.asarray(c0)
    eye = jnp.eye(Nb, dtype=jnp.float32)
    jac = jax.jit(jax.jacrev(flux_of))
    for _ in range(n_steps):
        J = jac(c)
        q = flux_of(c)
        r = (q_target_band - q) / qsc
        Js = (J.reshape(Nb, -1) / qsc).astype(jnp.float32)
        JJ = Js @ Js.T
        lam = ridge * float(jnp.maximum(jnp.mean(jnp.diag(JJ)), 1e-30))
        step = Js.T @ jnp.linalg.solve(JJ + lam * eye, r)
        r0 = float(jnp.linalg.norm(r))
        scale, accepted = 1.0, None
        for _ in range(6):
            c_try = c + scale * step.reshape(c.shape)
            r_try = float(jnp.linalg.norm((q_target_band - flux_of(c_try)) / qsc))
            if r_try < r0:
                accepted = c_try
                break
            scale *= 0.5
        if accepted is None:
            break
        c = accepted
        if float(jnp.linalg.norm(r)) < 1e-5:
            break
    return c


# ------------------------------------------------------------------ phi-aware (fast)
def _rademacher_probes(K, n=4, seed=7, probes=None):
    if probes is not None:
        return [jnp.asarray(p, dtype=jnp.float32) for p in probes]
    rng = np.random.default_rng(seed)
    return [jnp.asarray(rng.integers(0, 2, size=K).astype(np.float32) * 2.0 - 1.0)
            for _ in range(n)]


def phi_aware_fit(c0, recon_fn, phi_of, gt_field, lam=1.0, iters=40, tol=1e-8, probes=None):
    """Solve (R^T R + lam_eff G^T G) c = R^T f + lam_eff G^T phi_gt by CG (SPD, exact)."""
    def recon_c(c):
        return recon_fn(c).reshape(-1)

    def phi_c(c):
        return phi_of(recon_fn(c))

    phi_t = phi_of(gt_field)
    f_flat = gt_field.reshape(-1)
    c_flat = jnp.asarray(c0).reshape(-1)
    K = int(c_flat.shape[0])

    rc = jax.jit(recon_c)
    pc = jax.jit(phi_c)
    tr_r, tr_j = 0.0, 0.0
    for z in _rademacher_probes(K, 4, probes=probes):
        tr_r += float(jnp.sum(rc(z) ** 2))
        tr_j += float(jnp.sum(pc(z) ** 2))
    lam_eff = lam * tr_r / max(tr_j, 1e-30)

    @jax.jit
    def A_mv(v):
        def quad(vv):
            rec = recon_fn(vv)
            pv = phi_of(rec)
            r = rec.reshape(-1)
            return 0.5 * jnp.sum(r * r) + 0.5 * lam_eff * jnp.sum(pv * pv)
        return jax.grad(quad)(v)

    @jax.jit
    def rhs(cc):
        def lin(v):
            rec = recon_fn(v)
            pv = phi_of(rec)
            return jnp.sum(rec.reshape(-1) * f_flat) + lam_eff * jnp.sum(pv * phi_t)
        return jax.grad(lin)(cc)

    b = rhs(c_flat)
    x = c_flat
    r = b - A_mv(x)
    p = r
    rs = float(r @ r)
    rs0 = rs
    for _ in range(iters):
        Ap = A_mv(p)
        alpha = rs / float(p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        if rs_new < tol * rs0:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x.reshape(jnp.asarray(c0).shape), lam_eff


# ------------------------------------------------------------------ full-frame flux gradient
def fullframe_flux_grad(ops, zre, zim):
    """g = dQ_total/d(coeff) over the FULL frame (real & imag parts)."""
    def Q(zr, zi):
        Z = (zr + 1j * zi).reshape(CSHAPE)
        return ops.flux_total_of_field(ops.recon_field(Z))
    gre, gim = jax.grad(Q, argnums=(0, 1))(zre, zim)
    return gre, gim


def support_flux_grad(ops, idx, zre_full, zim_full):
    """Flux gradient over the full frame given only `idx` coeffs are active."""
    zre = jnp.zeros_like(zre_full).at[idx].set(zre_full[idx])
    zim = jnp.zeros_like(zim_full).at[idx].set(zim_full[idx])
    return fullframe_flux_grad(ops, zre, zim)


# ------------------------------------------------------------------ support selection
def _norm01(x):
    return x / jnp.maximum(x.max(), 1e-30)


def select_support(ops, c_full, budget_Kc, weights=(1.0, 1.0, 1.0),
                   n_pool_mult=2.0, n_refine=1):
    """Physics-importance per-snapshot support (frame indices). weights = (a_df, b_phi, g_flux).

    The phi and flux importance terms are skipped when their weight is 0 (exact — the term is
    multiplied by 0 either way), so amplitude-only / standard support (b=g=0) avoids the
    full-frame flux gradient and the refine pass entirely. b=g>0 behaviour is unchanged."""
    a, b, gmm = weights
    zre = c_full.real.reshape(-1)
    zim = c_full.imag.reshape(-1)
    amp = jnp.sqrt(zre ** 2 + zim ** 2)
    score = a * _norm01(amp)
    phi_term = b * _norm01(amp * jnp.sqrt(ops.phi_atom_energy)) if b else 0.0
    score = score + phi_term

    if not gmm:                                   # no flux importance -> pure amplitude/phi ranking
        return jnp.argsort(-score)[:budget_Kc]

    gre, gim = fullframe_flux_grad(ops, zre, zim)
    flux_imp = jnp.abs(gre * zre + gim * zim)
    score = score + gmm * _norm01(flux_imp)
    pool = jnp.argsort(-score)[:int(n_pool_mult * budget_Kc)]

    idx = pool[:budget_Kc]
    for _ in range(n_refine):
        gre, gim = support_flux_grad(ops, idx, zre, zim)
        flux_imp = jnp.abs(gre * zre + gim * zim)
        score = a * _norm01(amp) + phi_term + gmm * _norm01(flux_imp)
        ps = score[pool]
        order = jnp.argsort(-ps)
        idx = pool[order[:budget_Kc]]
    return idx


# ------------------------------------------------------------------ scoring
def fidelity(ops, pred_field, gt_field, gt_flux_full):
    peak = jnp.maximum(jnp.abs(gt_field).max(), 1e-30)
    mse = jnp.maximum(((pred_field - gt_field) ** 2).mean(), 1e-30)
    df_psnr = float(10.0 * jnp.log10(peak ** 2 / mse))
    phi_p = ops.phi_lin(pred_field)
    phi_g = ops.phi_lin(gt_field)
    ppk = jnp.maximum(jnp.abs(phi_g).max(), 1e-30)
    pmse = jnp.maximum(((phi_p - phi_g) ** 2).mean(), 1e-30)
    phi_psnr = float(10.0 * jnp.log10(ppk ** 2 / pmse))
    dff = ops.to_spec(pred_field)
    P = (ops.W * dff).sum((0, 1))
    phi_pre = (ops.pint.astype(dff.dtype) * dff).sum((0, 1))
    phi = ops.pdiag * phi_pre
    fp = float((P.real * (-phi.imag) + P.imag * phi.real).sum())
    phi_p_sp = ops.solve_phi_spectral(dff)
    zp = phi_p_sp[:, :, 0]
    dffg = ops.to_spec(gt_field)
    phi_g_sp = ops.solve_phi_spectral(dffg)
    zg = phi_g_sp[:, :, 0]
    ez_p = float((zp.real ** 2 + zp.imag ** 2).sum())
    ez_g = float((zg.real ** 2 + zg.imag ** 2).sum())
    return {"df_psnr": df_psnr, "phi_psnr": phi_psnr, "flux_pred": fp,
            "flux_gt": gt_flux_full,
            "flux_relerr": abs(fp - gt_flux_full) / (abs(gt_flux_full) + 1e-30),
            "zonal_phi_ratio": ez_p / (ez_g + 1e-30)}


# ------------------------------------------------------------------ one snapshot
def process_snapshot(ops, raw_input, budget_Kc, weights, n_pool_mult=2.0, n_refine=1,
                     gn_steps=5, phi_iters=40, do_phi=True, support_idx=None, probes=None):
    """raw_input: real field (version=real) or K (semispectral). Returns (tokens, fid, timings)."""
    t = {}
    field = raw_input if ops.version == "real" else fastops.K_to_real(raw_input)

    t0 = time.time()
    c_full = ops.fwd(raw_input)
    c_full.block_until_ready()
    t["proj"] = time.time() - t0

    qt = ops.flux_band_of_field(field)
    dff = ops.to_spec(field)
    Pg = (ops.W * dff).sum((0, 1))
    phig = ops.pdiag * (ops.pint.astype(dff.dtype) * dff).sum((0, 1))
    gt_flux_full = float((Pg.real * (-phig.imag) + Pg.imag * phig.real).sum())

    t0 = time.time()
    if support_idx is None:
        idx = select_support(ops, c_full, budget_Kc, weights, n_pool_mult, n_refine)
    else:
        idx = jnp.asarray(support_idx)
    idx.block_until_ready()
    t["select"] = time.time() - t0

    recon_fn, n = make_recon_S(ops, idx)
    zf = c_full.reshape(-1)
    c0 = jnp.concatenate([zf.real[idx], zf.imag[idx]])

    def flux_of(c):
        return ops.flux_band_of_field(recon_fn(c))

    lam_eff = 0.0
    t0 = time.time()
    if do_phi:
        cp, lam_eff = phi_aware_fit(c0, recon_fn, ops.phi_lin, field, iters=phi_iters,
                                    probes=probes)
    else:
        cp = c0
    cp.block_until_ready()
    t["phi"] = time.time() - t0

    t0 = time.time()
    cg = gauss_newton_flux(cp, flux_of, qt, n_steps=gn_steps)
    cg.block_until_ready()
    t["gn"] = time.time() - t0

    rec = recon_fn(cg)
    fid = fidelity(ops, rec, field, gt_flux_full)

    cg_np = np.asarray(cg)
    tokens = {"idx": np.asarray(idx).astype(np.int64),
              "re": cg_np[:n].astype(np.float32),
              "im": cg_np[n:].astype(np.float32)}
    fid["lam_eff"] = float(lam_eff)
    return tokens, fid, t
