"""Noise→target atom pairing for splat flow matching.

Independent x0↔x1 pairing makes the model regress the average over all
matchings — the classic blurred-set failure. These couplings re-order the
noise atoms so each target atom trains against the closest noise position,
always WITHIN a population (envelope atoms and each carrier bin separately —
an envelope atom never matches a carrier atom; the zf-stats token never moves).

Modes (measured mean-squared 5D transport cost on real latents, envelope
population n=750, as % of the independent→hungarian gap closed):

``morton``      z-order curve rank-match after per-axis quantile alignment —
                jittable, deterministic given the sets, ~67% gap closed. Default.
``sinkhorn``    log-domain entropic OT + greedy rounding — jittable, ~80%
                closed, O(n²·iters) per population.
``sliced``      exact 1-D OT along a random direction (rank-match after
                projection) — cheapest stochastic option, ~29% closed.
``hungarian``   exact assignment via scipy host callback — reference (~n³).
``independent`` identity (ablation baseline).

Blending with an independent coupling follows either the NSOT noise
perturbation (``blend="perturb"``: x0' = √(1−β)·x0 + √β·ε with β = 1 −
ot_fraction — keeps the assignment on every sample, softens the early field,
preserves the N(0,I) marginal) or a per-sample Bernoulli mixing of the two
couplings (``blend="mix"``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.gyrosplats.splat import MU_SL

PAIRING_MODES = ("morton", "sinkhorn", "sliced", "hungarian", "independent")


def _populations(bins: np.ndarray) -> list[np.ndarray]:
    """Static per-population index lists: envelope + one per carrier bin."""
    return [np.where(bins == m)[0] for m in np.unique(bins)]


def _rank_match_perm(k0, k1, pops, n, n_extra):
    """Permutation p with x0[p] rank-matched to x1 on scalar keys, per population."""
    perm = jnp.arange(n + n_extra)  # trailing extra tokens (e.g. stats) stay put
    for idx in pops:
        idx = jnp.asarray(idx)
        order0 = idx[jnp.argsort(k0[idx])]
        order1 = idx[jnp.argsort(k1[idx])]
        perm = perm.at[order1].set(order0)
    return perm


def _sliced_perm(key, mu0, mu1, pops, n_extra):
    u = jr.normal(key, (mu0.shape[1],))
    u = u / (jnp.linalg.norm(u) + 1e-8)
    return _rank_match_perm(mu0 @ u, mu1 @ u, pops, mu0.shape[0], n_extra)


def _morton_codes(mu, lo, hi, bits):
    """Fixed-point quantize each axis and interleave bits (z-order index)."""
    d = mu.shape[1]
    assert bits * d < 31, "morton code must fit int32"
    q = jnp.clip(
        ((mu - lo) / (hi - lo) * (1 << bits)).astype(jnp.int32), 0, (1 << bits) - 1
    )
    code = jnp.zeros(mu.shape[0], dtype=jnp.int32)
    for b in range(bits):
        for a in range(d):
            code = code | (((q[:, a] >> b) & 1) << (b * d + a))
    return code


def _morton_perm(mu0, mu1, pops, n_extra, bits=6):
    """Z-order rank-match; x0 axes are first quantile-aligned onto x1's marginals
    per population so both sides see comparable codes (n(0,1) vs [-1,1] data)."""
    n = mu0.shape[0]
    perm = jnp.arange(n + n_extra)
    for idx in pops:
        idx = jnp.asarray(idx)
        m0, m1 = mu0[idx], mu1[idx]
        # per-axis quantile alignment: sorted x1 values at the ranks of x0
        ranks = jnp.argsort(jnp.argsort(m0, axis=0), axis=0)
        m0q = jnp.take_along_axis(jnp.sort(m1, axis=0), ranks, axis=0)
        lo, hi = m1.min(0) - 1e-6, m1.max(0) + 1e-6
        c0 = _morton_codes(m0q, lo, hi, bits)
        c1 = _morton_codes(m1, lo, hi, bits)
        order0 = idx[jnp.argsort(c0)]
        order1 = idx[jnp.argsort(c1)]
        perm = perm.at[order1].set(order0)
    return perm


def _sinkhorn_round(logp):
    """Greedy rounding of a log-coupling to a permutation (scan over targets)."""
    n = logp.shape[0]

    def step(used, j):
        col = jnp.where(used, -jnp.inf, logp[:, j])
        i = jnp.argmax(col)
        return used.at[i].set(True), i

    _, rows = jax.lax.scan(step, jnp.zeros(n, bool), jnp.arange(n))
    return rows  # rows[j] = source index matched to target j


def _sinkhorn_perm(mu0, mu1, pops, n_extra, eps=0.05, iters=100):
    """Entropic OT (log-domain, uniform marginals) + greedy rounding, per population."""
    n = mu0.shape[0]
    perm = jnp.arange(n + n_extra)
    for idx in pops:
        idx = jnp.asarray(idx)
        m = idx.shape[0]
        cost = jnp.sum((mu0[idx][:, None] - mu1[idx][None]) ** 2, axis=-1)
        loga = -jnp.log(m)

        def it(_, fg):
            f, g = fg
            f = eps * (loga - jax.nn.logsumexp((g[None] - cost) / eps, axis=1))
            g = eps * (loga - jax.nn.logsumexp((f[:, None] - cost) / eps, axis=0))
            return f, g

        f, g = jax.lax.fori_loop(0, iters, it, (jnp.zeros(m), jnp.zeros(m)))
        rows = _sinkhorn_round((f[:, None] + g[None] - cost) / eps)
        perm = perm.at[idx].set(idx[rows])
    return perm


def _hungarian_perm_batch(mu0_b, mu1_b, pops, n_extra):
    """Exact within-population assignment, whole batch in one host callback.

    scipy's lap solver releases the gil, so a thread pool makes the batch cost
    ~one solve's wall time. squared-euclidean cost (the w2-correct choice for
    straight fm paths).
    """

    def host(m0, m1):
        from concurrent.futures import ThreadPoolExecutor

        import scipy.optimize

        bs, n = m0.shape[0], m0.shape[1]
        out = np.tile(np.arange(n + n_extra, dtype=np.int32), (bs, 1))

        def solve(b):
            for idx in pops:
                cost = ((m0[b][idx][:, None] - m1[b][idx][None]) ** 2).sum(-1)
                row, col = scipy.optimize.linear_sum_assignment(cost)
                # x0[idx[row[k]]] is the match for x1[idx[col[k]]]
                out[b][idx[col]] = idx[row]

        with ThreadPoolExecutor(max_workers=min(bs, 32)) as ex:
            list(ex.map(solve, range(bs)))
        return out

    bs, n = mu0_b.shape[0], mu0_b.shape[1]
    shape = jax.ShapeDtypeStruct((bs, n + n_extra), jnp.int32)
    return jax.pure_callback(host, shape, mu0_b, mu1_b)


def make_pair_fn(
    bins: np.ndarray,
    mode: str = "morton",
    ot_fraction: float = 0.8,
    blend: str = "perturb",
    morton_bits: int = 6,
    sinkhorn_eps: float = 0.05,
    sinkhorn_iters: int = 100,
    tie_groups: bool = False,
    n_extra_tokens: int = 0,
):
    """Batch-level ``pair_fn(key, x0, x1) -> x0`` for ``fm_forward_loss``.

    ``bins`` (n_atoms,) carrier modes define the populations; tokens are
    (n_atoms + 1, c) with the trailing zf-stats token excluded from pairing.
    ``tie_groups`` broadcasts each tied group's envelope-channel noise to all
    its rows AFTER pairing/blending, so the interpolant stays exactly tied
    (pairs with the tied-velocity group head in ``GyrosplatDiT``).
    """
    assert mode in PAIRING_MODES, mode
    assert blend in ("perturb", "mix")
    grp_idx = None
    if tie_groups:
        from neugk_jax.gyrosplats.splat import bank_structure, tie_group_channels

        _, grp_idx_np = bank_structure(bins)
        grp_idx = jnp.asarray(grp_idx_np)

    if mode == "independent":
        if grp_idx is None:
            return None

        def tie_only(key, x0, x1):
            return jax.vmap(lambda xb: tie_group_channels(xb, grp_idx))(x0)

        return tie_only
    pops = _populations(np.asarray(bins))

    ne = n_extra_tokens

    def _perm_one(key, mu0, mu1):
        if mode == "sliced":
            return _sliced_perm(key, mu0, mu1, pops, ne)
        if mode == "morton":
            return _morton_perm(mu0, mu1, pops, ne, bits=morton_bits)
        return _sinkhorn_perm(mu0, mu1, pops, ne, eps=sinkhorn_eps, iters=sinkhorn_iters)

    def _blend_one(key, x0, paired):
        if blend == "perturb":
            # nsot: keep the assignment, soften with fresh noise (marginal stays n(0,i))
            beta = 1.0 - ot_fraction
            eps_noise = jr.normal(key, x0.shape, x0.dtype)
            return jnp.sqrt(1.0 - beta) * paired + jnp.sqrt(beta) * eps_noise
        use = jr.bernoulli(key, p=ot_fraction)
        return jnp.where(use, paired, x0)

    def pair_fn(key, x0, x1):
        k_perm, k_mix = jr.split(key)
        atoms = slice(None, -ne) if ne else slice(None)
        mu0 = x0[:, atoms, MU_SL]
        mu1 = x1[:, atoms, MU_SL]
        if mode == "hungarian":
            perms = _hungarian_perm_batch(mu0, mu1, pops, ne)
        else:
            perms = jax.vmap(_perm_one)(jr.split(k_perm, x0.shape[0]), mu0, mu1)
        paired = jax.vmap(lambda xb, pb: xb[pb])(x0, perms)
        out = jax.vmap(_blend_one)(jr.split(k_mix, x0.shape[0]), x0, paired)
        if grp_idx is not None:
            from neugk_jax.gyrosplats.splat import tie_group_channels

            out = jax.vmap(lambda xb: tie_group_channels(xb, grp_idx))(out)
        return out

    return pair_fn
