"""PaiNN patch embedding / unpatch: one PaiNN interaction between a token and the points of its patch.

The points are the neighbours of the token (Schütt et al. 2021, the ``PaiNNInteraction`` /
``PaiNNMixing`` blocks of schnetpack). Configuration space is PaiNN's geometric space: a point sits at
its cell-centred offset ``r`` from the patch centre (the relative axes), with distance ``|r|`` and
direction ``r / |r|``. Velocity space and the channel are the point species: with the field value
they give the point's scalar embedding. Messages are ``phi(q) * (Linear(sinc rbf(|r|)) * cutoff)``,
split into a scalar part and the radial / vector parts of the equivariant message. Two deviations
keep a patch resolution independent: the sum over neighbours is a quadrature mean, and the cutoff
radius lies beyond the patch corners.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.patching.points import PointGrid, encoded_index
from neugk_jax.models.utils import MLP, Linear, silu


def sinc_rbf(d: jnp.ndarray, n: int, cutoff: float) -> jnp.ndarray:
    """``sin(k pi d / cutoff) / d`` for k = 1..n (PaiNN's radial basis), finite at d = 0."""
    k = jnp.arange(1, n + 1) * jnp.pi / cutoff
    safe = jnp.where(d > 0, d, 1.0)[..., None]
    return jnp.where(d[..., None] > 0, jnp.sin(k * safe) / safe, k)


def cosine_cutoff(d: jnp.ndarray, cutoff: float) -> jnp.ndarray:
    return 0.5 * (jnp.cos(jnp.pi * d / cutoff) + 1.0) * (d < cutoff)


class PaiNNMixing(eqx.Module):
    """PaiNN update: scalars and vectors exchange information through the norm and inner product of mixed vectors."""

    intra: MLP
    mu_mix: Linear

    def __init__(self, features: int, *, key):
        k1, k2 = jr.split(key)
        self.intra = MLP([2 * features, features, 3 * features], key=k1, act_fn=silu)
        self.mu_mix = Linear(features, 2 * features, key=k2, use_bias=False)

    def __call__(self, q: jnp.ndarray, mu: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # q (..., F), mu (..., D, F)
        mu_v, mu_w = jnp.split(self.mu_mix(mu), 2, axis=-1)
        mu_vn = jnp.sqrt(jnp.sum(mu_v**2, axis=-2) + 1e-8)
        dq, dmu, dqmu = jnp.split(self.intra(jnp.concatenate([q, mu_vn], -1)), 3, axis=-1)
        return q + dq + dqmu * jnp.sum(mu_v * mu_w, axis=-2), mu + dmu[..., None, :] * mu_w


class _Geometry(eqx.Module):
    """Point geometry and species of a patch: offsets on the relative axes, the rbf filters' inputs."""

    filter_net: Linear
    species: Linear
    features: int = eqx.field(static=True)
    n_rbf: int = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    species_index: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        grid: PointGrid,
        features: int,
        n_rbf: int,
        modes: int,
        n_cond: int,
        value_in: bool,
        *,
        key,
    ):
        n_rel, n = len(grid.rel_axes), grid.n_coords
        e = n * modes
        n_c = grid.n_channels
        # species: encoded absolute / folded coordinates, channel, patch scale, conditioning
        self.species_index = tuple(encoded_index(range(n_rel, n), modes)) + tuple(
            range(e, e + n_c + n_rel + n_cond)
        )
        self.features, self.n_rbf = features, n_rbf
        self.cutoff = math.sqrt(max(n_rel, 1)) + 0.5
        k1, k2 = jr.split(key)
        self.filter_net = Linear(n_rbf, 3 * features, key=k1)
        self.species = Linear(len(self.species_index) + int(value_in), features, key=k2)

    def __call__(self, grid: PointGrid, feats: jnp.ndarray):
        """Filters ``(P, 3F)``, directions ``(P, D)`` and species ``(*T_abs, P, S)`` of the points."""
        r = grid.offsets
        d = jnp.sqrt(jnp.sum(r**2, -1))
        direction = jnp.where(d[:, None] > 0, r / jnp.where(d > 0, d, 1.0)[:, None], 0.0)
        w = (
            self.filter_net(sinc_rbf(d, self.n_rbf, self.cutoff))
            * cosine_cutoff(d, self.cutoff)[:, None]
        )
        return w, direction, feats[..., jnp.asarray(self.species_index)]


def _chunked(fn, n_tok: int, size: int, *arrays):
    """``fn`` over chunks of ``size`` tokens (the first ``n_tok`` axes flattened), rematerialized per chunk."""
    lead = arrays[0].shape[:n_tok]
    n = math.prod(lead)
    k = -(-n // size)

    def chunks(a):
        a = a.reshape(n, *a.shape[n_tok:])
        a = jnp.pad(a, [(0, k * size - n)] + [(0, 0)] * (a.ndim - 1))
        return a.reshape(k, size, *a.shape[1:])

    out = jax.lax.map(jax.checkpoint(lambda a: fn(*a)), tuple(chunks(a) for a in arrays))
    return jax.tree_util.tree_map(
        lambda o: o.reshape(k * size, *o.shape[2:])[:n].reshape(*lead, *o.shape[2:]), out
    )


def token_width(grid: PointGrid, opts) -> int:
    """Width of the decoder's token state: F scalars and F vectors of the relative axes."""
    return opts["painn_point_features"] * (1 + len(grid.rel_axes))


class PaiNNEncoder(eqx.Module):
    """Points to token: the points' messages to a virtual atom at the patch centre, then a PaiNN update."""

    geometry: _Geometry
    context: MLP
    mixing: PaiNNMixing
    q0: jnp.ndarray
    chunk: int = eqx.field(static=True)

    def __init__(self, grid: PointGrid, opts, *, key):
        f = opts["painn_features"]
        k1, k2, k3 = jr.split(key, 3)
        self.geometry = _Geometry(
            grid, f, opts["n_rbf"], opts["modes"], opts["cond_features"], True, key=k1
        )
        self.context = MLP([f, f, 3 * f], key=k2, act_fn=silu)
        self.mixing = PaiNNMixing(f, key=k3)
        self.chunk = opts["painn_chunk"]
        self.q0 = jnp.zeros((f,))

    def __call__(self, patches: jnp.ndarray, grid: PointGrid, feats: jnp.ndarray) -> jnp.ndarray:
        """``(*T, F)`` token scalars of the folded patches ``(*T, P)``; ``feats`` the point features."""
        w, direction, species = self.geometry(grid, feats)
        quad = grid.weight / grid.weight.shape[-1]
        abs_first = 0 in grid.abs_axes
        n_tok = patches.ndim - 1

        f = self.geometry.features
        lin, (l1, l2) = self.geometry.species.inner, self.context.layers
        w2, b2 = l2.inner.weight, l2.inner.bias
        wq, wr, _ = jnp.split(w, 3, axis=-1)

        def messages(x, sp, qw):
            # x (n, P) values, sp (P, S) species, qw (P,) quadrature weights
            # q_j = s_j + a x_j and the first context layer act elementwise; the second is
            # contracted after the sum over the points, as one matmul per token
            s = sp @ lin.weight[:, :-1].T + lin.bias
            h = silu(l1(s) + x[..., None] * (l1.inner.weight @ lin.weight[:, -1]))
            gq = qw[:, None] * wq
            gr = jnp.einsum("pc,pd->pdc", qw[:, None] * wr, direction)
            dq = jnp.einsum("cf,pc,npf->nc", w2[:f], gq, h, optimize="optimal")
            dmu = jnp.einsum("cf,pdc,npf->ndc", w2[f : 2 * f], gr, h, optimize="optimal")
            return dq + gq.sum(0) * b2[:f], dmu + gr.sum(0) * b2[f : 2 * f]

        if abs_first and n_tok >= 1:
            # one absolute-axis row at a time (a vmap would batch the chunks of all rows)
            dq, dmu = jax.lax.map(
                lambda r: _chunked(
                    lambda xx: messages(xx, r[1], r[2]), r[0].ndim - 1, self.chunk, r[0]
                ),
                (patches, species, quad),
            )
        else:
            dq, dmu = _chunked(lambda xx: messages(xx, species, quad), n_tok, self.chunk, patches)
        q, _ = self.mixing(self.q0 + dq, dmu)
        return q


class PaiNNDecoder(eqx.Module):
    """Token to points: the token's message to every point, added to the point's species embedding, a PaiNN update and a linear readout.

    Every point runs the update block, so its width is ``painn_point_features``.
    """

    geometry: _Geometry
    context: MLP
    mixing: PaiNNMixing
    readout: Linear
    chunk: int = eqx.field(static=True)

    def __init__(self, grid: PointGrid, opts, *, key):
        f = opts["painn_point_features"]
        k1, k2, k3, k4 = jr.split(key, 4)
        self.geometry = _Geometry(
            grid, f, opts["n_rbf"], opts["modes"], opts["cond_features"], False, key=k1
        )
        self.context = MLP([f, f, 3 * f], key=k2, act_fn=silu)
        self.mixing = PaiNNMixing(f, key=k3)
        self.chunk = opts["painn_chunk"]
        readout = Linear(f, 1, key=k4)
        self.readout = eqx.tree_at(
            lambda m: (m.inner.weight, m.inner.bias), readout, (jnp.zeros((1, f)), jnp.zeros((1,)))
        )

    def __call__(self, tokens: jnp.ndarray, grid: PointGrid, feats: jnp.ndarray) -> jnp.ndarray:
        """``(*T, P)`` point values of the token states ``(*T, F + D F)`` (scalars, then vectors)."""
        w, direction, species = self.geometry(grid, feats)
        f, n_dir = self.geometry.features, direction.shape[-1]
        q_t, mu_t = tokens[..., :f], tokens[..., f:].reshape(*tokens.shape[:-1], n_dir, f)
        q0 = self.geometry.species(species)
        abs_first = 0 in grid.abs_axes

        def points(qt, mut, q0p):
            # qt (..., F), mut (..., D, F), q0p (P, F)
            dq, dmu_r, dmu_mu = jnp.split(self.context(qt)[..., None, :] * w, 3, axis=-1)
            q = q0p + dq
            mu = (
                dmu_r[..., :, None, :] * direction[:, :, None]
                + dmu_mu[..., :, None, :] * mut[..., None, :, :]
            )
            q, _ = self.mixing(q, mu)
            return self.readout(q)[..., 0]

        if abs_first and q_t.ndim >= 2:
            return jax.lax.map(
                lambda r: _chunked(
                    lambda x, y: points(x, y, r[2]), r[0].ndim - 1, self.chunk, r[0], r[1]
                ),
                (q_t, mu_t, q0),
            )
        return _chunked(lambda x, y: points(x, y, q0), q_t.ndim - 1, self.chunk, q_t, mu_t)
