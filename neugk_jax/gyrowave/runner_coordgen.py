"""gyrowave coord+value generation runner (Approach B).

Diffuses the FULL token z = [coord_norm (5), value_white (2)]; the model (WaveletCoordDiT)
generates support AND coefficients at fixed N. Rectified-flow loss lives HERE (the shared
fm_forward_loss uses a gaussian prior for x0 and has no coord-uniform / coord-pairing hook),
so flow_matching.py is untouched; the shared euler_sample IS reused (its prior_fn hook feeds
the uniform-coord / gaussian-value source).

Prior x0: coord ~ Uniform[0,1)^5 (bounded lattice, not gaussian); value ~ N(0, I).
Pairing (x0<->x1 on coord distance, config ``pairing``): 'sliced' (rank-match on a random
projection, jittable, DEFAULT), 'independent' (none), 'sinkhorn_sub' (entropic OT on a random
subsample <= sinkhorn_sub_m, argmax-rounded). Dense Sinkhorn on N~=27852 is infeasible
(N^2 ~ 7.7e8) so 'sinkhorn_sub' subsamples — see FIXED_LENGTH_PLAN.md fork #1.
"""
from __future__ import annotations

import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx, optax

from neugk_jax.diffusion.flow_matching import euler_sample, sample_time
from neugk_jax.gyrowave.runner import GyrowaveFMRunner
from neugk_jax.gyrowave.model_coordgen import WaveletCoordDiT


def _cg_model_fn(model):
    # coordgen cond = physical params (n_cond,) only; coords are part of the diffused state.
    return lambda z, t, c: model(z, t, c)


def _sliced_perm(key, c0, c1):
    """Rank-match perm on a single random projection of the coords: x0[perm][i] pairs x1[i]."""
    u = jr.normal(key, (c0.shape[1],))
    u = u / (jnp.linalg.norm(u) + 1e-8)
    o0 = jnp.argsort(c0 @ u)
    o1 = jnp.argsort(c1 @ u)
    return jnp.arange(c0.shape[0]).at[o1].set(o0)


def _sinkhorn_sub_perm(key, c0, c1, m, eps, iters):
    """Entropic OT on a random size-``m`` subsample (m static); rest stay identity.
    Greedy masked-argmax rounding of the log-coupling -> a valid within-subset permutation."""
    N = c0.shape[0]
    sub = jr.choice(key, N, (m,), replace=False)
    a, b = c0[sub], c1[sub]
    cost = jnp.sum((a[:, None] - b[None]) ** 2, axis=-1)          # (m, m)
    loga = -jnp.log(m)

    def it(_, fg):
        f, g = fg
        f = eps * (loga - jax.nn.logsumexp((g[None] - cost) / eps, axis=1))
        g = eps * (loga - jax.nn.logsumexp((f[:, None] - cost) / eps, axis=0))
        return f, g

    f, g = jax.lax.fori_loop(0, iters, it, (jnp.zeros(m), jnp.zeros(m)))
    logp = (f[:, None] + g[None] - cost) / eps

    def rnd(used, j):                                            # pick best unused source per target
        i = jnp.argmax(jnp.where(used, -jnp.inf, logp[:, j]))
        return used.at[i].set(True), i
    _, rows = jax.lax.scan(rnd, jnp.zeros(m, bool), jnp.arange(m))
    return jnp.arange(N).at[sub].set(sub[rows])


class GyrowaveCoordGenRunner(GyrowaveFMRunner):
    """Flow matching over the full (coord, value) token; support is generated, not conditioned."""

    def setup_data(self) -> None:
        super().setup_data()                                     # X (S,N,2) whitened vals, C (S,N,5) coords_n
        self.n_coord, self.val_dim = 5, 2
        self.x1 = jnp.concatenate([self.C, self.X], axis=-1)     # (S,N,7) = [coord_n | value_white]
        self.cshape_np = np.asarray(self.nrm["cshape"])          # (5,) int

    def setup_components(self) -> None:
        cfg, m = self.cfg, self.cfg.model
        self.model = WaveletCoordDiT(
            val_dim=self.val_dim, n_coord=self.n_coord, n_cond=self.n_cond,
            n_hidden=getattr(m, "n_hidden", 256), n_layers=getattr(m, "n_layers", 8),
            n_head=getattr(m, "n_head", 8), mlp_ratio=getattr(m, "mlp_ratio", 2),
            bands=getattr(m, "bands", 6), slice_num=getattr(m, "slice_num", 512),
            key=jr.PRNGKey(getattr(cfg, "seed", 0)))
        # pairing / loss config
        t = cfg.training
        self.pairing = getattr(t, "pairing", "sliced")
        self.ot_fraction = float(getattr(t, "ot_fraction", 1.0))
        self.coord_w = float(getattr(t, "coord_loss_w", 1.0))
        self.value_w = float(getattr(t, "value_loss_w", 1.0))
        self.sinkhorn_m = int(min(getattr(t, "sinkhorn_sub_m", 2048), self.N))
        self.sinkhorn_eps = float(getattr(t, "sinkhorn_eps", 0.05))
        self.sinkhorn_iters = int(getattr(t, "sinkhorn_iters", 100))
        # optimizer (same recipe as the parent, rebuilt because the model differs)
        self.steps_per_epoch = max(1, self.S // t.batch_size)
        total = t.n_epochs * self.steps_per_epoch
        peak = t.learning_rate
        self.schedule = optax.warmup_cosine_decay_schedule(
            0.0, peak, min(500, max(1, total // 20)), total,
            getattr(t, "final_learning_rate", peak * 1e-2))
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(getattr(t, "clip_to", 1.0)),
            optax.adamw(self.schedule, weight_decay=getattr(t, "weight_decay", 1e-4)))
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

    def _sample_x0(self, key, shape):
        # shape (B, N, 7): coords uniform[0,1)^5, values N(0,1)
        kc, kv = jr.split(key)
        c = jr.uniform(kc, (*shape[:-1], self.n_coord), dtype=jnp.float32)
        v = jr.normal(kv, (*shape[:-1], self.val_dim), dtype=jnp.float32)
        return jnp.concatenate([c, v], axis=-1)

    def _pair(self, key, x0, x1):
        if self.pairing == "independent":
            return x0
        kp, kmix = jr.split(key)

        def perm_one(k, a, b):
            c0, c1 = a[:, :self.n_coord], b[:, :self.n_coord]
            if self.pairing == "sliced":
                return _sliced_perm(k, c0, c1)
            return _sinkhorn_sub_perm(k, c0, c1, self.sinkhorn_m,
                                      self.sinkhorn_eps, self.sinkhorn_iters)

        perms = jax.vmap(perm_one)(jr.split(kp, x0.shape[0]), x0, x1)
        paired = jax.vmap(lambda xb, pb: xb[pb])(x0, perms)
        if self.ot_fraction < 1.0:                               # per-sample mix (keeps marginal)
            use = jr.bernoulli(kmix, self.ot_fraction, (x0.shape[0], 1, 1))
            paired = jnp.where(use, paired, x0)
        return paired

    @eqx.filter_jit
    def _train_step(self, model, opt_state, x1b, cb, key):
        def loss_fn(m):
            kp, kt, kx0 = jr.split(key, 3)
            x0 = self._pair(kp, self._sample_x0(kx0, x1b.shape), x1b)
            t = sample_time(kt, x1b.shape[0], dtype=x1b.dtype)
            tb = t.reshape(-1, 1, 1)
            xt = (1.0 - tb) * x0 + tb * x1b
            target = x1b - x0                                    # rectified-flow velocity
            pred = jax.vmap(_cg_model_fn(m))(xt, t, cb)
            w = jnp.concatenate([jnp.full((self.n_coord,), self.coord_w),
                                 jnp.full((self.val_dim,), self.value_w)])
            return jnp.mean(((pred - target) ** 2) * w)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        params, static = eqx.partition(model, eqx.is_array)
        g, _ = eqx.partition(grads, eqx.is_array)
        updates, opt_state = self.optimizer.update(g, opt_state, params)
        model = eqx.combine(eqx.apply_updates(params, updates), static)
        return model, opt_state, loss

    def train_epoch(self, epoch: int, key) -> dict:
        bs = self.cfg.training.batch_size
        perm = jr.permutation(key, self.S)
        losses = []
        for i in range(0, self.S, bs):
            idx = perm[i:i + bs]
            cb = self.P[idx]                                         # PER-SAMPLE params (pooled multi-traj)
            self.model, self.opt_state, l = self._train_step(
                self.model, self.opt_state, self.x1[idx], cb, jr.fold_in(key, i))
            losses.append(float(l))
        return {"loss": float(np.mean(losses))}

    def sample(self, *, key, batch, cond=None, steps: int = 50):
        if cond is None:
            cond = jnp.broadcast_to(self.pj, (batch, self.n_cond))
        return euler_sample(_cg_model_fn(self.model), key=key,
                            shape=(batch, self.N, self.n_coord + self.val_dim),
                            cond=cond, steps=steps, prior_fn=self._sample_x0)

    def _decode_tokens(self, samples):
        """(B,N,7) generated tokens -> per-sample int-lattice coords + de-whitened (re, im),
        collisions summed. Host/numpy (called from evaluate, not jitted)."""
        cshape = self.cshape_np.astype(np.int64)
        mu, sd = self.nrm["mu"], self.nrm["sd"]
        out = []
        for s in np.asarray(samples):
            coord, val = s[:, :self.n_coord], s[:, self.n_coord:]
            ci = np.clip(np.rint(coord * cshape).astype(np.int64), 0, cshape - 1)
            val = val * sd + mu                                  # de-whiten
            flat = np.ravel_multi_index([ci[:, k] for k in range(self.n_coord)], cshape)
            uniq, inv = np.unique(flat, return_inverse=True)
            vre, vim = np.zeros(len(uniq)), np.zeros(len(uniq))
            np.add.at(vre, inv, val[:, 0])
            np.add.at(vim, inv, val[:, 1])
            out.append({"coord_int": np.stack(np.unravel_index(uniq, cshape), 1),
                        "re": vre, "im": vim})
        return out

    def evaluate(self, epoch: int) -> dict:
        steps = getattr(self.cfg.validation, "eval_sample_steps", 50)
        samp = np.asarray(self.sample(key=jr.PRNGKey(epoch), batch=4, steps=steps))
        coord, val = samp[..., :self.n_coord], samp[..., self.n_coord:]
        toks = self._decode_tokens(samp)
        return {"samp_coord_min": float(coord.min()), "samp_coord_max": float(coord.max()),
                "samp_val_std": float(val.std()), "real_val_std": float(self.X_np.std()),
                "n_unique_mean": float(np.mean([len(t["re"]) for t in toks]))}
