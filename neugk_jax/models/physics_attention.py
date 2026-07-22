"""Physics Attention (irregular-mesh / slice-attention variant) from Transolver (Wu et al.,
2024), per-sample JAX/equinox port. Ported from
~/git/active-learning-agent/src/agentAL/surrogate/physics_attention.py, with two changes:
per-sample ``(N, C) -> (N, C)`` (batch handled by caller's vmap) and no jaxtyping (repo
convention).

Cost is ``O(N * slice_num)`` not ``O(N^2)`` — tokens are softly assigned to ``slice_num``
slices, attention runs among the slices, then scattered back, so there is NO token-token
interaction; RoPE does not apply to this path, positional info must enter via token features.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _orthogonal_init(key, shape, dtype):
    """torch.nn.init.orthogonal_ semantics for a 2D weight (out, in)."""
    flat = (shape[0], int(jnp.prod(jnp.array(shape[1:]))))
    a = jax.random.normal(key, flat, dtype=jnp.float32)
    trans = flat[0] < flat[1]
    if trans:
        a = a.T
    q, r = jnp.linalg.qr(a)
    q = q * jnp.sign(jnp.diag(r))[None, :]
    if trans:
        q = q.T
    return q.reshape(shape).astype(dtype)


def _trunc_normal_init(key, shape, std, dtype):
    return jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype=jnp.float32).astype(dtype) * std


class PhysicsAttentionIrregularMesh(eqx.Module):
    """Slice attention, per-sample ``(N, C) -> (N, C)``. Drop-in for a full
    self-attention layer at the ``(N, C)`` interface, but linear in N."""

    in_project_x: eqx.nn.Linear
    in_project_fx: eqx.nn.Linear
    in_project_slice: eqx.nn.Linear
    to_q: eqx.nn.Linear
    to_k: eqx.nn.Linear
    to_v: eqx.nn.Linear
    to_out: eqx.nn.Linear
    temperature: jax.Array
    heads: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    slice_num: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(self, dim, *, heads=8, dim_head=64, slice_num=64, key, dtype=jnp.float32):
        inner = dim_head * heads
        self.heads, self.dim_head, self.slice_num = heads, dim_head, slice_num
        self.scale, self.eps = dim_head ** -0.5, 1e-5
        kx, kfx, ks, kq, kk, kv, ko = jax.random.split(key, 7)

        def _mk(i, o, k, bias=True):
            lin = eqx.nn.Linear(i, o, use_bias=bias, key=k)
            lin = eqx.tree_at(lambda l: l.weight, lin, lin.weight.astype(dtype))
            if bias:
                lin = eqx.tree_at(lambda l: l.bias, lin, lin.bias.astype(dtype))
            return lin

        self.in_project_x = _mk(dim, inner, kx)
        self.in_project_fx = _mk(dim, inner, kfx)
        sl = _mk(dim_head, slice_num, ks)
        sl = eqx.tree_at(lambda l: l.weight, sl, _orthogonal_init(ks, (slice_num, dim_head), dtype))
        sl = eqx.tree_at(lambda l: l.bias, sl, jnp.zeros((slice_num,), dtype))
        self.in_project_slice = sl
        self.to_q = _mk(dim_head, dim_head, kq, bias=False)
        self.to_k = _mk(dim_head, dim_head, kk, bias=False)
        self.to_v = _mk(dim_head, dim_head, kv, bias=False)
        self.to_out = _mk(inner, dim, ko)
        self.temperature = jnp.full((heads, 1, 1), 0.5, dtype=dtype)  # per-head, learnable

    def __call__(self, x):
        # x: (N, C)
        N = x.shape[0]
        H, D, G = self.heads, self.dim_head, self.slice_num

        def ap(lin, t):
            y = jnp.einsum("...i,oi->...o", t, lin.weight)
            return y + lin.bias if lin.bias is not None else y

        fx_mid = ap(self.in_project_fx, x).reshape(N, H, D).transpose(1, 0, 2)   # (H, N, D)
        x_mid = ap(self.in_project_x, x).reshape(N, H, D).transpose(1, 0, 2)     # (H, N, D)
        slice_w = jax.nn.softmax(ap(self.in_project_slice, x_mid) / self.temperature, axis=-1)  # (H,N,G)
        slice_norm = slice_w.sum(axis=1)                                         # (H, G)
        slice_tok = jnp.einsum("hnd,hng->hgd", fx_mid, slice_w) / (slice_norm + self.eps)[:, :, None]
        q, k, v = ap(self.to_q, slice_tok), ap(self.to_k, slice_tok), ap(self.to_v, slice_tok)  # (H,G,D)
        # JAX flash attention among the G slices (dot_product_attention -> cuDNN flash when eligible),
        # replacing the manual einsum+softmax. dot_product_attention wants (seq, heads, head_dim).
        out_tok = jax.nn.dot_product_attention(
            q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2), scale=self.scale
        ).transpose(1, 0, 2)                                                     # (H, G, D)
        out = jnp.einsum("hgd,hng->hnd", out_tok, slice_w).transpose(1, 0, 2).reshape(N, H * D)
        return ap(self.to_out, out)
