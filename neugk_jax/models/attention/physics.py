"""Physics attention (Transolver, Wu et al. 2024) on ``(n_tokens, dim)`` tokens of any layout.

Per head the tokens are softly assigned to ``slice_num`` learned states, the states attend to each
other and are mapped back to the tokens through the same assignment, so the cost is linear in the
number of tokens and no grid or window is needed. Port of ``Physics_Attention_Irregular_Mesh``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.utils import Linear, RMSNorm, dropout, split_key


class PhysicsAttention(eqx.Module):
    """Slice attention: soft slicing into ``slice_num`` states per head, attention among them, deslicing.

    ``temperature`` (per head, initialized to 0.5) sharpens the slice assignment; the slice
    projection is orthogonally initialized as in the reference. ``qk_norm`` RMS-normalizes the
    queries / keys of the slice tokens.
    """

    in_x: Linear
    in_fx: Linear
    in_slice: Linear
    to_q: Linear
    to_k: Linear
    to_v: Linear
    proj: Linear
    q_norm: object | None
    k_norm: object | None
    temperature: jnp.ndarray
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    slice_num: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    attn_drop: float = eqx.field(static=True)
    proj_drop: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key,
        slice_num: int = 64,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.slice_num = slice_num
        self.scale = self.head_dim**-0.5
        k = jr.split(key, 8)
        self.in_x = Linear(dim, dim, key=k[0])
        self.in_fx = Linear(dim, dim, key=k[1])
        slice_proj = Linear(self.head_dim, slice_num, key=k[2])
        w = jax.nn.initializers.orthogonal()(k[3], (slice_num, self.head_dim))
        self.in_slice = eqx.tree_at(
            lambda m: (m.inner.weight, m.inner.bias), slice_proj, (w, jnp.zeros((slice_num,)))
        )
        self.to_q = Linear(self.head_dim, self.head_dim, key=k[4], use_bias=qkv_bias)
        self.to_k = Linear(self.head_dim, self.head_dim, key=k[5], use_bias=qkv_bias)
        self.to_v = Linear(self.head_dim, self.head_dim, key=k[6], use_bias=qkv_bias)
        self.proj = Linear(dim, dim, key=k[7])
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.temperature = jnp.full((num_heads, 1, 1), 0.5)
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    def slice_weights(self, x: jnp.ndarray) -> jnp.ndarray:
        """``(heads, n, slice_num)`` soft assignment of the tokens ``(n, dim)`` to the slices."""
        h = self.in_x(x).reshape(x.shape[0], self.num_heads, self.head_dim).transpose(1, 0, 2)
        return jax.nn.softmax(self.in_slice(h) / self.temperature, axis=-1)

    def __call__(self, x: jnp.ndarray, *, key=None, inference: bool = True) -> jnp.ndarray:
        n, dim = x.shape
        fx = self.in_fx(x).reshape(n, self.num_heads, self.head_dim).transpose(1, 0, 2)
        w = self.slice_weights(x)
        tokens = jnp.einsum("hnd,hng->hgd", fx, w) / (jnp.sum(w, axis=1)[..., None] + 1e-5)
        q, k, v = self.to_q(tokens), self.to_k(tokens), self.to_v(tokens)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        ka, kp = split_key(key, 2)
        attn = jax.nn.softmax(jnp.einsum("hgd,hfd->hgf", q, k) * self.scale, axis=-1)
        attn = dropout(attn, self.attn_drop, key=ka, inference=inference)
        out = jnp.einsum("hgd,hng->hnd", jnp.einsum("hgf,hfd->hgd", attn, v), w)
        out = out.transpose(1, 0, 2).reshape(n, dim)
        return dropout(self.proj(out), self.proj_drop, key=kp, inference=inference)
