"""Multi-head self-attention used by both Swin and ViT layers.

Operates on flattened token tensors of shape ``(n_tokens, dim)`` per sample.
Two backends:

* ``"einsum"`` — explicit q/k/v einsum + softmax. Always available.
* ``"flash"``  — ``jax.nn.dot_product_attention(implementation="cudnn")``
  which dispatches to cuDNN's flash attention kernel on Hopper/Ampere
  (falls back to XLA elsewhere). Available in JAX ≥ 0.4.31.

Backend is selected per-module at construction; a free function
``set_default_attention_backend`` lets benchmarks flip it globally without
rebuilding the model.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.embeddings import RPB
from neugk_jax.models.utils import Gate, Linear, RMSNorm, dropout

_DEFAULT_BACKEND = "einsum"


def set_default_attention_backend(backend: str) -> None:
    global _DEFAULT_BACKEND
    if backend not in ("einsum", "flash"):
        raise ValueError(f"unknown attention backend: {backend!r}")
    _DEFAULT_BACKEND = backend


def get_default_attention_backend() -> str:
    return _DEFAULT_BACKEND


def _has_flash() -> bool:
    return hasattr(jax.nn, "dot_product_attention")


def _pick_backend(backend: Optional[str]) -> str:
    chosen = backend or get_default_attention_backend()
    return "einsum" if chosen == "flash" and not _has_flash() else chosen


def _einsum_attention(q, k, v, scale, bias, attn_drop=0.0, key=None, inference=True):
    # q, k, v: (n, heads, head_dim)
    logits = jnp.einsum("nhd,mhd->hnm", q, k) * scale
    if bias is not None:
        logits = logits + bias
    attn = jax.nn.softmax(logits, axis=-1)
    attn = dropout(attn, attn_drop, key=key, inference=inference)
    return jnp.einsum("hnm,mhd->nhd", attn, v)


def _dropout_active(rate, key, inference):
    return rate > 0.0 and key is not None and not inference


_FLASH_DTYPES = (jnp.bfloat16, jnp.float16)


def _flash_attention(q, k, v, scale, bias):
    """Wrap ``jax.nn.dot_product_attention`` (expects (B, n, H, D)).

    Falls back to einsum when the dtype is unsupported by cuDNN (fp32, etc.).
    Inputs are kept in ``(n, H, D)`` layout so the head_dim axis stays
    stride-1 — cuDNN's flash kernel requires that.
    """
    if q.dtype not in _FLASH_DTYPES:
        return _einsum_attention(q, k, v, scale, bias)
    N, H, D = q.shape
    q4 = q[None]
    k4 = k[None]
    v4 = v[None]
    if bias is not None:
        b = bias
        if b.ndim == 2:
            b = jnp.broadcast_to(b, (H, N, N))
        elif b.ndim == 3 and b.shape[0] == 1:
            b = jnp.broadcast_to(b, (H, N, N))
        # force contiguous allocation via astype — cuDNN dislikes views
        b = jnp.asarray(b, dtype=q.dtype)
        bias = b[None]
    out = jax.nn.dot_product_attention(
        q4, k4, v4, bias=bias, scale=scale, implementation="cudnn",
    )
    return out[0]  # drop the added batch dim, back to (n, H, D)


class MultiHeadSelfAttention(eqx.Module):
    """Multi-head self-attention with optional extras.

    Switches:

    * ``qkv_bias`` — toggle the qkv bias term.
    * ``qk_norm``  — RMSNorm per-head on q and k pre-softmax.
    * ``use_rpb``  — additive relative-position-bias from an internal tiny MLP.
    * ``gated_attention`` — headwise multiplicative gate from a separate
      Linear, applied to the attention output.
    * ``attn_drop`` / ``proj_drop`` — dropout on the attention probabilities
      and on the output projection, active only with a key and ``inference=False``.

    Backend selection (``einsum`` / ``flash``) is independent; the einsum path
    is used whenever attention dropout is active.
    """

    qkv: Linear
    proj: Linear
    q_norm: object | None
    k_norm: object | None
    rpb: object | None
    gate: object | None
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    attn_drop: float = eqx.field(static=True)
    proj_drop: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        window_size: Optional[tuple[int, ...]] = None,
        backend: Optional[str] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        kqkv, kproj, kqn, kkn, krpb, kgate = jr.split(key, 6)
        self.qkv = Linear(dim, 3 * dim, key=kqkv, use_bias=qkv_bias)
        self.proj = Linear(dim, dim, key=kproj, use_bias=True)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        if use_rpb:
            assert window_size is not None, "use_rpb requires window_size"
            self.rpb = RPB(window_size, num_heads, key=krpb)
        else:
            self.rpb = None
        self.gate = Gate(self.head_dim, key=kgate) if gated_attention else None
        self.backend = _pick_backend(backend)
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    def __call__(
        self,
        x: jnp.ndarray,
        attn_bias: Optional[jnp.ndarray] = None,
        *,
        key=None,
        inference: bool = True,
    ) -> jnp.ndarray:
        n, dim = x.shape
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # fold rpb bias into the attention bias slot
        if self.rpb is not None:
            rpb_bias = self.rpb()  # shape: (heads, sl, sl)
            attn_bias = rpb_bias if attn_bias is None else attn_bias + rpb_bias
        ka, kp = (None, None) if key is None else jr.split(key)
        if self.backend == "flash" and not _dropout_active(self.attn_drop, ka, inference):
            out = _flash_attention(q, k, v, self.scale, attn_bias)
        else:
            out = _einsum_attention(q, k, v, self.scale, attn_bias, self.attn_drop, ka, inference)
        # out: (n, H, D); apply optional headwise gate before flattening to (n, dim)
        if self.gate is not None:
            out = self.gate(out, q)
        out = out.reshape(n, dim)
        return dropout(self.proj(out), self.proj_drop, key=kp, inference=inference)


class MultiHeadCrossAttention(eqx.Module):
    """Cross-attention: queries from ``left``, keys/values from ``right``.

    Used by the GyroSwin mixing layers — ``left`` attends to ``right``,
    output dim matches ``left``. Both inputs are tokenised ``(n, dim)``;
    the caller flattens spatial axes before the call. ``attn_drop`` /
    ``proj_drop`` act as in ``MultiHeadSelfAttention``.
    """

    q: Linear
    kv: Linear
    proj: Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    attn_drop: float = eqx.field(static=True)
    proj_drop: float = eqx.field(static=True)

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        *,
        key,
        out_dim: Optional[int] = None,
        qkv_bias: bool = False,
        backend: Optional[str] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        assert q_dim % num_heads == 0, f"q_dim={q_dim} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim ** -0.5
        kq, kkv, kp = jr.split(key, 3)
        out_dim = out_dim or q_dim
        self.q = Linear(q_dim, q_dim, key=kq, use_bias=qkv_bias)
        self.kv = Linear(kv_dim, 2 * q_dim, key=kkv, use_bias=qkv_bias)
        self.proj = Linear(q_dim, out_dim, key=kp, use_bias=True)
        self.backend = _pick_backend(backend)
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    def __call__(self, left: jnp.ndarray, right: jnp.ndarray, *, key=None,
                 inference: bool = True) -> jnp.ndarray:
        n_q, _ = left.shape
        n_kv, _ = right.shape
        q = self.q(left).reshape(n_q, self.num_heads, self.head_dim)
        kv = self.kv(right).reshape(n_kv, 2, self.num_heads, self.head_dim)
        k = kv[:, 0]
        v = kv[:, 1]
        ka, kp = (None, None) if key is None else jr.split(key)
        if self.backend == "flash" and not _dropout_active(self.attn_drop, ka, inference):
            out = _flash_attention(q, k, v, self.scale, None)
        else:
            out = _einsum_attention(q, k, v, self.scale, None, self.attn_drop, ka, inference)
        out = out.reshape(n_q, -1)
        return dropout(self.proj(out), self.proj_drop, key=kp, inference=inference)
