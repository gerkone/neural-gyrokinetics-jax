"""Cross-attention layers used to mix the df and phi U-Net latents.

``MixingBlock`` is a single cross-attention + MLP block; ``VSpaceReduce``
integrates over the velocity axes via a learned query token; ``RSpaceReduce``
does the same over real space (the ``integral`` flux-head reduction).
``FluxDecoder`` is the scalar flux head, optionally FiLM-conditioned on the
raw conditioning scalars.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from neugk_jax.models.attention import MultiHeadCrossAttention
from neugk_jax.models.embeddings import ContinuousConditionEmbed
from neugk_jax.models.swin import Film, _DropPath
from neugk_jax.models.utils import MLP, LayerNorm, Linear, dropout, gelu, split_key


def _pool_attention(q, k, v, scale, attn_drop, key, inference):
    # q: (g, h, d); k, v: (g, n, h, d) -> (g, h, d)
    attn = jax.nn.softmax(jnp.einsum("ghd,gnhd->ghn", q, k) * scale, axis=-1)
    attn = dropout(attn, attn_drop, key=key, inference=inference)
    return jnp.einsum("ghn,gnhd->ghd", attn, v)


class MixingBlock(eqx.Module):
    """Cross-attention + MLP. ``left`` queries kv from ``right``; output dim = left_dim.

    ``attn_drop`` drops attention probabilities; ``drop`` is the output-projection
    and MLP dropout.
    """

    norm1: LayerNorm
    attn: MultiHeadCrossAttention
    drop_path: _DropPath
    norm2: LayerNorm
    mlp: MLP

    def __init__(
        self,
        left_dim: int,
        right_dim: int,
        num_heads: int,
        *,
        key,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        act_fn=gelu,
    ):
        k1, k2 = jr.split(key, 2)
        self.norm1 = LayerNorm(left_dim, elementwise_affine=True)
        self.attn = MultiHeadCrossAttention(
            q_dim=left_dim,
            kv_dim=right_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            key=k1,
        )
        self.drop_path = _DropPath(drop_path)
        self.norm2 = LayerNorm(left_dim, elementwise_affine=True)
        self.mlp = MLP(
            [left_dim, int(left_dim * mlp_ratio), left_dim],
            act_fn=act_fn,
            drop=drop,
            key=k2,
        )

    def __call__(
        self,
        left: jnp.ndarray,
        right: Optional[jnp.ndarray] = None,
        *,
        key=None,
        inference: bool = True,
    ) -> jnp.ndarray:
        right = right if right is not None else left
        l_shape = left.shape
        l_tok = left.reshape(-1, l_shape[-1])
        r_tok = right.reshape(-1, right.shape[-1])
        k_attn, k_dp1, k_mlp, k_dp2 = split_key(key, 4)
        # post-norm on the attn output, pre-norm on the mlp branch
        h = self.norm1(self.attn(l_tok, r_tok, key=k_attn, inference=inference))
        x = l_tok + self.drop_path(h, key=k_dp1, inference=inference)
        h = self.mlp(self.norm2(x), key=k_mlp, inference=inference)
        x = x + self.drop_path(h, key=k_dp2, inference=inference)
        return x.reshape(l_shape)


class VSpaceReduce(eqx.Module):
    """Integrate velocity axes of a 5D df latent into a 3D phi-shaped latent.

    A learned query token (``integral_token``) cross-attends to the velocity
    tokens at each (s, x, y) position. Output shape: ``(s, x, y, out_dim)``.
    """

    kv: Linear
    proj: Linear
    integral_token: jax.Array
    buffer_fields = ("integral_token",)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    decouple_mu: bool = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    attn_drop: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int,
        *,
        key,
        decouple_mu: bool = False,
        gain: float = 1e-2,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
    ):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.out_dim = out_dim
        self.scale = self.head_dim**-0.5
        self.decouple_mu = decouple_mu
        self.attn_drop = attn_drop
        kkv, kp, ktoken = jr.split(key, 3)
        self.kv = Linear(dim, 2 * dim, key=kkv, use_bias=qkv_bias)
        self.proj = Linear(dim, out_dim, key=kp, use_bias=True)
        self.integral_token = gain * jr.normal(ktoken, (1, 1, dim))

    def __call__(self, df: jnp.ndarray, *, key=None, inference: bool = True) -> jnp.ndarray:
        # df comes as (vp, [mu,] s, x, y, C); decouple_mu controls whether mu is present
        if self.decouple_mu:
            vpar, ns, nx, ny, dim = df.shape
            df_t = rearrange(df, "vp s x y c -> (s x y) vp c")
        else:
            vpar, mu, ns, nx, ny, dim = df.shape
            df_t = rearrange(df, "vp mu s x y c -> (s x y) (vp mu) c")
        n_groups, n_tok, _ = df_t.shape
        kv = self.kv(df_t).reshape(n_groups, n_tok, 2, self.num_heads, self.head_dim)
        q = self.integral_token.reshape(1, self.num_heads, self.head_dim)
        q = jnp.broadcast_to(q, (n_groups, self.num_heads, self.head_dim))
        out = _pool_attention(
            q, kv[:, :, 0], kv[:, :, 1], self.scale, self.attn_drop, key, inference
        )
        out = self.proj(out.reshape(n_groups, self.num_heads * self.head_dim))
        return out.reshape(ns, nx, ny, self.out_dim)


class LatentMixingTransformer(eqx.Module):
    """A stack of ``depth`` cross-attention ``MixingBlock``s (one FluxDecoder stage).

    With ``cond_dim`` set, each block input is FiLM-modulated by a condition
    embedded from the raw scalars by this stage's own ``cond_embed``.
    """

    blocks: list
    cond_embed: Optional[ContinuousConditionEmbed]
    conditioning: Optional[list]

    def __init__(
        self,
        left_dim: int,
        right_dim: int,
        num_heads: int,
        depth: int,
        *,
        key,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        n_cond: int = 0,
        cond_embed_dim: int = 128,
    ):
        kb, kc, kf = jr.split(key, 3)
        self.blocks = [
            MixingBlock(
                left_dim,
                right_dim,
                num_heads,
                key=k,
                mlp_ratio=2.0,
                qkv_bias=True,
                attn_drop=attn_drop,
                drop=drop,
            )
            for k in jr.split(kb, depth)
        ]
        if n_cond > 0:
            self.cond_embed = ContinuousConditionEmbed(cond_embed_dim, n_cond, key=kc)
            self.conditioning = [
                Film(self.cond_embed.cond_dim, left_dim, key=k) for k in jr.split(kf, depth)
            ]
        else:
            self.cond_embed = None
            self.conditioning = None

    def __call__(
        self, left: jnp.ndarray, right: jnp.ndarray, cond=None, *, key=None, inference: bool = True
    ) -> jnp.ndarray:
        c = self.cond_embed(cond) if self.cond_embed is not None else None
        x = left
        for i, (blk, k) in enumerate(zip(self.blocks, split_key(key, len(self.blocks)))):
            if c is not None:
                x = self.conditioning[i](x, c)
            x = blk(x, right, key=k, inference=inference)
        return x


class FluxDecoder(eqx.Module):
    """Predict a scalar flux from the per-scale (phi, df) latents.

    One ``LatentMixingTransformer`` stage per scale: stage ``i`` cross-attends the
    phi latent (query) to the df latent (kv), pools over space (``max``, ``mean``
    or an ``integral`` query token) to a vector of ``left_dims[i]``, and the
    per-scale vectors are concatenated and fed to ``flux_mlp`` (sum -> half -> 1).
    ``n_cond > 0`` FiLM-conditions every stage on the raw conditioning scalars.
    """

    blocks: list
    reductions: Optional[list]
    flux_mlp: MLP
    reduction: str = eqx.field(static=True)
    detach_latents: bool = eqx.field(static=True)
    use_cond: bool = eqx.field(static=True)

    def __init__(
        self,
        left_dims,
        right_dims,
        num_heads: int,
        depth: int,
        *,
        key,
        reduction: str = "max",
        attn_drop: float = 0.1,
        drop: float = 0.0,
        detach_latents: bool = False,
        n_cond: int = 0,
        cond_embed_dim: int = 128,
    ):
        if reduction not in ("max", "mean", "integral"):
            raise ValueError(f"unknown flux reduction {reduction!r}")
        ks = jr.split(key, 2 * len(left_dims) + 1)
        self.blocks = [
            LatentMixingTransformer(
                left_dims[i],
                right_dims[i],
                num_heads,
                depth,
                key=ks[i],
                attn_drop=attn_drop,
                drop=drop,
                n_cond=n_cond,
                cond_embed_dim=cond_embed_dim,
            )
            for i in range(len(left_dims))
        ]
        if reduction == "integral":
            self.reductions = [
                RSpaceReduce(d, d, num_heads, key=ks[len(left_dims) + i], attn_drop=0.1)
                for i, d in enumerate(left_dims)
            ]
        else:
            self.reductions = None
        flux_latent = int(sum(left_dims))
        self.flux_mlp = MLP([flux_latent, flux_latent // 2, 1], act_fn=gelu, drop=drop, key=ks[-1])
        self.reduction = reduction
        self.detach_latents = detach_latents
        self.use_cond = n_cond > 0

    def mix(
        self,
        i: int,
        left: jnp.ndarray,
        right: jnp.ndarray,
        cond=None,
        *,
        key=None,
        inference: bool = True,
    ) -> jnp.ndarray:
        if self.detach_latents:
            left, right = jax.lax.stop_gradient(left), jax.lax.stop_gradient(right)
        k_mix, k_red = split_key(key, 2)
        x = self.blocks[i](
            left, right, cond if self.use_cond else None, key=k_mix, inference=inference
        )
        if self.reduction == "integral":
            return self.reductions[i](x, key=k_red, inference=inference)
        x = x.reshape(-1, x.shape[-1])
        return jnp.max(x, axis=0) if self.reduction == "max" else jnp.mean(x, axis=0)

    def __call__(self, flux_lats, *, key=None, inference: bool = True) -> jnp.ndarray:
        return self.flux_mlp(jnp.concatenate(flux_lats, axis=-1), key=key, inference=inference)


class RSpaceReduce(eqx.Module):
    """Pool every spatial axis into a single token (used by the flux head)."""

    kv: Linear
    proj: Linear
    integral_token: jax.Array
    buffer_fields = ("integral_token",)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    attn_drop: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int,
        *,
        key,
        gain: float = 1e-2,
        attn_drop: float = 0.0,
    ):
        assert dim % num_heads == 0
        self.attn_drop = attn_drop
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.out_dim = out_dim
        self.scale = self.head_dim**-0.5
        kkv, kp, kt = jr.split(key, 3)
        self.kv = Linear(dim, 2 * dim, key=kkv, use_bias=False)
        self.proj = Linear(dim, out_dim, key=kp, use_bias=True)
        self.integral_token = gain * jr.normal(kt, (1, 1, dim))

    def __call__(self, x: jnp.ndarray, *, key=None, inference: bool = True) -> jnp.ndarray:
        # x: (..., C) -> one group holding every token
        x_t = x.reshape(1, -1, x.shape[-1])
        kv = self.kv(x_t).reshape(1, x_t.shape[1], 2, self.num_heads, self.head_dim)
        q = self.integral_token.reshape(1, self.num_heads, self.head_dim)
        out = _pool_attention(
            q, kv[:, :, 0], kv[:, :, 1], self.scale, self.attn_drop, key, inference
        )
        out = out.reshape(1, self.num_heads * self.head_dim)
        return self.proj(out).reshape(self.out_dim)
