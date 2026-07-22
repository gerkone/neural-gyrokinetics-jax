"""WaveletDiT: the gyrowave flow-matching / diffusion model over wavelet-token sets.

A token = a wavelet/HL coefficient VALUE (diffused) tagged by its static physical
COORDINATE (s, x, log-scale, ky). The set is unordered and variable-length, so the model
is coordinate-driven, grid-free.

Design (AB-UPT / noether blueprint, NOTES.md): token embed = MLP(sincos(coord)) + MLP(value);
RoPE on the coordinate inside attention (full-attn path); AdaLN-Zero conditioning from
[t | physical params]; attention backend swappable — 'full' (O(N^2), RoPE) or 'phys'
(Transolver slice attention, O(N*slices), no RoPE) for the ~28k-token sets where full
attention is intractable. Output = velocity (rectified flow) on the value channels.

The outer model is identical for both attention backends; only the per-block attention differs.
"""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.attention import MultiHeadSelfAttention
from neugk_jax.models.embeddings import ContinuousConditionEmbed
from neugk_jax.models.physics_attention import PhysicsAttentionIrregularMesh
from neugk_jax.models.rope import rope_tables
from neugk_jax.models.utils import MLP, DiTModulation, LayerNorm, Linear


def _sincos(coords, bands):
    freqs = (2.0 ** jnp.arange(bands, dtype=coords.dtype)) * jnp.pi
    ang = coords[:, :, None] * freqs[None, None, :]                # (N,A,bands)
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1).reshape(coords.shape[0], -1)


class WaveletDiTBlock(eqx.Module):
    """Pre-norm DiT block with AdaLN-Zero; attention backend swappable.

    Composes SHARED pieces only — it re-implements no attention/softmax/modulation:
    ``models.attention.MultiHeadSelfAttention`` for ``attn_kind='full'`` (with rotary
    embedding on the physical coords, passed as the precomputed ``(cos, sin)`` rope
    table), or ``PhysicsAttentionIrregularMesh`` for ``attn_kind='phys'`` (slice
    attention, no token-token dot product so no rope). Norms/MLP come from
    ``models.utils``; ``DiTModulation`` (zero-inited for an identity-block start)
    supplies the 6-way AdaLN-Zero scales/shifts/gates. The block owns only the
    residual wiring.
    """
    norm1: LayerNorm
    norm2: LayerNorm
    attn: eqx.Module
    mlp: MLP
    mod: DiTModulation
    attn_kind: str = eqx.field(static=True)

    def __init__(self, dim, n_head, cond_dim, *, attn_kind="phys", slice_num=64,
                 mlp_ratio=2, key):
        ka, km, kmod = jr.split(key, 3)
        self.norm1 = LayerNorm(dim, elementwise_affine=False)
        self.norm2 = LayerNorm(dim, elementwise_affine=False)
        if attn_kind == "full":
            # shared MHSA; qkv_bias=False matches the coordinate-driven set-transformer design.
            self.attn = MultiHeadSelfAttention(dim, n_head, qkv_bias=False, key=ka)
        elif attn_kind == "phys":
            self.attn = PhysicsAttentionIrregularMesh(
                dim, heads=n_head, dim_head=dim // n_head, slice_num=slice_num, key=ka)
        else:
            raise ValueError(f"attn_kind must be 'full' or 'phys', got {attn_kind!r}")
        hidden = max(int(dim * mlp_ratio), dim)
        self.mlp = MLP([dim, hidden, dim], key=km)
        mod = DiTModulation(cond_dim, dim, key=kmod)                 # zero-init -> identity block
        mod = eqx.tree_at(lambda m: (m.proj.inner.weight, m.proj.inner.bias), mod,
                          (jnp.zeros_like(mod.proj.inner.weight), jnp.zeros_like(mod.proj.inner.bias)))
        self.mod = mod
        self.attn_kind = attn_kind

    def _attn(self, x, rope_cs):
        # full attention consumes the rope (cos, sin) table; slice attention has no
        # token-token dot product, so rope does not apply there.
        return self.attn(x, rope=rope_cs) if self.attn_kind == "full" else self.attn(x)

    def __call__(self, x, cond, rope_cs):
        s1, sh1, g1, s2, sh2, g2 = self.mod(cond)                    # 6-way adaLN-zero
        x = x + g1[None] * self._attn(self.norm1(x) * (1 + s1[None]) + sh1[None], rope_cs)
        x = x + g2[None] * self.mlp(self.norm2(x) * (1 + s2[None]) + sh2[None])
        return x


class WaveletDiT(eqx.Module):
    """Velocity-field model over a coordinate-tagged wavelet-token set.
    __call__(value, coords, t, cond) -> velocity, all per-sample (vmap the batch)."""
    time_embed: ContinuousConditionEmbed
    cond_embed: ContinuousConditionEmbed
    in_proj: MLP
    blocks: tuple
    ln_out: LayerNorm
    head: Linear
    bands: int = eqx.field(static=True)
    n_coord: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    rope: bool = eqx.field(static=True)

    def __init__(self, *, val_dim=2, n_coord=4, n_cond=4, n_hidden=256, n_layers=8,
                 n_head=8, mlp_ratio=2, bands=6, embed_dim=32, attn_kind="phys",
                 slice_num=64, key):
        ks = jr.split(key, 4 + n_layers)
        self.time_embed = ContinuousConditionEmbed(embed_dim, 1, key=ks[0])
        self.cond_embed = ContinuousConditionEmbed(embed_dim, n_cond, key=ks[1])
        cond_dim = self.time_embed.cond_dim + self.cond_embed.cond_dim
        in_dim = val_dim + n_coord * 2 * bands
        self.in_proj = MLP([in_dim, n_hidden * 2, n_hidden], key=ks[2])
        self.blocks = tuple(
            WaveletDiTBlock(n_hidden, n_head, cond_dim, attn_kind=attn_kind,
                            slice_num=slice_num, mlp_ratio=mlp_ratio, key=ks[4 + i])
            for i in range(n_layers))
        self.ln_out = LayerNorm(n_hidden, elementwise_affine=False)
        self.head = Linear(n_hidden, val_dim, key=ks[3])
        self.bands = bands
        self.n_coord = n_coord
        self.head_dim = n_hidden // n_head
        self.rope = (attn_kind == "full")

    def __call__(self, value, coords, t, cond):
        # value (N, val_dim) diffused; coords (N, n_coord) static physical; t scalar; cond (n_cond,)
        c = jnp.concatenate([self.time_embed(t.reshape(1)), self.cond_embed(cond)], axis=-1)
        h = self.in_proj(jnp.concatenate([value, _sincos(coords, self.bands)], axis=-1))
        rope_cs = rope_tables(coords, self.head_dim) if self.rope else None
        for blk in self.blocks:
            h = blk(h, c, rope_cs)
        return self.head(self.ln_out(h))
