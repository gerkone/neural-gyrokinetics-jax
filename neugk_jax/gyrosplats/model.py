"""GyrosplatDiT: velocity-field model over the splat atom bank.

Trivial-first design: a plain conditioned transformer (the repo's DiT block —
full self-attention + AdaLN-Zero, zero-initialized so every block starts as
identity) over FLAT per-atom tokens. No grouping, no stats token, no
positional identity — atoms are exchangeable within their carrier-bin
population.

Per-token features: the 17 noisy channels + a carrier-bin embedding
(mandatory: after normalization the ky channel is bin-centered drift, so
tokens otherwise carry no bin identity) + sincos featurization of the mu
channels (coordinates deserve a structured embedding). One shared linear head
emits the 17-channel velocity.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.gyrosplats.splat import MU_SL, N_CHANNELS
from neugk_jax.gyrosplats.window import N_STATE_CH
from neugk_jax.models.attention import _einsum_attention
from neugk_jax.models.embeddings import ContinuousConditionEmbed
from neugk_jax.models.utils import MLP, DiTModulation, LayerNorm, Linear, gelu
from neugk_jax.models.vit import DiTViTBlock as DiTBlock

N_TYPES = 8  # envelope + carrier bins 7..13


def _trunc_normal(key, shape, std=0.02):
    return jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype=jnp.float32) * std


def _linear(n_in, n_out, *, key):
    lin = eqx.nn.Linear(n_in, n_out, key=key)
    lin = eqx.tree_at(lambda m: m.weight, lin, _trunc_normal(key, (n_out, n_in)))
    return eqx.tree_at(lambda m: m.bias, lin, jnp.zeros((n_out,), jnp.float32))


def _apply(lin, x):
    return jnp.einsum("...i,oi->...o", x, lin.weight) + lin.bias


def _zero_modulation(block: DiTBlock) -> DiTBlock:
    # adaln-zero: gates (and scales/shifts) start at exactly 0 -> identity block
    return eqx.tree_at(
        lambda b: (b.mod.proj.inner.weight, b.mod.proj.inner.bias),
        block,
        (
            jnp.zeros_like(block.mod.proj.inner.weight),
            jnp.zeros_like(block.mod.proj.inner.bias),
        ),
    )


class GyrosplatDiT(eqx.Module):
    time_embed: ContinuousConditionEmbed
    cond_embed: ContinuousConditionEmbed
    type_embed: jax.Array  # (N_TYPES, type_dim)
    in_proj: MLP
    head: eqx.nn.Linear
    blocks: tuple
    ln_out: LayerNorm
    type_ids: jax.Array  # (n_atoms,) int32 buffer

    n_channels: int = eqx.field(static=True)
    mu_fourier_bands: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        bins: np.ndarray,
        n_cond: int = 4,
        n_channels: int = N_CHANNELS,
        n_hidden: int = 256,
        n_layers: int = 8,
        n_head: int = 8,
        mlp_ratio: int = 2,
        type_dim: int = 16,
        mu_fourier_bands: int = 4,
        embed_dim: int = 32,
        key,
        **_unused,
    ):
        bins = np.asarray(bins)
        # envelope -> 0, carrier bin m -> m - 6 (bins 7..13 -> 1..7)
        type_ids = np.where(bins == 0, 0, bins - 6).astype(np.int32)
        self.type_ids = jax.lax.stop_gradient(jnp.asarray(type_ids))
        self.n_channels = n_channels
        self.mu_fourier_bands = mu_fourier_bands

        keys = jax.random.split(key, 5 + n_layers)
        self.time_embed = ContinuousConditionEmbed(embed_dim, 1, key=keys[0])
        self.cond_embed = ContinuousConditionEmbed(embed_dim, n_cond, key=keys[1])
        cond_dim = self.time_embed.cond_dim + self.cond_embed.cond_dim

        self.type_embed = _trunc_normal(keys[2], (N_TYPES, type_dim))
        in_dim = n_channels + type_dim + 10 * mu_fourier_bands
        self.in_proj = MLP([in_dim, n_hidden * 2, n_hidden], key=keys[3])
        self.head = _linear(n_hidden, n_channels, key=keys[4])
        self.blocks = tuple(
            _zero_modulation(
                DiTBlock(
                    n_hidden,
                    n_head,
                    cond_dim,
                    key=k,
                    mlp_ratio=float(mlp_ratio),
                    norm_affine=False,
                )
            )
            for k in keys[5:]
        )
        self.ln_out = LayerNorm(n_hidden)

    def _features(self, x):
        # sincos featurization of the (noisy) mu channels — coordinates get a
        # structured embedding instead of riding through the mlp as raw scalars
        mu = x[:, MU_SL]
        freqs = (2.0 ** jnp.arange(self.mu_fourier_bands, dtype=x.dtype)) * jnp.pi
        ang = mu[:, :, None] * freqs[None, None, :]  # (n, 5, bands)
        return jnp.concatenate(
            [
                x,
                self.type_embed[self.type_ids],
                jnp.sin(ang).reshape(mu.shape[0], -1),
                jnp.cos(ang).reshape(mu.shape[0], -1),
            ],
            axis=-1,
        )

    def __call__(self, x, t, cond):
        c = jnp.concatenate(
            [self.time_embed(t.reshape(1)), self.cond_embed(cond)], axis=-1
        )
        h = self.in_proj(self._features(x))
        for block in self.blocks:
            h = block(h, c)
        return _apply(self.head, self.ln_out(h))


def _sincos(coords, bands):
    # sincos featurization of 5-d coordinates -> (n, 10*bands)
    freqs = (2.0 ** jnp.arange(bands, dtype=coords.dtype)) * jnp.pi
    ang = coords[:, :, None] * freqs[None, None, :]  # (n, 5, bands)
    return jnp.concatenate(
        [jnp.sin(ang).reshape(coords.shape[0], -1), jnp.cos(ang).reshape(coords.shape[0], -1)],
        axis=-1,
    )


class _RoPEAttention(eqx.Module):
    """Self-attention with rotary embedding keyed by token coordinates.

    Mirrors ``MultiHeadSelfAttention``'s einsum path but rotates q/k per head in
    2-d subplanes: head-dim channels are paired ``(2p, 2p+1)`` and rotated by
    angle ``coord[c_p] * freq_p`` where ``c_p`` cycles the 5 coordinates and the
    per-plane frequency doubles every full cycle (freq = pi * 2**(p // 5)).
    """

    qkv: Linear
    proj: Linear
    freqs: jax.Array  # (n_planes,)
    coord_idx: jax.Array  # (n_planes,) int32
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(self, dim, num_heads, *, n_coords=5, key):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "rope needs even head_dim"
        self.scale = self.head_dim**-0.5
        kq, kp = jr.split(key, 2)
        self.qkv = Linear(dim, 3 * dim, key=kq, use_bias=False)
        self.proj = Linear(dim, dim, key=kp, use_bias=True)
        n_planes = self.head_dim // 2
        rank = np.arange(n_planes) // n_coords
        self.freqs = jnp.asarray(np.pi * (2.0**rank), dtype=jnp.float32)
        self.coord_idx = jnp.asarray(np.arange(n_planes) % n_coords, dtype=jnp.int32)

    def _rotate(self, q, coords):
        # q (n, H, D); coords (n, n_coords)
        n, h, d = q.shape
        ang = coords[:, self.coord_idx] * self.freqs[None, :]  # (n, n_planes)
        cos, sin = jnp.cos(ang)[:, None, :], jnp.sin(ang)[:, None, :]
        q2 = q.reshape(n, h, d // 2, 2)
        qa, qb = q2[..., 0], q2[..., 1]
        ra = qa * cos - qb * sin
        rb = qa * sin + qb * cos
        return jnp.stack([ra, rb], axis=-1).reshape(n, h, d)

    def __call__(self, x, coords):
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = self._rotate(q, coords), self._rotate(k, coords)
        out = _einsum_attention(q, k, v, self.scale, None).reshape(n, -1)
        return self.proj(out)


class _RoPEDiTBlock(eqx.Module):
    """DiTViTBlock with the attention swapped for coordinate-keyed RoPE attention."""

    norm1: LayerNorm
    norm2: LayerNorm
    attn: _RoPEAttention
    mlp: MLP
    mod: DiTModulation

    def __init__(self, dim, num_heads, cond_dim, *, mlp_ratio=2.0, key):
        katt, kmlp, kmod = jr.split(key, 3)
        self.norm1 = LayerNorm(dim, elementwise_affine=False)
        self.norm2 = LayerNorm(dim, elementwise_affine=False)
        self.attn = _RoPEAttention(dim, num_heads, key=katt)
        hidden = max(int(dim * mlp_ratio), dim)
        self.mlp = MLP([dim, hidden, dim], key=kmlp, act_fn=gelu)
        self.mod = DiTModulation(cond_dim, dim, key=kmod)

    def __call__(self, x, cond, coords):
        s_msa, sh_msa, g_msa, s_mlp, sh_mlp, g_mlp = self.mod(cond)
        h = self.attn(self.norm1(x) * (1.0 + s_msa[None]) + sh_msa[None], coords)
        x = x + g_msa[None] * h
        h2 = self.mlp(self.norm2(x) * (1.0 + s_mlp[None]) + sh_mlp[None])
        return x + g_mlp[None] * h2


def _zero_rope_mod(block: _RoPEDiTBlock) -> _RoPEDiTBlock:
    return eqx.tree_at(
        lambda b: (b.mod.proj.inner.weight, b.mod.proj.inner.bias),
        block,
        (jnp.zeros_like(block.mod.proj.inner.weight), jnp.zeros_like(block.mod.proj.inner.bias)),
    )


class GyrosplatWindowDiT(eqx.Module):
    """v7 velocity field over the (n_env + n_window, 16) window state.

    Two token types share one transformer trunk: free envelope atoms and tied
    carrier "window" tokens. Per-type input MLPs feed 8 DiTViTBlocks (full
    attention, AdaLN-Zero on [t, theta]); per-type linear heads emit the 16-ch
    velocity (window pad channels masked in the loss). ``rope`` swaps the blocks
    for coordinate-keyed rotary attention (envelope: current noisy mu; window:
    the fixed scaffold mu).
    """

    time_embed: ContinuousConditionEmbed
    cond_embed: ContinuousConditionEmbed
    env_type: jax.Array  # (type_dim,)
    win_type: jax.Array  # (type_dim,)
    win_embed: jax.Array  # (n_window, win_embed_dim)
    scaffold_mu: jax.Array  # (n_window, 5) constant buffer
    env_in: MLP
    win_in: MLP
    blocks: tuple
    ln_out: LayerNorm
    env_head: eqx.nn.Linear
    win_head: eqx.nn.Linear

    n_env: int = eqx.field(static=True)
    n_window: int = eqx.field(static=True)
    mu_fourier_bands: int = eqx.field(static=True)
    rope: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        scaffold_mu: np.ndarray,
        n_env: int,
        n_cond: int = 4,
        n_hidden: int = 256,
        n_layers: int = 8,
        n_head: int = 8,
        mlp_ratio: int = 2,
        type_dim: int = 16,
        win_embed_dim: int = 16,
        mu_fourier_bands: int = 4,
        embed_dim: int = 32,
        rope: bool = False,
        key,
        **_unused,
    ):
        scaffold_mu = np.asarray(scaffold_mu, np.float32)
        self.n_env = int(n_env)
        self.n_window = int(scaffold_mu.shape[0])
        self.mu_fourier_bands = mu_fourier_bands
        self.rope = bool(rope)
        self.scaffold_mu = jax.lax.stop_gradient(jnp.asarray(scaffold_mu))

        keys = jr.split(key, 7 + n_layers)
        self.time_embed = ContinuousConditionEmbed(embed_dim, 1, key=keys[0])
        self.cond_embed = ContinuousConditionEmbed(embed_dim, n_cond, key=keys[1])
        cond_dim = self.time_embed.cond_dim + self.cond_embed.cond_dim

        self.env_type = _trunc_normal(keys[2], (type_dim,))
        self.win_type = _trunc_normal(keys[3], (type_dim,))
        self.win_embed = _trunc_normal(keys[4], (self.n_window, win_embed_dim))

        sincos_dim = 10 * mu_fourier_bands
        env_in_dim = N_STATE_CH + type_dim + sincos_dim
        win_in_dim = N_STATE_CH + type_dim + win_embed_dim + sincos_dim
        self.env_in = MLP([env_in_dim, n_hidden * 2, n_hidden], key=keys[5])
        self.win_in = MLP([win_in_dim, n_hidden * 2, n_hidden], key=keys[6])
        self.env_head = _linear(n_hidden, N_STATE_CH, key=keys[2])
        self.win_head = _linear(n_hidden, N_STATE_CH, key=keys[3])

        bkeys = keys[7:]
        if self.rope:
            self.blocks = tuple(
                _zero_rope_mod(
                    _RoPEDiTBlock(n_hidden, n_head, cond_dim, mlp_ratio=float(mlp_ratio), key=k)
                )
                for k in bkeys
            )
        else:
            self.blocks = tuple(
                _zero_modulation(
                    DiTBlock(n_hidden, n_head, cond_dim, key=k,
                             mlp_ratio=float(mlp_ratio), norm_affine=False)
                )
                for k in bkeys
            )
        self.ln_out = LayerNorm(n_hidden)

    def __call__(self, x, t, cond):
        c = jnp.concatenate([self.time_embed(t.reshape(1)), self.cond_embed(cond)], axis=-1)
        env, win = x[: self.n_env], x[self.n_env :]
        env_mu = env[:, MU_SL]
        b = self.mu_fourier_bands
        env_feat = jnp.concatenate(
            [env, jnp.broadcast_to(self.env_type, (self.n_env, self.env_type.shape[0])),
             _sincos(env_mu, b)], axis=-1
        )
        win_feat = jnp.concatenate(
            [win, jnp.broadcast_to(self.win_type, (self.n_window, self.win_type.shape[0])),
             self.win_embed, _sincos(self.scaffold_mu, b)], axis=-1
        )
        h = jnp.concatenate([self.env_in(env_feat), self.win_in(win_feat)], axis=0)
        if self.rope:
            coords = jnp.concatenate([env_mu, self.scaffold_mu], axis=0)
            for block in self.blocks:
                h = block(h, c, coords)
        else:
            for block in self.blocks:
                h = block(h, c)
        h = self.ln_out(h)
        out_env = _apply(self.env_head, h[: self.n_env])
        out_win = _apply(self.win_head, h[self.n_env :])
        return jnp.concatenate([out_env, out_win], axis=0)
