"""Latent DiT conditioned on the paired linear-run field.

``LinearFieldEncoder`` turns one 5D linear eigenmode field into conditioning:
patch embed → Swin stages (each ending in a PatchMerge downsample) → either the
grid tokens or a max-pooled code. It is the AE encoder's shape pipeline with the
bottleneck replaced by a pool, so it inherits the same patch/window config.

``LinearCondDiT`` consumes that in one of two switchable modes:

* ``cond_mode="adaln"`` — the pooled code is concatenated with the time
  embedding and drives the DiT scale/shift/gate modulation.
* ``cond_mode="cross"`` — the grid tokens are the cross-attention context in
  every block (Stable Diffusion's ``BasicTransformerBlock`` order); the time
  embedding alone drives the modulation.

There is no scalar-conditioning path: the physical parameters (itg, dg, s_hat, q)
cannot reach this model.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from neugk_jax.models.embeddings import APE, ContinuousConditionEmbed
from neugk_jax.models.gk_unet import SwinBlockDown, _as_seq
from neugk_jax.models.patching import PatchEmbed, pad_to_blocks
from neugk_jax.models.utils import LayerNorm, Linear, gelu
from neugk_jax.models.vit import CrossAttnDiTLayer, DiTLayer


class LinearFieldEncoder(eqx.Module):
    """5D field → conditioning tokens or a pooled code.

    ``tokens(field)`` returns the ``(n_tokens, out_dim)`` grid at the deepest
    stage; ``__call__`` pools those to a single ``(code_dim,)`` vector.
    """

    vel_pe: Optional[APE]
    patch_embed: PatchEmbed
    blocks: list
    norm: LayerNorm
    proj: Linear

    decouple_mu: bool = eqx.field(static=True)
    decoupled_dim: int = eqx.field(static=True)
    full_resolution: tuple[int, ...] = eqx.field(static=True)
    patch_size: tuple[int, ...] = eqx.field(static=True)
    grid_sizes: tuple = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    code_dim: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    field_ndim: int = eqx.field(static=True)
    pool: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        base_resolution: Sequence[int],
        in_channels: int,
        patch_size,
        window_size,
        dim: int,
        depth,
        num_heads,
        code_dim: int,
        key,
        space: int = 5,
        decouple_mu: bool = True,
        num_layers: int = 2,
        c_multiplier: int = 1,
        drop_path: float = 0.0,
        mlp_ratio: float = 2.0,
        merging_depth: int = 2,
        merging_hidden_ratio: float = 1.0,
        pool: str = "max",
        act_fn: Callable = gelu,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = True,
        gated_attention: bool = False,
        norm_affine: bool = False,
        rms_norm: bool = True,
    ):
        assert pool in ("max", "mean", "maxmean")
        full_resolution = tuple(base_resolution)
        patch_size = _as_seq(patch_size, space)
        window_size = _as_seq(window_size, space)
        decoupled_dim = 0
        if decouple_mu:
            # fold mu into the channel axis (mirrors Swin5DUnet), so the stack is 4D
            space = 4
            decoupled_dim = base_resolution[1]
            base_resolution = [base_resolution[0]] + list(base_resolution[2:])
            patch_size = [patch_size[0]] + list(patch_size[2:])
            window_size = [window_size[0]] + list(window_size[2:])
            embed_channels = in_channels * decoupled_dim
        else:
            embed_channels = in_channels
        depth = _as_seq(depth, num_layers)
        num_heads = _as_seq(num_heads, num_layers)
        padded_base = [
            s if p in (0, 1) or s % p == 0 else s + (p - s % p)
            for s, p in zip(base_resolution, patch_size)
        ]

        keys = jr.split(key, num_layers + 3)
        self.patch_embed = PatchEmbed(
            padded_base, patch_size, in_channels=embed_channels, embed_dim=dim,
            key=keys[0], mlp_depth=merging_depth, mlp_ratio=merging_hidden_ratio,
            rms_norm=rms_norm, act_fn=act_fn,
        )
        grid_sizes = [self.patch_embed.grid_size]
        dims = [dim]
        blocks = []
        for i in range(num_layers):
            blk = SwinBlockDown(
                space, dims[i], grid_size=grid_sizes[i], window_size=window_size,
                num_heads=num_heads[i], depth=depth[i], key=keys[i + 1],
                drop_path=drop_path, mlp_ratio=mlp_ratio, c_multiplier=c_multiplier,
                act_fn=act_fn, qkv_bias=qkv_bias, qk_norm=qk_norm,
                use_rpb=use_rpb, gated_attention=gated_attention,
                norm_affine=norm_affine, rms_norm=rms_norm,
            )
            blocks.append(blk)
            dims.append(blk.out_dim)
            grid_sizes.append(blk.resampled_grid_size)
        self.blocks = blocks
        self.grid_sizes = tuple(grid_sizes)
        self.out_dim = dims[-1]
        pooled = self.out_dim * (2 if pool == "maxmean" else 1)
        self.norm = LayerNorm(pooled)
        self.proj = Linear(pooled, code_dim, key=keys[-1])

        self.vel_pe = (
            APE(in_channels, (1, decoupled_dim, 1, 1, 1), init="normal",
                learnable=True, key=keys[-2])
            if decouple_mu else None
        )
        self.decouple_mu = decouple_mu
        self.decoupled_dim = decoupled_dim
        self.full_resolution = full_resolution
        self.patch_size = tuple(patch_size)
        self.code_dim = code_dim
        self.in_channels = in_channels
        self.field_ndim = 1 + len(full_resolution)
        self.pool = pool

    def tokens(self, field: jnp.ndarray, *, key=None, inference: bool = True) -> jnp.ndarray:
        x = jnp.moveaxis(field, 0, -1)
        if self.decouple_mu:
            x = self.vel_pe(x)
            x = rearrange(x, "vp mu s x y c -> vp s x y (c mu)")
        x, _ = pad_to_blocks(x, self.patch_size)
        x = self.patch_embed(x)
        keys = jr.split(key, len(self.blocks)) if key is not None else [None] * len(self.blocks)
        for blk, k in zip(self.blocks, keys):
            x = blk(x, key=k, inference=inference, return_skip=False)
        return x.reshape(-1, x.shape[-1])

    def __call__(self, field: jnp.ndarray, *, key=None, inference: bool = True) -> jnp.ndarray:
        tok = self.tokens(field, key=key, inference=inference)
        if self.pool == "max":
            pooled = jnp.max(tok, axis=0)
        elif self.pool == "mean":
            pooled = jnp.mean(tok, axis=0)
        else:
            pooled = jnp.concatenate([jnp.max(tok, axis=0), jnp.mean(tok, axis=0)])
        return self.proj(self.norm(pooled))


class LinearCondDiT(eqx.Module):
    """Latent DiT conditioned only on the linear field.

    Forward signature (per sample): ``__call__(x: (*grid, z_dim), tstep, condition)``
    where ``condition`` is the raw field ``(C, vp, mu, s, kx, ky)`` — encoded inline so
    the encoder trains — or its precomputed :meth:`encode_cond` output (a code in
    ``adaln`` mode, tokens in ``cross`` mode), which is what the sampler passes since
    the conditioning is constant along the integration path.
    """

    encoder: list
    ape: APE
    backbone: object
    decoder: Linear
    time_embed: ContinuousConditionEmbed
    lin_encoder: LinearFieldEncoder
    lin_proj: Optional[Linear]
    ctx_proj: Optional[Linear]
    act: Callable = eqx.field(static=True)
    cond_mode: str = eqx.field(static=True)
    z_dim: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)
    latent_shape: tuple[int, ...] = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    code_dim: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        space: int,
        z_dim: int,
        dim: int,
        grid_size: Sequence[int],
        depth: int,
        num_heads: int,
        linear_encoder: LinearFieldEncoder,
        key,
        cond_mode: str = "adaln",
        time_embed_dim: int = 32,
        lin_embed_dim: int = 32,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
    ):
        assert cond_mode in ("adaln", "cross")
        keys = jr.split(key, 6)
        self.time_embed = ContinuousConditionEmbed(time_embed_dim, 1, key=keys[0])
        self.lin_encoder = linear_encoder
        self.cond_mode = cond_mode

        if cond_mode == "adaln":
            # match the scalar-embed convention (4x width, silu) used for the timestep
            self.lin_proj = Linear(linear_encoder.code_dim, 4 * lin_embed_dim, key=keys[1])
            self.ctx_proj = None
            cdim = self.time_embed.cond_dim + 4 * lin_embed_dim
        else:
            self.lin_proj = None
            self.ctx_proj = Linear(linear_encoder.out_dim, dim, key=keys[1])
            cdim = self.time_embed.cond_dim
        self.cond_dim = cdim
        self.code_dim = linear_encoder.code_dim

        self.encoder = [Linear(z_dim, dim, key=keys[2], use_bias=False)]
        self.ape = APE(dim, grid_size, init="normal", learnable=True, key=keys[3])
        common = dict(
            space=space, dim=dim, depth=depth, num_heads=num_heads,
            grid_size=grid_size, key=keys[4], cond_dim=cdim,
            mlp_ratio=mlp_ratio, drop_path=drop_path, act_fn=act_fn,
        )
        self.backbone = (
            DiTLayer(**common) if cond_mode == "adaln"
            else CrossAttnDiTLayer(context_dim=dim, **common)
        )
        self.decoder = Linear(dim, z_dim, key=keys[5], use_bias=False)

        self.act = act_fn
        self.z_dim = z_dim
        self.dim = dim
        self.grid_size = tuple(grid_size)
        self.latent_shape = (*tuple(grid_size), z_dim)

    def encode_cond(self, field: jnp.ndarray, *, key=None, inference: bool = True) -> jnp.ndarray:
        # what the backbone consumes: a pooled code (adaln) or grid tokens (cross)
        if self.cond_mode == "adaln":
            return self.lin_encoder(field, key=key, inference=inference)
        return self.lin_encoder.tokens(field, key=key, inference=inference)

    def __call__(
        self,
        x: jnp.ndarray,
        tstep: jnp.ndarray,
        condition: jnp.ndarray,
        *,
        key=None,
        inference: bool = True,
    ) -> jnp.ndarray:
        k_lin, k_bb = (None, None) if key is None else jr.split(key, 2)
        if condition.ndim == self.lin_encoder.field_ndim:
            condition = self.encode_cond(condition, key=k_lin, inference=inference)

        t_emb = self.time_embed(jnp.asarray(tstep).reshape((1,)))
        h = self.act(self.encoder[0](x))
        h = self.ape(h)
        if self.cond_mode == "adaln":
            cond = jnp.concatenate([t_emb, jax.nn.silu(self.lin_proj(condition))], axis=-1)
            h = self.backbone(h, cond, key=k_bb, inference=inference)
        else:
            h = self.backbone(h, t_emb, self.ctx_proj(condition),
                              key=k_bb, inference=inference)
        return self.decoder(h)
