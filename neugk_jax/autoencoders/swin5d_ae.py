"""Swin5DAE — deterministic autoencoder built on Swin5DUnet.

The bottleneck inserts two extra global-attention stages (``middle_pre`` /
``middle_post``) around a channel projection that compresses the latent
dimension.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.gk_unet import Swin5DUnet
from neugk_jax.models.patching import PatchExpand
from neugk_jax.models.swin import BlockStack
from neugk_jax.models.utils import LayerNorm, Linear, gelu, split_key
from neugk_jax.models.vit import vit_layer


class Swin5DAE(eqx.Module):
    """Wraps Swin5DUnet with a bottleneck projection."""

    backbone: Swin5DUnet
    middle_pre: BlockStack
    middle_post: BlockStack
    middle_downproj: Linear
    middle_upproj: Linear
    middle_upscale: PatchExpand
    pre_z_norm: Optional[LayerNorm]
    post_z_norm: Optional[LayerNorm]

    bottleneck_dim: int = eqx.field(static=True)
    bottleneck_grid_size: tuple[int, ...] = eqx.field(static=True)
    normalized_latent: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        space: int = 5,
        decouple_mu: bool = False,
        dim: int,
        base_resolution: Sequence[int],
        in_channels: int,
        out_channels: int,
        patch_size,
        window_size,
        depth,
        num_heads,
        num_layers: int = 4,
        bottleneck_dim: Optional[int] = None,
        bottleneck_depth: int = 2,
        bottleneck_num_heads: int = 2,
        normalized_latent: bool = False,
        c_multiplier: int = 2,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        merging_hidden_ratio: float = 8.0,
        unmerging_hidden_ratio: float = 8.0,
        merging_depth: int = 2,
        unmerging_depth: int = 2,
        act_fn: Callable = gelu,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = False,
        legacy_double_shortcut: bool = False,
        decoder_rms_norm: bool = False,
        key,
    ):
        kb, k1, k2, k3, k4, k5 = jr.split(key, 6)
        self.backbone = Swin5DUnet(
            space=space,
            decouple_mu=decouple_mu,
            dim=dim,
            base_resolution=base_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            window_size=window_size,
            depth=depth,
            num_heads=num_heads,
            num_layers=num_layers,
            c_multiplier=c_multiplier,
            drop_path=drop_path,
            hidden_mlp_ratio=hidden_mlp_ratio,
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            merging_depth=merging_depth,
            unmerging_depth=unmerging_depth,
            act_fn=act_fn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_rpb=use_rpb,
            gated_attention=gated_attention,
            norm_affine=norm_affine,
            legacy_double_shortcut=legacy_double_shortcut,
            rms_norm=True,
            decoder_rms_norm=decoder_rms_norm,
            # ae has no encoder→decoder skips and its own bottleneck
            up_use_skip=False,
            build_middle=False,
            key=kb,
        )

        # bottleneck dims derived from the deepest encoder grid
        mid_dim = self.backbone.down_dims[-1]
        mid_grid = self.backbone.grid_sizes[-1]
        bd = bottleneck_dim or mid_dim

        self.bottleneck_dim = bd
        self.bottleneck_grid_size = mid_grid

        # bottleneck vit blocks use affine RMSNorm regardless of the encoder setting
        vit_kw = dict(
            mlp_ratio=hidden_mlp_ratio,
            drop_path=drop_path,
            act_fn=act_fn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            gated_attention=gated_attention,
            norm_affine=True,
            rms_norm=True,
        )
        self.middle_pre = vit_layer(
            mid_dim, bottleneck_depth, bottleneck_num_heads, key=k1, **vit_kw
        )
        self.middle_post = vit_layer(
            mid_dim, bottleneck_depth, bottleneck_num_heads, key=k2, **vit_kw
        )
        self.middle_downproj = Linear(mid_dim, bd, key=k3)
        self.middle_upproj = Linear(bd, mid_dim, key=k4)
        self.middle_upscale = PatchExpand(
            mid_dim,
            mid_grid,
            key=k5,
            target_grid_size=self.backbone.grid_sizes[-2],
            c_multiplier=c_multiplier,
            mlp_depth=1,
            rms_norm=decoder_rms_norm,
        )

        if normalized_latent:
            self.pre_z_norm = LayerNorm(bd)
            self.post_z_norm = LayerNorm(bd)
        else:
            self.pre_z_norm = None
            self.post_z_norm = None
        self.normalized_latent = normalized_latent

    def encode(self, df: jnp.ndarray, *, key=None, inference: bool = True) -> jnp.ndarray:
        keys = split_key(key, len(self.backbone.down_blocks) + 1)
        z = self.backbone.patch_encode(df)
        for blk, k in zip(self.backbone.down_blocks, keys):
            z = blk(z, return_skip=False, key=k, inference=inference)
        z = self.middle_downproj(self.middle_pre(z, key=keys[-1], inference=inference))
        return self.pre_z_norm(z) if self.normalized_latent else z

    def decode(self, z: jnp.ndarray, *, key=None, inference: bool = True):
        keys = split_key(key, len(self.backbone.up_blocks) + 1)
        if self.normalized_latent:
            z = self.post_z_norm(z)
        z = self.middle_post(self.middle_upproj(z), key=keys[0], inference=inference)
        z = self.middle_upscale(z)
        # no skip connections in the ae decoder
        for blk, k in zip(self.backbone.up_blocks, keys[1:]):
            z = blk(z, key=k, inference=inference)
        return {"df": self.backbone.patch_decode(z)}

    def __call__(
        self, df: jnp.ndarray, return_latent: bool = False, *, key=None, inference: bool = True
    ):
        k_enc, k_dec = split_key(key, 2)
        z = self.encode(df, key=k_enc, inference=inference)
        out = self.decode(z, key=k_dec, inference=inference)
        if return_latent:
            out["latent"] = z
        return out
