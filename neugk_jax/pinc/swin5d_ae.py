"""Swin5DAE — deterministic autoencoder built on Swin5DUnet.

The bottleneck inserts two extra global-attention stages (``middle_pre`` /
``middle_post``) around a channel projection that compresses the latent
dimension. Encoder and decoder can be conditioned separately on scalar
conditions (DiT modulation); the model takes the condition vector ordered by the
sorted union of both key sets.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.base import TokenLayerBase
from neugk_jax.models.embeddings import ContinuousConditionEmbed
from neugk_jax.models.gk_unet import Swin5DUnet
from neugk_jax.models.layers import token_layer
from neugk_jax.models.patching import CConvUnpatch
from neugk_jax.models.spec import Spec
from neugk_jax.models.tokens import TokenExpand
from neugk_jax.models.utils import LayerNorm, Linear, gelu, make_norm, split_key, trainable_mask


class Swin5DAE(eqx.Module):
    """Wraps Swin5DUnet with a bottleneck projection.

    ``layer`` / ``bottleneck_layer`` are the token-layer specs of the U-Net stages and of the two
    bottleneck stages; ``patching`` and ``token_pe`` as for :class:`SwinNDUnet`.
    """

    backbone: Swin5DUnet
    enc_cond_embed: Optional[ContinuousConditionEmbed]
    dec_cond_embed: Optional[ContinuousConditionEmbed]
    middle_pre: TokenLayerBase
    middle_post: TokenLayerBase
    middle_downproj: Linear
    middle_upproj: Linear
    middle_upscale: TokenExpand
    pre_z_norm: Optional[LayerNorm]
    post_z_norm: Optional[LayerNorm]
    input_norm: object | None

    bottleneck_dim: int = eqx.field(static=True)
    bottleneck_grid_size: tuple[int, ...] = eqx.field(static=True)
    normalized_latent: bool = eqx.field(static=True)
    condition_keys: tuple[str, ...] = eqx.field(static=True)
    enc_indices: Optional[tuple[int, ...]] = eqx.field(static=True)
    dec_indices: Optional[tuple[int, ...]] = eqx.field(static=True)

    def __init__(
        self,
        *,
        space: int = 5,
        decouple_mu: bool = False,
        mu_axis: int = 1,
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
        input_norm: bool = False,
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
        rms_norm: bool = True,
        legacy_double_shortcut: bool = False,
        decoder_rms_norm: bool = False,
        use_checkpoint: bool = False,
        encoder_conditioning: Sequence[str] = (),
        decoder_conditioning: Sequence[str] = (),
        cond_embed_dim: int = 32,
        merge_mask: Optional[Sequence[bool]] = None,
        readout_mult: float = 1.0,
        attention: str = "einsum",
        patching: Spec = "linear",
        patching_kwargs: Optional[Mapping] = None,
        layer: Spec = "swin",
        bottleneck_layer: Spec = "vit",
        token_pe: Optional[Spec] = None,
        key,
    ):
        kb, k1, k2, k3, k4, k5 = jr.split(key, 6)
        enc_keys = tuple(sorted(encoder_conditioning or ()))
        dec_keys = tuple(sorted(decoder_conditioning or ()))
        union = tuple(sorted(set(enc_keys) | set(dec_keys)))
        self.condition_keys = union
        self.enc_indices = tuple(union.index(k) for k in enc_keys) if enc_keys else None
        self.dec_indices = tuple(union.index(k) for k in dec_keys) if dec_keys else None

        def cond_embed(keys, i):
            if not keys:
                return None
            return ContinuousConditionEmbed(cond_embed_dim, len(keys), key=jr.fold_in(key, i))

        self.enc_cond_embed = cond_embed(enc_keys, 7)
        self.dec_cond_embed = cond_embed(dec_keys, 8)
        enc_cdim = self.enc_cond_embed.cond_dim if enc_keys else 0
        dec_cdim = self.dec_cond_embed.cond_dim if dec_keys else 0
        self.backbone = Swin5DUnet(
            space=space,
            decouple_mu=decouple_mu,
            mu_axis=mu_axis,
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
            use_checkpoint=use_checkpoint,
            rms_norm=rms_norm,
            decoder_rms_norm=decoder_rms_norm,
            enc_cond_dim=enc_cdim,
            dec_cond_dim=dec_cdim,
            merge_mask=merge_mask,
            readout_mult=readout_mult,
            attention=attention,
            patching=patching,
            patching_kwargs=patching_kwargs,
            layer=layer,
            token_pe=token_pe,
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

        # bottleneck vit blocks use an affine norm regardless of the encoder setting
        vit_kw = dict(
            mlp_ratio=hidden_mlp_ratio,
            drop_path=drop_path,
            act_fn=act_fn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            gated_attention=gated_attention,
            norm_affine=True,
            rms_norm=rms_norm,
            attention=attention,
        )
        # a windowed bottleneck layer takes the whole grid as its window
        grid_kw = dict(grid_size=mid_grid, window_size=mid_grid)
        self.middle_pre = token_layer(
            bottleneck_layer,
            mid_dim,
            bottleneck_depth,
            bottleneck_num_heads,
            key=k1,
            cond_dim=enc_cdim,
            **grid_kw,
            **vit_kw,
        )
        self.middle_post = token_layer(
            bottleneck_layer,
            mid_dim,
            bottleneck_depth,
            bottleneck_num_heads,
            key=k2,
            cond_dim=dec_cdim,
            **grid_kw,
            **vit_kw,
        )
        # norm of the merged tokens entering the bottleneck
        self.input_norm = make_norm(mid_dim, rms=rms_norm) if input_norm else None
        self.middle_downproj = Linear(mid_dim, bd, key=k3)
        self.middle_upproj = Linear(bd, mid_dim, key=k4)
        self.middle_upscale = TokenExpand(
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

    @staticmethod
    def _embed(embed, condition, indices):
        if embed is None:
            return None
        if condition is None:
            raise ValueError("this autoencoder is conditioned; pass `condition`")
        return embed(condition[jnp.asarray(indices)])

    def bottleneck_input(self, z: jnp.ndarray) -> jnp.ndarray:
        return z if self.input_norm is None else self.input_norm(z)

    def bottleneck(self, z: jnp.ndarray, *, inference: bool = True) -> tuple[jnp.ndarray, dict]:
        return z, {}

    def encode(
        self, df: jnp.ndarray, condition=None, *, geometry=None, key=None, inference: bool = True
    ) -> jnp.ndarray:
        cond = self._embed(self.enc_cond_embed, condition, self.enc_indices)
        keys = split_key(key, len(self.backbone.down_blocks) + 1)
        z = self.backbone.patch_encode(df, geometry)
        for blk, k in zip(self.backbone.down_blocks, keys):
            z = blk(z, cond, return_skip=False, key=k, inference=inference)
        z = self.middle_downproj(
            self.middle_pre(self.bottleneck_input(z), cond, key=keys[-1], inference=inference)
        )
        return self.pre_z_norm(z) if self.normalized_latent else z

    def decode(
        self, z: jnp.ndarray, condition=None, *, geometry=None, key=None, inference: bool = True
    ):
        cond = self._embed(self.dec_cond_embed, condition, self.dec_indices)
        keys = split_key(key, len(self.backbone.up_blocks) + 1)
        if self.normalized_latent:
            z = self.post_z_norm(z)
        z = self.middle_post(self.middle_upproj(z), cond, key=keys[0], inference=inference)
        z = self.middle_upscale(z)
        # no skip connections in the ae decoder
        for blk, k in zip(self.backbone.up_blocks, keys[1:]):
            z = blk(z, None, cond, key=k, inference=inference)
        return {"df": self.backbone.patch_decode(z, cond, geometry)}

    def __call__(
        self,
        df: jnp.ndarray,
        condition=None,
        return_latent: bool = False,
        *,
        geometry=None,
        key=None,
        inference: bool = True,
    ):
        k_enc, k_dec = split_key(key, 2)
        z = self.encode(df, condition, geometry=geometry, key=k_enc, inference=inference)
        z, extra = self.bottleneck(z, inference=inference)
        out = self.decode(z, condition, geometry=geometry, key=k_dec, inference=inference)
        out.update(extra)
        if return_latent:
            out["latent"] = z
        return out


def _is_none(x):
    return x is None


def _trunk(backbone):
    return backbone.down_blocks, backbone.up_blocks


def _trunk_param_index(backbone) -> tuple[int, ...]:
    # positions of the trainable trunk arrays among the trunk leaves (None nodes counted)
    trunk = _trunk(backbone)
    arrays = jax.tree_util.tree_leaves(trunk)
    params = {id(a) for a, m in zip(arrays, jax.tree_util.tree_leaves(trainable_mask(trunk))) if m}
    leaves = jax.tree_util.tree_leaves(trunk, is_leaf=_is_none)
    return tuple(i for i, leaf in enumerate(leaves) if id(leaf) in params)


def _trunk_params(backbone, index):
    leaves = jax.tree_util.tree_leaves(_trunk(backbone), is_leaf=_is_none)
    return [leaves[i] for i in index]


def _species_grid(grid):
    # species in front: a relative axis of patch 1, no coordinate of its own
    return {**grid, "axes": [{"kind": "relative"}, *grid["axes"]]}


def _grid_only(layer):
    return eqx.filter(layer, trainable_mask(layer), inverse=True)


class KineticSwin5DAE(Swin5DAE):
    """Swin5DAE over ``(C, species, vpar, mu, s, x, y)`` df with one stem per data grid.

    Species is an unpatched spatial axis with full-axis windows, identified by a learned
    ``species_embed``; with ``decouple_mu`` every stem folds its own mu axis into the channels.
    Every stem (``stems[name]``: ``resolution`` (vpar, mu, s, x, y),
    ``n_species``, ``patch_size``, optional ``window_size`` / ``in_channels`` /
    ``out_channels`` / ``grid``) has its own ``vel_pe`` and window layout buffers. With linear
    ``patching`` every stem has its own patch embedding and unpatch; with continuous-convolution
    ``patching`` (cconv, tucker) the stems share the primary stem's, each on the points of its
    ``grid`` (the point grid spec of the stem's (vpar, s, x, y), mu folded; species is prepended as
    a relative axis of patch 1). The trunk weights are those of the primary stem (the one with
    the most species, whatever the order of ``stems``), and every stem must give them the same
    shapes and reach the same latent grid apart from the species extent. A call runs the stem
    named by ``stem``, else the one whose input shape matches the df; ``decode`` defaults to the
    primary stem.
    """

    species_embed: jax.Array
    stem_backbones: dict
    primary_stem: str = eqx.field(static=True)
    trunk_index: tuple[int, ...] = eqx.field(static=True)
    stem_inputs: dict = eqx.field(static=True)

    def __init__(
        self,
        *,
        stems: Mapping[str, Mapping],
        in_channels: int,
        out_channels: int,
        window_size,
        n_species: int = 2,
        patching_kwargs: Optional[Mapping] = None,
        key,
        **kwargs,
    ):
        # the primary stem (shared layers, upscale target, DiT grid) has the most species
        names = sorted(stems, key=lambda n: -int(stems[n].get("n_species", n_species)))

        def stem_kwargs(spec):
            ns = int(spec.get("n_species", n_species))
            grid = {"grid": _species_grid(spec["grid"])} if spec.get("grid") else {}
            # species: unpatched, full-axis attention, never partitioned or shifted
            return dict(
                space=6,
                mu_axis=2,
                base_resolution=[ns, *spec["resolution"]],
                patch_size=[1, *spec["patch_size"]],
                window_size=[math.inf, *spec.get("window_size", window_size)],
                # species tokens are never merged (with each other or downsampled)
                merge_mask=[False, *[True] * len(spec["patch_size"])],
                in_channels=int(spec.get("in_channels", in_channels)),
                out_channels=int(spec.get("out_channels", out_channels)),
                patching_kwargs={**(patching_kwargs or {}), **grid},
            )

        k_ae, k_sp = jr.split(key)
        super().__init__(key=k_ae, **stem_kwargs(stems[names[0]]), **kwargs)
        self.stem_inputs = {}
        for name in names:
            kw = stem_kwargs(stems[name])
            self.stem_inputs[name] = (kw["in_channels"], *kw["base_resolution"])
        self.primary_stem = names[0]
        self.trunk_index = _trunk_param_index(self.backbone)
        shared = _trunk_params(self.backbone, self.trunk_index)
        self.stem_backbones = {}
        for i, name in enumerate(names[1:], 1):
            variant = Swin5DAE(key=jr.fold_in(k_ae, i), **stem_kwargs(stems[name]), **kwargs)
            bb = variant.backbone
            if _trunk_param_index(bb) != self.trunk_index:
                raise ValueError(f"stem {name!r}: trunk structure differs from stem {names[0]!r}")
            for a, b in zip(shared, _trunk_params(bb, self.trunk_index)):
                if a.shape != b.shape:
                    raise ValueError(f"stem {name!r}: trunk weight {b.shape} != {a.shape}")
            if bb.grid_sizes[-1][1:] != self.backbone.grid_sizes[-1][1:]:
                raise ValueError(
                    f"stem {name!r}: latent grid {bb.grid_sizes[-1]}"
                    f" != {self.backbone.grid_sizes[-1]}"
                )
            if isinstance(bb.unpatch, CConvUnpatch):
                bb = eqx.tree_at(
                    lambda t: (t.patch_embed, t.unpatch),
                    bb,
                    (_grid_only(bb.patch_embed), _grid_only(bb.unpatch)),
                )
            # trunk (and cconv patching) weights are taken from the primary backbone at call time
            self.stem_backbones[name] = eqx.tree_at(
                lambda t: _trunk_params(t, self.trunk_index),
                bb,
                replace=[None] * len(self.trunk_index),
                is_leaf=_is_none,
            )
        self.species_embed = 0.02 * jr.normal(k_sp, (n_species, self.backbone.down_dims[0]))

    def stem_for(self, shape) -> str:
        """The stem whose input shape ``(C, species, vpar, mu, s, x, y)`` is ``shape``."""
        hits = [n for n, s in self.stem_inputs.items() if tuple(s) == tuple(shape)]
        if len(hits) != 1:
            raise ValueError(
                f"input {tuple(shape)} matches stems {hits}; inputs: {self.stem_inputs}"
            )
        return hits[0]

    def stem_backbone(self, stem: Optional[str] = None):
        """The backbone of ``stem``: its own window layout and patching (or point grids of the shared
        cconv patching), the shared trunk."""
        if stem is None or stem == self.primary_stem:
            return self.backbone
        if stem not in self.stem_backbones:
            raise KeyError(
                f"unknown stem {stem!r}; one of {[self.primary_stem, *self.stem_backbones]}"
            )
        bb = eqx.tree_at(
            lambda t: _trunk_params(t, self.trunk_index),
            self.stem_backbones[stem],
            replace=_trunk_params(self.backbone, self.trunk_index),
            is_leaf=_is_none,
        )
        if not isinstance(bb.unpatch, CConvUnpatch):
            return bb
        primary = self.backbone
        return eqx.tree_at(
            lambda t: (t.patch_embed, t.unpatch),
            bb,
            (
                primary.patch_embed.with_grid(bb.patch_embed.grid),
                primary.unpatch.with_grid(bb.unpatch.grid),
            ),
        )

    def encode(self, df, condition=None, *, stem=None, key=None, inference: bool = True):
        bb = self.stem_backbone(stem or self.stem_for(df.shape))
        cond = self._embed(self.enc_cond_embed, condition, self.enc_indices)
        keys = split_key(key, len(bb.down_blocks) + 1)
        z = bb.patch_encode(df)
        z = z + self.species_embed[: z.shape[0]].reshape(-1, *(1,) * (z.ndim - 2), z.shape[-1])
        for blk, k in zip(bb.down_blocks, keys):
            z = blk(z, cond, return_skip=False, key=k, inference=inference)
        z = self.middle_downproj(
            self.middle_pre(self.bottleneck_input(z), cond, key=keys[-1], inference=inference)
        )
        return self.pre_z_norm(z) if self.normalized_latent else z

    def decode(self, z, condition=None, *, stem=None, key=None, inference: bool = True):
        bb = self.stem_backbone(stem)
        cond = self._embed(self.dec_cond_embed, condition, self.dec_indices)
        keys = split_key(key, len(bb.up_blocks) + 1)
        if self.normalized_latent:
            z = self.post_z_norm(z)
        z = self.middle_post(self.middle_upproj(z), cond, key=keys[0], inference=inference)
        # the shared upscale is built for the primary grid; crop to this stem's grid
        z = self.middle_upscale(z, target_grid_size=bb.grid_sizes[-2])
        for blk, k in zip(bb.up_blocks, keys[1:]):
            z = blk(z, None, cond, key=k, inference=inference)
        return {"df": bb.patch_decode(z, cond)}

    def __call__(
        self,
        df,
        condition=None,
        return_latent: bool = False,
        *,
        stem=None,
        key=None,
        inference: bool = True,
    ):
        k_enc, k_dec = split_key(key, 2)
        stem = stem or self.stem_for(df.shape)
        z = self.encode(df, condition, stem=stem, key=k_enc, inference=inference)
        z, extra = self.bottleneck(z, inference=inference)
        out = self.decode(z, condition, stem=stem, key=k_dec, inference=inference)
        out.update(extra)
        if return_latent:
            out["latent"] = z
        return out


class StemView(eqx.Module):
    """A ``KineticSwin5DAE`` bound to one stem, callable like a plain autoencoder."""

    model: KineticSwin5DAE
    stem: str = eqx.field(static=True)

    def __call__(
        self, df, condition=None, return_latent: bool = False, *, key=None, inference=True
    ):
        return self.model(
            df, condition, return_latent, stem=self.stem, key=key, inference=inference
        )
