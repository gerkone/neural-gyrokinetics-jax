"""GyroSwin multitask model — JAX/Equinox port of ``neugk/gyroswin/models/gyroswin.py``.

Composition (mirrors upstream ``GyroSwinMultitask`` after ``__init__``):

* ``df_unet``: ``Swin5DUnet`` — full 5D Swin U-Net on the distribution function.
* ``phi_unet``: ``SwinNDUnet`` (space=3) — only the **up** path is kept;
  the down path's outputs come from ``vspace_attn_down`` reducing the df features.
* ``vspace_attn_down`` / ``vspace_attn_middle`` / ``vspace_attn_patch_skip``:
  ``VSpaceReduce`` blocks that turn the 5D df latents into the 3D phi shape.
* ``df_mix_middle`` / ``phi_mix_middle``: bottleneck cross-attention.
* ``df_mix_up`` / ``phi_mix_up``: up-path cross-attention at each scale.

The flux head and baseline models (FNO/PointNet/...) are out of scope —
they were never needed by the user's training configs.

NOTE: conditioning (DiT modulation through the Swin blocks) is plumbed
via the existing ``Swin5DUnet`` API. For configs without conditioning the
forward still works.
"""

from __future__ import annotations

from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from neugk_jax.gyroswin.models.x_layers import FluxDecoder, MixingBlock, VSpaceReduce
from neugk_jax.models.embeddings import ContinuousConditionEmbed
from neugk_jax.models.gk_unet import Swin5DUnet, SwinNDUnet


class GyroSwinMultitask(eqx.Module):
    """5D df → 5D df + 3D phi prediction with cross-attention mixing."""

    df_unet: Swin5DUnet
    phi_unet: SwinNDUnet
    cond_embed: Optional[ContinuousConditionEmbed]
    vspace_attn_down: list
    vspace_attn_middle: VSpaceReduce
    vspace_attn_patch_skip: Optional[VSpaceReduce]
    df_mix_middle: MixingBlock
    phi_mix_middle: MixingBlock
    df_mix_up: list
    phi_mix_up: list
    df_mix_unpatch: MixingBlock
    phi_mix_unpatch: MixingBlock
    flux_head: Optional[FluxDecoder]
    use_flux: bool = eqx.field(static=True)
    use_phi: bool = eqx.field(static=True)
    patch_skip: bool = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    n_cond: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        dim: int,
        df_base_resolution: Sequence[int],
        df_patch_size: Sequence[int],
        df_window_size: Sequence[int],
        depth: int,
        num_heads: int,
        in_channels: int,
        out_channels: int,
        num_layers: int = 4,
        c_multiplier: int = 2,
        merging_hidden_ratio: float = 4.0,
        unmerging_hidden_ratio: float = 8.0,
        decouple_mu: bool = True,
        patch_skip: bool = True,
        swin_bottleneck: bool = True,
        use_rpb: bool = True,
        qk_norm: bool = True,
        gated_attention: bool = True,
        outputs: Sequence[str] = ("df", "phi"),
        n_cond: int = 0,
        cond_embed_dim: int = 128,
        cond_mode: str = "film",
        flux_num_heads: int = 4,
        flux_depth: int = 1,
        use_checkpoint: bool = False,
        key,
    ):
        self.latent_dim = dim
        self.patch_skip = patch_skip
        self.use_phi = "phi" in outputs
        self.use_flux = ("flux" in outputs) or ("fluxavg" in outputs)
        self.n_cond = n_cond

        phi_base_resolution = tuple(df_base_resolution[2:])  # (s, x, y)
        phi_patch_size = tuple(df_patch_size[2:])
        phi_window_size = tuple(df_window_size[2:])

        keys = jr.split(key, 18)
        # conditioning embeds live INSIDE each U-Net (torch parity: df_unet.cond_embed,
        # phi_unet.cond_embed), not at the top level.
        self.cond_embed = None
        cond_kw = dict(n_cond=n_cond, cond_embed_dim=cond_embed_dim, cond_mode=cond_mode,
                       middle_swin=True, unpatch_patch_skip=patch_skip)

        self.df_unet = Swin5DUnet(
            space=5,
            decouple_mu=decouple_mu,
            dim=dim,
            base_resolution=list(df_base_resolution),
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=list(df_patch_size),
            window_size=list(df_window_size),
            depth=depth,
            num_heads=num_heads,
            num_layers=num_layers,
            c_multiplier=c_multiplier,
            hidden_mlp_ratio=8.0,  # torch hardcodes this for gyroswin
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            qk_norm=qk_norm,
            use_rpb=use_rpb,
            gated_attention=gated_attention,
            use_checkpoint=use_checkpoint,
            key=keys[0],
            **cond_kw,
        )
        # phi unet is 3D — we only use the up path; down blocks are placeholders
        self.phi_unet = SwinNDUnet(
            space=3,
            dim=dim,
            base_resolution=list(phi_base_resolution),
            in_channels=1, out_channels=1,
            patch_size=list(phi_patch_size),
            window_size=list(phi_window_size),
            depth=depth,
            num_heads=num_heads,
            num_layers=num_layers,
            c_multiplier=c_multiplier,
            hidden_mlp_ratio=8.0,
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            qk_norm=qk_norm,
            use_rpb=use_rpb,
            gated_attention=gated_attention,
            use_checkpoint=use_checkpoint,
            key=keys[1],
            conv_patch=True,  # phi uses ConvTranspose patch ops (torch conv_patch=True)
            **cond_kw,
        )
        # torch deletes phi_unet.patch_embed + down_blocks (phi is produced from df via
        # velocity-space reduction, never encoded). Null them so they don't count as
        # unmatched params; the gyroswin forward never calls the phi encoder path.
        self.phi_unet = eqx.tree_at(
            lambda u: (u.patch_embed, u.down_blocks),
            self.phi_unet, (None, []),
            is_leaf=lambda x: x is None,
        )

        # SwinNDUnet exposes ``down_dims`` (input dim of each stage; len = num_layers+1).
        df_down_dims = list(self.df_unet.down_dims)
        phi_down_dims = list(self.phi_unet.down_dims)
        # torch builds ONE VSpaceReduce per actual df down block (num_layers), with
        # out_dim = the matching phi up block dim (zip(df_down_blocks, phi_up_blocks[::-1])).
        df_in_dims = df_down_dims[:-1]                  # input dim per down block
        phi_up_blk_dims = phi_down_dims[::-1][1:][::-1]  # phi up-block dims, zip order
        self.vspace_attn_down = [
            VSpaceReduce(
                dim=df_in_dims[i],
                out_dim=phi_up_blk_dims[i] if i < len(phi_up_blk_dims) else df_in_dims[i],
                num_heads=8, decouple_mu=decouple_mu, key=keys[2 + i],
            )
            for i in range(len(df_in_dims))
        ]
        bottleneck_dim = df_down_dims[-1] if df_down_dims else dim
        self.vspace_attn_middle = VSpaceReduce(
            dim=bottleneck_dim, out_dim=bottleneck_dim,
            num_heads=8, decouple_mu=decouple_mu, key=keys[8],
        )
        if patch_skip:
            self.vspace_attn_patch_skip = VSpaceReduce(
                dim=dim, out_dim=dim, num_heads=8,
                decouple_mu=decouple_mu, key=keys[9],
            )
        else:
            self.vspace_attn_patch_skip = None

        # bottleneck mixing — dims match the deepest stage
        self.df_mix_middle = MixingBlock(left_dim=bottleneck_dim, right_dim=bottleneck_dim,
                                         num_heads=8, key=keys[10])
        self.phi_mix_middle = MixingBlock(left_dim=bottleneck_dim, right_dim=bottleneck_dim,
                                          num_heads=8, key=keys[11])
        # up-path mixing — dims match the inputs to each SwinBlockUp (post middle_upscale)
        df_up_dims = df_down_dims[::-1][1:]
        phi_up_dims = phi_down_dims[::-1][1:]
        n_up = len(df_up_dims)
        self.df_mix_up = [
            MixingBlock(left_dim=df_up_dims[i],
                        right_dim=phi_up_dims[i] if i < len(phi_up_dims) else df_up_dims[i],
                        num_heads=8, key=k)
            for i, k in enumerate(jr.split(keys[12], n_up))
        ]
        self.phi_mix_up = [
            MixingBlock(left_dim=phi_up_dims[i] if i < len(phi_up_dims) else df_up_dims[i],
                        right_dim=df_up_dims[i],
                        num_heads=8, key=k)
            for i, k in enumerate(jr.split(keys[13], n_up))
        ]
        # patch-space mixing — operates AFTER the patch-skip concat, so the dim is
        # doubled when patch_skip (torch df_mix_unpatch.attn.q is (128,128) for dim=64).
        unpatch_dim = dim * (2 if patch_skip else 1)
        self.df_mix_unpatch = MixingBlock(left_dim=unpatch_dim, right_dim=unpatch_dim, num_heads=8, key=keys[14])
        self.phi_mix_unpatch = MixingBlock(left_dim=unpatch_dim, right_dim=unpatch_dim, num_heads=8, key=keys[15])

        # flux head: one cross-attn stage per scale (phi=query, df=kv), max-pooled
        # then concatenated -> scalar. dims = reversed down_dims (deepest first).
        if self.use_flux:
            self.flux_head = FluxDecoder(
                left_dims=phi_down_dims[::-1], right_dims=df_down_dims[::-1],
                num_heads=flux_num_heads, depth=flux_depth, key=keys[16],
            )
        else:
            self.flux_head = None

    def __call__(self, df: jnp.ndarray, cond: Optional[jnp.ndarray] = None,
                 *, key=None, inference: bool = True) -> dict:
        """Forward: df → (df, phi).

        df: ``(C, vp, mu, s, x, y)``; cond: ``(n_cond,)`` scalars (raw).
        """
        # per-U-Net conditioning embeddings (torch parity: df/phi have separate embeds)
        c_df = self.df_unet.condition(cond)
        c_phi = self.phi_unet.condition(cond)

        zdf, df_pad_axes = self.df_unet.patch_encode(df)
        # patch-skip residuals: df0 (full patch grid) and its velocity-reduced phi0
        df0 = zdf
        phi0 = self.vspace_attn_patch_skip(df0) if (self.patch_skip and self.vspace_attn_patch_skip is not None) else None
        # down path. The pre-downsample df skip feeds the df up block, and (via
        # vspace_attn_down[i]) the phi up block skip — torch passes phi_features[i]
        # to phi_up_blocks[i] as a KEYWORD s=..., so it is NOT dead.
        df_skips, phi_skips = [], []
        for i, blk in enumerate(self.df_unet.down_blocks):
            zdf, sk = blk(zdf, c_df, inference=inference, return_skip=True)
            df_skips.append(sk)
            if self.use_phi and i < len(self.vspace_attn_down):
                phi_skips.append(self.vspace_attn_down[i](sk))
        # bottleneck — vspace-reduce df → phi, then cross-mix, then middle swin/ViT
        if self.df_unet.middle_pe is not None:
            zdf = self.df_unet.middle_pe(zdf)
        zphi = self.vspace_attn_middle(zdf)
        if self.phi_unet.middle_pe is not None:
            zphi = self.phi_unet.middle_pe(zphi)
        zdf_new = self.df_mix_middle(zdf, zphi)
        zphi_new = self.phi_mix_middle(zphi, zdf)
        zdf, zphi = zdf_new, zphi_new
        zdf = self.df_unet.middle(zdf, c_df, inference=inference)
        zphi = self.phi_unet.middle(zphi, c_phi, inference=inference)
        # flux stage 0: bottleneck latents (phi=query, df=kv)
        flux_lats = []
        if self.use_flux and self.flux_head is not None:
            flux_lats.append(self.flux_head.mix(0, zphi, zdf))
        zdf = self.df_unet.middle_upscale(zdf)
        zphi = self.phi_unet.middle_upscale(zphi)
        # up path with per-scale cross-mix. torch updates df FIRST then mixes phi against
        # the UPDATED df (sequential, unlike the parallel middle mix above).
        for i, (df_blk, phi_blk) in enumerate(zip(self.df_unet.up_blocks, self.phi_unet.up_blocks)):
            zdf = self.df_mix_up[i](zdf, zphi)
            zphi = self.phi_mix_up[i](zphi, zdf)   # uses the just-updated zdf
            zdf = df_blk(zdf, df_skips[-(i + 1)], c_df, inference=inference)
            phi_sk = phi_skips[i] if (self.use_phi and i < len(phi_skips)) else None
            zphi = phi_blk(zphi, phi_sk, c_phi, inference=inference)
            # flux stage i+1: per-scale up-block latents (phi=query, df=kv)
            if self.use_flux and self.flux_head is not None:
                flux_lats.append(self.flux_head.mix(i + 1, zphi, zdf))
        # patch-skip concat (-> dim 2*latent), then final patch-space mixing at that dim
        if self.patch_skip:
            zdf = jnp.concatenate([zdf, df0], axis=-1)
            zphi = jnp.concatenate([zphi, phi0], axis=-1)
        zdf = self.df_mix_unpatch(zdf, zphi)
        zphi = self.phi_mix_unpatch(zphi, zdf)
        # unpatch reduces the concatenated dim back via its proj_concat + FiLM modulation
        df_out = self.df_unet.patch_decode(zdf, df_pad_axes, condition=c_df)
        # phi_unet output: (1, s, x, y) → rearrange to (x, s, y) to match the dataset's phi layout
        phi_out = self.phi_unet.patch_decode(zphi, df_pad_axes[2:], condition=c_phi)
        phi_out = jnp.squeeze(phi_out, axis=0)            # (s, x, y)
        phi_out = jnp.transpose(phi_out, (1, 0, 2))       # (x, s, y)
        out = {"df": df_out, "phi": phi_out}
        if self.use_flux and self.flux_head is not None:
            out["flux"] = self.flux_head(flux_lats)       # scalar
        return out


def build_gyroswin_from_config(cfg_path: str, *, key,
                               resolution: Optional[Sequence[int]] = None) -> GyroSwinMultitask:
    """Build a ``GyroSwinMultitask`` from a Hydra YAML (upstream torch config layout)."""
    import yaml
    from neugk_jax.translate import force_f32
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"] if "model" in cfg else cfg
    swin = mcfg["swin"]
    dataset = cfg.get("dataset", {})
    base_resolution = resolution or dataset.get("resolution") or (32, 8, 16, 85, 32)
    separate_zf = dataset.get("separate_zf", True)
    in_ch = 2 + (2 if separate_zf else 0)
    outputs = [k for k, w in (mcfg.get("loss_weights") or {}).items() if w and w > 0]
    n_cond = len(mcfg.get("conditioning", []) or [])
    model = GyroSwinMultitask(
        dim=mcfg["latent_dim"],
        df_base_resolution=base_resolution,
        df_patch_size=swin["patch_size"],
        df_window_size=swin["window_size"],
        depth=swin["depth"],
        num_heads=swin["num_heads"],
        in_channels=in_ch, out_channels=in_ch,
        num_layers=mcfg.get("num_layers", 4),
        c_multiplier=swin.get("c_multiplier", 2),
        merging_hidden_ratio=swin.get("merging_hidden_ratio", 4.0),
        unmerging_hidden_ratio=swin.get("unmerging_hidden_ratio", 8.0),
        decouple_mu=mcfg.get("decouple_mu", True),
        patch_skip=swin.get("patch_skip", True),
        swin_bottleneck=swin.get("swin_bottleneck", True),
        use_rpb=swin.get("use_rpb", True),
        # torch gyroswin swin blocks have no qk_norm / gated-attention (unlike the AE)
        qk_norm=swin.get("qk_norm", False),
        gated_attention=swin.get("gated_attention", False),
        cond_mode=swin.get("modulation", "film"),
        flux_num_heads=swin.get("flux_num_heads", 4),
        flux_depth=swin.get("flux_depth", 1),
        outputs=outputs or ["df", "phi"],
        n_cond=n_cond,
        use_checkpoint=bool(swin.get("gradient_checkpoint", False)),
        key=key,
    )
    return force_f32(model)
