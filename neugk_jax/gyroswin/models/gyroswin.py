"""GyroSwin multitask model — 5D df to 5D df + 3D phi with cross-attention mixing.

Composition:

* ``df_unet``: ``Swin5DUnet`` — full 5D Swin U-Net on the distribution function.
* ``phi_unet``: ``SwinNDUnet`` (space=3) — only the up path is kept;
  the down path's outputs come from ``vspace_attn_down`` reducing the df features.
* ``vspace_attn_down`` / ``vspace_attn_middle`` / ``vspace_attn_patch_skip``:
  ``VSpaceReduce`` blocks that turn the 5D df latents into the 3D phi shape.
* ``df_mix_middle`` / ``phi_mix_middle``: bottleneck cross-attention.
* ``df_mix_up`` / ``phi_mix_up``: up-path cross-attention at each scale.
* ``flux_head``: ``FluxDecoder`` scalar flux (``flux`` or ``fluxavg``) from the
  per-scale (phi, df) latents, optionally FiLM-conditioned.

Conditioning (DiT or FiLM modulation of the Swin blocks) goes through the
per-u-net condition embeds; configs without conditioning still work.
"""

from __future__ import annotations

from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.gyroswin.models.x_layers import FluxDecoder, MixingBlock, VSpaceReduce
from neugk_jax.models.gk_unet import Swin5DUnet, SwinNDUnet


class _Keys:
    """Hands out fresh subkeys of ``key`` on each call; ``None`` when ``key`` is None."""

    def __init__(self, key):
        self.key = key
        self.i = 0

    def __call__(self):
        if self.key is None:
            return None
        self.i += 1
        return jr.fold_in(self.key, self.i)


class GyroSwinMultitask(eqx.Module):
    """5D df → 5D df + 3D phi (+ scalar flux) prediction with cross-attention mixing.

    ``attn_drop`` is the attention-probability dropout of every mixing block and
    velocity-space reduction; ``flux_drop`` is the flux head's projection/MLP
    dropout. ``flux_conditioning`` FiLM-conditions the flux head on the raw
    conditioning scalars.
    """

    df_unet: Swin5DUnet
    phi_unet: SwinNDUnet
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
    flux_key: Optional[str] = eqx.field(static=True)
    use_phi: bool = eqx.field(static=True)
    patch_skip: bool = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    n_cond: int = eqx.field(static=True)
    detach_phi_cross_latents: bool = eqx.field(static=True)

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
        use_rpb: bool = True,
        qk_norm: bool = True,
        gated_attention: bool = True,
        outputs: Sequence[str] = ("df", "phi"),
        n_cond: int = 0,
        cond_embed_dim: int = 128,
        cond_mode: str = "film",
        flux_num_heads: int = 4,
        flux_depth: int = 1,
        flux_reduce: str = "max",
        flux_conditioning: bool = False,
        flux_drop: float = 0.1,
        attn_drop: float = 0.1,
        detach_flux_latents: bool = False,
        detach_phi_cross_latents: bool = False,
        rms_norm: bool = False,
        drop_path: float = 0.1,
        use_checkpoint: bool = False,
        legacy_double_shortcut: bool = False,
        key,
    ):
        self.latent_dim = dim
        self.patch_skip = patch_skip
        self.use_phi = "phi" in outputs
        flux_keys = [k for k in outputs if k in ("flux", "fluxavg")]
        if len(flux_keys) > 1:
            raise ValueError("cannot predict both flux and fluxavg")
        self.flux_key = flux_keys[0] if flux_keys else None
        self.n_cond = n_cond
        self.detach_phi_cross_latents = detach_phi_cross_latents

        phi_base_resolution = tuple(df_base_resolution[2:])  # (s, x, y)
        phi_patch_size = tuple(df_patch_size[2:])
        phi_window_size = tuple(df_window_size[2:])

        keys = jr.split(key, 18)
        unet_kw = dict(dim=dim, depth=depth, num_heads=num_heads, num_layers=num_layers,
                       hidden_mlp_ratio=8.0, merging_hidden_ratio=merging_hidden_ratio,
                       unmerging_hidden_ratio=unmerging_hidden_ratio, qk_norm=qk_norm,
                       use_rpb=use_rpb, gated_attention=gated_attention,
                       use_checkpoint=use_checkpoint, n_cond=n_cond,
                       cond_embed_dim=cond_embed_dim, cond_mode=cond_mode, middle_swin=True,
                       unpatch_patch_skip=patch_skip, rms_norm=rms_norm,
                       legacy_double_shortcut=legacy_double_shortcut, drop_path=drop_path)
        self.df_unet = Swin5DUnet(
            space=5,
            decouple_mu=decouple_mu,
            base_resolution=list(df_base_resolution),
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=list(df_patch_size),
            window_size=list(df_window_size),
            c_multiplier=c_multiplier,
            key=keys[0],
            **unet_kw,
        )
        # the phi unet only keeps its up path; its channel multiplier is fixed at 2
        self.phi_unet = SwinNDUnet(
            space=3,
            base_resolution=list(phi_base_resolution),
            in_channels=1, out_channels=1,
            patch_size=list(phi_patch_size),
            window_size=list(phi_window_size),
            c_multiplier=2,
            conv_patch=True,
            key=keys[1],
            **unet_kw,
        )
        self.phi_unet = eqx.tree_at(
            lambda u: (u.patch_embed, u.down_blocks),
            self.phi_unet, (None, []),
            is_leaf=lambda x: x is None,
        )

        df_down_dims = list(self.df_unet.down_dims)
        phi_down_dims = list(self.phi_unet.down_dims)
        # one VSpaceReduce per df down block; out_dim matches the corresponding phi up block
        df_in_dims = df_down_dims[:-1]
        phi_up_blk_dims = phi_down_dims[::-1][1:][::-1]
        vs_kw = dict(num_heads=8, decouple_mu=decouple_mu, attn_drop=attn_drop)
        self.vspace_attn_down = [
            VSpaceReduce(
                dim=df_in_dims[i],
                out_dim=phi_up_blk_dims[i] if i < len(phi_up_blk_dims) else df_in_dims[i],
                key=keys[2 + i], **vs_kw,
            )
            for i in range(len(df_in_dims))
        ]
        bottleneck_dim = df_down_dims[-1] if df_down_dims else dim
        self.vspace_attn_middle = VSpaceReduce(
            dim=bottleneck_dim, out_dim=bottleneck_dim, key=keys[8], **vs_kw,
        )
        if patch_skip:
            self.vspace_attn_patch_skip = VSpaceReduce(dim=dim, out_dim=dim, key=keys[9], **vs_kw)
        else:
            self.vspace_attn_patch_skip = None

        mix_kw = dict(num_heads=8, attn_drop=attn_drop)
        self.df_mix_middle = MixingBlock(bottleneck_dim, bottleneck_dim, key=keys[10], **mix_kw)
        self.phi_mix_middle = MixingBlock(bottleneck_dim, bottleneck_dim, key=keys[11], **mix_kw)
        # up-path mixing: dims match the inputs to each SwinBlockUp (post middle_upscale)
        df_up_dims = df_down_dims[::-1][1:]
        phi_up_dims = phi_down_dims[::-1][1:]
        n_up = len(df_up_dims)
        self.df_mix_up = [
            MixingBlock(df_up_dims[i],
                        phi_up_dims[i] if i < len(phi_up_dims) else df_up_dims[i],
                        key=k, **mix_kw)
            for i, k in enumerate(jr.split(keys[12], n_up))
        ]
        self.phi_mix_up = [
            MixingBlock(phi_up_dims[i] if i < len(phi_up_dims) else df_up_dims[i],
                        df_up_dims[i], key=k, **mix_kw)
            for i, k in enumerate(jr.split(keys[13], n_up))
        ]
        # patch-space mixing runs after the patch-skip concat, so the dim doubles with patch_skip
        unpatch_dim = dim * (2 if patch_skip else 1)
        self.df_mix_unpatch = MixingBlock(unpatch_dim, unpatch_dim, key=keys[14], **mix_kw)
        self.phi_mix_unpatch = MixingBlock(unpatch_dim, unpatch_dim, key=keys[15], **mix_kw)

        # flux head stages run deepest first (phi=query, df=kv)
        if self.flux_key is not None:
            self.flux_head = FluxDecoder(
                left_dims=phi_down_dims[::-1], right_dims=df_down_dims[::-1],
                num_heads=flux_num_heads, depth=flux_depth, key=keys[16],
                reduction=flux_reduce, attn_drop=attn_drop, drop=flux_drop,
                detach_latents=detach_flux_latents,
                n_cond=n_cond if flux_conditioning else 0, cond_embed_dim=cond_embed_dim,
            )
        else:
            self.flux_head = None

    def _phi_for_df(self, zphi):
        return jax.lax.stop_gradient(zphi) if self.detach_phi_cross_latents else zphi

    def __call__(self, df: jnp.ndarray, cond: Optional[jnp.ndarray] = None,
                 *, key=None, inference: bool = True) -> dict:
        """Forward: df → ``{"df", "phi"?, flux_key?}``.

        df: ``(C, vp, mu, s, x, y)``; cond: ``(n_cond,)`` raw scalars. ``key``
        drives drop-path and dropout when ``inference=False``.
        """
        nk = _Keys(key)
        kw = dict(inference=inference)
        c_df = self.df_unet.condition(cond)
        c_phi = self.phi_unet.condition(cond)

        zdf, df_pad_axes = self.df_unet.patch_encode(df)
        # patch-skip residuals: df0 (full patch grid) and its velocity-reduced phi0
        df0 = zdf
        phi0 = (self.vspace_attn_patch_skip(df0, key=nk(), **kw)
                if self.vspace_attn_patch_skip is not None else None)
        # down path: df skips feed the df up blocks, their velocity reductions the phi up blocks
        df_skips, phi_skips = [], []
        for i, blk in enumerate(self.df_unet.down_blocks):
            zdf, sk = blk(zdf, c_df, key=nk(), return_skip=True, **kw)
            df_skips.append(sk)
            if self.use_phi and i < len(self.vspace_attn_down):
                phi_skips.append(self.vspace_attn_down[i](sk, key=nk(), **kw))
        # bottleneck: vspace-reduce df → phi, parallel cross-mix, then the middle swin layers
        if self.df_unet.middle_pe is not None:
            zdf = self.df_unet.middle_pe(zdf)
        zphi = self.vspace_attn_middle(zdf, key=nk(), **kw)
        if self.phi_unet.middle_pe is not None:
            zphi = self.phi_unet.middle_pe(zphi)
        zdf, zphi = (self.df_mix_middle(zdf, self._phi_for_df(zphi), key=nk(), **kw),
                     self.phi_mix_middle(zphi, zdf, key=nk(), **kw))
        zdf = self.df_unet.middle(zdf, c_df, key=nk(), **kw)
        zphi = self.phi_unet.middle(zphi, c_phi, key=nk(), **kw)
        flux_lats = []
        if self.flux_head is not None:
            flux_lats.append(self.flux_head.mix(0, zphi, zdf, cond, key=nk(), **kw))
        zdf = self.df_unet.middle_upscale(zdf)
        zphi = self.phi_unet.middle_upscale(zphi)
        # up path: df mixes first, then phi mixes against the updated df
        for i, (df_blk, phi_blk) in enumerate(zip(self.df_unet.up_blocks, self.phi_unet.up_blocks)):
            zdf = self.df_mix_up[i](zdf, self._phi_for_df(zphi), key=nk(), **kw)
            zphi = self.phi_mix_up[i](zphi, zdf, key=nk(), **kw)
            zdf = df_blk(zdf, df_skips[-(i + 1)], c_df, key=nk(), **kw)
            phi_sk = phi_skips[i] if (self.use_phi and i < len(phi_skips)) else None
            zphi = phi_blk(zphi, phi_sk, c_phi, key=nk(), **kw)
            if self.flux_head is not None:
                flux_lats.append(self.flux_head.mix(i + 1, zphi, zdf, cond, key=nk(), **kw))
        if self.patch_skip:
            zdf = jnp.concatenate([zdf, df0], axis=-1)
            zphi = jnp.concatenate([zphi, phi0], axis=-1)
        zdf = self.df_mix_unpatch(zdf, self._phi_for_df(zphi), key=nk(), **kw)
        zphi = self.phi_mix_unpatch(zphi, zdf, key=nk(), **kw)
        df_out = self.df_unet.patch_decode(zdf, df_pad_axes, condition=c_df)
        # phi_unet output (1, s, x, y) → (x, s, y), the dataset's phi layout
        phi_out = self.phi_unet.patch_decode(zphi, df_pad_axes[2:], condition=c_phi)
        phi_out = jnp.transpose(jnp.squeeze(phi_out, axis=0), (1, 0, 2))
        out = {"df": df_out}
        if self.use_phi:
            out["phi"] = phi_out
        if self.flux_head is not None:
            out[self.flux_key] = self.flux_head(flux_lats, key=nk(), **kw)
        return out


_ACCEPTED_SWIN_KEYS = {
    "patch_size", "window_size", "num_heads", "depth", "gradient_checkpoint",
    "merging_hidden_ratio", "unmerging_hidden_ratio", "c_multiplier", "patch_skip",
    "modulation", "use_rpb", "qk_norm", "gated_attention", "norm_fn", "flux_reduce",
    "flux_num_heads", "flux_depth", "flux_conditioning", "detach_flux_latents",
    "detach_phi_cross_latents", "attn_drop", "flux_drop",
    # phi grids are derived from the df grids
    "phi_patch_size", "phi_window_size",
    # no effect on the model: unused by the multitask model, or init-only
    "norm_output", "drop_path", "init_weights", "patching_init_weights", "cond_init_weights",
}

# key -> the only supported value
_FIXED_SWIN_KEYS = {
    "swin_bottleneck": True, "latent_cross_attn": True, "use_abs_pe": False,
    "use_rope": False, "cosine_attn": False, "act_fn": "GELU",
}


def _check_config(mcfg: dict, dataset: dict, training: dict) -> None:
    swin = mcfg["swin"]
    unknown = set(swin) - _ACCEPTED_SWIN_KEYS - set(_FIXED_SWIN_KEYS)
    if unknown:
        raise ValueError(f"unknown model.swin keys: {sorted(unknown)}")
    for k, v in _FIXED_SWIN_KEYS.items():
        if k in swin and swin[k] != v:
            raise NotImplementedError(f"model.swin.{k}={swin[k]!r} (only {v!r} is supported)")
    if swin.get("modulation", "film") not in ("film", "dit"):
        raise NotImplementedError(f"model.swin.modulation={swin['modulation']!r}")
    if swin.get("norm_fn", "LayerNorm") not in ("LayerNorm", "RMSNorm"):
        raise NotImplementedError(f"model.swin.norm_fn={swin['norm_fn']!r}")
    if int(mcfg.get("bundle_seq_length", 1)) > 1:
        raise NotImplementedError("model.bundle_seq_length > 1")
    if not dataset.get("real_potens", True):
        raise NotImplementedError("dataset.real_potens=false (complex phi)")
    if training.get("predict_delta", False):
        raise NotImplementedError("training.predict_delta")
    if any(int(u) > 0 for u in (training.get("pushforward") or {}).get("unrolls") or []):
        raise NotImplementedError("training.pushforward unrolls")


def build_gyroswin_from_config(cfg_path, *, key,
                               resolution: Optional[Sequence[int]] = None,
                               legacy_double_shortcut: Optional[bool] = None) -> GyroSwinMultitask:
    """Build a ``GyroSwinMultitask`` from a YAML path or a ``{"model", "dataset"}`` mapping.

    ``legacy_double_shortcut`` defaults to ``model.legacy_swin_shortcut``, or
    to True when absent. Config keys the port does not implement raise.
    """
    from neugk_jax.translate import force_f32, load_config
    cfg = load_config(cfg_path)
    mcfg = cfg["model"] if "model" in cfg else cfg
    swin = mcfg["swin"]
    dataset = cfg.get("dataset", {}) or {}
    _check_config(mcfg, dataset, cfg.get("training", {}) or {})
    if legacy_double_shortcut is None:
        legacy_double_shortcut = bool(mcfg.get("legacy_swin_shortcut", True))
    base_resolution = resolution or dataset.get("resolution") or (32, 8, 16, 85, 32)
    separate_zf = dataset.get("separate_zf", True)
    in_ch = 2 + (2 if separate_zf else 0)
    sched = mcfg.get("loss_scheduler") or {}
    outputs = [k for k, w in (mcfg.get("loss_weights") or {}).items()
               if (w and w > 0) or sched.get(k)]
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
        use_rpb=swin.get("use_rpb", True),
        # gyroswin swin blocks default qk_norm/gated-attention off, unlike the ae
        qk_norm=swin.get("qk_norm", False),
        gated_attention=swin.get("gated_attention", False),
        cond_mode=swin.get("modulation", "film"),
        rms_norm=(swin.get("norm_fn") == "RMSNorm"),
        drop_path=float(mcfg.get("drop_path", 0.1)),
        flux_num_heads=swin.get("flux_num_heads", 4),
        flux_depth=swin.get("flux_depth", 1),
        flux_reduce=swin.get("flux_reduce", "max"),
        flux_conditioning=bool(swin.get("flux_conditioning", False)),
        flux_drop=float(swin.get("flux_drop", 0.1)),
        attn_drop=float(swin.get("attn_drop", 0.1)),
        detach_flux_latents=bool(swin.get("detach_flux_latents", False)),
        detach_phi_cross_latents=bool(swin.get("detach_phi_cross_latents", False)),
        outputs=outputs or ["df", "phi"],
        n_cond=len(mcfg.get("conditioning", []) or []),
        use_checkpoint=bool(swin.get("gradient_checkpoint", False)),
        legacy_double_shortcut=legacy_double_shortcut,
        key=key,
    )
    return force_f32(model)
