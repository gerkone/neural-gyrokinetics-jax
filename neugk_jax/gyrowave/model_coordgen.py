"""WaveletCoordDiT: coord+value generation over wavelet-token sets (Approach B).

Unlike WaveletDiT (values diffused, coords a STATIC conditioning input), here the whole
token z = [coord_norm (5), value_white (2)] is diffused, so the model generates BOTH the
support (coords) and the coefficients at a fixed length N. The velocity head therefore
covers all 7 channels.

Reuses the shared pieces verbatim — WaveletDiTBlock (AdaLN-Zero + attention),
ContinuousConditionEmbed (t + physical params), LayerNorm, MLP, Linear, and the model.py
``_sincos`` coord featurizer. The coord embed uses the CURRENT (diffused) coord, so there is
NO static RoPE table; attention is always 'phys' (slice attention, no token-token dot).
"""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.embeddings import ContinuousConditionEmbed
from neugk_jax.models.utils import MLP, LayerNorm, Linear
from .model import WaveletDiTBlock, _sincos


class WaveletCoordDiT(eqx.Module):
    """Velocity-field model over the FULL (coord, value) token set.
    __call__(z, t, cond) -> velocity, z (N, n_coord+val_dim), all per-sample (vmap the batch)."""
    time_embed: ContinuousConditionEmbed
    cond_embed: ContinuousConditionEmbed
    coord_embed: MLP
    value_embed: MLP
    blocks: tuple
    ln_out: LayerNorm
    head: Linear
    bands: int = eqx.field(static=True)
    n_coord: int = eqx.field(static=True)
    val_dim: int = eqx.field(static=True)

    def __init__(self, *, val_dim=2, n_coord=5, n_cond=4, n_hidden=256, n_layers=8,
                 n_head=8, mlp_ratio=2, bands=6, embed_dim=32, slice_num=512, key):
        ks = jr.split(key, 5 + n_layers)
        self.time_embed = ContinuousConditionEmbed(embed_dim, 1, key=ks[0])
        self.cond_embed = ContinuousConditionEmbed(embed_dim, n_cond, key=ks[1])
        cond_dim = self.time_embed.cond_dim + self.cond_embed.cond_dim
        # separate coord / value embeds, summed (AB-UPT token embed, coord side dynamic)
        self.coord_embed = MLP([n_coord * 2 * bands, n_hidden * 2, n_hidden], key=ks[2])
        self.value_embed = MLP([val_dim, n_hidden * 2, n_hidden], key=ks[3])
        self.blocks = tuple(
            WaveletDiTBlock(n_hidden, n_head, cond_dim, attn_kind="phys",
                            slice_num=slice_num, mlp_ratio=mlp_ratio, key=ks[5 + i])
            for i in range(n_layers))
        self.ln_out = LayerNorm(n_hidden, elementwise_affine=False)
        self.head = Linear(n_hidden, n_coord + val_dim, key=ks[4])
        self.bands = bands
        self.n_coord = n_coord
        self.val_dim = val_dim

    def __call__(self, z, t, cond):
        # z (N, n_coord+val_dim) diffused; t scalar; cond (n_cond,) physical params
        coord, value = z[:, :self.n_coord], z[:, self.n_coord:]
        c = jnp.concatenate([self.time_embed(t.reshape(1)), self.cond_embed(cond)], axis=-1)
        h = self.coord_embed(_sincos(coord, self.bands)) + self.value_embed(value)
        for blk in self.blocks:
            h = blk(h, c, None)                       # phys attention -> rope unused
        return self.head(self.ln_out(h))
