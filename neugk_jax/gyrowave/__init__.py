"""gyrowave — wavelet/HL-moment diffusion for gyrokinetic fields (companion to gyrosplats).

compress/  : linear transform + physics solves + per-trajectory processor (HL velocity
             moments x (s,x) wavelet, flux-GN + phi-aware, token cache).
analysis/  : token-cache building, FM training/sampling scripts.
model.py   : WaveletDiT — coordinate-tagged wavelet-token velocity model; RoPE on physical
             coords + AdaLN; swappable attention ('full' O(N^2) RoPE, or 'phys' Transolver
             slice attention O(N*slices)).
"""
from neugk_jax.models.physics_attention import PhysicsAttentionIrregularMesh
from neugk_jax.models.rope import apply_rope, rope_tables

from .model import WaveletDiT, WaveletDiTBlock
from .model_coordgen import WaveletCoordDiT

__all__ = ["WaveletDiT", "WaveletDiTBlock", "WaveletCoordDiT",
           "PhysicsAttentionIrregularMesh", "rope_tables", "apply_rope"]
