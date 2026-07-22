"""Rotary position embedding for the model package.

Two pieces live here:
  * the apply-math (``_rotate_half`` + ``apply_rope``): the rotation itself, consumed by any
    attention module (e.g. ``MultiHeadSelfAttention``) given a precomputed ``(cos, sin)`` table;
  * ``rope_tables``: builds that table from CONTINUOUS PHYSICAL coordinates (AB-UPT / noether
    blueprint) — the angle is a pure function of the coordinate value (no grid, no integer index,
    no learned table), so a token gets the same rotation regardless of local sampling density:
    the resolution-invariance mechanism. RoPE applies to q/k inside FULL attention only —
    slice/physics attention has no token-token dot product, so it does not apply there.
"""
from __future__ import annotations

import jax.numpy as jnp


def _rotate_half(x):
    h = x.shape[-1] // 2
    return jnp.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rope(x, cos, sin):
    """Rotate ``x`` by the precomputed ``(cos, sin)`` table (broadcast over ``x``)."""
    return x * cos + _rotate_half(x) * sin


def rope_tables(coords, head_dim, max_wavelength=10000.0):
    """coords: (N, A) physical coords -> (cos, sin) each (N, head_dim); head_dim split
    across axes (trailing axis absorbs the remainder). Coords must be ABSOLUTE physical
    units (not per-sample box-normalized) for cross-box-size consistency."""
    N, A = coords.shape
    assert head_dim % 2 == 0, "head_dim must be even"
    half = head_dim // 2
    per = half // A
    rem = half - per * (A - 1)
    angles = []
    for a in range(A):
        d = rem if a == A - 1 else per
        if d == 0:
            continue
        # geometric ladder in [1, 1/max_wavelength]
        omega = max_wavelength ** (-jnp.arange(0, d, dtype=jnp.float32) / max(d, 1))
        angles.append(coords[:, a:a + 1] * omega[None, :])          # (N, d)
    ang = jnp.concatenate(angles, axis=-1)                          # (N, half)
    cos = jnp.concatenate([jnp.cos(ang), jnp.cos(ang)], axis=-1)    # (N, head_dim)
    sin = jnp.concatenate([jnp.sin(ang), jnp.sin(ang)], axis=-1)
    return cos.astype(jnp.float32), sin.astype(jnp.float32)
