"""Patch embedding / unpatch layers (grid encoders / decoders).

``linear``: ``PatchEmbed`` / ``LinearUnpatch`` with weights tied to the patch size. ``points``: patch
point coordinates (``PointGrid``). ``field``: ``FieldPatchEmbed`` / ``FieldUnpatch``, whose weights
are functions of the point coordinates. The fold / unfold / padding ops live in
:mod:`neugk_jax.models.ops` and are re-exported here.
"""

from neugk_jax.models.ops import (
    _normalize_patch,
    fold_patches,
    pad_amounts,
    pad_to_blocks,
    unfold_patches,
    unpad,
)
from neugk_jax.models.patching.field import (
    DECODERS,
    ENCODERS,
    FIELD_OPTIONS,
    AxisBases,
    FieldPatchEmbed,
    FieldUnpatch,
    field_options,
)
from neugk_jax.models.patching.linear import LinearUnpatch, PatchEmbed
from neugk_jax.models.patching.points import AxisPoints, PointGrid

__all__ = [
    "DECODERS",
    "ENCODERS",
    "FIELD_OPTIONS",
    "AxisBases",
    "AxisPoints",
    "FieldPatchEmbed",
    "FieldUnpatch",
    "LinearUnpatch",
    "PatchEmbed",
    "PointGrid",
    "_normalize_patch",
    "field_options",
    "fold_patches",
    "pad_amounts",
    "pad_to_blocks",
    "unfold_patches",
    "unpad",
]
