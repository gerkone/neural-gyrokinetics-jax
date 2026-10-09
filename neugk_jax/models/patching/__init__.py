"""Patch embedding / unpatch layers (grid encoders / decoders).

``linear``: ``PatchEmbed`` / ``LinearUnpatch`` with weights tied to the patch size. ``points``: patch
point coordinates (``PointGrid``). ``field``: ``SmoothPatchEmbed`` / ``SmoothUnpatch`` and
``DCTPatchEmbed`` / ``DCTUnpatch``, whose weights are functions of the point coordinates.
:data:`PATCHINGS` maps a patching kind to its embedding / unpatch pair. The fold / unfold / padding ops
live in :mod:`neugk_jax.models.ops` and are re-exported here.
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
    CosineFilter,
    DCTBases,
    DCTPatchEmbed,
    DCTUnpatch,
    FieldPatchEmbed,
    FieldUnpatch,
    PointFilter,
    SmoothPatchEmbed,
    SmoothUnpatch,
)
from neugk_jax.models.patching.linear import LinearUnpatch, PatchEmbed
from neugk_jax.models.patching.points import AxisPoints, PointGrid

PATCHINGS = {
    "linear": (PatchEmbed, LinearUnpatch),
    "smooth": (SmoothPatchEmbed, SmoothUnpatch),
    "dct": (DCTPatchEmbed, DCTUnpatch),
}
FIELD_PATCHINGS = ("smooth", "dct")

__all__ = [
    "FIELD_PATCHINGS",
    "PATCHINGS",
    "AxisPoints",
    "CosineFilter",
    "DCTBases",
    "DCTPatchEmbed",
    "DCTUnpatch",
    "FieldPatchEmbed",
    "FieldUnpatch",
    "LinearUnpatch",
    "PatchEmbed",
    "PointFilter",
    "PointGrid",
    "SmoothPatchEmbed",
    "SmoothUnpatch",
    "_normalize_patch",
    "fold_patches",
    "pad_amounts",
    "pad_to_blocks",
    "unfold_patches",
    "unpad",
]
