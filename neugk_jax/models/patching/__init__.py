"""Patch embedding / unpatch layers.

``ops``: N-D fold / unfold and block padding. ``linear``: ``PatchEmbed`` / ``PatchMerge`` / ``PatchExpand``
with weights tied to the patch size. ``field``: ``FieldPatchEmbed`` / ``FieldUnpatch``, whose weights are
functions of the point coordinates.
"""

from neugk_jax.models.patching.field import (
    FIELD_OPTIONS,
    FieldPatchEmbed,
    FieldUnpatch,
    field_options,
)
from neugk_jax.models.patching.linear import (
    PatchEmbed,
    PatchExpand,
    PatchMerge,
    StridedConvTranspose,
    merge_grid,
)
from neugk_jax.models.patching.ops import (
    _normalize_patch,
    fold_patches,
    pad_amounts,
    pad_to_blocks,
    unfold_patches,
    unpad,
)

__all__ = [
    "FIELD_OPTIONS",
    "FieldPatchEmbed",
    "FieldUnpatch",
    "PatchEmbed",
    "PatchExpand",
    "PatchMerge",
    "StridedConvTranspose",
    "_normalize_patch",
    "field_options",
    "fold_patches",
    "merge_grid",
    "pad_amounts",
    "pad_to_blocks",
    "unfold_patches",
    "unpad",
]
