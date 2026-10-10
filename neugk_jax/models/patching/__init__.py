"""Patch embedding / unpatch layers (grid encoders / decoders).

``linear``: ``PatchEmbed`` / ``LinearUnpatch`` with weights tied to the patch size. ``points``: patch
point coordinates (``PointGrid``). ``cconv``: strided continuous convolutions,
``BandLimitedPatchEmbed`` / ``BandLimitedUnpatch`` and ``TuckerPatchEmbed`` / ``TuckerUnpatch``.
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
from neugk_jax.models.patching.cconv import (
    BandLimitedPatchEmbed,
    BandLimitedUnpatch,
    CConvPatchEmbed,
    CConvUnpatch,
    TuckerPatchEmbed,
    TuckerUnpatch,
)
from neugk_jax.models.patching.linear import LinearUnpatch, PatchEmbed
from neugk_jax.models.patching.points import AxisPoints, PointGrid

PATCHINGS = {
    "linear": (PatchEmbed, LinearUnpatch),
    "cconv": (BandLimitedPatchEmbed, BandLimitedUnpatch),
    "tucker": (TuckerPatchEmbed, TuckerUnpatch),
}
CCONV_PATCHINGS = ("cconv", "tucker")

__all__ = [
    "CCONV_PATCHINGS",
    "PATCHINGS",
    "AxisPoints",
    "BandLimitedPatchEmbed",
    "BandLimitedUnpatch",
    "CConvPatchEmbed",
    "CConvUnpatch",
    "LinearUnpatch",
    "PatchEmbed",
    "PointGrid",
    "TuckerPatchEmbed",
    "TuckerUnpatch",
    "_normalize_patch",
    "fold_patches",
    "pad_amounts",
    "pad_to_blocks",
    "unfold_patches",
    "unpad",
]
