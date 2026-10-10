"""Point coordinates of a patch for the field patching layers.

A grid spec describes the coordinates of a channel-last grid ``(*spatial, C)``: every spatial axis is
``relative`` (cell-centred offset from the centre of its patch, in [-1, 1], with a physical ``spacing``)
or ``absolute`` (node values mapped to [-1, 1] by ``range`` or the node extent, with quadrature
``weights``); absolute axes may be folded into the channels as ``(c, *folded)``. Without a spec every
axis is relative with unit spacing. A point at cell ``j`` of a patch of ``q`` cells sits at
``(2 j + 1) / q - 1``, so the same physical position keeps its coordinate when the resolution changes.
A relative axis may give its ``band``, the modes the data resolves per patch: the bases of an axis stop
at its band and at the modes its points represent (``AxisPoints.cap``).
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from neugk_jax.models.ops import _normalize_patch


def _cells(j, q):
    return (2 * np.asarray(j, np.float64) + 1) / q - 1


def _mapped(axis: Mapping, n_padded: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes in [-1, 1] and quadrature weights of an absolute axis padded to ``n_padded``."""
    nodes = np.asarray(axis["nodes"], np.float64)
    w = np.asarray(axis.get("weights", np.ones_like(nodes)), np.float64)
    extra = n_padded - len(nodes)
    step = nodes[-1] - nodes[-2] if len(nodes) > 1 else 1.0
    # padded nodes continue the grid with zero weight
    full = np.concatenate([nodes, nodes[-1] + step * np.arange(1, extra + 1)])
    lo, hi = axis.get("range", (nodes.min(), nodes.max()))
    unit = 2.0 * (full - lo) / max(hi - lo, 1e-12) - 1.0
    return unit, np.concatenate([w, np.zeros(extra)])


def _mean_one(w: np.ndarray) -> np.ndarray:
    return w / w[w > 0].mean()


class AxisPoints(eqx.Module):
    """One axis of a patch: coordinates and quadrature weights, ``(p,)`` or ``(T, p)`` per token, and
    ``cap``, the modes a basis can use on this grid."""

    buffer_fields = ("coords", "weight")

    coords: jnp.ndarray
    weight: jnp.ndarray
    kind: str = eqx.field(static=True)
    cap: int = eqx.field(static=True)

    def __init__(self, coords, weight, kind: str, cap: int):
        self.coords = jnp.asarray(coords, jnp.float32)
        self.weight = jnp.asarray(weight, jnp.float32)
        self.kind = kind
        self.cap = int(cap)


class PointGrid(eqx.Module):
    """Coordinates of the points of one channel of a patch, in ``fold_patches`` order.

    The points are the spatial cells of the patch times the folded nodes; every channel has the same
    points. Per point: ``pos`` (cell-centred position in the patch along every spatial axis) and the
    quadrature ``weight``; ``axes`` holds the coordinates and weights per spatial and folded axis.
    """

    buffer_fields = ("pos", "weight")

    pos: jnp.ndarray
    weight: jnp.ndarray
    axes: tuple[AxisPoints, ...]
    patch: tuple[int, ...] = eqx.field(static=True)
    rel_axes: tuple[int, ...] = eqx.field(static=True)
    abs_axes: tuple[int, ...] = eqx.field(static=True)
    n_fold: tuple[int, ...] = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    half: tuple[float, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    reference: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        padded: Sequence[int],
        patch: Sequence[int],
        channels: int,
        spec: Optional[Mapping] = None,
    ):
        patch = _normalize_patch(patch)
        n = len(patch)
        spec = dict(spec or {})
        axes = list(spec.get("axes") or [{"kind": "relative"}] * n)
        folded = list(spec.get("folded") or [])
        if len(axes) != n:
            raise ValueError(f"grid spec has {len(axes)} axes for {n} spatial axes")
        n_fold = tuple(len(f["nodes"]) for f in folded)
        if channels % max(1, math.prod(n_fold)):
            raise ValueError(f"{channels} channels do not hold the folded axes {n_fold}")
        idx = np.meshgrid(*(np.arange(k) for k in (*patch, *n_fold)), indexing="ij")
        idx = [i.reshape(-1) for i in idx]
        offs, ifold = idx[:n], idx[n:]
        n_points = math.prod(patch) * math.prod(n_fold)

        rel = [i for i, a in enumerate(axes) if a.get("kind", "relative") == "relative"]
        self.rel_axes = tuple(rel)
        self.abs_axes = tuple(i for i in range(n) if i not in rel)
        tokens = [padded[i] // patch[i] for i in self.abs_axes]
        weight, per_axis = np.ones((*tokens, n_points)), []
        for i in range(n):
            if i in rel:
                cells = _cells(np.arange(patch[i]), patch[i])
                cap = min(patch[i], int(axes[i].get("band", patch[i])))
                per_axis.append(AxisPoints(cells, np.ones(patch[i]), "relative", cap))
                continue
            j = self.abs_axes.index(i)
            unit, w = _mapped(axes[i], padded[i])
            rows = np.arange(tokens[j])[:, None] * patch[i] + np.arange(patch[i])[None]
            per_axis.append(
                AxisPoints(unit[rows], _mean_one(w[rows]), "absolute", len(axes[i]["nodes"]))
            )
            pos = np.arange(tokens[j])[:, None] * patch[i] + offs[i][None]
            shape = [1] * len(tokens) + [n_points]
            shape[j] = tokens[j]
            weight = weight * w[pos].reshape(shape)
        for f, i in zip(folded, ifold):
            unit, w = _mapped(f, len(f["nodes"]))
            per_axis.append(AxisPoints(unit, _mean_one(w), "folded", len(unit)))
            weight = weight * w[i]
        self.axes = tuple(per_axis)
        self.weight = jnp.asarray(_mean_one(weight), jnp.float32)
        self.pos = jnp.asarray(
            np.stack([_cells(o, p) for o, p in zip(offs, patch)], -1), jnp.float32
        )
        self.patch = patch
        self.n_fold = n_fold
        self.n_channels = channels // math.prod(n_fold)
        self.half = tuple(patch[i] / 2 for i in rel)
        self.spacing = tuple(float(axes[i].get("spacing", 1.0)) for i in rel)
        # physical half-width of a full patch at the nominal spacing, unless given
        self.reference = tuple(
            float(axes[i].get("reference", patch[i] / 2 * s)) for i, s in zip(rel, self.spacing)
        )

    def spacings(self, geometry=None) -> jnp.ndarray:
        return jnp.asarray(self.spacing if geometry is None else geometry, jnp.float32)

    def scale(self, geometry=None) -> jnp.ndarray:
        """Log physical half-width of a patch per relative axis, over the reference."""
        half = jnp.asarray(self.half, jnp.float32)
        return jnp.log(half * self.spacings(geometry) / jnp.asarray(self.reference, jnp.float32))
