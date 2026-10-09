"""Point coordinates of a patch for the field patching layers.

A grid spec describes the coordinates of a channel-last grid ``(*spatial, C)``: every spatial axis is
``relative`` (cell-centred offset from the centre of its patch, in [-1, 1], with a physical ``spacing``)
or ``absolute`` (node values mapped to [-1, 1] by ``range`` or the node extent, with quadrature
``weights``); absolute axes may be folded into the channels as ``(c, *folded)``. Without a spec every
axis is relative with unit spacing. A point at cell ``j`` of a patch of ``q`` cells sits at
``(2 j + 1) / q - 1``, so the same physical position keeps its coordinate when the resolution changes.
Any axis may give its ``band``, the number of modes the data resolves per patch: the coordinate
encoding fed to the filter MLPs is cut there (no cut without a ``band``), independently of the grid, so
the filters are the same functions on every grid. Fixed cosine bases are further cut at the points of
the grid (``AxisPoints.cap``), the modes it can represent.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from neugk_jax.models.ops import _normalize_patch

NO_BAND = 1 << 30


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
    """One axis of a patch: coordinates and quadrature weights, ``(p,)`` or ``(T, p)`` per token.

    ``cap``: the modes a fixed cosine basis can use on this grid; ``band``: the cut of the coordinate
    encoding (grid independent).
    """

    buffer_fields = ("coords", "weight")

    coords: jnp.ndarray
    weight: jnp.ndarray
    kind: str = eqx.field(static=True)
    cap: int = eqx.field(static=True)
    band: int = eqx.field(static=True)

    def __init__(self, coords, weight, kind: str, cap: int, band: int):
        self.coords = jnp.asarray(coords, jnp.float32)
        self.weight = jnp.asarray(weight, jnp.float32)
        self.kind = kind
        self.cap = int(cap)
        self.band = int(band)


class PointGrid(eqx.Module):
    """Coordinates of the points of one channel of a patch, in ``fold_patches`` order.

    The points are the spatial cells of the patch times the folded nodes; every channel has the same
    points. Per point: ``offsets`` (cell-centred offsets of the relative axes from the patch centre),
    ``coords`` (absolute spatial and folded coordinates per token, ``(*T_abs, P, n_abs)``), ``pos``
    (cell-centred position in the patch along every spatial axis) and the quadrature ``weight``.
    ``axes`` holds the same per spatial and folded axis, for separable bases.
    """

    buffer_fields = ("coords", "offsets", "pos", "weight")

    coords: jnp.ndarray
    offsets: jnp.ndarray
    pos: jnp.ndarray
    weight: jnp.ndarray
    axes: tuple[AxisPoints, ...]
    patch: tuple[int, ...] = eqx.field(static=True)
    tokens: tuple[int, ...] = eqx.field(static=True)
    rel_axes: tuple[int, ...] = eqx.field(static=True)
    abs_axes: tuple[int, ...] = eqx.field(static=True)
    n_fold: tuple[int, ...] = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    half: tuple[float, ...] = eqx.field(static=True)
    bands: tuple[int, ...] = eqx.field(static=True)
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
        band = [int(axes[i].get("band", NO_BAND)) for i in range(n)]
        self.rel_axes = tuple(rel)
        self.abs_axes = tuple(i for i in range(n) if i not in rel)
        tokens = [padded[i] // patch[i] for i in self.abs_axes]
        lead = (*tokens, n_points)
        coords, weight, per_axis = [], np.ones(lead), []
        for i in range(n):
            if i in rel:
                cells = _cells(np.arange(patch[i]), patch[i])
                cap = min(patch[i], band[i])
                per_axis.append(AxisPoints(cells, np.ones(patch[i]), "relative", cap, band[i]))
                continue
            j = self.abs_axes.index(i)
            unit, w = _mapped(axes[i], padded[i])
            rows = np.arange(tokens[j])[:, None] * patch[i] + np.arange(patch[i])[None]
            n_nodes = len(axes[i]["nodes"])
            per_axis.append(
                AxisPoints(unit[rows], _mean_one(w[rows]), "absolute", n_nodes, band[i])
            )
            pos = np.arange(tokens[j])[:, None] * patch[i] + offs[i][None]
            shape = [1] * len(tokens) + [n_points]
            shape[j] = tokens[j]
            coords.append(np.broadcast_to(unit[pos].reshape(shape), lead))
            weight = weight * w[pos].reshape(shape)
        for f, i in zip(folded, ifold):
            unit, w = _mapped(f, len(f["nodes"]))
            f_band = int(f.get("band", NO_BAND))
            per_axis.append(AxisPoints(unit, _mean_one(w), "folded", len(unit), f_band))
            coords.append(np.broadcast_to(unit[i], lead))
            weight = weight * w[i]
        self.axes = tuple(per_axis)
        self.coords = jnp.asarray(
            np.stack(coords, -1) if coords else np.zeros((*lead, 0)), jnp.float32
        )
        self.weight = jnp.asarray(_mean_one(weight), jnp.float32)
        pos = np.stack([_cells(o, p) for o, p in zip(offs, patch)], -1)
        self.pos = jnp.asarray(pos, jnp.float32)
        self.offsets = jnp.asarray(pos[:, list(rel)], jnp.float32)
        self.patch = patch
        self.tokens = tuple(padded[i] // patch[i] for i in range(n))
        self.n_fold = n_fold
        self.n_channels = channels // math.prod(n_fold)
        self.half = tuple(patch[i] / 2 for i in rel)
        # encoding cut per coordinate: offsets, absolute, then folded
        self.bands = (
            tuple(band[i] for i in rel)
            + tuple(band[i] for i in self.abs_axes)
            + tuple(int(f.get("band", NO_BAND)) for f in folded)
        )
        self.spacing = tuple(float(axes[i].get("spacing", 1.0)) for i in rel)
        # physical half-width of a full patch at the nominal spacing, unless given
        self.reference = tuple(
            float(axes[i].get("reference", patch[i] / 2 * s)) for i, s in zip(rel, self.spacing)
        )

    @property
    def n_coords(self) -> int:
        return self.offsets.shape[-1] + self.coords.shape[-1]

    def centres(self, k: int) -> jnp.ndarray:
        """``(T_k,)`` cell-centred patch centres along spatial axis ``k``, in [-1, 1] over the box."""
        return jnp.asarray(_cells(np.arange(self.tokens[k]), self.tokens[k]), jnp.float32)

    def spacings(self, geometry=None) -> jnp.ndarray:
        return jnp.asarray(self.spacing if geometry is None else geometry, jnp.float32)

    def scale(self, geometry=None) -> jnp.ndarray:
        """Log physical half-width of a patch per relative axis, over the reference."""
        half = jnp.asarray(self.half, jnp.float32)
        return jnp.log(half * self.spacings(geometry) / jnp.asarray(self.reference, jnp.float32))

    def features(self, geometry, modes: int) -> jnp.ndarray:
        """``(*T_abs, P, F)`` encoded coordinates and log physical half-widths of every point."""
        lead = self.coords.shape[:-1]
        x = jnp.concatenate(
            [jnp.broadcast_to(self.offsets, (*lead, self.offsets.shape[-1])), self.coords], -1
        )
        scale = self.scale(geometry)
        enc = encode(x, jnp.asarray(self.bands), modes)
        return jnp.concatenate([enc, jnp.broadcast_to(scale, (*lead, scale.shape[-1]))], -1)


def encode(x, caps, modes: int) -> jnp.ndarray:
    """Cosines ``cos(k pi (x + 1) / 2)`` of the coordinates ``x`` in [-1, 1], cut at the ``caps``."""
    k = jnp.arange(modes)
    c = jnp.cos(k * jnp.pi * (x[..., None] + 1) / 2) * (k < caps[:, None])
    return c.reshape(*x.shape[:-1], -1)
