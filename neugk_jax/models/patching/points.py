"""Point coordinates of a patch for the field patching layers.

A grid spec describes the coordinates of a channel-last grid ``(*spatial, C)``: every spatial axis is
``relative`` (cell-centred offset from the centre of its block, in [-1, 1], with a physical ``spacing``)
or ``absolute`` (node values mapped to [-1, 1] by ``range`` or the node extent, with quadrature
``weights``); absolute axes may be folded into the channels as ``(c, *folded)``. Without a spec every
axis is relative with unit spacing. A point at cell ``j`` of a block of ``q`` cells sits at
``(2 j + 1) / q - 1``, so the same physical position keeps its coordinate when the resolution changes.
A relative axis may give its ``band``, the number of modes the data resolves per patch (default: the
patch's point count); cosines are cut there, so a finer sampling of the same data adds no modes.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from neugk_jax.models.ops import _normalize_patch

ENCODINGS = ("fourier", "cosine", "ipe")


def _cells(j, q):
    return (2 * np.asarray(j, np.float64) + 1) / q - 1


def _mapped(axis: Mapping, n_padded: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes in [-1, 1], quadrature weights and cell widths of an absolute axis padded to ``n_padded``."""
    nodes = np.asarray(axis["nodes"], np.float64)
    w = np.asarray(axis.get("weights", np.ones_like(nodes)), np.float64)
    extra = n_padded - len(nodes)
    step = nodes[-1] - nodes[-2] if len(nodes) > 1 else 1.0
    # padded nodes continue the grid with zero weight
    full = np.concatenate([nodes, nodes[-1] + step * np.arange(1, extra + 1)])
    lo, hi = axis.get("range", (nodes.min(), nodes.max()))
    unit = 2.0 * (full - lo) / max(hi - lo, 1e-12) - 1.0
    cell = np.abs(np.gradient(unit)) if len(unit) > 1 else np.full(1, 2.0)
    return unit, np.concatenate([w, np.zeros(extra)]), cell


def _mean_one(w: np.ndarray) -> np.ndarray:
    return w / w[w > 0].mean()


class AxisPoints(eqx.Module):
    """One axis of a patch: coordinates, quadrature weights and cell widths, ``(p,)`` or ``(T, p)`` per token."""

    buffer_fields = ("coords", "weight", "cell")

    coords: jnp.ndarray
    weight: jnp.ndarray
    cell: jnp.ndarray
    kind: str = eqx.field(static=True)
    cap: int = eqx.field(static=True)

    def __init__(self, coords, weight, cell, kind: str, cap: int):
        self.coords = jnp.asarray(coords, jnp.float32)
        self.weight = jnp.asarray(weight, jnp.float32)
        self.cell = jnp.asarray(cell, jnp.float32)
        self.kind = kind
        self.cap = int(cap)


class PointGrid(eqx.Module):
    """Coordinates of the points of one patch, split into anchor blocks, in ``fold_patches`` order.

    Per point: ``offsets`` (cell-centred offsets of the relative axes from the centre of the point's
    block), ``coords`` (absolute spatial and folded coordinates per token ``(*T_abs, P, n_abs)``),
    ``cell`` (cell widths of all coordinates), ``pos`` (cell-centred position in the patch along every
    spatial axis), ``channel`` and ``weight``; ``perm`` sorts the points by block and ``unit`` places
    the block centres in the patch. ``axes`` holds the same per spatial and folded axis for separable
    bases.
    """

    buffer_fields = ("coords", "offsets", "cell", "pos", "channel", "weight", "unit", "perm", "inv")

    coords: jnp.ndarray
    offsets: jnp.ndarray
    cell: jnp.ndarray
    pos: jnp.ndarray
    channel: jnp.ndarray
    weight: jnp.ndarray
    unit: jnp.ndarray
    perm: jnp.ndarray
    inv: jnp.ndarray
    axes: tuple[AxisPoints, ...]
    patch: tuple[int, ...] = eqx.field(static=True)
    rel_axes: tuple[int, ...] = eqx.field(static=True)
    abs_axes: tuple[int, ...] = eqx.field(static=True)
    n_fold: tuple[int, ...] = eqx.field(static=True)
    half: tuple[float, ...] = eqx.field(static=True)
    caps: tuple[int, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    reference: tuple[float, ...] = eqx.field(static=True)
    n_anchor: int = eqx.field(static=True)

    def __init__(
        self,
        padded: Sequence[int],
        patch: Sequence[int],
        channels: int,
        spec: Optional[Mapping] = None,
        anchors: Optional[Sequence[int]] = None,
    ):
        patch = _normalize_patch(patch)
        n = len(patch)
        anchors = tuple(anchors or (1,) * n)
        spec = dict(spec or {})
        axes = list(spec.get("axes") or [{"kind": "relative"}] * n)
        folded = list(spec.get("folded") or [])
        if len(axes) != n:
            raise ValueError(f"grid spec has {len(axes)} axes for {n} spatial axes")
        n_fold = tuple(len(f["nodes"]) for f in folded)
        if channels % max(1, math.prod(n_fold)):
            raise ValueError(f"{channels} channels do not hold the folded axes {n_fold}")
        if any(p % a for p, a in zip(patch, anchors)):
            raise ValueError(f"anchors {anchors} do not divide the patch {patch}")
        n_c = channels // math.prod(n_fold)
        sub = [p // a for p, a in zip(patch, anchors)]
        idx = np.meshgrid(*(np.arange(k) for k in (*patch, n_c, *n_fold)), indexing="ij")
        idx = [i.reshape(-1) for i in idx]
        offs, ic, ifold = idx[:n], idx[n], idx[n + 1 :]
        block = [o // q for o, q in zip(offs, sub)]
        anchor = np.ravel_multi_index(block, anchors)
        first = np.array([np.flatnonzero(anchor == k)[0] for k in range(math.prod(anchors))])

        rel = [i for i, a in enumerate(axes) if a.get("kind", "relative") == "relative"]
        band = [min(patch[i], int(axes[i].get("band", patch[i]))) for i in range(n)]
        self.rel_axes = tuple(rel)
        self.abs_axes = tuple(i for i in range(n) if i not in rel)
        tokens = [padded[i] // patch[i] for i in self.abs_axes]
        lead = (*tokens, len(anchor))
        coords, cells, weight, per_axis = [], [], np.ones(lead), []
        for i in range(n):
            if i in rel:
                per_axis.append(
                    AxisPoints(
                        _cells(np.arange(patch[i]), patch[i]),
                        np.ones(patch[i]),
                        np.full(patch[i], 2.0 / patch[i]),
                        "relative",
                        band[i],
                    )
                )
                continue
            j = self.abs_axes.index(i)
            unit, w, cell = _mapped(axes[i], padded[i])
            rows = np.arange(tokens[j])[:, None] * patch[i] + np.arange(patch[i])[None]
            per_axis.append(
                AxisPoints(
                    unit[rows], _mean_one(w[rows]), cell[rows], "absolute", len(axes[i]["nodes"])
                )
            )
            pos = np.arange(tokens[j])[:, None] * patch[i] + offs[i][None]
            shape = [1] * len(tokens) + [len(anchor)]
            shape[j] = tokens[j]
            coords.append(np.broadcast_to(unit[pos].reshape(shape), lead))
            cells.append(np.broadcast_to(cell[pos].reshape(shape), lead))
            weight = weight * w[pos].reshape(shape)
        for f, i in zip(folded, ifold):
            unit, w, cell = _mapped(f, len(f["nodes"]))
            per_axis.append(AxisPoints(unit, _mean_one(w), cell, "folded", len(unit)))
            coords.append(np.broadcast_to(unit[i], lead))
            cells.append(np.broadcast_to(cell[i], lead))
            weight = weight * w[i]
        self.axes = tuple(per_axis)
        self.coords = jnp.asarray(
            np.stack(coords, -1) if coords else np.zeros((*lead, 0)), jnp.float32
        )
        self.weight = jnp.asarray(_mean_one(weight), jnp.float32)
        rel_cell = np.broadcast_to(np.asarray([2.0 / sub[i] for i in rel]), (*lead, len(rel)))
        self.cell = jnp.asarray(
            np.concatenate([rel_cell, np.stack(cells, -1) if cells else np.zeros((*lead, 0))], -1),
            jnp.float32,
        )
        local = [_cells(o - b * q, q) for o, b, q in zip(offs, block, sub)]
        self.offsets = jnp.asarray(
            np.stack([local[i] for i in rel], -1) if rel else np.zeros((len(anchor), 0)),
            jnp.float32,
        )
        self.pos = jnp.asarray(
            np.stack([_cells(o, p) for o, p in zip(offs, patch)], -1), jnp.float32
        )
        self.channel = jnp.asarray(np.eye(n_c)[ic], jnp.float32)
        self.unit = jnp.asarray(
            np.stack([((2 * b + 1) * q / p - 1)[first] for b, q, p in zip(block, sub, patch)], -1),
            jnp.float32,
        )
        self.patch = patch
        self.n_fold = n_fold
        self.half = tuple(sub[i] / 2 for i in rel)
        self.caps = (
            tuple(max(1, -(-band[i] * sub[i] // patch[i])) for i in rel)
            + tuple(len(axes[i]["nodes"]) for i in self.abs_axes)
            + n_fold
        )
        self.spacing = tuple(float(axes[i].get("spacing", 1.0)) for i in rel)
        # physical half-width of a full patch at the nominal spacing, unless given
        self.reference = tuple(
            float(axes[i].get("reference", patch[i] / 2 * s)) for i, s in zip(rel, self.spacing)
        )
        perm = np.argsort(anchor, kind="stable")
        self.perm, self.inv = jnp.asarray(perm), jnp.asarray(np.argsort(perm))
        self.n_anchor = len(first)

    @property
    def n_coords(self) -> int:
        return self.offsets.shape[-1] + self.coords.shape[-1]

    @property
    def n_channels(self) -> int:
        return self.channel.shape[-1]

    def spacings(self, geometry=None) -> jnp.ndarray:
        return jnp.asarray(self.spacing if geometry is None else geometry, jnp.float32)

    def scale(self, geometry=None, half=None) -> jnp.ndarray:
        """Log physical half-width of a block (``half`` cells per relative axis) over the reference."""
        half = jnp.asarray(self.half if half is None else half, jnp.float32)
        return jnp.log(half * self.spacings(geometry) / jnp.asarray(self.reference, jnp.float32))

    def features(self, geometry, encoding: str, n_freq: int, modes: int) -> jnp.ndarray:
        """``(*T_abs, P, F)`` encoded coordinates, channel and log physical half-width of every point."""
        lead = self.coords.shape[:-1]
        x = jnp.concatenate(
            [jnp.broadcast_to(self.offsets, (*lead, self.offsets.shape[-1])), self.coords], -1
        )
        enc = encode(x, self.cell, jnp.asarray(self.caps), encoding, n_freq, modes)
        scale = self.scale(geometry)
        return jnp.concatenate(
            [
                enc,
                jnp.broadcast_to(self.channel, (*lead, self.n_channels)),
                jnp.broadcast_to(scale, (*lead, scale.shape[-1])),
            ],
            -1,
        )

    def axis_features(
        self, k: int, geometry, encoding: str, n_freq: int, modes: int
    ) -> jnp.ndarray:
        """``(..., p, F)`` encoded coordinates of axis ``k`` (spatial, then folded), plus the log half-width of a relative axis."""
        ax = self.axes[k]
        enc = encode(
            ax.coords[..., None], ax.cell[..., None], jnp.asarray([ax.cap]), encoding, n_freq, modes
        )
        if ax.kind != "relative":
            return enc
        j = self.rel_axes.index(k)
        scale = self.scale(geometry, [self.patch[k] / 2 if i == k else 1.0 for i in self.rel_axes])[
            j
        ]
        return jnp.concatenate([enc, jnp.broadcast_to(scale, (*enc.shape[:-1], 1))], -1)


def encoded_index(
    ids: Sequence[int], n_coords: int, encoding: str, n_freq: int, modes: int
) -> list[int]:
    """Positions of the encodings of the coordinates ``ids`` in the :meth:`PointGrid.features` layout."""
    if encoding == "fourier":
        sin = [n_coords + c * n_freq + f for c in ids for f in range(n_freq)]
        return [*ids, *sin, *(i + n_coords * n_freq for i in sin)]
    return [c * modes + m for c in ids for m in range(modes)]


def n_encoded(n_coords: int, encoding: str, n_freq: int, modes: int) -> int:
    return n_coords * (1 + 2 * n_freq if encoding == "fourier" else modes)


def encode(x, cell, caps, encoding: str, n_freq: int, modes: int) -> jnp.ndarray:
    """Encoding of the coordinates ``x`` in [-1, 1]: fourier octaves, cosines cut at the resolution
    ``caps``, or cosines averaged over the cell (``ipe``, a box footprint of width ``cell``)."""
    lead = x.shape[:-1]
    if encoding == "fourier":
        z = x[..., None] * (jnp.pi * 2.0 ** jnp.arange(n_freq))
        return jnp.concatenate(
            [x, jnp.sin(z).reshape(*lead, -1), jnp.cos(z).reshape(*lead, -1)], -1
        )
    k = jnp.arange(modes)
    c = jnp.cos(k * jnp.pi * (x[..., None] + 1) / 2)
    if encoding == "cosine":
        c = c * (k < caps[:, None])
    elif encoding == "ipe":
        c = c * jnp.sinc(k * jnp.broadcast_to(cell, x.shape)[..., None] / 4)
    else:
        raise ValueError(f"encoding={encoding!r}; one of {ENCODINGS}")
    return c.reshape(*lead, -1)
