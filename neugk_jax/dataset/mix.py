"""A mix of ``CycloneDataset`` parts of different data types (grids, species), served jointly."""

from __future__ import annotations

import contextlib
from typing import Any, Mapping, Optional

import numpy as np

from neugk_jax.dataset.cyclone import CycloneDataset, CycloneSample


class CycloneMix:
    """Concatenation of named ``CycloneDataset`` parts, one per data type.

    Part ``i`` serves its samples with ``data_type = i`` and is decoded by the model stem
    named like the part (``stem``); ``groups`` holds the data type of
    every flat index so batches can be drawn from one part at a time. The parts share the
    conditions and the zonal-flow layout; per-part attributes (resolution, statistics,
    metadata) stay on ``parts``, and the dataset-level ones are those of the first part.

    ``mixing`` gives every part its sampling share, a constant or a schedule over the
    training progress as for the loss weights (``start``, ``end``, ``start_fraction``,
    ``end_fraction``, ``type``), or ``{share: ..., loss_weight: ...}`` to also weight the
    part's samples in the training loss; without it parts are sampled in proportion to
    their size.
    ``stems`` maps a part to the model stem decoding it (default: the part's name), so
    several parts on one grid share a stem.
    """

    def __init__(
        self,
        parts: Mapping[str, CycloneDataset],
        mixing: Optional[Mapping[str, Any]] = None,
        stems: Optional[Mapping[str, str]] = None,
    ):
        from neugk_jax.training.loss_scheduler import build_scheduler_dict

        self.parts = dict(parts)
        mixing = dict(mixing or {})
        if mixing and set(mixing) != set(self.parts):
            raise ValueError(f"dataset.mixing names {sorted(mixing)}, the parts are {sorted(self.parts)}")
        for name, spec in list(mixing.items()):
            if isinstance(spec, Mapping) and ("share" in spec or "loss_weight" in spec):
                self.parts[name].loss_weight = float(spec.get("loss_weight", 1.0))
                mixing[name] = spec["share"]
        constant = {n: v for n, v in mixing.items() if not isinstance(v, Mapping)}
        scheduled = build_scheduler_dict({n: v for n, v in mixing.items() if n not in constant})
        self.mixing = {
            **{n: (lambda _, v=float(v): v) for n, v in constant.items()},
            **scheduled,
        }
        first = next(iter(self.parts.values()))
        for name, ds in self.parts.items():
            if ds.conditions != first.conditions or ds.separate_zf != first.separate_zf:
                raise ValueError(f"part {name!r}: conditions / separate_zf differ from the first part")
        stems = dict(stems or {})
        for i, (name, ds) in enumerate(self.parts.items()):
            ds.data_type, ds.stem = i, stems.get(name, name)
        sizes = [len(ds) for ds in self.parts.values()]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.groups = np.repeat(np.arange(len(sizes)), sizes)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def group_weights(self, progress: float) -> Optional[np.ndarray]:
        """Sampling share of every part at training ``progress`` in [0, 1], or None (by size)."""
        if not self.mixing:
            return None
        w = np.asarray([self.mixing[n](1.0 - progress) for n in self.parts], np.float64)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"dataset.mixing shares {w} at progress {progress}")
        return w / w.sum()

    @property
    def batch_transform(self) -> bool:
        return next(iter(self.parts.values())).batch_transform

    @batch_transform.setter
    def batch_transform(self, on: bool) -> None:
        for ds in self.parts.values():
            ds.batch_transform = on

    def transform(self, df, fids, data_type):
        """:meth:`CycloneDataset.transform` of the part ``data_type`` (one per batch)."""
        return list(self.parts.values())[int(np.asarray(data_type).ravel()[0])].transform(df, fids)

    @contextlib.contextmanager
    def on_device(self, device):
        """Reads of this thread land on ``device``, whichever part serves them."""
        with contextlib.ExitStack() as stack:
            for ds in self.parts.values():
                on = getattr(ds.backend, "on_device", None)
                if on is not None:
                    stack.enter_context(on(device))
            yield

    def __getitem__(self, index: int) -> CycloneSample:
        part = int(self.groups[index])
        return list(self.parts.values())[part][int(index - self.offsets[part])]

    def __getattr__(self, name):
        # dataset-level attributes not defined here come from the first part
        if name in ("parts", "offsets", "groups", "mixing", "_batch_transform"):
            raise AttributeError(name)
        return getattr(next(iter(self.parts.values())), name)
