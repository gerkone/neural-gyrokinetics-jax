"""Component specs: a kind (``"swin"``) or a mapping ``{"kind": ..., **options}`` grouped by the caller."""

from __future__ import annotations

import inspect
from typing import Callable, Mapping, Union

Spec = Union[str, Mapping]


def parse_spec(spec: Spec, registry: Mapping) -> tuple[str, dict]:
    """``(kind, options)`` of a spec, checked against the ``registry`` kinds."""
    if isinstance(spec, str):
        kind, options = spec, {}
    else:
        options = dict(spec)
        kind = options.pop("kind")
    if kind not in registry:
        raise ValueError(f"kind {kind!r}; one of {sorted(registry)}")
    return kind, options


def accepted(factory: Callable, shared: Mapping) -> dict:
    """The entries of ``shared`` that ``factory`` takes as keywords."""
    params = inspect.signature(factory).parameters
    return {k: v for k, v in shared.items() if k in params}
