"""Generic utilities: config conversion, trace counting, atomic writes, progress bars,
separate/recombine zonal flow, running stats."""

from __future__ import annotations

import os
from collections import Counter
from typing import Callable

import numpy as np
from omegaconf import OmegaConf

# number of times each named jitted function has been traced
TRACE_COUNTS: Counter = Counter()


def count_trace(name: str) -> None:
    TRACE_COUNTS[name] += 1


def config_dict(cfg) -> dict:
    """Plain resolved container of an OmegaConf node or mapping (``{}`` for None)."""
    if cfg is None:
        return {}
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def atomic_write(path, write: Callable, mode: str = "wb") -> None:
    """Write ``path`` through ``write(file)`` into a temporary sibling, then rename it into place."""
    path = os.fspath(path)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, mode) as f:
        write(f)
    os.replace(tmp, path)


def progress(it, show: bool, **kwargs):
    """``it`` wrapped in a tqdm bar when ``show``."""
    if not show:
        return it
    from tqdm import tqdm

    return tqdm(it, **kwargs)


def separate_zf(x, axis: int = 0):
    """Separate Zonal Flow (ZF) and non-ZF components of a numpy or jax array.

    Layout: ``[zf, x - zf]`` along ``axis``. ZF is the **mean** over the
    last axis (ky), broadcast back; the "rest" is ``x - zf`` (so the
    decomposition is exact, ``zf + rest == x``).
    """
    xp = x.__array_namespace__()
    zf = xp.broadcast_to(x.mean(axis=-1, keepdims=True), x.shape)
    return xp.concatenate([zf, x - zf], axis=axis)


def recombine_zf(x, axis: int = 0):
    """Inverse of ``separate_zf``: ``[zf, non_zf]`` → ``zf + non_zf``."""
    if x.shape[axis] <= 2 or x.shape[axis] % 2 != 0:
        return x
    zf, non_zf = x.__array_namespace__().split(x, 2, axis=axis)
    return zf + non_zf


class RunningStats:
    """Elementwise running mean/var/min/max in float64, seeded with ``prior_count`` (variance 1).

    ``push`` adds one sample in place (Welford), ``merge`` a summary of ``count`` samples
    (Chan et al.); buffers take the shape of the first input. Unpickled ``RunningMeanStd``
    objects of the stored dataset statistics carry the same attributes.
    """

    def __init__(self, prior_count: float):
        self.count = prior_count
        self.mean = self.var = self.min = self.max = None

    def _start(self, shape) -> None:
        self.mean, self.var = np.zeros(shape), np.ones(shape)
        self.min, self.max = np.full(shape, np.inf), np.full(shape, -np.inf)

    def push(self, x) -> None:
        x = np.asarray(x)
        if self.mean is None:
            self._start(x.shape)
        c, n1 = self.count, self.count + 1.0
        d = np.asarray(x - self.mean)
        self.mean += d * (1.0 / n1)
        self.var *= c / n1
        np.multiply(d, d, out=d)
        d *= c / (n1 * n1)
        self.var += d
        np.minimum(self.min, x, out=self.min)
        np.maximum(self.max, x, out=self.max)
        self.count = n1

    def merge(self, mean, var, mn, mx, count: float = 1) -> None:
        mean, var = np.asarray(mean, np.float64), np.asarray(var, np.float64)
        if self.mean is None:
            self._start(mean.shape)
        new_count = self.count + count
        delta = mean - self.mean
        m2 = self.var * self.count + var * count + delta**2 * (self.count * count / new_count)
        self.mean = self.mean + delta * (count / new_count)
        self.var = m2 / new_count
        self.min = np.minimum(self.min, np.asarray(mn, np.float64))
        self.max = np.maximum(self.max, np.asarray(mx, np.float64))
        self.count = new_count

    def moments(self, dtype=None, axes=None) -> dict:
        """``mean``/``var``/``std``/``min``/``max``, optionally pooled over ``axes`` (keepdims).

        Pooling over ``axes`` takes the mean of the means, the mean variance plus the
        variance of the means, and the extreme min/max.
        """
        mean, var = np.asarray(self.mean, np.float64), np.asarray(self.var, np.float64)
        mn, mx = np.asarray(self.min, np.float64), np.asarray(self.max, np.float64)
        if axes:
            mean, var = (
                mean.mean(axis=axes, keepdims=True),
                var.mean(axis=axes, keepdims=True) + mean.var(axis=axes, keepdims=True),
            )
            mn, mx = mn.min(axis=axes, keepdims=True), mx.max(axis=axes, keepdims=True)
        out = {"mean": mean, "var": var, "std": np.sqrt(var), "min": mn, "max": mx}
        return out if dtype is None else {k: v.astype(dtype) for k, v in out.items()}
