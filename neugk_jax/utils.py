"""Generic utilities: config conversion, trace counting, separate/recombine zonal flow, running stats."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import jax.numpy as jnp
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


def separate_zf(x, axis: int = 0):
    """Separate Zonal Flow (ZF) and non-ZF components.

    Layout: ``[zf, x - zf]`` along ``axis``. ZF is the **mean** over the
    last axis (ky), broadcast back; the "rest" is ``x - zf`` (so the
    decomposition is exact, ``zf + rest == x``).

    Works on both numpy and jax arrays — picks the right namespace via
    duck typing.
    """
    if isinstance(x, jnp.ndarray):
        zf = jnp.broadcast_to(x.mean(axis=-1, keepdims=True), x.shape)
        return jnp.concatenate([zf, x - zf], axis=axis)
    zf = np.broadcast_to(x.mean(axis=-1, keepdims=True), x.shape)
    return np.concatenate([zf, x - zf], axis=axis)


def recombine_zf(x, axis: int = 0):
    """Inverse of ``separate_zf``: ``[zf, non_zf]`` → ``zf + non_zf``."""
    if x.shape[axis] <= 2 or x.shape[axis] % 2 != 0:
        return x
    if isinstance(x, jnp.ndarray):
        zf, non_zf = jnp.split(x, 2, axis=axis)
    else:
        zf, non_zf = np.split(x, 2, axis=axis)
    return zf + non_zf


@dataclass
class RunningMeanStd:
    """Numerically stable running mean/var over batches (numpy buffers)."""

    mean: np.ndarray | float = 0.0
    var: np.ndarray | float = 1.0
    min: np.ndarray | float = math.inf
    max: np.ndarray | float = -math.inf
    count: float = 0.0

    def __init__(self, shape: tuple[int, ...] | None = None):
        if shape is None:
            self.mean = 0.0
            self.var = 1.0
            self.min = math.inf
            self.max = -math.inf
        else:
            self.mean = np.zeros(shape, dtype=np.float64)
            self.var = np.ones(shape, dtype=np.float64)
            self.min = np.full(shape, math.inf, dtype=np.float64)
            self.max = np.full(shape, -math.inf, dtype=np.float64)
        self.count = 0.0

    def update(self, mean, var, mn, mx, count: int = 1) -> None:
        mean = np.asarray(mean, dtype=np.float64)
        var = np.asarray(var, dtype=np.float64)
        mn = np.asarray(mn, dtype=np.float64)
        mx = np.asarray(mx, dtype=np.float64)
        new_count = self.count + count
        delta = mean - np.asarray(self.mean)
        new_mean = np.asarray(self.mean) + delta * (count / new_count)
        m_a = np.asarray(self.var) * self.count
        m_b = var * count
        m2 = m_a + m_b + (delta**2) * (self.count * count / new_count)
        new_var = m2 / new_count
        self.mean = new_mean
        self.var = new_var
        self.min = np.minimum(self.min, mn)
        self.max = np.maximum(self.max, mx)
        self.count = new_count
