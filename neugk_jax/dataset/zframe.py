"""Lossless zstd shards: the ``.zstd16.bin`` sibling of a bf16 shard.

Layout: a header padded to 4 KiB (``MAGIC``, version, element bytes, raw bytes, chunk bytes, chunk
count, then the ``n + 1`` absolute chunk offsets), then the chunks back to back. A chunk holds up to
``chunk_bytes`` raw bytes, byte-shuffled (the first byte of every element, then the second, ...) and
compressed as one standard zstd frame, so it decodes on the CPU (numcodecs) or on the GPU (nvCOMP).
:func:`bf16_from_shuffled` undoes the shuffle on any jax device and is jittable.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numcodecs import Zstd

MAGIC = b"NGKZ"
VERSION = 1
ALIGN = 4096
CHUNK_BYTES = 4 << 20
# magic, version, element bytes, raw bytes, chunk bytes, chunk count
_FIXED = struct.Struct("<4sIIQQQ")


@dataclass(frozen=True)
class Header:
    elem_bytes: int
    raw_bytes: int
    chunk_bytes: int
    offsets: tuple[int, ...]

    @property
    def n_chunks(self) -> int:
        return len(self.offsets) - 1


def header_size(n_chunks: int) -> int:
    return -(-(_FIXED.size + 8 * (n_chunks + 1)) // ALIGN) * ALIGN


def _shuffle(raw: np.ndarray, elem_bytes: int) -> np.ndarray:
    return np.ascontiguousarray(raw.reshape(-1, elem_bytes).T).ravel()


def encode(
    raw: np.ndarray, *, elem_bytes: int = 2, chunk_bytes: int = CHUNK_BYTES, level: int = 1
) -> bytes:
    """``.zstd16`` bytes of the C-ordered payload ``raw`` (``elem_bytes`` bytes per element)."""
    raw = np.ascontiguousarray(raw).view(np.uint8).ravel()
    if raw.size % elem_bytes or chunk_bytes % elem_bytes:
        raise ValueError(
            f"payload {raw.size} B and chunk {chunk_bytes} B must be multiples of {elem_bytes}"
        )
    codec = Zstd(level=level)
    chunks = [
        codec.encode(_shuffle(raw[i : i + chunk_bytes], elem_bytes))
        for i in range(0, raw.size, chunk_bytes)
    ]
    start = header_size(len(chunks))
    offsets = np.cumsum([start] + [len(c) for c in chunks], dtype=np.uint64)
    head = (
        _FIXED.pack(MAGIC, VERSION, elem_bytes, raw.size, chunk_bytes, len(chunks))
        + offsets.tobytes()
    )
    return head.ljust(start, b"\0") + b"".join(chunks)


def parse_header(buf) -> Header:
    magic, version, elem_bytes, raw_bytes, chunk_bytes, n = _FIXED.unpack_from(buf, 0)
    if magic != MAGIC or version != VERSION:
        raise IOError(f"not a .zstd16 v{VERSION} shard (magic {magic!r}, version {version})")
    offsets = np.frombuffer(buf, dtype=np.uint64, count=n + 1, offset=_FIXED.size)
    return Header(elem_bytes, raw_bytes, chunk_bytes, tuple(int(o) for o in offsets))


def decode(
    buf, header: Header | None = None, *, out: np.ndarray | None = None, unshuffle: bool = True
) -> np.ndarray:
    """Raw payload bytes of the ``.zstd16`` shard in ``buf``; with ``unshuffle=False`` every chunk stays shuffled."""
    header = header or parse_header(buf)
    src = memoryview(buf).cast("B")
    out = np.empty(header.raw_bytes, np.uint8) if out is None else out[: header.raw_bytes]
    codec, cb, e = Zstd(), header.chunk_bytes, header.elem_bytes
    for i in range(header.n_chunks):
        dst = out[i * cb : min((i + 1) * cb, header.raw_bytes)]
        codec.decode(src[header.offsets[i] : header.offsets[i + 1]], out=dst)
        if unshuffle:
            dst[:] = dst.reshape(e, -1).T.ravel()
    return out


@partial(jax.jit, static_argnums=(1, 2))
def bf16_from_shuffled(shuffled: jax.Array, raw_bytes: int, chunk_bytes: int) -> jax.Array:
    """Flat bf16 values of a decoded, chunk-shuffled 2-byte payload (uint8 ``shuffled``)."""
    n_full, rem = divmod(raw_bytes, chunk_bytes)
    body = (
        shuffled[: n_full * chunk_bytes]
        .reshape(n_full, 2, chunk_bytes // 2)
        .transpose(0, 2, 1)
        .reshape(-1, 2)
    )
    if rem:
        body = jnp.concatenate(
            [body, shuffled[n_full * chunk_bytes : raw_bytes].reshape(2, rem // 2).T]
        )
    return jax.lax.bitcast_convert_type(
        jax.lax.bitcast_convert_type(body, jnp.uint16), jnp.bfloat16
    )
