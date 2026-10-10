"""Trajectory readers of the preprocessed cyclone dataset.

A backend owns every on-disk detail of a trajectory: the directory or file name, the
metadata files and their key conventions, the geometry defaults, the per-timestep shards
(fp32 or a quantized sibling) and the array type handed back. ``NumpyBackend`` and
``H5Backend`` return host ``numpy`` arrays, ``KvikIOBackend`` device ``jax`` arrays read
GPU-direct. :func:`make_backend` builds the backend of a ``dataset`` config.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib.util
import os
import pickle
import re
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np

from neugk_jax.dataset import quant, zframe
from neugk_jax.utils import atomic_write

# metadata keys left out of metadata_light (the per-element df and phi moments)
LIGHT_DROP_KEYS = (
    "df_min",
    "df_max",
    "df_var",
    "df_mean",
    "df_std",
    "phi_min",
    "phi_max",
    "phi_var",
)
# geometry scalars absent from a trajectory: single species, electrostatic, adiabatic electrons
GEOMETRY_DEFAULTS = {
    "mas": 1.0,
    "tmp": 1.0,
    "d2X": 1.0,
    "signz": 1.0,
    "signB": 1.0,
    "adiabatic": 1.0,
    "de": 1.0,
    "vthrat": 1.0,
    "beta": 0.0,
    "nlapar": 0.0,
    "nlbpar": 0.0,
}


def complete_geometry(geometry: dict) -> dict:
    """Numeric geometry entries with :data:`GEOMETRY_DEFAULTS` for the missing scalars.

    ``ffun`` (the flux-surface function, absent for cyclone s-α at ε→0) defaults to ones.
    """
    g = {k: np.asarray(v) for k, v in geometry.items() if np.asarray(v).dtype.kind in "fiub"}
    for k, v in GEOMETRY_DEFAULTS.items():
        g.setdefault(k, np.array(v, dtype=np.float64))
    if "ffun" not in g and "ints" in g:
        g["ffun"] = np.ones_like(g["ints"], dtype=np.float64)
    return g


def frame_name(kind: str, t: int) -> str:
    return f"{kind}_{int(t):05d}"


def expand_spec(spec) -> list[str]:
    """Trajectory names of a spec: one brace pattern (``iteration_{0-3,7}``) or a list of them."""
    if isinstance(spec, (list, tuple)):
        return [n for s in spec for n in expand_spec(s)]
    m = re.match(r"^(.*?)\{([^}]+)\}(.*?)$", spec)
    if not m:
        return [spec]
    prefix, ranges, suffix = m.groups()
    nums = []
    for part in ranges.split(","):
        lo, _, hi = part.partition("-")
        nums.extend(range(int(lo), int(hi or lo) + 1))
    return [f"{prefix}{n}{suffix}" for n in nums]


def _flatten_meta(meta):
    # npz stores flat string-keyed arrays; nest the geometry dict under "geometry/<key>"
    flat = {}
    for k, v in meta.items():
        if k == "geometry" and isinstance(v, dict):
            for gk, gv in v.items():
                flat[f"geometry/{gk}"] = np.asarray(gv)
        else:
            flat[k] = np.asarray(v)
    return flat


def _unflatten_meta(z):
    meta, geom = {}, {}
    for k in z.files:
        if k.startswith("geometry/"):
            geom[k[len("geometry/") :]] = z[k]
        else:
            meta[k] = z[k]
    if geom:
        meta["geometry"] = geom
    return meta


def meta_path(base: str) -> Optional[str]:
    for ext in (".npz", ".pkl"):
        if os.path.exists(base + ext):
            return base + ext
    return None


def load_meta(base: str) -> Optional[dict]:
    path = meta_path(base)
    if path is None:
        return None
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            return _unflatten_meta(z)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_meta(base: str, meta: dict, ext: str) -> None:
    if ext == ".npz":
        atomic_write(base + ext, lambda f: np.savez(f, **_flatten_meta(meta)))
    else:
        atomic_write(base + ext, lambda f: pickle.dump(meta, f))


def read_bin(file: str, shape: tuple, dtype=np.float32) -> np.ndarray:
    """Read a flat binary into a contiguous ``np.ndarray`` of given shape."""
    arr = np.fromfile(file, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise IOError(f"{file}: expected {expected} elements, got {arr.size}")
    return arr.reshape(shape)


class DataBackend(ABC):
    """Reader of one trajectory layout.

    ``read_metadata`` returns the metadata with the dataset conventions applied: ``flux``
    (not ``fluxes``), ``resolution`` as a tuple of ints and the geometry completed by
    :func:`complete_geometry`. ``read_df`` / ``read_phi`` return float32 frames.
    """

    @abstractmethod
    def trajectory_path(self, path: str) -> str: ...

    @abstractmethod
    def is_valid(self, path: str) -> bool: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def _read_metadata(self, path: str) -> dict: ...

    @abstractmethod
    @contextlib.contextmanager
    def open(self, path: str): ...

    @abstractmethod
    def _read(self, handle: Any, kind: str, t: int, shape: Sequence[int]): ...

    def read_metadata(self, path: str) -> dict:
        meta = self._read_metadata(path)
        if "flux" not in meta and "fluxes" in meta:
            meta["flux"] = meta.pop("fluxes")
        meta["resolution"] = tuple(int(r) for r in np.atleast_1d(meta["resolution"]))
        if meta.get("geometry") is not None:
            meta["geometry"] = complete_geometry(meta["geometry"])
        return meta

    def read_df(self, handle: Any, t: int, shape: Sequence[int]):
        return self._read(handle, "timestep", t, tuple(shape))

    def read_phi(self, handle: Any, t: int, shape: Sequence[int]):
        return self._read(handle, "poten", t, tuple(shape))


class NumpyBackend(DataBackend):
    """Plain numpy reader for the per-timestep .bin layout.

    Directory layout::

        traj_dir/                    (<name>_ifft_realpotens, or _ifft with real_potens=False)
        ├── metadata.{npz,pkl}
        ├── metadata_light.{npz,pkl}  (optional)
        └── data/
            ├── timestep_00000.bin
            ├── timestep_00000.bf16.bin  (optional, ``preprocess --mode=quantize``)
            ├── poten_00000.bin
            └── ...

    ``prefer_dtype`` (``fp16``/``bf16``/``zstd16``/``i8``/``i4``) reads the quantized sibling
    when present and otherwise the fp32 shard round-tripped through that dtype, so the model
    sees the same precision either way; ``zstd16`` is the losslessly compressed ``bf16``.
    ``lightweight_metadata`` reads ``metadata_light`` (no per-element moments) when it exists.
    ``split_into_bands`` names the zonal-flow band layout written by
    ``preprocess --split-into-bands``.
    """

    def __init__(
        self,
        *,
        prefer_dtype: Optional[str] = None,
        real_potens: bool = True,
        split_into_bands: Optional[int] = None,
        lightweight_metadata: bool = False,
    ):
        self.prefer_dtype = prefer_dtype or "fp32"
        self._zstd_announced = False
        self.real_potens = real_potens
        self.split_into_bands = split_into_bands
        self.lightweight_metadata = lightweight_metadata

    def trajectory_path(self, path: str) -> str:
        path = path.removesuffix("/").removesuffix(".h5")
        if self.split_into_bands:
            tag = f"_ifft_separate_zf_{self.split_into_bands}bands_realpotens"
        else:
            tag = "_ifft_realpotens" if self.real_potens else "_ifft"
        return path if tag in path else path + tag

    def is_valid(self, path: str) -> bool:
        return os.path.isdir(path)

    def exists(self, path: str) -> bool:
        # a trajectory is present if it has full or lightweight metadata
        return any(meta_path(os.path.join(path, n)) for n in ("metadata", "metadata_light"))

    def _read_metadata(self, path: str) -> dict:
        light, full = os.path.join(path, "metadata_light"), os.path.join(path, "metadata")
        if (self.lightweight_metadata or meta_path(full) is None) and meta_path(light):
            return load_meta(light)
        meta = load_meta(full)
        if self.lightweight_metadata:
            meta = {k: v for k, v in meta.items() if k not in LIGHT_DROP_KEYS}
        return meta

    @contextlib.contextmanager
    def open(self, path: str):
        # the directory path is the handle of per-timestep reads
        yield path

    def _read(self, handle: str, kind: str, t: int, shape: tuple):
        fp32 = os.path.join(handle, "data", frame_name(kind, t) + ".bin")
        path, bits = quant.resolve(fp32, self.prefer_dtype)
        if bits == "zstd16" and not self._zstd_announced:
            self._zstd_announced = True
            decode = getattr(self, "zstd_decode", "cpu")
            print(f"[data] reading zstd16 shards ({decode} decode), first: {path}", flush=True)
        if bits == "fp32" and self.prefer_dtype != "fp32":
            # no quantized sibling: quantize the fp32 shard on the fly
            return self._roundtrip(fp32, shape)
        return self._read_file(path, bits, shape)

    def _roundtrip(self, fp32: str, shape: tuple):
        return self._output(quant.roundtrip(read_bin(fp32, shape), quant.values(self.prefer_dtype)))

    def _read_file(self, path: str, bits: str, shape: tuple):
        if bits == "fp32":
            return read_bin(path, shape)
        return quant.read(path, bits, int(np.prod(shape))).reshape(shape)

    def _output(self, arr: np.ndarray):
        return arr


def kvikio_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("cupy", "kvikio"))


class KvikIOBackend(NumpyBackend):
    """GPU-direct reads via cupy + kvikio (NVIDIA GDS) into device ``jax`` arrays.

    Shards are read into a cupy buffer on device ``rank`` and handed to jax zero-copy over
    DLPack (fp32 / fp16 / bf16 / i8); i4 shards and on-the-fly quantization go through the
    host. ``zstd16`` shards are decompressed on the host (``zstd_decode="cpu"``) or by nvCOMP on the
    device (``"gpu"``) and unshuffled on the device. Requires the dataloader to run in-process.
    Same layout and options as ``NumpyBackend``.
    """

    # one nvcomp codec per (reader thread, device), shared by all instances
    _nvcomp_codecs: dict = {}

    def __init__(self, rank: int = 0, zstd_decode: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        if zstd_decode not in ("cpu", "gpu"):
            raise ValueError(f"zstd_decode={zstd_decode!r}; one of cpu, gpu")
        self.rank = rank
        self.zstd_decode = zstd_decode
        self._local = threading.local()
        # return 16-bit shards as stored; the consumer upcasts (exactly) per batch
        self.keep_half = False

    @contextlib.contextmanager
    def on_device(self, device):
        """Reads of this thread land on the jax ``device`` (default: local device ``rank``)."""
        prev = getattr(self._local, "device", None)
        self._local.device = device
        try:
            yield
        finally:
            self._local.device = prev

    def _target(self):
        return getattr(self._local, "device", None)

    def _output(self, arr: np.ndarray):
        import jax

        return jax.device_put(arr, self._target())

    def _roundtrip(self, fp32: str, shape: tuple):
        prefer = quant.values(self.prefer_dtype)
        if prefer not in ("bf16", "fp16"):
            return super()._roundtrip(fp32, shape)
        import jax.numpy as jnp

        dtype = jnp.bfloat16 if prefer == "bf16" else jnp.float16
        # on-device round-to-nearest-even, as ml_dtypes on the host
        half = self._read_file(fp32, "fp32", shape).astype(dtype)
        return half if self.keep_half else half.astype(jnp.float32)

    def _read_file(self, path: str, bits: str, shape: tuple):
        import cupy as cp
        import jax.dlpack as jdlp
        import jax.lax as lax
        import jax.numpy as jnp
        import kvikio

        n_elems = int(np.prod(shape))
        if bits == "zstd16":
            return self._read_zstd(path, shape)
        if bits == "i4":
            return self._output(quant.read(path, bits, n_elems).reshape(shape))
        header = quant.HEADER_BYTES if quant.has_header(bits) else 0
        buf_dtype = {"fp32": cp.float32, "fp16": cp.float16, "bf16": cp.uint16, "i8": cp.int8}[bits]
        expected = header + n_elems * np.dtype(buf_dtype).itemsize
        if os.path.getsize(path) != expected:
            raise IOError(f"{path}: expected {expected} bytes, got {os.path.getsize(path)}")
        dev = self._target()
        with cp.cuda.Device(self.rank if dev is None else dev.local_hardware_id):
            gpu = cp.empty(n_elems, dtype=buf_dtype)
            with kvikio.CuFile(path, "r") as fh:
                # payload only, after the scale header
                fh.read(gpu, file_offset=header)
        arr = jdlp.from_dlpack(gpu.reshape(shape))
        if bits == "bf16":
            half = lax.bitcast_convert_type(arr, jnp.bfloat16)
            out = half if self.keep_half else half.astype(jnp.float32)
        elif bits == "i8":
            scale = np.fromfile(path, dtype=np.float32, count=1)[0]
            out = arr.astype(jnp.float32) * jnp.float32(scale)
        elif bits == "fp16" and self.keep_half:
            return arr
        else:
            out = arr.astype(jnp.float32)
        # the cupy buffer is freed on return: every pending op reading it must have run
        return out.block_until_ready()

    def _buffer(self, name: str, nbytes: int, pinned: bool = False) -> np.ndarray:
        # a host buffer per reader thread, grown on demand and reused across reads
        buf = getattr(self._local, name, None)
        if buf is None or buf.size < nbytes:
            if pinned:
                import cupyx

                buf = cupyx.empty_pinned(nbytes, dtype=np.uint8)
            else:
                buf = np.empty(nbytes, np.uint8)
            setattr(self._local, name, buf)
        return buf[:nbytes]

    def _read_zstd(self, path: str, shape: tuple):
        import cupy as cp
        import jax.dlpack as jdlp
        import jax.numpy as jnp

        comp = self._buffer("zcomp", os.path.getsize(path))
        with open(path, "rb", buffering=0) as f:
            if f.readinto(comp) != comp.size:
                raise IOError(f"{path}: short read")
        header = zframe.parse_header(comp)
        if header.elem_bytes != 2 or header.raw_bytes != 2 * int(np.prod(shape)):
            raise IOError(
                f"{path}: {header.raw_bytes} bytes of {header.elem_bytes}-byte values"
                f" for shape {shape}"
            )
        dev = self._target()
        with cp.cuda.Device(self.rank if dev is None else dev.local_hardware_id) as d:
            if self.zstd_decode == "gpu":
                shuffled = self._nvcomp_decode(cp.asarray(comp), header, d.id)
            else:
                host = self._buffer("zraw", header.raw_bytes, pinned=True)
                shuffled = cp.asarray(zframe.decode(comp, header, out=host, unshuffle=False))
            cp.cuda.get_current_stream().synchronize()
        half = zframe.bf16_from_shuffled(
            jdlp.from_dlpack(shuffled), header.raw_bytes, header.chunk_bytes
        )
        half = half.reshape(shape)
        # the cupy buffer is freed on return: every pending op reading it must have run
        return (half if self.keep_half else half.astype(jnp.float32)).block_until_ready()

    def _nvcomp_decode(self, comp, header, device_id: int):
        import cupy as cp
        from nvidia import nvcomp

        key = (threading.get_ident(), device_id)
        if key not in self._nvcomp_codecs:
            if not self._nvcomp_codecs:
                # codecs must go before the cuda context at interpreter exit
                atexit.register(self._nvcomp_codecs.clear)
            self._nvcomp_codecs[key] = nvcomp.Codec(
                algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW, device_id=device_id
            )
        o = header.offsets
        chunks = [nvcomp.as_array(comp[o[i] : o[i + 1]]) for i in range(header.n_chunks)]
        return cp.concatenate([cp.asarray(c) for c in self._nvcomp_codecs[key].decode(chunks)])


class H5Backend(DataBackend):
    """Reader for single-file ``<trajectory>.h5`` trajectories (``ml-jku/gyroswin_cbc_id_ood``).

    Layout::

        traj.h5
        ├── data/timestep_00000            (2, vp, mu, s, x, y) float32, spatial
        ├── data/poten_00000               (x, s, y), optional
        ├── metadata/{timesteps, flux|fluxes, ion_temp_grad, density_grad, s_hat, q, resolution}
        └── geometry/<key>

    A uniform parallel grid without ``ds`` gets ``ds = ints[0]``.
    """

    def trajectory_path(self, path: str) -> str:
        return path if path.endswith(".h5") else path + ".h5"

    def is_valid(self, path: str) -> bool:
        return os.path.isfile(path)

    def exists(self, path: str) -> bool:
        return self.is_valid(path)

    def _read_metadata(self, path: str) -> dict:
        import h5py

        with h5py.File(path, "r") as f:
            meta = {k: np.asarray(v[()]) for k, v in f["metadata"].items()}
            if "geometry" in f:
                meta["geometry"] = {k: np.asarray(v[()]) for k, v in f["geometry"].items()}
        ints = np.asarray(meta.get("geometry", {}).get("ints", ()), np.float64)
        # uniform parallel grid: the s quadrature weight is the grid spacing
        if "ds" not in meta and ints.size and np.allclose(ints, ints[0]):
            meta["ds"] = np.float64(ints[0])
        return meta

    @contextlib.contextmanager
    def open(self, path: str):
        import h5py

        f = h5py.File(path, "r")
        try:
            yield f
        finally:
            f.close()

    def _read(self, f, kind: str, t: int, shape: tuple) -> np.ndarray:
        name = f"data/{frame_name(kind, t)}"
        if name not in f:
            raise FileNotFoundError(f"{f.filename} has no {name}")
        arr = np.asarray(f[name][()], dtype=np.float32)
        if arr.size != int(np.prod(shape)):
            raise IOError(
                f"{f.filename}:{name}: expected {int(np.prod(shape))} elements, got {arr.size}"
            )
        return arr.reshape(shape)


def make_backend(
    dcfg,
    *,
    local_rank: int = 0,
    prefer_dtype: Optional[str] = None,
    lightweight_metadata: bool = False,
) -> DataBackend:
    """Backend named by ``dataset.backend`` (``kvikio``, ``numpy`` or ``h5``).

    ``kvikio`` falls back to ``numpy`` when cupy or kvikio are not installed; ``io_threads``
    (default 8) sets the kvikio threads splitting each read unless ``KVIKIO_NTHREADS`` is set.
    """
    name = dcfg.get("backend", "kvikio")
    if name == "h5":
        return H5Backend()
    kwargs = dict(
        prefer_dtype=prefer_dtype,
        real_potens=bool(dcfg.get("real_potens", True)),
        lightweight_metadata=lightweight_metadata,
    )
    if name == "kvikio" and kvikio_available():
        # read before kvikio's first import
        os.environ.setdefault("KVIKIO_NTHREADS", str(int(dcfg.get("io_threads", 8))))
        return KvikIOBackend(rank=local_rank, zstd_decode=dcfg.get("zstd_decode", "cpu"), **kwargs)
    return NumpyBackend(**kwargs)
