"""Lossless ``.zstd16`` shards: encode / decode, the jitted device unshuffle, the backend read."""

from __future__ import annotations

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from neugk_jax.dataset import quant, zframe
from neugk_jax.dataset.backend import NumpyBackend


@pytest.mark.parametrize(
    "n, chunk_bytes", [(100_003, 1 << 12), (4096, 8192), (65_536, 1 << 12), (5, 4)]
)
def test_roundtrip_and_device_unshuffle(n, chunk_bytes):
    x = np.random.default_rng(n).standard_normal(n).astype(ml_dtypes.bfloat16)
    blob = zframe.encode(x, chunk_bytes=chunk_bytes)
    header = zframe.parse_header(blob)
    assert header.raw_bytes == x.nbytes and len(blob) == header.offsets[-1]
    assert np.array_equal(zframe.decode(blob).view(np.uint16), x.view(np.uint16))
    shuffled = zframe.decode(blob, unshuffle=False)
    dev = zframe.bf16_from_shuffled(jnp.asarray(shuffled), header.raw_bytes, header.chunk_bytes)
    assert np.array_equal(np.asarray(dev).view(np.uint16), x.view(np.uint16))


def test_rejects_other_files():
    with pytest.raises(IOError):
        zframe.parse_header(b"\0" * 4096)


def test_backend_reads_zstd16_as_the_bf16_shard(tmp_path):
    shape = (2, 1, 4, 3, 5, 7, 4)
    x = np.random.default_rng(0).standard_normal(shape).astype(ml_dtypes.bfloat16)
    traj = tmp_path / "traj"
    (traj / "data").mkdir(parents=True)
    x.tofile(traj / "data" / "timestep_00000.bf16.bin")
    ref = NumpyBackend(prefer_dtype="bf16").read_df(str(traj), 0, shape)
    (traj / "data" / "timestep_00000.zstd16.bin").write_bytes(zframe.encode(x, chunk_bytes=64))
    fp32 = str(traj / "data" / "timestep_00000.bin")
    # opt-in only: a bf16 preference never reads the compressed sibling
    assert quant.resolve(fp32, "bf16")[1] == "bf16" and quant.resolve(fp32, "zstd16")[1] == "zstd16"
    out = NumpyBackend(prefer_dtype="zstd16").read_df(str(traj), 0, shape)
    assert out.dtype == np.float32 and np.array_equal(out, ref)
    (traj / "data" / "timestep_00000.zstd16.bin").unlink()
    assert quant.resolve(fp32, "zstd16")[1] == "bf16"


def test_zstd16_preference_rounds_fp32_shards_to_bf16(tmp_path):
    shape = (2, 1, 4, 3, 5, 7, 4)
    traj = tmp_path / "traj"
    (traj / "data").mkdir(parents=True)
    np.random.default_rng(1).standard_normal(shape).astype(np.float32).tofile(
        traj / "data" / "timestep_00000.bin"
    )
    a = NumpyBackend(prefer_dtype="bf16").read_df(str(traj), 0, shape)
    b = NumpyBackend(prefer_dtype="zstd16").read_df(str(traj), 0, shape)
    assert np.array_equal(a, b)
