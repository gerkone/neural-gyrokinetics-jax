"""Distributed setup for jax.distributed (SLURM or torchrun) and data-parallel batch helpers.

One global ``Mesh`` with a single data axis spans every device of every process. Batches
are loaded per process (``process_batch_indices``) and assembled into global arrays
(``shard_batch``); model and optimizer state are replicated (``replicate``).
``training.batch_size`` is per device, so the global batch is ``batch_size * device_count``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@dataclass
class DistributedInfo:
    process_id: int
    num_processes: int
    local_device_count: int
    local_rank: int
    mesh: Mesh

    @property
    def is_rank0(self) -> bool:
        return self.process_id == 0

    @property
    def device_count(self) -> int:
        return self.mesh.size


def _torchrun_env() -> dict | None:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 or "RANK" not in os.environ:
        return None
    return dict(
        coordinator_address=f"{os.environ.get('MASTER_ADDR', 'localhost')}:"
                            f"{os.environ.get('MASTER_PORT', '29500')}",
        num_processes=int(os.environ["WORLD_SIZE"]),
        process_id=int(os.environ["RANK"]),
        # torchrun starts one process per gpu
        local_device_ids=[int(os.environ.get("LOCAL_RANK", "0"))],
    )


def _slurm_multiprocess() -> bool:
    return "SLURM_JOB_ID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1


def init_distributed(*, axis_name: str = "dp") -> DistributedInfo:
    """Initialise jax.distributed under torchrun or multi-task SLURM, else single process."""
    if not jax.distributed.is_initialized():
        tr = _torchrun_env()
        if tr is not None:
            jax.distributed.initialize(**tr)
        elif _slurm_multiprocess():
            # jax's slurm cluster detection resolves the coordinator from the nodelist
            jax.distributed.initialize()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    return DistributedInfo(
        process_id=jax.process_index(),
        num_processes=jax.process_count(),
        local_device_count=jax.local_device_count(),
        local_rank=local_rank,
        mesh=Mesh(np.asarray(jax.devices()), (axis_name,)),
    )


def data_sharding(mesh: Mesh, axis_name: str = "dp") -> NamedSharding:
    return NamedSharding(mesh, P(axis_name))


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def global_batch_size(dist: DistributedInfo, per_device: int) -> int:
    return per_device * dist.device_count


def process_batch_indices(dist: DistributedInfo, window: np.ndarray) -> np.ndarray:
    """This process's contiguous share of one global batch window of sample indices."""
    per_proc = len(window) // dist.num_processes
    return window[dist.process_id * per_proc:(dist.process_id + 1) * per_proc]


def shard_batch(dist: DistributedInfo, tree):
    """Assemble per-process host batches into arrays sharded on the leading axis."""
    if dist.device_count == 1:
        return tree
    sharding = data_sharding(dist.mesh)

    def put(x):
        if x is None:
            return None
        if dist.num_processes > 1:
            return jax.make_array_from_process_local_data(sharding, np.asarray(x))
        return jax.device_put(x, sharding)

    return jax.tree_util.tree_map(put, tree)


def replicate(dist: DistributedInfo, tree):
    """Place every array leaf of ``tree`` fully replicated on the mesh."""
    if dist.device_count == 1:
        return tree
    rep = replicated(dist.mesh)
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, rep) if isinstance(x, jax.Array | np.ndarray) else x, tree)


def eval_batch_owner(dist: DistributedInfo, batch_idx: int) -> bool:
    """Round-robin assignment of evaluation batches to processes."""
    return batch_idx % dist.num_processes == dist.process_id
