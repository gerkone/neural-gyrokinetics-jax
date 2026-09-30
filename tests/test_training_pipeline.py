"""Batch pipeline, jitted train step and checkpoint writer of the shared runner base."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neugk_jax.training.checkpoint import AsyncCheckpointer, CheckpointState, load_checkpoint
from neugk_jax.training.data import BatchLoader, eval_plans, train_plans
from neugk_jax.training.ddp import init_distributed, shard_batch
from neugk_jax.utils import TRACE_COUNTS


class _Rows:
    def __init__(self, n):
        self.rows = np.arange(n * 3, dtype=np.float32).reshape(n, 3)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _load(ds, indices, read):
    return {"x": np.stack(read(ds, indices)), "i": np.asarray(indices, np.int32)}


def test_eval_batches_keep_one_shape_and_mask_the_tail():
    ds = _Rows(7)
    dist = init_distributed()
    plans = eval_plans(dist, range(len(ds)), 3)
    loader = BatchLoader(workers=2, prefetch=2)
    seen, shapes = [], set()
    for plan, batch, _ in loader.iterate(ds, plans, _load, lambda b: shard_batch(dist, b)):
        assert isinstance(batch["x"], jax.Array)
        shapes.add(batch["x"].shape)
        seen.extend(np.asarray(batch["i"])[plan.mask > 0].tolist())
    loader.close()
    assert shapes == {(3, 3)}
    assert plans[-1].mask.tolist() == [1.0, 0.0, 0.0]
    assert plans[-1].indices.tolist() == [6, 6, 6]
    assert seen == list(range(7))


def test_eval_plans_max_batches_and_train_plans_drop_the_tail():
    dist = init_distributed()
    assert len(eval_plans(dist, range(10), 3, max_batches=2)) == 2
    perm = np.random.default_rng(0).permutation(7)
    plans = train_plans(dist, 7, 3 // dist.device_count or 1, perm)
    gbs = (3 // dist.device_count or 1) * dist.device_count
    assert len(plans) == 7 // gbs
    assert all(len(p.indices) == gbs // dist.num_processes for p in plans)


def test_prefetch_propagates_load_errors():
    def bad(ds, indices, read):
        raise RuntimeError("boom")

    loader = BatchLoader()
    with pytest.raises(RuntimeError, match="boom"):
        list(loader.iterate(_Rows(4), eval_plans(None, range(4), 2), bad, lambda b: b))
    loader.close()


def test_async_checkpoint_roundtrip(tmp_path):
    model = {"w": jnp.arange(4.0), "b": jnp.ones((2, 2))}
    opt = {"mu": jnp.zeros(4)}
    ck = AsyncCheckpointer()
    ck.save(tmp_path / "ckp.eqx", CheckpointState(model, opt, epoch=3, loss=0.5, meta={"best_val": 0.25}))
    # a second save waits for the first write and replaces the file atomically
    model2 = jax.tree_util.tree_map(lambda a: a + 1, model)
    ck.save(tmp_path / "ckp.eqx", CheckpointState(model2, opt, epoch=4, loss=0.4, meta={"best_val": 0.2}))
    ck.join()
    state = load_checkpoint(tmp_path / "ckp.eqx", model)
    assert state.epoch == 4 and state.meta["best_val"] == 0.2
    assert np.array_equal(state.model["w"], np.arange(4.0) + 1)
    assert not list(tmp_path.glob("*.tmp"))


def test_async_checkpoint_reports_write_errors(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    ck = AsyncCheckpointer()
    ck.save(blocker / "ckp.eqx", CheckpointState({"w": jnp.ones(2)}, None, epoch=1, loss=0.0))
    with pytest.raises(OSError):
        ck.join()


def test_ae_train_step_does_not_retrace_across_epochs(tmp_path):
    from test_e2e_loop import _make_traj, _tiny_ae_cfg

    from neugk_jax.autoencoders.runner import AERunner

    res = (4, 4, 4, 16, 8)
    _make_traj(tmp_path, "iteration_0", n_t=5, resolution=res)
    _make_traj(tmp_path, "iteration_1", n_t=4, resolution=res)
    cfg = _tiny_ae_cfg(tmp_path, res, tmp_path / "run")
    cfg.training.n_epochs = 2
    r = AERunner(cfg, output_path=cfg.output_path)
    logs1, info = r.train_epoch(1, jr.PRNGKey(0))
    r.evaluate(1)
    TRACE_COUNTS.clear()
    logs2, _ = r.train_epoch(2, jr.PRNGKey(1))
    r.evaluate(2)
    assert TRACE_COUNTS["train_step:AERunner"] == 0 and TRACE_COUNTS["ae_eval_step"] == 0
    assert set(logs1) == {"total", "df"} and np.isfinite(logs2["total"])
    assert {"data_ms", "step_ms", "first_step_ms"} <= set(info)
