"""Compression benchmark runner: scores reconstructors over test trajectories.

Per (method, trajectory) the scores are a row ``{metric: mean over snapshots} + {method, traj,
n_timesteps}`` written to ``<outdir>/rows/<LABEL>__<traj>.json``, so runs resume; with diagnostics
the per-snapshot spectra go to ``rows/diags/<LABEL>__<traj>.pkl``. :func:`aggregate` builds
``eval1k_<LABEL>.json`` (``{"rows", "agg": {metric: [mean, std] over trajectories}}``).
"""

from __future__ import annotations

import glob
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from neugk_jax.evaluate.integrals import precompute_geometry
from neugk_jax.pinc.benchmark.metrics import (
    moment_weights,
    snapshot_fields,
    snapshot_metrics,
    temporal_epe,
    time_averaged_spectral_metrics,
    velocity_moment_errors,
    zonal_profiles,
)
from neugk_jax.pinc.benchmark.reconstructors import Reconstructor

DIAG_KEYS = ("kxspec", "kyspec", "phi_zf", "qspec")


def traj_name(path: str) -> str:
    return Path(path).name.split("_ifft")[0]


class GroundTruth:
    """Raw df snapshots, geometry and field-line spacing of trajectories under ``path``."""

    def __init__(self, path: str, trajectories: Sequence[str]):
        from neugk_jax.dataset.factory import build_dataset

        dcfg = OmegaConf.create(
            {"path": path, "backend": "numpy", "validation_trajectories": list(trajectories)}
        )
        self.ds = build_dataset(dcfg, split="val", conditions=(), cond_filters=None, offset=0)
        self.fid = {traj_name(f): i for i, f in enumerate(self.ds.files)}

    def trajectory(self, traj: str, timesteps: Sequence[int]) -> "TrajectoryGT":
        fid = self.fid[traj]
        meta = self.ds.metadata[fid]
        frames = [np.asarray(self.ds.read_frame(fid, int(t))["df"], np.float32) for t in timesteps]
        return TrajectoryGT(frames, meta["geometry"], float(meta["ds"]))


def host_diag(fields: dict, geometry: dict) -> dict:
    out = {k: np.asarray(fields[k]) for k in DIAG_KEYS}
    out.update(zonal_profiles(np.asarray(fields["phi_spec"]), geometry))
    return out


class TrajectoryGT:
    """One trajectory's ground truth with its field solves and diagnostics, computed once."""

    def __init__(self, frames: list, geometry: dict, ds: float):
        self.frames, self.geometry, self.ds = frames, geometry, ds
        self.geom_t = precompute_geometry(geometry)
        self.weights = moment_weights(geometry)
        self.fields = [snapshot_fields(self.geom_t, jnp.asarray(f), ds) for f in frames]
        self.diags = [host_diag(f, geometry) for f in self.fields]

    @property
    def nbytes(self) -> int:
        return sum(f.nbytes for f in self.frames)


def evaluate(rec: Reconstructor, traj: str, timesteps: Sequence[int], gt: TrajectoryGT):
    """``(row, diags)`` of one reconstructor on one trajectory.

    ``row`` holds the snapshot metrics averaged over the timesteps, the compression ratio, the
    optical-flow end-point error and the time-averaged spectral metrics; ``diags`` the
    per-timestep spectra of the prediction and (``_gt``) of the ground truth.
    """
    dfs, csize = rec.reconstruct(traj, timesteps, gt.frames)
    metrics: dict[str, list] = defaultdict(list)
    if csize:
        metrics["cr"].append(gt.nbytes / csize)
    pred_diags = []
    for pred, frame, g in zip(dfs, gt.frames, gt.fields):
        pred, frame = jnp.asarray(pred, jnp.float32), jnp.asarray(frame)
        p = snapshot_fields(gt.geom_t, pred, gt.ds)
        for k, v in snapshot_metrics(pred, frame, p, g).items():
            metrics[k].append(float(v))
        for k, v in velocity_moment_errors(pred, frame, gt.weights).items():
            metrics[k].append(v)
        pred_diags.append(host_diag(p, gt.geometry))
    metrics["endpoint"].append(temporal_epe(gt.frames, dfs))
    for k, v in time_averaged_spectral_metrics(pred_diags, gt.diags).items():
        metrics[k].append(v)
    row = {k: float(np.mean(v)) for k, v in metrics.items()}
    diags = {k: [d[k] for d in pred_diags] for k in pred_diags[0]}
    diags.update({f"{k}_gt": [d[k] for d in gt.diags] for k in gt.diags[0]})
    return row, diags


def row_file(outdir: str, label: str, traj: str) -> str:
    return os.path.join(outdir, "rows", f"{label}__{traj}.json")


def write_row(outdir: str, label: str, traj: str, row: dict, diags=None) -> None:
    os.makedirs(os.path.join(outdir, "rows"), exist_ok=True)
    with open(row_file(outdir, label, traj), "w") as f:
        json.dump(row, f, indent=2)
    if diags is not None:
        ddir = os.path.join(outdir, "rows", "diags")
        os.makedirs(ddir, exist_ok=True)
        with open(os.path.join(ddir, f"{label}__{traj}.pkl"), "wb") as f:
            pickle.dump({f"{traj}.h5": diags}, f)


def aggregate(outdir: str, labels: Sequence[str]) -> dict[str, dict]:
    """(Re)build ``eval1k_<LABEL>.json`` from the row files on disk; returns the aggregates."""
    out = {}
    for label in labels:
        rows = []
        for fp in sorted(glob.glob(row_file(outdir, label, "*"))):
            with open(fp) as f:
                rows.append(json.load(f))
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["traj"].split("_")[1]))
        good = [r for r in rows if "error" not in r]
        keys = [k for k in good[0] if k not in ("method", "traj", "n_timesteps")] if good else []
        agg = {
            k: [float(np.mean([r[k] for r in good])), float(np.std([r[k] for r in good]))]
            for k in keys
        }
        agg["n_traj"] = len(good)
        agg["n_err"] = len(rows) - len(good)
        with open(os.path.join(outdir, f"eval1k_{label}.json"), "w") as f:
            json.dump({"rows": rows, "agg": agg}, f, indent=2)
        out[label] = agg
    return out


def scaling_entry(
    rec: Reconstructor, trajectories: Sequence[str], timesteps, gt: GroundTruth
) -> dict:
    """One rate-distortion point: the metrics pooled over every snapshot of ``trajectories``."""
    pooled: dict[str, list] = defaultdict(list)
    for traj in trajectories:
        row, _ = evaluate(rec, traj, timesteps, gt.trajectory(traj, timesteps))
        n = len(timesteps)
        for k, v in row.items():
            per_traj = k in ("cr", "endpoint") or k.startswith(("kyspec_", "qspec_", "zf"))
            pooled[k] += [v] if per_traj else [v] * n
    entry = {"cr": float(np.mean(pooled["cr"])), "name": rec.name}
    entry.update({k: float(np.mean(v)) for k, v in pooled.items()})
    return entry
