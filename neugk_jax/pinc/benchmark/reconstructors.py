"""Reconstructors of the compression benchmark.

Each turns one trajectory's ground-truth snapshots into reconstructed dfs ``(2, vp, mu, s, x, y)``
and the total compressed bytes: the traditional codecs (iso-CR per snapshot or a fixed knob),
the per-snapshot neural fields (torch or JAX checkpoints) and the autoencoders (JAX runs or torch
checkpoints, on the trajectory's training normalization).
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.pinc.benchmark.codecs import CODECS, encode_at_cr
from neugk_jax.pinc.neural_field import MLPNF, build_nf, from_torch_state, n_params, sample_field
from neugk_jax.pinc.nf_runner import snapshot_norm
from neugk_jax.training.checkpoint import load_model_only
from neugk_jax.utils import recombine_zf

# neural-field checkpoint prefixes: density warm-up (nf) and physics fine-tune (nf-pinc)
NF_PREFIX = {"nf": "best_mlp", "nf-pinc": "best_int_mlp"}


def discover(ckpt_dir: str, prefix: str) -> tuple[dict[str, dict[int, str]], Optional[int]]:
    """``{traj: {t: path}}`` of the ``<prefix>_<traj>_t<t>_x<cr>`` checkpoints and their CR.

    The checkpoints are torch ``.pt`` or ``NFRunner`` ``.eqx`` files.
    """
    rx = re.compile(re.escape(prefix) + r"_(iteration_\d+)_t(\d+)_x(\d+)\.(pt|eqx)$")
    weights, cr = defaultdict(dict), None
    for p in glob.glob(os.path.join(ckpt_dir, prefix + "_*")):
        m = rx.search(os.path.basename(p))
        if m:
            weights[m.group(1)][int(m.group(2))] = p
            cr = int(m.group(3))
    return dict(weights), cr


class Reconstructor:
    name: str

    def reconstruct(self, traj: str, timesteps: Sequence[int], gt: Sequence) -> tuple[list, int]:
        """Reconstructed dfs of ``gt`` (host arrays per timestep) and the total compressed bytes."""
        raise NotImplementedError


class Traditional(Reconstructor):
    """A codec at a fixed knob (``knob``) or searched per snapshot to ``target_cr``.

    The search is warm-started from the previous snapshot of the same trajectory.
    """

    def __init__(self, name: str, codec: str, *, target_cr: Optional[float] = None, knob=None):
        if (target_cr is None) == (knob is None):
            raise ValueError("give exactly one of target_cr and knob")
        self.name, self.codec, self.target_cr, self.knob = name, codec, target_cr, knob
        self._warm = None

    def reconstruct(self, traj, timesteps, gt):
        dfs, size, self._warm = [], 0, None
        for df in gt:
            if self.target_cr is not None:
                recon, nbytes, self._warm = encode_at_cr(
                    self.codec, df, self.target_cr, warm=self._warm
                )
            else:
                c = CODECS[self.codec]
                recon, _, nbytes = c.fn(df, **{c.knob: self.knob})
            dfs.append(recon)
            size += int(nbytes)
        return dfs, size


def load_torch_nf(path: str, grid_size: Sequence[int]) -> MLPNF:
    """A torch neural-field checkpoint (``{"state_dict", "cfg"}``) as a JAX :class:`MLPNF`."""
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    if cfg.get("name", "mlp") != "mlp" or cfg.get("embed_type", "discrete") != "discrete":
        raise NotImplementedError(f"{path}: only the mlp field with the discrete embedding")
    model = MLPNF(
        grid_size,
        key=jr.PRNGKey(0),
        dim=int(cfg.get("dim")),
        n_layers=int(cfg.get("n_layers")),
        skips=bool(cfg.get("skips", True)),
        act_fn=cfg.get("act_fn", "silu"),
    )
    state = {k.removeprefix("_orig_mod."): v.numpy() for k, v in blob["state_dict"].items()}
    return from_torch_state(model, state)


def load_nf(path: str, grid_size: Sequence[int]) -> MLPNF:
    """A torch ``.pt`` neural field, or an ``NFRunner`` ``.eqx`` next to its ``config.yaml``."""
    if path.endswith(".pt"):
        return load_torch_nf(path, grid_size)
    from omegaconf import OmegaConf

    mcfg = OmegaConf.load(Path(path).parent / "config.yaml").model
    return load_model_only(path, build_nf(mcfg, grid_size, key=jr.PRNGKey(0)))


@partial(jax.jit, static_argnums=1)
def _decode_nf(model, grid_size, df):
    scale, shift = snapshot_norm(df)
    return sample_field(model, grid_size) * scale + shift


class NeuralField(Reconstructor):
    """Per-snapshot neural fields ``weights[traj][t]``, denormalized with the snapshot's z-score."""

    def __init__(self, name: str, weights: dict[str, dict[int, str]]):
        self.name, self.weights = name, weights

    def reconstruct(self, traj, timesteps, gt):
        dfs, size = [], 0
        for t, df in zip(timesteps, gt):
            grid = tuple(int(n) for n in df.shape[1:])
            model = load_nf(self.weights[traj][int(t)], grid)
            dfs.append(_decode_nf(model, grid, jnp.asarray(df)))
            size += 4 * n_params(model)
        return dfs, size


@eqx.filter_jit
def _ae_forward(model, x, cond):
    def one(xi, ci):
        out = model(xi, *(() if ci is None else (ci,)), return_latent=True)
        return out["df"], out.get("vq_indices", out["latent"])

    return jax.vmap(one)(x, cond)


class Autoencoder(Reconstructor):
    """An AE / VQ-VAE over the trajectory snapshots, on its training normalization.

    ``dataset`` maps a trajectory name to a normalized one-trajectory dataset at frame offset 0;
    a latent value is stored in bfloat16, a VQ code index in 16 bits.
    """

    def __init__(self, name: str, model, dataset, cond_slots, *, batch: int = 6):
        self.name, self.model, self.dataset = name, model, dataset
        self.cond_slots, self.batch = cond_slots, batch

    def reconstruct(self, traj, timesteps, gt):
        ds = self.dataset(traj)
        dfs, size = [], 0
        for i in range(0, len(timesteps), self.batch):
            ts = [int(t) for t in timesteps[i : i + self.batch]]
            x = jnp.stack([jnp.asarray(ds.read_frame(0, t)["df"]) for t in ts])
            cond = np.stack([ds.conditioning(0, ds.metadata[0]["timesteps"][t]) for t in ts])
            cond = None if self.cond_slots is None else jnp.asarray(cond[:, self.cond_slots])
            pred, tokens = _ae_forward(self.model, x, cond)
            fids = jnp.zeros((len(ts),), jnp.int32)
            pred = recombine_zf(ds.norm.denormalize("df", pred, fids), axis=1)
            dfs += list(pred)
            size += 2 * int(tokens.size)
        return dfs, size


def build_autoencoder(run: str, *, path: str, traj: str, ckpt: Optional[str] = None):
    """``(model, dataset_fn, cond_slots)`` of an AE run directory, or of a torch ``.pth`` base.

    A run directory is rebuilt from its ``config.yaml`` (LoRA adapters attached for ``stage=peft``)
    and loaded from ``ckpt`` (default ``best.eqx``); a ``.pth`` is translated onto the
    ``pinc_revival`` AE. The data come from ``path`` (``traj`` is one of its trajectories), with
    the run's ``normalization_stats`` resolved under ``path``.
    """
    from omegaconf import OmegaConf

    from neugk_jax.models.build import ae_conditioning, build_ae_from_config, run_config
    from neugk_jax.training.checkpoint import load_checkpoint
    from neugk_jax.training.runner import conditioning_slots
    from neugk_jax.translate import attach_ae_lora, load_or_translate

    run_path = Path(run)
    if run_path.suffix == ".pth":
        from hydra import compose, initialize_config_dir

        configs = str(Path(__file__).resolve().parents[3] / "configs")
        with initialize_config_dir(config_dir=configs, version_base=None):
            cfg = compose("main", overrides=["experiment=pinc_revival", f"dataset.path={path}"])
    else:
        cfg = OmegaConf.load(run_path / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.path = path
    dcfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    dcfg.backend = "numpy"
    rc = run_config(cfg)
    ds0 = _ae_dataset(dcfg, traj, None)
    rc["dataset"]["resolution"] = [int(r) for r in ds0.resolution]
    rc["dataset"]["separate_zf"] = bool(ds0.separate_zf)
    model = build_ae_from_config(rc, key=jr.PRNGKey(0))
    if run_path.suffix == ".pth":
        model = load_or_translate(model, str(run_path), strict=True)
    else:
        if cfg.get("stage") == "peft":
            model = attach_ae_lora(model, cfg.model.peft.lora, key=jr.PRNGKey(1))
        model = load_checkpoint(run_path / (ckpt or "best.eqx"), model).model
    enc, dec = ae_conditioning(cfg.model)
    slots = conditioning_slots(ds0.conditions, set(enc) | set(dec))
    stats = ds0.stats
    return model, (lambda name: _ae_dataset(dcfg, name, stats)), slots


def _ae_dataset(dcfg, traj: str, stats: Optional[dict]):
    from neugk_jax.dataset.factory import build_dataset

    return build_dataset(
        dcfg,
        split="val",
        stats=stats,
        trajectories=[traj],
        cond_filters=None,
        offset=0,
        subsample=1,
    )
