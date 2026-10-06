"""CLI: a torch PINC-AE PEFT ``ckp.pth`` -> a JAX run directory that ``main.py load_ckpt=true`` resumes.

The JAX run is built from the ``pinc_revival`` experiment (plus the given Hydra overrides). The torch
adapters, their Adam moments (torch param groups: names without, then with an ``exclude_from_wd``
substring), the step count and epoch are copied into its ``ckp.eqx``. The best score starts empty:
the torch validation loss is not comparable (torch validates a per-rank shard of the set).

    python scripts/resume_pinc_torch.py <torch_run_dir> --out <jax_run_dir> ae_checkpoint=<base ae> [k=v ...]
    python main.py load_ckpt=true output_path=<jax_run_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from hydra import compose, initialize_config_dir


def torch_adam_moments(path: str, exclude: list[str]) -> tuple[dict, int, int, float]:
    """``{torch param name: (exp_avg, exp_avg_sq)}`` of the adapters, the step, epoch and loss."""
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    names = [k for k in blob["model_state_dict"] if ".lora_A." in k or ".lora_B." in k]
    no_wd = [n for n in names if any(e in n.lower() for e in exclude)]
    order = [n for n in names if n not in no_wd] + no_wd
    state = blob["optimizer_state_dict"]["state"]
    if len(state) != len(order):
        raise ValueError(f"{len(state)} optimizer states for {len(order)} adapter tensors")
    moments = {
        n: (state[i]["exp_avg"].numpy(), state[i]["exp_avg_sq"].numpy())
        for i, n in enumerate(order)
    }
    steps = {int(s["step"]) for s in state.values()}
    if len(steps) != 1:
        raise ValueError(f"adapter tensors at different steps {sorted(steps)}")
    return moments, steps.pop(), int(blob["epoch"]), float(blob["loss"])


def with_moments(opt_state, moments: dict, names: dict, step: int):
    """``opt_state`` with the Adam ``mu`` / ``nu`` of every adapter and every ``count`` at ``step``."""
    from neugk_jax.models.lora import get_path

    def fill(tree, which):
        for path, (ka, kb) in names.items():
            node = get_path(tree, path)
            for attr, key in (("lora_A", ka), ("lora_B", kb)):
                ref = getattr(node, attr)
                arr = jnp.asarray(moments[key][which], ref.dtype).reshape(ref.shape)
                tree = eqx.tree_at(lambda t, p=path, a=attr: getattr(get_path(t, p), a), tree, arr)
        return tree

    def visit(s):
        if isinstance(s, optax.ScaleByAdamState):
            return s._replace(
                count=jnp.asarray(step, s.count.dtype), mu=fill(s.mu, 0), nu=fill(s.nu, 1)
            )
        if "count" in getattr(s, "_fields", ()):
            return s._replace(count=jnp.asarray(step, s.count.dtype))
        return s

    return jax.tree_util.tree_map(visit, opt_state, is_leaf=lambda s: hasattr(s, "_fields"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("torch_run", help="torch pinc peft run directory (ckp.pth)")
    p.add_argument("--out", required=True, help="new jax run directory")
    p.add_argument("--ckpt", default="ckp.pth")
    # the remaining arguments are hydra overrides of the pinc_revival experiment
    args, overrides = p.parse_known_args()

    from neugk_jax.pinc.peft import PINCPEFTRunner
    from neugk_jax.training.ddp import local_view, replicate
    from neugk_jax.translate import import_ae_lora, load_torch_state

    configs = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=configs, version_base=None):
        cfg = compose("main", overrides=["experiment=pinc_revival", *overrides])
    cfg.output_path = str(Path(args.out).resolve())
    runner = PINCPEFTRunner(cfg, output_path=cfg.output_path)

    ckpt = str(Path(args.torch_run) / args.ckpt)
    model, names = import_ae_lora(local_view(runner.dist, runner.model), load_torch_state(ckpt))
    exclude = list(cfg.training.get("exclude_from_wd") or [])
    moments, step, epoch, loss = torch_adam_moments(ckpt, exclude)
    # torch keeps the partial last batch; the jax schedule resumes at the same epoch
    jax_step = epoch * runner.steps_per_epoch
    opt_state = with_moments(local_view(runner.dist, runner.opt_state), moments, names, jax_step)
    runner.model, runner.opt_state = (
        replicate(runner.dist, model),
        replicate(runner.dist, opt_state),
    )
    runner.save_checkpoint(epoch, loss, "ckp.eqx")
    runner.checkpointer.join()
    print(
        f"{len(names)} adapters at epoch {epoch} (torch step {step}, jax step {jax_step}, loss {loss:.4g})"
    )
    print(f"resume: python main.py load_ckpt=true output_path={cfg.output_path}")


if __name__ == "__main__":
    main()
