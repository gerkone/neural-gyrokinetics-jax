"""Hydra entry point for the JAX/Equinox port.

Mirrors the upstream torch ``main.py`` dispatch pattern: read the
Hydra config, build an output directory, log the resolved config, and
hand off to the workflow-appropriate runner. JAX's distributed setup
piggybacks on the same SLURM / torchrun env vars (see
``neugk_jax.training.ddp.init_distributed``), so there's no separate
launcher tier to thread through.

Usage::

    python main.py workflow=ae training.n_epochs=1
    python main.py workflow=diffusion ae_checkpoint=/path/to/ae_run_dir
    python main.py load_ckpt=true output_path=/path/to/run_dir   # resume in place
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict


def dispatch_runner(cfg: DictConfig) -> None:
    """Workflow → runner dispatch. Matches ``neugk.main.dispatch_runner``."""
    workflow = cfg.get("workflow", "ae")
    base = workflow.split("_")[0] if "_" in workflow else workflow
    if base in ("ae", "pinc"):  # accept the upstream label too
        from neugk_jax.autoencoders.runner import AERunner
        AERunner(cfg, output_path=cfg.output_path)()
    elif base == "diffusion":
        from neugk_jax.diffusion.runner import FlowMatchingRunner
        FlowMatchingRunner(cfg, output_path=cfg.output_path)()
    elif base == "gyroswin":
        from neugk_jax.gyroswin import GyroSwinRunner
        GyroSwinRunner(cfg, output_path=cfg.output_path)()
    else:
        raise NotImplementedError(f"unknown workflow: {workflow}")



def _drop_cli_overridden(cli: list[str], source: DictConfig, prefix: str = "") -> None:
    keys = {c.split("=")[0].lstrip("+~") for c in cli}
    for k in list(source.keys()):
        path = f"{prefix}.{k}" if prefix else str(k)
        if path in keys:
            del source[k]
        elif OmegaConf.is_dict(source[k]):
            _drop_cli_overridden(cli, source[k], path)


def resume_config(cfg: DictConfig) -> DictConfig:
    """Config for resuming ``cfg.output_path`` in place: its saved config, CLI overrides on top."""
    run = Path(cfg.output_path or "")
    if not (run / "ckp.eqx").exists():
        raise FileNotFoundError(f"load_ckpt=true but {run}/ckp.eqx does not exist")
    saved = OmegaConf.load(run / "config.yaml")
    cli = list(HydraConfig.get().overrides.task) if HydraConfig.initialized() else []
    _drop_cli_overridden(cli, saved)
    return OmegaConf.merge(cfg, saved)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rand_suffix = random.randint(0, 999)
    date_and_time = datetime.today().strftime("%Y%m%d_%H%M%S") + f"_{rand_suffix:03d}"

    if cfg.get("load_ckpt"):
        cfg = resume_config(cfg)
    elif cfg.get("output_path") is None:
        cfg.output_path = str(Path("outputs") / date_and_time)
    else:
        cfg.output_path = str(Path(cfg.output_path) / date_and_time)
    Path(cfg.output_path).mkdir(parents=True, exist_ok=True)

    # jax-trained models use the corrected swin residual; record it so rebuilds from config agree
    if cfg.get("model") is not None and cfg.model.get("legacy_swin_shortcut") is None:
        with open_dict(cfg):
            cfg.model.legacy_swin_shortcut = False
    OmegaConf.save(cfg, Path(cfg.output_path) / "config.yaml")
    print("#" * 88)
    print("Starting neugk-jax with configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 88)
    dispatch_runner(cfg)
    import jax
    if jax.distributed.is_initialized():
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
