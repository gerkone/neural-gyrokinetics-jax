"""CLI: JAX neural-field checkpoints -> torch ``{state_dict, cfg}`` ``.pt`` files for ``neugk.pinc.eval``.

Every ``<prefix>mlp_<traj>_t<t>_x<cr>.eqx`` of the checkpoint directory is written as the matching
``.pt``; ``best_`` (the density field) is also written without prefix, the torch final-density name.

    python scripts/export_nf_torch.py <jax_ckpt_dir> --out <torch_ckpt_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.random as jr
from omegaconf import OmegaConf

from neugk_jax.pinc.neural_field import build_nf, torch_state
from neugk_jax.training.checkpoint import load_model_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", help="jax neural-field checkpoint directory (with config.yaml)")
    p.add_argument("--out", required=True, help="torch checkpoint directory")
    p.add_argument("--resolution", type=int, nargs=5, default=(32, 8, 16, 85, 32))
    args = p.parse_args()

    import torch

    src, out = Path(args.ckpts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mcfg = OmegaConf.load(src / "config.yaml").model
    template = build_nf(mcfg, args.resolution, key=jr.PRNGKey(0))
    # torch load_nf reads the architecture from the saved cfg
    tcfg = OmegaConf.create(
        {k: mcfg[k] for k in ("name", "dim", "n_layers", "skips", "embed_type", "act_fn")}
    )
    files = sorted(src.glob("*.eqx"))
    for f in files:
        state = {
            k: torch.from_numpy(v) for k, v in torch_state(load_model_only(f, template)).items()
        }
        names = [f.stem] + (
            [f.stem[len("best_") :]]
            if f.stem.startswith("best_") and not f.stem.startswith("best_int_")
            else []
        )
        for name in names:
            torch.save({"state_dict": state, "cfg": tcfg}, out / f"{name}.pt")
    print(f"exported {len(files)} checkpoints to {out}")


if __name__ == "__main__":
    main()
