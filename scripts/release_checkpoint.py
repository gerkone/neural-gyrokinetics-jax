"""CLI: copy a trained model into a release tree and record it in ``<root>/manifest.json``.

A JAX run gives ``jax/model.eqx`` (model leaves, epoch, val loss, meta; no optimizer state) and
its ``config.yaml``; a torch export (or torch run) directory gives ``torch/best.pth`` and its
``config.yaml``.

    python scripts/release_checkpoint.py --root <release> --name <name> [--run <jax run dir>]
        [--ckpt best.eqx] [--torch <dir with best.pth>] [--row "<table label>"] [--note "..."]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from datetime import date
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neugk_jax.training.checkpoint import write_bundle


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def release_jax(run: Path, ckpt: str, out: Path) -> dict:
    with open(run / ckpt, "rb") as f:
        bundle = pickle.load(f)
    leaves = bundle["model_leaves"]
    keep = ("model_leaves", "epoch", "loss", "meta")
    write_bundle(out / "model.eqx", {k: bundle[k] for k in keep if k in bundle})
    shutil.copyfile(run / "config.yaml", out / "config.yaml")
    cfg = OmegaConf.load(run / "config.yaml")
    log = cfg.get("logging") or {}
    n = sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(leaves))
    return {
        "source": str(run / ckpt),
        "epoch": int(bundle.get("epoch", -1)),
        "val_loss": float(bundle.get("loss", float("nan"))),
        "n_stored_values": n,
        "experiment": str(cfg.get("experiment_id") or ""),
        "stage": cfg.get("stage"),
        "base": str(cfg.get("ae_checkpoint")) if cfg.get("stage") == "peft" else None,
        "wandb_run": log.get("run_id"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="release root")
    p.add_argument("--name", required=True, help="release entry, e.g. autoencoders/pinc_ae_1005x")
    p.add_argument("--run", default=None, help="jax run directory")
    p.add_argument("--ckpt", default="best.eqx")
    p.add_argument("--torch", default=None, help="directory with best.pth + config.yaml")
    p.add_argument("--row", default=None, help="results table label")
    p.add_argument("--note", default=None)
    args = p.parse_args()
    if not (args.run or args.torch):
        p.error("nothing to release: pass --run and/or --torch")

    root, entry = Path(args.root), {}
    out = root / args.name
    if args.run:
        (out / "jax").mkdir(parents=True, exist_ok=True)
        entry["jax"] = release_jax(Path(args.run), args.ckpt, out / "jax")
    if args.torch:
        src = Path(args.torch)
        (out / "torch").mkdir(parents=True, exist_ok=True)
        for f in ("best.pth", "config.yaml"):
            shutil.copyfile(src / f, out / "torch" / f)
        entry["torch"] = {"source": str(src / "best.pth")}
    entry["files"] = {
        str(f.relative_to(out)): sha256(f) for f in sorted(out.rglob("*")) if f.is_file()
    }
    entry.update(row=args.row, note=args.note, released=date.today().isoformat())

    manifest_file = root / "manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    manifest[args.name] = entry
    manifest_file.write_text(json.dumps(dict(sorted(manifest.items())), indent=1))
    print(f"released {args.name}: {', '.join(entry['files'])}")


if __name__ == "__main__":
    main()
