"""Write the lossless ``.zstd_bf16.bin`` sibling of every ``.bf16.bin`` shard under a dataset root.

With ``--dst`` the trajectories are mirrored into a new root (other files copied, shards
compressed), else the ``.zstd_bf16.bin`` files are written next to their ``.bf16.bin`` sources. Every
compressed shard is decoded and compared with its source before it is kept; existing outputs are
skipped.

Usage: pack_zstd.py <src_root> [--dst DIR] [--every K] [--workers N] [--level L]
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from neugk_jax.dataset import zframe
from neugk_jax.utils import atomic_write


def pack(task: tuple[str, str, int]) -> tuple[int, int]:
    src, dst, level = task
    if src.endswith(".bf16.bin"):
        if os.path.exists(dst):
            return 0, 0
        raw = np.fromfile(src, dtype=np.uint8)
        blob = zframe.encode(raw, level=level)
        if not np.array_equal(zframe.decode(blob), raw):
            raise RuntimeError(f"{src}: decoded shard differs from its source")
        atomic_write(dst, lambda f: f.write(blob))
        return raw.size, len(blob)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    return 0, 0


def tasks(src_root: Path, dst_root: Path, every: int, level: int):
    trajectories = sorted(p for p in src_root.iterdir() if p.is_dir())[::every]
    for traj in trajectories:
        for src in sorted(traj.rglob("*")):
            if src.is_dir():
                continue
            rel = src.relative_to(src_root)
            if rel.name.endswith(".bf16.bin"):
                rel = rel.with_name(rel.name.removesuffix(".bf16.bin") + ".zstd_bf16.bin")
            elif dst_root == src_root:
                continue
            (dst_root / rel).parent.mkdir(parents=True, exist_ok=True)
            yield str(src), str(dst_root / rel), level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_root", type=Path)
    ap.add_argument("--dst", type=Path, default=None)
    ap.add_argument("--every", type=int, default=1, help="pack every k-th trajectory")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--level", type=int, default=1)
    args = ap.parse_args()
    todo = list(tasks(args.src_root, args.dst or args.src_root, args.every, args.level))
    print(f"{len(todo)} files under {args.src_root}", flush=True)
    t0, raw_total, comp_total = time.time(), 0, 0
    with Pool(args.workers) as pool:
        for i, (raw, comp) in enumerate(pool.imap_unordered(pack, todo, chunksize=4), 1):
            raw_total, comp_total = raw_total + raw, comp_total + comp
            if i % 500 == 0 or i == len(todo):
                dt = time.time() - t0
                print(
                    f"{i}/{len(todo)} files, {raw_total / 1e9:.1f} GB packed at {raw_total / 1e9 / dt:.2f} GB/s, "
                    f"ratio {raw_total / max(comp_total, 1):.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
