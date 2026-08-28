"""Dataset preprocessing for the JAX port.

Currently a thin entrypoint with mode dispatch (similar to upstream
``neugk/dataset/preprocess.py``'s ``--metadata_only`` / ``--geometry_only``
flags). The full raw GKW → fp32 preprocessing path is still on the torch
side (it does the GKW IFFT, geometry parsing, flux verification against
the legacy integrator) and is left intentionally torch-bound for now —
all the moving parts live in upstream ``neugk/`` and reproducing them
brings in `load_geometry`, `K_files`, `parse_input_dat`, ... none of
which the JAX trainer needs at runtime.

What this file does cover today: take an existing fp32 ``.bin`` shard
and produce side-by-side quantized siblings (``.bf16.bin``, ``.fp16.bin``,
``.i8.bin``, ``.i4.bin``) used by the dataloader's
``prefer_dtype="…"`` graceful-fallback path.

Layout per file::

    fp16 / bf16:   raw 16-bit values, no header
    i8:            float32 scale (4 bytes) || raw int8 values
    i4:            float32 scale (4 bytes) || raw uint8 nibble-packed
                   (two int4 values per byte: low nibble = index 2k,
                    high nibble = index 2k+1)

Reads round-trip via ``neugk_jax.dataset.backend._read_quantized`` which
dequantizes back to fp32 transparently.

Usage::

    python -m neugk_jax.dataset.preprocess --mode=quantize \\
        --path /local00/bioinf/galletti/preprocessed_kvikio \\
        --trajs 'iteration_{0-299}_ifft_realpotens' \\
        --bits bf16 --num-workers 8

    # idempotent — skips files whose quantized sibling already exists

Linear-run conditioning fields (``--mode=linear``): each nonlinear trajectory
``iteration_N`` has a paired linear run ``iteration_N_Lin`` whose single ``FDS``
snapshot is the eigenmode of that operating point. This mode records the FDS path
in the trajectory metadata (and optionally materializes the transformed field as
``data/linear.bin``) so the dataloader can serve one linear field per trajectory::

    python -m neugk_jax.dataset.preprocess --mode=linear \\
        --path /local00/bioinf/galletti/preprocessed_kvikio \\
        --trajs 'iteration_{0-299}_ifft_realpotens' \\
        --raw-root /restricteddata/ukaea/gyrokinetics/raw \\
        --materialize --num-workers 8

Works from scratch or on an already-processed dataset — it only adds keys to the
existing metadata pickle.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from neugk_jax.dataset.backend import _meta_ext, load_meta, save_meta
from neugk_jax.dataset.linear import (
    default_linear_roots,
    load_linear_field,
    resolve_linear_dir,
)

# ---------- file naming -------------------------------------------------
_DTYPE_SUFFIX = {
    "fp16": ".fp16.bin",
    "bf16": ".bf16.bin",
    "i8":   ".i8.bin",
    "i4":   ".i4.bin",
}


def quantized_sibling(fp32_path: str, bits: str) -> str:
    """``foo.bin`` → ``foo.<dtype>.bin`` (the side-by-side quantized shard)."""
    if not fp32_path.endswith(".bin"):
        return fp32_path + _DTYPE_SUFFIX[bits]
    return fp32_path[:-4] + _DTYPE_SUFFIX[bits]


# ---------- quantize / dequantize core ---------------------------------
def quantize_array(arr_f32: np.ndarray, bits: str) -> tuple[np.ndarray, np.float32 | None]:
    """Quantize a flat fp32 array to ``bits`` precision.

    Returns ``(payload, scale)``. ``scale`` is ``None`` for IEEE 16-bit
    formats (the dtype itself encodes magnitude). For int8/int4 it's the
    per-tensor symmetric quantization scale (``max(|x|) / qmax``).
    """
    if bits == "fp16":
        return arr_f32.astype(np.float16), None
    if bits == "bf16":
        from ml_dtypes import bfloat16
        return arr_f32.astype(bfloat16), None
    if bits == "i8":
        qmax = 127
        mx = float(np.max(np.abs(arr_f32)))
        scale = np.float32(mx / qmax) if mx > 0 else np.float32(1.0)
        q = np.clip(np.round(arr_f32 / scale), -128, 127).astype(np.int8)
        return q, scale
    if bits == "i4":
        qmax = 7
        mx = float(np.max(np.abs(arr_f32)))
        scale = np.float32(mx / qmax) if mx > 0 else np.float32(1.0)
        q = np.clip(np.round(arr_f32 / scale), -8, 7).astype(np.int8)
        # nibble-pack: two int4 values per byte. low nibble = idx 2k, high = 2k+1.
        if q.size % 2:
            q = np.concatenate([q, np.zeros(1, dtype=np.int8)])
        lo = (q[0::2].astype(np.uint8)) & 0x0F
        hi = (q[1::2].astype(np.uint8)) & 0x0F
        packed = (hi << 4) | lo
        return packed.astype(np.uint8), scale
    raise ValueError(f"unknown bits={bits!r}; expected one of {list(_DTYPE_SUFFIX)}")


def dequantize_array(payload: np.ndarray, scale: np.float32 | None, bits: str, n_elems: int) -> np.ndarray:
    """Inverse of :func:`quantize_array` — returns fp32."""
    if bits == "fp16":
        return payload.astype(np.float32)
    if bits == "bf16":
        return payload.astype(np.float32)
    if bits == "i8":
        return payload.astype(np.float32) * float(scale)
    if bits == "i4":
        # unpack nibbles
        lo = payload & 0x0F
        hi = (payload >> 4) & 0x0F
        # sign-extend 4-bit two's complement
        lo = np.where(lo >= 8, lo.astype(np.int8) - 16, lo.astype(np.int8))
        hi = np.where(hi >= 8, hi.astype(np.int8) - 16, hi.astype(np.int8))
        out = np.empty(payload.size * 2, dtype=np.int8)
        out[0::2] = lo
        out[1::2] = hi
        out = out[:n_elems]  # drop the padding from quantize_array if any
        return out.astype(np.float32) * float(scale)
    raise ValueError(f"unknown bits={bits!r}")


# ---------- on-disk I/O ------------------------------------------------
def write_quantized(dst: str, payload: np.ndarray, scale: np.float32 | None) -> int:
    """Atomic-ish write of one quantized shard. Returns bytes written."""
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        if scale is not None:
            f.write(np.float32(scale).tobytes())
        f.write(payload.tobytes())
    n = os.path.getsize(tmp)
    os.replace(tmp, dst)
    return n


def read_quantized(path: str, bits: str, n_elems: int, *, return_raw: bool = False):
    """Read a quantized shard and (optionally) dequantize.

    ``return_raw=True`` returns ``(payload, scale)`` without dequantizing —
    used by the GPU-side bf16 / fp16 paths that prefer to keep the read in
    its original dtype until after dlpack hand-off to JAX.
    """
    with open(path, "rb") as f:
        scale = None
        if bits in ("i8", "i4"):
            scale = np.frombuffer(f.read(4), dtype=np.float32)[0]
        if bits == "fp16":
            payload = np.frombuffer(f.read(), dtype=np.float16)
        elif bits == "bf16":
            from ml_dtypes import bfloat16
            payload = np.frombuffer(f.read(), dtype=bfloat16)
        elif bits == "i8":
            payload = np.frombuffer(f.read(), dtype=np.int8)
        elif bits == "i4":
            payload = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"unknown bits={bits!r}")
    if return_raw:
        return payload, scale
    return dequantize_array(payload, scale, bits, n_elems)


# ---------- traversal --------------------------------------------------
def _resolve_trajs(root: str, spec) -> list[str]:
    """Expand a brace pattern (or accept an explicit list) → traj dir basenames."""
    if isinstance(spec, list) and len(spec) != 1:
        return list(spec)
    if isinstance(spec, list):
        spec = spec[0]
    m = re.match(r"^(.*?)\{([^}]+)\}(.*?)$", spec)
    if not m:
        return [spec]
    prefix, ranges_str, suffix = m.groups()
    nums = []
    for part in ranges_str.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            nums.extend(range(lo, hi + 1))
        else:
            nums.append(int(part))
    return [f"{prefix}{n}{suffix}" for n in nums]


def _src_bins(data_dir: str) -> list[str]:
    """List fp32 .bin sources (timestep + poten) inside ``traj/data``."""
    if not os.path.isdir(data_dir):
        return []
    out = []
    for name in os.listdir(data_dir):
        if not name.endswith(".bin"):
            continue
        if any(name.endswith(suf) for suf in _DTYPE_SUFFIX.values()):
            continue
        if not (name.startswith("timestep_") or name.startswith("poten_")):
            continue
        out.append(os.path.join(data_dir, name))
    return sorted(out)


def _quantize_file(src: str, bits: str, force: bool) -> tuple[str, int, str]:
    dst = quantized_sibling(src, bits)
    if os.path.exists(dst) and not force:
        return src, 0, "skip"
    try:
        arr = np.fromfile(src, dtype=np.float32)
        payload, scale = quantize_array(arr, bits)
        n = write_quantized(dst, payload, scale)
        return src, n, "written"
    except Exception as e:
        return src, 0, f"error: {e}"


def _process_traj(traj_dir: str, bits: str, force: bool) -> tuple[str, int, int, int]:
    files = _src_bins(os.path.join(traj_dir, "data"))
    n_written = n_skipped = bytes_written = 0
    for src in files:
        _, n, status = _quantize_file(src, bits, force=force)
        if status == "written":
            n_written += 1
            bytes_written += n
        elif status == "skip":
            n_skipped += 1
        else:
            print(f"  [{traj_dir}] {os.path.basename(src)}: {status}", file=sys.stderr)
    return traj_dir, n_written, n_skipped, bytes_written


def run_quantize(
    *, path: str, trajs: str | Sequence[str], bits: str,
    num_workers: int = 4, force: bool = False,
) -> None:
    traj_basenames = _resolve_trajs(path, trajs)
    traj_dirs = [os.path.join(path, n) for n in traj_basenames]
    traj_dirs = [d for d in traj_dirs if os.path.isdir(d)]
    if not traj_dirs:
        print(f"no trajectory dirs matched under {path}")
        sys.exit(1)
    print(f"quantizing {len(traj_dirs)} trajectories to {bits} from {path}")
    t0 = time.perf_counter()
    total_w = total_s = total_b = 0
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futures = {ex.submit(_process_traj, d, bits, force): d for d in traj_dirs}
        for i, fut in enumerate(as_completed(futures), 1):
            d, nw, ns, bw = fut.result()
            total_w += nw; total_s += ns; total_b += bw
            elapsed = time.perf_counter() - t0
            rate = total_b / max(elapsed, 1e-6) / 1e9
            print(
                f"  [{i}/{len(traj_dirs)}] {Path(d).name:<40}  "
                f"written={nw:4d}  skip={ns:4d}  bytes={bw / 1e9:6.2f} GB  "
                f"rate={rate:5.2f} GB/s",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    print(
        f"\ndone — {total_w} files written, {total_s} skipped, "
        f"{total_b / 1e9:.2f} GB in {elapsed:.0f}s "
        f"({total_b / max(elapsed, 1e-6) / 1e9:.2f} GB/s)"
    )



def linear_moments(arr: np.ndarray) -> dict:
    """Reduced second moments of one linear field, for the dataset-level profile.

    Full per-element moments would be ~90 MB per trajectory; these reductions are a few
    KB and are what the conditioning normalization needs. ``ky``/``kxky`` keep the
    trailing spectral axes so a shared profile can flatten the spectral decay without
    touching each trajectory's own deviation from it.
    """
    sq = np.square(arr, dtype=np.float64)
    return {
        "linear_rms": np.float64(np.sqrt(sq.mean())),
        "linear_rms_ky": np.sqrt(sq.mean(axis=(0, 1, 2, 3, 4))),          # (nky,)
        "linear_rms_cky": np.sqrt(sq.mean(axis=(1, 2, 3, 4))),            # (2, nky) re/im x ky
        "linear_rms_kxky": np.sqrt(sq.mean(axis=(0, 1, 2, 3))),           # (nkx, nky)
        "linear_rms_ckxky": np.sqrt(sq.mean(axis=(1, 2, 3))),             # (2, nkx, nky)
        "linear_rms_count": np.int64(arr.size),
    }


def _meta_bases(traj_dir: str, which: str) -> list[str]:
    bases = []
    if which in ("full", "both"):
        bases.append(os.path.join(traj_dir, "metadata"))
    if which in ("light", "both"):
        bases.append(os.path.join(traj_dir, "metadata_light"))
    return [b for b in bases if _meta_ext(b) is not None]


def _save_meta_atomic(base: str, meta: dict, ext: str) -> None:
    # a trajectory's full metadata is ~450 MB; never truncate the original in place
    save_meta(base + ".tmp", meta, ext)
    os.replace(base + ".tmp" + ext, base + ext)


def _link_one(
    traj_dir: str, roots: Sequence[str], *, materialize: bool, bits: str | None,
    meta_which: str, force: bool, dry_run: bool = False, space: str = "real",
    stats: bool = False,
) -> tuple[str, str]:
    bases = _meta_bases(traj_dir, meta_which)
    if not bases:
        return traj_dir, "no metadata"
    lin_dir = resolve_linear_dir(traj_dir, roots)
    if lin_dir is None:
        return traj_dir, "no _Lin pair"
    fds = os.path.join(lin_dir, "FDS")
    if dry_run:
        return traj_dir, f"would link -> {lin_dir}"

    # linear_space describes what is STORED: only the materialized bin is pre-transformed
    updates: dict = {"linear_fds": fds}
    if materialize:
        meta0 = load_meta(bases[0])
        resolution = tuple(int(x) for x in np.atleast_1d(meta0["resolution"]))
        shape = (2, *resolution)
        # "raw" keeps the semispectral field as GKW wrote it, no kx/ky inversion
        name = "linear.bin" if space == "real" else "linear_k.bin"
        dst = os.path.join(traj_dir, "data", name)
        if not os.path.exists(dst) or force:
            arr = load_linear_field(fds, resolution, to_real=(space == "real"))
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                tmp = dst + ".tmp"
                arr.tofile(tmp)
                os.replace(tmp, dst)
                if bits:
                    payload, scale = quantize_array(arr.ravel(), bits)
                    write_quantized(quantized_sibling(dst, bits), payload, scale)
            except OSError as e:
                # some trajectories are read-only symlinks into another user's tree
                return traj_dir, f"not writable ({e.strerror})"
        key = "linear_bin" if space == "real" else "linear_k_bin"
        updates[key] = f"data/{name}"
        if stats:
            arr = np.fromfile(dst, dtype=np.float32).reshape(shape)
            updates.update(linear_moments(arr))
        updates[f"{key}_shape" if space != "real" else "linear_shape"] = np.asarray(
            shape, dtype=np.int64)
        if space == "real":
            updates["linear_space"] = "real"

    if stats and not materialize:
        meta0 = load_meta(bases[0])
        resolution = tuple(int(x) for x in np.atleast_1d(meta0["resolution"]))
        rel = meta0.get("linear_k_bin" if space != "real" else "linear_bin")
        local = os.path.join(traj_dir, str(rel)) if rel is not None else None
        if local is not None and os.path.exists(local):
            arr = np.fromfile(local, dtype=np.float32).reshape((2, *resolution))
        else:
            arr = load_linear_field(fds, resolution, to_real=(space == "real"))
        updates.update(linear_moments(arr))

    for base in bases:
        meta = load_meta(base)
        if meta is None:
            continue
        if all(k in meta for k in updates) and not force:
            continue
        meta.update(updates)
        try:
            _save_meta_atomic(base, meta, _meta_ext(base))
        except OSError as e:
            # some trajectories are read-only symlinks into another user's tree
            return traj_dir, f"not writable ({e.strerror})"
    return traj_dir, "linked"


def run_link_linear(
    *, path: str, trajs: str | Sequence[str] | None, raw_root: str,
    linear_roots: Sequence[str] | None = None, materialize: bool = False,
    bits: str | None = None, meta: str = "both", num_workers: int = 4,
    force: bool = False, dry_run: bool = False, space: str = "real",
    stats: bool = False,
) -> None:
    """Record each trajectory's paired ``_Lin`` FDS file in its metadata.

    Idempotent and incremental: runs on an already-preprocessed dataset and only
    edits the metadata (plus, with ``materialize``, one extra ``data/linear.bin``
    holding the transformed field so training reads never touch the raw dir).
    """
    if trajs in (None, "all", ["all"]):
        traj_dirs = sorted(
            os.path.join(path, n) for n in os.listdir(path)
            if os.path.isdir(os.path.join(path, n))
        )
    else:
        traj_dirs = [os.path.join(path, n) for n in _resolve_trajs(path, trajs)]
        traj_dirs = [d for d in traj_dirs if os.path.isdir(d)]
    if not traj_dirs:
        print(f"no trajectory dirs matched under {path}")
        sys.exit(1)
    roots = list(linear_roots) if linear_roots else default_linear_roots(raw_root)
    print(f"linking linear FDS for {len(traj_dirs)} trajectories (roots: {roots})")
    t0 = time.perf_counter()
    counts: dict[str, int] = {}
    unmatched = []
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futures = [
            ex.submit(_link_one, d, roots, materialize=materialize, bits=bits,
                      meta_which=meta, force=force, dry_run=dry_run, space=space,
                      stats=stats)
            for d in traj_dirs
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            d, status = fut.result()
            # bucket by outcome, not by the resolved path (which is unique per trajectory)
            key = status.split(" ->")[0]
            counts[key] = counts.get(key, 0) + 1
            if key not in ("linked", "would link"):
                unmatched.append((Path(d).name, status))
            print(f"  [{i}/{len(traj_dirs)}] {Path(d).name:<45} {status}", flush=True)
    print(f"\ndone in {time.perf_counter() - t0:.0f}s — " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for name, status in unmatched[:20]:
        print(f"  unmatched: {name}  ({status})")

# ---------- CLI --------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode",
        choices=("quantize", "linear"),
        default="quantize",
        help="``quantize``: write side-by-side quantized shards. ``linear``: record "
             "each trajectory's paired ``_Lin`` FDS file in its metadata (see "
             "--materialize). The full GKW raw → fp32 preprocessing path stays on "
             "the torch side for now (``neugk/dataset/preprocess.py``).",
    )
    ap.add_argument("--path", default="/local00/bioinf/galletti/preprocessed_kvikio")
    ap.add_argument(
        "--trajs", nargs="+", default=["iteration_{0-299}_ifft_realpotens"],
        help="brace pattern (single string) OR explicit list of trajectory dirs",
    )
    ap.add_argument(
        "--bits", choices=tuple(_DTYPE_SUFFIX), default="bf16",
        help="quantization target (fp16 / bf16 / i8 / i4)",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing quantized shards / re-link metadata")
    # --mode=linear
    ap.add_argument("--raw-root", default="/restricteddata/ukaea/gyrokinetics/raw",
                    help="root holding the raw ``<traj>_Lin`` simulation dirs")
    ap.add_argument("--linear-roots", nargs="+", default=None,
                    help="explicit raw roots to search (overrides the defaults derived "
                         "from --raw-root)")
    ap.add_argument("--materialize", action="store_true",
                    help="also write ``data/linear.bin`` (the FDS field transformed to "
                         "the df layout) so training never reads the raw dir")
    ap.add_argument("--meta", choices=("both", "full", "light"), default="both",
                    help="which metadata file(s) to edit")
    ap.add_argument("--linear-bits", choices=tuple(_DTYPE_SUFFIX), default=None,
                    help="also write a quantized sibling of ``data/linear.bin``")
    ap.add_argument("--space", choices=("real", "raw"), default="real",
                    help="materialize the kx/ky-inverted field (real) or the semispectral "
                         "field as written by GKW (raw)")
    ap.add_argument("--stats", action="store_true",
                    help="also store the field's reduced second moments (global / per-ky / "
                         "per-(kx,ky)) in the metadata, for the shared normalization profile")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the resolved pairing without writing anything")
    args = ap.parse_args(argv)
    if args.mode == "quantize":
        run_quantize(
            path=args.path, trajs=args.trajs, bits=args.bits,
            num_workers=args.num_workers, force=args.force,
        )
    elif args.mode == "linear":
        run_link_linear(
            path=args.path, trajs=args.trajs, raw_root=args.raw_root,
            linear_roots=args.linear_roots, materialize=args.materialize,
            bits=args.linear_bits,
            meta=args.meta, num_workers=args.num_workers, force=args.force,
            dry_run=args.dry_run, space=args.space, stats=args.stats,
        )


if __name__ == "__main__":
    main()
