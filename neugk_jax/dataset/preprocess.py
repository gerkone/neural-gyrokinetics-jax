"""Dataset preprocessing: raw simulations to the kvikio layout, potential rewrites, quantization.

Modes (``python -m neugk_jax.dataset.preprocess --mode=<mode>``):

* ``preprocess``: raw GKW runs (K-files, Poten/Spc3d dumps, input.dat, geom.dat) to
  ``<target>/preprocessed_kvikio/<traj>_ifft_realpotens/`` with ``data/timestep_XXXXX.bin``
  (fp32 ``(2, vpar, mu, s, x, y)``), ``data/poten_XXXXX.bin`` (fp32 real ``(x, s, y)``),
  ``metadata.pkl`` and ``metadata_light.pkl``.
* ``rewrite-phi``: overwrite ``poten_*.bin`` of preprocessed trajectories with the field solve
  of their df (backups, resumable ``DONE`` markers, recomputed phi statistics).
* ``gyaradax``: gyaradax run folders (``step_*.npz`` + ``config.yaml`` + ``geometry.pkl``)
  to the same layout, with heat-flux verification.
* ``quantize``: side-by-side quantized siblings of the fp32 shards (``.bf16.bin``,
  ``.fp16.bin``, ``.i8.bin``, ``.i4.bin``) read by the dataloader's ``prefer_dtype`` path.

Quantized layout per file::

    fp16 / bf16:   raw 16-bit values, no header
    i8:            float32 scale (4 bytes) || raw int8 values
    i4:            float32 scale (4 bytes) || raw uint8 nibble-packed
                   (two int4 values per byte: low nibble = index 2k,
                    high nibble = index 2k+1)

Usage::

    python -m neugk_jax.dataset.preprocess --mode=preprocess --trajs 'iteration_{0-9}' \\
        --target-dir /local00/bioinf/galletti
    python -m neugk_jax.dataset.preprocess --mode=rewrite-phi \\
        --path /local00/bioinf/galletti/preprocessed_kvikio --poten-backup /path/to/backup
    python -m neugk_jax.dataset.preprocess --mode=gyaradax --gyaradax-dirs /path/to/run
    python -m neugk_jax.dataset.preprocess --mode=quantize \\
        --path /local00/bioinf/galletti/preprocessed_kvikio \\
        --trajs 'iteration_{0-299}_ifft_realpotens' --bits bf16 --num-workers 8
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

RAW_ROOT = "/restricteddata/ukaea/gyrokinetics"
TARGET_DIR = "/local00/bioinf/galletti"
KVIKIO_SUBDIR = "preprocessed_kvikio"
LIGHT_DROP_KEYS = ("df_min", "df_max", "df_var", "df_mean", "df_std", "phi_min", "phi_max",
                   "phi_var")
PHI_STAT_KEYS = ("phi_mean", "phi_var", "phi_std", "phi_min", "phi_max")

_DTYPE_SUFFIX = {
    "fp16": ".fp16.bin",
    "bf16": ".bf16.bin",
    "i8":   ".i8.bin",
    "i4":   ".i4.bin",
}


def quantized_sibling(fp32_path: str, bits: str) -> str:
    # foo.bin -> foo.<dtype>.bin
    if not fp32_path.endswith(".bin"):
        return fp32_path + _DTYPE_SUFFIX[bits]
    return fp32_path[:-4] + _DTYPE_SUFFIX[bits]


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
        # nibble-pack: two int4 values per byte, low nibble = idx 2k, high = 2k+1
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
        lo = payload & 0x0F
        hi = (payload >> 4) & 0x0F
        # sign-extend 4-bit two's complement
        lo = np.where(lo >= 8, lo.astype(np.int8) - 16, lo.astype(np.int8))
        hi = np.where(hi >= 8, hi.astype(np.int8) - 16, hi.astype(np.int8))
        out = np.empty(payload.size * 2, dtype=np.int8)
        out[0::2] = lo
        out[1::2] = hi
        out = out[:n_elems]
        return out.astype(np.float32) * float(scale)
    raise ValueError(f"unknown bits={bits!r}")


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

    ``return_raw=True`` returns ``(payload, scale)`` without dequantizing, keeping
    the read in its on-disk dtype.
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


def expand_spec(spec) -> list[str]:
    """Expand one brace pattern (``iteration_{0-3,7}``) or pass an explicit list through."""
    if isinstance(spec, (list, tuple)) and len(spec) != 1:
        return list(spec)
    s = spec[0] if isinstance(spec, (list, tuple)) else spec
    m = re.match(r"^(.*?)\{([^}]+)\}(.*?)$", s)
    if not m:
        return [s]
    prefix, ranges_str, suffix = m.groups()
    nums = []
    for part in ranges_str.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            nums.extend(range(lo, hi + 1))
        else:
            nums.append(int(part))
    return [f"{prefix}{n}{suffix}" for n in nums]


def resolve_traj_dirs(root_dir: str, spec=None) -> list[str]:
    """Trajectory dirs under ``root_dir`` from basenames or one brace pattern.

    ``spec=None`` selects every ``*_ifft_realpotens`` directory.
    """
    if spec is None:
        return sorted(
            os.path.join(root_dir, n)
            for n in os.listdir(root_dir)
            if n.endswith("_ifft_realpotens") and os.path.isdir(os.path.join(root_dir, n))
        )
    return [os.path.join(root_dir, n) for n in expand_spec(spec)]


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
    arr = np.fromfile(src, dtype=np.float32)
    payload, scale = quantize_array(arr, bits)
    return src, write_quantized(dst, payload, scale), "written"


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
    traj_dirs = [d for d in resolve_traj_dirs(path, trajs) if os.path.isdir(d)]
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
            total_w += nw
            total_s += ns
            total_b += bw
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


def parse_input_dat(file_path: str) -> dict:
    """Parse a GKW ``input.dat`` namelist into ``{section: {param: value}}``."""
    parsed = {}
    with open(file_path, "r") as file:
        content = file.read()
    sections = re.split(r"&\w+", content)
    headers = re.findall(r"&(\w+)", content)
    sections = [s.strip() for s in sections if len(s) and s[0] != "!" and s.strip()]
    for header, section in zip(headers, sections):
        section_dict = {}
        for param, value in re.findall(r"(\w+)\s*=\s*([-\d\.e\w]+)", section):
            try:
                section_dict[param] = float(value) if "e" in value or "." in value else int(value)
            except ValueError:
                section_dict[param] = value.strip()
        # repeated sections (species) get a trailing 0 per repeat
        while header in parsed:
            header = f"{header}0"
        parsed[header] = section_dict
    return parsed


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_geom_dat(file_path: str) -> dict:
    """Parse a GKW ``geom.dat`` into scalars and float64 arrays keyed by block name."""
    data, key, values = {}, None, []
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) == 1 and not _is_number(parts[0]):
            if key is not None:
                data[key] = np.array(values, dtype=np.float64)
            key, values = parts[0], []
        else:
            values.extend(map(float, parts))
    if key is not None:
        data[key] = np.array(values, dtype=np.float64)
    return data


def _gkw_bool(val) -> float:
    if isinstance(val, str):
        v = val.lower().strip()
        if v == ".true.":
            return 1.0
        if v == ".false.":
            return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _default_mu_grid(n: int = 8, mumax: float = 4.5) -> tuple[np.ndarray, np.ndarray]:
    dvperp = np.sqrt(2.0 * mumax) / n
    vperp = (np.arange(n + 1) - 0.5) * dvperp
    mugr = vperp**2 / 2.0
    intmu = np.abs(np.pi * ((vperp + 0.5 * dvperp) ** 2 - (vperp - 0.5 * dvperp) ** 2))
    return mugr[1:], intmu[1:]


def load_geometry(directory: str) -> dict[str, np.ndarray]:
    """Flux-integral geometry of one GKW run as float64 numpy arrays."""
    f64 = np.float64
    geom = load_geom_dat(os.path.join(directory, "geom.dat"))
    inp = parse_input_dat(os.path.join(directory, "input.dat"))
    g = {k: np.array(1.0, dtype=f64) for k in ("signz", "vthrat", "tmp", "mas", "d2X", "signB")}
    control = inp.get("control", {})
    g["nlapar"] = np.array(_gkw_bool(control.get("nlapar", 0.0)), dtype=f64)
    g["nlbpar"] = np.array(_gkw_bool(control.get("nlbpar", 0.0)), dtype=f64)
    g["beta"] = np.array(float(inp.get("parameters", {}).get("beta", 0.0)), dtype=f64)

    num_sp = 1
    for sec in inp.values():
        if "number_of_species" in sec:
            num_sp = int(sec["number_of_species"])
            break
    species = [inp[k] for k in inp if k.startswith("species")][:num_sp]
    if not species:
        species = [{}]
    for key, name in (("mas", "mass"), ("tmp", "temp"), ("de", "dens"), ("signz", "z")):
        g[key] = np.array([sp.get(name, 1.0) for sp in species], dtype=f64)
    g["vthrat"] = np.sqrt(g["tmp"] / g["mas"])
    g["adiabatic"] = np.array(1.0, dtype=f64)

    kxrh = np.loadtxt(os.path.join(directory, "kxrh"))[0]
    krho = np.loadtxt(os.path.join(directory, "krho")).T[0] / geom["kthnorm"]
    g["kxrh"] = np.asarray(kxrh, dtype=f64)
    g["krho"] = np.asarray(krho, dtype=f64)
    g["parseval"] = np.array([1.0] + [float(len(krho))] * (len(krho) - 1), dtype=f64)

    mugr_d, intmu_d = _default_mu_grid()
    if os.path.exists(os.path.join(directory, "intmu.dat")):
        intmu = np.loadtxt(os.path.join(directory, "intmu.dat"))
        intmu = intmu[:, 0] if intmu.ndim == 2 else intmu
    else:
        intmu = intmu_d
    g["intmu"] = np.asarray(intmu, dtype=f64)
    if os.path.exists(os.path.join(directory, "vperp.dat")):
        vperp = np.loadtxt(os.path.join(directory, "vperp.dat"))
        vperp = vperp[:, 0] if vperp.ndim == 2 else vperp
        mugr = vperp**2 / 2.0
    else:
        mugr = mugr_d
    g["mugr"] = np.asarray(mugr, dtype=f64)
    g["intvp"] = np.asarray(np.loadtxt(os.path.join(directory, "intvp.dat"))[0], dtype=f64)
    g["vpgr"] = np.asarray(np.loadtxt(os.path.join(directory, "vpgr.dat"))[0], dtype=f64)

    sgrid = np.loadtxt(os.path.join(directory, "sgrid"))
    ints = np.concatenate([np.array([0.0]), np.diff(sgrid)])
    ints[0] = ints[1]
    g["ints"] = ints.astype(f64)
    g["efun"] = np.asarray(-geom["E_eps_zeta"], dtype=f64)
    g["little_g"] = np.stack([geom["g_zeta_zeta"], geom["g_eps_zeta"], geom["g_eps_eps"]], -1)
    g["bn"] = np.asarray(geom["bn"], dtype=f64)
    g["bt_frac"] = np.asarray(geom["Bt_frac"], dtype=f64)
    g["rfun"] = np.asarray(geom["R"], dtype=f64)
    if len(g["de"]) > 1:
        g["adiabatic"] = np.array(0.0, dtype=f64)
    else:
        g["adiabatic"] = np.array(np.squeeze(geom.get("adiabatic", 1.0)), dtype=f64)
    for k in ("mas", "tmp", "de", "signz", "vthrat"):
        if k in geom:
            g[k] = np.asarray(geom[k], dtype=f64)
    return g


def k_files(directory: str) -> list[str]:
    # k* dumps sorted, then numeric names by value
    files = os.listdir(directory)
    digits = sorted((f for f in files if f.isdigit()), key=int)
    ks = sorted(f for f in files if f.startswith("K") and not f.endswith(".dat"))
    return ks + digits


def poten_files(directory: str) -> tuple[list[str], np.ndarray]:
    pots = sorted(f for f in os.listdir(directory) if f.startswith("Poten"))
    return pots, np.array([int(f.replace("Poten", "")) for f in pots]) - 1


def read_dump_time(dat_path: str) -> float:
    with open(dat_path, "r") as file:
        for line in file:
            parts = line.split("=")
            if parts[0].strip() == "TIME":
                return float(parts[1].strip().strip(",").strip())
    raise ValueError(f"no TIME entry in {dat_path}")


def load_k_dump(path: str, resolution: tuple) -> np.ndarray:
    # fp32 (2, vpar, mu, s, kx, ky), kx zero-centred
    ff = np.fromfile(path, dtype=np.float64)
    return np.reshape(ff, (2, *resolution), order="F").astype("float32").copy()


def do_ifft(knth: np.ndarray) -> np.ndarray:
    knth = np.fft.ifftn(knth, axes=(3, 4), norm="forward")
    return np.stack([knth.real, knth.imag]).squeeze().astype("float32")


def check_ifft(transformed: np.ndarray, orig: np.ndarray, zf_separated: bool = False,
               atol: float = 1e-5) -> bool:
    """True when the real-space df transforms back onto the raw spectral dump within ``atol``."""
    if zf_separated:
        cplx = np.sum(transformed[::2], axis=0) + 1j * np.sum(transformed[1::2], axis=0)
    else:
        cplx = transformed[0] + 1j * transformed[1]
    spec = np.fft.fftn(cplx.astype(np.complex64), axes=(3, 4), norm="forward")
    spec = np.fft.ifftshift(spec, axes=(3,))
    err_re = np.max(np.abs(spec.real.astype(np.float32) - orig[0]))
    err_im = np.max(np.abs(spec.imag.astype(np.float32) - orig[1]))
    return bool(max(err_re, err_im) <= atol)


def _check_spc(abs_phi_fft: np.ndarray, spc: np.ndarray) -> bool:
    return np.allclose(abs_phi_fft, spc, rtol=0.0, atol=1e-3)


def phi_to_spc(phi, gt_spc=None, out_shape=None, norm: str = "forward") -> np.ndarray:
    """Real GKW potential ``(nx, s, ny)`` to its kx-centred one-sided spectrum ``(nkx, s, nky)``.

    Asserts ``|spectrum|`` against the ``Spc3d`` dump when ``gt_spc`` is given.
    """
    phi_fft = np.fft.fftn(phi, axes=(0, 2), norm=norm)
    phi_fft = np.fft.fftshift(phi_fft, axes=(0, 2))
    phi_fft = phi_fft[..., phi_fft.shape[-1] // 2:]
    nkx, _, nky = out_shape
    xpad = (phi_fft.shape[0] - nkx) // 2
    xpad = xpad + 1 if (phi_fft.shape[0] % 2 == 0) else xpad
    phi_fft = phi_fft[xpad:nkx + xpad, :, :nky]
    if gt_spc is not None:
        assert _check_spc(np.abs(phi_fft), gt_spc), "Spectral space of Phi incorrect"
    return phi_fft


def phi_fft_to_real(fft: np.ndarray, out_shape, norm: str = "forward") -> np.ndarray:
    """Kx-centred one-sided spectrum (zero-padded to ``out_shape``) to a real potential."""
    if fft.shape != tuple(out_shape):
        nkx, _, nky = out_shape
        nx, _, ny = fft.shape
        xpad = (nkx - nx) // 2 + 1
        padded = np.zeros(out_shape).astype(fft.dtype)
        padded[xpad:xpad + nx, :, :ny] = fft
    else:
        nkx, _, nky = fft.shape
        padded = fft
    # ifftshift (not fftshift) inverts the centring for odd nkx
    phi = np.fft.ifftshift(padded, axes=(0,))
    return np.fft.irfftn(phi, axes=(0, 2), norm=norm, s=[nkx, nky])


def solver_df_to_realspace(df_spec: np.ndarray) -> np.ndarray:
    # inverse of the solver-side fftshift + spatial fft
    un = np.fft.fftshift(df_spec, axes=(3,))
    phys = np.fft.ifftn(un, axes=(3, 4), norm="forward")
    return np.stack([phys.real, phys.imag]).astype("float32")


def realspace_to_solver_df(df: np.ndarray) -> np.ndarray:
    spec = np.fft.fftn(df[0] + 1j * df[1], axes=(3, 4), norm="forward")
    return np.fft.ifftshift(spec, axes=(3,)).astype(np.complex64)


def _numeric_geometry(geometry: dict) -> dict:
    return {k: np.asarray(v) for k, v in geometry.items()
            if np.asarray(v).dtype.kind in "fiub"}


def _eflux_ky(g: dict, spec, phi, apar, bpar):
    import jax.numpy as jnp

    chi = (g["bessel"] * phi - 2.0 * g["vthrat"] * g["vpgr"] * g["bessel"] * apar
           + 2.0 * g["mugr"] * g["tmp"] / g["signz"] * g["bessel_bpar"] * bpar)
    dum1 = jnp.imag(g["parseval"] * g["ints"] * g["efun"] * g["krho"] * spec * jnp.conj(chi))
    d3v = g["ints"] * g["d2X"] * g["intmu"] * g["bn"] * g["intvp"]
    ef = d3v * (g["vpgr"] ** 2 * dum1 + 2.0 * g["mugr"] * g["bn"] * dum1) * g["de"] * g["tmp"]
    return jnp.sum(ef, axis=(0, 1, 2, 3))


def _flux_spectrum(g: dict, df):
    from neugk_jax.evaluate.integrals import _df_fft, _solve_fields

    spec = _df_fft(df)
    phi, apar, bpar = _solve_fields(g, spec)
    return _eflux_ky(g, spec, phi, apar, bpar)


class FieldSolver:
    """Jitted field solve, heat flux and per-ky heat-flux spectrum of real-space dfs.

    Built once per trajectory geometry; ``x64`` runs the solve in float64 (potentials are
    returned as fp32 ``(x, s, y)``).
    """

    _jits: dict = {}

    def __init__(self, geometry: dict, x64: bool = True):
        import jax
        import jax.numpy as jnp

        from neugk_jax.evaluate.integrals import flux_integral, precompute_geometry

        self.x64 = x64
        self.dtype = np.float64 if x64 else np.float32
        if not FieldSolver._jits:
            FieldSolver._jits["solve"] = jax.jit(flux_integral)
            FieldSolver._jits["spectrum"] = jax.jit(_flux_spectrum)
        gt = precompute_geometry(_numeric_geometry(geometry), dtype=self.dtype)
        with jax.enable_x64(x64):
            self.geom = {k: jnp.asarray(v) for k, v in gt.items()}

    def __call__(self, df: np.ndarray) -> tuple[np.ndarray, float]:
        import jax
        import jax.numpy as jnp

        with jax.enable_x64(self.x64):
            phi, (_, eflux, _) = self._jits["solve"](self.geom, jnp.asarray(df, self.dtype))
            return np.asarray(phi, dtype=np.float32), float(eflux)

    def flux_spectrum(self, df: np.ndarray) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        with jax.enable_x64(self.x64):
            out = self._jits["spectrum"](self.geom, jnp.asarray(df, self.dtype))
            return np.asarray(out, dtype=np.float64)


def field_solve_phi(df: np.ndarray, geometry: dict, x64: bool = True) -> np.ndarray:
    return FieldSolver(geometry, x64=x64)(df)[0]


class StreamStats:
    """Elementwise running mean/var/min/max over single samples, updated in place in float64.

    Seeded with a prior of count ``1e-4`` and unit variance, the convention of the stored
    dataset statistics.
    """

    def __init__(self, prior_count: float = 1e-4):
        self.count = prior_count
        self.mean = self.var = self.min = self.max = None
        self._d = None

    def update(self, x) -> None:
        x = np.asarray(x)
        if self.mean is None:
            shape = x.shape
            self.mean, self.var = np.zeros(shape), np.ones(shape)
            self.min, self.max = np.full(shape, np.inf), np.full(shape, -np.inf)
            self._d = np.empty(shape)
        c, n1 = self.count, self.count + 1.0
        d = self._d
        np.subtract(x, self.mean, out=d)
        self.mean += d * (1.0 / n1)
        self.var *= c / n1
        np.multiply(d, d, out=d)
        d *= c / (n1 * n1)
        self.var += d
        np.minimum(self.min, x, out=self.min)
        np.maximum(self.max, x, out=self.max)
        self.count = n1


def _running_stats():
    return StreamStats()


def _stats_dict(prefix: str, stats, dtype=None) -> dict:
    cast = (lambda x: np.asarray(x, dtype=dtype)) if dtype is not None else np.asarray
    return {
        f"{prefix}_mean": cast(stats.mean),
        f"{prefix}_var": cast(stats.var),
        f"{prefix}_std": cast(np.sqrt(stats.var)),
        f"{prefix}_min": cast(stats.min),
        f"{prefix}_max": cast(stats.max),
    }


def _meta_path(base: str) -> Optional[str]:
    for ext in (".npz", ".pkl"):
        if os.path.exists(base + ext):
            return base + ext
    return None


def _save_meta_atomic(base: str, meta: dict, ext: str = ".pkl") -> None:
    from neugk_jax.dataset.backend import save_meta

    save_meta(base + ".tmp", meta, ext)
    os.replace(base + ".tmp" + ext, base + ext)


def write_metadata(traj_dir: str, metadata: dict) -> None:
    _save_meta_atomic(os.path.join(traj_dir, "metadata"), metadata)
    light = {k: v for k, v in metadata.items() if k not in LIGHT_DROP_KEYS}
    _save_meta_atomic(os.path.join(traj_dir, "metadata_light"), light)


def _write_bin(path: str, arr: np.ndarray, overwrite: bool = False) -> None:
    if overwrite or not os.path.exists(path):
        np.ascontiguousarray(arr).tofile(path)


def _merge_old_metadata(traj_dir: str, metadata: dict) -> dict:
    from neugk_jax.dataset.backend import load_meta

    old = load_meta(os.path.join(traj_dir, "metadata"))
    if old is None:
        return metadata
    return {**metadata, **{k: v for k, v in old.items() if k not in metadata}}


def _progress(it, show: bool, **kwargs):
    if not show:
        return it
    from tqdm import tqdm

    return tqdm(it, **kwargs)


def preprocess(
    filename: str,
    backend=None,
    spatial_ifft: bool = True,
    separate_zf: bool = False,
    split_into_bands: Optional[int] = None,
    root: str = RAW_ROOT,
    raw_subdir: str = "raw",
    target_dir: Optional[str] = TARGET_DIR,
    metadata_only: bool = False,
    geometry_only: bool = False,
    phi_source: str = "field_solve",
    max_timesteps: Optional[int] = None,
    x64: bool = True,
    show_tqdm: bool = False,
    position: int = 0,
) -> tuple[str, bool]:
    """Convert one raw GKW run into the kvikio layout. Returns ``(out_path, skipped)``.

    Every dump is checked: the real-space df transforms back onto the K-file, the potential
    spectrum matches ``Spc3d``, the heat flux of the df matches ``fluxes.dat`` and the GKW
    potential matches the field solve of the df. ``phi_source`` picks the stored potential:
    ``"field_solve"`` (the field solve of the stored df) or ``"gkw"`` (the ``Poten`` dump).
    ``max_timesteps`` truncates the trajectory (data, series and statistics consistently).
    """
    from neugk_jax.dataset.backend import NumpyBackend

    assert spatial_ifft, "only the real-space (ifft) layout is supported"
    assert phi_source in ("field_solve", "gkw"), phi_source
    if "Lin" in filename:
        raise ValueError(f"{filename}: linear runs are not converted by preprocess")
    backend = backend or NumpyBackend()
    target_dir = root if target_dir is None else target_dir
    dir_in = f"{root}/{raw_subdir}/{filename}"
    dir_out = os.path.join(target_dir, KVIKIO_SUBDIR)
    os.makedirs(dir_out, exist_ok=True)
    out_path = backend.format_path(
        os.path.join(dir_out, filename.replace("/", "_")), spatial_ifft, split_into_bands,
        real_potens=True,
    )
    if backend.exists(out_path) and not (metadata_only or geometry_only):
        return out_path, True

    ks = k_files(dir_in)
    potens, _ = poten_files(dir_in)
    if not ks:
        # dump names follow the same sampling as iteration_0
        ref = f"{root}/{raw_subdir}/iteration_0"
        ks = k_files(ref)
        potens, _ = poten_files(ref)
    if max_timesteps is not None:
        ks, potens = ks[:max_timesteps], potens[:max_timesteps]
    timesteps = np.array([read_dump_time(f"{dir_in}/{k}.dat") for k in ks])

    sgrid = np.loadtxt(f"{dir_in}/sgrid")
    xphi = np.loadtxt(f"{dir_in}/xphi")
    krho = np.loadtxt(f"{dir_in}/krho")
    vpgr = np.loadtxt(f"{dir_in}/vpgr.dat")
    ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]
    nx, ny = xphi.shape[1], xphi.shape[0]
    nkx, nky = krho.shape[1], krho.shape[0]
    nvpar, nmu = vpgr.shape[1], vpgr.shape[0]
    resolution = (nvpar, nmu, ns, nkx, nky)

    fluxes = np.loadtxt(f"{dir_in}/fluxes.dat")[:, 1]
    orig_times = np.loadtxt(f"{dir_in}/time.dat")
    ts_slices = [np.isclose(orig_times, t).nonzero()[0][0] for t in timesteps]
    orig_fluxes = fluxes[ts_slices].copy()
    fluxes = np.clip(orig_fluxes, a_min=0.0, a_max=None)

    config = parse_input_dat(f"{dir_in}/input.dat")
    geometry = load_geometry(dir_in)
    metadata = {
        "timesteps": timesteps,
        "resolution": resolution,
        "ds": float(np.ravel(sgrid)[1] - np.ravel(sgrid)[0]),
        "ion_temp_grad": np.array([config["species"]["rlt"]]),
        "density_grad": np.array([config["species"]["rln"]]),
        "flux": fluxes,
        "s_hat": np.array([config["geom"]["shat"]]),
        "q": np.array([config["geom"]["q"]]),
        "geometry": geometry,
        "kyspec": np.loadtxt(f"{dir_in}/kyspec")[ts_slices],
        "fluxspec": np.loadtxt(f"{dir_in}/eflux_spectra.dat")[ts_slices],
    }

    if geometry_only:
        os.makedirs(os.path.join(out_path, "data"), exist_ok=True)
        write_metadata(out_path, _merge_old_metadata(out_path, metadata))
        return out_path, False

    solver = FieldSolver(geometry, x64=x64)
    df_stats, phi_stats, flux_stats = _running_stats(), _running_stats(), _running_stats()
    os.makedirs(os.path.join(out_path, "data"), exist_ok=True)
    it = _progress(enumerate(zip(ks, potens)), show_tqdm, desc=filename, total=len(ks),
                   position=position, leave=False)
    for idx, (k, pot) in it:
        knth = load_k_dump(f"{dir_in}/{k}", resolution)
        orig_knth = knth.copy()
        knth = np.moveaxis(knth, 0, -1).copy().view(dtype=np.complex64)
        knth = np.fft.fftshift(knth, axes=(3,))
        if separate_zf:
            knth = np.concatenate(_split_modes(knth, split_into_bands), axis=0)
        else:
            knth = do_ifft(knth)
        assert check_ifft(knth, orig_knth, zf_separated=separate_zf), \
            "error transforming back to original space"

        a = np.loadtxt(f"{dir_in}/{pot}")
        phi_gkw = np.reshape(a, (nx, ns, ny), order="F").astype("float32").copy()
        b = np.loadtxt(f"{dir_in}/{pot.replace('Poten', 'Spc3d')}")
        gt_spc = np.reshape(b, (nkx, ns, nky), order="F")
        phi_fft = phi_to_spc(phi_gkw, gt_spc, out_shape=(nkx, ns, nky))
        phi_gkw = phi_fft_to_real(phi_fft, out_shape=phi_fft.shape).astype(np.float32)

        df2 = knth.reshape(-1, 2, *knth.shape[1:]).sum(0) if knth.shape[0] != 2 else knth
        phi_int, eflux = solver(df2)
        if not np.isclose(eflux, orig_fluxes[idx], rtol=0.0, atol=1e-2):
            warnings.warn(f"Flux integral does not match original flux! "
                          f"Computed: {eflux}, Original: {orig_fluxes[idx]}")
        assert np.isclose(eflux, orig_fluxes[idx], rtol=0.0, atol=1.0), "strong deviation for flux"
        rel = np.linalg.norm(phi_gkw - phi_int) / np.linalg.norm(phi_int)
        assert rel < 1e-2, f"poten {pot} does not match the field solve of {k} (rel-L2 {rel:.3e})"
        phi = phi_int if phi_source == "field_solve" else phi_gkw

        df_stats.update(knth)
        flux_stats.update(fluxes[idx])
        phi_stats.update(phi)
        if not metadata_only:
            _write_bin(os.path.join(out_path, "data", f"timestep_{idx:05d}.bin"), knth)
            _write_bin(os.path.join(out_path, "data", f"poten_{idx:05d}.bin"), phi)

    metadata.update(_stats_dict("df", df_stats, np.float32))
    metadata.update(_stats_dict("phi", phi_stats, np.float32))
    metadata.update({k: np.float64(v) for k, v in _stats_dict("flux", flux_stats).items()})
    if metadata_only:
        metadata = _merge_old_metadata(out_path, metadata)
    write_metadata(out_path, metadata)
    return out_path, False


def _split_modes(knth: np.ndarray, split_into_bands: Optional[int]) -> list[np.ndarray]:
    """Zonal (ky=0) and turbulent (optionally ky-banded) real-space parts of a spectral df."""
    nky = knth.shape[4]
    zf, no_zf = knth.copy(), knth.copy()
    zf[..., 1:, :] = 0.0
    no_zf[..., 0, :] = 0.0
    out = [do_ifft(zf)]
    if not split_into_bands:
        return out + [do_ifft(no_zf)]
    per = nky // split_into_bands
    for band in range(split_into_bands):
        cur = np.zeros_like(no_zf)
        lo = 1 + band * per
        hi = None if band == split_into_bands - 1 else lo + per
        cur[..., lo:hi, :] = no_zf[..., lo:hi, :]
        out.append(do_ifft(cur))
    return out


def rewrite_poten(traj_dir: str, backup_dir: str, x64: bool = True) -> str:
    """Overwrite ``poten_*.bin`` of a preprocessed trajectory with the field solve of its df.

    The originals and both metadata files are copied to ``backup_dir/<name>`` first and the
    phi statistics are recomputed in both metadata files. Trajectories with a ``DONE``
    marker in their backup are skipped.
    """
    from neugk_jax.dataset.backend import NumpyBackend, load_meta, save_meta

    name = os.path.basename(traj_dir.rstrip("/"))
    bdir = os.path.join(backup_dir, name)
    if os.path.exists(os.path.join(bdir, "DONE")):
        return f"{name}: skip (done)"
    os.makedirs(bdir, exist_ok=True)
    data = os.path.join(traj_dir, "data")
    potens = sorted(glob.glob(os.path.join(data, "poten_[0-9][0-9][0-9][0-9][0-9].bin")))
    metas = [p for p in (_meta_path(os.path.join(traj_dir, m)) for m in ("metadata", "metadata_light"))
             if p is not None]
    for p in potens + metas:
        b = os.path.join(bdir, os.path.basename(p))
        if not os.path.exists(b):
            shutil.copy2(p, b)

    meta = NumpyBackend().read_metadata(traj_dir)
    shape = (2, *meta["resolution"])
    solver = FieldSolver(meta["geometry"], x64=x64)
    stats = _running_stats()
    for p in potens:
        idx = os.path.basename(p)[6:11]
        df = np.fromfile(os.path.join(data, f"timestep_{idx}.bin"), dtype=np.float32).reshape(shape)
        phi = solver(df)[0]
        assert phi.nbytes == os.path.getsize(p), (p, phi.shape)
        phi.tofile(p + ".tmp")
        os.replace(p + ".tmp", p)
        stats.update(phi)

    new = _stats_dict("phi", stats)
    for mp in metas:
        base, ext = os.path.splitext(mp)
        m = load_meta(base)
        for k in [k for k in new if k in m]:
            m[k] = np.asarray(new[k], dtype=np.asarray(m[k]).dtype)
        save_meta(base + ".tmp", m, ext)
        os.replace(base + ".tmp" + ext, mp)
    with open(os.path.join(bdir, "DONE"), "w") as f:
        f.write(f"{len(potens)}\n")
    return f"{name}: rewrote {len(potens)} potentials"


def preprocess_gyaradax(
    traj_dir: str,
    backend=None,
    target_dir: str = TARGET_DIR,
    out_name: Optional[str] = None,
    metadata_only: bool = False,
    verify: bool = True,
    flux_atol: float = 1.0,
    show_tqdm: bool = False,
    x64: bool = True,
) -> str:
    """Convert a gyaradax run folder (``step_*.npz`` + ``config.yaml`` + ``geometry.pkl``).

    Writes the kvikio layout (real-space 2-channel df, field-solved real phi, metadata with
    kyspec/fluxspec and statistics) with the geometry in the GKW convention. The flux of each
    df is checked against the gyaradax-reported heat flux. Returns the output path.
    """
    from omegaconf import OmegaConf

    from neugk_jax.dataset.backend import NumpyBackend

    backend = backend or NumpyBackend()
    traj_dir = str(traj_dir)
    name = out_name or os.path.basename(os.path.normpath(traj_dir))
    dir_out = os.path.join(target_dir, KVIKIO_SUBDIR)
    os.makedirs(dir_out, exist_ok=True)
    out_path = backend.format_path(os.path.join(dir_out, name), spatial_ifft=True,
                                   split_into_bands=None, real_potens=True)

    cfg = OmegaConf.load(os.path.join(traj_dir, "config.yaml"))
    with open(os.path.join(traj_dir, "geometry.pkl"), "rb") as fh:
        np_geom = pickle.load(fh)
    adiabatic = 1.0 if bool(cfg.grid.get("adiabatic_electrons", True)) else 0.0
    np_geom["adiabatic"] = np.array(adiabatic, dtype=np.float64)
    np_geom["beta"] = np.array(float(cfg.physics.get("beta", 0.0)), dtype=np.float64)
    np_geom["nlapar"] = np.array(0.0, dtype=np.float64)
    np_geom["nlbpar"] = np.array(0.0, dtype=np.float64)

    ints = np.asarray(np_geom["ints"])
    ns = len(ints)
    resolution = (len(np.asarray(np_geom["intvp"])), len(np.asarray(np_geom["intmu"])), ns,
                  len(np.asarray(np_geom["kxrh"])), len(np.asarray(np_geom["krho"])))
    # gyaradax ky weights [1, 2, 2, ...] -> gkw convention [1, 2ns, 2ns, ...]
    parseval = np.asarray(np_geom["parseval"], dtype=np.float64).copy()
    parseval[1:] *= float(ns)
    np_geom["parseval"] = parseval
    sgrid = np.asarray(np_geom["sgrid"]).ravel()
    ds = float(sgrid[1] - sgrid[0]) if sgrid.size > 1 else 1.0 / ns

    steps = sorted(glob.glob(os.path.join(traj_dir, "step_*.npz")))
    if not steps:
        raise FileNotFoundError(f"no step_*.npz dumps in {traj_dir}")
    solve_geom = {"de": np.array(1.0), **_numeric_geometry(np_geom)}
    solver = FieldSolver(solve_geom, x64=x64)

    times, fluxes, kyspecs, fluxspecs = [], [], [], []
    df_stats, phi_stats, flux_stats = _running_stats(), _running_stats(), _running_stats()
    os.makedirs(os.path.join(out_path, "data"), exist_ok=True)
    for idx, step_path in _progress(enumerate(steps), show_tqdm, total=len(steps), desc=name,
                                    leave=False):
        d = np.load(step_path)
        df_real = solver_df_to_realspace(d["df"])
        phi, eflux_total = solver(df_real)
        reported = float(d["fluxes"][1])
        if verify and not np.isclose(eflux_total, reported, rtol=0.0, atol=flux_atol):
            warnings.warn(f"{name} step {int(d['step'])}: flux {eflux_total:.4f} != reported "
                          f"{reported:.4f}")
        fluxspecs.append(solver.flux_spectrum(df_real).astype(np.float32))
        kyspecs.append(np.asarray(d["ky_spec"], dtype=np.float32))
        times.append(float(d["time"]))
        fluxes.append(reported)
        df_stats.update(df_real)
        phi_stats.update(phi)
        flux_stats.update(reported)
        if not metadata_only:
            _write_bin(os.path.join(out_path, "data", f"timestep_{idx:05d}.bin"), df_real)
            _write_bin(os.path.join(out_path, "data", f"poten_{idx:05d}.bin"), phi)

    metadata = {
        "timesteps": np.asarray(times),
        "resolution": resolution,
        "ds": ds,
        "ion_temp_grad": np.array([float(cfg.physics.rlt)]),
        "density_grad": np.array([float(cfg.physics.rln)]),
        "flux": np.clip(np.asarray(fluxes), a_min=0.0, a_max=None),
        "s_hat": np.array([float(cfg.geometry.shat)]),
        "q": np.array([float(cfg.geometry.q)]),
        "geometry": np_geom,
        "kyspec": np.asarray(kyspecs),
        "fluxspec": np.asarray(fluxspecs),
        **_stats_dict("df", df_stats, np.float32),
        **_stats_dict("phi", phi_stats, np.float32),
        **{k: np.float32(v) for k, v in _stats_dict("flux", flux_stats).items()},
    }
    write_metadata(out_path, metadata)
    return out_path


def _gkw_datasets(args) -> list[str]:
    if args.trajs_file:
        with open(args.trajs_file) as fh:
            return [line.strip() for line in fh if line.strip()]
    if args.trajs:
        return expand_spec(args.trajs)
    return [f"iteration_{i}" for i in range(args.num_iterations)]


def _run_preprocess(args) -> None:
    datasets = _gkw_datasets(args)
    kwargs = dict(
        spatial_ifft=True, separate_zf=args.separate_zf, split_into_bands=args.split_into_bands,
        root=args.root, raw_subdir=args.raw_subdir, target_dir=args.target_dir,
        metadata_only=args.metadata_only, geometry_only=args.geometry_only,
        phi_source=args.phi_source, max_timesteps=args.max_timesteps, x64=not args.fp32,
        show_tqdm=args.tqdm,
    )

    def one(i_name):
        i, name = i_name
        try:
            return name, *preprocess(name, position=1 + i % max(1, args.num_workers), **kwargs), None
        except (OSError, ValueError, AssertionError, IndexError, KeyError) as e:
            return name, None, False, e

    skipped = []
    with ThreadPoolExecutor(max(1, min(len(datasets), args.num_workers))) as ex:
        for name, out_path, was_skipped, err in ex.map(one, enumerate(datasets)):
            if err is not None:
                print(f"Error processing {name}: {err}", file=sys.stderr, flush=True)
            elif was_skipped:
                skipped.append(name)
            else:
                from neugk_jax.dataset.backend import load_meta

                meta = load_meta(os.path.join(out_path, "metadata"))
                msg = f"{out_path}: {len(meta['timesteps'])} points"
                if "df_mean" in meta:
                    msg += (f", mean {meta['df_mean'][0].mean():.2e}, "
                            f"std {meta['df_std'][0].mean():.2e}")
                print(msg, flush=True)
    if skipped:
        print(f"Skipped {len(skipped)} trajectories (already processed).")


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("preprocess", "rewrite-phi", "gyaradax", "quantize"),
                    default="quantize")
    ap.add_argument("--path", default=os.path.join(TARGET_DIR, KVIKIO_SUBDIR),
                    help="preprocessed dataset root (rewrite-phi, quantize)")
    ap.add_argument("--trajs", nargs="+", default=None,
                    help="brace pattern (single string) OR explicit list of trajectories; "
                         "raw run names for preprocess, trajectory dirs otherwise")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tqdm", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="field solve in float32 (default float64)")
    g = ap.add_argument_group("preprocess")
    g.add_argument("--root", default=RAW_ROOT)
    g.add_argument("--raw-subdir", default="raw")
    g.add_argument("--target-dir", default=TARGET_DIR,
                   help="output root; data goes to <target-dir>/preprocessed_kvikio")
    g.add_argument("--num-iterations", type=int, default=300,
                   help="iteration_0 .. iteration_N-1 when neither --trajs nor --trajs-file is set")
    g.add_argument("--trajs-file", default=None, help="file with one raw run name per line")
    g.add_argument("--metadata-only", action="store_true",
                   help="only rewrite metadata (with statistics), no field data")
    g.add_argument("--geometry-only", action="store_true",
                   help="only rewrite metadata without statistics (existing ones are kept)")
    g.add_argument("--phi-source", choices=("field_solve", "gkw"), default="field_solve")
    g.add_argument("--max-timesteps", type=int, default=None)
    g.add_argument("--separate-zf", action="store_true")
    g.add_argument("--split-into-bands", type=int, default=None)
    g = ap.add_argument_group("rewrite-phi")
    g.add_argument("--poten-backup", default=None, help="backup directory (required)")
    g = ap.add_argument_group("gyaradax")
    g.add_argument("--gyaradax-dirs", nargs="+", default=None)
    g = ap.add_argument_group("quantize")
    g.add_argument("--bits", choices=tuple(_DTYPE_SUFFIX), default="bf16",
                   help="quantization target (fp16 / bf16 / i8 / i4)")
    g.add_argument("--force", action="store_true", help="overwrite existing quantized shards")
    args = ap.parse_args(argv)

    if args.mode == "quantize":
        run_quantize(path=args.path, trajs=args.trajs or ["iteration_{0-299}_ifft_realpotens"],
                     bits=args.bits, num_workers=args.num_workers, force=args.force)
    elif args.mode == "preprocess":
        _run_preprocess(args)
    elif args.mode == "rewrite-phi":
        if not args.poten_backup:
            ap.error("--mode=rewrite-phi requires --poten-backup")
        traj_dirs = resolve_traj_dirs(args.path, args.trajs)
        with ThreadPoolExecutor(max(1, args.num_workers)) as ex:
            futures = [ex.submit(rewrite_poten, d, args.poten_backup, not args.fp32)
                       for d in traj_dirs]
            for fut in as_completed(futures):
                print(fut.result(), flush=True)
    elif args.mode == "gyaradax":
        if not args.gyaradax_dirs:
            ap.error("--mode=gyaradax requires --gyaradax-dirs")
        from neugk_jax.dataset.backend import load_meta

        for traj_dir in args.gyaradax_dirs:
            out = preprocess_gyaradax(traj_dir, target_dir=args.target_dir,
                                      metadata_only=args.metadata_only, show_tqdm=args.tqdm,
                                      x64=not args.fp32)
            meta = load_meta(os.path.join(out, "metadata"))
            print(f"{out}: {len(meta['timesteps'])} steps, "
                  f"df mean/std {meta['df_mean'].mean():.3e}/{meta['df_std'].mean():.3e}, "
                  f"flux mean {float(np.mean(meta['flux'])):.3f}", flush=True)


if __name__ == "__main__":
    main()
