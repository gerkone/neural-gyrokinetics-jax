"""Traditional lossy baselines of the compression benchmark: ZFP, SZ3, wavelet, PCA and JPEG2000.

Each codec takes the real-space df ``(2, vp, mu, s, x, y)`` in physical units and returns
``(reconstruction, payload, nbytes)``. The methods are kept plain (library or textbook form, no
extra transforms, entropy coders or lossless post-passes); only the array layout and the library
settings are chosen. The reconstruction is decoded from the payload and ``nbytes`` counts all of
it. :func:`encode_at_cr` searches the single rate knob of a codec per snapshot to match a target
compression ratio.
"""

from __future__ import annotations

import io
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

SHAPE = (2, 32, 8, 16, 85, 32)


def _wrap(encode: Callable, decode: Callable, df, knob):
    x = np.asarray(df, dtype=np.float32)
    assert x.shape == SHAPE, x.shape
    payload = encode(x, knob)
    return decode(payload), payload, len(payload)


def _to_layout(df: np.ndarray, perm, shape) -> np.ndarray:
    return np.ascontiguousarray(df.transpose(perm), dtype=np.float32).reshape(shape)


def _from_layout(a: np.ndarray, perm) -> np.ndarray:
    a = a.reshape(tuple(SHAPE[i] for i in perm))
    return np.ascontiguousarray(a.transpose(np.argsort(perm)), dtype=np.float32)


# zfp: fixed accuracy on a 4d array ((re/im mu x), vpar, s, y)
_ZFP_PERM = (0, 2, 4, 1, 3, 5)
_ZFP_SHAPE = (2 * 8 * 85, 32, 16, 32)


def zfp_encode(df: np.ndarray, tolerance: float) -> bytes:
    import zfpy

    tol = np.float32(tolerance)
    # unit tolerance on the rescaled field
    a = _to_layout(df, _ZFP_PERM, _ZFP_SHAPE) / tol
    return struct.pack("<f", tol) + zfpy.compress_numpy(a, tolerance=1.0, write_header=False)


def zfp_decode(payload: bytes) -> np.ndarray:
    import zfpy

    tol = np.float32(struct.unpack("<f", payload[:4])[0])
    a = zfpy._decompress(payload[4:], zfpy.type_float, list(_ZFP_SHAPE), tolerance=1.0) * tol
    return _from_layout(a, _ZFP_PERM)


def zfp_recon(df, tolerance: float = 300.0):
    return _wrap(zfp_encode, zfp_decode, df, float(tolerance))


# sz3: linear interpolation predictor on a 4d array ((mu re/im s), vpar, y, x)
_SZ3_PERM = (2, 0, 3, 1, 5, 4)
_SZ3_SHAPE = (8 * 2 * 16, 32, 32, 85)
_SZ3_INI = (
    "[GlobalSettings]\nCmprAlgo = ALGO_INTERP\n"
    "[AlgoSettings]\nInterpolationAlgo = INTERP_ALGO_LINEAR\nInterpolationAlpha = 4\n"
    "InterpolationBeta = 16\n"
)
_SZ3_CFG_PATH: list[str] = []


def _sz3_config(error_bound: float):
    from pysz import szConfig

    if not _SZ3_CFG_PATH:
        fd, path = tempfile.mkstemp(suffix=".ini")
        os.write(fd, _SZ3_INI.encode())
        os.close(fd)
        _SZ3_CFG_PATH.append(path)
    cfg = szConfig()
    cfg.loadcfg(_SZ3_CFG_PATH[0])
    cfg.errorBoundMode = 0
    cfg.absErrorBound = float(error_bound)
    return cfg


def sz3_encode(df: np.ndarray, error_bound: float) -> bytes:
    from pysz import sz

    stream, _ = sz.compress(_to_layout(df, _SZ3_PERM, _SZ3_SHAPE), _sz3_config(error_bound))
    return bytes(stream)


def sz3_decode(payload: bytes) -> np.ndarray:
    from pysz import sz

    a, _ = sz.decompress(np.frombuffer(payload, dtype=np.uint8).copy(), np.float32, _SZ3_SHAPE)
    return _from_layout(a, _SZ3_PERM)


def sz3_recon(df, error_bound: float = 5.0):
    return _wrap(sz3_encode, sz3_decode, df, float(error_bound))


# wavelet: separable periodized dwt, global top-k on the re/im magnitude, sparse float16 storage
_WAV_AXES = (
    (1, "bior6.8", 2),
    (2, "haar", 1),
    (3, "bior6.8", 2),
    (4, "bior6.8", 4),
    (5, "bior6.8", 4),
)
_WAV_HDR = "<If"


def _wav_sizes(n, wav, level):
    import pywt

    flen = pywt.Wavelet(wav).dec_len
    lens = [n]
    for _ in range(level):
        lens.append(pywt.dwt_coeff_len(lens[-1], flen, "periodization"))
    return [lens[-1]] + lens[:0:-1]


def wavelet_encode(df: np.ndarray, k: float) -> bytes:
    import pywt

    a = df.astype(np.float64)
    for ax, wav, lv in _WAV_AXES:
        a = np.concatenate(pywt.wavedec(a, wav, mode="periodization", level=lv, axis=ax), axis=ax)
    k = max(int(round(k)), 1)
    re, im = a[0].ravel(), a[1].ravel()
    idx = np.sort(np.argpartition(re * re + im * im, -k)[-k:]).astype(np.uint32)
    vals = np.stack([re[idx], im[idx]], axis=1)
    scale = float(np.abs(vals).max()) or 1.0
    # uint32 flat index plus float16 re/im per kept coefficient
    body = idx.tobytes() + (vals / scale).astype(np.float16).tobytes()
    return struct.pack(_WAV_HDR, k, scale) + body


def wavelet_decode(payload: bytes) -> np.ndarray:
    import pywt

    k, scale = struct.unpack_from(_WAV_HDR, payload)
    off = struct.calcsize(_WAV_HDR)
    idx = np.frombuffer(payload, np.uint32, k, off)
    vals = np.frombuffer(payload, np.float16, 2 * k, off + 4 * k).astype(np.float64)
    vals = vals.reshape(k, 2) * scale
    cshape = list(SHAPE)
    for ax, wav, lv in _WAV_AXES:
        cshape[ax] = sum(_wav_sizes(SHAPE[ax], wav, lv))
    c = np.zeros(cshape, np.float64)
    c[0].reshape(-1)[idx] = vals[:, 0]
    c[1].reshape(-1)[idx] = vals[:, 1]
    for ax, wav, lv in reversed(_WAV_AXES):
        parts = np.split(c, np.cumsum(_wav_sizes(SHAPE[ax], wav, lv))[:-1], axis=ax)
        rec = pywt.waverec(parts, wav, mode="periodization", axis=ax)
        c = np.take(rec, np.arange(SHAPE[ax]), axis=ax)
    return c.astype(np.float32)


def wavelet_recon(df, k: float = 9000.0):
    return _wrap(wavelet_encode, wavelet_decode, df, float(k))


# pca: truncated svd of the (vpar mu s) x (re/im x y) unfolding, uncentered, float16 factors
_PCA_PERM = (1, 2, 3, 0, 4, 5)
_PCA_M = 32 * 8 * 16


def pca_encode(df: np.ndarray, rank: int) -> bytes:
    from scipy.linalg import eigh

    rank = int(rank)
    a = df.astype(np.float64).transpose(_PCA_PERM).reshape(_PCA_M, -1)
    # top singular triplets from the smaller gram matrix
    w, u = eigh(a @ a.T, subset_by_index=(max(_PCA_M - rank, 0), _PCA_M - 1))
    u, s = u[:, ::-1], np.sqrt(np.maximum(w[::-1], 0.0))
    v = (a.T @ u) / np.where(s > 0, s, 1.0)[None, :]
    scores = (u * s[None, :]).astype(np.float16)
    return struct.pack("<H", rank) + scores.tobytes() + v.astype(np.float16).tobytes()


def pca_decode(payload: bytes) -> np.ndarray:
    (rank,) = struct.unpack_from("<H", payload)
    n = int(np.prod(SHAPE)) // _PCA_M
    scores = np.frombuffer(payload, np.float16, _PCA_M * rank, 2)
    v = np.frombuffer(payload, np.float16, n * rank, 2 + 2 * _PCA_M * rank)
    scores = scores.reshape(_PCA_M, rank).astype(np.float64)
    v = v.reshape(n, rank).astype(np.float64)
    return _from_layout(scores @ v.T, _PCA_PERM)


def pca_recon(df, rank: int = 4):
    return _wrap(pca_encode, pca_decode, df, int(rank))


# jpeg2000: one mosaic with rows (re/im vpar x) and cols (mu s y), 16 bit, irreversible 9/7
_J2K_PERM = (0, 1, 4, 2, 3, 5)
_J2K_ROWS = 2 * 32 * 85


def jpeg2000_encode(df: np.ndarray, ratio: float) -> bytes:
    from PIL import Image

    img = df.transpose(_J2K_PERM).reshape(_J2K_ROWS, -1)
    mn, mx = float(img.min()), float(img.max())
    sc = (mx - mn) / 65535.0 if mx > mn else 1.0
    u16 = np.round((img.astype(np.float64) - mn) / sc).clip(0, 65535).astype(np.uint16)
    buf = io.BytesIO()
    # openjpeg rates are relative to the 16 bit image, half the float32 ratio
    Image.fromarray(u16).save(
        buf,
        format="JPEG2000",
        no_jp2=True,
        quality_mode="rates",
        quality_layers=[float(ratio) / 2.0],
        irreversible=True,
        num_resolutions=9,
        codeblock_size=(64, 64),
    )
    return struct.pack("<ff", mn, mx) + buf.getvalue()


def jpeg2000_decode(payload: bytes) -> np.ndarray:
    from PIL import Image

    mn, mx = struct.unpack_from("<ff", payload)
    sc = (mx - mn) / 65535.0 if mx > mn else 1.0
    img = np.asarray(Image.open(io.BytesIO(payload[8:]))).astype(np.float32) * sc + mn
    return _from_layout(img, _J2K_PERM)


def jpeg2000_recon(df, ratio: float = 1167.0):
    return _wrap(jpeg2000_encode, jpeg2000_decode, df, float(ratio))


@dataclass(frozen=True)
class Codec:
    """A codec, its rate knob and range, whether the ratio grows with it, and an integer knob."""

    fn: Callable[..., Tuple[np.ndarray, bytes, int]]
    knob: str
    lo: float
    hi: float
    increases_cr: bool = True
    discrete: bool = False


CODECS = {
    "zfp": Codec(zfp_recon, "tolerance", 1e-3, 1e5),
    "sz3": Codec(sz3_recon, "error_bound", 1e-5, 1e5),
    "wavelet": Codec(wavelet_recon, "k", 20.0, 3e6, increases_cr=False),
    "pca": Codec(pca_recon, "rank", 1, 4000, increases_cr=False, discrete=True),
    "jpeg2000": Codec(jpeg2000_recon, "ratio", 5.0, 2e5),
}


def encode_at_cr(
    name: str,
    df,
    target_cr: float,
    warm: Optional[float] = None,
    tol: float = 0.02,
    max_iter: int = 14,
):
    """Encode ``df`` with codec ``name`` at about ``target_cr``; returns ``(recon, nbytes, knob)``.

    Secant search in log(knob)-log(CR) space, warm-started from ``warm`` (the previous snapshot's
    knob); the probe closest to the target is returned, since some codecs reach the ratio in steps.
    """
    codec = CODECS[name]
    fn, knob, disc = codec.fn, codec.knob, codec.discrete
    lo, hi, inc = codec.lo, codec.hi, codec.increases_cr
    df = np.asarray(df, dtype=np.float32)
    nbytes = df.nbytes
    cache: dict = {}

    def ev(v):
        v = min(max(float(v), lo), hi)
        key = int(round(v)) if disc else v
        if key not in cache:
            r, _, s = fn(df, **{knob: key})
            cache[key] = (r, s)
        r, s = cache[key]
        return key, r, s, (nbytes / s if s else float("inf"))

    lt = math.log(target_cr)
    best = [float("inf"), None]

    def probe(v):
        k, r, s, cr = ev(v)
        e = abs(math.log(cr) - lt)
        if e < best[0]:
            best[0], best[1] = e, (r, s, k)
        return k, cr

    # knob bracket: klow -> lower cr, khigh -> higher cr
    klow, khigh = (lo, hi) if inc else (hi, lo)
    seed = (
        warm
        if (warm is not None and lo <= warm <= hi)
        else math.exp((math.log(lo) + math.log(hi)) / 2)
    )
    ka, ca = probe(seed)
    if best[0] < tol:
        return best[1]
    if ca < target_cr:
        klow = ka
    else:
        khigh = ka
    kb, cb = probe(math.exp((math.log(klow) + math.log(khigh)) / 2))
    for _ in range(max_iter):
        if best[0] < tol:
            break
        if cb < target_cr:
            klow = kb
        else:
            khigh = kb
        lca, lcb = math.log(ca), math.log(cb)
        if abs(lcb - lca) < 1e-9 or abs(math.log(khigh) - math.log(klow)) < 1e-4:
            break
        # clamp the secant step in log space
        lm = math.log(kb) + (lt - lcb) * (math.log(kb) - math.log(ka)) / (lcb - lca)
        m = math.exp(min(max(lm, math.log(min(klow, khigh))), math.log(max(klow, khigh))))
        ka, ca = kb, cb
        kb, cb = probe(m)
    return best[1]
