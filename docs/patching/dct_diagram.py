"""Diagram of DCT field patching with linear heads, as one SVG (same style as ``smooth_diagram.py``).

The drawings come from a trained model on a real frame (``extract_dct.py``).
Usage: dct_diagram.py <data.npz> <out.svg>
"""

import sys
from pathlib import Path

import numpy as np
import smooth_diagram as sd
from smooth_diagram import (
    ACCENT,
    CW,
    INK,
    MUTED,
    Canvas,
    M,
    card,
    dots,
    heat,
    label,
    matrix,
    vstrip,
)

AXES = [r"v_\parallel", "s", "x", "y", r"\mu"]


def basis_row(cv, x, y, bases, c, gap=40):
    """1D bases ``(points, modes)`` of every axis, modes as rows over points as columns."""
    for name, b in zip(AXES, bases):
        w, h = heat(cv, x, y, b.T, c)
        label(cv, x + w / 2, y + h + 26, rf"B^{{{name}}}", size=20, color=INK)
        cv.text(
            x + w / 2,
            y + h + 48,
            f"{b.shape[0]} × {b.shape[1]}",
            size=15,
            color=MUTED,
            anchor="middle",
        )
        x += w + gap
    return x


def main(data, out):
    d = np.load(data, allow_pickle=True)
    cv = Canvas()
    plane, (_, ps, px, _) = d["plane"], d["patch"]
    tok = d["token"]
    eb = list(d["bases_enc"])
    ranks = tuple(int(r) for r in d["ranks"])
    n_c, dim = int(d["channels"]), int(d["token_dim"])
    big = 26
    cx = M + CW / 2
    enc_c, dec_c = ACCENT["enc"][0], ACCENT["dec"][0]
    y = 30

    # 1: tiling
    h1 = 300
    card(cv, y, h1, "tile", "Patch tiling", "")
    cell, zc = 8.4, 38
    total = plane.shape[1] * cell + 110 + px * zc + 130
    gx, gy = cx - total / 2, y + 96
    pw, ph = heat(cv, gx, gy, plane, cell, gap=0.6, r=1)
    for j in range(0, plane.shape[1] + 1, px):
        cv.line(
            gx + j * cell - 0.3, gy - 4, gx + j * cell - 0.3, gy + ph + 4, color="#334155", sw=1
        )
    for i in range(0, plane.shape[0] + 1, ps):
        cv.line(
            gx - 4, gy + i * cell - 0.3, gx + pw + 4, gy + i * cell - 0.3, color="#334155", sw=1
        )
    hx, hy = gx + tok[2] * px * cell, gy + tok[1] * ps * cell
    cv.rect(hx - 2, hy - 2, px * cell + 3.4, ps * cell + 3.4, "none", stroke=INK, sw=3.4, r=2)
    cv.text(gx + pw / 2, gy + ph + 38, "x", size=big, color=MUTED, anchor="middle")
    cv.text(gx - 14, gy + ph / 2 + 9, "s", size=big, color=MUTED, anchor="end")
    zx, zy = gx + pw + 110, gy + ph / 2 - ps * zc / 2
    cv.line(hx + px * cell, hy, zx - 8, zy - 4, color=INK, sw=2.6, dash="7 5")
    cv.line(hx + px * cell, hy + ps * cell, zx - 8, zy + ps * zc + 4, color=INK, sw=2.6, dash="7 5")
    zw, zh = dots(cv, zx, zy, d["truth"], zc)
    cv.rect(zx - 6, zy - 6, zw + 12, zh + 12, "none", stroke=INK, sw=3.6, r=10)
    cv.math(zx + zw + 28, zy + zh / 2 - 6, r"x_p", size=34)
    cv.text(zx + zw + 28, zy + zh / 2 + 28, "one patch", size=20, color=MUTED)
    cv.text(zx + zw + 28, zy + zh / 2 + 52, "= one token", size=20, color=MUTED)
    y += h1 + 44

    # 2: per-axis dct bases and their tensor product
    learned = bool(d["learned"])
    h2 = 690 if learned else 500
    card(
        cv,
        y,
        h2,
        "basis",
        "Per-axis DCT bases",
        "one 1D basis per axis, combined by a Tucker product",
    )
    c = 15
    x0, bx = M + 60, M + 560
    ry = y + 130
    cv.text(x0, ry - 30, "every axis: DCT-II modes", size=21, weight=600)
    cv.math(x0, ry + 22, r"c_m(u) = \cos\!\big(m\pi\tfrac{u+1}{2}\big)", size=22)
    cv.text(x0, ry + 56, "u: cell-centred coordinate in the window", size=17, color=MUTED)
    cv.text(x0, ry + 78, "fixed, the same on every grid", size=17, color=MUTED)
    basis_row(cv, bx, ry - 10, list(d["modes"]), c)
    cv.text(x0, ry + 100, "maps: modes (rows) over points (columns)", size=17, color=MUTED)
    if learned:
        ry += 200
        cv.text(x0, ry - 30, "learned: the modes re-mixed", size=21, weight=600)
        cv.math(x0, ry + 22, r"B^d_r = \sum_m A^d_{mr}(\text{ctx})\, c_m", size=22)
        cv.text(x0, ry + 56, "A = A0 + hypernetwork(log patch scale)", size=17, color=MUTED)
        cv.text(x0, ry + 78, "starts at the DCT", size=17, color=MUTED)
        basis_row(cv, bx, ry - 10, eb, c)
    ry3 = ry + 205
    cv.text(
        x0,
        ry3 - 30,
        "Tucker product: the functions the patch is projected onto",
        size=21,
        weight=600,
    )
    cv.math(
        x0,
        ry3 + 22,
        r"b_{aijbm}(p) = B^{v_\parallel}_a\, B^s_i\, B^x_j\, B^y_b\, B^{\mu}_m",
        size=22,
    )
    n_fun = int(np.prod(ranks)) * n_c
    rk = "·".join(str(r) for r in ranks)
    cv.text(
        x0, ry3 + 56, f"ranks {rk} × {n_c} channels = {n_fun} projections", size=17, color=MUTED
    )
    tx, pc = bx + 90, 15
    for i, j in ((0, 1), (1, 2), (2, 1)):
        heat(cv, tx, ry3 - 10, np.outer(eb[1][:, i], eb[2][:, j]), pc)
        label(cv, tx + px * pc / 2, ry3 + ps * pc + 16, rf"B^s_{i}\,B^x_{j}", size=18, color=INK)
        tx += px * pc + 24
    basis_out = (tx - 14, ry3 - 10 + ps * pc / 2)
    y2_end = y + h2
    y += h2 + 44

    # 3: encoder
    h3 = 280
    card(cv, y, h3, "enc", "Encoder", "projection onto the Tucker basis, per channel")
    c = 22
    ey = y + 110
    mid = ey + ps * c / 2
    x = M + 70
    dots(cv, x, ey, d["truth"], c)
    label(cv, x + px * c / 2, ey + ps * c + 36, r"x_p", size=26, color=INK)
    cv.math(x + px * c + 10, mid + 10, r"\times", size=28, color=enc_c)
    bx = x + px * c + 40
    heat(cv, bx, ey, np.outer(eb[1][:, 1], eb[2][:, 2]), c)
    label(cv, bx + px * c / 2, ey + ps * c + 36, r"b(p)", size=24, color=INK)
    enc_in = (bx + px * c / 2, ey - 6)
    x = bx + px * c + 16
    cv.arrow(x, mid, x + 36, color=enc_c)
    x += 50
    cv.math(
        x,
        mid - 10,
        r"h = x \times_{v} B^{v_\parallel} \times_{s} B^{s} \times_{x} B^{x} \times_{y} B^{y} \times_{\mu} B^{\mu}",
        size=20,
    )
    cv.text(x, mid + 26, "one weighted contraction per axis", size=18, color=MUTED)
    x += 470
    heat(cv, x, ey - 6, d["core_enc_sx"], 16)
    label(
        cv,
        x + d["core_enc_sx"].shape[1] * 8,
        ey + 76,
        r"h_{\cdot ij \cdot\cdot}",
        size=22,
        color=INK,
    )
    cv.text(
        x + d["core_enc_sx"].shape[1] * 8,
        ey + 100,
        "core slice",
        size=16,
        color=MUTED,
        anchor="middle",
    )
    x += d["core_enc_sx"].shape[1] * 16 + 20
    cv.arrow(x, mid, x + 36, color=enc_c)
    x += 46
    matrix(cv, x, mid - 46, 96, 92, enc_c, ACCENT["enc"][1], r"W")
    cv.text(x + 48, mid + 72, f"{n_fun} → {dim}", size=17, color=MUTED, anchor="middle")
    x += 106
    cv.arrow(x, mid, x + 36, color=enc_c)
    x += 46
    vstrip(cv, x, mid - 60, d["z"], 26, 120)
    label(cv, x + 13, mid + 98, r"z", size=26, color=INK)
    enc_out = (x + 13, mid + 60)
    y += h3 + 30

    # backbone
    hb = 110
    card(cv, y, hb, "back", "", "")
    cv.text(cx, y + hb / 2 + 13, "Backbone", size=38, weight=700, anchor="middle")
    cv.varrow(enc_out[0], enc_out[1] + 50, y - 2, color=MUTED)
    back_out = (M, y + hb / 2)
    yb_end = y + hb
    y += hb + 30

    # 4: decoder
    h4 = 330
    card(cv, y, h4, "dec", "Decoder", "")
    c = 22
    dy = y + 120
    mid = dy + ps * c / 2
    terms = d["terms"]
    energy = np.sqrt((terms**2).sum((2, 3)))
    picks = list(zip(*np.unravel_index(np.argsort(-energy.ravel())[:3], energy.shape)))
    scale = np.abs(d["truth"]).max()
    x = M + 70
    vstrip(cv, x, mid - 60, d["z"], 26, 120)
    label(cv, x + 13, mid + 98, r"z", size=26, color=INK)
    dec_zmid = (x, mid)
    x += 36
    cv.arrow(x, mid, x + 36, color=dec_c)
    x += 46
    matrix(cv, x, mid - 46, 96, 92, dec_c, ACCENT["dec"][1], r"B")
    cv.text(x + 48, mid + 72, f"{dim} → {n_fun}", size=17, color=MUTED, anchor="middle")
    x += 106
    cv.arrow(x, mid, x + 36, color=dec_c)
    x += 70
    heat(cv, x, mid - 34, d["core_dec_sx"], 16)
    label(cv, x + d["core_dec_sx"].shape[1] * 8, mid + 62, r"c_{ij}", size=22, color=INK)
    cv.text(
        x + d["core_dec_sx"].shape[1] * 8,
        mid + 88,
        "core (slice)",
        size=16,
        color=MUTED,
        anchor="middle",
    )
    x += d["core_dec_sx"].shape[1] * 16 + 20
    cv.arrow(x, mid, x + 36, color=dec_c)
    x += 50
    t0 = x
    for n, (i, j) in enumerate(picks):
        heat(cv, x, dy, terms[i, j], c, scale=scale)
        label(
            cv, x + px * c / 2, dy + ps * c + 30, rf"c_{{{i}{j}}}B^s_{i}B^x_{j}", size=17, color=INK
        )
        x += px * c + 10
        cv.math(x + 6, mid + 12, r"+" if n < len(picks) - 1 else r"+\cdots", size=30, color=dec_c)
        x += 42 if n < len(picks) - 1 else 96
    cv.math(x, mid + 12, r"=", size=30, color=dec_c)
    x += 42
    dots(cv, x, dy, d["recon"], c, scale=scale)
    label(cv, x + px * c / 2, dy + ps * c + 36, r"u_p", size=26, color=INK)
    dec_in = (t0 + px * c / 2, dy - 6)
    cv.text(
        cx,
        y + h4 - 22,
        "the patch as a sum of rank-one tensor products of the axis bases",
        size=22,
        color=MUTED,
        anchor="middle",
    )
    y += h4 + 34

    # connectors
    lane1, lane2 = M + CW + 22, M + CW + 50
    cv.elbow(
        [
            (basis_out[0] + 10, basis_out[1]),
            (lane1, basis_out[1]),
            (lane1, y2_end + 22),
            (enc_in[0], y2_end + 22),
            enc_in,
        ],
        enc_c,
    )
    cv.elbow(
        [
            (basis_out[0] + 10, basis_out[1] + 14),
            (lane2, basis_out[1] + 14),
            (lane2, yb_end + 11),
            (dec_in[0], yb_end + 11),
            dec_in,
        ],
        dec_c,
    )
    side = M - 20
    cv.elbow(
        [back_out, (side, back_out[1]), (side, dec_zmid[1]), (dec_zmid[0] - 4, dec_zmid[1])], MUTED
    )

    Path(out).write_text(cv.render(y))
    print(out, sd.W, y)


if __name__ == "__main__":
    main(*sys.argv[1:])
