"""Diagram of the band-limited continuous convolution with linear heads, as one SVG (formulas as MathJax paths).

The drawings come from a trained model on a real frame (``extract_cconv.py``). Needs node with
``mathjax-full`` (``NODE_PATH`` pointing at its ``node_modules``) for ``tex2svg.js``.
Usage: cconv_diagram.py <data.npz> <out.svg>
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
W, M = 1400, 40
LANE = 70
CW = W - 2 * M - LANE
FS = 1.1
FONT = "'Courier New', 'Liberation Mono', 'Nimbus Mono PS', monospace"
INK, MUTED, LINE, CARD = "#0f172a", "#64748b", "#e2e8f0", "#f8fafc"
ARROWS = ["#64748b", "#b45309", "#1d4ed8", "#047857"]
ACCENT = {
    "tile": ("#475569", "#f1f5f9"),
    "basis": ("#b45309", "#fef3c7"),
    "enc": ("#1d4ed8", "#dbeafe"),
    "dec": ("#047857", "#d1fae5"),
    "back": ("#64748b", "#f1f5f9"),
}
# card tints: the sidebar colour, faint
TINT = {
    "tile": ("#f5f7fa", "#e2e8f0"),
    "basis": ("#fffaf0", "#f3e3c3"),
    "enc": ("#f4f8ff", "#d6e2fb"),
    "dec": ("#f2fbf7", "#cdebdc"),
    "back": ("#f3f5f8", "#cbd5e1"),
}


def tex_svgs(texs):
    out = subprocess.run(
        ["node", str(HERE / "tex2svg.js")],
        input=json.dumps(texs),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class Canvas:
    def __init__(self):
        self.parts, self.texs, self.top = [], [], []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=14, color=INK, weight=400, anchor="start", italic=False):
        style = " font-style='italic'" if italic else ""
        size *= FS
        self.add(
            f"<text x='{x:.1f}' y='{y:.1f}' font-family=\"{FONT}\" font-size='{size}' font-weight='{weight}' fill='{color}' text-anchor='{anchor}'{style}>{s}</text>"
        )

    def math(self, x, y, tex, size=17, color=INK, anchor="start"):
        size *= FS
        # placeholder, filled with the mathjax svg (x, y: baseline start)
        self.texs.append((x, y, tex, size, color, anchor))
        self.add(f"@@MATH{len(self.texs) - 1}@@")

    def rect(self, x, y, w, h, fill, stroke="none", r=0, sw=1, dash=None):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.add(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{r}' fill='{fill}' stroke='{stroke}' stroke-width='{sw}'{d}/>"
        )

    def line(self, x1, y1, x2, y2, color=MUTED, sw=1.5, dash=None, arrow=False):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        if arrow:
            u = np.array([x2 - x1, y2 - y1], float)
            u /= np.linalg.norm(u)
            x2, y2 = x2 - u[0] * 14, y2 - u[1] * 14
        el = f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='{sw}' stroke-linecap='round'{d}/>"
        (self.top if arrow else self.parts).append(el)
        if arrow:
            self.head(x2 + u[0] * 14, y2 + u[1] * 14, u, color, size=19)

    def head(self, x, y, u, color, size=12):
        """Filled triangle with its tip at (x, y) pointing along ``u``."""
        n = np.array([-u[1], u[0]])
        b = np.array([x, y]) - u * size
        p1, p2 = b + n * size * 0.45, b - n * size * 0.45
        self.top.append(
            f"<path d='M {x:.1f} {y:.1f} L {p1[0]:.1f} {p1[1]:.1f} L {p2[0]:.1f} {p2[1]:.1f} z' fill='{color}' stroke='{color}' stroke-width='1.5' stroke-linejoin='round'/>"
        )

    def arrow(self, x1, y, x2, color=MUTED):
        self.line(x1, y, x2, y, color=color, sw=4.2, arrow=True)

    def varrow(self, x, y1, y2, color=MUTED):
        self.line(x, y1, x, y2, color=color, sw=4.2, arrow=True)

    def elbow(self, pts, color, r=16, sw=4.6):
        """Polyline through ``pts`` with rounded corners and an arrowhead at the end."""
        pts = [np.array(q, float) for q in pts]
        u = pts[-1] - pts[-2]
        u /= np.linalg.norm(u)
        end = pts[-1] - u * 18
        d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"
        for p0, p1, p2 in zip(pts, pts[1:], pts[2:]):
            a = p1 - (p1 - p0) / np.linalg.norm(p1 - p0) * r
            b = p1 + (p2 - p1) / np.linalg.norm(p2 - p1) * r
            d += f" L {a[0]:.1f} {a[1]:.1f} Q {p1[0]:.1f} {p1[1]:.1f} {b[0]:.1f} {b[1]:.1f}"
        d += f" L {end[0]:.1f} {end[1]:.1f}"
        self.top.append(
            f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{sw}' stroke-linecap='round' stroke-linejoin='round'/>"
        )
        self.head(pts[-1][0], pts[-1][1], u, color, size=22)

    def render(self, height):
        svgs = tex_svgs([t[2] for t in self.texs])
        body = "\n".join(self.parts + self.top)
        for i, ((x, y, _, size, color, anchor), svg) in enumerate(zip(self.texs, svgs)):
            body = body.replace(f"@@MATH{i}@@", place_math(svg, x, y, size, color, anchor))
        defs = ""
        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{height}' viewBox='0 0 {W} {height}'>"
            f"{defs}<rect width='100%' height='100%' fill='white'/>\n{body}\n</svg>"
        )


def place_math(svg, x, y, size, color, anchor):
    ex = size * 0.442
    w = float(re.search(r'width="([\d.]+)ex"', svg).group(1)) * ex
    h = float(re.search(r'height="([\d.]+)ex"', svg).group(1)) * ex
    va = re.search(r"vertical-align: (-?[\d.]+)ex", svg)
    drop = -float(va.group(1)) * ex if va else 0.0
    x0 = x - w / 2 if anchor == "middle" else (x - w if anchor == "end" else x)
    inner = re.sub(r"^<svg[^>]*>", "", svg)[: -len("</svg>")]
    vb = re.search(r'viewBox="([^"]+)"', svg).group(1)
    return f"<svg x='{x0:.1f}' y='{y - h + drop:.1f}' width='{w:.1f}' height='{h:.1f}' viewBox='{vb}' color='{color}' style='color:{color}' overflow='visible'>{inner}</svg>"


def cmap(v):
    # diverging blue - white - red for a value in [-1, 1]
    stops = [
        (-1.0, (37, 78, 168)),
        (-0.5, (116, 158, 214)),
        (0.0, (247, 247, 245)),
        (0.5, (229, 135, 104)),
        (1.0, (170, 32, 42)),
    ]
    v = float(np.clip(v, -1, 1))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            c = [round(ca[i] + t * (cb[i] - ca[i])) for i in range(3)]
            return f"rgb({c[0]},{c[1]},{c[2]})"
    return "rgb(170,32,42)"


def heat(cv, x, y, a, cell, scale=None, r=2, gap=1.5, cell_h=None):
    """``a`` (rows, cols) as rounded cells; returns the drawn width, height."""
    s = scale or (np.abs(a).max() + 1e-12)
    ch = cell_h or cell
    rows, cols = a.shape
    for i in range(rows):
        for j in range(cols):
            cv.rect(x + j * cell, y + i * ch, cell - gap, ch - gap, cmap(a[i, j] / s), r=r)
    return cols * cell - gap, rows * ch - gap


def dots(cv, x, y, a, cell, scale=None):
    """``a`` (rows, cols) as points (cell-centred samples)."""
    s = scale or (np.abs(a).max() + 1e-12)
    rows, cols = a.shape
    for i in range(rows):
        for j in range(cols):
            cx, cy = x + (j + 0.5) * cell, y + (i + 0.5) * cell
            cv.add(
                f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{cell * 0.36:.1f}' fill='{cmap(a[i, j] / s)}' stroke='#94a3b8' stroke-width='0.8'/>"
            )
    return cols * cell, rows * cell


def card(cv, y, h, key, title, subtitle, dash=None):
    color, _ = ACCENT[key]
    fill, stroke = TINT[key]
    cv.rect(M, y, CW, h, fill, stroke=stroke, r=20, sw=1.4, dash=dash)
    cv.rect(M, y, 9, h, color, r=4)
    cv.text(M + 36, y + 56, title, size=36, weight=700)
    if subtitle:
        cv.text(M + CW - 34, y + 52, subtitle, size=21, color=MUTED, anchor="end")
    return color


def label(cv, x, y, tex, size=22, color=MUTED):
    cv.math(x, y, tex, size=size, color=color, anchor="middle")


def group(cv, x, y, maps, c, scale=None, gap=14):
    """Maps side by side; returns the drawn width."""
    w = 0
    for a in maps:
        heat(cv, x + w, y, a, c, scale=scale)
        w += a.shape[1] * c + gap
    return w - gap


def group_width(n, cols, c, gap=14):
    return n * cols * c + (n - 1) * gap


def vstrip(cv, x, y, v, w, h, n=16):
    """A vector as a column of ``n`` binned cells."""
    v = np.asarray(v).ravel()
    vals = np.array([b[np.argmax(np.abs(b))] for b in np.array_split(v, n)])
    s = np.abs(vals).max() + 1e-12
    ch = h / n
    for i, val in enumerate(vals):
        cv.rect(x, y + i * ch, w, ch + 0.3, cmap(val / s))
    cv.rect(x, y, w, h, "none", stroke="#94a3b8", r=3, sw=1)


def matrix(cv, x, y, w, h, color, light, tex):
    cv.rect(x, y, w, h, light, stroke=color, r=8, sw=1.6)
    for f in (1 / 3, 2 / 3):
        cv.line(x + 6, y + h * f, x + w - 6, y + h * f, color=color, sw=0.6)
        cv.line(x + w * f, y + 6, x + w * f, y + h - 6, color=color, sw=0.6)
    cv.math(x + w / 2, y + h / 2 + 11, tex, size=30, color=color, anchor="middle")


def main(data, out):
    d = np.load(data)
    cv = Canvas()
    plane, (_, ps, px, _) = d["plane"], d["patch"]
    tok, phi, k_maps, psi = d["token"], d["phi_maps"], d["k_maps"], d["psi_maps"]
    order = np.argsort(-k_maps.reshape(len(k_maps), -1).std(1))
    order_psi = np.argsort(-psi.reshape(len(psi), -1).std(1))
    cm = tuple(int(v) for v in d["code_modes"])
    n_k = int(np.prod(cm))
    rank, n_c, dim = int(d["rank"]), int(d["channels"]), int(d["token_dim"])
    bands = [int(b) for b in d["bands"]]
    filter_name, filter_shape = "DCT filter", f"{int(np.prod(bands))} modes → {rank}"
    width = n_k * n_c * rank
    sx = [(ks * cm[2] + kxm) * cm[3] for ks in range(cm[1]) for kxm in range(cm[2])]
    big = 26
    cx = M + CW / 2
    enc_c, dec_c, basis_c = ACCENT["enc"][0], ACCENT["dec"][0], ACCENT["basis"][0]
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

    # 2: coordinate basis; decoder row on top so the connectors do not cross
    h2 = 500
    card(
        cv,
        y,
        h2,
        "basis",
        "Coordinate basis",
        "functions on the patch, from the point coordinates only",
    )
    c = 14
    n_w = group_width(2, px, c)
    row_w = 330 + 60 + 150 + 50 + n_w + 70 + n_w + 70 + n_w
    x0 = cx - row_w / 2
    rows = [
        (
            y + 140,
            order_psi,
            psi,
            r"\psi_r(p)",
            r"\varphi_k\psi_r",
            "dec",
            f"{filter_name} (dec)",
            filter_shape,
            f"synthesis basis: {n_k} × {rank}",
        ),
        (
            y + 320,
            order,
            k_maps,
            r"K_r(p)",
            r"b_{kr}",
            "enc",
            f"{filter_name} (enc)",
            filter_shape,
            f"projection basis: {n_k} × {rank}",
        ),
    ]
    pmid = (rows[0][0] + rows[1][0]) / 2 + ps * c / 2
    cv.math(x0, pmid - 18, r"p", size=34)
    cv.math(x0, pmid + 22, r"(u_s, u_x, u_y, v_\parallel, \mu)", size=20, color=MUTED)
    cv.math(x0, pmid + 66, r"K_r = \sum_m A_{mr}\, \Phi_m(p)", size=20, color=INK)
    cv.text(x0, pmid + 94, "Φ_m: DCT modes of the", size=16, color=MUTED)
    cv.text(x0, pmid + 115, "window coordinates", size=16, color=MUTED)
    cv.text(x0, pmid + 136, "·".join(map(str, bands)) + " modes", size=16, color=MUTED)
    outs = {}
    for ry, idx, maps, tex, out_tex, key, mlp_name, mlp_shape, basis_name in rows:
        mid = ry + ps * c / 2
        x = x0 + 330
        cv.elbow(
            [(x0 + 262, pmid), (x0 + 288, pmid), (x0 + 288, mid), (x + 58, mid)],
            basis_c,
            r=12,
            sw=4.2,
        )
        x += 64
        matrix(cv, x + 20, mid - 34, 80, 68, basis_c, ACCENT["basis"][1], r"A")
        cv.text(x + 60, ry - 40, mlp_name, size=21, weight=600, anchor="middle")
        cv.text(x + 60, ry - 16, mlp_shape, size=18, color=MUTED, anchor="middle")
        x += 140
        cv.arrow(x, mid, x + 40, color=basis_c)
        x += 50
        group(cv, x, ry, [maps[idx[0]], maps[idx[1]]], c)
        label(cv, x + n_w / 2, ry + ps * c + 34, tex, size=24, color=INK)
        x += n_w + 20
        cv.math(x + 6, mid + 11, r"\times", size=28, color=basis_c)
        x += 50
        group(cv, x, ry, [phi[sx[1]], phi[sx[3]]], c, scale=1.0)
        label(cv, x + n_w / 2, ry + ps * c + 34, r"\varphi_k(p)", size=24, color=INK)
        cv.text(x + n_w / 2, ry - 16, "fixed cosines", size=17, color=MUTED, anchor="middle")
        x += n_w + 20
        cv.math(x + 6, mid + 11, r"=", size=28, color=basis_c)
        x += 50
        group(cv, x, ry, [phi[sx[1]] * maps[idx[0]], phi[sx[3]] * maps[idx[1]]], c)
        label(cv, x + n_w / 2, ry + ps * c + 34, out_tex, size=24, color=INK)
        cv.text(
            x + n_w / 2, ry - 16, basis_name.split(":")[0], size=17, color=MUTED, anchor="middle"
        )
        cv.text(
            x + n_w / 2,
            ry + ps * c + 64,
            basis_name.split(": ")[1],
            size=17,
            color=MUTED,
            anchor="middle",
        )
        outs[key] = (x + n_w, mid)
    cv.text(
        cx,
        y + h2 - 22,
        "computed per point, one basis for every channel: the same weights on every grid",
        size=20,
        color=MUTED,
        anchor="middle",
    )
    y2_end = y + h2
    y += h2 + 44

    # 3: encoder
    h3 = 270
    card(cv, y, h3, "enc", "Encoder", "a linear combination of the patch values, per channel c")
    c = 22
    ey = y + 110
    mid = ey + ps * c / 2
    row_w = px * c + 40 + px * c + 60 + 360 + 50 + 130 + 50 + 40
    x = cx - row_w / 2
    dots(cv, x, ey, d["truth"], c)
    label(cv, x + px * c / 2, ey + ps * c + 36, r"x_{pc}", size=26, color=INK)
    cv.math(x + px * c + 10, mid + 10, r"\times", size=28, color=enc_c)
    bx = x + px * c + 40
    heat(cv, bx, ey, phi[sx[1]] * k_maps[order[0]], c)
    label(cv, bx + px * c / 2, ey + ps * c + 36, r"b_{kr}(p)", size=26, color=INK)
    enc_in = (bx + px * c / 2, ey - 6)
    x = bx + px * c + 14
    cv.arrow(x, mid, x + 40, color=enc_c)
    x += 56
    cv.math(x, mid + 12, r"h_{kcr} = \sum_p w_p\, b_{kr}(p)\, x_{pc}", size=28)
    cv.text(
        x + 150,
        mid + 60,
        f"{n_k} × {n_c} × {rank} = {width} projections",
        size=19,
        color=MUTED,
        anchor="middle",
    )
    x += 360
    cv.arrow(x, mid, x + 40, color=enc_c)
    x += 50
    matrix(cv, x, mid - 50, 110, 100, enc_c, ACCENT["enc"][1], r"W")
    cv.text(x + 55, mid + 78, f"{width} → {dim}", size=19, color=MUTED, anchor="middle")
    x += 120
    cv.arrow(x, mid, x + 40, color=enc_c)
    x += 50
    vstrip(cv, x, mid - 60, d["z"], 26, 120)
    label(cv, x + 13, mid + 98, r"z", size=26, color=INK)
    enc_out = (x + 13, mid + 60)
    y += h3 + 30

    # backbone: a small block under the token
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
    picks = [(k, int(np.argmax(energy[k]))) for k in (sx[0], sx[1], sx[3])]
    scale = np.abs(d["truth"]).max()
    terms_w = 3 * px * c + 2 * 52
    row_w = 26 + 50 + 110 + 50 + 26 + 60 + terms_w + 110 + 44 + px * c
    x = cx - row_w / 2
    vstrip(cv, x, mid - 60, d["z"], 26, 120)
    label(cv, x + 13, mid + 98, r"z", size=26, color=INK)
    dec_zmid = (x, mid)
    x += 36
    cv.arrow(x, mid, x + 40, color=dec_c)
    x += 50
    matrix(cv, x, mid - 50, 110, 100, dec_c, ACCENT["dec"][1], r"B")
    cv.text(x + 55, mid + 78, f"{dim} → {width}", size=19, color=MUTED, anchor="middle")
    x += 120
    cv.arrow(x, mid, x + 40, color=dec_c)
    x += 50
    vstrip(cv, x, mid - 60, d["codes"], 26, 120)
    label(cv, x + 13, mid + 98, r"c_{kcr}", size=26, color=INK)
    x += 36
    cv.arrow(x, mid, x + 40, color=dec_c)
    x += 56
    t0 = x
    for i, (k, r) in enumerate(picks):
        heat(cv, x, dy, terms[k, r], c, scale=scale)
        x += px * c + 10
        if i < len(picks) - 1:
            cv.math(x + 6, mid + 12, r"+", size=30, color=dec_c)
            x += 42
    cv.math(x + 6, mid + 12, r"+\cdots", size=30, color=dec_c)
    x += 110
    cv.math(x, mid + 12, r"=", size=30, color=dec_c)
    x += 44
    dots(cv, x, dy, d["recon"], c, scale=scale)
    label(cv, x + px * c / 2, dy + ps * c + 36, r"u_{pc}", size=26, color=INK)
    label(
        cv,
        t0 + terms_w / 2,
        dy + ps * c + 40,
        r"u_{pc} = \sum_{k,r} c_{kcr}\, \varphi_k(p)\,\psi_r(p)",
        size=26,
        color=INK,
    )
    dec_in = (t0 + terms_w / 2, dy - 6)
    cv.text(
        cx,
        y + h4 - 22,
        "the patch as a linear combination of basis functions",
        size=22,
        color=MUTED,
        anchor="middle",
    )
    y += h4 + 34

    # connectors
    lane1, lane2 = M + CW + 22, M + CW + 50
    cv.elbow(
        [
            (outs["enc"][0] + 10, outs["enc"][1]),
            (lane1, outs["enc"][1]),
            (lane1, y2_end + 22),
            (enc_in[0], y2_end + 22),
            enc_in,
        ],
        enc_c,
    )
    cv.elbow(
        [
            (outs["dec"][0] + 10, outs["dec"][1]),
            (lane2, outs["dec"][1]),
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
    print(out, W, y)


if __name__ == "__main__":
    main(*sys.argv[1:])
