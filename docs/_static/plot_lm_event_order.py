"""Figures explaining the sorting of TOF listmode events (issue #54).

Creates (small palette PNGs, see save_png)
  lm_event_order.png  (a) TOF center of an event and the image region its projection
                      touches, (b) 3 warps (32 consecutive events each) in random order,
                      (c) the same for events sorted by their sort key
  lm_sort_key.png     (a) Morton (Z-order) code of a 2D grid of cells,
                      (b) bit layout of the 64-bit sort key

2D illustration (one transaxial plane); the library uses the same ideas in 3D.
Run from this directory: python plot_lm_event_order.py
"""

import io

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image

rng = np.random.default_rng(3)

# --- 2D geometry (mm): image of 64 x 64 voxels of 4 mm, scanner radius 150 mm,
# cylinder phantom of radius 100 mm, TOF bins of 15 mm, TOF sigma 15 mm, 3 sigma
NVOX, VOX = 64, 4.0
ORIGIN = -0.5 * (NVOX - 1) * VOX  # center of voxel 0
R_SCANNER, R_PHANTOM = 150.0, 100.0
W, SIGMA, NSIG = 15.0, 15.0, 3.0
NB = int(2 * R_SCANNER / W) | 1  # number of TOF bins (odd)
CELL = 2  # Morton grid cell in voxels
COLORS = ["tab:blue", "tab:orange", "tab:green"]


def save_png(fig, fname, dpi=90, colors=64):
    """save a figure as a PNG with a palette of `colors` colors (~4x smaller files than
    RGB PNGs or SVGs of these figures)"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE).save(fname, optimize=True)


def make_events(n):
    """random LORs (two points on the scanner circle) through the phantom; TOF bin of a
    uniformly distributed emission point inside the phantom plus Gaussian TOF noise"""
    s, e, it = [], [], []
    while len(s) < n:
        p = rng.uniform(-R_PHANTOM, R_PHANTOM, 2)
        if p @ p > R_PHANTOM**2:
            continue
        phi = rng.uniform(0, np.pi)
        d = np.array([np.cos(phi), np.sin(phi)])
        # intersections of the line p + t d with the scanner circle
        b = p @ d
        t = np.sqrt(b**2 - (p @ p - R_SCANNER**2))
        a_, e_ = p + (-b - t) * d, p + (-b + t) * d
        u = (p - 0.5 * (a_ + e_)) @ d + rng.normal(0, SIGMA)  # TOF position rel. to the LOR center
        s.append(a_)
        e.append(e_)
        it.append(int(np.clip(np.round(u / W + 0.5 * (NB - 1)), 0, NB - 1)))
    return np.array(s), np.array(e), np.array(it)


def tof_center(s, e, it):
    """center of TOF bin it on the LOR from s to e (as in the projectors)"""
    d = (e - s) / np.linalg.norm(e - s, axis=-1, keepdims=True)
    return 0.5 * (s + e) + ((it - 0.5 * (NB - 1)) * W)[..., None] * d


def spread2(x):
    """insert a zero bit between the bits of x (2D Morton code)"""
    x = x & 0xFFFF
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x


def sort_key(s, e, it):
    """2D version of the sort key: principal axis, then Morton code of the TOF center cell"""
    d = np.abs(e - s)
    axis = (d[:, 1] > d[:, 0]).astype(np.int64)
    q = np.clip(np.floor((tof_center(s, e, it) - (ORIGIN - 0.5 * VOX)) / (CELL * VOX)), 0, 1023).astype(np.int64)
    return (axis << 40) | spread2(q[:, 0]) | (spread2(q[:, 1]) << 1)


def draw_image(ax, title):
    ax.add_patch(Circle((0, 0), R_SCANNER, fill=False, lw=1.5, color="0.3"))
    ax.add_patch(Circle((0, 0), R_PHANTOM, color="0.92", zorder=0))
    lo = ORIGIN - 0.5 * VOX
    ax.add_patch(Rectangle((lo, lo), NVOX * VOX, NVOX * VOX, fill=False, lw=0.8, color="0.5"))
    ax.set_xlim(-R_SCANNER - 8, R_SCANNER + 8)
    ax.set_ylim(-R_SCANNER - 8, R_SCANNER + 8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize="medium")


def draw_window(ax, s, e, it, color, lw=2.5, alpha=0.9):
    """the part of the LOR that the projection of the event visits (TOF window)"""
    d = (e - s) / np.linalg.norm(e - s)
    c = tof_center(s, e, np.array(it))
    half = NSIG * SIGMA + 0.5 * W
    ax.plot(*np.array([c - half * d, c + half * d]).T, color=color, lw=lw, alpha=alpha, solid_capstyle="butt")
    ax.plot(*c, "o", color=color, ms=2.5)


s, e, it = make_events(30000)
key = sort_key(s, e, it)
order = np.argsort(key, kind="stable")

# ------------------------------------------------------------------ figure 1
fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.4), layout="constrained")

# (a) one event: LOR, TOF bins, TOF center, TOF window, grid cell of the TOF center
ax = axs[0]
draw_image(ax, "(a) one event: TOF center and TOF window")
k = 7
d = (e[k] - s[k]) / np.linalg.norm(e[k] - s[k])
ax.plot(*np.array([s[k], e[k]]).T, color="0.4", lw=0.8)
mid = 0.5 * (s[k] + e[k])
for t in range(NB + 1):  # TOF bin edges
    pt = mid + (t - 0.5 * NB) * W * d
    nrm = np.array([-d[1], d[0]])
    ax.plot(*np.array([pt - 3 * nrm, pt + 3 * nrm]).T, color="0.4", lw=0.6)
draw_window(ax, s[k], e[k], it[k], "tab:red", lw=4, alpha=0.5)
c = tof_center(s[k], e[k], np.array(it[k]))
cell_lo = np.floor((c - (ORIGIN - 0.5 * VOX)) / (CELL * VOX)) * CELL * VOX + (ORIGIN - 0.5 * VOX)
ax.add_patch(Rectangle(cell_lo, CELL * VOX, CELL * VOX, color="tab:red", alpha=0.6))
ax.annotate("TOF center c\n(center of the event's TOF bin)", c, c + np.array([25, 55]), fontsize="small",
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("planes visited by the event\n(TOF window: $\\pm$(3$\\sigma$ + bin/2))", c - 40 * d,
            np.array([10, -95]), fontsize="small", arrowprops=dict(arrowstyle="->", lw=0.8))
ax.plot(*s[k], "s", color="0.3", ms=4)
ax.plot(*e[k], "s", color="0.3", ms=4)
ax.text(-R_SCANNER, R_SCANNER, "small ticks: TOF bins\nred square: grid cell (2 x 2 voxels)\ncontaining c",
        fontsize="x-small", va="top")

# (b) random order: 3 warps of 32 consecutive events
ax = axs[1]
draw_image(ax, "(b) random order: 3 warps of 32 events\n(events of a warp spread over the image)")
for g, col in enumerate(COLORS):
    for j in range(32 * g, 32 * (g + 1)):
        draw_window(ax, s[j], e[j], it[j], col, lw=1.5, alpha=0.8)

# (c) sorted by the sort key: 3 warps of 32 consecutive events
ax = axs[2]
draw_image(ax, "(c) sorted by sort key: 3 warps of 32 events\n(events of a warp touch the same image region)")
for g, (start, col) in enumerate(zip([4000, 12000, 21000], COLORS)):
    for j in order[start:start + 32]:
        draw_window(ax, s[j], e[j], it[j], col, lw=1.5, alpha=0.8)
save_png(fig, "lm_event_order.png")

# ------------------------------------------------------------------ figure 2
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1.5]))

# (a) Morton code on an 8 x 8 grid of cells
ax = axs[0]
n = 8
xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
codes = spread2(xs) | (spread2(ys) << 1)
path = np.array([(x, y) for _, x, y in sorted(zip(codes.ravel(), xs.ravel(), ys.ravel()))])
ax.plot(path[:, 0], path[:, 1], "-", color="tab:blue", lw=1.2, zorder=1)
for x in range(n):
    for y in range(n):
        hl = (x, y) == (5, 3)
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=hl, color="tab:red" if hl else "0.6",
                               alpha=0.35 if hl else 1, lw=0.5))
        ax.text(x, y, str(codes[x, y]), ha="center", va="center", fontsize="small", zorder=2,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
ax.set_xlim(-0.6, n - 0.4)
ax.set_ylim(-0.6, n - 0.4)
ax.set_aspect("equal")
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xlabel("cell x")
ax.set_ylabel("cell y")
ax.set_title("(a) Morton (Z-order) code of grid cells:\nfollowing the numbers = following the blue curve",
             fontsize="medium")
ax.text(0.5, -0.24, "cell (x, y) = (5, 3):  x = 101$_2$, y = 011$_2$\n"
        "interleave bits y$_2$x$_2$ y$_1$x$_1$ y$_0$x$_0$ = 01 10 11$_2$ = 27\n"
        "close numbers $\\Rightarrow$ close cells (mostly)", transform=ax.transAxes, ha="center", va="top",
        fontsize="small")

# (b) bit layout of the 64-bit sort key (3D)
ax = axs[1]
ax.set_xlim(-1, 65)
ax.set_ylim(-3.0, 2.2)
ax.axis("off")
ax.set_title("(b) 64-bit sort key of an event (3D)", fontsize="medium")
fields = [(63, 42, "0", "0.9"), (41, 40, "axis", "tab:orange"),
          (39, 30, "0", "0.9"), (29, 0, "Morton code of the grid cell of the TOF center\n"
                                    "(x, y, z: 10 bits each, interleaved z y x z y x ...)", "tab:blue")]
for hi, lo, label, col in fields:
    x0 = 63 - hi
    ax.add_patch(Rectangle((x0, 0), hi - lo + 1, 1.2, color=col, alpha=0.5, ec="k", lw=0.8))
    ax.text(x0 + 0.5 * (hi - lo + 1), 0.6, label, ha="center", va="center", fontsize="small")
    ax.text(x0 + 0.5, 1.3, str(hi), ha="center", va="bottom", fontsize="x-small")
    ax.text(x0 + hi - lo + 0.5, -0.1, str(lo), ha="center", va="top", fontsize="x-small")
ax.text(32, -0.8,
        "axis: principal axis of the LOR (0, 1 or 2)\n"
        "sorting by this number = grouping the events by principal axis (all threads of a warp step\n"
        "through the image along the same axis), then ordering them along the Z-order curve of their\n"
        "TOF centers (threads of a warp read / write the same image region at the same time).\n"
        "TOF center: center of the event's TOF bin on its LOR; grid cell: 2 x 2 x 2 voxels.\n"
        "The events are sorted once (e.g. per subset) and reused in all iterations.",
        ha="center", va="top", fontsize="small")
save_png(fig, "lm_sort_key.png")
