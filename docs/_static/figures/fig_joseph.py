"""Joseph's method: sampling planes along the principal axis and bilinear interpolation.

(a) one transaxial slice: a LOR whose principal axis is x is sampled once in every image
    plane x = const (through the voxel centers); cf = length of the LOR between two planes.
(b) inside one plane: bilinear interpolation between the 4 voxel centers around the
    intersection point (the back projection adds with the same 4 weights).
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

import ppstyle as st

NX, NY = 9, 6
P0 = np.array([-1.2, 0.30])
P1 = np.array([9.3, 4.18])
SLOPE = (P1[1] - P0[1]) / (P1[0] - P0[0])
K = 4  # highlighted plane


def y_at(x):
    return P0[1] + (x - P0[0]) * SLOPE


fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1.75, 1]))

# ---------------------------------------------------------------- (a) sampling planes
for i in range(NX):
    for j in range(NY):
        ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fc="white", ec=st.MID, lw=0.8))
# the two voxels interpolated in plane K
yk = y_at(K)
j0 = int(np.floor(yk))
for j in (j0, j0 + 1):
    ax.add_patch(Rectangle((K - 0.5, j - 0.5), 1, 1, fc=st.RED, alpha=0.18, ec="none"))
# voxel centers
gx, gy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
ax.plot(gx.ravel(), gy.ravel(), "o", color=st.MID, ms=2.5, zorder=2)
# sampling planes through the voxel centers
for i in range(NX):
    ax.plot([i, i], [-0.5, NY - 0.5], ls=(0, (3, 3)), color=st.GRAY, lw=0.8, zorder=1)
# the LOR and its samples
ax.plot(*np.array([P0, P1]).T, color=st.BLUE, lw=2.2, zorder=3)
xs = np.arange(NX)
ax.plot(xs, y_at(xs), "o", color=st.BLUE, ms=6, mec="white", mew=1, zorder=4)
ax.plot(K, yk, "o", color=st.RED, ms=8, mec="white", mew=1.2, zorder=5)
for p, lab in [(P0, "lor_start"), (P1, "lor_end")]:
    ax.plot(*p, "s", color=st.GRAY, ms=6, zorder=4)
ax.text(P0[0] - 0.1, P0[1] - 0.45, "lor_start", ha="center", va="top", fontsize=9, color=st.GRAY)
ax.text(P1[0] + 0.1, P1[1] + 0.35, "lor_end", ha="center", va="bottom", fontsize=9, color=st.GRAY)
# cf between plane K and K + 1
a, b = np.array([K, yk]), np.array([K + 1, y_at(K + 1)])
nrm = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
off = 0.42 * nrm
ax.add_patch(FancyArrowPatch(a + off, b + off, arrowstyle="<->", mutation_scale=9, color=st.RED, lw=1.2))
ax.text(*(0.5 * (a + b) + 2.2 * off), "cf", color=st.RED, ha="center", va="center", fontsize=11,
        fontstyle="italic", fontweight="bold")
# principal axis arrow
ax.add_patch(FancyArrowPatch((-0.5, -1.7), (2.0, -1.7), arrowstyle="-|>", mutation_scale=12, color=st.GRAY, lw=1.4))
ax.text(2.15, -1.7, "principal axis x (the LOR changes most along x)", va="center", fontsize=9, color=st.GRAY)
ax.text(NX - 0.5, NY - 0.35, "planes x = const through the voxel centers", ha="right", va="bottom",
        fontsize=9, color=st.GRAY)
ax.annotate("sample in plane k: interpolate between the 2 nearest voxels (4 in 3D)", (K, yk - 0.12),
            (K + 0.6, -1.0), fontsize=9, color=st.RED, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=st.RED, lw=1, connectionstyle="arc3,rad=0.25"))
ax.set_xlim(-1.9, NX + 0.6)
ax.set_ylim(-2.0, NY + 0.1)
st.clean_axes(ax)
ax.set_title("(a) one sample per image plane along the principal axis", loc="left")

# ---------------------------------------------------------------- (b) bilinear interpolation
wa, wb = 0.32, 0.62  # fractional position of the intersection point between the 4 voxel centers
for i in range(2):
    for j in range(2):
        bx.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fc="white", ec=st.MID, lw=0.8))
corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
weights = [(1 - wa) * (1 - wb), wa * (1 - wb), (1 - wa) * wb, wa * wb]
labels = ["(1−α)(1−β)", "α(1−β)", "(1−α)β", "αβ"]
for (cx, cy), w, lab in zip(corners, weights, labels):
    bx.plot([wa, cx], [wb, cy], color=st.RED, lw=0.6 + 3.5 * w, alpha=0.75, zorder=2)
    bx.plot(cx, cy, "o", color=st.GRAY, ms=7, zorder=3)
    ty = cy + (-0.2 if cy == 0 else 0.2)
    bx.text(cx, ty, f"{lab}\n= {w:.2f}", ha="center", va="top" if cy == 0 else "bottom", fontsize=8.5,
            color="#333333")
bx.plot(wa, wb, "o", color=st.RED, ms=9, mec="white", mew=1.2, zorder=4)
# fractional distances
bx.annotate("", (wa, -0.62), (0, -0.62), arrowprops=dict(arrowstyle="<->", color=st.GRAY, lw=1))
bx.text(wa / 2, -0.7, "α", ha="center", va="top", fontsize=11, color=st.GRAY)
bx.annotate("", (1.62, wb), (1.62, 0), arrowprops=dict(arrowstyle="<->", color=st.GRAY, lw=1))
bx.text(1.7, wb / 2, "β", ha="left", va="center", fontsize=11, color=st.GRAY)
bx.text(0.5, -1.05, "contribution of plane k = cf × Σ weight × voxel value\n"
        "back projection: add cf × value with the same 4 weights",
        ha="center", va="top", fontsize=9, color="#333333")
bx.set_xlim(-0.8, 2.0)
bx.set_ylim(-1.55, 1.8)
st.clean_axes(bx)
bx.set_title("(b) inside plane k: bilinear interpolation", loc="left")

fig.tight_layout()
st.save(fig, "joseph", png_preview_dir=sys.argv[1] if len(sys.argv) > 1 else None)
