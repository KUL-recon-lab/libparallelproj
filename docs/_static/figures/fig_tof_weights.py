"""TOF along a LOR: TOF bins and the TOF weights of one ray sample.

(a) the LOR with its TOF bins (numbered towards lor_end, centered on the LOR center plus the
    TOF center offset) and the truncated Gaussian TOF kernel around one ray sample s_k; the
    weight of a TOF bin is the area of the kernel inside the bin (bars).
(b) the same with the cumulative kernel F: weight of bin t = F(upper edge) - F(lower edge);
    neighbouring bins share an edge (one evaluation of F per edge).
"""

import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

import ppstyle as st

L = 240.0  # LOR length (mm), s from -L/2 (lor_start) to +L/2 (lor_end)
W = 20.0  # TOF bin width
NB = 11  # number of TOF bins
OFF = 6.0  # TOF center offset
SIG = 15.0  # TOF sigma
NS = 3.0  # num_sigmas
SK = 58.0  # position of the ray sample

centers = (np.arange(NB) - 0.5 * (NB - 1)) * W + OFF
edges = np.concatenate([centers - W / 2, [centers[-1] + W / 2]])


def F(s):
    """truncated, normalized cumulative kernel around SK"""
    u = np.clip((np.asarray(s, float) - SK) / SIG, -NS, NS)
    e = np.vectorize(math.erf)(u / math.sqrt(2))
    return 0.5 * (e / math.erf(NS / math.sqrt(2)) + 1)


weights = F(edges[1:]) - F(edges[:-1])

fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))

# ---------------------------------------------------------------- (a)
ybase = 0.0
for t in range(NB):
    ax.add_patch(Rectangle((edges[t], ybase - 0.075), W, 0.075, fc=st.LIGHT if t % 2 else "white", ec=st.MID, lw=0.8))
    ax.text(centers[t], ybase - 0.085, str(t), ha="center", va="top", fontsize=8.5, color=st.GRAY)
ax.text(edges[0] - 4, ybase - 0.085, "TOF bin", ha="right", va="top", fontsize=8.5, color=st.GRAY)
# the LOR
ax.plot([-L / 2, L / 2], [ybase, ybase], color=st.BLUE, lw=2.4, zorder=3)
for x, lab, ha in [(-L / 2, "lor_start", "right"), (L / 2, "lor_end", "left")]:
    ax.plot(x, ybase, "s", color=st.GRAY, ms=7, zorder=4)
    ax.text(x + (-6 if ha == "right" else 6), ybase, lab, ha=ha, va="center", fontsize=9, color=st.GRAY)
# LOR center and offset of the central bin
c_mid = centers[NB // 2]
ax.plot([0, 0], [ybase, ybase + 0.19], color=st.GRAY, lw=1, ls=(0, (2, 2)))
ax.text(0, ybase + 0.2, "LOR center", ha="center", va="bottom", fontsize=8.5, color=st.GRAY)
ax.add_patch(FancyArrowPatch((0, ybase + 0.15), (c_mid, ybase + 0.15), arrowstyle="-|>", mutation_scale=8,
                             color=st.PURPLE, lw=1.2))
ax.text(c_mid + 2, ybase + 0.15, "tof_center_offset", ha="left", va="center", fontsize=8.5, color=st.PURPLE)
ax.add_patch(FancyArrowPatch((-L / 2 + 10, ybase - 0.2), (-L / 2 + 70, ybase - 0.2), arrowstyle="-|>",
                             mutation_scale=10, color=st.GRAY, lw=1.1))
ax.text(-L / 2 + 74, ybase - 0.2, "bin numbers increase towards lor_end", va="center", fontsize=8.5, color=st.GRAY)
# truncated kernel around the sample and the weights as areas
s = np.linspace(SK - NS * SIG, SK + NS * SIG, 600)
pdf = np.exp(-0.5 * ((s - SK) / SIG) ** 2)
pdf = 0.62 * pdf / pdf.max()
for t in range(NB):
    m = (s >= edges[t]) & (s <= edges[t + 1])
    if m.any():
        ax.fill_between(s[m], ybase + 0.012, ybase + 0.012 + pdf[m], color=st.ORANGE, alpha=0.32 if t % 2 else 0.55,
                        lw=0)
ax.plot(s, ybase + 0.012 + pdf, color=st.ORANGE, lw=1.8)
for x in (SK - NS * SIG, SK + NS * SIG):
    ax.plot([x, x], [ybase, ybase + 0.08], color=st.ORANGE, lw=1.2)
ax.text(SK + NS * SIG + 3, ybase + 0.13, "truncated at\n±num_sigmas · σ", fontsize=8.5, color=st.ORANGE, va="center")
ax.plot(SK, ybase, "o", color=st.RED, ms=9, mec="white", mew=1.3, zorder=5)
ax.text(SK, ybase + 0.68, "TOF kernel of the ray sample sₖ", ha="center", va="bottom", fontsize=9, color=st.ORANGE)
ax.annotate("ray sample sₖ (intersection\nwith image plane k)", (SK, ybase), (SK - 120, ybase + 0.5), fontsize=9,
            color=st.RED, ha="center", arrowprops=dict(arrowstyle="->", color=st.RED, lw=1,
                                                      connectionstyle="arc3,rad=-0.25"))
ax.set_ylim(-0.32, 0.86)
ax.set_xlim(-L / 2 - 38, L / 2 + 38)
ax.axis("off")
ax.set_title("(a) TOF bins of a LOR and the TOF kernel of one ray sample: weight of a bin = area inside the bin",
             loc="left")

# ---------------------------------------------------------------- (b)
ss = np.linspace(edges[0], edges[-1], 800)
bx.plot(ss, F(ss), color=st.ORANGE, lw=2)
for e_ in edges:
    bx.axvline(e_, color=st.MID, lw=0.8, zorder=0)
bx.plot(edges, F(edges), "o", color=st.ORANGE, ms=4.5, zorder=3)
tb = int(np.argmax(weights))  # bin with the largest weight
lo, hi = F(edges[tb]), F(edges[tb + 1])
for y in (lo, hi):
    bx.plot([edges[tb] - 40, edges[tb + 1]], [y, y], color=st.RED, lw=0.8, ls=(0, (3, 2)))
bx.add_patch(FancyArrowPatch((edges[tb] - 36, lo), (edges[tb] - 36, hi), arrowstyle="<->", mutation_scale=9,
                             color=st.RED, lw=1.3))
bx.text(edges[tb] - 40, 0.5 * (lo + hi), f"weight of bin {tb}\n= F(upper edge) − F(lower edge)\n= {hi - lo:.2f}",
        ha="right", va="center", fontsize=9, color=st.RED)
bx.text(edges[-1] + 4, 0.5, "the upper edge of a bin is the\nlower edge of the next bin:\none evaluation of F per edge",
        ha="left", va="center", fontsize=8.5, color=st.GRAY)
bx.set_ylabel("cumulative kernel F")
bx.set_xlabel("position along the LOR [mm]")
bx.set_ylim(-0.05, 1.08)
bx.spines[["top", "right"]].set_visible(False)
bx.set_title("(b) the same with the cumulative kernel F (truncated, normalized: weights sum to 1)", loc="left")

fig.tight_layout()
st.save(fig, "tof_weights", png_preview_dir=sys.argv[1] if len(sys.argv) > 1 else None)
