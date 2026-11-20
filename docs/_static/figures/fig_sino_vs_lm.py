"""TOF sinogram vs. TOF listmode projection of one LOR.

(a) TOF sinogram: one pass over all image planes of the LOR computes all TOF bins.
(b) TOF listmode: an event is a LOR plus one TOF bin; only the planes whose TOF kernel
    overlaps this bin are visited, the result is one value (identical to that TOF bin of the
    sinogram projection).
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import ppstyle as st

NPL = 30  # image planes crossed by the LOR
BW = 3.0  # TOF bin width in plane spacings
NB = 10
HALF = 1.6  # half width of the kernel support in TOF bins
TB = 6  # TOF bin of the listmode event

edges = np.arange(NB + 1) * BW - 0.5


def draw(ax, mode):
    # image planes and the LOR
    ax.add_patch(Rectangle((-0.5, -0.9), NPL, 1.8, fc=st.LIGHT, ec="none"))
    for p in range(NPL):
        ax.plot([p, p], [-0.9, 0.9], color=st.MID, lw=0.8)
    ax.plot([-2.5, NPL + 1.5], [0, 0], color=st.BLUE, lw=2.2, zorder=2)
    ax.text(-2.7, 0, "LOR", ha="right", va="center", fontsize=9, color=st.BLUE)
    if mode == "sino":
        ax.text(NPL / 2 - 0.5, 1.05, "image planes along the principal axis: all are visited", ha="center",
                va="bottom", fontsize=8.5, color=st.GRAY)
    # TOF bins
    yb = -1.9
    it_f = (np.arange(NPL) - (BW - 1) / 2) / BW  # plane p lies in bin round(it_f)
    visited = np.ones(NPL, bool)
    if mode == "lm":
        visited = (it_f > TB - 0.5 - HALF) & (it_f < TB + 0.5 + HALF)
    for t in range(NB):
        if mode == "sino":
            fc, ec = "#F6D2B8", st.ORANGE
        else:
            fc, ec = ("#F6D2B8", st.ORANGE) if t == TB else ("white", st.MID)
        ax.add_patch(Rectangle((edges[t] + 0.08, yb - 0.4), BW - 0.16, 0.8, fc=fc, ec=ec, lw=1.0))
        ax.text(edges[t] + BW / 2, yb, str(t), ha="center", va="center", fontsize=9, color="#333333")
    ax.text(-2.7, yb, "TOF bins", ha="right", va="center", fontsize=9, color=st.GRAY)
    # samples
    for p in range(NPL):
        if visited[p]:
            ax.plot(p, 0, "o", color=st.RED if mode == "lm" else st.BLUE, ms=6.5, mec="white", mew=1, zorder=3)
        else:
            ax.plot(p, 0, "o", color="white", ms=5, mec=st.MID, mew=1, zorder=3)
    if mode == "lm":
        p_lo, p_hi = np.where(visited)[0][[0, -1]]
        ax.annotate("", (p_lo - 0.4, 1.35), (p_hi + 0.4, 1.35), arrowprops=dict(arrowstyle="<->", color=st.RED, lw=1.2))
        ax.text(0.5 * (p_lo + p_hi), 1.45, "visited planes: TOF kernel overlaps TOF bin 6 of the event", ha="center",
                va="bottom", fontsize=8.5, color=st.RED)
    ax.set_xlim(-6.0, NPL + 2.0)
    ax.set_ylim(-2.6, 2.0)
    st.clean_axes(ax)
    ax.set_aspect("auto")


fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.5, 5.6))
draw(ax, "sino")
ax.set_title("(a) TOF sinogram: one pass over all planes of the LOR computes all TOF bins", loc="left")
draw(bx, "lm")
bx.set_title("(b) TOF listmode: event = LOR + TOF bin; only the planes around this bin, one value per event", loc="left")
fig.tight_layout(h_pad=1.5)
st.save(fig, "sino_vs_lm", png_preview_dir=sys.argv[1] if len(sys.argv) > 1 else None)
