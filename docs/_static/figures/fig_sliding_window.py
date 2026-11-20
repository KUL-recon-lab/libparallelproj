"""TOF sinogram forward projection: the sliding accumulator in registers.

Rows: the image planes of one LOR, visited in the order of increasing TOF position.
Columns: the TOF bins of the LOR. For every plane, the TOF window (bins overlapping the
kernel support) gets a contribution (orange). The accumulator (W bins in registers, blue
frame) moves forward with the window; a bin that leaves the accumulator is final and is
written to memory once (green). The back projection is the mirror image (bins are read once
into the register buffer).
"""

import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

import ppstyle as st

NB = 16  # TOF bins
NPL = 7  # planes shown
W = 8  # accumulator size
HALF = 2.4  # half width of the kernel support in TOF bins
IT0, DIT = 2.6, 1.15  # TOF position of the first plane, TOF bins per plane

fig, ax = plt.subplots(figsize=(11, 5.2))

base = None
for k in range(NPL):
    y = NPL - 1 - k
    it_f = IT0 + DIT * k
    lo = int(math.floor(it_f - HALF + 0.5))
    hi = int(math.floor(it_f + HALF + 0.5))
    if base is None:
        base = lo
    written_up_to = base  # bins < base were written before this plane
    while lo >= base + 1:  # move the accumulator forward (store width V = 1)
        base += 1
    for t in range(NB):
        if t < base:
            fc, ec = "#D7EDDF", st.GREEN  # written (final)
        elif lo <= t <= hi:
            fc, ec = "#F6D2B8", st.ORANGE  # window of this plane
        elif t < base + W:
            fc, ec = "white", st.MID  # in the accumulator, no contribution from this plane
        else:
            fc, ec = "white", "#E3E7EC"  # not reached yet
        ax.add_patch(Rectangle((t + 0.06, y + 0.1), 0.88, 0.8, fc=fc, ec=ec, lw=0.9))
        if base <= t < written_up_to + 0 and False:
            pass
    # bins written at this step (they left the accumulator)
    for t in range(written_up_to, base):
        ax.text(t + 0.5, y + 0.5, "✓", ha="center", va="center", fontsize=10, color=st.GREEN, fontweight="bold")
    # accumulator frame
    ax.add_patch(FancyBboxPatch((base + 0.02, y + 0.04), W - 0.04, 0.92, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="none", ec=st.BLUE, lw=2.0))
    # kernel position
    ax.plot(it_f + 0.5, y + 0.5, "o", color=st.RED, ms=6, mec="white", mew=1, zorder=5)
    ax.text(-0.3, y + 0.5, f"plane {k}", ha="right", va="center", fontsize=9, color="#333333")

for t in range(NB):
    ax.text(t + 0.5, NPL + 0.12, str(t), ha="center", va="bottom", fontsize=8.5, color=st.GRAY)
ax.text(-0.3, NPL + 0.12, "TOF bin", ha="right", va="bottom", fontsize=8.5, color=st.GRAY)
ax.annotate("", (-1.6, -0.15), (-1.6, NPL - 0.2), arrowprops=dict(arrowstyle="-|>", color=st.GRAY, lw=1.2))
ax.text(-1.8, NPL / 2, "planes in the order of\nincreasing TOF position", rotation=90, ha="right", va="center",
        fontsize=8.5, color=st.GRAY)

# legend
items = [("#F6D2B8", st.ORANGE, "TOF window of the plane (gets a contribution)"),
         ("white", st.MID, "in the accumulator (registers)"),
         ("#D7EDDF", st.GREEN, "final: written to memory once (✓ = written at this step)")]
for i, (fc, ec, lab) in enumerate(items):
    yl = -0.75 - 0.5 * i
    ax.add_patch(Rectangle((0.1, yl - 0.18), 0.6, 0.36, fc=fc, ec=ec, lw=0.9))
    ax.text(0.95, yl, lab, va="center", fontsize=9)
ax.add_patch(FancyBboxPatch((9.1, -0.93), 0.6, 0.36, boxstyle="round,pad=0,rounding_size=0.08", fc="none",
                            ec=st.BLUE, lw=2.0))
ax.text(9.95, -0.75, f"accumulator: W = {W} TOF bins in registers,\nmoves forward with the window", va="center",
        fontsize=9)
ax.plot(9.4, -1.6, "o", color=st.RED, ms=6, mec="white", mew=1)
ax.text(9.95, -1.6, "TOF position of the plane's ray sample", va="center", fontsize=9)

ax.set_xlim(-2.6, NB + 0.2)
ax.set_ylim(-2.0, NPL + 0.6)
st.clean_axes(ax)
ax.set_title("TOF sinogram forward projection of one LOR: every TOF bin is written exactly once", loc="left")

fig.tight_layout()
st.save(fig, "sliding_window", png_preview_dir=sys.argv[1] if len(sys.argv) > 1 else None)
