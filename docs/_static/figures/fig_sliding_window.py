"""TOF sinogram projection of one LOR: the sliding window of TOF bins in registers.

Rows: the image planes of one LOR, visited in the order of increasing TOF position.
Columns: the TOF bins of the LOR.

(a) forward projection: for every plane, the TOF window (bins overlapping the kernel support)
    gets a contribution. The accumulator (W bins in registers) moves forward with the window;
    a bin that leaves the accumulator is final and is written to memory exactly once.
(b) back projection (the mirror image): the buffer (W bins in registers) moves forward with
    the window; every TOF bin is loaded from memory exactly once, when it enters the buffer,
    and the plane's value is the TOF-weighted sum of its window bins.
"""

import math
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

import ppstyle as st

NB = 16  # TOF bins
NPL = 7  # planes shown
W = 8  # accumulator / buffer size
HALF = 2.4  # half width of the kernel support in TOF bins
IT0, DIT = 2.6, 1.15  # TOF position of the first plane, TOF bins per plane

WINDOW = ("#F6D2B8", st.ORANGE)
DONE = ("#D7EDDF", st.GREEN)
LOADED = ("#DCE6F5", st.BLUE)
PASSED = ("#F2F2F2", "#D5D5D5")


def windows():
    """(it_f, first bin, last bin, accumulator start) for every plane (store width 1)"""
    out, base = [], None
    for k in range(NPL):
        it_f = IT0 + DIT * k
        lo = int(math.floor(it_f - HALF + 0.5))
        hi = int(math.floor(it_f + HALF + 0.5))
        base = lo if base is None else max(base, lo)
        out.append((it_f, lo, hi, base))
    return out


def draw(ax, mode):
    prev_base = None
    for k, (it_f, lo, hi, base) in enumerate(windows()):
        y = NPL - 1 - k
        for t in range(NB):
            if lo <= t <= hi:
                fc, ec = WINDOW
            elif t < base:
                fc, ec = DONE if mode == "fwd" else PASSED
            elif t < base + W:
                fc, ec = ("white", st.MID) if mode == "fwd" else LOADED
            else:
                fc, ec = "white", "#E3E7EC"
            ax.add_patch(Rectangle((t + 0.06, y + 0.1), 0.88, 0.8, fc=fc, ec=ec, lw=0.9))
        if mode == "fwd":
            # bins written at this step: they left the accumulator
            marks = range(prev_base, base) if prev_base is not None else []
            sym, col = "✓", st.GREEN
        else:
            # bins loaded at this step: they entered the buffer
            new = range(base, base + W) if prev_base is None else range(prev_base + W, base + W)
            marks, sym, col = new, "↓", st.BLUE
        for t in marks:
            if 0 <= t < NB:
                ax.text(t + 0.5, y + 0.5, sym, ha="center", va="center", fontsize=10, color=col, fontweight="bold",
                        zorder=6)
        ax.add_patch(FancyBboxPatch((base + 0.02, y + 0.04), W - 0.04, 0.92, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="none", ec=st.BLUE, lw=2.0))
        ax.plot(it_f + 0.5, y + 0.5, "o", color=st.RED, ms=6, mec="white", mew=1, zorder=5)
        ax.text(-0.3, y + 0.5, f"plane {k}", ha="right", va="center", fontsize=9, color="#333333")
        prev_base = base
    for t in range(NB):
        ax.text(t + 0.5, NPL + 0.12, str(t), ha="center", va="bottom", fontsize=8.5, color=st.GRAY)
    ax.text(-0.3, NPL + 0.12, "TOF bin", ha="right", va="bottom", fontsize=8.5, color=st.GRAY)
    ax.annotate("", (-1.6, -0.1), (-1.6, NPL - 0.2), arrowprops=dict(arrowstyle="-|>", color=st.GRAY, lw=1.2))
    ax.text(-1.8, NPL / 2, "planes in the order of\nincreasing TOF position", rotation=90, ha="right", va="center",
            fontsize=8.5, color=st.GRAY)


def legend(ax, items, frame_text):
    for i, (fc, ec, lab) in enumerate(items):
        yl = -0.75 - 0.5 * i
        ax.add_patch(Rectangle((0.1, yl - 0.18), 0.6, 0.36, fc=fc, ec=ec, lw=0.9))
        ax.text(0.95, yl, lab, va="center", fontsize=9)
    ax.add_patch(FancyBboxPatch((9.1, -0.93), 0.6, 0.36, boxstyle="round,pad=0,rounding_size=0.08", fc="none",
                                ec=st.BLUE, lw=2.0))
    ax.text(9.95, -0.75, frame_text, va="center", fontsize=9)
    ax.plot(9.4, -1.6, "o", color=st.RED, ms=6, mec="white", mew=1)
    ax.text(9.95, -1.6, "TOF position of the plane's ray sample", va="center", fontsize=9)
    ax.set_xlim(-2.6, NB + 0.2)
    ax.set_ylim(-2.0, NPL + 0.6)
    st.clean_axes(ax)


fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 10.6))
draw(ax, "fwd")
legend(ax, [(*WINDOW, "TOF window of the plane: gets a contribution"),
            ("white", st.MID, "in the accumulator (registers)"),
            (*DONE, "final: written to memory once (✓ = written at this step)")],
       f"accumulator: W = {W} TOF bins in registers,\nmoves forward with the window")
ax.set_title("(a) forward projection: every TOF bin of the LOR is written exactly once", loc="left")
draw(bx, "back")
legend(bx, [(*WINDOW, "TOF window of the plane: weighted sum → back projected"),
            (*LOADED, "in the buffer (registers), loaded once (↓ = loaded at this step)"),
            (*PASSED, "not needed anymore")],
       f"buffer: W = {W} TOF bins in registers,\nmoves forward with the window")
bx.set_title("(b) back projection: every TOF bin of the LOR is read exactly once", loc="left")

fig.tight_layout(h_pad=2.0)
st.save(fig, "sliding_window", png_preview_dir=sys.argv[1] if len(sys.argv) > 1 else None)
