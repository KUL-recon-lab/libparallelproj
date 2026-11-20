"""Why the order of LORs / events matters on GPUs: memory transactions of a warp.

32 neighbouring GPU threads (a warp, 8 shown) access the image at the same time. Memory is
transferred in sectors of 32 bytes (8 floats). If the threads access nearby voxels, few
sectors are transferred and all their bytes are used (left); if they access voxels far apart,
every thread needs its own sector and most transferred bytes are wasted (right).
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

import ppstyle as st

NT = 8  # threads shown
NS = 8  # sectors shown
SEC = 8  # floats per sector


def draw(ax, targets, title, note):
    # threads
    for t in range(NT):
        x = 4 + t * 7.4
        ax.add_patch(FancyBboxPatch((x - 2.2, 6.6), 4.4, 2.0, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc="#E3ECF7", ec=st.BLUE, lw=1.3))
        ax.text(x, 7.6, f"T{t}", ha="center", va="center", fontsize=8.5, color=st.BLUE)
    ax.text(30, 9.8, "warp (32 threads run together; 8 shown)", ha="center", va="center", fontsize=9, color=st.BLUE)
    # memory: NS sectors of SEC floats
    used = set(targets)
    fetched = {c // SEC for c in targets}
    for c in range(NS * SEC):
        sec = c // SEC
        fc = "white"
        if sec in fetched:
            fc = "#F6D2B8" if c in used else "#FBE9DC"
        ax.add_patch(Rectangle((c, 0), 1, 1.6, fc=fc, ec=st.MID, lw=0.5))
    for s in range(NS):
        ax.add_patch(Rectangle((s * SEC, 0), SEC, 1.6, fc="none", ec=st.ORANGE if s in fetched else st.GRAY,
                               lw=1.6 if s in fetched else 0.9))
    ax.text(32, -0.9, "image in memory: sectors of 32 bytes (8 floats)", ha="center", va="top", fontsize=8.5,
            color=st.GRAY)
    for t, c in enumerate(targets):
        x = 4 + t * 7.4
        ax.add_patch(FancyArrowPatch((x, 6.5), (c + 0.5, 1.7), arrowstyle="-|>", mutation_scale=8, color=st.BLUE,
                                     lw=0.9, alpha=0.8))
    nbytes = len(fetched) * 32
    ax.text(32, -2.9, f"{len(fetched)} sector{'s' if len(fetched) > 1 else ''} transferred ({nbytes} bytes) "
            f"for {NT * 4} bytes used", ha="center", va="top", fontsize=9.5, fontweight="bold",
            color=st.GREEN if len(fetched) <= 2 else st.RED)
    ax.text(32, -4.5, note, ha="center", va="top", fontsize=8.5, color=st.GRAY)
    ax.set_xlim(-1, 65)
    ax.set_ylim(-6.6, 10.6)
    st.clean_axes(ax)
    ax.set_aspect("auto")
    ax.set_title(title, loc="left")


near = [17, 18, 19, 20, 21, 22, 23, 24]
far = [2, 13, 21, 30, 38, 44, 53, 60]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4))
draw(ax, near, "(a) neighbouring threads access nearby voxels",
     "sinogram with planes fastest + axial image axis last,\nlistmode events sorted with tof_lm_sort_key")
draw(bx, far, "(b) neighbouring threads access voxels far apart",
     "unfavorable sinogram order / image layout,\nlistmode events in time (random) order")
fig.tight_layout(w_pad=3)
st.save(fig, "warp_memory", png_preview_dir=sys.argv[1] if len(sys.argv) > 1 else None)
