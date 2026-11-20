"""Where the projectors sit in a PET system model.

Expected data = multiplicative corrections x geometric (TOF) projection of the image
                + additive contamination (scatter, randoms).
libparallelproj computes the geometric projection (forward) and its adjoint (back
projection); everything else is applied by the reconstruction framework.
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import ppstyle as st

fig, ax = plt.subplots(figsize=(11, 3.6))


def box(x, y, w, h, title, sub, fc, ec, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold" if bold else "normal", color="#222222")
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=8.5, color=st.GRAY)


def arrow(x0, y0, x1, y1, col=st.GRAY, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13, color=col, lw=1.4,
                                 connectionstyle=f"arc3,rad={rad}"))


Y, H = 1.2, 1.15
box(0.0, Y, 1.7, H, "image x", "activity in voxels", "white", st.GRAY)
box(2.2, Y, 1.9, H, "resolution model", "(optional, image space)", "white", st.MID)
box(4.6, Y, 2.4, H, "geometric projection", "Joseph, non-TOF or TOF\nsinogram or listmode", "#E3ECF7", st.BLUE,
    bold=True)
box(7.5, Y, 2.0, H, "× corrections", "attenuation,\nnormalization", "white", st.GRAY)
box(10.0, Y, 1.8, H, "+ contamination", "scatter, randoms", "white", st.GRAY)
box(12.3, Y, 1.7, H, "expected data", "per LOR / TOF bin\nor per event", "white", st.GRAY)
for x0, x1 in [(1.7, 2.2), (4.1, 4.6), (7.0, 7.5), (9.5, 10.0), (11.8, 12.3)]:
    arrow(x0 + 0.03, Y + H / 2, x1 - 0.03, Y + H / 2)
ax.text(5.8, Y + H + 0.32, "libparallelproj", ha="center", va="bottom", fontsize=11, fontweight="bold", color=st.BLUE)
ax.text(5.8, Y + H + 0.1, "geometric forward projection and its exact adjoint (back projection)", ha="center",
        va="bottom", fontsize=8.5, color=st.BLUE)
# back projection path
arrow(13.15, Y - 0.05, 5.8, Y - 0.05, col=st.BLUE, rad=-0.12)
ax.text(9.5, 0.5, "back projection (e.g. of the ratio data / expected data in MLEM) with the same corrections",
        ha="center", va="center", fontsize=8.5, color=st.BLUE)

ax.set_xlim(-0.2, 14.2)
ax.set_ylim(0.3, 3.1)
st.clean_axes(ax)
ax.set_aspect("auto")
fig.tight_layout()
st.save(fig, "system_model", png_preview_dir=sys.argv[1] if len(sys.argv) > 1 else None)
