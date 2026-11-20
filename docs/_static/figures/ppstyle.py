"""Common style of the figures of the libparallelproj documentation.

All figure scripts in this directory import this module, so that the figures share colors,
fonts and line widths. Figures are saved as SVG with the text kept as text (small files,
sharp at any zoom, the browser renders the text with its sans-serif font) on an opaque white
background (readable in the light and the dark theme of the documentation).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- colors (colorblind-friendly, muted)
BLUE = "#2F6DB5"  # LORs, rays
ORANGE = "#E07B39"  # TOF kernels, TOF windows
GREEN = "#3A9D5D"  # finished / written results
RED = "#C8433A"  # highlights (sample points, the "current" element)
PURPLE = "#7A5BA6"  # secondary highlight
GRAY = "#6B6B6B"  # text, axes, outlines
LIGHT = "#E9EDF2"  # light fills (voxels, background shapes)
MID = "#C9D1DB"  # grid lines

# --- fonts and lines
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": "#333333",
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "text.color": "#222222",
        "lines.linewidth": 1.6,
        "svg.fonttype": "none",  # keep text as text in SVGs
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

OUT_DIR = Path(__file__).resolve().parent


def save(fig, name, png_preview_dir=None, dpi=110):
    """save the figure as <name>.svg next to the scripts (and optionally a PNG preview)"""
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight", pad_inches=0.08)
    if png_preview_dir is not None:
        fig.savefig(Path(png_preview_dir) / f"{name}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def clean_axes(ax):
    """axes without frame and ticks (for schematic drawings)"""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
