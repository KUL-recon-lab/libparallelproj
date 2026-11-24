"""TOF listmode on 2 GPUs: assigning samples with different numbers of events to the GPUs.

The time of a listmode projection is roughly proportional to the number of events, and the
mini batch is done when the slowest GPU is done. Left: alternating assignment (sample k to GPU
k mod 2). Right: greedy assignment (largest sample first, to the GPU with the fewest events so far).

Run in this directory with ``python fig_lm_load_balancing.py .``: writes the SVG (shown in the gallery
example 03_multi_gpu_tof_listmode, which is not executed when the docs are built) and a PNG next to it (the
thumbnail of that example, sphinx_gallery_thumbnail_path).
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

import ppstyle as st

EVENTS = np.array([3.6, 1.2, 2.8, 1.9, 3.1, 1.4])  # million events of the samples s0 ... s5
COLORS = [st.BLUE, st.ORANGE, st.GREEN, st.RED, st.PURPLE, st.GRAY]


def draw(ax, assignment, title):
    load = [0.0, 0.0]
    for k, g in assignment:
        ax.barh(g, EVENTS[k], left=load[g], height=0.6, color=COLORS[k], alpha=0.85, edgecolor="white", lw=1.5)
        ax.text(load[g] + EVENTS[k] / 2, g, f"s{k}\n{EVENTS[k]:.1f} M", ha="center", va="center", color="white",
                fontsize=8.5)
        load[g] += EVENTS[k]
    t_end = max(load)
    ax.axvline(t_end, color=st.RED, lw=1.2, ls=(0, (4, 2)))
    ax.text(t_end + 0.12, 1.5, f"batch done\nafter {t_end:.1f} M", color=st.RED, fontsize=8.5, va="center")
    ax.set_yticks([0, 1], ["GPU 0", "GPU 1"])
    ax.set_xlabel("million events projected (≈ time)")
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-0.5, 1.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=10.5)


fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.4))
draw(a, [(k, k % 2) for k in range(len(EVENTS))], "(a) alternating: sample k on GPU k mod 2")
greedy, load = [], [0.0, 0.0]
for k in np.argsort(-EVENTS):  # largest sample first ...
    g = int(np.argmin(load))  # ... to the GPU with the fewest events so far
    greedy.append((int(k), g))
    load[g] += EVENTS[k]
draw(b, greedy, "(b) greedy: largest first, to the least loaded GPU")
fig.tight_layout(w_pad=3)
st.save(fig, "lm_load_balancing", png_preview_dir=sys.argv[1] if len(sys.argv) > 1 else None)
