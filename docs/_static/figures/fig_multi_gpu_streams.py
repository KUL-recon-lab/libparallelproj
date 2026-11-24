"""Synchronous vs. asynchronous projector calls with 2 GPUs (timeline).

A mini batch of 4 samples, 2 per GPU, projected from a single Python thread.
Top: stream=None, every call waits until its GPU is done, the GPUs work one after another.
Bottom: stream=<current stream>, the CPU only queues the work, both GPUs work at the same time.

Run in this directory with ``python fig_multi_gpu_streams.py .``: writes the SVG (shown in the gallery
example 02_multi_gpu_streams, which is not executed when the docs are built) and a PNG next to it (the
thumbnail of that example, sphinx_gallery_thumbnail_path).
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import ppstyle as st

GPU_COLORS = (st.BLUE, st.ORANGE)
H = 0.5  # height of a block
T = 1.0  # duration of one projection on a GPU
Q = 0.08  # time to queue one call on the CPU


def block(ax, t0, dt, y, color, text="", alpha=0.85, text_color="white"):
    ax.add_patch(FancyBboxPatch((t0 + 0.02, y), dt - 0.04, H, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=color, ec="none", alpha=alpha))
    if text:
        ax.text(t0 + dt / 2, y + H / 2, text, ha="center", va="center", fontsize=8.5, color=text_color)


def lanes(ax, y_top, title):
    """y positions of the CPU, GPU 0 and GPU 1 lanes"""
    ax.text(-1.05, y_top + 0.45, title, fontsize=10, fontweight="bold")
    ys = {"CPU": y_top - 0.2, "GPU 0": y_top - 0.9, "GPU 1": y_top - 1.6}
    for name, y in ys.items():
        ax.text(-0.12, y + H / 2, name, ha="right", va="center", fontsize=9, color=st.GRAY)
        ax.plot([0, 4.35], [y - 0.06, y - 0.06], color=st.MID, lw=0.8)
    return ys


fig, ax = plt.subplots(figsize=(10.5, 5.6))

# --- synchronous: call k starts when call k-1 is done
ys = lanes(ax, 4.5, "synchronous: stream=None (the default)")
order = [(0, 0), (1, 1), (2, 0), (3, 1)]  # (sample, GPU)
for k, (s, g) in enumerate(order):
    t0 = k * T
    block(ax, t0, T, ys["CPU"], st.LIGHT, "call + wait", alpha=1.0, text_color=st.GRAY)
    block(ax, t0, T, ys[f"GPU {g}"], GPU_COLORS[g], f"sample {s}")
ax.annotate("", (4 * T, ys["GPU 1"] - 0.25), (0, ys["GPU 1"] - 0.25),
            arrowprops=dict(arrowstyle="<->", color=st.GRAY, lw=0.9))
ax.text(2 * T, ys["GPU 1"] - 0.42, "4 projections one after another", ha="center", va="top", fontsize=8.5,
        color=st.GRAY)

# --- asynchronous: the CPU only queues the calls, both GPUs work in parallel
ys = lanes(ax, 1.45, "asynchronous: stream=torch.cuda.current_stream(device).cuda_stream")
for k, (s, g) in enumerate(order):
    block(ax, k * Q, Q, ys["CPU"], st.GRAY, alpha=0.7)
    t0 = k * Q + (k // 2) * T  # queued work starts as soon as the GPU is free
    block(ax, t0, T, ys[f"GPU {g}"], GPU_COLORS[g], f"sample {s}")
ax.text(4 * Q + 0.05, ys["CPU"] + H / 2, "queue 4 calls, then continue (e.g. queue the next operations)",
        ha="left", va="center", fontsize=8.5, color=st.GRAY)
t_end = 3 * Q + 2 * T
ax.annotate("", (t_end, ys["GPU 1"] - 0.25), (0, ys["GPU 1"] - 0.25),
            arrowprops=dict(arrowstyle="<->", color=st.GRAY, lw=0.9))
ax.text(t_end / 2, ys["GPU 1"] - 0.42, "both GPUs at the same time", ha="center", va="top", fontsize=8.5,
        color=st.GRAY)
ax.annotate("", (4 * T, ys["GPU 1"] - 0.25), (t_end, ys["GPU 1"] - 0.25),
            arrowprops=dict(arrowstyle="<->", color=st.GREEN, lw=1.2))
ax.text((t_end + 4 * T) / 2, ys["GPU 1"] - 0.42, "time saved", ha="center", va="top", fontsize=9,
        color=st.GREEN, fontweight="bold")

ax.annotate("time", (4.35, -1.05), (3.6, -1.05), fontsize=9, color=st.GRAY, va="center",
            arrowprops=dict(arrowstyle="-|>", color=st.GRAY))
ax.set_xlim(-1.1, 4.45)
ax.set_ylim(-1.2, 5.3)
ax.axis("off")
ax.set_title("A mini batch of 4 samples on 2 GPUs, called from one Python thread", loc="left")
st.save(fig, "multi_gpu_streams", png_preview_dir=sys.argv[1] if len(sys.argv) > 1 else None)
