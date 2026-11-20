"""Estimated time of a TOF OSEM reconstruction (4 iterations x 28 subsets) vs. number of counts.

SIGNA-like geometry (351 radial bins x 224 views x 2025 ring pairs, image 256 x 256 x 89),
NVIDIA RTX A4500, measured with scripts/benchmark_tof_sinogram.py (preset signa_subset /
signa_fine_subset: one subset of the full sinogram) and
scripts/benchmark_tof_listmode_scaling.py (sorted / unsorted events):

  TOF sinogram, 27 TOF bins (~sigma):    28 x 0.110 s + ~0.25 s per iteration  ->  ~3.3 s
  TOF sinogram, 134 TOF bins (sigma/5):  28 x 0.392 s + ~0.8 s per iteration   ->  ~11.8 s
  TOF listmode, sorted events:           ~9.4 ns per event and iteration (incl. sorting once)
  TOF listmode, unsorted (time order):   ~61 ns per event and iteration

The sensitivity image (the same for both) is not included.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

import ppstyle as st

N_IT = 4
counts = np.logspace(7, 9.6, 200)
t_lm = N_IT * 9.4e-9 * counts
t_lm_unsorted = N_IT * 61e-9 * counts
t_sino27 = N_IT * 3.3
t_sino134 = N_IT * 11.8

fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.loglog(counts, t_lm, color=st.RED, lw=2.2, label="TOF listmode, sorted events")
ax.loglog(counts, t_lm_unsorted, color=st.RED, lw=1.4, ls=(0, (4, 3)), alpha=0.7,
          label="TOF listmode, unsorted events (time order)")
ax.axhline(t_sino27, color=st.BLUE, lw=2.2, label="TOF sinogram, 27 TOF bins (≈ σ)")
ax.axhline(t_sino134, color=st.BLUE, lw=1.6, ls=(0, (6, 2)), label="TOF sinogram, 134 TOF bins (σ / 5)")
for t_s, lab in [(t_sino27, "≈ 3.5·10⁸"), (t_sino134, "≈ 1.3·10⁹")]:
    n = t_s / (N_IT * 9.4e-9)
    ax.plot(n, t_s, "o", color=st.GRAY, ms=7, mec="white", mew=1.2, zorder=5)
    ax.annotate(f"break-even\n{lab} counts", (n, t_s), (n * 1.25, t_s * 0.45), fontsize=8.5, color=st.GRAY,
                arrowprops=dict(arrowstyle="->", color=st.GRAY, lw=0.9))
ax.set_xlabel("number of counts (events)")
ax.set_ylabel("time for 4 iterations × 28 subsets [s]")
ax.set_xlim(counts[0], counts[-1])
# plain (unicode) tick labels instead of mathtext: the SVG keeps text as text
ax.set_xticks([1e7, 1e8, 1e9], ["10⁷", "10⁸", "10⁹"])
ax.set_yticks([1, 10, 100], ["1", "10", "100"])
ax.minorticks_off()
ax.set_ylim(0.3, 300)
ax.grid(True, which="major", color=st.MID, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
ax.set_title("TOF OSEM, SIGNA-like geometry, one GPU (estimate)", loc="left")
ax.text(0.99, 0.02, "RTX A4500; listmode time ∝ counts,\nsinogram time independent of the counts",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=st.GRAY)
fig.tight_layout()
st.save(fig, "cost_vs_counts", png_preview_dir=sys.argv[1] if len(sys.argv) > 1 else None)
