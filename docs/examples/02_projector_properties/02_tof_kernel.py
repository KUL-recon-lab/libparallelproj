"""
The TOF Kernel: Truncation, Bin Width and Normalization
=======================================================

The TOF projectors distribute the line integral of a LOR over its TOF bins with a Gaussian
TOF kernel (see :doc:`/theory/tof`). Three things determine the TOF weights:

- ``tof_sigma``: the standard deviation of the Gaussian (spatial units, here mm),
- ``num_sigmas`` (default 3): the kernel is **truncated** at ±``num_sigmas`` · σ and
  **renormalized**,
- ``tof_bin_width``: the weight of a TOF bin is the kernel **integrated over the bin**.

This example shows their effect on the TOF projection of a point source, and that the
weights of every image sample sum to 1: **the sum over all TOF bins equals the non-TOF
projection**, as long as the TOF bins cover the kernel.
"""

# sphinx_gallery_start_ignore
import pathlib, sys
try:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
except NameError:  # run by sphinx-gallery: docs/conf.py adds docs/examples to the path
    pass
# sphinx_gallery_end_ignore
import math

import numpy as np
import matplotlib.pyplot as plt
import parallelproj_core
from utils import COLORS, array_backend, disk_phantom, lor_2d, set_style, to_numpy

xp, dev = array_backend()
set_style()

# %%
# A point source and a LOR through it
# -----------------------------------
# A point source (one voxel with value 1) in voxel [42, 32, 0] of a 64 x 64 x 1 image with
# 4 mm voxels, i.e. at x = 42 mm, y = 2 mm, and a LOR along the x-axis through it. Its
# non-TOF projection is 4 (1 voxel x 4 mm). TOF: σ = 12 mm, 45 TOF bins of 8 mm.

n, vs = 64, 4.0
image_origin = xp.asarray([-(n - 1) / 2 * vs, -(n - 1) / 2 * vs, 0.0], dtype=xp.float32, device=dev)
voxel_size = xp.asarray([vs, vs, vs], dtype=xp.float32, device=dev)
point = xp.zeros((n, n, 1), dtype=xp.float32, device=dev)
point[42, 32, 0] = 1.0
x_src = 42.0  # position of the point source along the LOR (from the LOR center)

lor_start = xp.asarray([[-170.0, 2.0, 0.0]], dtype=xp.float32, device=dev)
lor_end = xp.asarray([[170.0, 2.0, 0.0]], dtype=xp.float32, device=dev)
sigma = 12.0
tof_sigma = xp.asarray([sigma], dtype=xp.float32, device=dev)
tof_offset = xp.asarray([0.0], dtype=xp.float32, device=dev)
tof_bin_width, num_tof_bins = 8.0, 45


def tof_profile(bin_width, num_bins, **kwargs):
    """TOF sinogram projection of the point source (one value per TOF bin)"""
    p = xp.zeros((1, num_bins), dtype=xp.float32, device=dev)
    parallelproj_core.joseph3d_tof_sino_fwd(lor_start, lor_end, point, image_origin, voxel_size, p, bin_width,
                                            tof_sigma, tof_offset, num_bins, **kwargs)
    return to_numpy(p)[0]


# %%
# 1. Truncation: num_sigmas = 2, 3, 4
# -----------------------------------
# Outside ±``num_sigmas`` · σ around an image sample, the TOF weights are exactly 0 (only
# the TOF bins that overlap this interval get a weight). Inside, the truncated Gaussian is
# renormalized, so the weights still sum to 1.

profiles_ns = {ns: tof_profile(tof_bin_width, num_tof_bins, num_sigmas=ns) for ns in (2.0, 3.0, 4.0)}
for ns, p in profiles_ns.items():
    nz = np.nonzero(p)[0]
    print(f"num_sigmas = {ns:g}: non-zero TOF bins {nz[0]}..{nz[-1]} ({nz.size} bins), maximum {p.max():.3f}, "
          f"sum {p.sum():.4f}")

# %%
# 2. TOF bin width
# ----------------
# The same point source with TOF bins of 4, 12 and 24 mm (and as many bins as needed to
# cover the LOR). Divided by the bin width, the TOF projections are comparable: they are
# the Gaussian (centered at the point source) averaged over each TOF bin.

profiles_w = {}
for w in (4.0, 12.0, 24.0):
    nb = int(round(num_tof_bins * tof_bin_width / w)) | 1  # odd: the central bin is centered on the LOR center
    pos = (np.arange(nb) - (nb - 1) / 2) * w  # TOF bin centers along the LOR (from the LOR center)
    profiles_w[w] = (pos, tof_profile(w, nb))

# %%
# 3. Sum over the TOF bins = non-TOF projection
# ---------------------------------------------
# For the disk phantom and 240 LORs (at 0° and 33°): the sum over 45 TOF bins of 8 mm (360
# mm, the whole LOR) equals the non-TOF projection. With only the 15 central TOF bins (120
# mm), the parts of the LORs outside the TOF bins are lost - as for a real scanner whose
# TOF bins do not cover the whole LOR.

phantom_np, _, _, _ = disk_phantom()
phantom = xp.asarray(phantom_np, device=dev)
lors = [lor_2d(0, ri) for ri in np.linspace(-118, 118, 120)] + [lor_2d(33, ri) for ri in np.linspace(-118, 118, 120)]
ls = xp.asarray(np.array([s for s, _ in lors]), device=dev)
le = xp.asarray(np.array([e for _, e in lors]), device=dev)

nontof = xp.zeros(len(lors), dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_fwd(ls, le, phantom, image_origin, voxel_size, nontof)
tof_sums = {}
for nb in (45, 15):
    p = xp.zeros((len(lors), nb), dtype=xp.float32, device=dev)
    parallelproj_core.joseph3d_tof_sino_fwd(ls, le, phantom, image_origin, voxel_size, p, tof_bin_width, tof_sigma,
                                            tof_offset, nb)
    tof_sums[nb] = to_numpy(xp.sum(p, axis=1))
nontof = to_numpy(nontof)
print(f"max. |sum over 45 TOF bins - non-TOF| / max. non-TOF: {np.abs(tof_sums[45] - nontof).max() / nontof.max():.1e}")

# %%
fig, axs = plt.subplots(2, 2, figsize=(13, 9.5), layout="constrained")
cols = {2.0: COLORS["red"], 3.0: COLORS["blue"], 4.0: COLORS["green"]}
bins = np.arange(num_tof_bins)
for ns, p in profiles_ns.items():
    lab = f"num_sigmas = {ns:g} ({np.count_nonzero(p)} bins, sum {p.sum():.2f})"
    axs[0, 0].plot(bins, p, ".-", color=cols[ns], label=lab)
    axs[0, 1].semilogy(bins, np.where(p > 0, p, np.nan), ".-", color=cols[ns], label=lab)
for ax in axs[0]:
    ax.set_xlim(17, 37)
    ax.set_xlabel("TOF bin")
    ax.axvline((num_tof_bins - 1) / 2 + x_src / tof_bin_width, color=COLORS["gray"], lw=0.6, ls=":")
    ax.legend(loc="upper left")
axs[0, 0].set_ylim(0, 1.45)
axs[0, 0].set_ylabel("TOF projection of the point source")
axs[0, 0].set_title("(a) num_sigmas = 2, 3, 4: linear scale")
axs[0, 1].set_ylim(1e-5, 3)
axs[0, 1].set_ylabel("TOF projection of the point source (log scale)")
axs[0, 1].text(0.99, 0.03, "weight 0 outside ± num_sigmas · σ\n(not shown on the log scale)", ha="right",
               transform=axs[0, 1].transAxes, fontsize=8, color=COLORS["gray"])
axs[0, 1].set_title("(b) the same on a log scale: the tails")

xx = np.linspace(-40, 120, 400)
axs[1, 0].plot(xx, np.exp(-0.5 * ((xx - x_src) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)),
               color=COLORS["gray"], lw=2.5, alpha=0.4, label=f"Gaussian kernel, σ = {sigma:g} mm")
for (w, (pos, p)), col in zip(profiles_w.items(), [COLORS["purple"], COLORS["orange"], COLORS["blue"]]):
    axs[1, 0].step(pos, p / (4.0 * w), where="mid", color=col, label=f"TOF bin width {w:g} mm")
axs[1, 0].axvline(x_src, color=COLORS["gray"], lw=0.6, ls=":")
axs[1, 0].set_xlim(-40, 120)
axs[1, 0].set_xlabel("TOF bin center along the LOR [mm]")
axs[1, 0].set_ylabel("TOF projection / (non-TOF projection · bin width) [1/mm]")
axs[1, 0].legend(loc="upper left")
axs[1, 0].set_title("(c) TOF bin width: the kernel integrated over each bin")

axs[1, 1].plot(nontof, tof_sums[45], "o", ms=3, alpha=0.7, color=COLORS["green"],
               label="45 TOF bins of 8 mm (whole LOR)")
axs[1, 1].plot(nontof, tof_sums[15], "o", ms=3, alpha=0.7, color=COLORS["red"], label="15 central TOF bins only")
m = float(nontof.max())
axs[1, 1].plot([0, m], [0, m], color=COLORS["gray"], lw=1, ls=(0, (3, 2)))
axs[1, 1].set_xlabel("non-TOF projection")
axs[1, 1].set_ylabel("sum over the TOF bins")
axs[1, 1].legend(loc="upper left")
axs[1, 1].set_title("(d) sum over the TOF bins = non-TOF projection")
fig.show()

# %%
# - **(a), (b)** ``num_sigmas`` sets how many TOF bins get a weight (here 7, 10 and 13 bins
#   of 8 mm for σ = 12 mm): fewer bins make the projections faster. On the linear scale
#   the profiles are almost identical (3 and 4 overlap); the renormalization raises the maximum for
#   ``num_sigmas = 2``, and the sum is the non-TOF projection (4) in all cases. The log
#   scale shows the difference: the tails, two to four orders of magnitude below the
#   maximum. The default of 3 keeps 99.7 % of the Gaussian.
# - **(c)** Wide TOF bins average the kernel over the bin: the TOF profile becomes a coarser
#   staircase, and the *effective* kernel of a TOF bin is wider than the Gaussian. The TOF
#   resolution of a scanner and its TOF bin width therefore both matter; σ describes only
#   the timing resolution. (The projectors integrate the Gaussian over each TOF bin exactly,
#   so no TOF bin is too wide for the projector.)
# - **(d)** The sum over the TOF bins equals the non-TOF projection (up to float32
#   rounding) only if the TOF bins cover the LOR inside the image. LORs through the center
#   (large non-TOF projections) are longer than the 120 mm of 15 TOF bins.
#
# Kernels that are not Gaussian (e.g. measured ones) can be used with the tabulated-kernel
# projectors, see
# :ref:`sphx_glr_auto_examples_02_projector_properties_03_tof_kernel_table.py`.
