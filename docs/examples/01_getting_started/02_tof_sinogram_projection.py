"""
TOF Sinogram Forward and Back Projection
========================================

With time-of-flight (TOF), every LOR is divided into TOF bins, and the projection of a LOR
becomes a **TOF profile**: one value per TOF bin. The TOF sinogram projectors
:func:`parallelproj_core.joseph3d_tof_sino_fwd` and
:func:`parallelproj_core.joseph3d_tof_sino_back` compute all TOF bins of a LOR in one pass
(see :doc:`/theory/tof` and :doc:`/theory/sino_vs_lm`).

Additional arguments compared with the non-TOF projectors:

- ``tof_bin_width``: width of the TOF bins along the LOR (spatial units),
- ``tof_sigma``: standard deviation of the Gaussian TOF kernel (spatial units; one value or
  one per LOR),
- ``tof_center_offset``: shift of the TOF bins along the LOR (one value or one per LOR),
- ``num_tof_bins``: number of TOF bins, centered on the LOR center (plus offset),
- ``num_sigmas`` (optional, default 3): truncation of the Gaussian kernel.

The TOF projections have the shape ``(num_lors, num_tof_bins)``.
"""

# sphinx_gallery_start_ignore
import pathlib, sys
try:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
except NameError:  # run by sphinx-gallery: docs/conf.py adds docs/examples to the path
    pass
# sphinx_gallery_end_ignore
import numpy as np
import matplotlib.pyplot as plt
import parallelproj_core
from utils import COLORS, array_backend, disk_phantom, set_style, show_image, to_numpy

xp, dev = array_backend()
set_style()

image_np, image_origin_np, voxel_size_np, extent = disk_phantom()
image = xp.asarray(image_np, device=dev)
image_origin = xp.asarray(image_origin_np, device=dev)
voxel_size = xp.asarray(voxel_size_np, device=dev)

# %%
# TOF parameters and one LOR
# --------------------------
# A Gaussian TOF kernel with σ = 12 mm (FWHM 28 mm, i.e. a coincidence time resolution of
# about 190 ps), 45 TOF bins of 8 mm (covering 360 mm, the whole LOR), and one LOR that
# passes close to both hot spots.

tof_bin_width = 8.0
tof_sigma = xp.asarray([12.0], dtype=xp.float32, device=dev)
tof_center_offset = xp.asarray([0.0], dtype=xp.float32, device=dev)
num_tof_bins = 45

lor_start = xp.asarray([[-170.0, -55.0, 0.0]], dtype=xp.float32, device=dev)
lor_end = xp.asarray([[170.0, 64.0, 0.0]], dtype=xp.float32, device=dev)

# %%
# TOF forward projection
# ----------------------

tof_proj = xp.zeros((1, num_tof_bins), dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_tof_sino_fwd(lor_start, lor_end, image, image_origin, voxel_size, tof_proj,
                                        tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)

# the non-TOF projection of the same LOR for comparison
nontof_proj = xp.zeros(1, dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_fwd(lor_start, lor_end, image, image_origin, voxel_size, nontof_proj)

print(f"sum over the TOF bins: {float(xp.sum(tof_proj)):.2f}, non-TOF projection: {float(nontof_proj[0]):.2f}")

# %%
# The TOF bins together give the non-TOF projection: the TOF kernel only *distributes* the
# line integral along the LOR (the TOF weights of every image sample sum to 1, as long as the
# TOF bins cover the whole kernel).
#
# TOF back projection of a single TOF bin
# ---------------------------------------
# Back projecting a TOF sinogram that is 1 in one TOF bin and 0 elsewhere shows where along
# the LOR this TOF bin "sees" the image: a short segment, blurred by the TOF kernel.

t_max = int(np.argmax(to_numpy(tof_proj)[0]))
one_bin = xp.zeros((1, num_tof_bins), dtype=xp.float32, device=dev)
one_bin[0, t_max] = 1.0
back = xp.zeros(image.shape, dtype=xp.float32, device=dev)  # the back projection adds to the image
parallelproj_core.joseph3d_tof_sino_back(lor_start, lor_end, back, image_origin, voxel_size, one_bin,
                                         tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)

# %%
s, e = to_numpy(lor_start)[0], to_numpy(lor_end)[0]
d = (e - s) / np.linalg.norm(e - s)
c = 0.5 * (s + e)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1.35, 1]))
show_image(axs[0], image, extent, title="(a) one LOR and its TOF bins")
axs[0].plot([s[0], e[0]], [s[1], e[1]], color=COLORS["blue"], lw=2)
for t in range(0, num_tof_bins + 1, 4):  # every 4th TOF bin edge
    q = c + (t - num_tof_bins / 2) * tof_bin_width * d
    n = np.array([-d[1], d[0], 0]) * 6
    axs[0].plot([q[0] - n[0], q[0] + n[0]], [q[1] - n[1], q[1] + n[1]], color=COLORS["blue"], lw=0.8)
axs[0].text(e[0], e[1] + 10, "lor_end", ha="right", fontsize=8.5, color=COLORS["gray"])
axs[1].bar(np.arange(num_tof_bins), to_numpy(tof_proj)[0], color=COLORS["orange"], alpha=0.85, width=0.9)
axs[1].set_xlabel("TOF bin (numbered towards lor_end)")
axs[1].set_ylabel("TOF sinogram value")
axs[1].set_title("(b) TOF forward projection: one value per TOF bin")
axs[1].text(0.02, 0.95, f"sum over TOF bins = {float(xp.sum(tof_proj)):.1f}\n"
            f"non-TOF projection = {float(nontof_proj[0]):.1f}", transform=axs[1].transAxes, va="top",
            fontsize=9, color=COLORS["green"])
show_image(axs[2], back, extent, cmap="Oranges", title=f"(c) back projection of TOF bin {t_max} only")
axs[2].plot([s[0], e[0]], [s[1], e[1]], color=COLORS["blue"], lw=0.8, alpha=0.6)
for ax in (axs[0], axs[2]):
    ax.set_xlim(-190, 190)
    ax.set_ylim(-190, 190)
fig.show()

# %%
# The TOF profile shows where along the LOR the activity is: the plateau of the disk (value 1)
# and the peak of the large hot spot (value 4) closer to ``lor_end``. For a whole
# TOF sinogram, the arrays simply have one row per LOR. How the TOF weights are computed (and
# how ``num_sigmas`` and the TOF bin width change them) is shown in
# :ref:`sphx_glr_auto_examples_02_projector_properties_02_tof_kernel.py` (theory:
# :doc:`/theory/tof`).
