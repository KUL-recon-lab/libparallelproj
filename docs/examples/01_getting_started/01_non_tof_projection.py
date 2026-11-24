"""
Non-TOF Forward and Back Projection
===================================

The non-TOF projectors :func:`parallelproj_core.joseph3d_fwd` (forward projection) and
:func:`parallelproj_core.joseph3d_back` (back projection) compute line integrals through a
3D image along lines of response (LORs), using Joseph's method (see
:doc:`/theory/joseph`):

- the **forward projection** of an image gives one value per LOR: the line integral of the
  image along the LOR (in units of *image value × length*),
- the **back projection** distributes one value per LOR back into the image along the LOR
  (the adjoint of the forward projection).

Here, the image has a single plane (shape ``(128, 128, 1)``) and the LORs lie in this plane,
which makes the result easy to plot. Everything works the same way in 3D.
"""

# sphinx_gallery_start_ignore
# the helper module utils.py is in the parent directory (docs/examples)
import pathlib, sys
try:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
except NameError:  # run by sphinx-gallery: docs/conf.py adds docs/examples to the path
    pass
# sphinx_gallery_end_ignore
import numpy as np
import matplotlib.pyplot as plt
import parallelproj_core
from utils import COLORS, array_backend, disk_phantom, lor_2d, set_style, show_image, to_numpy

xp, dev = array_backend()  # CuPy on the GPU (CUDA build) or NumPy on the CPU
set_style()
print(f"parallelproj_core {parallelproj_core.__version__}, using {xp.__name__} on {dev}")

# %%
# The image
# ---------
# The projectors need the image as a 3D ``float32`` array (C order), the position of the
# **center of voxel [0, 0, 0]** (``image_origin``) and the voxel size, in the same units as
# the LOR coordinates (here mm). ``disk_phantom`` creates a disk with two hot spots in an image
# with 128 x 128 x 1 voxels of 2 mm.

image_np, image_origin_np, voxel_size_np, extent = disk_phantom()
image = xp.asarray(image_np, device=dev)
image_origin = xp.asarray(image_origin_np, device=dev)
voxel_size = xp.asarray(voxel_size_np, device=dev)
print(f"image shape {image.shape}, origin {image_origin_np} mm, voxel size {voxel_size_np} mm")

# %%
# LORs and forward projection
# ---------------------------
# A LOR is given by its two end points ``lor_start`` and ``lor_end`` (arrays of shape
# ``(num_lors, 3)``). ``lor_2d`` gives the end points of a LOR in the image plane (direction
# angle and distance from the center) on a "scanner" circle with a radius of 180 mm.

lors = [lor_2d(20, 25), lor_2d(75, -40), lor_2d(-35, 60)]
lor_start = xp.asarray(np.array([s for s, _ in lors]), device=dev)
lor_end = xp.asarray(np.array([e for _, e in lors]), device=dev)

# the forward projection writes one value per LOR into an existing array
proj = xp.zeros(lor_start.shape[0], dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_fwd(lor_start, lor_end, image, image_origin, voxel_size, proj)
print("forward projection:", to_numpy(proj))

# %%
# Back projection
# ---------------
# The back projection **adds** to the image passed to it, so it starts from zeros. Back
# projecting a value of 1 for each LOR shows which voxels contribute to the LORs and with
# which weights.

back = xp.zeros(image.shape, dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_back(lor_start, lor_end, back, image_origin, voxel_size,
                                xp.ones(lor_start.shape[0], dtype=xp.float32, device=dev))

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
show_image(a, image, extent, title="(a) forward projection: one value per LOR")
cols = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
ls_np, le_np, p_np = to_numpy(lor_start), to_numpy(lor_end), to_numpy(proj)
for k in range(len(lors)):
    a.plot([ls_np[k, 0], le_np[k, 0]], [ls_np[k, 1], le_np[k, 1]], color=cols[k], lw=2)
    a.text(le_np[k, 0], le_np[k, 1], f"  LOR {k}: {p_np[k]:.0f}", color=cols[k], fontsize=9, va="center",
           ha="left" if le_np[k, 0] > 0 else "right")
show_image(b, back, extent, cmap="Blues", title="(b) back projection of ones along the same LORs")
for ax in (a, b):
    ax.set_xlim(-190, 190)
    ax.set_ylim(-190, 190)
fig.show()

# %%
# The projection values are line integrals: e.g. LOR 2 crosses ~160 mm of the disk with value
# 1 (and no hot spot), so its projection is about 160 (image value × mm). The back projection
# of ones is non-zero only in the voxels next to the LORs, with the interpolation weights of
# Joseph's method.
#
# Next: :ref:`the same with TOF <sphx_glr_auto_examples_01_getting_started_02_tof_sinogram_projection.py>`.
