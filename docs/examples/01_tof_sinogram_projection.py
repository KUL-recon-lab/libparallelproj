"""
Sinogram (histogram) TOF Joseph Forward and Back Projection
===========================================================

This minimal example demonstrates how to call python API
for the sinogram (histogram) TOF Joseph forward and back projection functions, which are implemented in
:func:`parallelproj_backend.joseph3d_tof_sino_fwd` and :func:`parallelproj_backend.joseph3d_tof_sino_back`.

.. note::
  In this educational example, the TOF resolution is much smaller compared to the voxel size,
  which is not typical for real PET scanners, but better for visualization.
"""

import importlib
import parallelproj_backend
import matplotlib.pyplot as plt
from utils import show_voxel_cube, show_lors, to_numpy

# %%
# import array API compatible library (CuPy if CUDA is available, otherwise NumPy).
if (
    parallelproj_backend.cuda_enabled == 1
    and importlib.util.find_spec("cupy") is not None
):
    import array_api_compat.cupy as xp

    dev = xp.cuda.Device(0)
else:
    import array_api_compat.numpy as xp

    dev = "cpu"


# %%
# Print backend and device info.
print(f"parallelproj_backend version: {parallelproj_backend.__version__}")
print(f"parallelproj_backend cuda enabled: {parallelproj_backend.cuda_enabled}")
print(f"using array API compatible library: {xp.__name__} on device {dev}")

# %%
# Define a mini sparse demo image.
image = xp.zeros((5, 5, 5), dtype=xp.float32, device=dev)
image[0, 2, 4] = 0.25
image[4, 2, 4] = 0.25
image[0, 0, 0] = 0.5
image[4, 4, 4] = 0.5

voxel_size = xp.asarray([2.0, 2.0, 2.0], device=dev, dtype=xp.float32)
img_origin = xp.asarray([-1.0, -1.0, -1.0], device=dev, dtype=xp.float32)

# %%
# Define LOR start and end points.
lor_start = xp.asarray(
    [[14.0, 3.5, 7.0], [-1, -1, -8], [7, -6, -6]],
    device=dev,
    dtype=xp.float32,
)
lor_end = xp.asarray(
    [[-8.0, 3.5, 7.0], [-1, -1, 14], [7, 12, 12]],
    device=dev,
    dtype=xp.float32,
)

# %%
# TOF parameter setup

# standard deviation of the TOF kernel in spatial units (mm)
sigma_tof = xp.asarray([2.0], device=dev, dtype=xp.float32)
# width of the TOF sinogram bins in mm
tofbin_width = float(sigma_tof[0]) / 2.0
# offset of the TOF center in mm
tof_center_offset = xp.asarray([0.0], device=dev, dtype=xp.float32)
# number of sigmas for truncation of Gaussian TOF kernel
num_sigmas = 3.0
# number of TOF bins needed to cover an LOR of 22mm length
num_tofbins = int(22 / tofbin_width) + 1

# %%
# Forward projection of the demo image.
img_fwd = xp.zeros((lor_start.shape[0], num_tofbins), dtype=xp.float32, device=dev)
parallelproj_backend.joseph3d_tof_sino_fwd(
    lor_start,
    lor_end,
    image,
    img_origin,
    voxel_size,
    img_fwd,
    tofbin_width,
    sigma_tof,
    tof_center_offset,
    num_tofbins,
    num_sigmas,
)
print(img_fwd)

# %%
fig = plt.figure(figsize=(12, 6), layout="constrained")
ax = fig.add_subplot(121, projection="3d")
show_voxel_cube(ax, image, voxel_size, img_origin)
show_lors(
    ax,
    lor_start,
    lor_end,
    labels=[f"LOR-{i}" for i in range(lor_start.shape[0])],
    num_tofbins=num_tofbins,
    tofbin_width=tofbin_width,
    tof_center_offset=tof_center_offset,
)

ax.set_xlim(-10, 15)
ax.set_ylim(-10, 15)
ax.set_zlim(-10, 15)
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")
ax.set_box_aspect([1, 1, 1])
fig.suptitle(
    "TOF sinogram forward projection of a 3D image using the Joseph's method",
    fontsize="medium",
)

ax_b = fig.add_subplot(122)

for i in range(img_fwd.shape[0]):
    ax_b.plot(to_numpy(img_fwd[i, ...]), ".-", label=f"LOR {i}")

ax_b.set_xlabel("TOF bin")
ax_b.set_ylabel("TOF projection profile")
ax_b.grid(ls=":")
ax_b.legend()

fig.show()

# %%
# back projection of ones along the same LORs.
back_ones = xp.zeros(image.shape, dtype=xp.float32, device=dev)
parallelproj_backend.joseph3d_tof_sino_back(
    lor_start,
    lor_end,
    back_ones,
    img_origin,
    voxel_size,
    xp.ones(img_fwd.shape, dtype=xp.float32, device=dev),
    tofbin_width,
    sigma_tof,
    tof_center_offset,
    num_tofbins,
    num_sigmas,
)

# %%
fig2 = plt.figure(figsize=(6, 6), layout="constrained")
ax2 = fig2.add_subplot(111, projection="3d")
show_voxel_cube(ax2, back_ones, voxel_size, img_origin)
show_lors(
    ax2,
    lor_start,
    lor_end,
    labels=[f"LOR-{i}" for i in range(lor_start.shape[0])],
    num_tofbins=num_tofbins,
    tofbin_width=tofbin_width,
    tof_center_offset=tof_center_offset,
)

ax2.set_xlim(-10, 15)
ax2.set_ylim(-10, 15)
ax2.set_zlim(-10, 15)
ax2.set_xlabel("x [mm]")
ax2.set_ylabel("y [mm]")
ax2.set_zlabel("z [mm]")
ax2.set_box_aspect([1, 1, 1])
ax2.set_title(
    "Back projection of ones for a set of LORs using the Joseph's method",
    fontsize="medium",
)

fig2.show()
