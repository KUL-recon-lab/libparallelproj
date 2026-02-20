"""
Non-TOF Joseph Forward and Back Projection
==========================================

This minimal example demonstrates how to call python API
for the non-TOF Joseph forward and back projection functions, which are implemented in
:func:`parallelproj_backend.joseph3d_fwd` and :func:`parallelproj_backend.joseph3d_back`.
"""

import parallelproj_backend

import array_api_compat.numpy as xp

dev = "cpu"

import matplotlib.pyplot as plt

from utils import show_voxel_cube, show_lors


# %%
image = xp.zeros((5, 5, 5), dtype=xp.float32, device=dev)
image[1, 2, 4] = 0.25
image[2, 2, 4] = 1.0
image[0, 0, 0] = 0.5
image[4, 4, 4] = 0.5

voxel_size = xp.asarray([2.0, 2.0, 2.0], device=dev, dtype=xp.float32)
img_origin = xp.asarray([-1.0, -1.0, -1.0], device=dev, dtype=xp.float32)

# %%
lor_start = xp.asarray(
    [[12.0, 3.5, 7.0], [-1, -1, -6], [7, -4, -4]],
    device=dev,
    dtype=xp.float32,
)
lor_end = xp.asarray(
    [[-6.0, 3.5, 7.0], [-1, -1, 12], [7, 10, 10]],
    device=dev,
    dtype=xp.float32,
)

# forward projection
img_fwd = xp.zeros(lor_start.shape[0], dtype=xp.float32, device=dev)
parallelproj_backend.joseph3d_fwd(
    lor_start, lor_end, image, img_origin, voxel_size, img_fwd
)
print(img_fwd)

# %%
# visualize the image using a 3D matplotlib plot
# every voxel should be rendered as a cube, so we use the "nearest" interpolation method
# the voxel values should be rendered as transparancy, so we use the "alpha" colormap
fig = plt.figure(figsize=(6, 6), layout="constrained")
ax = fig.add_subplot(111, projection="3d")
show_voxel_cube(ax, image, voxel_size, img_origin)
show_lors(
    ax, lor_start, lor_end, labels=[f"LOR-{i}: {x:.2f}" for i, x in enumerate(img_fwd)]
)

ax.set_xlim(-8, 13)
ax.set_ylim(-8, 13)
ax.set_zlim(-8, 13)
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")
ax.set_box_aspect([1, 1, 1])
ax.set_title(
    "Forward projection of a 3D image using the Joseph's method", fontsize="medium"
)
fig.show()

# %%
back_ones = xp.zeros(image.shape, dtype=xp.float32, device=dev)
parallelproj_backend.joseph3d_back(
    lor_start,
    lor_end,
    back_ones,
    img_origin,
    voxel_size,
    xp.ones(img_fwd.shape, dtype=xp.float32, device=dev),
)

fig2 = plt.figure(figsize=(6, 6), layout="constrained")
ax2 = fig2.add_subplot(111, projection="3d")
show_voxel_cube(ax2, back_ones, voxel_size, img_origin)
show_lors(ax2, lor_start, lor_end)

ax2.set_xlim(-8, 13)
ax2.set_ylim(-8, 13)
ax2.set_zlim(-8, 13)
ax2.set_xlabel("x [mm]")
ax2.set_ylabel("y [mm]")
ax2.set_zlabel("z [mm]")
ax2.set_box_aspect([1, 1, 1])
ax2.set_title(
    "Back projection of ones for a set of LORs using the Joseph's method",
    fontsize="medium",
)

fig2.show()
