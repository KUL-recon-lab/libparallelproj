import matplotlib.pyplot as plt
import array_api_compat.numpy as xp

dev = "cpu"

import parallelproj_backend as ppb

image = xp.zeros((5, 5, 5), dtype=xp.float32, device=dev)
image[2, 2, 2] = 1.0
image[0, 4, 4] = 2.0
image[0, 0, 0] = 3.0

voxel_size = (2.0, 2.0, 2.0)
img_origin = (-1.0, -1.0, -1.0)

# %%
x = xp.arange(image.shape[0] + 1) * voxel_size[0] + (img_origin[0] - voxel_size[0] / 2)
y = xp.arange(image.shape[1] + 1) * voxel_size[1] + (img_origin[1] - voxel_size[1] / 2)
z = xp.arange(image.shape[2] + 1) * voxel_size[2] + (img_origin[2] - voxel_size[2] / 2)

X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")

num_bins = 64
norm_image = xp.astype(num_bins * image / image.max(), xp.int16)

# visualize the image using a 3D matplotlib plot
# every voxel should be rendered as a cube, so we use the "nearest" interpolation method
# the voxel values should be rendered as transparancy, so we use the "alpha" colormap
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
for b in xp.unique(norm_image):
    ax.voxels(
        X,
        Y,
        Z,
        norm_image == b,
        facecolors="C4",
        edgecolor=(0, 0, 0, 0.2),
        alpha=0.5 * (b / num_bins),
    )
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim(-8, 14)
ax.set_ylim(-8, 14)
ax.set_zlim(-8, 14)
ax.set_box_aspect([1, 1, 1])

fig.show()
