import matplotlib.pyplot as plt
import array_api_compat.numpy as xp

dev = "cpu"

import parallelproj_backend as ppb

image = xp.zeros((5, 10, 5), dtype=xp.float32, device=dev)
image[2, 2, 2] += 1.0

voxel_size = (2.0, 1.0, 2.0)
img_origin = (-1.0, -1.0, -1.0)

x = xp.arange(image.shape[0] + 1) * voxel_size[0] + (img_origin[0] - voxel_size[0] / 2)
y = xp.arange(image.shape[1] + 1) * voxel_size[1] + (img_origin[1] - voxel_size[1] / 2)
z = xp.arange(image.shape[2] + 1) * voxel_size[2] + (img_origin[2] - voxel_size[2] / 2)

X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")

# visualize the image using a 3D matplotlib plot
# every voxel should be rendered as a cube, so we use the "nearest" interpolation method
# the voxel values should be rendered as transparancy, so we use the "alpha" colormap
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.voxels(X, Y, Z, image == 0, facecolors="C0", edgecolor=(0, 0, 0, 0.3), alpha=0.0)
ax.voxels(X, Y, Z, image > 0, facecolors="C0", edgecolor=(0, 0, 0, 0.3), alpha=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Image")

ax.set_xlim(-5, 12)
ax.set_ylim(-5, 12)
ax.set_zlim(-5, 12)
ax.set_box_aspect([1, 1, 1])

plt.show()
