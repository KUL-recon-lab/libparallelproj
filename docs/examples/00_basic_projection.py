import matplotlib.pyplot as plt
import array_api_compat.numpy as xp

dev = "cpu"

import parallelproj_backend as ppb

image = xp.zeros((5, 5, 5), dtype=xp.float32, device=dev)
image[2, 2, 2] = 1.0
image[1, 4, 4] = 2.0
image[0, 0, 0] = 3.0

voxel_size = xp.asarray([2.0, 2.0, 2.0], device=dev, dtype=xp.float32)
img_origin = xp.asarray([-1.0, -1.0, -1.0], device=dev, dtype=xp.float32)

# %%
lor_start = xp.asarray(
    [[-6.0, 4.0, 3.0], [-1, -1, -6], [2, 12, 6]], device=dev, dtype=xp.float32
)
lor_end = xp.asarray(
    [[12.0, 4.0, 3.0], [-1, -1, 12], [2, -6, 6]], device=dev, dtype=xp.float32
)

# forward projection
img_fwd = xp.zeros(lor_start.shape[0], dtype=xp.float32, device=dev)
ppb.joseph3d_fwd(lor_start, lor_end, image, img_origin, voxel_size, img_fwd)
print(img_fwd)

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

for i in range(lor_start.shape[0]):
    col = plt.cm.tab10(i)
    # plor the start and end points of the line of response as red dots
    ax.scatter(
        lor_start[i, 0], lor_start[i, 1], lor_start[i, 2], color=col, s=50, marker="^"
    )
    ax.scatter(lor_end[i, 0], lor_end[i, 1], lor_end[i, 2], color=col, s=50, marker="X")
    # plot the line of response as a red line
    ax.plot(
        [lor_start[i, 0], lor_end[i, 0]],
        [lor_start[i, 1], lor_end[i, 1]],
        [lor_start[i, 2], lor_end[i, 2]],
        color=col,
        linewidth=2,
    )

plt.show()
