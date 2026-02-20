import numpy as np
import matplotlib.pyplot as plt


def show_voxel_cube(ax, img, voxel_size, img_origin, num_bins=64):
    x = np.arange(img.shape[0] + 1) * voxel_size[0] + (
        img_origin[0] - voxel_size[0] / 2
    )
    y = np.arange(img.shape[1] + 1) * voxel_size[1] + (
        img_origin[1] - voxel_size[1] / 2
    )
    z = np.arange(img.shape[2] + 1) * voxel_size[2] + (
        img_origin[2] - voxel_size[2] / 2
    )

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    norm_image = np.asarray(num_bins * img / img.max()).astype(np.int16)

    for b in np.unique(norm_image):
        ax.voxels(
            X,
            Y,
            Z,
            norm_image == b,
            facecolors="C4",
            edgecolor=(0.2, 0.2, 0.2, 0.3),
            alpha=0.5 * (b / num_bins),
        )


def show_lors(ax, lor_start, lor_end, labels=None):
    for i in range(lor_start.shape[0]):
        col = plt.cm.tab10(i)
        # plor the start and end points of the line of response as red dots
        ax.scatter(
            lor_start[i, 0],
            lor_start[i, 1],
            lor_start[i, 2],
            color=col,
            s=50,
            marker="^",
        )
        ax.scatter(
            lor_end[i, 0], lor_end[i, 1], lor_end[i, 2], color=col, s=50, marker="X"
        )
        # plot the line of response as a red line
        ax.plot(
            [lor_start[i, 0], lor_end[i, 0]],
            [lor_start[i, 1], lor_end[i, 1]],
            [lor_start[i, 2], lor_end[i, 2]],
            color=col,
            linewidth=2,
        )

        if labels is not None:
            ax.text(
                lor_start[i, 0],
                lor_start[i, 1],
                lor_start[i, 2],
                labels[i],
                color="black",
                fontsize=10,
            )
