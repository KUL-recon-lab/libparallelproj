import numpy as np
import matplotlib.pyplot as plt

import numpy as np


def to_numpy(x):
    """
    Convert x (NumPy / CuPy / torch.Tensor) to a NumPy ndarray on CPU.

    - torch CPU: zero-copy when possible (via .numpy()).
    - torch CUDA: moves to CPU, then converts.
    - cupy: moves to host (asnumpy).
    """
    # Fast path: already NumPy
    if isinstance(x, np.ndarray):
        return x

    # PyTorch
    try:
        import torch

        if isinstance(x, torch.Tensor):
            # detach to avoid autograd surprises; then ensure CPU
            if x.device.type != "cpu":
                x = x.detach().to("cpu")
            else:
                x = x.detach()
            return x.numpy()
    except Exception:
        pass

    # CuPy
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)  # same as x.get()
    except Exception:
        pass

    # Fallbacks:
    # - Works for objects implementing __array__ (e.g., some CPU array-likes)
    # - Will NOT magically pull from GPU for CuPy/torch CUDA
    return np.asarray(x)


def show_voxel_cube(ax, img, voxel_size, img_origin, num_bins=64):
    x = np.arange(img.shape[0] + 1) * float(voxel_size[0]) + (
        float(img_origin[0]) - float(voxel_size[0]) / 2
    )
    y = np.arange(img.shape[1] + 1) * float(voxel_size[1]) + (
        float(img_origin[1]) - float(voxel_size[1]) / 2
    )
    z = np.arange(img.shape[2] + 1) * float(voxel_size[2]) + (
        float(img_origin[2]) - float(voxel_size[2]) / 2
    )

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    norm_image = to_numpy(img)
    norm_image = np.asarray(num_bins * norm_image / np.max(norm_image)).astype(np.int16)

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


def show_lors(
    ax,
    lor_start,
    lor_end,
    labels=None,
    num_tofbins=None,
    tofbin_width=None,
    tof_center_offset=None,
):
    l_start = to_numpy(lor_start)
    l_end = to_numpy(lor_end)

    for i in range(lor_start.shape[0]):
        col = plt.cm.tab10(i)
        # plor the start and end points of the line of response as red dots
        ax.scatter(
            l_start[i, 0],
            l_start[i, 1],
            l_start[i, 2],
            color=col,
            s=50,
            marker="^",
        )
        ax.scatter(l_end[i, 0], l_end[i, 1], l_end[i, 2], color=col, s=50, marker="X")
        # plot the line of response as a red line
        ax.plot(
            [l_start[i, 0], l_end[i, 0]],
            [l_start[i, 1], l_end[i, 1]],
            [l_start[i, 2], l_end[i, 2]],
            color=col,
            linewidth=2,
        )

        if num_tofbins is not None and tofbin_width is not None:
            # calculate the LOR center
            center = (l_start[i] + l_end[i]) / 2
            # calculat the unit vector along the LOR
            lor_vec = l_end[i] - l_start[i]
            lor_unit_vec = lor_vec / np.linalg.norm(lor_vec)

            if tof_center_offset is not None:
                to = to_numpy(tof_center_offset)
                if len(to) == 1:
                    center += to[0] * lor_unit_vec
                elif len(to) == lor_start.shape[0]:
                    center += to[i] * lor_unit_vec
                else:
                    raise ValueError(
                        "tof_center_offset should be either a scalar or an array of shape (num_lors,)"
                    )

            # calculate the signed tof bins
            signed_tof_bins = np.arange(num_tofbins) - num_tofbins / 2 + 0.5

            for it, signed_tof_bin in enumerate(signed_tof_bins):
                tofbin_center = center + signed_tof_bin * tofbin_width * lor_unit_vec
                ax.scatter(
                    tofbin_center[0],
                    tofbin_center[1],
                    tofbin_center[2],
                    color=col,
                    s=20,
                    marker="o",
                    alpha=0.5,
                )

        if labels is not None:
            ax.text(
                l_start[i, 0],
                l_start[i, 1],
                l_start[i, 2],
                labels[i],
                color="black",
                fontsize=10,
            )
