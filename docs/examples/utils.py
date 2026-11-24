"""Helpers shared by the gallery examples (not an example itself).

- ``to_numpy``: convert NumPy / CuPy / PyTorch arrays to NumPy (for plotting),
- ``array_backend``: CuPy on the GPU if available (CUDA build), otherwise NumPy on the CPU,
- a simple 2D geometry (one image plane, LORs in that plane): ``disk_phantom``, ``lor_2d``,
- ``set_style`` / ``COLORS`` / ``show_image``: a common, readable plotting style.
"""

import importlib.util
import math

import numpy as np
import matplotlib as mpl

# colors used in all examples (the same as in the figures of the documentation)
COLORS = dict(blue="#2F6DB5", orange="#E07B39", green="#3A9D5D", red="#C8433A", purple="#7A5BA6",
              gray="#6B6B6B", light="#E9EDF2", mid="#C9D1DB")


def set_style():
    """a common, light plotting style for all examples"""
    mpl.rcParams.update({
        "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": COLORS["gray"], "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": COLORS["gray"], "ytick.color": COLORS["gray"], "lines.linewidth": 1.6,
        "legend.frameon": False, "legend.fontsize": 8.5, "figure.facecolor": "white",
    })


def to_numpy(x):
    """Convert x (NumPy / CuPy / torch.Tensor, CPU or GPU) to a NumPy array on the CPU."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):  # cupy
        return x.get()
    return np.asarray(x)


def array_backend():
    """(xp, dev): CuPy on GPU 0 if libparallelproj was built with CUDA and CuPy is installed,
    otherwise NumPy on the CPU (both wrapped by array_api_compat)"""
    import parallelproj_core

    if parallelproj_core.cuda_enabled == 1 and importlib.util.find_spec("cupy") is not None:
        import array_api_compat.cupy as xp

        return xp, xp.cuda.Device(0)
    import array_api_compat.numpy as xp

    return xp, "cpu"


def disk_phantom(n=64, voxel_size=4.0):
    """2D phantom in one image plane (image shape (n, n, 1)): a disk of radius 100 mm with
    value 1 and two hot spots (values 4 and 3). Returns (image, image_origin, voxel_size,
    extent) as float32 NumPy arrays; extent is the (x, y) range for plt.imshow."""
    vs = np.array([voxel_size, voxel_size, voxel_size], np.float32)
    origin = np.array([-(n - 1) / 2 * voxel_size, -(n - 1) / 2 * voxel_size, 0.0], np.float32)
    c = origin[0] + np.arange(n) * voxel_size  # voxel centers along x and y
    X, Y = np.meshgrid(c, c, indexing="ij")
    img = (X**2 + Y**2 < 100**2).astype(np.float32)
    img += 3 * ((X - 40) ** 2 + (Y - 20) ** 2 < 15**2)
    img += 2 * ((X + 45) ** 2 + (Y + 30) ** 2 < 10**2)
    half = n / 2 * voxel_size
    return img[:, :, None].astype(np.float32), origin, vs, (-half, half, -half, half)


def lor_2d(phi_deg, r, radius=180.0):
    """end points (start, end) of the LOR in the plane z = 0 with direction angle phi_deg whose
    closest point to the center is at signed distance r; both end points lie on a circle
    (the "scanner") of the given radius"""
    ph = math.radians(phi_deg)
    d = np.array([math.cos(ph), math.sin(ph), 0.0])
    nrm = np.array([-math.sin(ph), math.cos(ph), 0.0])
    h = math.sqrt(radius**2 - r**2)
    return (r * nrm - h * d).astype(np.float32), (r * nrm + h * d).astype(np.float32)


def show_image(ax, img, extent, cmap="Greys", title=None, vmax=None):
    """show the (single) plane of an image of shape (n0, n1, 1); x horizontal, y vertical"""
    ax.imshow(to_numpy(img)[:, :, 0].T, origin="lower", extent=extent, cmap=cmap, vmax=vmax)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    if title:
        ax.set_title(title)
