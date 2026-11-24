"""
Listmode TOF Point Projection (TOF Histo-Image)
===============================================

This example demonstrates the listmode TOF "point" projectors
:func:`parallelproj_core.tof_lm_point_fwd` and :func:`parallelproj_core.tof_lm_point_back`.

Every event is reduced to a single point: the center of its TOF bin on the LOR.
The forward projector trilinearly interpolates the image at that point, the
back projector (its adjoint) splats the event value onto the surrounding voxels.
Back projecting events with value 1 gives a TOF "histo-image", a fast, blurred
estimate of the activity distribution -- no ray tracing and no TOF kernel needed.

We simulate TOF events from a simple phantom with TOF bins much wider than the
voxels (as typical for PET scanners), show the ring artifacts this causes in a
plain histo-image, remove them by randomly dithering the events within their TOF
bins, and compare to the TOF listmode Joseph back projection of the same events.
"""

# sphinx_gallery_start_ignore
import pathlib, sys
try:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
except NameError:  # run by sphinx-gallery: docs/conf.py adds docs/examples to the path
    pass
# sphinx_gallery_end_ignore
import time

import numpy as np
import matplotlib.pyplot as plt
import parallelproj_core
from utils import COLORS, array_backend, set_style, to_numpy

xp, dev = array_backend()  # CuPy on the GPU (CUDA build) or NumPy on the CPU
set_style()
print(f"parallelproj_core {parallelproj_core.__version__}, using {xp.__name__} on {dev}")

# %%
# Image grid: 81 x 81 x 3 voxels of 3 mm, centered at the origin.
img_shape = (81, 81, 3)
voxel_size = np.array([3.0, 3.0, 3.0], dtype=np.float32)
img_origin = ((-np.array(img_shape) / 2 + 0.5) * voxel_size).astype(np.float32)

# %%
# Simulate events: emission points from a phantom (a large disk plus two hot spots
# in the central plane), random in-plane LOR directions on a ring with radius 400 mm.
rng = np.random.default_rng(1)
num_events = 1_000_000

centers = np.array([[0.0, 0.0], [40.0, 30.0], [-50.0, -20.0]])
radii = np.array([90.0, 12.0, 8.0])
fractions = np.array([0.7, 0.2, 0.1])

comp = rng.choice(3, size=num_events, p=fractions)
r = radii[comp] * np.sqrt(rng.uniform(0, 1, num_events))
phi = rng.uniform(0, 2 * np.pi, num_events)
emission = np.zeros((num_events, 3))
emission[:, 0] = centers[comp, 0] + r * np.cos(phi)
emission[:, 1] = centers[comp, 1] + r * np.sin(phi)

# LOR through the emission point with a random direction, endpoints on the ring
theta = rng.uniform(0, np.pi, num_events)
u = np.stack([np.cos(theta), np.sin(theta), np.zeros(num_events)], axis=1)
b = (emission * u).sum(1)
c = (emission**2).sum(1) - 400.0**2
t_plus = -b + np.sqrt(b**2 - c)
t_minus = -b - np.sqrt(b**2 - c)
lor_start = (emission + t_minus[:, None] * u).astype(np.float32)
lor_end = (emission + t_plus[:, None] * u).astype(np.float32)

# %%
# TOF binning as done by a scanner: signed distance of the (blurred) emission
# point from the LOR midpoint towards the end point, TOF resolution 30 mm FWHM,
# 10 mm TOF bins -- much wider than the 3 mm voxels.
tof_bin_width = 10.0
num_tof_bins = 81
tof_sigma = 30.0 / 2.355
midpoint = 0.5 * (lor_start + lor_end)
dist = ((emission - midpoint) * u).sum(1) + rng.normal(0, tof_sigma, num_events)
tof_bin = np.clip(np.round(dist / tof_bin_width + (num_tof_bins - 1) / 2), 0, num_tof_bins - 1)
tof_bin = tof_bin.astype(np.int16)

# %%
# Dithering within the TOF bins
# -----------------------------
# The point projector puts every event exactly at the center of its TOF bin. All
# events of a LOR therefore land on points spaced by the TOF bin width. If the TOF
# bins are much wider than the voxels, this spacing shows up as ring / stripe
# artifacts in the histo-image.
#
# A quick fix that needs no change of the projector: split every TOF bin into
# ``k`` sub-bins and assign each event to a random sub-bin of its bin.
# Sub-bin ``s`` of bin ``i`` gets the index ``i * k + s``, the sub-bins have width
# ``tof_bin_width / k`` and there are ``num_tof_bins * k`` of them. The sub-bin
# centers evenly cover the original bin, and the TOF center offset stays the same.
#
# .. note::
#   - choose ``k`` such that ``tof_bin_width / k`` is about the voxel size or
#     smaller, and ``num_tof_bins * k <= 32767`` (int16 bin index)
#   - dithering adds a uniform position spread with a standard deviation of
#     ``tof_bin_width / sqrt(12)`` (2.9 mm here), small compared to the TOF resolution
#   - the result depends on the random draw (use a seed for reproducible results).
#     For a fixed draw, forward and back projector stay exactly adjoint, so draw
#     once when using the dithered events in an iterative reconstruction.
k = 4
tof_bin_dithered = (tof_bin.astype(np.int32) * k + rng.integers(0, k, num_events)).astype(np.int16)

# %%
# Move the data to the device.
d_start = xp.asarray(lor_start, device=dev)
d_end = xp.asarray(lor_end, device=dev)
d_origin = xp.asarray(img_origin, device=dev)
d_voxsize = xp.asarray(voxel_size, device=dev)
d_offset = xp.asarray([0.0], dtype=xp.float32, device=dev)
d_ones = xp.ones(num_events, dtype=xp.float32, device=dev)


def histo_image(tof_bins, bin_width, num_bins):
    """TOF histo-image of all events (event value 1) with the point back projector"""
    img = xp.zeros(img_shape, dtype=xp.float32, device=dev)
    parallelproj_core.tof_lm_point_back(
        d_start, d_end, img, d_origin, d_voxsize, d_ones,
        bin_width, d_offset, xp.asarray(tof_bins, device=dev), num_bins,
    )
    return img


# %%
# TOF histo-images without and with dithering.
histo_plain = histo_image(tof_bin, tof_bin_width, num_tof_bins)
histo_dithered = histo_image(tof_bin_dithered, tof_bin_width / k, num_tof_bins * k)

# %%
# For comparison: TOF listmode Joseph back projection of the same events
# (original 10 mm TOF bins, Gaussian TOF kernel).
d_tof_bin = xp.asarray(tof_bin, device=dev)
d_tof_sigma = xp.asarray([tof_sigma], dtype=xp.float32, device=dev)


def joseph_back():
    img = xp.zeros(img_shape, dtype=xp.float32, device=dev)
    parallelproj_core.joseph3d_tof_lm_back(
        d_start, d_end, img, d_origin, d_voxsize, d_ones,
        tof_bin_width, d_tof_sigma, d_offset, d_tof_bin, num_tof_bins,
    )
    return img


joseph_image = joseph_back()

# %%
# Timing
# ------
# GPU work can run asynchronously: a call may return before the GPU is done
# (e.g. projections with ``stream=...``, and all PyTorch / CuPy operations). To
# get meaningful timings, synchronize the GPU before reading the clock, and
# exclude the first call (one-time GPU initialization) with a warm-up call.
# Alternatives are CUDA events, or ``torch.utils.benchmark.Timer`` /
# ``cupyx.profiler.benchmark``, which take care of both.


def synchronize():
    """wait until all queued GPU work is done (no-op for NumPy arrays)"""
    if hasattr(dev, "synchronize"):  # CuPy device
        dev.synchronize()


def best_time(fn, repeats=3):
    """best wall-clock time of fn() over a few runs, after a warm-up call"""
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        synchronize()
        t0 = time.perf_counter()
        fn()
        synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


# (both timings use event data that is already on the device)
t_point = best_time(lambda: histo_image(d_tof_bin, tof_bin_width, num_tof_bins))
t_joseph = best_time(joseph_back)

print(f"tof_lm_point_back:    {1e3 * t_point:8.2f} ms")
print(f"joseph3d_tof_lm_back: {1e3 * t_joseph:8.2f} ms ({t_joseph / t_point:.0f}x slower)")
print(f"events in the histo-image: {float(xp.sum(histo_plain)):.0f} of {num_events}")

# %%
# Show the central plane of the images and a profile through the large hot spot (y = 30 mm).
# The plain histo-image shows rings (centered at the scanner center) and stripes in the hot
# spots. Dithering removes them, and the result is close to the Joseph back projection.
extent = [img_origin[0] - voxel_size[0] / 2, img_origin[0] + (img_shape[0] - 0.5) * voxel_size[0],
          img_origin[1] - voxel_size[1] / 2, img_origin[1] + (img_shape[1] - 0.5) * voxel_size[1]]
j_prof = int(round((30.0 - img_origin[1]) / voxel_size[1]))  # image row at y = 30 mm
x_prof = img_origin[0] + np.arange(img_shape[0]) * voxel_size[0]

fig, ax = plt.subplots(1, 4, figsize=(17, 4.5), layout="constrained",
                       gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35]))
images = {
    f"(a) tof_lm_point_back\n{tof_bin_width:.0f} mm TOF bins": (histo_plain, COLORS["red"]),
    f"(b) tof_lm_point_back, dithered\n({k} sub-bins of {tof_bin_width / k:.1f} mm)": (histo_dithered, COLORS["green"]),
    f"(c) joseph3d_tof_lm_back\n{tof_bin_width:.0f} mm TOF bins": (joseph_image, COLORS["blue"]),
}
for a, (title, (img, col)) in zip(ax, images.items()):
    sl = to_numpy(img)[:, :, 1]
    a.imshow(sl.T, origin="lower", cmap="Greys", extent=extent)
    a.axhline(30.0, color=col, lw=0.8, ls=(0, (3, 2)))
    a.set_title(title)
    a.set_xlabel("x [mm]")
    a.set_ylabel("y [mm]")
    prof = sl[:, j_prof]
    ax[3].plot(x_prof, prof / prof.mean(), color=col, label=title.split("\n")[0][4:])
ax[3].set_xlabel("x [mm]")
ax[3].set_ylabel("profile at y = 30 mm (/ mean)")
ax[3].legend(loc="upper left")
ax[3].set_title("(d) profiles through the large hot spot")
fig.show()
