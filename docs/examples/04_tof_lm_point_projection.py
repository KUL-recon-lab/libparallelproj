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

We simulate TOF events from a simple phantom, and compare the histo-image to the
TOF listmode Joseph back projection of the same events.
"""

import importlib
import time
import numpy as np
import parallelproj_core
import matplotlib.pyplot as plt

# %%
# import array API compatible library (CuPy if CUDA is available, otherwise NumPy).
if parallelproj_core.cuda_enabled == 1 and importlib.util.find_spec("cupy") is not None:
    import array_api_compat.cupy as xp

    dev = xp.cuda.Device(0)
else:
    import array_api_compat.numpy as xp

    dev = "cpu"

print(f"parallelproj_core version: {parallelproj_core.__version__}")
print(f"using array API compatible library: {xp.__name__} on device {dev}")

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
# TOF binning: signed distance of the (blurred) emission point from the LOR
# midpoint towards the end point, TOF resolution 30 mm FWHM, 3 mm bins.
#
# .. note::
#   All events of a LOR land on the TOF bin centers, which are spaced by the TOF
#   bin width. If the TOF bins are much wider than the voxels, this spacing shows
#   up as ring / stripe artifacts in the histo-image. Use TOF bins of about the
#   voxel size (e.g. by subdividing the bins of the scanner and assigning events
#   randomly within their bin) or voxels of about the TOF bin width.
tof_bin_width = 3.0
num_tof_bins = 271
tof_sigma = 30.0 / 2.355
midpoint = 0.5 * (lor_start + lor_end)
dist = ((emission - midpoint) * u).sum(1) + rng.normal(0, tof_sigma, num_events)
tof_bin = np.clip(np.round(dist / tof_bin_width + (num_tof_bins - 1) / 2), 0, num_tof_bins - 1)
tof_bin = tof_bin.astype(np.int16)

# %%
# Move the data to the device.
d_start = xp.asarray(lor_start, device=dev)
d_end = xp.asarray(lor_end, device=dev)
d_origin = xp.asarray(img_origin, device=dev)
d_voxsize = xp.asarray(voxel_size, device=dev)
d_tof_bin = xp.asarray(tof_bin, device=dev)
d_offset = xp.asarray([0.0], dtype=xp.float32, device=dev)
d_ones = xp.ones(num_events, dtype=xp.float32, device=dev)

# %%
# TOF histo-image with the point back projector.
histo_image = xp.zeros(img_shape, dtype=xp.float32, device=dev)
t0 = time.perf_counter()
parallelproj_core.tof_lm_point_back(
    d_start, d_end, histo_image, d_origin, d_voxsize, d_ones,
    tof_bin_width, d_offset, d_tof_bin, num_tof_bins,
)
t_point = time.perf_counter() - t0

# %%
# For comparison: TOF listmode Joseph back projection of the same events.
joseph_image = xp.zeros(img_shape, dtype=xp.float32, device=dev)
t0 = time.perf_counter()
parallelproj_core.joseph3d_tof_lm_back(
    d_start, d_end, joseph_image, d_origin, d_voxsize, d_ones,
    tof_bin_width, xp.asarray([tof_sigma], dtype=xp.float32, device=dev), d_offset,
    d_tof_bin, num_tof_bins,
)
t_joseph = time.perf_counter() - t0

print(f"tof_lm_point_back:    {t_point:.3f} s")
print(f"joseph3d_tof_lm_back: {t_joseph:.3f} s")
print(f"events in the histo-image: {float(xp.sum(histo_image)):.0f} of {num_events}")

# %%
# Show the central plane of both images.
fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
for a, img, title in zip(
    ax,
    (histo_image, joseph_image),
    ("tof_lm_point_back\n(TOF histo-image)", "joseph3d_tof_lm_back"),
):
    sl = np.asarray(img[:, :, 1].get() if hasattr(img, "get") else img[:, :, 1])
    a.imshow(sl.T, origin="lower", cmap="Greys")
    a.set_title(title)
    a.set_axis_off()
fig.show()
