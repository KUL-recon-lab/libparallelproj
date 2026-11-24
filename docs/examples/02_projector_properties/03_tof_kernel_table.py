"""
TOF Projections with a Tabulated (Non-Gaussian) TOF Kernel
==========================================================

The TOF projectors with the suffix ``_tab``
(:func:`parallelproj_core.joseph3d_tof_sino_fwd_tab`,
:func:`parallelproj_core.joseph3d_tof_sino_back_tab`,
:func:`parallelproj_core.joseph3d_tof_lm_fwd_tab`,
:func:`parallelproj_core.joseph3d_tof_lm_back_tab`) accept an arbitrary TOF kernel, e.g.
one measured for a detector, instead of a Gaussian. This example shows

- how to turn a measured distribution of **arrival time differences (in ps)** into the
  table of the **cumulative kernel F(u) (in mm)** that the projectors expect,
- the sign convention for asymmetric kernels,
- how the per-LOR **kernel scale** makes the kernel wider or narrower,
- the TOF profiles of a point source with a Gaussian and with the tabulated kernel.

What goes into the table
------------------------

The table holds the cumulative TOF kernel F(u) of the detector pair on a uniform grid of
u from ``u_min`` to ``u_max``, where u is the distance (in the units of the LOR coordinates,
here mm) between the TOF position and the true annihilation point, **positive towards the
end point of the LOR**. The TOF bin width is *not* part of the table: the projectors
integrate the kernel over each TOF bin themselves. See the section on tabulated TOF kernels
in :doc:`/theory/tof`.

For an arrival time difference :math:`\\Delta t = t_\\text{start} - t_\\text{end}` (arrival
time at the detector at ``lor_start`` minus arrival time at the detector at ``lor_end``),

.. math::

   u = \\frac{c \\, \\Delta t}{2} \\approx 0.1499 \\, \\frac{\\text{mm}}{\\text{ps}} \\, \\Delta t .

An annihilation closer to the end detector reaches it earlier, so :math:`\\Delta t > 0`
and :math:`u > 0`.
"""

# %%
import importlib.util
import math
import numpy as np
import matplotlib.pyplot as plt
import parallelproj_core
# sphinx_gallery_start_ignore
# the helper modules (utils.py, gpu_utils.py) are in the parent directory (docs/examples)
import pathlib, sys
try:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
except NameError:  # run by sphinx-gallery: docs/conf.py adds docs/examples to the path
    pass
# sphinx_gallery_end_ignore
from utils import to_numpy

if parallelproj_core.cuda_enabled == 1 and importlib.util.find_spec("cupy") is not None:
    import array_api_compat.cupy as xp

    dev = xp.cuda.Device(0)
else:
    import array_api_compat.numpy as xp

    dev = "cpu"

print(f"parallelproj_core {parallelproj_core.__version__}, using {xp.__name__} on {dev}")

# %%
# A "measured" kernel: arrival time differences of a point source
# ---------------------------------------------------------------
# We simulate the arrival time differences of a point source in the middle between two
# detectors. The detector at ``lor_start`` has a fast component (Gaussian jitter) and a
# slow component (an exponential delay of some events, e.g. scintillation light of a slow
# material); the detector at ``lor_end`` only has the Gaussian jitter. The delays of the
# start detector make :math:`\Delta t = t_\text{start} - t_\text{end}` larger, so the
# kernel has a tail towards positive u (towards ``lor_end``).

rng = np.random.default_rng(1)
num_coinc = 2_000_000
jitter_ps = 90.0  # Gaussian jitter (sigma) of each detector in ps
slow_fraction, slow_decay_ps = 0.35, 120.0  # fraction and mean delay of slow events (start detector)

t_start = rng.normal(0, jitter_ps, num_coinc) + np.where(
    rng.uniform(size=num_coinc) < slow_fraction, rng.exponential(slow_decay_ps, num_coinc), 0.0
)
t_end = rng.normal(0, jitter_ps, num_coinc)
dt_ps = t_start - t_end

# histogram of the time differences (this is what a measurement would give)
hist_edges_ps = np.arange(-450.0, 900.0 + 5.0, 5.0)  # 5 ps bins
counts, _ = np.histogram(dt_ps, bins=hist_edges_ps)

# %%
# From the histogram to the table of F(u)
# ---------------------------------------
# The cumulative sum of the histogram counts is F at the histogram bin edges (starting with
# 0 at the first edge). Linear interpolation of F between the edges corresponds exactly to
# the histogram as a (piecewise constant) density, so the table can be used as it is.
# The range of the histogram is the support of the kernel: choose it wide enough to contain
# (almost) all events, but not much wider (see the note on the 64-bin limit at the end).
# Here, -450 ps to 900 ps contains all but ~0.02 % of the events. F does not need to be
# normalized; the projectors divide by F(u_max) - F(u_min).

C_MM_PER_PS = 0.299792458 / 2  # c / 2 in mm / ps

tof_kernel_cdf = np.concatenate([[0.0], np.cumsum(counts)]).astype(np.float32)
tof_kernel_cdf /= tof_kernel_cdf[-1]
u_min = float(hist_edges_ps[0] * C_MM_PER_PS)
u_max = float(hist_edges_ps[-1] * C_MM_PER_PS)
print(f"table: {tof_kernel_cdf.shape[0]} points from u_min = {u_min:.1f} mm to u_max = {u_max:.1f} mm")

# for comparison: a Gaussian kernel with the same standard deviation (in mm)
sigma_mm = float(np.std(dt_ps)) * C_MM_PER_PS
print(f"standard deviation of the measured kernel: {sigma_mm:.1f} mm (FWHM {2.355 * sigma_mm:.1f} mm)")

# %%
u_edges = hist_edges_ps * C_MM_PER_PS
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
ax[0].stairs(counts / counts.sum() / np.diff(u_edges), u_edges, label="measured kernel f(u)")
uu = np.linspace(u_min, u_max, 500)
ax[0].plot(uu, np.exp(-0.5 * (uu / sigma_mm) ** 2) / (sigma_mm * math.sqrt(2 * math.pi)),
           "--", label="Gaussian, same std. dev.")
ax[0].set_xlabel("u [mm] (> 0: towards lor_end)")
ax[0].set_ylabel("density [1/mm]")
ax[0].legend()
ax[1].plot(np.linspace(u_min, u_max, tof_kernel_cdf.shape[0]), tof_kernel_cdf)
ax[1].set_xlabel("u [mm]")
ax[1].set_ylabel("F(u) (table)")
for a in ax:
    a.grid(ls=":")
fig.show()

# %%
# Note that the mean of the measured kernel is not 0 (the slow events shift it towards
# positive u). The table describes the kernel *relative to the true position*, so this
# shift is part of the kernel. A known constant shift of a detector pair could also be
# described by the TOF center offset instead.
#
# TOF profiles of a point source
# ------------------------------
# A single LOR along the x-axis from x = -150 mm (``lor_start``) to x = +150 mm (``lor_end``)
# and a point source at x = 20 mm. We compare the TOF sinogram profiles for the Gaussian
# kernel, the tabulated kernel, and the tabulated kernel scaled by 1.5 (same shape, 50 %
# wider).

voxel_size = xp.asarray([1.0, 1.0, 1.0], dtype=xp.float32, device=dev)
img_shape = (201, 1, 1)
img_origin = xp.asarray([-100.0, 0.0, 0.0], dtype=xp.float32, device=dev)
image = xp.zeros(img_shape, dtype=xp.float32, device=dev)
image[120, 0, 0] = 1.0  # x = -100 + 120 = 20 mm

lor_start = xp.asarray([[-150.0, 0.0, 0.0]], dtype=xp.float32, device=dev)
lor_end = xp.asarray([[150.0, 0.0, 0.0]], dtype=xp.float32, device=dev)
tof_bin_width = 5.0  # mm
num_tof_bins = 161  # +-402 mm: covers the support of the scaled kernel (up to 1.5 x 135 mm)
tof_center_offset = xp.asarray([0.0], dtype=xp.float32, device=dev)

p_gauss = xp.zeros((1, num_tof_bins), dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_tof_sino_fwd(
    lor_start, lor_end, image, img_origin, voxel_size, p_gauss, tof_bin_width,
    xp.asarray([sigma_mm], dtype=xp.float32, device=dev), tof_center_offset, num_tof_bins, num_sigmas=4.0,
)

cdf_dev = xp.asarray(tof_kernel_cdf, dtype=xp.float32, device=dev)
profiles = {}
for scale in [1.0, 1.5]:
    p = xp.zeros((1, num_tof_bins), dtype=xp.float32, device=dev)
    parallelproj_core.joseph3d_tof_sino_fwd_tab(
        lor_start, lor_end, image, img_origin, voxel_size, p, tof_bin_width, tof_center_offset, num_tof_bins,
        cdf_dev, u_min, u_max, xp.asarray([scale], dtype=xp.float32, device=dev),
    )
    profiles[scale] = to_numpy(p)[0]

# position of the TOF bin centers along the LOR (bin (num_tof_bins - 1) / 2 is the LOR center)
x_bins = (np.arange(num_tof_bins) - (num_tof_bins - 1) / 2) * tof_bin_width

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(x_bins, to_numpy(p_gauss)[0], ".-", label="Gaussian (same std. dev.)")
ax.plot(x_bins, profiles[1.0], ".-", label="tabulated, scale 1")
ax.plot(x_bins, profiles[1.5], ".-", label="tabulated, scale 1.5")
ax.axvline(20.0, color="k", ls=":", label="point source")
ax.set_xlabel("TOF bin center along the LOR [mm]")
ax.set_ylabel("TOF sinogram value")
ax.set_xlim(-60, 120)
ax.grid(ls=":")
ax.legend()
fig.show()

# %%
# The tabulated profiles have the tail of the measured kernel towards ``lor_end`` (positive
# positions). All profiles sum to the non-TOF projection (here 1 voxel x 1 mm): the TOF
# weights of every image plane sum to 1, also for the tabulated kernel - as long as the TOF
# bins cover the whole support of the kernel (here 161 bins of 5 mm; with fewer bins, the
# part of the kernel outside the TOF bins is lost, as in a real scanner).

print(f"sum over TOF bins: Gaussian {float(xp.sum(p_gauss)):.4f}, table {profiles[1.0].sum():.4f}, "
      f"table scaled {profiles[1.5].sum():.4f}")

# %%
# Notes
# -----
# - **Per-LOR scale:** ``tof_kernel_scale`` can have one value per LOR (or event), e.g. for
#   detector pairs with the same pulse shape but different timing resolution.
# - **Table resolution:** use table points that are finer than the TOF bins (e.g. at most a
#   tenth of the TOF bin width); details of the kernel finer than the table spacing are
#   smoothed out. Here: 5 ps = 0.75 mm vs. 5 mm TOF bins.
# - **Width of the support (64-bin limit):** the projectors handle TOF windows (the TOF
#   bins of one image plane) of up to 64 TOF bins. A window has at most
#   min(floor(s * (u_max - u_min) / tof_bin_width) + 2, num_tof_bins) bins, so with at most
#   64 TOF bins the projection is always exact. With more TOF bins (as here: 161), the
#   support s * (u_max - u_min) must be shorter than 63 TOF bin widths; otherwise the
#   window is truncated to 64 bins around the center of the support and renormalized,
#   which cuts off parts of the kernel (visible as a kink in the profile), and the
#   projectors issue a ``UserWarning``. Trim empty tails of a measured kernel from the
#   table. Here: 202 mm * 1.5 = 303 mm = 61 bins of 5 mm.
# - **Mirrored LORs:** if ``lor_start`` and ``lor_end`` of a LOR are swapped, u changes its
#   sign: the same detector pair then needs the mirrored kernel
#   (F'(u) = 1 - F(-u), on the grid from -u_max to -u_min).
# - **Listmode:** :func:`parallelproj_core.joseph3d_tof_lm_fwd_tab` and
#   :func:`parallelproj_core.joseph3d_tof_lm_back_tab` use the same table; a listmode event
#   gets the value of its TOF sinogram bin.
