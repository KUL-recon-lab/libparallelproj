"""
TOF Projections with a Tabulated TOF Kernel
===========================================

Instead of a Gaussian, the TOF projectors with the suffix ``_tab``
(:func:`parallelproj_core.joseph3d_tof_sino_fwd_tab`,
:func:`parallelproj_core.joseph3d_tof_sino_back_tab`, and the listmode versions) use any TOF
kernel given as a **table of its cumulative distribution F(u)**:

- ``tof_kernel_cdf``: F at n equidistant points from ``u_min`` to ``u_max`` (linearly
  interpolated in between); u is the distance along the LOR from the true annihilation
  point (spatial units, positive towards ``lor_end``). F does not need to be normalized.
- ``tof_kernel_scale``: stretches the kernel (one value, or one per LOR).
- the TOF bin width is not part of the table: the projectors integrate the kernel over
  each TOF bin themselves.

This example projects one LOR with

1. the Gaussian kernel of :func:`parallelproj_core.joseph3d_tof_sino_fwd`,
2. the **same Gaussian as a table** (gives the same projection),
3. a **non-Gaussian kernel** with the same FWHM but wider tails.

How to build the table from measured time differences is shown in
:ref:`sphx_glr_auto_examples_02_projector_properties_03_tof_kernel_table.py`.
"""

# sphinx_gallery_start_ignore
import pathlib, sys
try:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
except NameError:  # run by sphinx-gallery: docs/conf.py adds docs/examples to the path
    pass
# sphinx_gallery_end_ignore
import math

import numpy as np
import matplotlib.pyplot as plt
import parallelproj_core
from utils import COLORS, array_backend, disk_phantom, set_style, to_numpy

xp, dev = array_backend()
set_style()

# %%
# Image, one LOR and TOF parameters
# ---------------------------------
# The same as in
# :ref:`sphx_glr_auto_examples_01_getting_started_02_tof_sinogram_projection.py`: the disk
# phantom, one LOR close to both hot spots, 45 TOF bins of 8 mm, σ = 12 mm.

image_np, image_origin_np, voxel_size_np, extent = disk_phantom()
image = xp.asarray(image_np, device=dev)
image_origin = xp.asarray(image_origin_np, device=dev)
voxel_size = xp.asarray(voxel_size_np, device=dev)

lor_start = xp.asarray([[-170.0, -55.0, 0.0]], dtype=xp.float32, device=dev)
lor_end = xp.asarray([[170.0, 64.0, 0.0]], dtype=xp.float32, device=dev)

tof_bin_width, num_tof_bins = 8.0, 45
sigma = 12.0  # mm (FWHM 28.3 mm)
tof_center_offset = xp.asarray([0.0], dtype=xp.float32, device=dev)
tof_kernel_scale = xp.asarray([1.0], dtype=xp.float32, device=dev)

# %%
# 1. Gaussian kernel (no table)
# -----------------------------

p_gauss = xp.zeros((1, num_tof_bins), dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_tof_sino_fwd(lor_start, lor_end, image, image_origin, voxel_size, p_gauss, tof_bin_width,
                                        xp.asarray([sigma], dtype=xp.float32, device=dev), tof_center_offset,
                                        num_tof_bins, num_sigmas=3.0)

# %%
# 2. The same Gaussian as a table
# -------------------------------
# F(u) = Φ(u / σ), the cumulative standard normal distribution, on u = -3σ ... 3σ (the
# support of the Gaussian above, which is truncated at 3σ), 601 points (0.12 mm apart).


def tabulated_projection(cdf, u_min, u_max):
    """TOF sinogram forward projection of the LOR with a tabulated kernel"""
    p = xp.zeros((1, num_tof_bins), dtype=xp.float32, device=dev)
    parallelproj_core.joseph3d_tof_sino_fwd_tab(lor_start, lor_end, image, image_origin, voxel_size, p,
                                                tof_bin_width, tof_center_offset, num_tof_bins,
                                                xp.asarray(cdf, dtype=xp.float32, device=dev), u_min, u_max,
                                                tof_kernel_scale)
    return p


u_gauss = np.linspace(-3 * sigma, 3 * sigma, 601)
cdf_gauss = np.array([0.5 * (1 + math.erf(u / (sigma * math.sqrt(2)))) for u in u_gauss])
p_gauss_tab = tabulated_projection(cdf_gauss, float(u_gauss[0]), float(u_gauss[-1]))

# %%
# 3. A non-Gaussian kernel: same FWHM, wider tails
# ------------------------------------------------
# A Gaussian core plus a 3 times wider Gaussian "tail" (25 % of the events), e.g. events with a
# poorer timing resolution. The width s of the core is chosen such that the kernel has the
# same FWHM as the Gaussian above. The table covers ±9 s (the tail component up to 3 of its
# standard deviations).


def mixture_pdf(u, s):
    g = lambda x, sd: np.exp(-0.5 * (x / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    return 0.75 * g(u, s) + 0.25 * g(u, 3 * s)


def full_width(s, fraction=0.5):
    """full width of the mixture (core width s) at the given fraction of its maximum
    (0.5: FWHM, 0.1: FWTM), computed numerically"""
    u = np.linspace(0, 20 * s, 200001)
    f = mixture_pdf(u, s)
    return 2 * u[np.argmax(f < fraction * f[0])]


def fwhm(s):
    return full_width(s, 0.5)


fwhm_gauss = 2 * math.sqrt(2 * math.log(2)) * sigma
s_lo, s_hi = 1.0, sigma  # bisection: the FWHM grows with s
for _ in range(60):
    s_mid = 0.5 * (s_lo + s_hi)
    s_lo, s_hi = (s_mid, s_hi) if fwhm(s_mid) < fwhm_gauss else (s_lo, s_mid)
s_core = 0.5 * (s_lo + s_hi)

u_tail = np.linspace(-9 * s_core, 9 * s_core, 1201)
cdf_tail = np.concatenate([[0.0], np.cumsum(0.5 * (mixture_pdf(u_tail[1:], s_core) + mixture_pdf(u_tail[:-1], s_core))
                                            * np.diff(u_tail))])  # trapezoidal integration of the density
p_tail = tabulated_projection(cdf_tail, float(u_tail[0]), float(u_tail[-1]))

fwtm_gauss = 2 * math.sqrt(2 * math.log(10)) * sigma  # full width at tenth maximum of a Gaussian
fwtm_tail = full_width(s_core, 0.1)
print(f"FWHM: Gaussian {fwhm_gauss:.2f} mm, core + tail {fwhm(s_core):.2f} mm (core s = {s_core:.2f} mm)")
print(f"FWTM: Gaussian {fwtm_gauss:.2f} mm, core + tail {fwtm_tail:.2f} mm")
p_nontof = xp.zeros(1, dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_fwd(lor_start, lor_end, image, image_origin, voxel_size, p_nontof)
print(f"non-TOF projection: {float(p_nontof[0]):.3f}")
for name, p in [("Gaussian", p_gauss), ("Gaussian table", p_gauss_tab), ("core + tail table", p_tail)]:
    print(f"{name:18s}: sum over TOF bins {float(xp.sum(p)):.3f}")
print(f"max. |Gaussian table - Gaussian| / max. Gaussian: "
      f"{float(xp.max(xp.abs(p_gauss_tab - p_gauss)) / xp.max(p_gauss)):.1e}")

# %%
# The "true" TOF profile (perfect TOF)
# ------------------------------------
# With a perfect timing resolution, every TOF bin would contain the line integral of the
# image over its own 8 mm section of the LOR, without blurring. We compute it numerically:
# bilinear interpolation of the image (as in Joseph's method) at points 0.05 mm apart along
# the LOR, summed per TOF bin.

s_np, e_np = to_numpy(lor_start)[0].astype(np.float64), to_numpy(lor_end)[0].astype(np.float64)
lor_length = np.linalg.norm(e_np - s_np)
step = 0.05  # mm
t = np.arange(step / 2, lor_length, step)  # distance from lor_start
pts = s_np + t[:, None] * (e_np - s_np) / lor_length
fx = (pts[:, 0] - image_origin_np[0]) / voxel_size_np[0]  # continuous voxel coordinates
fy = (pts[:, 1] - image_origin_np[1]) / voxel_size_np[1]
img2d = image_np[:, :, 0]
ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
wx, wy = fx - ix, fy - iy
vals = np.zeros_like(t)
for dx, dy, w in [(0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)]:
    jx, jy = ix + dx, iy + dy
    inside = (jx >= 0) & (jx < img2d.shape[0]) & (jy >= 0) & (jy < img2d.shape[1])
    vals[inside] += w[inside] * img2d[jx[inside], jy[inside]]
# TOF bin of every point: bins of tof_bin_width centered on the LOR center, numbered towards lor_end
tof_bin = np.floor((t - 0.5 * lor_length) / tof_bin_width + 0.5 * num_tof_bins).astype(int)
ok = (tof_bin >= 0) & (tof_bin < num_tof_bins)
p_true = np.bincount(tof_bin[ok], weights=vals[ok] * step, minlength=num_tof_bins)
print(f"perfect TOF: sum over TOF bins {p_true.sum():.3f} (non-TOF projection {float(p_nontof[0]):.3f})")

# %%
# The back projection with a table
# --------------------------------
# :func:`parallelproj_core.joseph3d_tof_sino_back_tab` takes the same table and is the
# adjoint of the forward projection: for an image x and TOF data y,
# ⟨A x, y⟩ = ⟨x, Aᵀ y⟩.

y = xp.asarray(np.random.default_rng(0).uniform(size=(1, num_tof_bins)), dtype=xp.float32, device=dev)
back = xp.zeros(image.shape, dtype=xp.float32, device=dev)  # the back projection adds to it
parallelproj_core.joseph3d_tof_sino_back_tab(lor_start, lor_end, back, image_origin, voxel_size, y, tof_bin_width,
                                             tof_center_offset, num_tof_bins,
                                             xp.asarray(cdf_tail, dtype=xp.float32, device=dev),
                                             float(u_tail[0]), float(u_tail[-1]), tof_kernel_scale)
print(f"<A x, y> = {float(xp.sum(p_tail * y)):.4f},  <x, A^T y> = {float(xp.sum(image * back)):.4f}")

# %%
uu = np.linspace(-110, 110, 2201)  # covers the support of both kernels (+-3 sigma, +-9 s)
# The kernels as the projectors use them: densities [1/mm] with integral 1 over their range
# (the projectors normalize every kernel over its support: the Gaussian truncated at 3 sigma,
# the table by F(u_max) - F(u_min)).
gauss_norm = math.erf(3 / math.sqrt(2))  # integral of the Gaussian over -3 sigma ... 3 sigma
k_gauss = np.where(np.abs(uu) <= 3 * sigma,
                   np.exp(-0.5 * (uu / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi) * gauss_norm), np.nan)
tail_norm = cdf_tail[-1] - cdf_tail[0]  # integral of the core + tail density over its table range
k_tail = np.where(np.abs(uu) <= 9 * s_core, mixture_pdf(uu, s_core) / tail_norm, np.nan)
du = uu[1] - uu[0]
print(f"integral of the kernels (as used by the projectors): Gaussian {np.nansum(k_gauss) * du:.4f}, "
      f"core + tail {np.nansum(k_tail) * du:.4f}")
max_gauss = 1 / (sigma * math.sqrt(2 * math.pi) * gauss_norm)
max_tail = mixture_pdf(0.0, s_core) / tail_norm
bins = np.arange(num_tof_bins)
pg, pgt, pt = to_numpy(p_gauss)[0], to_numpy(p_gauss_tab)[0], to_numpy(p_tail)[0]
cg, ct = COLORS["blue"], COLORS["orange"]

fig, ax = plt.subplot_mosaic([["a", "b", "c"], ["d", "d", "e"]], figsize=(17, 9.5), layout="constrained")
for a, log in [(ax["a"], False), (ax["b"], True)]:
    a.plot(uu, k_gauss, color=cg, label=f"Gaussian, σ = {sigma:g} mm (truncated at 3σ)")
    a.plot(uu, k_tail, color=ct, label="core + tail (same FWHM)")
    a.fill_between(uu, k_tail, color=ct, alpha=0.08, lw=0)
    a.fill_between(uu, k_gauss, color=cg, alpha=0.08, lw=0)
    # FWHM (the same for both) and FWTM (different): bars at half / a tenth of each kernel's
    # own maximum, with ticks at their ends (in the color of the kernel)
    for w_h, w_t, m, c in [(fwhm_gauss, fwtm_gauss, max_gauss, cg), (fwhm_gauss, fwtm_tail, max_tail, ct)]:
        for w, frac in [(w_h, 0.5), (w_t, 0.1)]:
            a.hlines(frac * m, -0.5 * w, 0.5 * w, color=c, lw=1.2)
            a.plot([-0.5 * w, 0.5 * w], [frac * m, frac * m], "|", ms=10, mew=2, color=c)
    a.text(45, 0.022 if not log else 0.02,
           f"FWHM (at half maximum):\n{fwhm_gauss:.1f} mm both\n\nFWTM (at a tenth of the maximum):\n"
           f"{fwtm_gauss:.1f} mm Gaussian\n{fwtm_tail:.1f} mm core + tail", va="center", fontsize=8.5,
           color=COLORS["gray"])
    a.set_xlabel("u: distance from the annihilation point [mm]")
    if log:
        a.set_yscale("log")
        a.set_ylim(1e-5, 0.1)
        a.set_ylabel("kernel density [1/mm] (log scale)")
        a.set_title("(b) the same on a log scale: the tails")
    else:
        a.set_ylim(0, 0.039)
        a.set_ylabel("kernel density [1/mm]")
        a.set_title("(a) the TOF kernels: area 1, same FWHM, different FWTM")
        a.legend(loc="upper left")

# (c) the tables F(u) as passed to the projector (normalized: F(u_max) - F(u_min) = 1)
for u_tab, cdf, c, name in [(u_gauss, cdf_gauss, cg, "Gaussian table (601 points)"),
                            (u_tail, cdf_tail, ct, "core + tail table (1201 points)")]:
    F = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    ax["c"].plot(u_tab, F, color=c, label=name)
    ax["c"].plot(u_tab[::50], F[::50], "o", ms=3.5, color=c)  # every 50th table point
ax["c"].axhline(0.5, color=COLORS["gray"], lw=0.6, ls=":")
ax["c"].set_xlim(-110, 110)
ax["c"].set_xlabel("u: distance from the annihilation point [mm]")
ax["c"].set_ylabel("F(u) (tof_kernel_cdf, normalized)")
ax["c"].legend(loc="upper left")
ax["c"].set_title("(c) the tables: cumulative kernels F(u)")

ax["d"].step(bins, p_true, where="mid", color=COLORS["gray"], lw=1.2, alpha=0.8, label="perfect TOF (no blur)")
ax["d"].plot(bins, pg, "o", ms=6, mfc="none", color=cg, label="1. Gaussian")
ax["d"].plot(bins, pgt, ".", ms=5, color=COLORS["purple"], label="2. Gaussian table")
ax["d"].plot(bins, pt, ".-", ms=4, color=ct, label="3. core + tail table")
ax["d"].set_xlabel("TOF bin")
ax["d"].set_ylabel("TOF sinogram value")
ax["d"].legend(loc="upper left")
ax["d"].set_title("(d) TOF projections of the LOR")

ax["e"].plot(bins, pgt - pg, ".-", color=COLORS["purple"], label="2. Gaussian table − 1.")
ax["e"].plot(bins, pt - pg, ".-", color=ct, label="3. core + tail − 1.")
ax["e"].axhline(0, color=COLORS["gray"], lw=0.8)
ax["e"].set_xlabel("TOF bin")
ax["e"].set_ylabel("difference to the Gaussian projection")
ax["e"].legend(loc="lower left")
ax["e"].set_title("(e) differences to 1.")
fig.show()

# %%
# - **Perfect TOF** (gray steps in (d)): the line integral of the image per TOF bin. Every TOF
#   kernel blurs this profile along the LOR; its sum is the same as that of the TOF
#   projections (up to 0.03 %: Joseph's method samples the LOR once per image plane, the
#   reference integrates it in 0.05 mm steps).
# - **1. and 2.** give the same projection (up to float32 rounding and the resolution of the
#   table): a table of a Gaussian reproduces the Gaussian kernel.
# - **3.** has the same FWHM, but a wider FWTM (full width at tenth maximum). Both kernels
#   have the area 1 (the projectors normalize every kernel over its support), so the kernel
#   with tails has a lower peak: its tails move counts from the peaks of the TOF profile
#   into the neighboring TOF bins (and further out). The sum over all TOF bins is
#   the non-TOF projection, as for every kernel, as long as the TOF bins cover the kernel
#   of every image sample: here, the 45 TOF bins (±180 mm around the LOR center) miss the
#   outermost part of the ±101 mm tails of the samples at the edge of the disk, so the sum
#   is 0.01 % smaller.
# - **Table range:** the support ``u_max - u_min`` should only cover the region where the
#   kernel is not (practically) zero: the projectors handle TOF windows of up to 64 TOF bins
#   (here: ±9 s = ±101 mm, i.e. 25 TOF bins of 8 mm).
