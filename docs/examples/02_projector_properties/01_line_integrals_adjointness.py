"""
Line Integrals and Adjointness
==============================

Two properties that iterative reconstruction algorithms (MLEM, OSEM, gradient methods)
rely on, checked numerically:

1. The **forward projection is a line integral** of the image along the LOR. For a uniform
   disk with value 1, the projection of a LOR is the length of the LOR inside the disk,
   which is known analytically: :math:`2\\sqrt{R^2 - r^2}` for a LOR at distance
   :math:`r` from the center of a disk with radius :math:`R`.
2. The **back projection is the exact adjoint** (transpose) of the forward projection: for
   any image :math:`x` and any data :math:`y`,

   .. math::

      \\langle A x, y \\rangle = \\langle x, A^T y \\rangle .

   In a reconstruction, the back projection is used to compute gradients (:math:`A^T` of
   the data mismatch) and the sensitivity image (:math:`A^T 1`). If it were not the exact
   adjoint, algorithms like MLEM would not converge to the solution of the model.

All projector pairs of ``parallelproj_core`` are exact adjoints of each other (up to
float32 rounding): non-TOF, TOF sinogram, TOF listmode and the TOF listmode "point"
projectors. See :doc:`/theory/joseph` for how the line integrals are computed.
"""

# sphinx_gallery_start_ignore
import pathlib, sys
try:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
except NameError:  # run by sphinx-gallery: docs/conf.py adds docs/examples to the path
    pass
# sphinx_gallery_end_ignore
import numpy as np
import matplotlib.pyplot as plt
import parallelproj_core
from utils import COLORS, array_backend, lor_2d, set_style, to_numpy

xp, dev = array_backend()
set_style()

# %%
# A uniform disk and LORs at different distances from the center
# --------------------------------------------------------------
# A disk with radius R = 100 mm and value 1 in an image of 64 x 64 x 1 voxels of 4 mm. 120
# LORs parallel to the x-axis (0°) and 120 LORs at 33°, at distances from -118 mm to 118 mm
# from the center.

n, vs, R = 64, 4.0, 100.0
origin_np = np.array([-(n - 1) / 2 * vs, -(n - 1) / 2 * vs, 0.0], np.float32)
c = origin_np[0] + np.arange(n) * vs  # voxel centers along x and y
X, Y = np.meshgrid(c, c, indexing="ij")
disk = ((X**2 + Y**2) < R**2).astype(np.float32)[:, :, None]

image = xp.asarray(disk, device=dev)
image_origin = xp.asarray(origin_np, device=dev)
voxel_size = xp.asarray([vs, vs, vs], dtype=xp.float32, device=dev)

r = np.linspace(-118, 118, 120)
lors = [lor_2d(0, ri) for ri in r] + [lor_2d(33, ri) for ri in r]
lor_start = xp.asarray(np.array([s for s, _ in lors]), device=dev)
lor_end = xp.asarray(np.array([e for _, e in lors]), device=dev)
num_lors = len(lors)

proj = xp.zeros(num_lors, dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_fwd(lor_start, lor_end, image, image_origin, voxel_size, proj)
proj_np = to_numpy(proj)

chord = 2 * np.sqrt(np.clip(R**2 - r**2, 0, None))  # analytic length of the LORs inside the disk
dev_mm = np.abs(proj_np - np.concatenate([chord, chord]))
print(f"|projection - analytic chord length|: median {np.median(dev_mm):.1f} mm, max. {dev_mm.max():.1f} mm")

# %%
# The projections follow the analytic chord lengths. The deviations (typically about a mm,
# up to ~20 mm for LORs that just graze the edge) come from the image, not from the
# projector: the disk is represented by 4 mm voxels, so its edge is a staircase, not a
# circle, and Joseph's method interpolates linearly between the voxel centers.
#
# Adjointness: the dot-product test
# ---------------------------------
# For random images x and random data y, compare :math:`\langle Ax, y \rangle` with
# :math:`\langle x, A^T y \rangle` for all four projector pairs (20 random pairs (x, y)
# each; same LORs, TOF bins of 8 mm, σ = 12 mm, 3 events per LOR for listmode). The inner
# products are summed in float64; the projections themselves are computed in float32.

tof_bin_width, num_tof_bins = 8.0, 45
tof_sigma = xp.asarray([12.0], dtype=xp.float32, device=dev)
tof_offset = xp.asarray([0.0], dtype=xp.float32, device=dev)
rng = np.random.default_rng(3)
# listmode events: 3 events per LOR with random TOF bins
ev = np.repeat(np.arange(num_lors), 3)
ev_start, ev_end = xp.take(lor_start, xp.asarray(ev, device=dev), axis=0), xp.take(lor_end, xp.asarray(ev, device=dev), axis=0)
ev_tof_bin = xp.asarray(rng.integers(10, 35, ev.size).astype(np.int16), device=dev)

# (forward projection, back projection, shape of the data) of each projector pair
projectors = {
    "non-TOF": (
        lambda x, y: parallelproj_core.joseph3d_fwd(lor_start, lor_end, x, image_origin, voxel_size, y),
        lambda y, x: parallelproj_core.joseph3d_back(lor_start, lor_end, x, image_origin, voxel_size, y),
        (num_lors,)),
    "TOF sinogram": (
        lambda x, y: parallelproj_core.joseph3d_tof_sino_fwd(lor_start, lor_end, x, image_origin, voxel_size, y,
                                                             tof_bin_width, tof_sigma, tof_offset, num_tof_bins),
        lambda y, x: parallelproj_core.joseph3d_tof_sino_back(lor_start, lor_end, x, image_origin, voxel_size, y,
                                                              tof_bin_width, tof_sigma, tof_offset, num_tof_bins),
        (num_lors, num_tof_bins)),
    "TOF listmode": (
        lambda x, y: parallelproj_core.joseph3d_tof_lm_fwd(ev_start, ev_end, x, image_origin, voxel_size, y,
                                                           tof_bin_width, tof_sigma, tof_offset, ev_tof_bin,
                                                           num_tof_bins),
        lambda y, x: parallelproj_core.joseph3d_tof_lm_back(ev_start, ev_end, x, image_origin, voxel_size, y,
                                                            tof_bin_width, tof_sigma, tof_offset, ev_tof_bin,
                                                            num_tof_bins),
        (ev.size,)),
    "TOF listmode point": (
        lambda x, y: parallelproj_core.tof_lm_point_fwd(ev_start, ev_end, x, image_origin, voxel_size, y,
                                                        tof_bin_width, tof_offset, ev_tof_bin, num_tof_bins),
        lambda y, x: parallelproj_core.tof_lm_point_back(ev_start, ev_end, x, image_origin, voxel_size, y,
                                                         tof_bin_width, tof_offset, ev_tof_bin, num_tof_bins),
        (ev.size,)),
}

rel_diff = {}
for name, (fwd, back, data_shape) in projectors.items():
    d = []
    for _ in range(20):
        x = xp.asarray(rng.uniform(size=image.shape).astype(np.float32), device=dev)
        y = xp.asarray(rng.uniform(size=data_shape).astype(np.float32), device=dev)
        Ax = xp.zeros(data_shape, dtype=xp.float32, device=dev)
        fwd(x, Ax)
        ATy = xp.zeros(image.shape, dtype=xp.float32, device=dev)  # the back projection adds to it
        back(y, ATy)
        ip_data = float(np.sum(to_numpy(Ax).astype(np.float64) * to_numpy(y)))
        ip_image = float(np.sum(to_numpy(x).astype(np.float64) * to_numpy(ATy)))
        d.append((ip_image - ip_data) / ip_data)
    rel_diff[name] = np.array(d)
    print(f"{name:20s}: max. |relative difference| {np.abs(rel_diff[name]).max():.1e}")

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
a.plot(r, chord, color=COLORS["gray"], lw=3, alpha=0.45, label="analytic chord length 2√(R² − r²)")
a.plot(r, proj_np[:120], ".", ms=4, color=COLORS["blue"], label="joseph3d_fwd, LORs at 0°")
a.plot(r, proj_np[120:], ".", ms=4, color=COLORS["orange"], label="joseph3d_fwd, LORs at 33°")
a.set_xlabel("distance r of the LOR from the center [mm]")
a.set_ylabel("projection of a uniform disk (value 1) [mm]")
a.legend(loc="lower center")
a.set_title("(a) the projection is a line integral (here: the length in the disk)")
cols = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]
for k, (name, d) in enumerate(rel_diff.items()):
    b.plot(k + np.linspace(-0.25, 0.25, d.size), d * 1e9, "o", ms=4, color=cols[k])
b.axhline(0, color=COLORS["gray"], lw=0.8)
for lim in (-np.finfo(np.float32).eps * 1e9, np.finfo(np.float32).eps * 1e9):
    b.axhline(lim, color=COLORS["red"], lw=0.8, ls=(0, (3, 2)))
b.text(3.45, np.finfo(np.float32).eps * 1e9, "± float32 eps", color=COLORS["red"], fontsize=8.5, va="bottom",
       ha="right")
b.set_xticks(range(len(rel_diff)), list(rel_diff), fontsize=9)
b.set_ylim(-150, 150)
b.set_ylabel("(⟨x, Aᵀy⟩ − ⟨Ax, y⟩) / ⟨Ax, y⟩  [×10⁻⁹]")
b.set_title("(b) dot-product test: 20 random (x, y) per projector pair")
fig.show()

# %%
# The relative differences are far below the float32 machine epsilon
# (:math:`1.2 \cdot 10^{-7}`, red lines): the back projectors are the exact adjoints of the
# forward projectors, for all LOR directions and TOF settings. Small differences of this
# size come from the different order of the float32 summations in the forward and the
# back projection.
#
# The same test is useful for your own operators built from the projectors (e.g. with
# attenuation, normalization or resolution modeling): an operator and its adjoint pass it
# only if all their parts are consistent.
