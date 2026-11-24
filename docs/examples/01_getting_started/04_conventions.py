"""
Conventions: Image Origin, LOR Direction and TOF Bins
=====================================================

Most errors when calling projectors directly come from the conventions of their arguments.
This example shows them with small, checkable cases (see also :doc:`/theory/projectors`):

1. ``image_origin`` is the position of the **center** of voxel ``[0, 0, 0]``; voxel
   ``[i, j, k]`` is centered at ``image_origin + (i, j, k) * voxel_size``.
2. Non-TOF projections do not depend on the direction of a LOR (start and end can be
   swapped).
3. TOF bins are numbered **from** ``lor_start`` **towards** ``lor_end``; the central bin lies
   at the LOR center. Swapping start and end mirrors the TOF profile.
4. A positive ``tof_center_offset`` shifts the TOF bins towards ``lor_end``.
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
from matplotlib.patches import Rectangle
import parallelproj_core
from utils import COLORS, array_backend, set_style, to_numpy

xp, dev = array_backend()
set_style()

# %%
# 1. Image origin and voxel centers
# ---------------------------------
# A 64 x 64 x 1 image with 4 mm voxels, centered on (0, 0, 0): the center of voxel [0, 0, 0]
# is half an image width (minus half a voxel) away from the center.

n, vs = 64, 4.0
image_origin_np = np.array([-(n - 1) / 2 * vs, -(n - 1) / 2 * vs, 0.0], np.float32)
voxel_size_np = np.array([vs, vs, vs], np.float32)
for idx in [(0, 0, 0), (1, 2, 0), (n - 1, n - 1, 0)]:
    print(f"center of voxel {idx}: {image_origin_np + np.array(idx) * voxel_size_np} mm")

image_origin = xp.asarray(image_origin_np, device=dev)
voxel_size = xp.asarray(voxel_size_np, device=dev)

# %%
# The projectors use the **voxel centers** for the interpolation: a point source in voxel
# [i, j, 0] is at the position printed above, and the image covers
# ``image_origin - voxel_size / 2`` to ``image_origin + (shape - 1/2) * voxel_size``.
#
# 2.-4. LOR direction, TOF bins and TOF center offset
# ---------------------------------------------------
# A point source in voxel [42, 32, 0], i.e. at x = 42 mm, y = 2 mm, and a LOR parallel to the
# x-axis through it. We compare the TOF profiles for the LOR from -x to +x, the same LOR with start
# and end swapped, and a TOF center offset of +20 mm.

image = xp.zeros((n, n, 1), dtype=xp.float32, device=dev)
i_src = n // 2 + 10
image[i_src, n // 2, 0] = 1.0
print(f"point source at {image_origin_np + np.array([i_src, n // 2, 0]) * voxel_size_np} mm")

tof_bin_width, num_tof_bins = 8.0, 45
tof_sigma = xp.asarray([12.0], dtype=xp.float32, device=dev)
a = xp.asarray([[-170.0, 2.0, 0.0]], dtype=xp.float32, device=dev)  # -x side, through the voxel centers at y = 2 mm
b = xp.asarray([[170.0, 2.0, 0.0]], dtype=xp.float32, device=dev)  # +x side

profiles, nontof = {}, {}
for name, start, end, offset in [("start → end (-x → +x)", a, b, 0.0), ("swapped (+x → -x)", b, a, 0.0),
                                 ("tof_center_offset = +20 mm", a, b, 20.0)]:
    p = xp.zeros((1, num_tof_bins), dtype=xp.float32, device=dev)
    parallelproj_core.joseph3d_tof_sino_fwd(start, end, image, image_origin, voxel_size, p, tof_bin_width,
                                            tof_sigma, xp.asarray([offset], dtype=xp.float32, device=dev),
                                            num_tof_bins)
    profiles[name] = to_numpy(p)[0]
    q = xp.zeros(1, dtype=xp.float32, device=dev)
    parallelproj_core.joseph3d_fwd(start, end, image, image_origin, voxel_size, q)
    nontof[name] = float(q[0])
    print(f"{name:28s}: TOF bin with the maximum {int(np.argmax(profiles[name])):2d}, non-TOF projection "
          f"{nontof[name]:.2f}")

# %%
# - The non-TOF projection is the same in all cases (it does not depend on the direction).
# - From -x to +x, the source (42 mm towards ``lor_end`` from the LOR center) is in TOF bin
#   22 + 42 / 8 ≈ 27; with start and end swapped, it is 42 mm towards ``lor_start``, i.e. in
#   bin 22 - 5 ≈ 17: the profile is mirrored around the central bin 22.
# - With an offset of +20 mm, all TOF bin centers move 20 mm towards ``lor_end``, so the
#   source now falls into a bin with a smaller number (27 - 20 / 8 ≈ 24.5).

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1.3]))
# (a) sketch of the corner of the image: voxels, centers, origin
v, o = 10.0, np.array([-15.0, -15.0])
for i in range(4):
    for j in range(4):
        ax0.add_patch(Rectangle((o[0] + (i - 0.5) * v, o[1] + (j - 0.5) * v), v, v, fc="white", ec=COLORS["mid"]))
        ax0.plot(o[0] + i * v, o[1] + j * v, "o", color=COLORS["mid"], ms=3)
ax0.plot(*o, "o", color=COLORS["red"], ms=8)
ax0.annotate("image_origin = center of voxel [0, 0, 0]", o, (o[0] - 2, o[1] - 17), color=COLORS["red"], fontsize=9,
             arrowprops=dict(arrowstyle="->", color=COLORS["red"]))
ax0.annotate("", (o[0] + 0.5 * v, o[1] + 3.5 * v + 2), (o[0] - 0.5 * v, o[1] + 3.5 * v + 2),
             arrowprops=dict(arrowstyle="<->", color=COLORS["gray"]))
ax0.text(o[0], o[1] + 3.5 * v + 4, "voxel_size[0]", fontsize=8.5, color=COLORS["gray"], ha="center", va="bottom")
ax0.text(o[0] + v, o[1] + 2 * v + 2.5, "[1,2]", ha="center", va="bottom", fontsize=8)
ax0.text(o[0] + 3 * v, o[1] + 2.5, "[3,0]", ha="center", va="bottom", fontsize=8)
ax0.annotate("", (-10, 36), (-28, 36), arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"]))
ax0.annotate("", (-28, 54), (-28, 36), arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"]))
ax0.text(-8, 36, "axis 0", fontsize=8.5, va="center", color=COLORS["gray"])
ax0.text(-28, 56, "axis 1", fontsize=8.5, ha="center", color=COLORS["gray"])
ax0.set_xlim(-40, 40)
ax0.set_ylim(-38, 60)
ax0.set_aspect("equal")
ax0.axis("off")
ax0.set_title("(a) voxel [i, j, k] is centered at\n     image_origin + (i, j, k) · voxel_size")
for (name, prof), col in zip(profiles.items(), [COLORS["blue"], COLORS["orange"], COLORS["purple"]]):
    ax1.plot(np.arange(num_tof_bins), prof, ".-", color=col, label=name)
ax1.axvline((num_tof_bins - 1) / 2, color=COLORS["gray"], lw=0.8, ls=(0, (3, 2)))
ax1.annotate("central bin 22\n= LOR center", ((num_tof_bins - 1) / 2, 0.45), (34, 0.55), fontsize=8.5,
             color=COLORS["gray"], va="center", arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=0.8))
ax1.set_xlabel("TOF bin")
ax1.set_ylabel("TOF projection of the point source")
ax1.legend(loc="upper left")
ax1.set_title("(b) TOF bins are numbered from lor_start towards lor_end;\n"
              "     a positive tof_center_offset shifts them towards lor_end")
fig.show()

# %%
# **Units:** all lengths (LOR end points, image origin, voxel size, TOF bin width, σ, offset)
# must use the same unit (usually mm). TOF parameters given in time units have to be
# converted: a time difference Δt corresponds to c Δt / 2 ≈ 0.15 mm/ps along the LOR.
