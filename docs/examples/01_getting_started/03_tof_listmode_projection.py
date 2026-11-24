"""
TOF Listmode Forward and Back Projection
========================================

In listmode, the data are a list of **events**: every detected coincidence has its LOR (two
end points) and the TOF bin in which it was detected (``tof_bin_index``, ``int16``). The TOF
listmode projectors :func:`parallelproj_core.joseph3d_tof_lm_fwd` and
:func:`parallelproj_core.joseph3d_tof_lm_back` compute one value per event: the value of the
event's TOF bin. Only the image planes around this TOF bin are visited, which makes listmode
projections cheap for few counts (see :doc:`/theory/sino_vs_lm`).

The TOF parameters are the same as for the TOF sinogram projectors. This example shows that
**the listmode projection of an event is identical to the value of its TOF bin in the TOF
sinogram projection** of its LOR.
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
from utils import COLORS, array_backend, disk_phantom, lor_2d, set_style, show_image, to_numpy

xp, dev = array_backend()
set_style()

image_np, image_origin_np, voxel_size_np, extent = disk_phantom()
image = xp.asarray(image_np, device=dev)
image_origin = xp.asarray(image_origin_np, device=dev)
voxel_size = xp.asarray(voxel_size_np, device=dev)

tof_bin_width = 8.0
tof_sigma = xp.asarray([12.0], dtype=xp.float32, device=dev)
tof_center_offset = xp.asarray([0.0], dtype=xp.float32, device=dev)
num_tof_bins = 45

# %%
# LORs and events
# ---------------
# 300 random LORs, and 3 events with random TOF bins on every LOR (900 events). For the
# listmode projectors, every event needs its own end points, so the LORs are repeated.

rng = np.random.default_rng(1)
lors = [lor_2d(phi, r) for phi, r in zip(rng.uniform(0, 180, 300), rng.uniform(-120, 120, 300))]
lor_start_np = np.array([s for s, _ in lors])
lor_end_np = np.array([e for _, e in lors])
event_lor = np.repeat(np.arange(300), 3)  # the LOR of every event
event_tof_bin = rng.integers(10, 35, event_lor.size).astype(np.int16)

event_start = xp.asarray(lor_start_np[event_lor], device=dev)
event_end = xp.asarray(lor_end_np[event_lor], device=dev)
tof_bin_index = xp.asarray(event_tof_bin, device=dev)

# %%
# Listmode forward projection: one value per event
# ------------------------------------------------

lm_proj = xp.zeros(event_lor.size, dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_tof_lm_fwd(event_start, event_end, image, image_origin, voxel_size, lm_proj,
                                      tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)

# the TOF sinogram projection of the 300 LORs, for comparison
sino = xp.zeros((300, num_tof_bins), dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_tof_sino_fwd(xp.asarray(lor_start_np, device=dev), xp.asarray(lor_end_np, device=dev),
                                        image, image_origin, voxel_size, sino, tof_bin_width, tof_sigma,
                                        tof_center_offset, num_tof_bins)
sino_of_events = to_numpy(sino)[event_lor, event_tof_bin]
print(f"max. difference listmode vs. TOF sinogram bin: {np.abs(to_numpy(lm_proj) - sino_of_events).max():.1e}")

# %%
# Listmode back projection
# ------------------------
# Back projecting the value 1 for a few events shows that every event only "sees" the image
# around its TOF bin. (The back projection adds to the image, so it starts from zeros.)

sel = np.array([5, 40, 77, 120, 200])  # 5 LORs, with one TOF bin each
sel_bins = np.array([22, 18, 26, 20, 24], np.int16)
back = xp.zeros(image.shape, dtype=xp.float32, device=dev)
parallelproj_core.joseph3d_tof_lm_back(xp.asarray(lor_start_np[sel], device=dev),
                                       xp.asarray(lor_end_np[sel], device=dev), back, image_origin, voxel_size,
                                       xp.ones(sel.size, dtype=xp.float32, device=dev), tof_bin_width, tof_sigma,
                                       tof_center_offset, xp.asarray(sel_bins, device=dev), num_tof_bins)

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
show_image(a, back, extent, cmap="Purples", title="(a) back projection of 5 listmode events")
for k in sel:
    a.plot([lor_start_np[k, 0], lor_end_np[k, 0]], [lor_start_np[k, 1], lor_end_np[k, 1]], color=COLORS["gray"],
           lw=0.6, alpha=0.6)
a.set_xlim(-190, 190)
a.set_ylim(-190, 190)
b.plot(sino_of_events, to_numpy(lm_proj), "o", ms=3, color=COLORS["purple"], alpha=0.6)
m = float(sino_of_events.max())
b.plot([0, m], [0, m], color=COLORS["gray"], lw=1, ls=(0, (3, 2)))
b.set_xlabel("TOF sinogram value of the event's (LOR, TOF bin)")
b.set_ylabel("TOF listmode forward projection")
b.set_title("(b) a listmode event = its TOF sinogram bin")
fig.show()

# %%
# Listmode reconstructions therefore use exactly the same model as TOF sinogram
# reconstructions of the same data. For large numbers of events, **sort the events** once
# (e.g. per subset) to make the listmode projections much faster on GPUs, see
# :ref:`sphx_glr_auto_examples_03_pytorch_gpus_performance_04_tof_listmode_event_order.py`.
