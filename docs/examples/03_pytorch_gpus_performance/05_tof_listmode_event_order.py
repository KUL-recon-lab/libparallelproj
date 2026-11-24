"""
Faster TOF Listmode Projections by Sorting the Events
=====================================================

The TOF listmode projectors (:func:`parallelproj_core.joseph3d_tof_lm_fwd`,
:func:`parallelproj_core.joseph3d_tof_lm_back`) process consecutive events at the same time
(on GPUs: 32 events per "warp"). In time-ordered listmode data, these events lie anywhere
in the image, and almost every voxel access needs its own memory transaction. If
consecutive events have nearby **TOF centers** (the center of the event's TOF bin on its
LOR), they read and write the same image region and the voxels come from the caches.

:func:`parallelproj_core.tof_lm_sort_key` computes a 64-bit sort key per event: the
principal axis of its LOR and the 3D Morton (Z-order) code of the grid cell (2 x 2 x 2
voxels) that contains its TOF center. Sorting the events by this key - once, e.g. once per
subset, before the iterations of a reconstruction - made the TOF listmode projections ~9x
faster on a desktop GPU (see :ref:`listmode-event-order`).

This example shows

- how to compute the keys and sort the events,
- that sorting does not change the projections (only their order),
- the TOF centers of consecutive events before and after sorting,
- the projection times before and after sorting (on a GPU with 10 million events, on the
  CPU with 300 000; the large gains are on GPUs),
- how to sort the events of the subsets of a listmode OSEM reconstruction.
"""

# %%
import importlib.util
import time

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
# A simple scanner, image and listmode data
# -----------------------------------------
# A cylindrical scanner (radius 300 mm), a cylinder phantom (radius 150 mm) and events with
# random LORs through the phantom. The TOF bin of an event is that of its emission point
# plus Gaussian TOF noise. Like time-ordered listmode data, the events are in random order.
#
# The problem size depends on the device: on a GPU, 10 million events and an image of
# 256 x 256 x 80 voxels of 2 mm (20 MB). With fewer events, the projections take less than
# a millisecond on a GPU and the timings show no difference; with a small image, it fits
# into the L2 cache of the GPU and the memory access pattern hardly matters. On the CPU
# (e.g. when the documentation is built), 300 000 events and 128 x 128 x 40 voxels of 4 mm.
# (10 million events need ~0.8 GB of GPU memory in this example.)

on_gpu = xp.__name__.endswith("cupy")
if on_gpu:
    num_events, img_shape, vs = 10_000_000, (256, 256, 80), np.array([2.0, 2.0, 2.0], np.float32)
else:
    num_events, img_shape, vs = 300_000, (128, 128, 40), np.array([4.0, 4.0, 4.0], np.float32)
origin = (-0.5 * (np.array(img_shape) - 1) * vs).astype(np.float32)
radius, phantom_radius, axial_half = 300.0, 150.0, 75.0
tof_bin_width, sigma_tof, num_tof_bins = 20.0, 25.0, 31


def random_events(n, rng):
    """n events: LOR end points on the scanner (float32) and TOF bins (int16)"""
    # emission points uniformly in the phantom, random directions
    r = phantom_radius * np.sqrt(rng.uniform(size=n))
    phi = rng.uniform(0, 2 * np.pi, n)
    p = np.stack([r * np.cos(phi), r * np.sin(phi), rng.uniform(-axial_half, axial_half, n)], axis=1)
    theta = rng.uniform(0, np.pi, n)
    d = np.stack([np.cos(theta), np.sin(theta), rng.uniform(-0.2, 0.2, n)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    # intersections of the lines p + t d with the scanner cylinder (transaxial radius 300 mm)
    a2 = d[:, 0] ** 2 + d[:, 1] ** 2
    b = p[:, 0] * d[:, 0] + p[:, 1] * d[:, 1]
    c = p[:, 0] ** 2 + p[:, 1] ** 2 - radius**2
    t = np.sqrt(b**2 - a2 * c) / a2
    s = (p + (-b / a2 - t)[:, None] * d).astype(np.float32)
    e = (p + (-b / a2 + t)[:, None] * d).astype(np.float32)
    # TOF bin: distance of the emission point from the LOR center (towards e) + TOF noise
    u = np.sum((p - 0.5 * (s + e)) * d, axis=1) + rng.normal(0, sigma_tof, n)
    tb = np.clip(np.round(u / tof_bin_width + 0.5 * (num_tof_bins - 1)), 0, num_tof_bins - 1).astype(np.int16)
    return s, e, tb


# generated in chunks of (at most) 1 million events to limit the memory on the host
rng = np.random.default_rng(1)
ev_start = np.empty((num_events, 3), np.float32)
ev_end = np.empty((num_events, 3), np.float32)
tof_bin = np.empty(num_events, np.int16)
for i in range(0, num_events, 1_000_000):
    j = min(i + 1_000_000, num_events)
    ev_start[i:j], ev_end[i:j], tof_bin[i:j] = random_events(j - i, rng)
print(f"{num_events:,} events, image {img_shape} voxels of {vs[0]} mm")

image = np.zeros(img_shape, np.float32)
X, Y = np.meshgrid(*[origin[k] + np.arange(img_shape[k]) * vs[k] for k in range(2)], indexing="ij")
image[(X**2 + Y**2) < phantom_radius**2, :] = 1.0

# arrays on the device
ev_start_d, ev_end_d = xp.asarray(ev_start, device=dev), xp.asarray(ev_end, device=dev)
tof_bin_d = xp.asarray(tof_bin, device=dev)
image_d, origin_d, vs_d = xp.asarray(image, device=dev), xp.asarray(origin, device=dev), xp.asarray(vs, device=dev)
sigma_d = xp.asarray([sigma_tof], dtype=xp.float32, device=dev)
offset_d = xp.asarray([0.0], dtype=xp.float32, device=dev)

# %%
# Sort keys and sorting
# ---------------------
# One key per event (int64). Then sort the events - and all other per-event arrays - with
# the argsort of the array library.

key = xp.zeros(num_events, dtype=xp.int64, device=dev)
parallelproj_core.tof_lm_sort_key(ev_start_d, ev_end_d, origin_d, vs_d, key, tof_bin_width, offset_d, tof_bin_d,
                                  num_tof_bins)
order = xp.argsort(key)

ev_start_s = xp.take(ev_start_d, order, axis=0)
ev_end_s = xp.take(ev_end_d, order, axis=0)
tof_bin_s = xp.take(tof_bin_d, order, axis=0)

# %%
# The projections do not depend on the order
# ------------------------------------------
# The forward projection of the sorted events contains the same values (in the new order),
# the back projection gives the same image up to the rounding of the summation order.


def fwd(s, e, tb):
    out = xp.zeros(s.shape[0], dtype=xp.float32, device=dev)
    parallelproj_core.joseph3d_tof_lm_fwd(s, e, image_d, origin_d, vs_d, out, tof_bin_width, sigma_d, offset_d, tb,
                                          num_tof_bins)
    return out


def back(s, e, tb, values):
    x = xp.zeros(img_shape, dtype=xp.float32, device=dev)
    parallelproj_core.joseph3d_tof_lm_back(s, e, x, origin_d, vs_d, values, tof_bin_width, sigma_d, offset_d, tb,
                                           num_tof_bins)
    return x


ones = xp.ones(num_events, dtype=xp.float32, device=dev)
p_unsorted, p_sorted = fwd(ev_start_d, ev_end_d, tof_bin_d), fwd(ev_start_s, ev_end_s, tof_bin_s)
b_unsorted, b_sorted = back(ev_start_d, ev_end_d, tof_bin_d, ones), back(ev_start_s, ev_end_s, tof_bin_s, ones)
order_np = to_numpy(order)
print("forward projections equal (after reordering):", np.array_equal(to_numpy(p_sorted), to_numpy(p_unsorted)[order_np]))
print("max. rel. difference of the back projections:",
      float(np.abs(to_numpy(b_sorted) - to_numpy(b_unsorted)).max() / np.abs(to_numpy(b_unsorted)).max()))

# %%
# TOF centers of consecutive events
# ---------------------------------
# The TOF centers of the first 3 x 32 events (3 GPU warps) before and after sorting
# (transaxial view). Before sorting, the events of a warp are spread over the whole image;
# after sorting, they are close together.


def tof_centers(s, e, tb):
    s, e = to_numpy(s).astype(np.float64), to_numpy(e).astype(np.float64)
    dd = e - s
    x = (to_numpy(tb) - 0.5 * (num_tof_bins - 1)) * tof_bin_width
    return 0.5 * (s + e) + (x / np.linalg.norm(dd, axis=1))[:, None] * dd


start = num_events // 3  # some events in the middle of the sorted list
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained", sharex=True, sharey=True)
for k, (title, (s, e, tb)) in enumerate(
    [("random (time) order", (ev_start_d, ev_end_d, tof_bin_d)), ("sorted by tof_lm_sort_key", (ev_start_s, ev_end_s, tof_bin_s))]
):
    c = tof_centers(s[start:start + 96, :], e[start:start + 96, :], tb[start:start + 96])
    for w in range(3):
        ax[k].plot(c[32 * w:32 * (w + 1), 0], c[32 * w:32 * (w + 1), 1], "o", ms=4, label=f"warp {w + 1}")
    ax[k].add_patch(plt.Circle((0, 0), phantom_radius, fill=False, color="0.5"))
    ax[k].set_title(title)
    ax[k].set_aspect("equal")
    ax[k].set_xlabel("x [mm]")
ax[0].set_ylabel("y [mm]")
ax[0].legend(loc="upper right", fontsize="small")
fig.show()

# %%
# Projection times before and after sorting
# -----------------------------------------
# The large speed-ups (~9x) were measured on GPUs. On CPUs, the effect is smaller (back
# projection ~1.5x faster, forward projection about the same). The times shown below are
# those of the device used to build this documentation.


def best_time(fn, repeats=3):
    times = []
    for _ in range(repeats):
        if on_gpu:
            xp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        fn()
        if on_gpu:
            xp.cuda.Device(0).synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


t_key = best_time(lambda: (parallelproj_core.tof_lm_sort_key(ev_start_d, ev_end_d, origin_d, vs_d, key, tof_bin_width,
                                                             offset_d, tof_bin_d, num_tof_bins), xp.argsort(key)))
t = {name: (best_time(lambda: fwd(*args)), best_time(lambda: back(*args, ones)))
     for name, args in [("random", (ev_start_d, ev_end_d, tof_bin_d)), ("sorted", (ev_start_s, ev_end_s, tof_bin_s))]}
# times in ms (a few ms or less on a GPU for this problem size)
print(f"sort key + argsort: {1e3 * t_key:8.2f} ms")
for name, (tf, tb_) in t.items():
    print(f"{name:7s} events: fwd {1e3 * tf:8.2f} ms, back {1e3 * tb_:8.2f} ms")
print(f"speed-up by sorting: fwd {t['random'][0] / t['sorted'][0]:.1f}x, back {t['random'][1] / t['sorted'][1]:.1f}x")

# %%
# Listmode OSEM: sort the events of every subset once
# ---------------------------------------------------
# In listmode OSEM, the events are split into subsets (e.g. every n-th event). Sort the
# events of every subset once, before the iterations, and use the sorted subsets in all
# iterations. All per-event arrays (LOR end points, TOF bins, per-event TOF parameters and
# corrections such as attenuation, normalization or contamination) must be reordered
# consistently. The order of the events within a subset does not matter for the update.

num_subsets = 4
subsets = []
for k in range(num_subsets):
    idx = xp.arange(k, num_events, num_subsets, device=dev)
    s, e, tb = xp.take(ev_start_d, idx, axis=0), xp.take(ev_end_d, idx, axis=0), xp.take(tof_bin_d, idx, axis=0)
    key_k = xp.zeros(s.shape[0], dtype=xp.int64, device=dev)
    parallelproj_core.tof_lm_sort_key(s, e, origin_d, vs_d, key_k, tof_bin_width, offset_d, tb, num_tof_bins)
    o = xp.argsort(key_k)
    subsets.append((xp.take(s, o, axis=0), xp.take(e, o, axis=0), xp.take(tb, o, axis=0)))

# ... then, in every iteration and for every subset k, project subsets[k] (e.g. MLEM update
# with the expectation fwd(subsets[k]) + contamination and the back projection of the ratio)
print(f"{num_subsets} sorted subsets with {[int(s[0].shape[0]) for s in subsets]} events")
