"""
Multiple GPUs: TOF Listmode Projections
=======================================

The TOF listmode version of the previous examples: every sample of a mini batch has its
own **list of events**, and the number of events differs from sample to sample. The
samples are distributed across all GPUs, projected with the TOF listmode projectors
:func:`parallelproj_core.joseph3d_tof_lm_fwd` and
:func:`parallelproj_core.joseph3d_tof_lm_back`, and used in a training step with the
Poisson listmode log-likelihood.

Compared to the sinogram examples, three things are new:

- the event lists of the samples have **different lengths**, so they cannot be stacked
  into one tensor: every sample keeps its own event tensors on its GPU,
- the samples are assigned to the GPUs such that every GPU gets about the **same number
  of events** (the projection time is roughly proportional to the number of events),
- the listmode data fidelity needs the **sensitivity image** (the back projection of all
  possible LORs), computed once per GPU.

.. note::
  This example needs a CUDA build of ``parallelproj-core``, PyTorch with CUDA support
  and at least one CUDA GPU (two or more to see the speed-up). It is not executed when
  the documentation is built. See the first multi-GPU example for the background on
  synchronous and asynchronous calls and on timing GPU code.
"""

# %%
import torch
import parallelproj_core
# sphinx_gallery_start_ignore
# the helper modules (utils.py, gpu_utils.py) are in the parent directory (docs/examples)
import pathlib, sys
try:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
except NameError:  # run by sphinx-gallery: docs/conf.py adds docs/examples to the path
    pass
# sphinx_gallery_end_ignore
from gpu_utils import (cuda_devices, current_stream, best_time, sinogram_lors,
                       image_origin, random_images, simulate_lm_events)

devices = cuda_devices()  # exits with a message if no GPU can be used
num_gpus = len(devices)
print(f"parallelproj_core {parallelproj_core.__version__}, PyTorch {torch.__version__}")
print(f"using {num_gpus} GPU(s): {[torch.cuda.get_device_name(d) for d in devices]}")

# %%
# Scanner, TOF parameters and per-GPU constants
# ---------------------------------------------
# The same cylindrical scanner as in the first example. Per GPU we keep the image
# geometry, the TOF parameters and the **sensitivity image** :math:`s = A^T 1`, the
# (non-TOF) back projection of all LORs of the scanner. It is needed by the listmode
# log-likelihood and computed once per GPU.

img_shape = (128, 128, 48)
voxel_size = torch.tensor([2.0, 2.0, 2.0])
img_origin = image_origin(img_shape, voxel_size)
scanner_radius = 350.0

tof_bin_width = 10.0  # mm
num_tof_bins = 61  # covers +/- 305 mm around the LOR centers
tof_sigma = torch.tensor([45.0 / 2.355])  # mm (300 ps FWHM)
tof_center_offset = torch.tensor([0.0])  # mm
num_sigmas = 3.0

lor_start, lor_end = sinogram_lors(radius=scanner_radius)

per_gpu = {}
for d in devices:
    c = dict(img_origin=img_origin.to(d), voxel_size=voxel_size.to(d),
             tof_sigma=tof_sigma.to(d), tof_center_offset=tof_center_offset.to(d))
    sens = torch.zeros(img_shape, dtype=torch.float32, device=d)
    ls, le = lor_start.to(d), lor_end.to(d)
    parallelproj_core.joseph3d_back(ls, le, sens, c["img_origin"], c["voxel_size"],
                                    torch.ones(ls.shape[0], dtype=torch.float32, device=d))
    c["sensitivity"] = sens
    per_gpu[d] = c
    del ls, le  # the sinogram LORs are only needed for the sensitivity image

# %%
# Simulate listmode data with different numbers of events per sample
# -------------------------------------------------------------------
# Every sample of the mini batch is a phantom with a different total activity, so the
# number of events differs from sample to sample (here between 1 and 4 million).

batch_size = 2 * num_gpus
images = random_images(batch_size, img_shape)
gen = torch.Generator().manual_seed(1)
num_events = torch.randint(1_000_000, 4_000_000, (batch_size,), generator=gen).tolist()

events = []  # one dict of event tensors (on the CPU for now) per sample
for b in range(batch_size):
    start, end, tof_bin = simulate_lm_events(
        images[b], img_origin, voxel_size, num_events[b], scanner_radius,
        tof_bin_width, num_tof_bins, float(tof_sigma), seed=b,
    )
    events.append(dict(lor_start=start, lor_end=end, tof_bin=tof_bin))
print("events per sample:", [e["tof_bin"].shape[0] for e in events])

# %%
# Distribute the samples across the GPUs (load balancing)
# -------------------------------------------------------
# The time of a listmode projection is roughly proportional to the number of events.
# Assigning "2 samples per GPU" could give one GPU much more work than another, and
# all GPUs would wait for the slowest one. A simple greedy assignment (largest sample
# first, to the GPU with the fewest events so far) balances the load.

order = sorted(range(batch_size), key=lambda b: -events[b]["tof_bin"].shape[0])
assignment = {d: [] for d in devices}
load = {d: 0 for d in devices}
for b in order:
    d = min(devices, key=lambda dev: load[dev])
    assignment[d].append(b)
    load[d] += events[b]["tof_bin"].shape[0]

for d in devices:
    print(f"{d}: samples {assignment[d]}, {load[d] / 1e6:.2f} million events")

# move every sample (image and events) to its GPU
samples = []  # (device, image, events) for all samples
for d in devices:
    for b in assignment[d]:
        ev = {k: v.to(d) for k, v in events[b].items()}
        samples.append((d, images[b].to(d), ev))

# %%
# TOF listmode forward and back projection of one sample
# ------------------------------------------------------
# The forward projection gives one value per event (the expected value of the TOF
# bin of the event's LOR), the back projection maps one value per event back to the
# image.


def lm_fwd_project(x, ev, stream=None):
    c = per_gpu[x.device]
    y = torch.zeros(ev["tof_bin"].shape[0], dtype=torch.float32, device=x.device)
    parallelproj_core.joseph3d_tof_lm_fwd(
        ev["lor_start"], ev["lor_end"], x, c["img_origin"], c["voxel_size"], y,
        tof_bin_width, c["tof_sigma"], c["tof_center_offset"], ev["tof_bin"], num_tof_bins,
        num_sigmas=num_sigmas, stream=stream,
    )
    return y


def lm_back_project(y, ev, stream=None):
    c = per_gpu[y.device]
    x = torch.zeros(img_shape, dtype=torch.float32, device=y.device)
    parallelproj_core.joseph3d_tof_lm_back(
        ev["lor_start"], ev["lor_end"], x, c["img_origin"], c["voxel_size"], y,
        tof_bin_width, c["tof_sigma"], c["tof_center_offset"], ev["tof_bin"], num_tof_bins,
        num_sigmas=num_sigmas, stream=stream,
    )
    return x


# %%
# Synchronous vs. asynchronous
# ----------------------------
# As an example workload we compute the listmode MLEM "ratio" back projection
# :math:`A^T (1 / (A x))` of every sample, the core of an MLEM / OSEM update.


def ratio_backprojection(samples, use_streams):
    out = []
    for d, x, ev in samples:
        stream = current_stream(x) if use_streams else None
        out.append(lm_back_project(1.0 / (lm_fwd_project(x, ev, stream) + 1e-6), ev, stream))
    return out


t_sync = best_time(lambda: ratio_backprojection(samples, use_streams=False), devices)
t_async = best_time(lambda: ratio_backprojection(samples, use_streams=True), devices)

print(f"{num_gpus} GPU(s), batch of {batch_size} samples, "
      f"{sum(load.values()) / 1e6:.1f} million events in total")
print(f"  synchronous calls (stream=None):     {t_sync:.3f} s")
print(f"  asynchronous calls (current stream): {t_async:.3f} s  ->  speed-up {t_sync / t_async:.2f}")

res_sync = ratio_backprojection(samples, use_streams=False)
res_async = ratio_backprojection(samples, use_streams=True)
rel_diff = max(
    float((a.cpu() - b.cpu()).abs().max() / b.cpu().abs().max()) for a, b in zip(res_sync, res_async)
)
print(f"max. relative difference synchronous vs. asynchronous: {rel_diff:.2e}")

# %%
# Autograd layer with event lists
# -------------------------------
# The listmode projection depends on the image **and** on the events of the sample.
# The events are passed to ``apply`` as an extra (non-differentiable) argument;
# ``backward`` returns ``None`` for it. As before, the current stream of the tensor's
# GPU is passed in ``forward`` and ``backward``.


class LMForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ev):
        ctx.ev = ev
        return lm_fwd_project(x, ev, stream=current_stream(x))

    @staticmethod
    def backward(ctx, grad_y):
        grad_x = lm_back_project(grad_y.contiguous(), ctx.ev, stream=current_stream(grad_y))
        return grad_x, None  # no gradient for the events


# %%
# Training step with the Poisson listmode log-likelihood
# ------------------------------------------------------
# For listmode data, the negative Poisson log-likelihood of an image :math:`x` is
#
# .. math::
#
#    L(x) = \sum_i s_i x_i - \sum_e \log\left( (A x)_e \right),
#
# where :math:`s = A^T 1` is the sensitivity image and :math:`(A x)_e` the TOF
# listmode forward projection of event :math:`e`. Its gradient is
# :math:`s - A^T \left(1 / (A x)\right)`. Every GPU computes the loss of its samples,
# the losses are summed on GPU 0, and one call of ``backward()`` computes the
# gradients of all samples, each on its own GPU.

# initial images (the parameters to optimize), one per sample, on the sample's GPU
eps = 1e-6  # avoids log(0) for events with zero expectation
x = [torch.ones(img_shape, device=d, requires_grad=True) for d, _, _ in samples]


def training_step():
    for xi in x:
        xi.grad = None
    losses = []
    for xi, (d, _, ev) in zip(x, samples):
        expectation = LMForwardProjection.apply(xi, ev)
        losses.append((per_gpu[d]["sensitivity"] * xi).sum() - torch.log(expectation + eps).sum())
    loss = sum(l.to(devices[0]) for l in losses)
    loss.backward()
    return loss


t_step = best_time(training_step, devices)
loss = training_step()
print(f"training step on {num_gpus} GPU(s): {t_step:.3f} s, loss {loss.item():.6e}")

# %%
# Check the gradient of the first sample against its analytic value
# :math:`s - A^T (1 / (A x + \epsilon))`, computed with synchronous calls:

d0, _, ev0 = samples[0]
x0 = x[0].detach()
expected = per_gpu[d0]["sensitivity"] - lm_back_project(1.0 / (lm_fwd_project(x0, ev0) + eps), ev0)
rel_err = float((x[0].grad - expected).abs().max() / expected.abs().max())
print(f"relative error of the gradient: {rel_err:.2e}")

# %%
# Notes
# -----
# - **Load balancing** matters more for listmode than for sinograms: with very
#   different event counts, assign samples by number of events (as above), or split
#   large samples into several event chunks (a listmode projection of a subset of the
#   events is just the corresponding part of the full projection).
# - **Data transfer:** here all events are moved to the GPUs once. When streaming new
#   events for every mini batch, keep them in pinned host memory
#   (``tensor.pin_memory()``) and copy with ``.to(device, non_blocking=True)``, so that
#   the copies of all GPUs overlap as well.
# - **Event chunks** (e.g. subsets in OSEM) work the same way: every chunk is just a
#   shorter event list.
