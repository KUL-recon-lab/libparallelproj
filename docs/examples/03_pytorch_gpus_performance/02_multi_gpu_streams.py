"""
Multiple GPUs and CUDA Streams
==============================

This example shows how to use the projectors of :mod:`parallelproj_core` with one or
several GPUs from PyTorch, e.g. to process the samples of a mini batch on all GPUs in
parallel. You will learn

- what the ``stream`` argument of the projectors does (synchronous vs. asynchronous calls),
- how to split a mini batch across several GPUs so that all GPUs work at the same time,
- how to write a PyTorch autograd layer that works with one or several GPUs,
- how to time GPU code correctly.

.. note::
  This example needs a CUDA build of ``parallelproj-core``, PyTorch with CUDA support
  and at least one CUDA GPU. With two or more GPUs, it shows the speed-up of running
  them in parallel. It is not executed when the documentation is built (no GPU).

Background: synchronous and asynchronous calls
----------------------------------------------

GPU work is queued on *CUDA streams*. PyTorch (and CuPy) run every operation on the
*current stream* of a GPU and return immediately, before the GPU is done: the CPU can
already queue the next operations, and operations on the same stream run in order.
Anything that needs the result on the CPU (``.item()``, ``.cpu()``, ``print(x)``, ...)
waits automatically.

The projectors of :mod:`parallelproj_core` support both modes via their ``stream`` argument:

- ``stream=None`` (default): the call is **synchronous**. It returns when the result
  is complete. Simple and safe, but the CPU waits for every projection.
- ``stream=<handle>``: the call is **asynchronous** on that stream, exactly like a
  PyTorch operation. Pass the current stream of the GPU of your tensors,
  ``torch.cuda.current_stream(device).cuda_stream``. ``0`` is PyTorch's default stream.

With asynchronous calls, projections on different GPUs run **at the same time**, even
if they are called one after another from a single Python thread.
"""

# %%
# The helper module ``gpu_utils.py`` (next to the examples) contains the device
# check, the current stream of a tensor, the timing function and the scanner
# geometry / phantoms used by all multi-GPU examples. The two GPU-specific helpers are
#
# .. code-block:: python
#
#     def current_stream(x):
#         """handle of the current CUDA stream of the GPU of tensor x"""
#         return torch.cuda.current_stream(x.device).cuda_stream
#
#     def synchronize_all(devices):
#         """wait until all queued work on all GPUs is done"""
#         for d in devices:
#             torch.cuda.synchronize(d)

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
                       image_origin, random_images)

devices = cuda_devices()  # exits with a message if no GPU can be used
num_gpus = len(devices)
print(f"parallelproj_core {parallelproj_core.__version__}, PyTorch {torch.__version__}")
print(f"using {num_gpus} GPU(s): {[torch.cuda.get_device_name(d) for d in devices]}")

# %%
# Scanner geometry and images
# ---------------------------
# A simple cylindrical scanner (sinogram LORs) and an image grid. The LOR
# coordinates are copied **once** to every GPU, so that the projections on each GPU
# only use data that is already on that GPU.


img_shape = (128, 128, 48)
voxel_size = torch.tensor([2.0, 2.0, 2.0])
img_origin = image_origin(img_shape, voxel_size)
lor_start, lor_end = sinogram_lors()
print(f"{lor_start.shape[0] / 1e6:.1f} million LORs, image {img_shape}")

# one copy of the geometry per GPU
geometry = {
    d: dict(lor_start=lor_start.to(d), lor_end=lor_end.to(d),
            img_origin=img_origin.to(d), voxel_size=voxel_size.to(d))
    for d in devices
}


# %%
# Forward and back projection of one image
# ----------------------------------------
# Both functions take an image on one GPU and use the geometry copy of that GPU. The
# ``stream`` argument is passed through: ``None`` for synchronous calls, or a stream
# handle for asynchronous calls.


def fwd_project(x, stream=None):
    geo = geometry[x.device]
    y = torch.zeros(geo["lor_start"].shape[0], dtype=torch.float32, device=x.device)
    parallelproj_core.joseph3d_fwd(
        geo["lor_start"], geo["lor_end"], x, geo["img_origin"], geo["voxel_size"], y, stream=stream
    )
    return y


def back_project(y, stream=None):
    geo = geometry[y.device]
    x = torch.zeros(img_shape, dtype=torch.float32, device=y.device)
    parallelproj_core.joseph3d_back(
        geo["lor_start"], geo["lor_end"], x, geo["img_origin"], geo["voxel_size"], y, stream=stream
    )
    return x


# %%
# Split a mini batch across all GPUs
# ----------------------------------
# A mini batch of ``2 * num_gpus`` images is split into one chunk per GPU (2 images
# per GPU). Each chunk is moved to its GPU.

batch = random_images(2 * num_gpus, img_shape)
chunks = [c.to(d) for c, d in zip(torch.tensor_split(batch, num_gpus), devices)]


def project_batch(chunks, use_streams):
    """forward and back project all images of all chunks (one chunk per GPU)"""
    results = []
    for chunk in chunks:  # loop over GPUs
        out = []
        for x in chunk:  # loop over the images of this GPU
            stream = current_stream(x) if use_streams else None
            out.append(back_project(fwd_project(x, stream), stream))
        results.append(torch.stack(out))
    return results


# %%
# Timing GPU code correctly
# -------------------------
# Because GPU calls can return before the GPU is done, a timer around them would only
# measure how long it takes to *queue* the work. To measure the real time:
#
# - **synchronize** all GPUs before starting and before stopping the clock,
# - do a **warm-up** call first (the first call includes one-time initialization),
# - repeat a few times and take the best (or median) time.
#
# Alternatives are CUDA events (``torch.cuda.Event(enable_timing=True)``) or
# ``torch.utils.benchmark.Timer``, which take care of synchronization and warm-up.
#
# ``best_time(fn, devices)`` from ``gpu_utils`` does exactly that:
#
# .. code-block:: python
#
#     def best_time(fn, devices, repeats=3):
#         fn()  # warm-up
#         times = []
#         for _ in range(repeats):
#             synchronize_all(devices)
#             t0 = time.perf_counter()
#             fn()
#             synchronize_all(devices)
#             times.append(time.perf_counter() - t0)
#         return min(times)

# %%
# Synchronous vs. asynchronous calls
# ----------------------------------
# **Synchronous** (``stream=None``): every call waits until its GPU is done, so the
# GPUs work one after another. **Asynchronous** (current stream of each GPU): the
# loop only queues the work, and all GPUs run at the same time.

t_sync = best_time(lambda: project_batch(chunks, use_streams=False), devices)
t_async = best_time(lambda: project_batch(chunks, use_streams=True), devices)

print(f"{num_gpus} GPU(s), batch of {batch.shape[0]} images (forward + back projection)")
print(f"  synchronous calls (stream=None):     {t_sync:.3f} s")
print(f"  asynchronous calls (current stream): {t_async:.3f} s  ->  speed-up {t_sync / t_async:.2f}")

# %%
# Both give the same results, up to float32 rounding: the back projection adds the
# contributions of many LORs to each voxel with atomic additions, whose order varies
# from run to run. Note that no explicit synchronization is needed here: moving the
# results to the CPU (``.cpu()``) waits for the GPUs automatically, as for every
# PyTorch operation.

res_sync = project_batch(chunks, use_streams=False)
res_async = project_batch(chunks, use_streams=True)
rel_diff = max(
    float((a.cpu() - b.cpu()).abs().max() / b.cpu().abs().max()) for a, b in zip(res_sync, res_async)
)
print(f"max. relative difference synchronous vs. asynchronous: {rel_diff:.2e}")

# %%
# An autograd layer for training
# ------------------------------
# To use the projector in a network, wrap it in a :class:`torch.autograd.Function`.
# The only thing needed for one or several GPUs is to pass the **current stream of
# the tensor's GPU** in ``forward`` and ``backward``:
#
# - the projector then behaves like any other PyTorch operation (asynchronous,
#   correctly ordered with all other operations, no extra synchronization needed),
# - PyTorch runs the backward pass of each GPU in its own thread and on the same
#   stream as the forward pass, so the backward projections of all GPUs run in
#   parallel as well.


class ForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return fwd_project(x, stream=current_stream(x))

    @staticmethod
    def backward(ctx, grad_y):
        # the adjoint of the forward projection is the back projection
        return back_project(grad_y.contiguous(), stream=current_stream(grad_y))


def forward_projection_batch(x_chunk):
    """forward project all images of a chunk (images on one GPU)"""
    return torch.stack([ForwardProjection.apply(x) for x in x_chunk])


# %%
# A toy "training step": every GPU computes the data fidelity loss
# :math:`\|A x - y\|^2` of its chunk, the losses are summed on GPU 0, and one call of
# ``backward()`` computes the gradients of all chunks, each on its own GPU.

# simulated data of every chunk, on the GPU of the chunk
data = [forward_projection_batch(c).detach() for c in chunks]

# initial images (the parameters to optimize), one tensor per GPU
x = [torch.ones_like(c, requires_grad=True) for c in chunks]


def training_step():
    for xi in x:
        xi.grad = None
    losses = [((forward_projection_batch(xi) - yi) ** 2).sum() for xi, yi in zip(x, data)]
    loss = sum(l.to(devices[0]) for l in losses)
    loss.backward()
    return loss


t_step = best_time(training_step, devices)
loss = training_step()
print(f"training step on {num_gpus} GPU(s): {t_step:.3f} s, loss {loss.item():.4e}")

# %%
# Check the gradient of the first chunk against its analytic value
# :math:`2 A^T (A x - y)`, computed with synchronous calls:

expected = torch.stack(
    [2 * back_project(fwd_project(xi.detach()) - yi) for xi, yi in zip(x[0], data[0])]
)
rel_err = float((x[0].grad - expected).abs().max() / expected.abs().max())
print(f"relative error of the gradient: {rel_err:.2e}")

# %%
# Summary and further notes
# -------------------------
# - **One GPU:** the same code works. Asynchronous calls (current stream) avoid that the
#   CPU waits for every projection; there is no speed-up from parallel GPUs, of course.
#   No extra synchronization is needed in normal code, only for timing.
# - **DistributedDataParallel (one process per GPU, e.g. with** ``torchrun`` **):** use
#   the autograd layer as above. Every process uses only its own GPU, and passing the
#   current stream lets the projections overlap with the gradient all-reduce.
# - **Side streams:** inside ``with torch.cuda.stream(s):`` the current stream is ``s``,
#   so passing the current stream keeps the projector correctly ordered with the other
#   operations on ``s``.
# - **CUDA graphs:** projections with GPU tensors and the current stream can be recorded
#   with ``torch.cuda.graph`` (the current stream is the capture stream).
# - **CuPy:** use ``cupy.cuda.get_current_stream().ptr`` as stream handle. During CUDA
#   graph capture, pass CuPy arrays as ``x.__dlpack__(stream=handle)``.
# - **CPU arrays** (NumPy) work as well; such calls always return complete results.
# - **TOF:** the next examples do the same with the TOF sinogram projectors and with
#   the TOF listmode projectors (different numbers of events per sample).
