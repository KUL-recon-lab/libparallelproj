"""
Multiple GPUs: TOF Sinogram Projections
=======================================

The same as the previous example ("Multiple GPUs and CUDA Streams"), but with the
**TOF sinogram projectors** :func:`parallelproj_core.joseph3d_tof_sino_fwd` and
:func:`parallelproj_core.joseph3d_tof_sino_back`: a mini batch of images is split
across all GPUs, projected synchronously and asynchronously, and used in a training
step with an autograd layer.

The only differences to the non-TOF case are the TOF parameters and the shape of the
projections: a TOF sinogram has one value per LOR **and** TOF bin, i.e. it is
``num_tof_bins`` times larger than a non-TOF sinogram. We therefore use a smaller
scanner (fewer rings) to keep the memory per GPU moderate.

.. note::
  This example needs a CUDA build of ``parallelproj-core``, PyTorch with CUDA support
  and at least one CUDA GPU (two or more to see the speed-up). It is not executed when
  the documentation is built. See the previous example for the background on
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
                       image_origin, random_images)

devices = cuda_devices()  # exits with a message if no GPU can be used
num_gpus = len(devices)
print(f"parallelproj_core {parallelproj_core.__version__}, PyTorch {torch.__version__}")
print(f"using {num_gpus} GPU(s): {[torch.cuda.get_device_name(d) for d in devices]}")

# %%
# Scanner geometry, TOF parameters and images
# -------------------------------------------
# A cylindrical scanner with 8 rings (64 ring pairs) and TOF bins of 20 mm with a TOF
# resolution of 300 ps FWHM (about 45 mm FWHM, sigma 19 mm). The geometry and the TOF
# parameters are copied **once** to every GPU.

img_shape = (128, 128, 32)
voxel_size = torch.tensor([2.0, 2.0, 2.0])
img_origin = image_origin(img_shape, voxel_size)
lor_start, lor_end = sinogram_lors(num_rings=8, ring_spacing=8.0)

tof_bin_width = 20.0  # mm
num_tof_bins = 21  # covers +/- 210 mm around the LOR centers
tof_sigma = torch.tensor([45.0 / 2.355])  # mm
tof_center_offset = torch.tensor([0.0])  # mm
num_sigmas = 3.0

num_lors = lor_start.shape[0]
print(f"{num_lors / 1e6:.2f} million LORs x {num_tof_bins} TOF bins, image {img_shape}")
print(f"TOF sinogram of one image: {num_lors * num_tof_bins * 4 / 1e6:.0f} MB")

geometry = {
    d: dict(lor_start=lor_start.to(d), lor_end=lor_end.to(d),
            img_origin=img_origin.to(d), voxel_size=voxel_size.to(d),
            tof_sigma=tof_sigma.to(d), tof_center_offset=tof_center_offset.to(d))
    for d in devices
}

# %%
# TOF forward and back projection of one image
# --------------------------------------------
# As in the non-TOF case, both functions use the geometry copy of the image's GPU and
# pass the ``stream`` argument through. The TOF sinogram has the shape
# ``(num_lors, num_tof_bins)``.


def tof_fwd_project(x, stream=None):
    geo = geometry[x.device]
    y = torch.zeros((num_lors, num_tof_bins), dtype=torch.float32, device=x.device)
    parallelproj_core.joseph3d_tof_sino_fwd(
        geo["lor_start"], geo["lor_end"], x, geo["img_origin"], geo["voxel_size"], y,
        tof_bin_width, geo["tof_sigma"], geo["tof_center_offset"], num_tof_bins,
        num_sigmas=num_sigmas, stream=stream,
    )
    return y


def tof_back_project(y, stream=None):
    geo = geometry[y.device]
    x = torch.zeros(img_shape, dtype=torch.float32, device=y.device)
    parallelproj_core.joseph3d_tof_sino_back(
        geo["lor_start"], geo["lor_end"], x, geo["img_origin"], geo["voxel_size"], y,
        tof_bin_width, geo["tof_sigma"], geo["tof_center_offset"], num_tof_bins,
        num_sigmas=num_sigmas, stream=stream,
    )
    return x


# %%
# Mini batch across all GPUs: synchronous vs. asynchronous
# --------------------------------------------------------
# A mini batch of ``2 * num_gpus`` images, one chunk of 2 images per GPU.

batch = random_images(2 * num_gpus, img_shape)
chunks = [c.to(d) for c, d in zip(torch.tensor_split(batch, num_gpus), devices)]


def project_batch(chunks, use_streams):
    """TOF forward and back project all images of all chunks (one chunk per GPU)"""
    results = []
    for chunk in chunks:  # loop over GPUs
        out = []
        for x in chunk:  # loop over the images of this GPU
            stream = current_stream(x) if use_streams else None
            out.append(tof_back_project(tof_fwd_project(x, stream), stream))
        results.append(torch.stack(out))
    return results


t_sync = best_time(lambda: project_batch(chunks, use_streams=False), devices)
t_async = best_time(lambda: project_batch(chunks, use_streams=True), devices)

print(f"{num_gpus} GPU(s), batch of {batch.shape[0]} images (TOF forward + back projection)")
print(f"  synchronous calls (stream=None):     {t_sync:.3f} s")
print(f"  asynchronous calls (current stream): {t_async:.3f} s  ->  speed-up {t_sync / t_async:.2f}")

res_sync = project_batch(chunks, use_streams=False)
res_async = project_batch(chunks, use_streams=True)
rel_diff = max(
    float((a.cpu() - b.cpu()).abs().max() / b.cpu().abs().max()) for a, b in zip(res_sync, res_async)
)
print(f"max. relative difference synchronous vs. asynchronous: {rel_diff:.2e}")

# %%
# Autograd layer and training step
# --------------------------------
# The autograd layer is the same as in the non-TOF case: pass the current stream of
# the tensor's GPU in ``forward`` and ``backward``. The output of ``forward`` is a TOF
# sinogram, and ``backward`` receives the gradient with respect to the TOF sinogram.


class TOFForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return tof_fwd_project(x, stream=current_stream(x))

    @staticmethod
    def backward(ctx, grad_y):
        # the adjoint of the TOF forward projection is the TOF back projection
        return tof_back_project(grad_y.contiguous(), stream=current_stream(grad_y))


def tof_projection_batch(x_chunk):
    """TOF forward project all images of a chunk (images on one GPU)"""
    return torch.stack([TOFForwardProjection.apply(x) for x in x_chunk])


# simulated TOF data of every chunk, on the GPU of the chunk
data = [tof_projection_batch(c).detach() for c in chunks]

# initial images (the parameters to optimize), one tensor per GPU
x = [torch.ones_like(c, requires_grad=True) for c in chunks]


def training_step():
    for xi in x:
        xi.grad = None
    losses = [((tof_projection_batch(xi) - yi) ** 2).sum() for xi, yi in zip(x, data)]
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
    [2 * tof_back_project(tof_fwd_project(xi.detach()) - yi) for xi, yi in zip(x[0], data[0])]
)
rel_err = float((x[0].grad - expected).abs().max() / expected.abs().max())
print(f"relative error of the gradient: {rel_err:.2e}")

# %%
# Notes
# -----
# - **Memory:** a TOF sinogram is ``num_tof_bins`` times larger than a non-TOF
#   sinogram. With large scanners, the TOF sinograms of a mini batch often dominate
#   the GPU memory; the next example uses listmode data instead.
# - **LOR-dependent TOF parameters:** ``tof_sigma`` and ``tof_center_offset`` can also
#   have one value per LOR (shape ``(num_lors,)``), copied once to every GPU like the
#   geometry.
