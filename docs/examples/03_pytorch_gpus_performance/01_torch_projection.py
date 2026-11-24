"""
The Projector as a PyTorch Layer
================================

To use a projector in a PyTorch model (e.g. in an unrolled reconstruction network, or to
fit an image with a PyTorch optimizer), wrap it in a :class:`torch.autograd.Function`:

- ``forward`` computes the forward projection :math:`y = A x`,
- ``backward`` gets the gradient of the loss with respect to :math:`y` and returns the
  gradient with respect to :math:`x`. For a linear operator, this is the **back
  projection** :math:`A^T` of the incoming gradient (the adjoint, see
  :ref:`sphx_glr_auto_examples_02_projector_properties_01_line_integrals_adjointness.py`).

The projectors accept PyTorch tensors directly (CPU or CUDA, no copies). On a GPU, pass
the **current CUDA stream** of the tensor, so that the projections run asynchronously and
in order with all other PyTorch operations (see
:ref:`sphx_glr_auto_examples_03_pytorch_gpus_performance_02_multi_gpu_streams.py`).

This example fits an image to its projections with the Adam optimizer, using a 2D
geometry (one image plane). It runs on the GPU if available, otherwise on the CPU.
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
import torch
import parallelproj_core
from utils import COLORS, disk_phantom, lor_2d, set_style, show_image

set_style()
dev = "cuda" if (torch.cuda.is_available() and parallelproj_core.cuda_enabled == 1) else "cpu"
print(f"parallelproj_core {parallelproj_core.__version__}, PyTorch {torch.__version__}, device {dev}")

# %%
# The geometry: a 2D "sinogram"
# -----------------------------
# 90 directions x 150 parallel LORs (2 mm apart) through the disk phantom (128 x 128 x 1
# voxels of 2 mm). All arrays are PyTorch tensors on ``dev``.

image_np, origin_np, voxel_size_np, extent = disk_phantom()
x_true = torch.from_numpy(image_np).to(dev)
img_origin = torch.from_numpy(origin_np).to(dev)
voxel_size = torch.from_numpy(voxel_size_np).to(dev)
img_shape = tuple(x_true.shape)

lors = [lor_2d(phi, r) for phi in np.linspace(0, 180, 90, endpoint=False) for r in np.linspace(-149, 149, 150)]
lor_start = torch.from_numpy(np.array([s for s, _ in lors])).to(dev)
lor_end = torch.from_numpy(np.array([e for _, e in lors])).to(dev)
print(f"{lor_start.shape[0]} LORs, image {img_shape}")


# %%
# The autograd layer
# ------------------
# ``stream`` is the current CUDA stream for GPU tensors and ``None`` (synchronous) for
# CPU tensors. The geometry is not differentiated, so it is taken from the enclosing
# scope instead of being passed to ``apply``.


def stream_of(t):
    return torch.cuda.current_stream(t.device).cuda_stream if t.is_cuda else None


class ForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.zeros(lor_start.shape[0], dtype=torch.float32, device=x.device)
        parallelproj_core.joseph3d_fwd(lor_start, lor_end, x.contiguous(), img_origin, voxel_size, y,
                                       stream=stream_of(x))
        return y

    @staticmethod
    def backward(ctx, grad_y):
        # the gradient with respect to x is the back projection of the gradient w.r.t. y
        grad_x = torch.zeros(img_shape, dtype=torch.float32, device=grad_y.device)
        parallelproj_core.joseph3d_back(lor_start, lor_end, grad_x, img_origin, voxel_size, grad_y.contiguous(),
                                        stream=stream_of(grad_y))
        return grad_x


project = ForwardProjection.apply

# %%
# Check the gradient
# ------------------
# For the loss :math:`L(x) = \frac{1}{2}\|Ax - y\|^2`, the gradient is
# :math:`A^T (Ax - y)`. The gradient computed by autograd (through ``backward``) must be
# the same as the explicit back projection:

y = project(x_true)  # "measured" data: the projections of the true image
x = torch.rand(img_shape, device=dev, requires_grad=True)
loss = 0.5 * ((project(x) - y) ** 2).sum()
loss.backward()

expected = torch.zeros(img_shape, device=dev)
parallelproj_core.joseph3d_back(lor_start, lor_end, expected, img_origin, voxel_size, (project(x) - y).detach())
print(f"max. relative difference autograd vs. A^T(Ax - y): "
      f"{float((x.grad - expected).abs().max() / expected.abs().max()):.1e}")

# %%
# Fit an image with Adam
# ----------------------
# Start from zeros and minimize the mean squared difference between the projections of
# the image and the data with ``torch.optim.Adam`` (here without noise and without
# regularization, to keep it simple). Every iteration calls the forward projection
# (``project``) and, through ``loss.backward()``, the back projection.

x = torch.zeros(img_shape, device=dev, requires_grad=True)
optimizer = torch.optim.Adam([x], lr=0.2)
losses = []
for it in range(200):
    optimizer.zero_grad()
    loss = ((project(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
print(f"loss: {losses[0]:.3e} (iteration 1) -> {losses[-1]:.3e} (iteration {len(losses)})")

# %%
x_fit = x.detach().cpu()
fig, ax = plt.subplots(1, 4, figsize=(17, 4.2), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2]))
show_image(ax[0], x_true.cpu(), extent, title="(a) true image", vmax=4.2)
show_image(ax[1], x_fit, extent, title=f"(b) fitted (Adam, {len(losses)} iterations)", vmax=4.2)
show_image(ax[2], x_fit - x_true.cpu(), extent, cmap="RdBu_r", title="(c) difference (b) − (a)")
ax[2].images[0].set_clim(-1, 1)
ax[3].semilogy(np.arange(1, len(losses) + 1), losses, color=COLORS["blue"])
ax[3].set_xlabel("iteration")
ax[3].set_ylabel("loss: mean (A x − y)²")
ax[3].set_title("(d) the projector as a differentiable layer")
for a in ax[:3]:
    a.set_xlim(-130, 130)
    a.set_ylim(-130, 130)
fig.show()

# %%
# The image is recovered well; the remaining differences are at the sharp edges of the
# disk and the hot spots, which converge slowest (and are not uniquely determined by a
# limited number of LORs). The learning rate matters: with ``lr=0.05`` instead of 0.2,
# the loss after 200 iterations is about 100 times higher.
#
# Notes
# -----
# - **Batches and channels:** a network works with tensors of shape (batch, channel,
#   ...). Loop over the batch (and channels) in ``forward`` / ``backward``, or reshape;
#   the projectors work on one image at a time.
# - **TOF and listmode:** the same pattern works with the TOF sinogram and TOF listmode
#   projectors; for listmode, the events are passed to ``apply`` as an extra
#   (non-differentiable) argument, see
#   :ref:`sphx_glr_auto_examples_03_pytorch_gpus_performance_03_multi_gpu_tof_listmode.py`.
# - **Poisson data:** for PET data, use the Poisson log-likelihood instead of the squared
#   difference (also shown in the listmode example).
# - **parallelproj** provides ready-to-use operators and PyTorch layers built on these
#   projectors (scanner geometries, sinogram and listmode, attenuation, resolution
#   models), see the examples of `parallelproj <https://parallelproj.readthedocs.io>`_.
