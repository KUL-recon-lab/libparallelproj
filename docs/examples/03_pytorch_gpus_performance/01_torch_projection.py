"""
Torch Projection Layer
======================

This minimal example demonstrates how to use the `parallelproj_core` library
with torch tensors and how to backward propagate gradients through the forward projection operation.
"""

import parallelproj_core
import array_api_compat.torch as torch
from array_api_compat import device


# %%
# Setup a custom torch autograd function to wraps the forward projection.
# We also implement the backward path (Jacobian-vector product) which in case of a linear operator
# is equivalent to its adjoint (back projection).
# Here we operator on a singel 3D tensor. In practice, this would probably be a batch of 3D tensors, with a signle channel (5D tensor).


class FwdProjLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lor_start, lor_end, img_origin, voxel_size):
        dev = device(x)
        img_fwd = torch.zeros(lor_start.shape[0], dtype=torch.float32, device=dev)
        parallelproj_core.joseph3d_fwd(
            lor_start, lor_end, x, img_origin, voxel_size, img_fwd
        )
        # save context variables for backward pass
        image_shape = torch.asarray(x.shape)
        ctx.save_for_backward(lor_start, lor_end, img_origin, voxel_size, image_shape)
        return img_fwd

    @staticmethod
    def backward(ctx, grad_output):
        dev = device(grad_output)
        lor_start, lor_end, img_origin, voxel_size, image_shape = ctx.saved_tensors
        grad_input = torch.zeros(tuple(image_shape), dtype=torch.float32, device=dev)
        parallelproj_core.joseph3d_back(
            lor_start,
            lor_end,
            grad_input,
            img_origin,
            voxel_size,
            grad_output,
        )
        return grad_input, None, None, None, None


# %%
# select device
if torch.cuda.is_available() and parallelproj_core.cuda_enabled == 1:
    dev = "cuda"
else:
    dev = "cpu"


# %%
# Print backend and device info.
print(f"parallelproj_core version: {parallelproj_core.__version__}")
print(f"parallelproj_core cuda enabled: {parallelproj_core.cuda_enabled}")
print(f"using array API compatible library: {torch.__name__} on device {dev}")

# %%
# Define a mini sparse demo image.
image = torch.zeros((3, 3, 3), dtype=torch.float32, device=dev)
image[0, 0, 2] = 0.25
image[2, 0, 0] = 0.25
image[2, 2, 2] = 0.5

voxel_size = torch.asarray([2.0, 2.0, 2.0], device=dev, dtype=torch.float32)
img_origin = torch.asarray([-1.0, -1.0, -1.0], device=dev, dtype=torch.float32)

# %%
# Define LOR start and end points.
lor_start = torch.asarray(
    [[7.0, 2.0, 2.0], [-1, -1, -5], [3, -3, -3]],
    device=dev,
    dtype=torch.float32,
)
lor_end = torch.asarray(
    [[-5.0, 2.0, 2.0], [-1, -1, 7], [3, 6, 6]],
    device=dev,
    dtype=torch.float32,
)

# %%
fwd_proj_layer = FwdProjLayer.apply
img_fwd = fwd_proj_layer(image, lor_start, lor_end, img_origin, voxel_size)
print("Forward projection result:", img_fwd)

# %%
# Run a torch gradient check to verify that the backward projection correctly computes gradients.
# Since all parallelproj_core functions use float32, we set the eps, atol, and rtol parameters to higher values
# to account for numerical precision issues.

x = torch.rand(image.shape, dtype=torch.float32, device=dev, requires_grad=True)

print("Running forward projection layer gradient test")
grad_test_fwd = torch.autograd.gradcheck(
    fwd_proj_layer,
    (x, lor_start, lor_end, img_origin, voxel_size),
    eps=1e-2,
    atol=1e-5,
    rtol=1e-5,
)

assert grad_test_fwd, "Gradient check failed!"

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

fig, ax = plt.subplots(figsize=(2.6, 1.6), layout="constrained")
ax.axis("off")

r = 0.035  # neuron radius

x_in, x_h1, x_h2, x_out = 0.12, 0.38, 0.64, 0.88
ys_in = [0.35, 0.65]  # 2 inputs
ys_h1 = [0.25, 0.50, 0.75]  # 3 hidden
ys_h2 = [0.25, 0.50, 0.75]  # 3 hidden
ys_out = [0.50]  # 1 output


def connect(x0, ys0, x1, ys1):
    for y0 in ys0:
        for y1 in ys1:
            ax.plot([x0 + r, x1 - r], [y0, y1], lw=0.8, alpha=0.35)


connect(x_in, ys_in, x_h1, ys_h1)
connect(x_h1, ys_h1, x_h2, ys_h2)
connect(x_h2, ys_h2, x_out, ys_out)


def neurons(x, ys):
    for y in ys:
        ax.add_patch(Circle((x, y), r, fill=True, alpha=0.95))


neurons(x_in, ys_in)
neurons(x_h1, ys_h1)
neurons(x_h2, ys_h2)
neurons(x_out, ys_out)

ax.text(x_in, 0.06, "in", ha="center", va="center", fontsize=9)
ax.text(x_h1, 0.06, "h1", ha="center", va="center", fontsize=9)
ax.text(x_h2, 0.06, "h2", ha="center", va="center", fontsize=9)
ax.text(x_out, 0.06, "out", ha="center", va="center", fontsize=9)

fig.show()
# sphinx_gallery_end_ignore
