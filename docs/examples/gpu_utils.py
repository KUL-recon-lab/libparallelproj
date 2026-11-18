"""Helpers shared by the multi-GPU examples (``*_multi_gpu_*.py``), not an example itself."""

import sys
import time
import torch
import parallelproj_core


################################################################################
# GPUs, streams, timing
################################################################################


def cuda_devices() -> list[torch.device]:
    """all CUDA devices; exits with a message if no GPU can be used"""
    if not (parallelproj_core.cuda_enabled == 1 and torch.cuda.is_available()):
        print("This example needs a CUDA build of parallelproj_core, PyTorch with CUDA and a GPU.")
        sys.exit(0)
    return [torch.device(f"cuda:{k}") for k in range(torch.cuda.device_count())]


def current_stream(x: torch.Tensor) -> int:
    """handle of the current CUDA stream of the GPU of tensor x (for the ``stream`` argument)"""
    return torch.cuda.current_stream(x.device).cuda_stream


def synchronize_all(devices):
    """wait until all queued work on all GPUs is done"""
    for d in devices:
        torch.cuda.synchronize(d)


def best_time(fn, devices, repeats=3) -> float:
    """best wall-clock time of fn() over a few runs, after a warm-up call.

    GPU calls can return before the GPU is done, so all GPUs are synchronized
    before starting and before stopping the clock."""
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        synchronize_all(devices)
        t0 = time.perf_counter()
        fn()
        synchronize_all(devices)
        times.append(time.perf_counter() - t0)
    return min(times)


################################################################################
# scanner geometry, images, listmode events
################################################################################


def sinogram_lors(radius=350.0, num_rad=200, num_views=180, num_rings=16, ring_spacing=4.0):
    """start and end points of all LORs of a cylindrical scanner (direct and oblique planes),
    shape (num_rings**2 * num_views * num_rad, 3)"""
    phi = torch.arange(num_views) * torch.pi / num_views
    s = (torch.arange(num_rad) - (num_rad - 1) / 2) * 2.5
    P, S = torch.meshgrid(phi, s, indexing="ij")
    a = torch.arccos(S / radius)
    p1 = torch.stack([radius * torch.cos(P + a), radius * torch.sin(P + a)], -1).reshape(-1, 2)
    p2 = torch.stack([radius * torch.cos(P - a), radius * torch.sin(P - a)], -1).reshape(-1, 2)
    z = (torch.arange(num_rings) - (num_rings - 1) / 2) * ring_spacing
    z1, z2 = torch.meshgrid(z, z, indexing="ij")
    z1, z2 = z1.reshape(-1, 1, 1), z2.reshape(-1, 1, 1)
    n = p1.shape[0]
    start = torch.cat([p1.expand(z1.shape[0], n, 2), z1.expand(-1, n, 1)], -1).reshape(-1, 3)
    end = torch.cat([p2.expand(z2.shape[0], n, 2), z2.expand(-1, n, 1)], -1).reshape(-1, 3)
    return start.float().contiguous(), end.float().contiguous()


def image_origin(img_shape, voxel_size: torch.Tensor) -> torch.Tensor:
    """world coordinates of the center of voxel [0,0,0] of an image centered at the origin"""
    return (-torch.tensor(img_shape) / 2 + 0.5) * voxel_size


def random_images(batch_size, img_shape, seed=0) -> torch.Tensor:
    """a batch of simple phantoms: a cylinder with a few random hot spheres"""
    g = torch.Generator().manual_seed(seed)
    x = torch.arange(img_shape[0]) - img_shape[0] / 2
    y = torch.arange(img_shape[1]) - img_shape[1] / 2
    z = torch.arange(img_shape[2]) - img_shape[2] / 2
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    r_cyl = 0.4 * min(img_shape[0], img_shape[1])
    imgs = torch.zeros((batch_size,) + tuple(img_shape))
    for b in range(batch_size):
        img = ((X**2 + Y**2) < r_cyl**2).float()
        for _ in range(4):
            c = (torch.rand(3, generator=g) - 0.5) * torch.tensor(
                [1.2 * r_cyl, 1.2 * r_cyl, 0.6 * img_shape[2]]
            )
            r = 0.04 * r_cyl + 0.12 * r_cyl * torch.rand(1, generator=g)
            img += 3.0 * (((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) < r**2).float()
        imgs[b] = img
    return imgs


def simulate_lm_events(img, img_origin, voxel_size, num_events, radius, tof_bin_width,
                       num_tof_bins, tof_sigma, max_tilt=0.1, seed=0):
    """simulate TOF listmode events of an activity image (toy model, no attenuation /
    scatter / randoms / detector effects).

    Emission points are drawn from the image, LOR directions are random (transaxial
    angle uniform, small axial tilt), and the LOR end points are the intersections
    with a cylinder of the given radius. The TOF bin is the (Gaussian blurred) signed
    distance of the emission point from the LOR center towards the end point.

    Returns lor_start, lor_end (float32, (n, 3)) and tof_bin (int16, (n,)); events
    whose TOF bin is outside of the bin range are dropped."""
    g = torch.Generator().manual_seed(seed)
    p = img.flatten().double()
    idx = torch.multinomial(p / p.sum(), num_events, replacement=True, generator=g)
    vox = torch.stack(torch.unravel_index(idx, img.shape), -1).float()
    pos = img_origin + (vox + torch.rand(num_events, 3, generator=g) - 0.5) * voxel_size

    phi = torch.rand(num_events, generator=g) * torch.pi
    tilt = (2 * torch.rand(num_events, generator=g) - 1) * max_tilt
    u = torch.stack([torch.cos(phi), torch.sin(phi), tilt], -1)
    u = u / u.norm(dim=1, keepdim=True)

    # intersection with the cylinder x^2 + y^2 = radius^2
    a = u[:, 0] ** 2 + u[:, 1] ** 2
    b = 2 * (pos[:, 0] * u[:, 0] + pos[:, 1] * u[:, 1])
    c = pos[:, 0] ** 2 + pos[:, 1] ** 2 - radius**2
    sq = torch.sqrt(b**2 - 4 * a * c)
    t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    start = pos + t1[:, None] * u
    end = pos + t2[:, None] * u

    # TOF bin: distance of the emission point from the LOR center towards the end point
    dist = -0.5 * (t1 + t2) + tof_sigma * torch.randn(num_events, generator=g)
    tof_bin = torch.round(dist / tof_bin_width + (num_tof_bins - 1) / 2)
    keep = (tof_bin >= 0) & (tof_bin < num_tof_bins)
    return (start[keep].float().contiguous(), end[keep].float().contiguous(),
            tof_bin[keep].to(torch.int16).contiguous())
