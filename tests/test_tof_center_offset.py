"""Tests for the convention of the TOF center offset.

Documented convention (Doxyfile.in, pp_param_tof_center_offset): the offset
shifts the central TOF bin from the LOR midpoint, and a positive value means a
shift towards the *end point* of the LOR -- independent of the orientation of
the LOR relative to the coordinate axes.
"""

import pytest
import math
import random
import numpy as np
import parallelproj_core as ppb

from types import ModuleType
from .config import pytestmark


def _tof_fwd(xp, dev, kind, xstart, xend, img, img_origin, voxsize, w, sigma,
             offset, nbins, tofbins=None, num_sigmas=5.0):
    """TOF forward projection returning an (nlors, nbins) numpy array
    (sino) or an (nlors,) numpy array (lm)."""
    f32 = xp.float32
    xs = xp.asarray(xstart, dtype=f32, device=dev)
    xe = xp.asarray(xend, dtype=f32, device=dev)
    im = xp.asarray(img, dtype=f32, device=dev)
    org = xp.asarray(img_origin, dtype=f32, device=dev)
    vs = xp.asarray(voxsize, dtype=f32, device=dev)
    sig = xp.asarray([sigma], dtype=f32, device=dev)
    off = xp.asarray(offset, dtype=f32, device=dev)

    if kind == "sino":
        p = xp.zeros((xstart.shape[0], nbins), dtype=f32, device=dev)
        ppb.joseph3d_tof_sino_fwd(xs, xe, im, org, vs, p, w, sig, off, nbins, num_sigmas=num_sigmas)
    else:
        p = xp.zeros(xstart.shape[0], dtype=f32, device=dev)
        tb = xp.asarray(tofbins, dtype=xp.int16, device=dev)
        ppb.joseph3d_tof_lm_fwd(xs, xe, im, org, vs, p, w, sig, off, tb, nbins, num_sigmas=num_sigmas)
    return to_numpy(p).astype(np.float64)


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "get"):  # cupy
        return x.get()
    if hasattr(x, "cpu"):  # torch
        x = x.cpu()
    return np.from_dlpack(x)


@pytest.mark.parametrize("kind", ["sino", "lm"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("end_greater_start", [True, False])
@pytest.mark.parametrize("offset", [-3.0, 2.5])
def test_tof_center_offset_towards_end_point(
    xp: ModuleType, dev: str, kind: str, axis: int, end_greater_start: bool, offset: float
):
    """A point source at the LOR center must appear at a distance -offset from
    the center of the TOF profile, measured from the LOR center towards the end
    point, for both orientations of the LOR."""
    n, w, sigma, nbins = 41, 4.0, 3.0, 31
    c = 0.5 * (nbins - 1)

    dim = [1, 1, 1]
    dim[axis] = n
    img = np.zeros(dim, np.float32)
    img[tuple(n // 2 if k == axis else 0 for k in range(3))] = 1.0
    img_origin = [(-(m / 2) + 0.5) * 1.0 for m in dim]

    s = 1.0 if end_greater_start else -1.0
    xstart = np.zeros((1, 3), np.float32)
    xstart[0, axis] = -60.0 * s
    xend = -xstart

    if kind == "sino":
        p = _tof_fwd(xp, dev, kind, xstart, xend, img, img_origin, [1.0] * 3, w,
                     sigma, [offset], nbins)[0]
    else:
        p = _tof_fwd(xp, dev, kind, np.repeat(xstart, nbins, 0), np.repeat(xend, nbins, 0),
                     img, img_origin, [1.0] * 3, w, sigma, [offset], nbins,
                     tofbins=np.arange(nbins))

    centroid = float((np.arange(nbins) * p).sum() / p.sum())
    # bin i is centered at (i - c) * w + offset (towards the end point), so the
    # source at the LOR center (distance 0) gives a centroid of c - offset / w
    assert math.isclose(centroid, c - offset / w, abs_tol=1e-3)


@pytest.mark.parametrize("kind", ["sino", "lm"])
def test_tof_center_offset_mirroring(xp: ModuleType, dev: str, kind: str):
    """Swapping start and end point of oblique LORs together with mirroring the
    TOF bin index (i -> N - 1 - i) and negating the TOF center offset (since it
    is defined towards the end point) must give identical projections."""
    rng = np.random.default_rng(1)
    nvox, nlors, nbins, w, sigma = 17, 300, 25, 3.0, 4.0
    voxsize = [2.2, 2.5, 2.7]
    img = rng.uniform(0, 1, (nvox, nvox, nvox)).astype(np.float32)
    img_origin = [(-(nvox / 2) + 0.5) * v for v in voxsize]

    def sphere(m):
        v = rng.normal(size=(m, 3))
        return (40.0 * v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    xstart, xend = sphere(nlors), sphere(nlors)
    offsets = rng.uniform(-6.0, 6.0, nlors).astype(np.float32)

    if kind == "sino":
        p = _tof_fwd(xp, dev, kind, xstart, xend, img, img_origin, voxsize, w, sigma,
                     offsets, nbins)
        p_swap = _tof_fwd(xp, dev, kind, xend, xstart, img, img_origin, voxsize, w,
                          sigma, -offsets, nbins)
        p_swap = p_swap[:, ::-1]
    else:
        tb = rng.integers(0, nbins, nlors)
        p = _tof_fwd(xp, dev, kind, xstart, xend, img, img_origin, voxsize, w, sigma,
                     offsets, nbins, tofbins=tb)
        p_swap = _tof_fwd(xp, dev, kind, xend, xstart, img, img_origin, voxsize, w,
                          sigma, -offsets, nbins, tofbins=nbins - 1 - tb)

    assert p.max() > 0
    assert np.allclose(p, p_swap, atol=1e-4 * p.max(), rtol=1e-4)
