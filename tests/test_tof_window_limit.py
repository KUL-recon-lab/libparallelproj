"""Tests of the 64-bin limit of the TOF window (PARALLELPROJ_TOF_MAX_WINDOW_BINS).

Background (see src/tof_kernel.h, tof_window): for every image plane along a LOR, the TOF
projectors compute the weights of the TOF bins that overlap the support of the TOF kernel
(the "TOF window"). The window only contains TOF bins that exist, so it has at most

    min(floor(support_bins) + 2, num_tof_bins)

bins, where support_bins is the width of the kernel support in TOF bins
(Gaussian: 2 num_sigmas sigma / tof_bin_width; table: scale (u_max - u_min) / tof_bin_width).
If this is more than 64, the window is truncated to 64 bins (approximation) and the Python
interface issues a UserWarning. These tests check

1. the warning is issued exactly when truncation can occur (all 8 TOF projectors), also
   as an exception with warnings turned into errors,
2. a kernel wider than 64 TOF bins is projected exactly if there are at most 64 TOF bins
   (no truncation, no warning),
3. the module constant parallelproj_core.tof_max_window_bins.

The values of sigma / the kernel scale are only checked if they are CPU arrays (reading
GPU arrays would force a synchronization), so the warning tests run on the CPU only.
"""

import math
import warnings

import numpy as np
import pytest
import parallelproj_core as ppb

from types import ModuleType
from .config import pytestmark
from .tof_reference import tof_sino_fwd_reference

IMG_SHAPE = (11, 13, 9)
VOXEL_SIZE = (2.0, 2.0, 2.0)
NUM_LORS = 40
WARNING_MATCH = "at most 64 TOF bins per window"


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "get"):  # cupy
        return x.get()
    if hasattr(x, "cpu"):  # torch
        x = x.cpu()
    return np.from_dlpack(x)


def _geometry():
    rng = np.random.default_rng(11)
    vs = np.array(VOXEL_SIZE, np.float32)
    origin = (-0.5 * (np.array(IMG_SHAPE) - 1) * vs).astype(np.float32)

    def points():
        d = rng.normal(size=(NUM_LORS, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return (d * 25 + rng.normal(scale=2, size=(NUM_LORS, 3))).astype(np.float32)

    xs, xe = points(), points()
    img = rng.uniform(size=IMG_SHAPE).astype(np.float32)
    return xs, xe, img, origin, vs


def _gauss_table(sigma, num_sigmas=3.0, n=1001):
    u = np.linspace(-num_sigmas * sigma, num_sigmas * sigma, n)
    cdf = np.array([0.5 * (1 + math.erf(x / (math.sqrt(2) * sigma))) for x in u], np.float32)
    return cdf, float(u[0]), float(u[-1])


def _call(xp, dev, projector: str, kernel: str, w: float, nbins: int, sigma: float, num_sigmas: float = 3.0,
          scale: float = 1.0):
    """Call one of the 8 TOF projectors (projector: sino_fwd, sino_back, lm_fwd, lm_back;
    kernel: gauss or table) with a Gaussian TOF kernel of width sigma (table: the table of
    the same Gaussian, stretched by scale)."""
    xs, xe, img, origin, vs = _geometry()
    f32 = xp.float32

    def a(x, dtype=f32):
        return xp.asarray(x, dtype=dtype, device=dev)

    off = a(np.zeros(1, np.float32))
    lm = projector.startswith("lm")
    if lm:  # one event per LOR, TOF bin in the middle
        tb = a(np.full(NUM_LORS, nbins // 2, np.int16), xp.int16)
        proj_shape = (NUM_LORS,)
    else:
        proj_shape = (NUM_LORS, nbins)
    fwd = projector.endswith("fwd")
    image = a(img) if fwd else xp.zeros(IMG_SHAPE, dtype=f32, device=dev)
    proj = xp.zeros(proj_shape, dtype=f32, device=dev) if fwd else xp.ones(proj_shape, dtype=f32, device=dev)
    common = (a(xs), a(xe), image, a(origin), a(vs), proj, w)
    if kernel == "gauss":
        fn = getattr(ppb, f"joseph3d_tof_{projector}")
        sig = a(np.array([sigma], np.float32))
        if lm:
            fn(*common, sig, off, tb, nbins, num_sigmas=num_sigmas)
        else:
            fn(*common, sig, off, nbins, num_sigmas=num_sigmas)
    else:
        fn = getattr(ppb, f"joseph3d_tof_{projector}_tab")
        cdf, u_min, u_max = _gauss_table(sigma, num_sigmas)
        sc = a(np.array([scale], np.float32))
        if lm:
            fn(*common, off, tb, nbins, a(cdf), u_min, u_max, sc)
        else:
            fn(*common, off, nbins, a(cdf), u_min, u_max, sc)
    return to_numpy(proj if fwd else image)


PROJECTORS = ["sino_fwd", "sino_back", "lm_fwd", "lm_back"]


def _skip_if_not_cpu(xp, dev):
    if not (dev == "cpu" or dev is None):
        pytest.skip("the window check needs CPU arrays (sigma / kernel scale)")


def test_tof_max_window_bins_constant(xp: ModuleType, dev: str):
    assert ppb.tof_max_window_bins == 64


@pytest.mark.parametrize("kernel", ["gauss", "table"])
@pytest.mark.parametrize("projector", PROJECTORS)
def test_warning_if_window_truncated(xp: ModuleType, dev: str, projector: str, kernel: str):
    """support 2 * 3 * 5 / 0.4 = 75 TOF bins and 100 TOF bins: windows of up to 77 bins
    would be needed -> warning (for the table: support of the stretched table)"""
    _skip_if_not_cpu(xp, dev)
    with pytest.warns(UserWarning, match=WARNING_MATCH):
        _call(xp, dev, projector, kernel, w=0.4, nbins=100, sigma=5.0)
    # table: the same via the kernel scale (support 2 * 3 * 2.5 * 2 / 0.4 = 75 bins)
    if kernel == "table":
        with pytest.warns(UserWarning, match=WARNING_MATCH):
            _call(xp, dev, projector, kernel, w=0.4, nbins=100, sigma=2.5, scale=2.0)


@pytest.mark.parametrize("kernel", ["gauss", "table"])
@pytest.mark.parametrize("projector", PROJECTORS)
def test_no_warning_if_window_fits(xp: ModuleType, dev: str, projector: str, kernel: str):
    """No warning (warnings are turned into errors here) if
    - the kernel is wide (75 bins), but there are at most 64 TOF bins, or
    - there are many TOF bins, but the windows have at most 64 bins: support
      2 * 3 * 4.1 / 0.4 = 61.5 bins -> windows of at most floor(61.5) + 2 = 63 bins."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _call(xp, dev, projector, kernel, w=0.4, nbins=64, sigma=5.0)
        _call(xp, dev, projector, kernel, w=0.4, nbins=200, sigma=4.1)


def test_warning_boundary(xp: ModuleType, dev: str):
    """The warning uses the same formula as the projectors: windows of 64 bins are fine
    (support 62.x bins), 65 bins are not (support 63.x bins)."""
    _skip_if_not_cpu(xp, dev)
    w = 0.5
    sigma_64 = 62.5 * w / 6.0  # support 62.5 bins -> windows of up to 64 bins
    sigma_65 = 63.5 * w / 6.0  # support 63.5 bins -> windows of up to 65 bins
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _call(xp, dev, "sino_fwd", "gauss", w=w, nbins=150, sigma=sigma_64)
    with pytest.warns(UserWarning, match="windows of up to 65 TOF bins"):
        _call(xp, dev, "sino_fwd", "gauss", w=w, nbins=150, sigma=sigma_65)
    # LOR-dependent sigma: the widest kernel counts
    xs, xe, img, origin, vs = _geometry()
    f32 = xp.float32
    sig = np.full(NUM_LORS, sigma_64, np.float32)
    sig[7] = sigma_65
    p = xp.zeros((NUM_LORS, 150), dtype=f32, device=dev)
    with pytest.warns(UserWarning, match=WARNING_MATCH):
        ppb.joseph3d_tof_sino_fwd(*(xp.asarray(x, dtype=f32, device=dev) for x in (xs, xe, img, origin, vs)), p, w,
                                  xp.asarray(sig, dtype=f32, device=dev), xp.zeros(1, dtype=f32, device=dev), 150)


def test_warning_as_error(xp: ModuleType, dev: str):
    """With warnings turned into errors (python -W error), the call raises the warning."""
    _skip_if_not_cpu(xp, dev)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match=WARNING_MATCH):
            _call(xp, dev, "sino_fwd", "gauss", w=0.4, nbins=100, sigma=5.0)
        with pytest.raises(UserWarning, match=WARNING_MATCH):
            _call(xp, dev, "lm_back", "table", w=0.4, nbins=100, sigma=5.0)


@pytest.mark.parametrize("nbins", [20, 64])
def test_wide_kernel_few_bins_is_exact(xp: ModuleType, dev: str, nbins: int):
    """A kernel support of 75 TOF bins with at most 64 TOF bins: the projection equals the
    float64 reference WITHOUT any window limit (exact; nothing is truncated)."""
    xs, xe, img, origin, vs = _geometry()
    w, sigma = 0.4, 5.0
    p = _call(xp, dev, "sino_fwd", "gauss", w=w, nbins=nbins, sigma=sigma).astype(np.float64)
    ref = np.array([tof_sino_fwd_reference(xs[k], xe[k], img, origin, vs, w, sigma, 0.0, nbins,
                                           max_window_bins=10**6) for k in range(NUM_LORS)])
    assert np.abs(ref).max() > 0
    assert np.abs(p - ref).max() < 1e-4 * np.abs(ref).max()
    # the same for the table of this Gaussian
    p_tab = _call(xp, dev, "sino_fwd", "table", w=w, nbins=nbins, sigma=sigma).astype(np.float64)
    assert np.abs(p_tab - ref).max() < 1e-4 * np.abs(ref).max()
