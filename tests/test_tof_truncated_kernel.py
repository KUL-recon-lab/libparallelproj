"""Tests of the truncated TOF kernel and the register-accumulator TOF sinogram forward
projector (see src/tof_kernel.h and src/joseph3d_tof_sino_fwd_worker.h).

The cases cover the different code paths of the forward projector: accumulator sizes from
8 to 64 TOF bins (also mixed within one call via LOR-dependent TOF sigma), TOF windows
wider than 64 bins (truncation), and - on the GPU - TOF bins written one at a time or in
aligned groups of 4 (TOF window moving >= 0.7 bins per image plane).
"""

import functools
import math
import zlib

import numpy as np
import pytest
import parallelproj_core as ppb

from types import ModuleType
from .config import pytestmark
from .tof_reference import tof_sino_fwd_reference

IMG_SHAPE = (23, 31, 17)
VOXEL_SIZE = (2.0, 1.5, 2.5)
NUM_LORS = 150

# name: (TOF bin width, TOF sigma (scalar or (min, max) for LOR-dependent), number of TOF bins)
CASES = {
    # TOF bin width ~ sigma (SIGNA-like ratio), window ~8 bins
    "w_eq_sigma": (6.0, 6.0, 25),
    # TOF bin width sigma / 5: window ~32 bins
    "w_sigma_5": (1.2, 6.0, 120),
    # TOF bins narrower than the voxels: the window moves >= 0.7 bins per plane
    # (groups of 4 on the GPU)
    "narrow_bins": (1.0, 1.0, 140),
    # LOR-dependent sigma: windows from 5 to ~40 bins in one call (several accumulator sizes)
    "lor_dependent_sigma": (2.0, (1.0, 13.0), 70),
    # window wider than 64 bins: truncated to 64 bins and renormalized
    "wide_window": (0.4, 5.0, 300),
}


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "get"):  # cupy
        return x.get()
    if hasattr(x, "cpu"):  # torch
        x = x.cpu()
    return np.from_dlpack(x)


@functools.lru_cache(maxsize=None)
def _setup(case: str):
    """LORs, TOF parameters, image and float64 reference projection of a case"""
    w, sigma, nbins = CASES[case]
    rng = np.random.default_rng(zlib.crc32(case.encode()))
    vs = np.array(VOXEL_SIZE, np.float32)
    origin = (-0.5 * (np.array(IMG_SHAPE) - 1) * vs).astype(np.float32)

    def points():
        d = rng.normal(size=(NUM_LORS, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return (d * 45 + rng.normal(scale=4, size=(NUM_LORS, 3))).astype(np.float32)

    xs, xe = points(), points()
    if isinstance(sigma, tuple):
        sig = rng.uniform(*sigma, NUM_LORS).astype(np.float32)
    else:
        sig = np.full(NUM_LORS, sigma, np.float32)
    off = rng.uniform(-8, 8, NUM_LORS).astype(np.float32)
    img = rng.uniform(size=IMG_SHAPE).astype(np.float32)
    ref = np.array([tof_sino_fwd_reference(xs[k], xe[k], img, origin, vs, w, float(sig[k]), float(off[k]), nbins)
                    for k in range(NUM_LORS)])
    return xs, xe, sig, off, img, origin, vs, w, nbins, ref


def _tof_sino_fwd(xp, dev, xs, xe, img, origin, vs, w, sig, off, nbins, init=0.0):
    f32 = xp.float32
    p = xp.full((xs.shape[0], nbins), init, dtype=f32, device=dev)
    ppb.joseph3d_tof_sino_fwd(
        xp.asarray(xs, dtype=f32, device=dev), xp.asarray(xe, dtype=f32, device=dev),
        xp.asarray(img, dtype=f32, device=dev), xp.asarray(origin, dtype=f32, device=dev),
        xp.asarray(vs, dtype=f32, device=dev), p, w,
        xp.asarray(sig, dtype=f32, device=dev), xp.asarray(off, dtype=f32, device=dev), nbins,
    )
    return to_numpy(p).astype(np.float64)


@pytest.mark.parametrize("case", list(CASES))
def test_tof_sino_fwd_vs_reference(xp: ModuleType, dev: str, case: str):
    """Forward projection vs. float64 reference; every output bin must be written
    (the output is prefilled with NaN)."""
    xs, xe, sig, off, img, origin, vs, w, nbins, ref = _setup(case)
    p = _tof_sino_fwd(xp, dev, xs, xe, img, origin, vs, w, sig, off, nbins, init=float("nan"))
    assert np.all(np.isfinite(p)), "not every TOF bin was written"
    scale = np.abs(ref).max()
    assert scale > 0
    assert np.abs(p - ref).max() < 1e-4 * scale


@pytest.mark.parametrize("case", ["w_eq_sigma", "narrow_bins", "lor_dependent_sigma"])
def test_tof_sum_equals_nontof(xp: ModuleType, dev: str, case: str):
    """With TOF bins covering the whole kernel support of every plane, the sum over the
    TOF bins equals the non-TOF projection (the weights of a plane sum to 1)."""
    xs, xe, sig, off, img, origin, vs, w, nbins, _ = _setup(case)
    # enough TOF bins (centered on the LOR centers, positions unchanged) to cover all planes
    lor_len = float(np.linalg.norm(xe - xs, axis=1).max())
    m = 3.0 * float(sig.max()) / w
    nbins_ext = int(math.ceil((lor_len + 2 * 8.0) / w + 2 * m + 4)) | 1
    p = _tof_sino_fwd(xp, dev, xs, xe, img, origin, vs, w, sig, off, nbins_ext)

    f32 = xp.float32
    p_nt = xp.zeros(xs.shape[0], dtype=f32, device=dev)
    ppb.joseph3d_fwd(xp.asarray(xs, dtype=f32, device=dev), xp.asarray(xe, dtype=f32, device=dev),
                     xp.asarray(img, dtype=f32, device=dev), xp.asarray(origin, dtype=f32, device=dev),
                     xp.asarray(vs, dtype=f32, device=dev), p_nt)
    p_nt = to_numpy(p_nt).astype(np.float64)
    assert p_nt.max() > 0
    assert np.abs(p.sum(axis=1) - p_nt).max() < 1e-5 * p_nt.max()


@pytest.mark.parametrize("case", list(CASES))
def test_tof_sino_adjointness_truncated_kernel(xp: ModuleType, dev: str, case: str):
    """<A x, y> = <x, A^T y> for all accumulator sizes / store widths."""
    xs, xe, sig, off, img, origin, vs, w, nbins, _ = _setup(case)
    rng = np.random.default_rng(5)
    y = rng.uniform(size=(xs.shape[0], nbins)).astype(np.float32)
    p = _tof_sino_fwd(xp, dev, xs, xe, img, origin, vs, w, sig, off, nbins)

    f32 = xp.float32
    back = xp.zeros(IMG_SHAPE, dtype=f32, device=dev)
    ppb.joseph3d_tof_sino_back(
        xp.asarray(xs, dtype=f32, device=dev), xp.asarray(xe, dtype=f32, device=dev), back,
        xp.asarray(origin, dtype=f32, device=dev), xp.asarray(vs, dtype=f32, device=dev),
        xp.asarray(y, dtype=f32, device=dev), w,
        xp.asarray(sig, dtype=f32, device=dev), xp.asarray(off, dtype=f32, device=dev), nbins,
    )
    ip1 = float((p * y).sum())
    ip2 = float((img.astype(np.float64) * to_numpy(back).astype(np.float64)).sum())
    assert math.isclose(ip1, ip2, rel_tol=1e-5)


@pytest.mark.parametrize("case", ["w_eq_sigma", "narrow_bins", "lor_dependent_sigma", "wide_window"])
def test_tof_lm_equals_sino_bins(xp: ModuleType, dev: str, case: str):
    """A listmode event gets the value of its TOF sinogram bin (same TOF kernel)."""
    xs, xe, sig, off, img, origin, vs, w, nbins, _ = _setup(case)
    p = _tof_sino_fwd(xp, dev, xs, xe, img, origin, vs, w, sig, off, nbins)

    # one event for every TOF bin of the first 20 LORs
    nl = 20
    lor = np.repeat(np.arange(nl), nbins)
    tb = np.tile(np.arange(nbins), nl).astype(np.int16)
    f32 = xp.float32
    p_lm = xp.zeros(lor.shape[0], dtype=f32, device=dev)
    ppb.joseph3d_tof_lm_fwd(
        xp.asarray(xs[lor], dtype=f32, device=dev), xp.asarray(xe[lor], dtype=f32, device=dev),
        xp.asarray(img, dtype=f32, device=dev), xp.asarray(origin, dtype=f32, device=dev),
        xp.asarray(vs, dtype=f32, device=dev), p_lm, w,
        xp.asarray(sig[lor], dtype=f32, device=dev), xp.asarray(off[lor], dtype=f32, device=dev),
        xp.asarray(tb, dtype=xp.int16, device=dev), nbins,
    )
    p_lm = to_numpy(p_lm).astype(np.float64).reshape(nl, nbins)
    scale = np.abs(p[:nl]).max()
    assert scale > 0
    assert np.abs(p_lm - p[:nl]).max() < 1e-5 * scale


def test_tof_kernel_continuous_in_position(xp: ModuleType, dev: str):
    """Moving a point source by tiny steps along a LOR changes every TOF bin by a tiny
    amount (no jump when a bin enters or leaves the truncated TOF window)."""
    nbins, w, sigma = 15, 4.0, 4.0
    img = np.zeros((101, 1, 1), np.float32)
    img[50, 0, 0] = 1.0
    vs = np.array([1.0, 1.0, 1.0], np.float32)
    origin = np.array([-50.0, 0.0, 0.0], np.float32)
    xs = np.array([[-80.0, 0.0, 0.0]], np.float32)
    xe = np.array([[80.0, 0.0, 0.0]], np.float32)
    prev = None
    max_jump = 0.0
    # steps of 0.002 mm over more than one TOF bin (4 mm), so that bins enter and leave
    # the TOF window
    for off in np.linspace(-2.2, 2.2, 2201).astype(np.float32):
        p = _tof_sino_fwd(xp, dev, xs, xe, img, origin, vs, w, np.array([sigma], np.float32),
                          np.array([off], np.float32), nbins)[0]
        if prev is not None:
            max_jump = max(max_jump, float(np.abs(p - prev).max()))
        prev = p
    # smooth change per step: <= step * max |d weight / d x| ~ 0.002 mm * 0.025 / mm = 5e-5;
    # with the old (non-truncated) kernel, a bin entering the window jumped by ~1.3e-3
    assert max_jump < 2e-4
