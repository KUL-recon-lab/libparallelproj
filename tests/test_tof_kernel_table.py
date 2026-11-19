"""Tests of the TOF projectors with a tabulated TOF kernel (joseph3d_tof_*_tab, issue #46).

The tabulated kernel is given as a table of the cumulative kernel F on a uniform grid of
u (spatial units, positive towards lor_end), stretched per LOR / event by a scale s
(F_s(u) = F(u / s)). See src/tof_kernel.h and the docstrings of the functions.
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

# "asymmetric_wide" deliberately needs TOF windows of more than 64 bins (truncation), for
# which the Python interface issues a UserWarning (tested in test_tof_window_limit.py)
pytestmark = [pytestmark, pytest.mark.filterwarnings("ignore:.*at most 64 TOF bins per window:UserWarning")]

IMG_SHAPE = (23, 31, 17)
VOXEL_SIZE = (2.0, 1.5, 2.5)
NUM_LORS = 120
SIGMA0 = 3.0  # sigma (mm) of the tabulated Gaussian


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "get"):  # cupy
        return x.get()
    if hasattr(x, "cpu"):  # torch
        x = x.cpu()
    return np.from_dlpack(x)


def gauss_table(sigma=SIGMA0, num_sigmas=3.0, n=2001):
    """cumulative Gaussian F on [-num_sigmas sigma, num_sigmas sigma]"""
    u = np.linspace(-num_sigmas * sigma, num_sigmas * sigma, n)
    cdf = np.array([0.5 * (1 + math.erf(x / (math.sqrt(2) * sigma))) for x in u], np.float32)
    return cdf, float(u[0]), float(u[-1])


def asymmetric_table(n=1501):
    """an asymmetric kernel: narrow Gaussian core + exponential tail towards +u (mm),
    support [-8, 30] mm; F computed by numerical integration and normalized to [0, 1]"""
    u = np.linspace(-8.0, 30.0, n)
    core = 0.6 * np.exp(-0.5 * (u / 1.5) ** 2) / (1.5 * math.sqrt(2 * math.pi))
    tail = 0.4 * np.where(u > 0, np.exp(-u / 6.0) / 6.0, 0.0)
    pdf = core + tail
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(u))])
    cdf /= cdf[-1]
    return cdf.astype(np.float32), float(u[0]), float(u[-1])


# name: (TOF bin width, table, scale (scalar or (min, max) for LOR-dependent), number of TOF bins)
CASES = {
    "gauss_table": (3.0, gauss_table, 1.0, 41),
    "gauss_table_lor_dependent_scale": (3.0, gauss_table, (0.5, 2.5), 41),
    "asymmetric": (2.0, asymmetric_table, (0.7, 1.4), 61),
    # narrow TOF bins (groups of 4 on the GPU)
    "asymmetric_narrow_bins": (0.8, asymmetric_table, 1.0, 140),
    # support wider than 64 bins: window truncated to 64 bins
    "asymmetric_wide": (0.4, asymmetric_table, 1.0, 300),
    # support of 95 bins, but only 50 TOF bins: no truncation (windows clipped to 50 bins)
    "asymmetric_wide_few_bins": (0.4, asymmetric_table, 1.0, 50),
}


@functools.lru_cache(maxsize=None)
def _setup(case: str):
    w, table_fn, scale, nbins = CASES[case]
    rng = np.random.default_rng(zlib.crc32(case.encode()))
    vs = np.array(VOXEL_SIZE, np.float32)
    origin = (-0.5 * (np.array(IMG_SHAPE) - 1) * vs).astype(np.float32)

    def points():
        d = rng.normal(size=(NUM_LORS, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return (d * 45 + rng.normal(scale=4, size=(NUM_LORS, 3))).astype(np.float32)

    xs, xe = points(), points()
    if isinstance(scale, tuple):
        sc = rng.uniform(*scale, NUM_LORS).astype(np.float32)
    else:
        sc = np.full(1, scale, np.float32)
    off = rng.uniform(-4, 4, NUM_LORS).astype(np.float32)
    img = rng.uniform(size=IMG_SHAPE).astype(np.float32)
    cdf, u_min, u_max = table_fn()
    return xs, xe, sc, off, img, origin, vs, w, nbins, cdf, u_min, u_max


def _arr(xp, dev, a, dtype=None):
    return xp.asarray(a, dtype=dtype or xp.float32, device=dev)


def sino_fwd_tab(xp, dev, xs, xe, img, origin, vs, w, off, nbins, cdf, u_min, u_max, sc, init=0.0):
    p = xp.full((xs.shape[0], nbins), init, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_sino_fwd_tab(_arr(xp, dev, xs), _arr(xp, dev, xe), _arr(xp, dev, img), _arr(xp, dev, origin),
                                  _arr(xp, dev, vs), p, w, _arr(xp, dev, off), nbins, _arr(xp, dev, cdf), u_min,
                                  u_max, _arr(xp, dev, sc))
    return to_numpy(p).astype(np.float64)


def sino_back_tab(xp, dev, xs, xe, y, origin, vs, w, off, nbins, cdf, u_min, u_max, sc):
    x = xp.zeros(IMG_SHAPE, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_sino_back_tab(_arr(xp, dev, xs), _arr(xp, dev, xe), x, _arr(xp, dev, origin), _arr(xp, dev, vs),
                                   _arr(xp, dev, y), w, _arr(xp, dev, off), nbins, _arr(xp, dev, cdf), u_min, u_max,
                                   _arr(xp, dev, sc))
    return to_numpy(x).astype(np.float64)


def lm_fwd_tab(xp, dev, xs, xe, img, origin, vs, w, off, tb, nbins, cdf, u_min, u_max, sc):
    p = xp.zeros(xs.shape[0], dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_lm_fwd_tab(_arr(xp, dev, xs), _arr(xp, dev, xe), _arr(xp, dev, img), _arr(xp, dev, origin),
                                _arr(xp, dev, vs), p, w, _arr(xp, dev, off), _arr(xp, dev, tb, xp.int16), nbins,
                                _arr(xp, dev, cdf), u_min, u_max, _arr(xp, dev, sc))
    return to_numpy(p).astype(np.float64)


def lm_back_tab(xp, dev, xs, xe, y, origin, vs, w, off, tb, nbins, cdf, u_min, u_max, sc):
    x = xp.zeros(IMG_SHAPE, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_lm_back_tab(_arr(xp, dev, xs), _arr(xp, dev, xe), x, _arr(xp, dev, origin), _arr(xp, dev, vs),
                                 _arr(xp, dev, y), w, _arr(xp, dev, off), _arr(xp, dev, tb, xp.int16), nbins,
                                 _arr(xp, dev, cdf), u_min, u_max, _arr(xp, dev, sc))
    return to_numpy(x).astype(np.float64)


# ---------------------------------------------------------------------------------------
# 1. a table of the Gaussian reproduces the built-in Gaussian kernel (also with a per-LOR
#    scale, which corresponds to sigma = scale * SIGMA0)


@pytest.mark.parametrize("case", ["gauss_table", "gauss_table_lor_dependent_scale"])
@pytest.mark.parametrize("projector", ["sino_fwd", "sino_back", "lm_fwd", "lm_back"])
def test_gauss_table_equals_gaussian(xp: ModuleType, dev: str, case: str, projector: str):
    xs, xe, sc, off, img, origin, vs, w, nbins, cdf, u_min, u_max = _setup(case)
    sigma = (sc * SIGMA0).astype(np.float32)  # per LOR (or one value)
    f32 = xp.float32
    if projector.startswith("sino"):
        if projector == "sino_fwd":
            tab = sino_fwd_tab(xp, dev, xs, xe, img, origin, vs, w, off, nbins, cdf, u_min, u_max, sc)
            ref = xp.zeros((NUM_LORS, nbins), dtype=f32, device=dev)
            ppb.joseph3d_tof_sino_fwd(_arr(xp, dev, xs), _arr(xp, dev, xe), _arr(xp, dev, img), _arr(xp, dev, origin),
                                      _arr(xp, dev, vs), ref, w, _arr(xp, dev, sigma), _arr(xp, dev, off), nbins)
        else:
            y = np.random.default_rng(1).uniform(size=(NUM_LORS, nbins)).astype(np.float32)
            tab = sino_back_tab(xp, dev, xs, xe, y, origin, vs, w, off, nbins, cdf, u_min, u_max, sc)
            ref = xp.zeros(IMG_SHAPE, dtype=f32, device=dev)
            ppb.joseph3d_tof_sino_back(_arr(xp, dev, xs), _arr(xp, dev, xe), ref, _arr(xp, dev, origin),
                                       _arr(xp, dev, vs), _arr(xp, dev, y), w, _arr(xp, dev, sigma),
                                       _arr(xp, dev, off), nbins)
    else:
        tb = np.random.default_rng(2).integers(0, nbins, NUM_LORS).astype(np.int16)
        if projector == "lm_fwd":
            tab = lm_fwd_tab(xp, dev, xs, xe, img, origin, vs, w, off, tb, nbins, cdf, u_min, u_max, sc)
            ref = xp.zeros(NUM_LORS, dtype=f32, device=dev)
            ppb.joseph3d_tof_lm_fwd(_arr(xp, dev, xs), _arr(xp, dev, xe), _arr(xp, dev, img), _arr(xp, dev, origin),
                                    _arr(xp, dev, vs), ref, w,
                                    _arr(xp, dev, sigma),
                                    _arr(xp, dev, off), _arr(xp, dev, tb, xp.int16), nbins)
        else:
            y = np.random.default_rng(3).uniform(size=NUM_LORS).astype(np.float32)
            tab = lm_back_tab(xp, dev, xs, xe, y, origin, vs, w, off, tb, nbins, cdf, u_min, u_max, sc)
            ref = xp.zeros(IMG_SHAPE, dtype=f32, device=dev)
            ppb.joseph3d_tof_lm_back(_arr(xp, dev, xs), _arr(xp, dev, xe), ref, _arr(xp, dev, origin),
                                     _arr(xp, dev, vs), _arr(xp, dev, y), w, _arr(xp, dev, sigma),
                                     _arr(xp, dev, off), _arr(xp, dev, tb, xp.int16), nbins)
    ref = to_numpy(ref).astype(np.float64)
    assert np.abs(ref).max() > 0
    # differences come from the linear interpolation of the table (~1e-7) and float rounding
    assert np.abs(tab - ref).max() < 1e-5 * np.abs(ref).max()


# ---------------------------------------------------------------------------------------
# 2. forward projection vs. a float64 reference of the same model; every bin written


@pytest.mark.parametrize("case", list(CASES))
def test_tab_sino_fwd_vs_reference(xp: ModuleType, dev: str, case: str):
    xs, xe, sc, off, img, origin, vs, w, nbins, cdf, u_min, u_max = _setup(case)
    p = sino_fwd_tab(xp, dev, xs, xe, img, origin, vs, w, off, nbins, cdf, u_min, u_max, sc, init=float("nan"))
    assert np.all(np.isfinite(p)), "not every TOF bin was written"
    s_lor = np.broadcast_to(sc, (NUM_LORS,))
    ref = np.array([tof_sino_fwd_reference(xs[k], xe[k], img, origin, vs, w, None, float(off[k]), nbins,
                                           table=(cdf, u_min, u_max, float(s_lor[k]))) for k in range(NUM_LORS)])
    assert np.abs(ref).max() > 0
    assert np.abs(p - ref).max() < 1e-4 * np.abs(ref).max()


# ---------------------------------------------------------------------------------------
# 3. sum over TOF bins = non-TOF projection, adjointness, listmode = sinogram bins


@pytest.mark.parametrize("case", ["asymmetric", "asymmetric_narrow_bins"])
def test_tab_tof_sum_equals_nontof(xp: ModuleType, dev: str, case: str):
    xs, xe, sc, off, img, origin, vs, w, nbins, cdf, u_min, u_max = _setup(case)
    # enough TOF bins to cover the kernel support of every plane (centered, positions unchanged)
    lor_len = float(np.linalg.norm(xe - xs, axis=1).max())
    extra = (max(abs(u_min), abs(u_max)) * float(np.max(sc)) + 8.0) / w
    nbins_ext = int(math.ceil(lor_len / w + 2 * extra + 4)) | (nbins & 1)
    p = sino_fwd_tab(xp, dev, xs, xe, img, origin, vs, w, off, nbins_ext, cdf, u_min, u_max, sc)
    p_nt = xp.zeros(NUM_LORS, dtype=xp.float32, device=dev)
    ppb.joseph3d_fwd(_arr(xp, dev, xs), _arr(xp, dev, xe), _arr(xp, dev, img), _arr(xp, dev, origin),
                     _arr(xp, dev, vs), p_nt)
    p_nt = to_numpy(p_nt).astype(np.float64)
    assert np.abs(p.sum(axis=1) - p_nt).max() < 1e-5 * p_nt.max()


@pytest.mark.parametrize("case", list(CASES))
def test_tab_sino_adjointness(xp: ModuleType, dev: str, case: str):
    xs, xe, sc, off, img, origin, vs, w, nbins, cdf, u_min, u_max = _setup(case)
    y = np.random.default_rng(5).uniform(size=(NUM_LORS, nbins)).astype(np.float32)
    p = sino_fwd_tab(xp, dev, xs, xe, img, origin, vs, w, off, nbins, cdf, u_min, u_max, sc)
    b = sino_back_tab(xp, dev, xs, xe, y, origin, vs, w, off, nbins, cdf, u_min, u_max, sc)
    assert math.isclose(float((p * y).sum()), float((img * b).sum()), rel_tol=1e-5)


@pytest.mark.parametrize("case", ["asymmetric", "asymmetric_wide"])
def test_tab_lm_adjointness_and_equals_sino_bins(xp: ModuleType, dev: str, case: str):
    xs, xe, sc, off, img, origin, vs, w, nbins, cdf, u_min, u_max = _setup(case)
    p_sino = sino_fwd_tab(xp, dev, xs, xe, img, origin, vs, w, off, nbins, cdf, u_min, u_max, sc)
    # one event for every TOF bin of the first 15 LORs
    nl = 15
    lor = np.repeat(np.arange(nl), nbins)
    tb = np.tile(np.arange(nbins), nl).astype(np.int16)
    sc_ev = sc[lor] if sc.size > 1 else sc
    p_lm = lm_fwd_tab(xp, dev, xs[lor], xe[lor], img, origin, vs, w, off[lor], tb, nbins, cdf, u_min, u_max, sc_ev)
    scale = np.abs(p_sino[:nl]).max()
    assert scale > 0
    assert np.abs(p_lm.reshape(nl, nbins) - p_sino[:nl]).max() < 1e-5 * scale

    y = np.random.default_rng(6).uniform(size=lor.shape[0]).astype(np.float32)
    b = lm_back_tab(xp, dev, xs[lor], xe[lor], y, origin, vs, w, off[lor], tb, nbins, cdf, u_min, u_max, sc_ev)
    assert math.isclose(float((p_lm * y).sum()), float((img * b).sum()), rel_tol=1e-5)


# ---------------------------------------------------------------------------------------
# 4. direction convention of asymmetric kernels: reversing the LORs (start <-> end) with
#    the mirrored kernel (u -> -u) and the negated center offset gives the same projection
#    with the TOF bins in reversed order


def test_tab_asymmetric_mirror(xp: ModuleType, dev: str):
    xs, xe, sc, off, img, origin, vs, w, nbins, cdf, u_min, u_max = _setup("asymmetric")
    p = sino_fwd_tab(xp, dev, xs, xe, img, origin, vs, w, off, nbins, cdf, u_min, u_max, sc)
    # mirrored kernel f'(u) = f(-u): F'(u) = 1 - F(-u), on the grid from -u_max to -u_min
    cdf_m = (1.0 - cdf[::-1]).astype(np.float32)
    p_rev = sino_fwd_tab(xp, dev, xe, xs, img, origin, vs, w, -off, nbins, cdf_m, -u_max, -u_min, sc)
    assert np.abs(p_rev[:, ::-1] - p).max() < 1e-4 * np.abs(p).max()
    # the kernel really is asymmetric: without mirroring the table, the results differ
    p_no_mirror = sino_fwd_tab(xp, dev, xe, xs, img, origin, vs, w, -off, nbins, cdf, u_min, u_max, sc)
    assert np.abs(p_no_mirror[:, ::-1] - p).max() > 1e-2 * np.abs(p).max()


# ---------------------------------------------------------------------------------------
# 5. input checks (CPU arrays) and degenerate parameters


def test_tab_input_validation(xp: ModuleType, dev: str):
    if dev not in ("cpu", None):
        pytest.skip("value checks of the table are done for CPU arrays only")
    xs, xe, sc, off, img, origin, vs, w, nbins, cdf, u_min, u_max = _setup("gauss_table")

    def call(cdf=cdf, u_min=u_min, u_max=u_max, sc=sc, off=off):
        return sino_fwd_tab(xp, dev, xs, xe, img, origin, vs, w, off, nbins, cdf, u_min, u_max, sc)

    with pytest.raises(ValueError, match="at least 2"):
        call(cdf=cdf[:1])
    with pytest.raises(ValueError, match="u_min < u_max"):
        call(u_min=1.0, u_max=1.0)
    with pytest.raises(ValueError, match="must not decrease"):
        call(cdf=cdf[::-1].copy())
    with pytest.raises(ValueError, match="must increase"):
        call(cdf=np.zeros(5, np.float32))
    with pytest.raises(ValueError, match="finite"):
        call(cdf=np.array([0.0, np.nan, 1.0], np.float32))
    with pytest.raises(ValueError, match="> 0"):
        call(sc=np.array([0.0], np.float32))
    with pytest.raises(ValueError, match="tof_kernel_scale"):
        call(sc=np.ones(NUM_LORS + 1, np.float32))
    with pytest.raises(ValueError, match="tof_center_offset"):
        call(off=np.zeros(3, np.float32))


def test_degenerate_gaussian_sigma(xp: ModuleType, dev: str):
    """sigma <= 0 (degenerate kernel): no TOF contribution, no NaN"""
    xs, xe, sc, off, img, origin, vs, w, nbins, cdf, u_min, u_max = _setup("gauss_table")
    sigma = np.zeros(1, np.float32)
    p = xp.full((NUM_LORS, nbins), float("nan"), dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_sino_fwd(_arr(xp, dev, xs), _arr(xp, dev, xe), _arr(xp, dev, img), _arr(xp, dev, origin),
                              _arr(xp, dev, vs), p, w, _arr(xp, dev, sigma), _arr(xp, dev, off), nbins)
    assert np.all(to_numpy(p) == 0)
    x = xp.zeros(IMG_SHAPE, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_sino_back(_arr(xp, dev, xs), _arr(xp, dev, xe), x, _arr(xp, dev, origin), _arr(xp, dev, vs),
                               _arr(xp, dev, np.ones((NUM_LORS, nbins), np.float32)), w, _arr(xp, dev, sigma),
                               _arr(xp, dev, off), nbins)
    assert np.all(to_numpy(x) == 0)


def test_tab_shape_checks(xp: ModuleType, dev: str):
    """wrong array shapes of the 4 *_tab functions raise ValueError"""
    xs, xe, sc, off, img, origin, vs, w, nbins, cdf, u_min, u_max = _setup("gauss_table")
    a = functools.partial(_arr, xp, dev)
    o, v, c, s1, off1 = a(origin), a(vs), a(cdf), a(np.ones(1, np.float32)), a(np.zeros(1, np.float32))
    image = a(img)
    n = NUM_LORS
    p_ok = xp.zeros((n, nbins), dtype=xp.float32, device=dev)
    tab = (c, u_min, u_max, s1)

    def sino(fn, start, end, img_, proj):
        fn(start, end, img_, o, v, proj, w, off1, nbins, *tab)

    for fn, img_, proj in [(ppb.joseph3d_tof_sino_fwd_tab, image, p_ok),
                           (ppb.joseph3d_tof_sino_back_tab, xp.zeros(IMG_SHAPE, dtype=xp.float32, device=dev), p_ok)]:
        with pytest.raises(ValueError, match="at least 2 dims"):
            sino(fn, a(xs[0]), a(xe[0]), img_, proj)
        with pytest.raises(ValueError, match="same number of dimensions"):
            sino(fn, a(xs), a(xe[None]), img_, proj)
        with pytest.raises(ValueError, match="same shape"):
            sino(fn, a(xs), a(xe[:-1]), img_, proj)
        with pytest.raises(ValueError, match="same number of dimensions as lor_start"):
            sino(fn, a(xs), a(xe), img_, xp.zeros((n * nbins,), dtype=xp.float32, device=dev))
        with pytest.raises(ValueError, match=r"projection_values\[:-1\]"):
            sino(fn, a(xs), a(xe), img_, xp.zeros((n + 1, nbins), dtype=xp.float32, device=dev))
        with pytest.raises(ValueError, match="num_tof_bins"):
            sino(fn, a(xs), a(xe), img_, xp.zeros((n, nbins + 1), dtype=xp.float32, device=dev))

    tb_ok = a(np.zeros(n, np.int16), xp.int16)
    y_ok = xp.zeros(n, dtype=xp.float32, device=dev)

    def lm(fn, start, end, img_, proj, tb):
        fn(start, end, img_, o, v, proj, w, off1, tb, nbins, *tab)

    for fn, img_ in [(ppb.joseph3d_tof_lm_fwd_tab, image),
                     (ppb.joseph3d_tof_lm_back_tab, xp.zeros(IMG_SHAPE, dtype=xp.float32, device=dev))]:
        with pytest.raises(ValueError, match="2 dimensions"):
            lm(fn, a(xs[None]), a(xe[None]), img_, y_ok, tb_ok)
        with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
            lm(fn, a(np.zeros((n, 4), np.float32)), a(np.zeros((n, 4), np.float32)), img_, y_ok, tb_ok)
        with pytest.raises(ValueError, match="same number of events"):
            lm(fn, a(xs), a(xe[:-1]), img_, y_ok, tb_ok)
        with pytest.raises(ValueError, match="projection_values must have shape"):
            lm(fn, a(xs), a(xe), img_, xp.zeros(n + 1, dtype=xp.float32, device=dev), tb_ok)
        with pytest.raises(ValueError, match="tof_bin_index must have shape"):
            lm(fn, a(xs), a(xe), img_, y_ok, a(np.zeros(n + 1, np.int16), xp.int16))
