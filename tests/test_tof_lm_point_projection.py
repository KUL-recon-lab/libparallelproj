"""Tests for the TOF listmode "point" projectors (tof_lm_point_fwd / tof_lm_point_back).

Every event is reduced to the center of its TOF bin, located at a distance
(i - (N - 1) / 2) * tof_bin_width + tof_center_offset from the LOR midpoint
towards the LOR end point. The forward projector trilinearly interpolates the
image there, the back projector is its adjoint.
"""

import pytest
import math
import numpy as np
import parallelproj_core as ppb

from types import ModuleType
from .config import pytestmark


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "get"):  # cupy
        return x.get()
    if hasattr(x, "cpu"):  # torch
        x = x.cpu()
    return np.from_dlpack(x)


################################################################################
# NumPy (float64) reference implementation
################################################################################


def ref_tof_bin_centers(xstart, xend, w, offset, tofbins, nbins):
    """world coordinates of the TOF bin centers, shape (num_events, 3)"""
    d = xend.astype(np.float64) - xstart.astype(np.float64)
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    x = (tofbins.astype(np.float64) - 0.5 * (nbins - 1)) * w + offset
    return 0.5 * (xstart + xend).astype(np.float64) + x[:, None] * u


def ref_trilinear_weights(points, img_shape, img_origin, voxsize):
    """list of (flat voxel indices, weights) per point (neighbors inside of the image)"""
    f = (points - np.asarray(img_origin, np.float64)) / np.asarray(voxsize, np.float64)
    fl = np.floor(f)
    r = f - fl
    fl = fl.astype(np.int64)
    out = []
    for i in range(points.shape[0]):
        idx, wts = [], []
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    j = fl[i] + (a, b, c)
                    if np.all(j >= 0) and np.all(j < img_shape):
                        wt = (
                            (r[i, 0] if a else 1 - r[i, 0])
                            * (r[i, 1] if b else 1 - r[i, 1])
                            * (r[i, 2] if c else 1 - r[i, 2])
                        )
                        idx.append(np.ravel_multi_index(tuple(j), img_shape))
                        wts.append(wt)
        out.append((np.array(idx, dtype=np.int64), np.array(wts)))
    return out


def ref_fwd(img, img_origin, voxsize, xstart, xend, w, offset, tofbins, nbins):
    pts = ref_tof_bin_centers(xstart, xend, w, offset, tofbins, nbins)
    flat = img.astype(np.float64).ravel()
    return np.array(
        [(flat[k] * wt).sum() for k, wt in ref_trilinear_weights(pts, img.shape, img_origin, voxsize)]
    )


def ref_back(img_shape, img_origin, voxsize, xstart, xend, values, w, offset, tofbins, nbins):
    pts = ref_tof_bin_centers(xstart, xend, w, offset, tofbins, nbins)
    out = np.zeros(int(np.prod(img_shape)), np.float64)
    for v, (k, wt) in zip(values, ref_trilinear_weights(pts, img_shape, img_origin, voxsize)):
        np.add.at(out, k, v * wt)
    return out.reshape(img_shape)


################################################################################
# helpers calling the library
################################################################################


def lib_fwd(xp, dev, img, img_origin, voxsize, xstart, xend, w, offset, tofbins, nbins, init=0.0):
    f32 = xp.float32
    p = xp.full(xstart.shape[0], init, dtype=f32, device=dev)
    ppb.tof_lm_point_fwd(
        xp.asarray(xstart, dtype=f32, device=dev),
        xp.asarray(xend, dtype=f32, device=dev),
        xp.asarray(img, dtype=f32, device=dev),
        xp.asarray(img_origin, dtype=f32, device=dev),
        xp.asarray(voxsize, dtype=f32, device=dev),
        p,
        w,
        xp.asarray(np.atleast_1d(offset), dtype=f32, device=dev),
        xp.asarray(tofbins, dtype=xp.int16, device=dev),
        nbins,
    )
    return to_numpy(p).astype(np.float64)


def lib_back(xp, dev, img_shape, img_origin, voxsize, xstart, xend, values, w, offset, tofbins, nbins):
    f32 = xp.float32
    img = xp.zeros(img_shape, dtype=f32, device=dev)
    ppb.tof_lm_point_back(
        xp.asarray(xstart, dtype=f32, device=dev),
        xp.asarray(xend, dtype=f32, device=dev),
        img,
        xp.asarray(img_origin, dtype=f32, device=dev),
        xp.asarray(voxsize, dtype=f32, device=dev),
        xp.asarray(values, dtype=f32, device=dev),
        w,
        xp.asarray(np.atleast_1d(offset), dtype=f32, device=dev),
        xp.asarray(tofbins, dtype=xp.int16, device=dev),
        nbins,
    )
    return to_numpy(img).astype(np.float64)


def random_setup(seed=0, nlors=2000, img_shape=(17, 19, 13), voxsize=(2.2, 2.5, 2.7),
                 nbins=25, w=4.0, radius=50.0):
    """random LORs between points on a sphere (both orientations, oblique),
    random TOF bins (some placing events outside of the image) and
    event-dependent TOF center offsets"""
    rng = np.random.default_rng(seed)
    img_origin = tuple((-(n / 2) + 0.5) * v for n, v in zip(img_shape, voxsize))

    def sphere(m):
        v = rng.normal(size=(m, 3))
        return (radius * v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    xstart, xend = sphere(nlors), sphere(nlors)
    tofbins = rng.integers(0, nbins, nlors).astype(np.int16)
    offsets = rng.uniform(-5.0, 5.0, nlors).astype(np.float32)
    img = rng.uniform(0.0, 1.0, img_shape).astype(np.float32)
    return dict(img=img, img_origin=img_origin, voxsize=voxsize, xstart=xstart, xend=xend,
                w=w, offset=offsets, tofbins=tofbins, nbins=nbins, rng=rng)


################################################################################
# tests
################################################################################


def test_tof_lm_point_fwd_vs_reference(xp: ModuleType, dev: str):
    s = random_setup(seed=1)
    args = (s["xstart"], s["xend"], s["w"], s["offset"], s["tofbins"], s["nbins"])
    p = lib_fwd(xp, dev, s["img"], s["img_origin"], s["voxsize"], *args)
    p_ref = ref_fwd(s["img"], s["img_origin"], s["voxsize"], *args)

    # make sure the test covers events inside, at the border and outside of the image
    assert (p_ref == 0).sum() > 100
    assert (p_ref > 0).sum() > 100
    assert np.allclose(p, p_ref, atol=1e-5, rtol=1e-4)


def test_tof_lm_point_back_vs_reference(xp: ModuleType, dev: str):
    s = random_setup(seed=2)
    values = s["rng"].uniform(0.0, 1.0, s["xstart"].shape[0]).astype(np.float32)
    args = (s["xstart"], s["xend"], values, s["w"], s["offset"], s["tofbins"], s["nbins"])
    b = lib_back(xp, dev, s["img"].shape, s["img_origin"], s["voxsize"], *args)
    b_ref = ref_back(s["img"].shape, s["img_origin"], s["voxsize"], *args)

    assert b_ref.max() > 0
    assert np.allclose(b, b_ref, atol=1e-5 * b_ref.max(), rtol=1e-4)
    assert math.isclose(b.sum(), b_ref.sum(), rel_tol=1e-5)


@pytest.mark.parametrize("scalar_offset", [False, True])
def test_tof_lm_point_adjointness(xp: ModuleType, dev: str, scalar_offset: bool):
    s = random_setup(seed=3)
    offset = np.float32(1.7) if scalar_offset else s["offset"]
    y = s["rng"].uniform(0.0, 1.0, s["xstart"].shape[0]).astype(np.float32)

    args = (s["xstart"], s["xend"], s["w"], offset, s["tofbins"], s["nbins"])
    p = lib_fwd(xp, dev, s["img"], s["img_origin"], s["voxsize"], *args)
    b = lib_back(xp, dev, s["img"].shape, s["img_origin"], s["voxsize"],
                 s["xstart"], s["xend"], y, s["w"], offset, s["tofbins"], s["nbins"])

    ip1 = float((p * y).sum())
    ip2 = float((s["img"].astype(np.float64) * b).sum())
    assert math.isclose(ip1, ip2, rel_tol=1e-5)


def test_tof_lm_point_scalar_vs_event_dependent_offset(xp: ModuleType, dev: str):
    s = random_setup(seed=4, nlors=300)
    n = s["xstart"].shape[0]
    common = (s["xstart"], s["xend"], s["w"])
    p1 = lib_fwd(xp, dev, s["img"], s["img_origin"], s["voxsize"], *common,
                 np.float32(-2.3), s["tofbins"], s["nbins"])
    p2 = lib_fwd(xp, dev, s["img"], s["img_origin"], s["voxsize"], *common,
                 np.full(n, -2.3, np.float32), s["tofbins"], s["nbins"])
    assert np.array_equal(p1, p2)


def test_tof_lm_point_mirroring(xp: ModuleType, dev: str):
    """swapping start and end point, mirroring the TOF bin (i -> N - 1 - i) and
    negating the TOF center offset describes the same point"""
    s = random_setup(seed=5)
    nb = s["nbins"]
    y = s["rng"].uniform(0.0, 1.0, s["xstart"].shape[0]).astype(np.float32)

    p = lib_fwd(xp, dev, s["img"], s["img_origin"], s["voxsize"], s["xstart"], s["xend"],
                s["w"], s["offset"], s["tofbins"], nb)
    p_m = lib_fwd(xp, dev, s["img"], s["img_origin"], s["voxsize"], s["xend"], s["xstart"],
                  s["w"], -s["offset"], nb - 1 - s["tofbins"], nb)
    assert np.allclose(p, p_m, atol=1e-5, rtol=1e-4)

    b = lib_back(xp, dev, s["img"].shape, s["img_origin"], s["voxsize"], s["xstart"], s["xend"],
                 y, s["w"], s["offset"], s["tofbins"], nb)
    b_m = lib_back(xp, dev, s["img"].shape, s["img_origin"], s["voxsize"], s["xend"], s["xstart"],
                   y, s["w"], -s["offset"], nb - 1 - s["tofbins"], nb)
    assert np.allclose(b, b_m, atol=1e-5 * b.max(), rtol=1e-4)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("end_greater_start", [True, False])
def test_tof_lm_point_offset_direction(xp: ModuleType, dev: str, axis: int, end_greater_start: bool):
    """point source at +5 along an axis-aligned LOR, 21 TOF bins of 1 mm: the
    central bin (10) hits the source for offset +5 if the end point is on the
    positive side and -5 otherwise"""
    n = 41
    img_shape = tuple(n if k == axis else 1 for k in range(3))
    img_origin = tuple(-20.0 if k == axis else 0.0 for k in range(3))
    img = np.zeros(img_shape, np.float32)
    img[tuple(25 if k == axis else 0 for k in range(3))] = 1.0  # voxel center at +5

    s = 1.0 if end_greater_start else -1.0
    xstart = np.zeros((21, 3), np.float32)
    xstart[:, axis] = -100.0 * s
    xend = -xstart
    tofbins = np.arange(21, dtype=np.int16)

    p = lib_fwd(xp, dev, img, img_origin, (1.0, 1.0, 1.0), xstart, xend, 1.0,
                np.float32(5.0 * s), tofbins, 21)
    expected = np.zeros(21)
    expected[10] = 1.0
    assert np.allclose(p, expected, atol=1e-6)


def test_tof_lm_point_histo_image_totals(xp: ModuleType, dev: str):
    """events inside of the box spanned by the outermost voxel centers add exactly
    their value to the image sum, events further than one voxel outside add nothing"""
    s = random_setup(seed=6, nlors=5000)
    shape, origin, vs = s["img"].shape, s["img_origin"], s["voxsize"]
    pts = ref_tof_bin_centers(s["xstart"], s["xend"], s["w"], s["offset"], s["tofbins"], s["nbins"])
    f = (pts - np.asarray(origin)) / np.asarray(vs)
    eps = 1e-3
    inside = np.all((f > eps) & (f < np.asarray(shape) - 1 - eps), axis=1)
    outside = np.any((f < -1 - eps) | (f > np.asarray(shape) + eps), axis=1)
    assert inside.sum() > 100 and outside.sum() > 100

    ones = np.ones(s["xstart"].shape[0], np.float32)
    b_in = lib_back(xp, dev, shape, origin, vs, s["xstart"][inside], s["xend"][inside],
                    ones[inside], s["w"], s["offset"][inside], s["tofbins"][inside], s["nbins"])
    assert math.isclose(b_in.sum(), inside.sum(), rel_tol=1e-5)

    b_out = lib_back(xp, dev, shape, origin, vs, s["xstart"][outside], s["xend"][outside],
                     ones[outside], s["w"], s["offset"][outside], s["tofbins"][outside], s["nbins"])
    assert b_out.sum() == 0.0

    p_out = lib_fwd(xp, dev, s["img"], origin, vs, s["xstart"][outside], s["xend"][outside],
                    s["w"], s["offset"][outside], s["tofbins"][outside], s["nbins"], init=7.0)
    assert np.all(p_out == 0.0)


def test_tof_lm_point_voxel_centers(xp: ModuleType, dev: str):
    """an event whose TOF bin center coincides with a voxel center reads / writes
    exactly that voxel"""
    img_shape, voxsize = (9, 9, 9), (2.0, 2.0, 2.0)
    img_origin = (-8.0, -8.0, -8.0)
    rng = np.random.default_rng(7)
    img = rng.uniform(0.0, 1.0, img_shape).astype(np.float32)

    # oblique LOR through voxel [6, 3, 4] (world (4, -2, 0)), 5 bins of width 3,
    # central bin (2) at the LOR midpoint
    target = np.array([4.0, -2.0, 0.0])
    u = np.array([1.0, 2.0, 2.0]) / 3.0
    xstart = (target - 40.0 * u).astype(np.float32)[None]
    xend = (target + 40.0 * u).astype(np.float32)[None]

    p = lib_fwd(xp, dev, img, img_origin, voxsize, xstart, xend, 3.0, np.float32(0.0),
                np.array([2], np.int16), 5)
    assert math.isclose(p[0], img[6, 3, 4], rel_tol=1e-5)

    b = lib_back(xp, dev, img_shape, img_origin, voxsize, xstart, xend, np.array([2.5], np.float32),
                 3.0, np.float32(0.0), np.array([2], np.int16), 5)
    assert math.isclose(b[6, 3, 4], 2.5, rel_tol=1e-5)
    assert math.isclose(b.sum(), 2.5, rel_tol=1e-5)


def test_tof_lm_point_degenerate_lor(xp: ModuleType, dev: str):
    """events with identical start and end point do not contribute"""
    img = np.ones((5, 5, 5), np.float32)
    origin, vs = (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0)
    xstart = np.zeros((2, 3), np.float32)
    xend = np.zeros((2, 3), np.float32)
    xend[1, 0] = 10.0  # second event is regular, through the image center
    xstart[1, 0] = -10.0
    tb = np.array([1, 1], np.int16)

    p = lib_fwd(xp, dev, img, origin, vs, xstart, xend, 1.0, np.float32(0.0), tb, 3, init=np.nan)
    assert p[0] == 0.0
    assert math.isclose(p[1], 1.0, rel_tol=1e-6)

    b = lib_back(xp, dev, img.shape, origin, vs, xstart, xend, np.ones(2, np.float32), 1.0,
                 np.float32(0.0), tb, 3)
    assert math.isclose(b.sum(), 1.0, rel_tol=1e-6)


def test_tof_lm_point_invalid_arguments(xp: ModuleType, dev: str):
    f32 = xp.float32
    xs = xp.zeros((4, 3), dtype=f32, device=dev)
    xe = xp.ones((4, 3), dtype=f32, device=dev)
    img = xp.zeros((3, 3, 3), dtype=f32, device=dev)
    org = xp.zeros(3, dtype=f32, device=dev)
    vs = xp.ones(3, dtype=f32, device=dev)
    vals = xp.zeros(4, dtype=f32, device=dev)
    off = xp.zeros(1, dtype=f32, device=dev)
    tb = xp.zeros(4, dtype=xp.int16, device=dev)

    for fn in (ppb.tof_lm_point_fwd, ppb.tof_lm_point_back):
        # valid call
        fn(xs, xe, img, org, vs, vals, 1.0, off, tb, 3)

        with pytest.raises(ValueError):  # different number of events in start / end
            fn(xs, xe[:3, ...], img, org, vs, vals, 1.0, off, tb, 3)
        with pytest.raises(ValueError):  # wrong number of projection values
            fn(xs, xe, img, org, vs, vals[:3], 1.0, off, tb, 3)
        with pytest.raises(ValueError):  # wrong number of TOF bin indices
            fn(xs, xe, img, org, vs, vals, 1.0, off, tb[:3], 3)
        with pytest.raises(ValueError):  # wrong shape of TOF center offset
            fn(xs, xe, img, org, vs, vals, 1.0, xp.zeros(2, dtype=f32, device=dev), tb, 3)
        with pytest.raises(ValueError):  # non-positive TOF bin width
            fn(xs, xe, img, org, vs, vals, 0.0, off, tb, 3)
        with pytest.raises(ValueError):  # non-finite TOF bin width
            fn(xs, xe, img, org, vs, vals, float("nan"), off, tb, 3)
        with pytest.raises(ValueError):  # no TOF bins
            fn(xs, xe, img, org, vs, vals, 1.0, off, tb, 0)
        with pytest.raises(TypeError):  # wrong dtype of TOF bin indices
            fn(xs, xe, img, org, vs, vals, 1.0, off, xp.zeros(4, dtype=xp.int32, device=dev), 3)
