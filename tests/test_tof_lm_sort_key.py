"""Tests of the sort keys of TOF listmode events (tof_lm_sort_key, issue #54).

The key of an event is (principal axis of the LOR) << 40 | 3D Morton code of the grid cell
(2 x 2 x 2 voxels) that contains the TOF center of the event (see
src/tof_lm_sort_key_worker.h). Sorting the events by this key brings events with nearby
TOF centers together, which makes the TOF listmode projectors much faster on GPUs; the
projection of an event must not depend on the order.
"""

import math

import numpy as np
import pytest
import parallelproj_core as ppb

from types import ModuleType
from .config import pytestmark

IMG_SHAPE = (40, 36, 24)
VOXEL_SIZE = (2.0, 2.5, 3.0)
NUM_TOF_BINS = 31
TOF_BIN_WIDTH = 6.0
CELL = 2  # grid cell of the key in voxels (TOF_LM_SORT_KEY_CELL_VOXELS)


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "get"):  # cupy
        return x.get()
    if hasattr(x, "cpu"):  # torch
        x = x.cpu()
    return np.from_dlpack(x)


def make_events(n, seed=0):
    """random events through (and partly outside of) the image, random TOF bins and
    per-event TOF center offsets; plus a degenerate event (start == end) at index 3"""
    rng = np.random.default_rng(seed)
    vs = np.array(VOXEL_SIZE, np.float32)
    origin = (-0.5 * (np.array(IMG_SHAPE) - 1) * vs).astype(np.float32)

    def points():
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return (d * 70 + rng.normal(scale=10, size=(n, 3))).astype(np.float32)

    xs, xe = points(), points()
    xe[3] = xs[3]
    it = rng.integers(0, NUM_TOF_BINS, n).astype(np.int16)
    off = rng.uniform(-5, 5, n).astype(np.float32)
    return xs, xe, it, off, origin, vs


def ref_key_parts(xs, xe, it, off, origin, vs):
    """principal axis and grid cell (float64 reference) of every event"""
    d = xe.astype(np.float64) - xs.astype(np.float64)
    length = np.linalg.norm(d, axis=1)
    x = (it.astype(np.float64) - 0.5 * (NUM_TOF_BINS - 1)) * TOF_BIN_WIDTH + off
    with np.errstate(invalid="ignore", divide="ignore"):
        c = 0.5 * (xs + xe).astype(np.float64) + (x / length)[:, None] * d
    f = (c - origin) / vs
    q = np.clip(np.nan_to_num(np.floor((f + 0.5) / CELL)), 0, 1023).astype(np.int64)
    s = d**2
    axis = np.where((s[:, 1] >= s[:, 0]) & (s[:, 1] >= s[:, 2]), 1,
                    np.where((s[:, 2] >= s[:, 0]) & (s[:, 2] >= s[:, 1]), 2, 0))
    axis[length == 0] = 3
    return axis, q


def decode(key):
    """principal axis and grid cell (qx, qy, qz) from the keys"""
    key = key.astype(np.int64)
    q = np.zeros((key.shape[0], 3), np.int64)
    for bit in range(10):
        for k in range(3):
            q[:, k] |= ((key >> (3 * bit + k)) & 1) << bit
    return key >> 40, q


def compute_keys(xp, dev, xs, xe, it, off, origin, vs):
    key = xp.zeros(xs.shape[0], dtype=xp.int64, device=dev)
    ppb.tof_lm_sort_key(xp.asarray(xs, device=dev), xp.asarray(xe, device=dev), xp.asarray(origin, device=dev),
                        xp.asarray(vs, device=dev), key, TOF_BIN_WIDTH, xp.asarray(off, device=dev),
                        xp.asarray(it, device=dev), NUM_TOF_BINS)
    return to_numpy(key)


def test_sort_key_vs_reference(xp: ModuleType, dev: str):
    """axis and grid cell of the key vs. a float64 reference (the cell can differ by one
    for TOF centers within float rounding of a cell boundary)"""
    xs, xe, it, off, origin, vs = make_events(5000)
    key = compute_keys(xp, dev, xs, xe, it, off, origin, vs)
    axis, q = decode(key)
    axis_ref, q_ref = ref_key_parts(xs, xe, it, off, origin, vs)
    assert np.array_equal(axis, axis_ref)
    ok = axis_ref < 3
    # degenerate event: key = 3 << 40 exactly (sorted last)
    assert key[3] == 3 << 40
    assert np.abs(q[ok] - q_ref[ok]).max() <= 1
    assert np.mean(np.all(q[ok] == q_ref[ok], axis=1)) > 0.99
    # the keys use only bits 0-29 and 40-41
    assert np.all((key & ~((np.int64(3) << 40) | np.int64((1 << 30) - 1))) == 0)


def test_sort_key_scalar_offset(xp: ModuleType, dev: str):
    """one TOF center offset for all events = the same offset for every event"""
    xs, xe, it, off, origin, vs = make_events(500, seed=1)
    off1 = np.full(1, 3.5, np.float32)
    k1 = compute_keys(xp, dev, xs, xe, it, off1, origin, vs)
    k2 = compute_keys(xp, dev, xs, xe, it, np.full(xs.shape[0], 3.5, np.float32), origin, vs)
    assert np.array_equal(k1, k2)


def test_sort_key_morton_order(xp: ModuleType, dev: str):
    """events along a line of grid cells along one axis: the Morton code increases with
    the cell coordinate (bits of axis 0 at positions 0, 3, 6, ...)"""
    n = 16
    vs = np.ones(3, np.float32)
    origin = np.zeros(3, np.float32)
    # LORs along axis 0 (principal axis 0) whose TOF center (central TOF bin, zero offset)
    # is the LOR center, at x = 2 k (cell k), y = z = 0
    xc = 2.0 * np.arange(n, dtype=np.float32)
    xs = np.stack([xc - 50, np.zeros(n), np.zeros(n)], axis=1).astype(np.float32)
    xe = np.stack([xc + 50, np.zeros(n), np.zeros(n)], axis=1).astype(np.float32)
    it = np.full(n, NUM_TOF_BINS // 2, np.int16)
    key = compute_keys(xp, dev, xs, xe, it, np.zeros(1, np.float32), origin, vs)
    expected = [sum(((k >> b) & 1) << (3 * b) for b in range(10)) for k in range(n)]
    assert key.tolist() == expected


def test_sorting_brings_tof_centers_together(xp: ModuleType, dev: str):
    """after sorting, consecutive events have much closer TOF centers than before"""
    xs, xe, it, off, origin, vs = make_events(20000, seed=2)
    xe[3] = xe[2]  # no degenerate event here
    key = compute_keys(xp, dev, xs, xe, it, off, origin, vs)
    d = xe.astype(np.float64) - xs
    c = 0.5 * (xs + xe) + (((it - 0.5 * (NUM_TOF_BINS - 1)) * TOF_BIN_WIDTH + off) / np.linalg.norm(d, axis=1))[:, None] * d
    inside = np.all(np.abs(c) < 0.5 * np.array(IMG_SHAPE) * np.array(VOXEL_SIZE), axis=1)
    order = np.argsort(key[inside], kind="stable")
    ci = c[inside]
    dist_random = np.linalg.norm(np.diff(ci, axis=0), axis=1).mean()
    dist_sorted = np.linalg.norm(np.diff(ci[order], axis=0), axis=1).mean()
    assert dist_sorted < 0.2 * dist_random


@pytest.mark.parametrize("kernel", ["gauss", "table"])
def test_projections_do_not_depend_on_the_order(xp: ModuleType, dev: str, kernel: str):
    """forward projection of the sorted events = reordered forward projection (exactly);
    back projection of the sorted events = back projection of the unsorted events (up to
    the rounding of the different summation order)"""
    xs, xe, it, off, origin, vs = make_events(3000, seed=3)
    key = compute_keys(xp, dev, xs, xe, it, off, origin, vs)
    order = np.argsort(key, kind="stable")
    rng = np.random.default_rng(4)
    img = rng.uniform(size=IMG_SHAPE).astype(np.float32)
    vals = rng.uniform(size=xs.shape[0]).astype(np.float32)
    sigma = np.array([5.0], np.float32)
    u = np.linspace(-15.0, 15.0, 501)
    cdf = np.array([0.5 * (1 + math.erf(x / (math.sqrt(2) * 5.0))) for x in u], np.float32)

    def a(x):
        return xp.asarray(x, device=dev)

    def fwd(s, e, t, o):
        p = xp.zeros(s.shape[0], dtype=xp.float32, device=dev)
        if kernel == "gauss":
            ppb.joseph3d_tof_lm_fwd(a(s), a(e), a(img), a(origin), a(vs), p, TOF_BIN_WIDTH, a(sigma), a(o), a(t),
                                    NUM_TOF_BINS)
        else:
            ppb.joseph3d_tof_lm_fwd_tab(a(s), a(e), a(img), a(origin), a(vs), p, TOF_BIN_WIDTH, a(o), a(t),
                                        NUM_TOF_BINS, a(cdf), float(u[0]), float(u[-1]), a(np.ones(1, np.float32)))
        return to_numpy(p)

    def back(s, e, t, o, v):
        x = xp.zeros(IMG_SHAPE, dtype=xp.float32, device=dev)
        if kernel == "gauss":
            ppb.joseph3d_tof_lm_back(a(s), a(e), x, a(origin), a(vs), a(v), TOF_BIN_WIDTH, a(sigma), a(o), a(t),
                                     NUM_TOF_BINS)
        else:
            ppb.joseph3d_tof_lm_back_tab(a(s), a(e), x, a(origin), a(vs), a(v), TOF_BIN_WIDTH, a(o), a(t),
                                         NUM_TOF_BINS, a(cdf), float(u[0]), float(u[-1]), a(np.ones(1, np.float32)))
        return to_numpy(x).astype(np.float64)

    p = fwd(xs, xe, it, off)
    p_sorted = fwd(xs[order], xe[order], it[order], off[order])
    assert np.abs(p).max() > 0
    assert np.array_equal(p_sorted, p[order])
    b = back(xs, xe, it, off, vals)
    b_sorted = back(xs[order], xe[order], it[order], off[order], vals[order])
    assert np.abs(b).max() > 0
    assert np.abs(b_sorted - b).max() < 1e-5 * np.abs(b).max()


def test_sort_key_input_validation(xp: ModuleType, dev: str):
    xs, xe, it, off, origin, vs = make_events(10)

    def call(xs=xs, xe=xe, key_n=10, it=it, off=off, w=TOF_BIN_WIDTH, nb=NUM_TOF_BINS):
        key = xp.zeros(key_n, dtype=xp.int64, device=dev)
        ppb.tof_lm_sort_key(xp.asarray(xs, device=dev), xp.asarray(xe, device=dev), xp.asarray(origin, device=dev),
                            xp.asarray(vs, device=dev), key, w, xp.asarray(off, device=dev),
                            xp.asarray(it, device=dev), nb)

    call()
    with pytest.raises(ValueError, match="event_start and event_end must have the same number"):
        call(xe=xe[:5])
    with pytest.raises(ValueError, match="2 dimensions"):
        call(xs=xs[None], xe=xe[None])
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
        call(xs=np.ascontiguousarray(xs[:, :2]), xe=np.ascontiguousarray(xe[:, :2]))
    with pytest.raises(ValueError, match="sort_key must have shape"):
        call(key_n=9)
    with pytest.raises(ValueError, match="tof_bin_index must have shape"):
        call(it=it[:5])
    with pytest.raises(ValueError, match="tof_center_offset"):
        call(off=off[:5])
    with pytest.raises(ValueError, match="tof_bin_width"):
        call(w=0.0)
    with pytest.raises(ValueError, match="num_tof_bins"):
        call(nb=0)


def test_sort_key_cell_clamping(xp: ModuleType, dev: str):
    """TOF centers far outside the grid: cell coordinates clamped to 0 and 1023"""
    vs = np.ones(3, np.float32)
    origin = np.zeros(3, np.float32)
    # LOR centers (= TOF centers, central TOF bin) at -1e4 and +1e4 voxels along all axes
    c = np.array([[-1e4, -1e4, -1e4], [1e4, 1e4, 1e4]], np.float32)
    xs = c - np.array([50.0, 0.0, 0.0], np.float32)
    xe = c + np.array([50.0, 0.0, 0.0], np.float32)
    it = np.full(2, NUM_TOF_BINS // 2, np.int16)
    key = compute_keys(xp, dev, xs, xe, it, np.zeros(1, np.float32), origin, vs)
    axis, q = decode(key)
    assert axis.tolist() == [0, 0]
    assert q.tolist() == [[0, 0, 0], [1023, 1023, 1023]]
