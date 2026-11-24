"""Every CUDA projector must accept host, CUDA device and CUDA managed arrays.

In a CUDA build, the projectors look at every array argument with
``cudaPointerGetAttributes`` (``handle_cuda_input_array`` in src/cuda_utils.cu):

- **host** arrays are copied to temporary device buffers (outputs are copied back),
- **device** arrays are used directly,
- **managed** arrays are used directly (after best-effort prefetch / advise hints).

This test runs all 13 projectors with all their arrays of one memory type and
compares the results with those for host arrays. The TOF parameter arrays
(``tof_sigma``, ``tof_center_offset``, ``tof_kernel_cdf``, ``tof_kernel_scale``)
are either of the same memory type as the main arrays or host arrays (the Python
bindings allow host parameter arrays with GPU main arrays).

It also checks that

- the inputs are not modified,
- managed and device main arrays can not be mixed in one call (all main arrays
  must be of the same type),
- the managed arrays really are managed memory (``cudaMemoryTypeManaged``) and the
  device arrays really are device memory (``cudaMemoryTypeDevice``), so that each
  code path of ``handle_cuda_input_array`` is exercised,
- calls on a side stream give the same results (after synchronizing the stream).

Needs a CUDA build, a CUDA device and CuPy (managed arrays are allocated with
``cupy.cuda.malloc_managed``); skipped otherwise.
"""

import importlib

import numpy as np
import pytest

import parallelproj_core as ppb

from ._cuda_driver import has_physical_cuda_device

CUDA_MEMORY_TYPE_DEVICE = 2  # cudaMemoryTypeDevice
CUDA_MEMORY_TYPE_MANAGED = 3  # cudaMemoryTypeManaged


def _cupy_with_device():
    if not int(ppb.cuda_enabled) or importlib.util.find_spec("cupy") is None or not has_physical_cuda_device():
        return None
    import cupy

    try:
        return cupy if cupy.cuda.runtime.getDeviceCount() > 0 else None
    except Exception:
        return None


cp = _cupy_with_device()
needs_gpu = pytest.mark.skipif(cp is None, reason="needs a CUDA build, a CUDA device and CuPy")

################################################################################
# memory types
################################################################################


def _memory_type(a):
    return cp.cuda.runtime.pointerGetAttributes(a.data.ptr).type


def to_kind(x, kind):
    """numpy array -> array of the given memory type ("host", "device", "managed")"""
    x = np.ascontiguousarray(x)
    if kind == "host":
        return x.copy()
    if kind == "device":
        a = cp.asarray(x)
        assert _memory_type(a) == CUDA_MEMORY_TYPE_DEVICE
        return a
    if kind == "managed":
        with cp.cuda.using_allocator(cp.cuda.malloc_managed):
            a = cp.asarray(x)
        assert _memory_type(a) == CUDA_MEMORY_TYPE_MANAGED
        return a
    raise ValueError(kind)


def to_numpy(a):
    return a if isinstance(a, np.ndarray) else cp.asnumpy(a)


################################################################################
# the test problem
################################################################################

NB = 9  # number of TOF bins
W = 2.0  # TOF bin width
SIGMA = 3.0  # TOF sigma
U = 4.0  # kernel table range [-U, U] in units of the kernel scale


def _gauss_cdf(u):
    from math import erf, sqrt

    return np.array([0.5 * (1 + erf(v / sqrt(2))) for v in u])


def _problem(seed=0, n=300):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 1, (11, 12, 13)).astype(np.float32)
    org = np.array([-5.0, -5.5, -6.0], np.float32)
    vs = np.array([1.0, 0.9, 1.1], np.float32)

    def sphere():
        v = rng.normal(size=(n, 3))
        return (20 * v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    p = dict(
        xs=sphere(),
        xe=sphere(),
        img=img,
        org=org,
        vs=vs,
        y_lm=rng.uniform(0.5, 1.5, n).astype(np.float32),
        y_sino=rng.uniform(0.5, 1.5, (n, NB)).astype(np.float32),
        tb=rng.integers(0, NB, n).astype(np.int16),
        # parameter arrays: tof_sigma, tof_center_offset, tof_kernel_cdf, tof_kernel_scale
        sigma=np.array([SIGMA], np.float32),
        offset=rng.uniform(-1, 1, n).astype(np.float32),  # LOR dependent
        cdf=_gauss_cdf(np.linspace(-U, U, 129)).astype(np.float32),
        scale=rng.uniform(2.5, 3.5, n).astype(np.float32),  # LOR dependent
    )
    return p


PARAMS = ("sigma", "offset", "cdf", "scale")


def run(name, p, main, param, stream=None, kinds=None):
    """Run projector `name` on problem p with main arrays of kind `main` and parameter
    arrays of kind `param` (kinds: {input name: kind} overrides single inputs).
    Returns (output array, converted inputs)."""
    kinds = kinds or {}
    a = {k: to_kind(v, kinds.get(k, param if k in PARAMS else main)) for k, v in p.items()}
    n = p["xs"].shape[0]
    if name == "tof_lm_sort_key":
        out = to_kind(np.zeros(n, np.int64), main)
    elif "back" in name:
        out = to_kind(np.zeros(p["img"].shape, np.float32), main)
    elif "sino" in name:
        out = to_kind(np.zeros((n, NB), np.float32), main)
    else:
        out = to_kind(np.zeros(n, np.float32), main)
    if stream is not None and cp is not None:
        # the copies to device / managed memory above run on CuPy's current stream, which
        # is not ordered with a non-blocking side stream: wait for them
        cp.cuda.runtime.deviceSynchronize()
    geo = (a["xs"], a["xe"])
    kw = dict(stream=stream)
    tab = (a["cdf"], -U, U, a["scale"])

    if name == "joseph3d_fwd":
        ppb.joseph3d_fwd(*geo, a["img"], a["org"], a["vs"], out, **kw)
    elif name == "joseph3d_back":
        ppb.joseph3d_back(*geo, out, a["org"], a["vs"], a["y_lm"], **kw)
    elif name == "joseph3d_tof_sino_fwd":
        ppb.joseph3d_tof_sino_fwd(*geo, a["img"], a["org"], a["vs"], out, W, a["sigma"], a["offset"], NB, **kw)
    elif name == "joseph3d_tof_sino_back":
        ppb.joseph3d_tof_sino_back(*geo, out, a["org"], a["vs"], a["y_sino"], W, a["sigma"], a["offset"], NB, **kw)
    elif name == "joseph3d_tof_lm_fwd":
        ppb.joseph3d_tof_lm_fwd(*geo, a["img"], a["org"], a["vs"], out, W, a["sigma"], a["offset"], a["tb"], NB,
                                **kw)
    elif name == "joseph3d_tof_lm_back":
        ppb.joseph3d_tof_lm_back(*geo, out, a["org"], a["vs"], a["y_lm"], W, a["sigma"], a["offset"], a["tb"], NB,
                                 **kw)
    elif name == "joseph3d_tof_sino_fwd_tab":
        ppb.joseph3d_tof_sino_fwd_tab(*geo, a["img"], a["org"], a["vs"], out, W, a["offset"], NB, *tab, **kw)
    elif name == "joseph3d_tof_sino_back_tab":
        ppb.joseph3d_tof_sino_back_tab(*geo, out, a["org"], a["vs"], a["y_sino"], W, a["offset"], NB, *tab, **kw)
    elif name == "joseph3d_tof_lm_fwd_tab":
        ppb.joseph3d_tof_lm_fwd_tab(*geo, a["img"], a["org"], a["vs"], out, W, a["offset"], a["tb"], NB, *tab, **kw)
    elif name == "joseph3d_tof_lm_back_tab":
        ppb.joseph3d_tof_lm_back_tab(*geo, out, a["org"], a["vs"], a["y_lm"], W, a["offset"], a["tb"], NB, *tab,
                                     **kw)
    elif name == "tof_lm_point_fwd":
        ppb.tof_lm_point_fwd(*geo, a["img"], a["org"], a["vs"], out, W, a["offset"], a["tb"], NB, **kw)
    elif name == "tof_lm_point_back":
        ppb.tof_lm_point_back(*geo, out, a["org"], a["vs"], a["y_lm"], W, a["offset"], a["tb"], NB, **kw)
    elif name == "tof_lm_sort_key":
        ppb.tof_lm_sort_key(*geo, a["org"], a["vs"], out, W, a["offset"], a["tb"], NB, **kw)
    else:
        raise ValueError(name)
    return out, a


PROJECTORS = [
    "joseph3d_fwd",
    "joseph3d_back",
    "joseph3d_tof_sino_fwd",
    "joseph3d_tof_sino_back",
    "joseph3d_tof_lm_fwd",
    "joseph3d_tof_lm_back",
    "joseph3d_tof_sino_fwd_tab",
    "joseph3d_tof_sino_back_tab",
    "joseph3d_tof_lm_fwd_tab",
    "joseph3d_tof_lm_back_tab",
    "tof_lm_point_fwd",
    "tof_lm_point_back",
    "tof_lm_sort_key",
]


def _assert_same(out, ref, name):
    out = to_numpy(out)
    if name == "tof_lm_sort_key":
        np.testing.assert_array_equal(out, ref)
    else:
        # back projections use atomicAdd: the summation order may differ between calls
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6 * float(np.abs(ref).max()))


################################################################################
# tests
################################################################################


def test_all_projectors_covered():
    """every projector of the Python API is in PROJECTORS"""
    exported = {n for n in dir(ppb) if n.startswith(("joseph3d_", "tof_lm_"))}
    assert exported == set(PROJECTORS)


@needs_gpu
@pytest.mark.parametrize("param", ["same", "host"])
@pytest.mark.parametrize("main", ["host", "device", "managed"])
@pytest.mark.parametrize("name", PROJECTORS)
def test_memory_types(name, main, param):
    """all main arrays of one memory type (and the parameter arrays of the same type or
    on the host) give the same result as host arrays, and leave the inputs unchanged"""
    p = _problem()
    ref, _ = run(name, p, "host", "host")
    out, a = run(name, p, main, main if param == "same" else "host")
    # synchronous call (stream=None): the result is complete on return, also for GPU arrays
    _assert_same(out, ref, name)
    for k, v in p.items():
        np.testing.assert_array_equal(to_numpy(a[k]), v, err_msg=f"input {k} was modified")


@needs_gpu
@pytest.mark.parametrize("main", ["host", "device", "managed"])
@pytest.mark.parametrize("name", PROJECTORS)
def test_memory_types_side_stream(name, main):
    """same on a non-blocking side stream (asynchronous for device / managed arrays)"""
    p = _problem(1)
    ref, _ = run(name, p, "host", "host")
    s = cp.cuda.Stream(non_blocking=True)
    out, _ = run(name, p, main, main, stream=s.ptr)
    s.synchronize()
    _assert_same(out, ref, name)


@needs_gpu
@pytest.mark.parametrize("name", PROJECTORS)
def test_mixed_device_and_managed_main_arrays_rejected(name):
    """main arrays of different memory types (one managed, the others device arrays) must
    be rejected with a ValueError: all main arrays must be of the same type"""
    if to_kind(np.zeros(3, np.float32), "managed").__dlpack_device__()[0] != 13:  # kDLCUDAManaged
        pytest.skip("this CuPy exports managed memory as kDLCUDA (same type as device memory)")
    with pytest.raises(ValueError, match="same device type"):
        run(name, _problem(), "device", "device", kinds={"xs": "managed"})
