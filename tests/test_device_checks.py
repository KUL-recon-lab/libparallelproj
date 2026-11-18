"""Tests for the device checks of the Python bindings.

The checks only look at the device type / ID of the array arguments (via DLPack)
and raise before any memory is accessed. They can therefore be tested without a
GPU, using small array-like objects that export a DLPack capsule claiming to live
on a CUDA device (the data is host memory that is never read).

Tests with real GPU arrays only run on machines with (at least two) CUDA devices.
"""

import ctypes
import importlib
import pytest
import numpy as np
import parallelproj_core as ppb

KDL_CPU = 1
KDL_CUDA = 2


################################################################################
# fake device arrays exporting a DLPack capsule
################################################################################


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


class FakeDeviceArray:
    """C-contiguous array-like object whose DLPack capsule claims the given device."""

    def __init__(self, shape, dtype=np.float32, device_type=KDL_CUDA, device_id=0):
        self._host = np.zeros(shape, dtype=dtype)
        self._shape = (ctypes.c_int64 * len(shape))(*shape)
        dt = np.dtype(dtype)
        code = {"f": 2, "i": 0, "u": 1}[dt.kind]
        self._managed = _DLManagedTensor()
        self._managed.dl_tensor = _DLTensor(
            self._host.ctypes.data,
            _DLDevice(device_type, device_id),
            len(shape),
            _DLDataType(code, dt.itemsize * 8, 1),
            self._shape,
            None,
            0,
        )
        self._managed.manager_ctx = None
        self._managed.deleter = None
        self._device = (device_type, device_id)

    def __dlpack__(self, *args, **kwargs):
        return _PyCapsule_New(ctypes.addressof(self._managed), b"dltensor", None)

    def __dlpack_device__(self):
        return self._device


################################################################################
# argument sets of all projector functions
################################################################################

N = 4  # number of LORs / events
NB = 5  # number of TOF bins

# (function, argument spec). "M": main array (shape, dtype), "P": parameter array
# (tof_sigma, tof_center_offset), anything else: passed as is
FUNCTIONS = {
    "joseph3d_fwd": [("M", (N, 3)), ("M", (N, 3)), ("M", (3, 3, 3)), ("M", (3,)), ("M", (3,)), ("M", (N,))],
    "joseph3d_back": [("M", (N, 3)), ("M", (N, 3)), ("M", (3, 3, 3)), ("M", (3,)), ("M", (3,)), ("M", (N,))],
    "joseph3d_tof_sino_fwd": [("M", (N, 3)), ("M", (N, 3)), ("M", (3, 3, 3)), ("M", (3,)), ("M", (3,)),
                              ("M", (N, NB)), 1.0, ("P", (1,)), ("P", (1,)), NB],
    "joseph3d_tof_sino_back": [("M", (N, 3)), ("M", (N, 3)), ("M", (3, 3, 3)), ("M", (3,)), ("M", (3,)),
                               ("M", (N, NB)), 1.0, ("P", (1,)), ("P", (1,)), NB],
    "joseph3d_tof_lm_fwd": [("M", (N, 3)), ("M", (N, 3)), ("M", (3, 3, 3)), ("M", (3,)), ("M", (3,)),
                            ("M", (N,)), 1.0, ("P", (1,)), ("P", (1,)), ("M", (N,), np.int16), NB],
    "joseph3d_tof_lm_back": [("M", (N, 3)), ("M", (N, 3)), ("M", (3, 3, 3)), ("M", (3,)), ("M", (3,)),
                             ("M", (N,)), 1.0, ("P", (1,)), ("P", (1,)), ("M", (N,), np.int16), NB],
    "tof_lm_point_fwd": [("M", (N, 3)), ("M", (N, 3)), ("M", (3, 3, 3)), ("M", (3,)), ("M", (3,)),
                         ("M", (N,)), 1.0, ("P", (1,)), ("M", (N,), np.int16), NB],
    "tof_lm_point_back": [("M", (N, 3)), ("M", (N, 3)), ("M", (3, 3, 3)), ("M", (3,)), ("M", (3,)),
                          ("M", (N,)), 1.0, ("P", (1,)), ("M", (N,), np.int16), NB],
}


def build_args(name, main, param, override=None):
    """Arguments of projector `name`. main / param: factories (shape, dtype) -> array.
    override: {argument index: factory} for single arguments."""
    args = []
    for k, a in enumerate(FUNCTIONS[name]):
        if isinstance(a, tuple) and a[0] in ("M", "P"):
            shape = a[1]
            dtype = a[2] if len(a) > 2 else np.float32
            factory = (override or {}).get(k, main if a[0] == "M" else param)
            args.append(factory(shape, dtype))
        else:
            args.append(a)
    return args


def cuda(device_id):
    return lambda shape, dtype: FakeDeviceArray(shape, dtype, KDL_CUDA, device_id)


def host(shape, dtype):
    return np.zeros(shape, dtype=dtype)


def param_indices(name):
    return [k for k, a in enumerate(FUNCTIONS[name]) if isinstance(a, tuple) and a[0] == "P"]


cuda_build = bool(int(ppb.cuda_enabled))

################################################################################
# tests without GPU
################################################################################


def test_all_projectors_covered():
    for name in FUNCTIONS:
        assert hasattr(ppb, name)


@pytest.mark.skipif(cuda_build, reason="CPU-only build behavior")
@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_cpu_build_rejects_gpu_arrays(name):
    """A CPU-only build must raise for GPU arrays instead of dereferencing device pointers."""
    fn = getattr(ppb, name)
    with pytest.raises(ValueError, match="not a CPU array"):
        fn(*build_args(name, cuda(0), cuda(0)))


@pytest.mark.skipif(cuda_build, reason="CPU-only build behavior")
@pytest.mark.parametrize("name", [n for n in FUNCTIONS if param_indices(n)])
def test_cpu_build_rejects_gpu_parameter_arrays(name):
    """tof_sigma / tof_center_offset on a GPU must be rejected in CPU-only builds."""
    fn = getattr(ppb, name)
    for k in param_indices(name):
        with pytest.raises(ValueError, match="not a CPU array"):
            fn(*build_args(name, host, host, override={k: cuda(0)}))


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_main_arrays_on_different_gpus(name):
    fn = getattr(ppb, name)
    with pytest.raises(ValueError, match="same device ID"):
        fn(*build_args(name, cuda(0), cuda(0), override={2: cuda(1)}))


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_main_arrays_on_cpu_and_gpu(name):
    fn = getattr(ppb, name)
    with pytest.raises(ValueError, match="same device type"):
        fn(*build_args(name, host, host, override={2: cuda(0)}))


@pytest.mark.parametrize("name", [n for n in FUNCTIONS if param_indices(n)])
def test_parameter_array_on_other_gpu(name):
    """tof_sigma / tof_center_offset on another GPU than the main arrays must be rejected."""
    fn = getattr(ppb, name)
    for k in param_indices(name):
        with pytest.raises(ValueError):
            fn(*build_args(name, cuda(0), cuda(0), override={k: cuda(1)}))


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_device_id_mismatch(name):
    """An explicit device_id different from the GPU of the arrays must be rejected."""
    fn = getattr(ppb, name)
    with pytest.raises(ValueError):
        fn(*build_args(name, cuda(0), cuda(0)), device_id=1)


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_host_arrays_accepted(name):
    """CPU arrays (incl. the default device_id) must be accepted by all builds."""
    if cuda_build and not _num_cuda_devices():
        pytest.skip("CUDA build without CUDA device")
    getattr(ppb, name)(*build_args(name, host, host))


################################################################################
# decision logic of the device checks for both build types
# (exposed as _check_devices, so the CUDA-build logic is also tested in CPU builds)
################################################################################

KDL_CUDA_HOST = 3
KDL_ROCM = 10
KDL_CUDA_MANAGED = 13


def _main(device_type, device_id=0, n=3):
    return [(f"array{k}", device_type, device_id) for k in range(n)]


@pytest.mark.parametrize("cuda_build_flag", [False, True])
@pytest.mark.parametrize("host_type", [KDL_CPU, KDL_CUDA_HOST])
def test_check_devices_host_arrays(cuda_build_flag, host_type):
    # host arrays: device_id is passed on unchanged (-1: automatic / current device)
    for device_id in (-1, 0, 2):
        assert ppb._check_devices(_main(host_type), [("p", KDL_CPU, 0)], device_id, cuda_build_flag) == device_id


def test_check_devices_cpu_build_rejects_device_arrays():
    for t in (KDL_CUDA, KDL_CUDA_MANAGED, KDL_ROCM):
        with pytest.raises(ValueError, match="array0 is not a CPU array"):
            ppb._check_devices(_main(t), [], -1, False)
        with pytest.raises(ValueError, match="p is not a CPU array"):
            ppb._check_devices(_main(KDL_CPU), [("p", t, 0)], -1, False)


@pytest.mark.parametrize("cuda_type", [KDL_CUDA, KDL_CUDA_MANAGED])
def test_check_devices_cuda_build_gpu_arrays(cuda_type):
    # GPU arrays: the computation runs on their device
    assert ppb._check_devices(_main(cuda_type, 1), [], -1, True) == 1
    assert ppb._check_devices(_main(cuda_type, 1), [], 1, True) == 1
    # an explicit, different device_id is an error
    with pytest.raises(ValueError, match="does not match the CUDA device"):
        ppb._check_devices(_main(cuda_type, 1), [], 0, True)
    # parameter arrays: on the CPU or on the same device
    assert ppb._check_devices(_main(cuda_type, 1), [("p", KDL_CPU, 0), ("q", cuda_type, 1)], -1, True) == 1
    with pytest.raises(ValueError, match="q must be a CPU array or on the same device"):
        ppb._check_devices(_main(cuda_type, 1), [("q", cuda_type, 0)], -1, True)
    with pytest.raises(ValueError, match="q must be a CPU array or on the same device"):
        ppb._check_devices(_main(cuda_type, 1), [("q", KDL_ROCM, 1)], -1, True)


def test_check_devices_cuda_build_other_cases():
    # GPU parameter arrays with host main arrays are an error
    with pytest.raises(ValueError, match="q must be a CPU array or on the same device"):
        ppb._check_devices(_main(KDL_CPU), [("q", KDL_CUDA, 0)], -1, True)
    # unsupported device types
    with pytest.raises(ValueError, match="unsupported device type of array0"):
        ppb._check_devices(_main(KDL_ROCM), [], -1, True)


@pytest.mark.parametrize("cuda_build_flag", [False, True])
def test_check_devices_main_arrays_must_match(cuda_build_flag):
    with pytest.raises(ValueError, match="same device type \\(array0 and x differ\\)"):
        ppb._check_devices(_main(KDL_CUDA, 0) + [("x", KDL_CPU, 0)], [], -1, cuda_build_flag)
    with pytest.raises(ValueError, match="same device ID \\(array0 and x differ\\)"):
        ppb._check_devices(_main(KDL_CUDA, 0) + [("x", KDL_CUDA, 1)], [], -1, cuda_build_flag)
    with pytest.raises(ValueError, match="no main arrays"):
        ppb._check_devices([], [], -1, cuda_build_flag)


################################################################################
# tests with real GPU arrays
################################################################################


def _gpu_xp():
    """array namespace with GPU support (torch preferred, then cupy) or None"""
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            return "torch"
    if importlib.util.find_spec("cupy") is not None:
        import cupy

        try:
            if cupy.cuda.runtime.getDeviceCount() > 0:
                return "cupy"
        except Exception:
            pass
    return None


def _num_cuda_devices():
    lib = _gpu_xp()
    if lib == "torch":
        import torch

        return torch.cuda.device_count()
    if lib == "cupy":
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    return 0


def _to_gpu(x, device_id):
    if _gpu_xp() == "torch":
        import torch

        return torch.from_numpy(np.ascontiguousarray(x)).to(f"cuda:{device_id}")
    import cupy

    with cupy.cuda.Device(device_id):
        return cupy.asarray(x)


def _current_cuda_device():
    """current CUDA device of the process (as seen by the CUDA runtime)"""
    if _gpu_xp() == "torch":
        import torch

        return torch.cuda.current_device()
    import cupy

    return cupy.cuda.runtime.getDevice()


def _to_numpy(x):
    if _gpu_xp() == "torch":
        return x.cpu().numpy()
    return x.get()


def _random_problem(seed=0, n=500):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 1, (11, 12, 13)).astype(np.float32)
    org = np.array([-5.0, -5.5, -6.0], np.float32)
    vs = np.ones(3, np.float32)
    v = rng.normal(size=(n, 3))
    xs = (30 * v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    v = rng.normal(size=(n, 3))
    xe = (30 * v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    tb = rng.integers(0, 15, n).astype(np.int16)
    return img, org, vs, xs, xe, tb


def _fwd_and_back(xs, xe, img, org, vs, tb, sigma, offset, to_dev, back_shape, **kw):
    """TOF LM fwd and point back projection with all main arrays converted by to_dev"""
    p = to_dev(np.zeros(xs.shape[0], np.float32))
    ppb.joseph3d_tof_lm_fwd(to_dev(xs), to_dev(xe), to_dev(img), to_dev(org), to_dev(vs), p, 4.0,
                            sigma, offset, to_dev(tb), 15, **kw)
    b = to_dev(np.zeros(back_shape, np.float32))
    ppb.tof_lm_point_back(to_dev(xs), to_dev(xe), b, to_dev(org), to_dev(vs),
                          to_dev(np.ones(xs.shape[0], np.float32)), 4.0, offset, to_dev(tb), 15, **kw)
    return p, b


@pytest.mark.skipif(not cuda_build or _num_cuda_devices() < 1, reason="needs a CUDA build and a CUDA device")
def test_gpu_arrays_with_host_parameter_arrays():
    """GPU main arrays with CPU tof_sigma / tof_center_offset (copied to the GPU) give the
    same result as everything on the CPU."""
    img, org, vs, xs, xe, tb = _random_problem(1)
    sig = np.array([3.0], np.float32)
    off = np.array([0.5], np.float32)
    p_ref, b_ref = _fwd_and_back(xs, xe, img, org, vs, tb, sig, off, lambda x: x, img.shape)
    p, b = _fwd_and_back(xs, xe, img, org, vs, tb, sig, off, lambda x: _to_gpu(x, 0), img.shape)
    assert np.allclose(_to_numpy(p), p_ref, rtol=1e-4, atol=1e-5)
    assert np.allclose(_to_numpy(b), b_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not cuda_build or _num_cuda_devices() < 2, reason="needs a CUDA build and >= 2 CUDA devices")
def test_arrays_on_second_gpu():
    """Arrays on GPU 1 with the default device_id must be processed on GPU 1, and the
    projectors must not change the caller's current CUDA device."""
    current_device = _current_cuda_device()
    img, org, vs, xs, xe, tb = _random_problem(2)
    to1 = lambda x: _to_gpu(x, 1)
    sig, off = to1(np.array([3.0], np.float32)), to1(np.array([0.5], np.float32))
    p_ref, b_ref = _fwd_and_back(xs, xe, img, org, vs, tb, np.array([3.0], np.float32),
                                 np.array([0.5], np.float32), lambda x: x, img.shape)
    p, b = _fwd_and_back(xs, xe, img, org, vs, tb, sig, off, to1, img.shape)
    assert np.allclose(_to_numpy(p), p_ref, rtol=1e-4, atol=1e-5)
    assert np.allclose(_to_numpy(b), b_ref, rtol=1e-4, atol=1e-5)

    # explicitly requesting the matching device is fine, another device is an error
    _fwd_and_back(xs, xe, img, org, vs, tb, sig, off, to1, img.shape, device_id=1)
    with pytest.raises(ValueError):
        _fwd_and_back(xs, xe, img, org, vs, tb, sig, off, to1, img.shape, device_id=0)

    assert _current_cuda_device() == current_device


@pytest.mark.skipif(not cuda_build or _num_cuda_devices() < 2, reason="needs a CUDA build and >= 2 CUDA devices")
def test_host_arrays_on_other_device_id():
    """Host arrays with an explicit device_id that is not the current device: same result,
    and the caller's current CUDA device is unchanged afterwards."""
    current_device = _current_cuda_device()
    other_device = (current_device + 1) % _num_cuda_devices()
    img, org, vs, xs, xe, tb = _random_problem(3)
    sig, off = np.array([3.0], np.float32), np.array([0.5], np.float32)
    p_ref, b_ref = _fwd_and_back(xs, xe, img, org, vs, tb, sig, off, lambda x: x, img.shape)
    p, b = _fwd_and_back(xs, xe, img, org, vs, tb, sig, off, lambda x: x, img.shape, device_id=other_device)
    assert np.allclose(p, p_ref, rtol=1e-4, atol=1e-5)
    assert np.allclose(b, b_ref, rtol=1e-4, atol=1e-5)
    assert _current_cuda_device() == current_device
