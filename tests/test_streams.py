"""Tests for the CUDA stream support of the projectors (``stream`` argument).

- ``stream=None`` (default): synchronous call, results are complete on return.
- ``stream=<handle>``: the call is queued on that stream; for GPU arrays it
  returns immediately (like a PyTorch / CuPy operation on that stream).
  ``0`` is the default stream of PyTorch / CuPy (asynchronous).
- CPU-only builds accept and ignore ``stream``.

Most tests need a CUDA build and a GPU (PyTorch and / or CuPy); they are skipped
otherwise. The multi-GPU tests need at least two GPUs.
"""

import importlib
import threading
import pytest
import numpy as np
import parallelproj_core as ppb

cuda_build = bool(int(ppb.cuda_enabled))

################################################################################
# helpers
################################################################################


def _problem(seed=0, n=20000, shape=(41, 43, 37)):
    """random LORs through an image, TOF bins, event values"""
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 1, shape).astype(np.float32)
    org = (-(np.array(shape) / 2) + 0.5).astype(np.float32)
    vs = np.ones(3, np.float32)
    v = rng.normal(size=(n, 3))
    xs = (60 * v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    v = rng.normal(size=(n, 3))
    xe = (60 * v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    tb = rng.integers(0, 25, n).astype(np.int16)
    y = rng.uniform(0, 1, n).astype(np.float32)
    return img, org, vs, xs, xe, tb, y


def _fwd(xs, xe, img, org, vs, out, **kw):
    ppb.joseph3d_fwd(xs, xe, img, org, vs, out, **kw)


def _point_back(xs, xe, out_img, org, vs, y, offset, tb, **kw):
    ppb.tof_lm_point_back(xs, xe, out_img, org, vs, y, 3.0, offset, tb, 25, **kw)


class _Torch:
    name = "torch"

    def __init__(self):
        import torch

        self.t = torch

    def available_devices(self):
        return self.t.cuda.device_count() if self.t.cuda.is_available() else 0

    def asarray(self, x, dev=0):
        return self.t.from_numpy(np.ascontiguousarray(x)).to(f"cuda:{dev}")

    def zeros(self, shape, dtype=np.float32, dev=0):
        return self.t.zeros(shape, dtype={np.float32: self.t.float32}[dtype], device=f"cuda:{dev}")

    def to_numpy(self, x):
        return x.cpu().numpy()

    def current_stream(self, dev=0):
        return self.t.cuda.current_stream(dev).cuda_stream

    def side_stream(self, dev=0):
        return self.t.cuda.Stream(device=dev)

    def stream_ctx(self, s):
        return self.t.cuda.stream(s)

    def handle(self, s):
        return s.cuda_stream

    def synchronize(self, dev=0):
        self.t.cuda.synchronize(dev)

    def busy(self, x, n=30):
        """queue slow work on the current stream that ends by overwriting x with 2 * x"""
        a = self.t.rand(2048, 2048, device=x.device)
        for _ in range(n):
            a = a @ a
            a = a / a.norm()
        x.mul_(2.0).add_(0.0 * a[0, 0])


class _Cupy:
    name = "cupy"

    def __init__(self):
        import cupy

        self.cp = cupy

    def available_devices(self):
        try:
            return self.cp.cuda.runtime.getDeviceCount()
        except Exception:
            return 0

    def asarray(self, x, dev=0):
        with self.cp.cuda.Device(dev):
            return self.cp.asarray(np.ascontiguousarray(x))

    def zeros(self, shape, dtype=np.float32, dev=0):
        with self.cp.cuda.Device(dev):
            return self.cp.zeros(shape, dtype=dtype)

    def to_numpy(self, x):
        return x.get()

    def current_stream(self, dev=0):
        with self.cp.cuda.Device(dev):
            return self.cp.cuda.get_current_stream().ptr

    def side_stream(self, dev=0):
        with self.cp.cuda.Device(dev):
            return self.cp.cuda.Stream(non_blocking=True)

    def stream_ctx(self, s):
        return s

    def handle(self, s):
        return s.ptr

    def synchronize(self, dev=0):
        self.cp.cuda.Device(dev).synchronize()

    def busy(self, x, n=30):
        a = self.cp.random.rand(2048, 2048, dtype=self.cp.float32)
        for _ in range(n):
            a = a @ a
            a = a / self.cp.linalg.norm(a)
        x *= 2.0
        x += 0.0 * a[0, 0]


def _gpu_libs():
    libs = []
    if not cuda_build:
        return libs
    if importlib.util.find_spec("torch") is not None:
        lib = _Torch()
        if lib.available_devices() > 0:
            libs.append(lib)
    if importlib.util.find_spec("cupy") is not None:
        lib = _Cupy()
        if lib.available_devices() > 0:
            libs.append(lib)
    return libs


GPU_LIBS = _gpu_libs()
NUM_GPUS = max([lib.available_devices() for lib in GPU_LIBS], default=0)

needs_gpu = pytest.mark.skipif(not GPU_LIBS, reason="needs a CUDA build, a GPU and PyTorch or CuPy")
needs_2_gpus = pytest.mark.skipif(NUM_GPUS < 2, reason="needs a CUDA build and >= 2 GPUs")
gpu_lib_params = pytest.mark.parametrize("lib", GPU_LIBS, ids=[lib.name for lib in GPU_LIBS])

################################################################################
# tests without GPU
################################################################################


def test_stream_pointer_mapping():
    # None -> nullptr (synchronous), 0 -> cudaStreamLegacy (asynchronous default
    # stream), anything else is passed through
    assert ppb._stream_pointer(None) == 0
    assert ppb._stream_pointer() == 0
    assert ppb._stream_pointer(0) == 1
    assert ppb._stream_pointer(2) == 2
    assert ppb._stream_pointer(0x7F0012345600) == 0x7F0012345600
    with pytest.raises(TypeError):
        ppb._stream_pointer(-1)


@pytest.mark.skipif(cuda_build, reason="CPU-only build behavior")
def test_cpu_build_ignores_stream():
    img, org, vs, xs, xe, tb, y = _problem(1, n=500)
    p_ref = np.zeros(xs.shape[0], np.float32)
    _fwd(xs, xe, img, org, vs, p_ref)
    for stream in (0, 1, 2, 123456789):
        p = np.zeros(xs.shape[0], np.float32)
        _fwd(xs, xe, img, org, vs, p, stream=stream)
        assert np.array_equal(p, p_ref)


def test_all_projectors_accept_stream_none():
    img, org, vs, xs, xe, tb, y = _problem(2, n=50)
    nb = 25
    sig, off = np.array([2.0], np.float32), np.array([0.0], np.float32)
    lm = np.zeros(xs.shape[0], np.float32)
    sino = np.zeros((xs.shape[0], nb), np.float32)
    im = np.zeros(img.shape, np.float32)
    ppb.joseph3d_fwd(xs, xe, img, org, vs, lm, stream=None)
    ppb.joseph3d_back(xs, xe, im, org, vs, lm, stream=None)
    ppb.joseph3d_tof_sino_fwd(xs, xe, img, org, vs, sino, 3.0, sig, off, nb, stream=None)
    ppb.joseph3d_tof_sino_back(xs, xe, im, org, vs, sino, 3.0, sig, off, nb, stream=None)
    ppb.joseph3d_tof_lm_fwd(xs, xe, img, org, vs, lm, 3.0, sig, off, tb, nb, stream=None)
    ppb.joseph3d_tof_lm_back(xs, xe, im, org, vs, lm, 3.0, sig, off, tb, nb, stream=None)
    ppb.tof_lm_point_fwd(xs, xe, img, org, vs, lm, 3.0, off, tb, nb, stream=None)
    ppb.tof_lm_point_back(xs, xe, im, org, vs, lm, 3.0, off, tb, nb, stream=None)


################################################################################
# tests with GPU arrays
################################################################################


def _reference(img, org, vs, xs, xe, tb, y, offset):
    p = np.zeros(xs.shape[0], np.float32)
    _fwd(xs, xe, img, org, vs, p)
    b = np.zeros(img.shape, np.float32)
    _point_back(xs, xe, b, org, vs, y, offset, tb)
    return p, b


def _gpu_fwd_back(lib, dev, img, org, vs, xs, xe, tb, y, offset, stream):
    a = lambda x: lib.asarray(x, dev)
    p = lib.zeros(xs.shape[0], dev=dev)
    _fwd(a(xs), a(xe), a(img), a(org), a(vs), p, stream=stream)
    b = lib.zeros(img.shape, dev=dev)
    _point_back(a(xs), a(xe), b, a(org), a(vs), a(y), a(offset), a(tb), stream=stream)
    return p, b


@needs_gpu
@gpu_lib_params
def test_async_default_stream(lib):
    """stream=0: asynchronous on the default stream, results are correct once the
    stream is synchronized (or the data is moved to the host)."""
    img, org, vs, xs, xe, tb, y = _problem(3)
    off = np.array([0.5], np.float32)
    p_ref, b_ref = _reference(img, org, vs, xs, xe, tb, y, off)
    p, b = _gpu_fwd_back(lib, 0, img, org, vs, xs, xe, tb, y, off, stream=0)
    # moving the results to the host synchronizes with the default stream
    assert np.allclose(lib.to_numpy(p), p_ref, rtol=1e-4, atol=1e-5)
    assert np.allclose(lib.to_numpy(b), b_ref, rtol=1e-4, atol=1e-5)


@needs_gpu
@gpu_lib_params
def test_input_produced_on_side_stream(lib):
    """An input written on a non-blocking side stream and projected on the same
    stream (passed as stream=...) must be read only after it is complete."""
    img, org, vs, xs, xe, tb, y = _problem(4)
    p_ref = np.zeros(xs.shape[0], np.float32)
    _fwd(xs, xe, 2 * img, org, vs, p_ref)

    s = lib.side_stream(0)
    x_gpu = lib.asarray(img, 0)
    xs_gpu, xe_gpu = lib.asarray(xs, 0), lib.asarray(xe, 0)
    org_gpu, vs_gpu = lib.asarray(org, 0), lib.asarray(vs, 0)
    lib.synchronize(0)
    p = lib.zeros(xs.shape[0], dev=0)
    with lib.stream_ctx(s):
        lib.busy(x_gpu)  # slow work on s that ends by doubling the image
        _fwd(xs_gpu, xe_gpu, x_gpu, org_gpu, vs_gpu, p, stream=lib.handle(s))
    s.synchronize()
    assert np.allclose(lib.to_numpy(p), p_ref, rtol=1e-4, atol=1e-5)


@needs_gpu
@gpu_lib_params
def test_host_arrays_with_stream(lib):
    """CPU arrays with an explicit stream: results are complete on return."""
    img, org, vs, xs, xe, tb, y = _problem(5)
    off = np.array([0.5], np.float32)
    p_ref, b_ref = _reference(img, org, vs, xs, xe, tb, y, off)
    s = lib.side_stream(0)
    p = np.zeros(xs.shape[0], np.float32)
    _fwd(xs, xe, img, org, vs, p, stream=lib.handle(s))
    b = np.zeros(img.shape, np.float32)
    _point_back(xs, xe, b, org, vs, y, off, tb, stream=lib.handle(s))
    assert np.allclose(p, p_ref, rtol=1e-4, atol=1e-5)
    assert np.allclose(b, b_ref, rtol=1e-4, atol=1e-5)


@needs_gpu
@gpu_lib_params
def test_cuda_graph_capture(lib):
    """A forward projection with GPU arrays can be recorded in a CUDA graph and replayed."""
    img, org, vs, xs, xe, tb, y = _problem(6)
    x_static = lib.asarray(img, 0)
    xs_gpu, xe_gpu = lib.asarray(xs, 0), lib.asarray(xe, 0)
    org_gpu, vs_gpu = lib.asarray(org, 0), lib.asarray(vs, 0)
    p_static = lib.zeros(xs.shape[0], dev=0)

    if lib.name == "torch":
        torch = lib.t
        # warm-up on a side stream, as recommended for torch.cuda.graph
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _fwd(xs_gpu, xe_gpu, x_static, org_gpu, vs_gpu, p_static, stream=s.cuda_stream)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _fwd(xs_gpu, xe_gpu, x_static, org_gpu, vs_gpu, p_static,
                 stream=torch.cuda.current_stream().cuda_stream)
        replay = g.replay
    else:
        cp = lib.cp
        s = cp.cuda.Stream(non_blocking=True)
        with s:
            _fwd(xs_gpu, xe_gpu, x_static, org_gpu, vs_gpu, p_static, stream=s.ptr)
            s.begin_capture()
            _fwd(xs_gpu, xe_gpu, x_static, org_gpu, vs_gpu, p_static, stream=s.ptr)
            g = s.end_capture()
        replay = lambda: g.launch(stream=s)

    # change the input in place and replay the graph
    img2 = (img[::-1, :, :] * 3).copy()
    x_static[...] = lib.asarray(img2, 0)
    lib.synchronize(0)
    replay()
    lib.synchronize(0)

    p_ref = np.zeros(xs.shape[0], np.float32)
    _fwd(xs, xe, img2, org, vs, p_ref)
    assert np.allclose(lib.to_numpy(p_static), p_ref, rtol=1e-4, atol=1e-5)


@needs_2_gpus
@gpu_lib_params
def test_minibatch_split_across_gpus_async(lib):
    """A mini batch split across all GPUs, one chunk per GPU, queued asynchronously
    on each GPU's current stream in one Python thread."""
    ngpu = lib.available_devices()
    batch = [_problem(10 + k) for k in range(2 * ngpu)]
    off = np.array([0.5], np.float32)
    refs = [_reference(img, org, vs, xs, xe, tb, y, off) for (img, org, vs, xs, xe, tb, y) in batch]

    results = []
    for k, (img, org, vs, xs, xe, tb, y) in enumerate(batch):
        dev = k % ngpu
        results.append(_gpu_fwd_back(lib, dev, img, org, vs, xs, xe, tb, y, off,
                                     stream=lib.current_stream(dev)))
    for dev in range(ngpu):
        lib.synchronize(dev)

    for (p, b), (p_ref, b_ref) in zip(results, refs):
        assert np.allclose(lib.to_numpy(p), p_ref, rtol=1e-4, atol=1e-5)
        assert np.allclose(lib.to_numpy(b), b_ref, rtol=1e-4, atol=1e-5)


@needs_2_gpus
@gpu_lib_params
def test_one_thread_per_gpu(lib):
    """One Python thread per GPU without a stream (synchronous calls): correct
    results (the GIL is released during the calls, so the threads can overlap)."""
    ngpu = lib.available_devices()
    off = np.array([0.5], np.float32)
    batch = [_problem(20 + k) for k in range(ngpu)]
    refs = [_reference(img, org, vs, xs, xe, tb, y, off) for (img, org, vs, xs, xe, tb, y) in batch]
    results = [None] * ngpu
    errors = []

    def work(dev):
        try:
            img, org, vs, xs, xe, tb, y = batch[dev]
            p, b = _gpu_fwd_back(lib, dev, img, org, vs, xs, xe, tb, y, off, stream=None)
            results[dev] = (lib.to_numpy(p), lib.to_numpy(b))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(dev,)) for dev in range(ngpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    for (p, b), (p_ref, b_ref) in zip(results, refs):
        assert np.allclose(p, p_ref, rtol=1e-4, atol=1e-5)
        assert np.allclose(b, b_ref, rtol=1e-4, atol=1e-5)
