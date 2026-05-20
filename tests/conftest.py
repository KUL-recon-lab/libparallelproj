# tests/conftest.py
import pytest


def _cuda_build_enabled() -> bool:
    import parallelproj_core as pp
    return bool(int(pp.cuda_enabled))


def _has_physical_cuda_device() -> bool:
    """Detect a physical CUDA device via the CUDA Driver API (ctypes only).

    Loads libcuda / nvcuda directly so this works without cupy or torch.
    """
    import ctypes
    import sys

    candidates = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "libcuda.so"]
    lib = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return False

    # cuInit(0) returns CUDA_SUCCESS (0) when the driver is present
    try:
        if lib.cuInit(0) != 0:
            return False
    except AttributeError:
        return False

    # cuDeviceGetCount(&n) returns CUDA_SUCCESS (0) and writes the device count
    count = ctypes.c_int(0)
    try:
        if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return False
    except AttributeError:
        return False

    return count.value > 0


def pytest_configure(config):
    # Decide once per session
    try:
        cuda_build = _cuda_build_enabled()
    except Exception:
        # Don't skip on import error (packaging problem): let pytest fail normally
        config._pp_mode = "import_failed"
        return

    if not cuda_build:
        config._pp_mode = "cpu_build"
    else:
        config._pp_mode = "cuda_build_with_gpu" if _has_physical_cuda_device() else "cuda_build_no_gpu"


def pytest_report_header(config):
    return [f"parallelproj_core test mode: {getattr(config, '_pp_mode', 'unknown')}"]


def pytest_collection_modifyitems(config, items):
    # Your policy:
    # - CPU build: run all tests
    # - CUDA build + GPU: run all tests
    # - CUDA build + no GPU: skip all tests
    if getattr(config, "_pp_mode", None) == "cuda_build_no_gpu":
        skip = pytest.mark.skip(
            reason="parallelproj_core built with CUDA (cuda_enabled==1) but no CUDA device is available"
        )
        for item in items:
            item.add_marker(skip)
