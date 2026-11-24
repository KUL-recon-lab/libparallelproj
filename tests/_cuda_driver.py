"""Detection of a physical CUDA device via the CUDA driver API (ctypes only).

Used by conftest.py (test mode) and by all test modules before they ask CuPy or PyTorch for
the number of devices: on a system without an NVIDIA driver (e.g. CI runners that build with
CUDA but have no GPU), cupy.cuda.runtime.getDeviceCount() can crash the process (access
violation with CuPy 14.2 on Windows) instead of raising an exception, so CuPy must not be
asked at all there.
"""

import functools


@functools.lru_cache(maxsize=None)
def has_physical_cuda_device() -> bool:
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
