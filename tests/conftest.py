# tests/conftest.py
import pytest


def _cuda_build_enabled() -> bool:
    import parallelproj_core as pp
    return bool(int(pp.cuda_enabled))


def _has_physical_cuda_device() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        try:
            import torch
            return torch.cuda.device_count() > 0
        except Exception:
            return False


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
