#pragma once

#include <cuda_runtime.h>
#include <iostream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX // keep windows.h from defining min/max macros
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Exit code that tells CTest a test was skipped (not passed, not failed).
// Must match the SKIP_RETURN_CODE test property set in CMakeLists.txt.
constexpr int CUDA_TEST_SKIP_RETURN_CODE = 77;

// Runtime guard for CUDA tests.
//
// Building with the CUDA toolkit does not imply that a physical GPU (or an
// NVIDIA driver) is present at runtime - e.g. on CI runners or build-only
// machines. Call this at the top of main():
//
//   int main()
//   {
//     if (!cuda_device_available())
//       return CUDA_TEST_SKIP_RETURN_CODE;
//     ...
//   }
//
// IMPORTANT implementation detail: we deliberately do NOT use the CUDA
// runtime API (cudaGetDeviceCount) for the initial detection. The runtime
// lazily loads the driver library (nvcuda.dll / libcuda.so.1), and on
// Windows machines without an NVIDIA driver this crashes with an access
// violation inside cudart instead of returning cudaErrorInsufficientDriver
// (observed on GitHub windows runners with the conda-forge CUDA toolchain).
// Instead we load the *driver* library dynamically ourselves and call
// cuInit / cuDeviceGetCount through function pointers - if the library is
// not even present we can skip gracefully. This mirrors the detection logic
// in tests/conftest.py used for the Python test suite.
inline bool cuda_device_available()
{
    using cuInit_t = int (*)(unsigned int);
    using cuDeviceGetCount_t = int (*)(int *);

    cuInit_t cu_init = nullptr;
    cuDeviceGetCount_t cu_device_get_count = nullptr;

#ifdef _WIN32
    HMODULE driver_lib = LoadLibraryA("nvcuda.dll");
    if (driver_lib == nullptr)
    {
        std::cout << "SKIP: CUDA driver library (nvcuda.dll) not found - no GPU driver installed"
                  << std::endl;
        return false;
    }
    cu_init = reinterpret_cast<cuInit_t>(
        reinterpret_cast<void *>(GetProcAddress(driver_lib, "cuInit")));
    cu_device_get_count = reinterpret_cast<cuDeviceGetCount_t>(
        reinterpret_cast<void *>(GetProcAddress(driver_lib, "cuDeviceGetCount")));
#else
    void *driver_lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (driver_lib == nullptr)
        driver_lib = dlopen("libcuda.so", RTLD_NOW);
    if (driver_lib == nullptr)
    {
        std::cout << "SKIP: CUDA driver library (libcuda.so) not found - no GPU driver installed"
                  << std::endl;
        return false;
    }
    cu_init = reinterpret_cast<cuInit_t>(dlsym(driver_lib, "cuInit"));
    cu_device_get_count = reinterpret_cast<cuDeviceGetCount_t>(dlsym(driver_lib, "cuDeviceGetCount"));
#endif
    // We intentionally keep the driver library loaded for the lifetime of the
    // process - the CUDA runtime will need it anyway.

    if (cu_init == nullptr || cu_device_get_count == nullptr)
    {
        std::cout << "SKIP: cuInit / cuDeviceGetCount not found in CUDA driver library"
                  << std::endl;
        return false;
    }

    // cuInit(0) returns CUDA_SUCCESS (0) when a usable driver is present.
    if (cu_init(0) != 0)
    {
        std::cout << "SKIP: cuInit failed - no usable CUDA driver" << std::endl;
        return false;
    }

    int driver_device_count = 0;
    if (cu_device_get_count(&driver_device_count) != 0 || driver_device_count == 0)
    {
        std::cout << "SKIP: no physical CUDA device found" << std::endl;
        return false;
    }

    // Driver and physical device are present, so the CUDA *runtime* is now
    // safe to use. Cross-check via the runtime API (this also catches
    // driver-too-old-for-this-runtime situations).
    int runtime_device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&runtime_device_count);
    if (err != cudaSuccess || runtime_device_count == 0)
    {
        std::cout << "SKIP: CUDA runtime reports no usable device (cudaGetDeviceCount: "
                  << cudaGetErrorString(err) << ")" << std::endl;
        // Clear the error state in case the caller decides to continue anyway.
        cudaGetLastError();
        return false;
    }

    return true;
}
