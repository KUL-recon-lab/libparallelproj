#pragma once

#include <cuda_runtime.h>
#include <iostream>

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
// Returns true if at least one CUDA device is available. Returns false and
// prints the reason otherwise (cudaGetDeviceCount returns cudaErrorNoDevice
// or cudaErrorInsufficientDriver when no GPU / driver is present).
inline bool cuda_device_available()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess)
    {
        std::cout << "SKIP: no usable CUDA device (cudaGetDeviceCount: "
                  << cudaGetErrorString(err) << ")" << std::endl;
        // Clear the error state in case the caller decides to continue anyway.
        cudaGetLastError();
        return false;
    }

    if (device_count == 0)
    {
        std::cout << "SKIP: no CUDA device found" << std::endl;
        return false;
    }

    return true;
}
