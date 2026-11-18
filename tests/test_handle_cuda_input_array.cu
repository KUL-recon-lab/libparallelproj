// Unit test for handle_cuda_input_array() (src/cuda_utils.cu).
//
// This test compiles src/cuda_utils.cu directly into the test executable
// (see CMakeLists.txt) instead of linking the symbol from the parallelproj
// shared library: handle_cuda_input_array is an internal utility and is
// deliberately NOT exported (no PARALLELPROJ_API), so on Windows it is not
// visible in the DLL. Compiling it in keeps the test identical on Linux,
// Windows and WSL and avoids throwing exceptions across a DLL boundary.
//
// Tested behavior:
//   1. An allocation that cannot possibly succeed makes the function throw
//      std::runtime_error mentioning "cudaMalloc" (cudaMalloc or
//      cudaMallocAsync), and leaves the CudaDevicePtr empty (ptr == nullptr,
//      owns == false).
//   2. The CUDA context stays usable after the failed allocation
//      (recovery check with a small real allocation).
//   3. Happy path on the legacy default stream: a small host array is copied
//      to the device (owns == true) and the device copy matches element-wise.
//   4. Same on a non-blocking stream: the copy is queued on that stream
//      (stream-ordered), the buffer is freed stream-ordered on that stream.
//   5. Device memory is borrowed, not copied (owns == false, same pointer).

#include "cuda_utils.h"
#include "cuda_test_utils.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 1. allocation failure must throw and leave clean state
// ---------------------------------------------------------------------------
bool test_oversized_alloc_throws(int device_id)
{
    // Small, valid host buffer ...
    std::vector<float> host(16, 1.0f);

    // ... but an allocation size that can never succeed on any GPU:
    // 2^62 bytes (4 EiB) exceeds the virtual address space of every CUDA
    // device, on Linux, Windows (WDDM and TCC) and WSL alike. Note that a
    // "few times free VRAM" would NOT be reliable here: Windows WDDM can
    // page device memory into system RAM and oversubscribe.
    //
    // handle_cuda_input_array allocates BEFORE the host-to-device copy, so
    // the deliberately wrong size never causes an out-of-bounds read of the
    // small host buffer: the function must throw at the allocation stage.
    const std::size_t impossible_size = std::size_t(1) << 62;

    CudaDevicePtr<float> d;
    bool threw_expected = false;

    try
    {
        handle_cuda_input_array(host.data(), d, impossible_size, device_id,
                                cudaMemAdviseSetReadMostly, cudaStreamLegacy);
    }
    catch (const std::runtime_error &e)
    {
        const std::string msg = e.what();
        if (msg.find("cudaMalloc") != std::string::npos)
        {
            threw_expected = true;
            std::cout << "  got expected exception: " << msg << std::endl;
        }
        else
        {
            std::cerr << "  FAIL: std::runtime_error thrown, but not from cudaMalloc(Async): "
                      << msg << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "  FAIL: unexpected exception type: " << e.what() << std::endl;
    }

    if (!threw_expected)
    {
        std::cerr << "  FAIL: no std::runtime_error thrown for impossible allocation"
                  << std::endl;
        return false;
    }

    // The failure path must not leave dangling state behind.
    if (d.ptr != nullptr)
    {
        std::cerr << "  FAIL: ptr not reset to nullptr after failure" << std::endl;
        return false;
    }
    if (d.owns)
    {
        std::cerr << "  FAIL: owns set to true although allocation failed" << std::endl;
        return false;
    }

    // The library must clear the CUDA runtime's last-error state before
    // throwing. CUDA only resets this state when it is read, so a stale
    // error would be misattributed to the kernel launch check of a later,
    // successful call (observed in practice: a tiny projection right after
    // a failed one raised "CUDA kernel launch failed: out of memory").
    cudaError_t stale = cudaGetLastError();
    if (stale != cudaSuccess)
    {
        std::cerr << "  FAIL: last-error state not cleared before throwing: "
                  << cudaGetErrorString(stale) << std::endl;
        return false;
    }

    std::cout << "  PASS: oversized allocation throws and leaves clean state" << std::endl;
    return true;
}

// ---------------------------------------------------------------------------
// 2. CUDA context must stay usable after the failed allocation
// ---------------------------------------------------------------------------
bool test_context_recovers(int /*device_id*/)
{
    // A failed allocation does not corrupt the context, and the library is
    // responsible for clearing the last-error state itself (checked in
    // test 1) - so WITHOUT any cleanup here, a small real allocation must
    // succeed afterwards.
    float *small_ptr = nullptr;
    cudaError_t err = cudaMalloc(&small_ptr, 16 * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "  FAIL: small allocation after failed oversized allocation: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    cudaFree(small_ptr);

    std::cout << "  PASS: context usable after failed allocation" << std::endl;
    return true;
}

// ---------------------------------------------------------------------------
// 3. / 4. Happy path: host array gets a device copy that matches
// ---------------------------------------------------------------------------
bool test_host_array_roundtrip(int device_id, cudaStream_t stream, const char *label)
{
    const std::size_t n = 256;
    std::vector<float> host(n);
    for (std::size_t i = 0; i < n; ++i)
        host[i] = static_cast<float>(i) * 0.5f;

    bool ok = true;
    {
        CudaDevicePtr<float> d;

        try
        {
            handle_cuda_input_array(host.data(), d, n * sizeof(float), device_id,
                                    cudaMemAdviseSetReadMostly, stream);
        }
        catch (const std::exception &e)
        {
            std::cerr << "  FAIL: unexpected exception on happy path: " << e.what() << std::endl;
            return false;
        }

        if (!d.owns)
        {
            std::cerr << "  FAIL: owns should be true for a host input array" << std::endl;
            return false;
        }
        if (d.ptr == nullptr)
        {
            std::cerr << "  FAIL: ptr is null on happy path" << std::endl;
            return false;
        }
        if (d.stream != stream)
        {
            std::cerr << "  FAIL: buffer does not remember the stream it belongs to" << std::endl;
            return false;
        }

        // read back, stream-ordered after the H2D copy
        std::vector<float> roundtrip(n, -1.0f);
        cudaError_t err = cudaMemcpyAsync(roundtrip.data(), d.ptr, n * sizeof(float),
                                          cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess)
            err = cudaStreamSynchronize(stream);
        ok = (err == cudaSuccess);
        if (!ok)
            std::cerr << "  FAIL: D2H copy failed: " << cudaGetErrorString(err) << std::endl;

        for (std::size_t i = 0; ok && i < n; ++i)
        {
            if (roundtrip[i] != host[i])
            {
                std::cerr << "  FAIL: device copy differs at index " << i << std::endl;
                ok = false;
            }
        }
    } // d frees its buffer here (stream-ordered on `stream` if supported)

    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        std::cerr << "  FAIL: error after freeing the buffer: " << cudaGetErrorString(err) << std::endl;
        ok = false;
    }

    if (ok)
        std::cout << "  PASS: host array round trip (" << label << ")" << std::endl;
    return ok;
}

// ---------------------------------------------------------------------------
// 5. Device memory is borrowed
// ---------------------------------------------------------------------------
bool test_device_array_borrowed(int device_id)
{
    float *dev_ptr = nullptr;
    if (cudaMalloc(&dev_ptr, 64 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "  FAIL: cudaMalloc" << std::endl;
        return false;
    }

    bool ok = true;
    {
        CudaDevicePtr<float> d;
        handle_cuda_input_array(static_cast<const float *>(dev_ptr), d, 64 * sizeof(float), device_id,
                                cudaMemAdviseSetReadMostly, cudaStreamLegacy);
        if (d.ptr != dev_ptr || d.owns)
        {
            std::cerr << "  FAIL: device array should be borrowed (same pointer, owns == false)" << std::endl;
            ok = false;
        }
    } // must not free dev_ptr

    // dev_ptr must still be valid (freeing it twice would fail)
    if (cudaFree(dev_ptr) != cudaSuccess)
    {
        std::cerr << "  FAIL: borrowed device pointer was freed by CudaDevicePtr" << std::endl;
        ok = false;
    }

    if (ok)
        std::cout << "  PASS: device array borrowed" << std::endl;
    return ok;
}

int main()
{
    if (!cuda_device_available())
        return CUDA_TEST_SKIP_RETURN_CODE;

    const int device_id = 0;
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        std::cerr << "FAIL: cudaSetDevice: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    bool all_passed = true;

    std::cout << "test 1: oversized allocation must throw" << std::endl;
    all_passed = test_oversized_alloc_throws(device_id) && all_passed;

    std::cout << "test 2: context recovery after failed allocation" << std::endl;
    all_passed = test_context_recovers(device_id) && all_passed;

    std::cout << "test 3: host array round trip on the legacy default stream" << std::endl;
    all_passed = test_host_array_roundtrip(device_id, cudaStreamLegacy, "legacy default stream") && all_passed;

    std::cout << "test 4: host array round trip on a non-blocking stream" << std::endl;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
    {
        std::cerr << "FAIL: cudaStreamCreateWithFlags" << std::endl;
        return 1;
    }
    all_passed = test_host_array_roundtrip(device_id, stream, "non-blocking stream") && all_passed;
    cudaStreamDestroy(stream);

    std::cout << "test 5: device arrays are borrowed" << std::endl;
    all_passed = test_device_array_borrowed(device_id) && all_passed;

    if (all_passed)
    {
        std::cout << "all tests passed" << std::endl;
        return 0;
    }
    return 1;
}
