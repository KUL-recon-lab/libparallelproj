#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// CudaDevicePtr: RAII wrapper for CUDA device memory.
//
// RAII ("Resource Acquisition Is Initialization") ties a resource — here a
// device memory allocation — to the lifetime of a stack object. C++
// guarantees that the destructor of a stack object runs whenever the object
// goes out of scope, on EVERY exit path: normal return, early return, or an
// exception unwinding the stack. Putting cudaFree() in the destructor
// therefore guarantees the device buffer is released even when an error is
// thrown halfway through a projector function.
//
// Why this replaced the old "T* d_x; bool free_x; ... if (free_x)
// cudaFree(d_x);" pattern: the manual cudaFree() calls at the end of each
// projector only ran if execution reached the end. Since the projectors now
// throw std::runtime_error on CUDA errors (failed H2D copy, kernel launch
// error, failed cudaDeviceSynchronize, ...), a throw would have skipped the
// cleanup block and leaked every device buffer allocated up to that point.
// With this wrapper no explicit free is needed anywhere — the compiler
// inserts the destructor calls on all paths.
//
// Ownership semantics (the `owns` flag, set by handle_cuda_input_array):
//   * owns == true  : the input was a host array; a temporary device copy
//                     was allocated and must be freed.
//                     If the buffer was allocated stream-ordered
//                     (cudaMallocAsync, stream_ordered == true), the destructor
//                     frees it with cudaFreeAsync on the same stream, i.e. after
//                     all work queued on that stream (copies, kernel) has used it,
//                     without synchronizing the host. Otherwise (GPUs without
//                     memory pool support) it calls cudaFree.
//   * owns == false : the input was already device or managed memory; `ptr`
//                     just borrows the caller's pointer. The destructor does
//                     nothing — we must not free memory we do not own.
//
// The wrapper is non-copyable: two copies owning the same pointer would
// both free it (double free). It is movable, so ownership can be
// transferred; the moved-from object is reset so its destructor is a no-op.
//
// Note: freeing in a destructor must not throw (destructors that throw
// during stack unwinding terminate the program), so the return value is
// deliberately ignored. This is the same trade-off std::unique_ptr makes.
// ---------------------------------------------------------------------------
template <typename T>
struct CudaDevicePtr {
    T*           ptr            = nullptr; // device pointer used in kernel launches
    bool         owns           = false;   // true only if we allocated ptr ourselves
    bool         stream_ordered = false;   // allocated with cudaMallocAsync on `stream`
    cudaStream_t stream         = nullptr; // stream used for a stream-ordered free

    CudaDevicePtr() = default;

    // Non-copyable: copies would double-free the same allocation.
    CudaDevicePtr(const CudaDevicePtr&)            = delete;
    CudaDevicePtr& operator=(const CudaDevicePtr&) = delete;

    // Movable: transfers ownership and disarms the source object.
    CudaDevicePtr(CudaDevicePtr&& other) noexcept
        : ptr(other.ptr), owns(other.owns), stream_ordered(other.stream_ordered), stream(other.stream)
    { other.disarm(); }
    CudaDevicePtr& operator=(CudaDevicePtr&& other) noexcept {
        if (this != &other) {
            release(); // release current allocation first
            ptr = other.ptr; owns = other.owns;
            stream_ordered = other.stream_ordered; stream = other.stream;
            other.disarm();
        }
        return *this;
    }

    // Runs automatically on scope exit (including exception unwinding).
    ~CudaDevicePtr() { release(); }

    T* get() const { return ptr; }

private:
    void release() noexcept {
        if (owns && ptr) {
            if (stream_ordered)
                cudaFreeAsync(ptr, stream);
            else
                cudaFree(ptr);
        }
        disarm();
    }
    void disarm() noexcept { ptr = nullptr; owns = false; stream_ordered = false; stream = nullptr; }
};

// ---------------------------------------------------------------------------
// CudaCallStream: the stream a projector call runs on.
//
// stream == nullptr (default of the C API): legacy default stream and the call
//   is synchronous: before returning it waits for its stream (not for the whole
//   device, i.e. not for unrelated work on other streams).
// otherwise: all work (copies, kernel, frees) is queued on the given stream and,
//   for device / managed arrays, the call returns without waiting.
// Calls with host arrays always wait for their stream before returning (host
// outputs must be complete, host inputs must not be read after returning).
// ---------------------------------------------------------------------------
struct CudaCallStream
{
    cudaStream_t stream;
    bool synchronous;

    explicit CudaCallStream(void *user_stream)
        : stream(user_stream ? static_cast<cudaStream_t>(user_stream) : cudaStreamLegacy),
          synchronous(user_stream == nullptr) {}
};

// Image dimensions passed to the kernels by value (no device copy needed).
struct ImageDim
{
    int d[3];
    std::size_t nvoxels() const
    {
        return static_cast<std::size_t>(d[0]) * static_cast<std::size_t>(d[1]) * static_cast<std::size_t>(d[2]);
    }
};

// Makes device_id the current CUDA device for the lifetime of the guard and restores the
// caller's current device afterwards (also if an exception is thrown). A library call must
// not change the current device of the caller (e.g. CuPy / PyTorch create new arrays on it).
// device_id < 0: keep the current device.
// Declare the guard before any CudaDevicePtr, so that buffers are freed while device_id is
// still current (destruction in reverse order of declaration).
class CudaDeviceGuard
{
public:
    explicit CudaDeviceGuard(int device_id)
    {
        if (device_id < 0)
            return;

        cudaError_t err = cudaGetDevice(&previous_device_);
        if (err != cudaSuccess)
        {
            cudaGetLastError(); // clear last-error state before throwing
            throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
        }
        if (previous_device_ == device_id)
            return;

        err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            cudaGetLastError(); // clear last-error state before throwing
            throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(err));
        }
        restore_ = true;
    }

    ~CudaDeviceGuard()
    {
        if (restore_ && cudaSetDevice(previous_device_) != cudaSuccess)
            cudaGetLastError(); // never throw from a destructor
    }

    CudaDeviceGuard(const CudaDeviceGuard &) = delete;
    CudaDeviceGuard &operator=(const CudaDeviceGuard &) = delete;

private:
    int previous_device_ = 0;
    bool restore_ = false;
};

// Checked device-to-host copy. Throws std::runtime_error on failure.
inline void cuda_memcpy_d2h(void* dst, const void* src, std::size_t size)
{
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaGetLastError(); // clear last-error state before throwing
        throw std::runtime_error(
            std::string("cudaMemcpy (D2H) failed: ") + cudaGetErrorString(err));
    }
}

// Read the image dimensions when img_dim may be a host, CUDA managed or device
// pointer. A device / managed img_dim is copied on `stream` and waited for
// (only this stream); pass a host pointer for fully asynchronous calls and for
// CUDA graph capture (the Python bindings always do). Throws
// std::invalid_argument or std::runtime_error on error.
inline ImageDim cuda_image_dim_to_host(const int *img_dim_ptr, cudaStream_t stream)
{
    if (!img_dim_ptr)
        throw std::invalid_argument("image_dim is null");

    ImageDim h = {{0, 0, 0}};
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, static_cast<const void *>(img_dim_ptr));
    if (err != cudaSuccess)
    {
        // Expected for plain host pointers on older CUDA versions. Clear the
        // last-error state so it cannot surface in a later kernel launch check.
        cudaGetLastError();
    }

    // If pointer known to CUDA and points to device/managed memory, copy to host.
    if (err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged))
    {
        err = cudaMemcpyAsync(h.d, img_dim_ptr, 3 * sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess)
            err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            cudaGetLastError(); // clear last-error state before throwing
            throw std::runtime_error(std::string("copy of image_dim to host failed: ") + cudaGetErrorString(err));
        }
    }
    else
    {
        // Treat as host pointer (or pointer attributes not available) — read directly.
        h.d[0] = img_dim_ptr[0];
        h.d[1] = img_dim_ptr[1];
        h.d[2] = img_dim_ptr[2];
    }

    if (h.d[0] <= 0 || h.d[1] <= 0 || h.d[2] <= 0)
        throw std::invalid_argument("invalid image_dim values");

    return h;
}

// Queue a device-to-host copy on `stream` (completed by cuda_finish_call).
inline void cuda_memcpy_d2h_async(void *dst, const void *src, std::size_t size, cudaStream_t stream)
{
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        cudaGetLastError(); // clear last-error state before throwing
        throw std::runtime_error(std::string("cudaMemcpyAsync (D2H) failed: ") + cudaGetErrorString(err));
    }
}

// Check the kernel launch (launch errors are reported immediately, also for
// asynchronous calls). Throws std::runtime_error on failure.
inline void cuda_check_launch()
{
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(launch_err));
}

// Finish a projector call: wait for the call's stream if the call is
// synchronous or involves host buffers (copied inputs / outputs), otherwise
// return immediately (asynchronous call with device / managed arrays). Errors
// during kernel execution are reported here for waiting calls; for asynchronous
// calls they surface at the caller's next synchronizing operation.
inline void cuda_finish_call(const CudaCallStream &call_stream, bool has_host_buffers)
{
    if (!call_stream.synchronous && !has_host_buffers)
        return;

    cudaError_t sync_err = cudaStreamSynchronize(call_stream.stream);
    if (sync_err != cudaSuccess)
    {
        cudaGetLastError(); // clear last-error state before throwing
        throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(sync_err));
    }
}

// Make an input / output array available on the GPU for a projector call.
// Device and managed arrays are used directly (managed arrays get memory hints
// and a prefetch on `stream`). Host arrays are copied to a temporary device
// buffer, allocated and copied stream-ordered on `stream` (cudaMallocAsync /
// cudaMemcpyAsync; cudaMalloc on GPUs without memory pool support). The
// CudaDevicePtr owns the temporary buffer and frees it stream-ordered.
// Throws std::runtime_error on failure (out is left empty).

// Overload for constant input_ptr (const T*)
template <typename T>
void handle_cuda_input_array(const T *input_ptr, CudaDevicePtr<T> &out, std::size_t size, int device_id,
                             cudaMemoryAdvise memory_hint, cudaStream_t stream);

// Overload for non-constant input_ptr (T*)
template <typename T>
void handle_cuda_input_array(T *input_ptr, CudaDevicePtr<T> &out, std::size_t size, int device_id,
                             cudaMemoryAdvise memory_hint, cudaStream_t stream);
